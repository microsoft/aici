use std::{fmt::Debug, sync::Mutex};

use crate::Tensor;
use aici_abi::TokenId;
use aicirt::api::SequenceResult;
use serde::{Deserialize, Serialize};

use crate::{blocks::BlockRef, config::SamplingParams, LogitsProcessor};

pub type Token = u32;
pub type SeqId = usize;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum FinishReason {
    /// EOS token was generated.
    FoundEos,
    /// Stopped by AICI.
    AiciStop,
    /// SamplingParams.max_tokens reached.
    MaxTokensReached,
    /// Explicit abort request on the engine.
    Aborted,
    /// The scheduler didn't like the sequence.
    Failed,
}

impl FinishReason {
    pub fn short_name(&self) -> String {
        let r = match self {
            FinishReason::FoundEos => "eos",
            FinishReason::MaxTokensReached => "length",
            FinishReason::Aborted => "abort",
            FinishReason::Failed => "fail",
            FinishReason::AiciStop => "aici",
        };
        r.to_string()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum SchedulingPhase {
    Waiting,
    Running,
    Suspended,
    Swapped,
    Finished(FinishReason),
}

#[derive(Debug, Clone)]
pub enum AiciSampling {
    Regular,
    SampleWithBias {
        offset: usize,
    },
    Splice {
        backtrack: u32,
        ff_tokens: Vec<TokenId>,
    },
}

impl Default for AiciSampling {
    fn default() -> Self {
        Self::Regular
    }
}

pub struct Sequence {
    pub seq_id: SeqId,
    pub index: usize, // within the sequence group
    pub tokens: Vec<Token>,
    pub prompt_len: usize,
    pub(crate) output_ptr: usize,
    pub(crate) num_kv_computed: usize,
    pub(crate) has_aici: bool,
    pub(crate) aici_sampling: AiciSampling,
    pub aici_logs: Vec<SequenceResult>,
    pub pending_fork_ids: Vec<SeqId>,

    // state for Scheduler and BlockSpaceManager
    pub(crate) sched_phase: SchedulingPhase,
    pub(crate) gpu_blocks: Vec<BlockRef>,
    pub(crate) cpu_blocks: Vec<BlockRef>,
    block_size: usize,
}

impl Debug for Sequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequence")
            .field("seq_id", &self.seq_id)
            .field("sched_phase", &self.sched_phase)
            .field("kv_computed", &self.num_kv_computed)
            .field("aici_sampling", &self.aici_sampling)
            .field("tokens", &self.tokens)
            .field("prompt_len", &self.prompt_len)
            .finish()
    }
}

impl Sequence {
    pub(crate) fn new(seq_id: SeqId, tokens: &[Token], block_size: usize) -> Self {
        let prompt_len = tokens.len();
        Self {
            seq_id,
            index: 0,
            sched_phase: SchedulingPhase::Waiting,
            tokens: tokens.to_vec(),
            num_kv_computed: 0,
            prompt_len,
            output_ptr: prompt_len,
            gpu_blocks: Vec::new(),
            cpu_blocks: Vec::new(),
            block_size,
            has_aici: false,
            aici_logs: Vec::new(),
            aici_sampling: AiciSampling::Regular,
            pending_fork_ids: Vec::new(),
        }
    }

    pub fn get_len(&self) -> usize {
        self.tokens.len()
    }

    pub fn get_gen_len(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }

    pub fn get_gpu_slot(&self, position: usize) -> usize {
        let block_index = self.gpu_blocks[position / self.block_size].get_index();
        let block_offset = position % self.block_size;
        block_index * self.block_size + block_offset
    }

    #[allow(dead_code)]
    pub(crate) fn fork_as(&self, seq_id: SeqId, index: usize) -> Self {
        Self {
            seq_id,
            index,
            sched_phase: self.sched_phase,
            num_kv_computed: self.num_kv_computed,
            tokens: self.tokens.clone(),
            output_ptr: self.prompt_len,
            prompt_len: self.prompt_len,
            gpu_blocks: self.gpu_blocks.iter().map(|x| x.fork()).collect(),
            cpu_blocks: self.cpu_blocks.iter().map(|x| x.fork()).collect(),
            block_size: self.block_size,
            has_aici: self.has_aici,
            aici_logs: Vec::new(),
            pending_fork_ids: Vec::new(),
            aici_sampling: AiciSampling::Regular,
        }
    }

    pub fn append_token_id(&mut self, token_id: Token) {
        self.tokens.push(token_id)
    }

    pub fn finish_reason(&self) -> Option<FinishReason> {
        match self.sched_phase {
            SchedulingPhase::Finished(reason) => Some(reason),
            _ => None,
        }
    }

    pub fn gen_output(&mut self) -> SeqOutput {
        let new_output_tokens = self.tokens[self.output_ptr..].to_vec();
        self.output_ptr = self.tokens.len();
        SeqOutput {
            seq_id: self.seq_id,
            index: self.index,
            new_output_tokens,
            new_text: String::new(),
            output_tokens: self.tokens[self.prompt_len..].to_vec(),
            finish_reason: self.finish_reason(),
            aici_logs: std::mem::take(&mut self.aici_logs),
        }
    }

    pub fn is_finished(&self) -> bool {
        self.finish_reason().is_some()
    }
}

/// A group of sequences that are generated from the same prompt.
pub struct SequenceGroup {
    pub request_id: String,
    pub prompt: String,
    pub seqs: Vec<Sequence>,
    pub sampling_params: SamplingParams,
    pub arrival_time: std::time::Instant,
    pub logits_processor: LogitsProcessor,
    pub max_index: usize,
}

pub struct BatchInfo {
    pub tokens: Tensor,         // u32, [num_tokens]
    pub positions: Tensor,      // i64, [num_tokens]
    pub seqlens_q: Tensor,      // u32, [batch_size + 1]; points to tokens/positions
    pub seqlens_k: Tensor,      // u32, [batch_size + 1]; can go outside tokens/positions
    pub gather_mapping: Tensor, // u32, [sum(context_len + prompt_len)]
    pub slot_mapping: Tensor,   // u32, [num_tokens]
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub kv_cache: Vec<(Tensor, Tensor)>,

    pub infer_log: Mutex<Vec<(String, Tensor)>>,
    pub step_no: usize,
}

impl BatchInfo {
    pub fn log_tensor(&self, key: &str, value: &Tensor) {
        if false {
            self.infer_log
                .lock()
                .unwrap()
                .push((key.to_string(), value.copy()));
        }
    }

    pub fn save_log(&self, filename: &str) {
        let mut lck = self.infer_log.lock().unwrap();
        if lck.len() == 0 {
            return;
        }
        let tensors = lck
            .iter()
            .enumerate()
            .map(|(i, (k, v))| (format!("{:0>4}_{}", i, k), v.copy()))
            .collect::<Vec<_>>();
        lck.clear();
        Tensor::write_safetensors(&tensors, filename).unwrap();
    }
}

impl Debug for BatchInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchInfo")
            .field("tokens", &self.tokens)
            .field("positions", &self.positions)
            .field("seqlens_q", &self.seqlens_q)
            .field("seqlens_k", &self.seqlens_k)
            // .field("gather_mapping", &self.gather_mapping)
            // .field("slot_mapping", &self.slot_mapping)
            .field("max_seqlen_q", &self.max_seqlen_q)
            .field("max_seqlen_k", &self.max_seqlen_k)
            .finish()
    }
}

impl SequenceGroup {
    /// The maximum number of sequences running in parallel in the remaining
    /// lifetime of the request.
    pub fn get_max_num_running_seqs(&self) -> usize {
        if self.sampling_params.use_beam_search {
            // For beam search, maximally there will always be `best_of` beam
            // candidates running in the future.
            self.sampling_params.best_of
        } else {
            if self.sampling_params.best_of > self.num_seqs(None) {
                // At prompt stage, the sequence group is not yet filled up
                // and only have one sequence running. However, in the
                // generation stage, we will have `best_of` sequences running.
                self.sampling_params.best_of
            } else {
                // At sampling stages, return the number of actual sequences
                // running.
                self.num_seqs(Some(SchedulingPhase::Running))
            }
        }
    }

    pub fn only_seq(&self) -> &Sequence {
        if self.seqs.len() == 1 {
            &self.seqs[0]
        } else {
            panic!("num seq {} != 1", self.seqs.len());
        }
    }

    /// Retrieves sequences, optionally filtered by status.
    pub fn get_seqs(&self, status: Option<SchedulingPhase>) -> Vec<&Sequence> {
        match status {
            Some(status_filter) => self
                .seqs
                .iter()
                .filter(|seq| seq.sched_phase == status_filter)
                .collect(),
            None => self.seqs.iter().collect(),
        }
    }

    /// Returns the number of sequences, optionally filtered by status.
    pub fn num_seqs(&self, status: Option<SchedulingPhase>) -> usize {
        self.get_seqs(status).len()
    }

    /// Checks if all sequences are finished.
    pub fn is_finished(&self) -> bool {
        self.seqs.iter().all(|seq| seq.is_finished())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqOutput {
    pub seq_id: SeqId,
    pub index: usize, // within the sequence group
    pub new_output_tokens: Vec<Token>,
    pub new_text: String,
    /// The tokens generated by the model. Doesn't include prompt tokens.
    pub output_tokens: Vec<Token>,
    pub finish_reason: Option<FinishReason>,
    pub aici_logs: Vec<SequenceResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestOutput {
    pub request_id: String,
    pub seq_outputs: Vec<SeqOutput>,
    pub is_final: bool,
    pub is_ambiguous: bool,
}

/*
You are PyRust Translator, designed to assist users in translating Python code into Rust.
- only translate code, do not explain differences between Python and Rust
- if Python code is using the 'pytorch' package, the Rust should use 'candle' (assuming similar APIs to 'tch' and 'pytorch')
- keep comments and docstrings; attach docstrings to struct fields or parameters as appropriate in Rust
- keep asserts
- provide complete translations, filling out all methods and their bodies; avoid comments like "// Similar to Python" or "// Implement other methods"
- always translate code, even if it won't work to provide a base line for the user

*/

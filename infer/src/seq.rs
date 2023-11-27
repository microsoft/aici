#![allow(dead_code)]

use std::fmt::Debug;

use anyhow::Result;
use candle::Tensor;

use crate::{block::BlockRef, config::SamplingParams};

pub type Token = u32;
pub type SeqId = u32;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FinishReason {
    Stopped,
    LengthCapped,
    Aborted,
    Ignored,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum SchedulingPhase {
    Waiting,
    Running,
    Swapped,
    Finished(FinishReason),
}

#[derive(Debug, Clone, Copy)]
pub enum StepType {
    Prompt,
    Fixed(usize),
    Gen,
}

pub struct Sequence {
    pub seq_id: SeqId,
    pub step_type: StepType,
    pub tokens: Vec<Token>,
    pub prompt_len: usize,

    // state for Scheduler and BlockManager
    pub(crate) sched_phase: SchedulingPhase,
    pub(crate) phys_blocks: Vec<BlockRef>,
    block_size: usize,
}

impl Debug for Sequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequence")
            .field("seq_id", &self.seq_id)
            .field("sched_phase", &self.sched_phase)
            .field("step_type", &self.step_type)
            .field("tokens", &self.tokens)
            .field("prompt_len", &self.prompt_len)
            .finish()
    }
}

impl Sequence {
    pub(crate) fn new(seq_id: SeqId, tokens: &[Token], block_size: usize) -> Self {
        let prompt_len = tokens.len();
        let mut seq = Self {
            seq_id,
            sched_phase: SchedulingPhase::Waiting,
            step_type: StepType::Prompt,
            tokens: Vec::new(),
            prompt_len,
            phys_blocks: Vec::new(),
            block_size,
        };
        seq._append_tokens_to_blocks(tokens);
        seq
    }

    pub fn get_len(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn fork_as(&self, seq_id: SeqId) -> Self {
        let mut seq = Self {
            seq_id,
            sched_phase: self.sched_phase,
            step_type: self.step_type,
            tokens: self.tokens.clone(),
            prompt_len: self.prompt_len,
            phys_blocks: self.phys_blocks.iter().map(|x| x.fork()).collect(),
            block_size: self.block_size,
        };
        seq._append_tokens_to_blocks(&self.tokens);
        seq
    }

    fn _append_tokens_to_blocks(&mut self, token_ids: &[Token]) {
        self.tokens.extend_from_slice(token_ids);
    }

    pub fn append_token_id(&mut self, token_id: Token) {
        self._append_tokens_to_blocks(&[token_id]);
    }

    pub fn is_finished(&self) -> bool {
        match self.sched_phase {
            SchedulingPhase::Finished(_) => true,
            _ => false,
        }
    }
}

/// A group of sequences that are generated from the same prompt.
pub struct SequenceGroup {
    pub request_id: String,
    pub seqs: Vec<Sequence>,
    pub sampling_params: SamplingParams,
    pub arrival_time: std::time::Instant,
}

// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
//
// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
// `seqlen_1 + seqlen_2`, etc.

#[derive(Debug)]
pub struct BatchInfo {
    pub tokens: Tensor,    // u32, [num_tokens]
    pub positions: Tensor, // i64, [num_tokens]
    pub seqlens_q: Tensor, // u32, [batch_size + 1]; points to tokens/positions
    pub seqlens_k: Tensor, // u32, [batch_size + 1]; can go outside tokens/positions
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
}

impl BatchInfo {
    pub fn from_seqs(seqs: &[Sequence], device: &candle::Device) -> Result<Self> {
        let mut k_ptr = 0u32;
        let mut q_ptr = 0u32;
        let mut positions = Vec::new();
        let mut seqlens_q = vec![q_ptr];
        let mut seqlens_k = vec![k_ptr];
        let mut tokens = Vec::new();
        let mut max_seqlen_q = 0;
        let mut max_seqlen_k = 0;
        for seq in seqs {
            let seq_len = seq.tokens.len();
            let k_len = seq_len;
            let q_len = match seq.step_type {
                StepType::Prompt => seq_len,
                StepType::Fixed(len) => len,
                StepType::Gen => 1,
            };
            let off = k_len - q_len;
            for idx in off..off + q_len {
                positions.push(idx as i64);
                tokens.push(seq.tokens[idx]);
            }
            q_ptr += q_len as u32;
            k_ptr += k_len as u32;
            seqlens_q.push(q_ptr);
            seqlens_k.push(k_ptr);

            max_seqlen_q = max_seqlen_q.max(q_len);
            max_seqlen_k = max_seqlen_k.max(k_len);
        }

        let positions = Tensor::new(positions.as_slice(), device)?;
        let seqlens_q = Tensor::new(seqlens_q.as_slice(), device)?;
        let seqlens_k = Tensor::new(seqlens_k.as_slice(), device)?;
        let tokens = Tensor::new(tokens.as_slice(), device)?;
        Ok(BatchInfo {
            tokens,
            positions,
            seqlens_q,
            seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
        })
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

    /// Retrieves finished sequences.
    fn get_finished_seqs(&self) -> Vec<&Sequence> {
        self.seqs.iter().filter(|seq| seq.is_finished()).collect()
    }

    /// Returns the number of sequences, optionally filtered by status.
    pub fn num_seqs(&self, status: Option<SchedulingPhase>) -> usize {
        self.get_seqs(status).len()
    }

    /// Adds a sequence.
    fn add(&mut self, seq: Sequence) {
        self.seqs.push(seq)
    }

    /// Checks if all sequences are finished.
    pub fn is_finished(&self) -> bool {
        self.seqs.iter().all(|seq| seq.is_finished())
    }
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

use crate::{
    config::SamplingParams, engine::ExpectedGeneration, LogitsProcessor, SeqId, SequenceManager,
};
use aici_abi::{toktree::TokTrie, TokenId};
use aicirt::api::SequenceResult;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub type Token = u32;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum FinishReason {
    /// EOS token was generated.
    FoundEos,
    /// Stopped by AICI.
    AiciStop,
    /// Too many prompt/generation tokens in the current request (sequence group)
    AiciOutOfFuel,
    /// SamplingParams.max_tokens reached.
    MaxTokensReached,
    /// Explicit abort request on the engine.
    Aborted,
    /// The scheduler didn't like the sequence.
    Failed,
    /// All sequences in the group are suspended.
    Deadlock,
}

impl FinishReason {
    pub fn short_name(&self) -> String {
        let r = match self {
            FinishReason::FoundEos => "eos",
            FinishReason::MaxTokensReached => "length",
            FinishReason::Aborted => "abort",
            FinishReason::Failed => "fail",
            FinishReason::AiciStop => "aici-stop",
            FinishReason::Deadlock => "deadlock",
            FinishReason::AiciOutOfFuel => "aici-out-of-fuel",
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
    tokens: Vec<Token>,
    pub prompt_len: usize,
    pub(crate) output_ptr: usize,
    pub(crate) output_pending: Vec<u8>,
    pub(crate) num_kv_computed: usize,
    pub(crate) has_aici: bool,
    pub(crate) aici_sampling: AiciSampling,
    pub aici_logs: Vec<SequenceResult>,
    pub pending_fork_ids: Vec<SeqId>,
    pub(crate) expected: Option<ExpectedGeneration>,

    // state for Scheduler and BlockSpaceManager
    pub(crate) sched_phase: SchedulingPhase,
}

impl Debug for Sequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequence")
            .field("seq_id", &self.seq_id.to_num())
            .field("sched_phase", &self.sched_phase)
            .field("kv_computed", &self.num_kv_computed)
            .field("aici_sampling", &self.aici_sampling)
            .field("tokens", &self.tokens)
            .field("prompt_len", &self.prompt_len)
            .finish()
    }
}

impl Sequence {
    pub(crate) fn new(seq_id: SeqId, tokens: &[Token]) -> Self {
        let prompt_len = tokens.len();
        Self {
            seq_id,
            index: 0,
            sched_phase: SchedulingPhase::Waiting,
            tokens: tokens.to_vec(),
            num_kv_computed: 0,
            prompt_len,
            output_ptr: prompt_len,
            output_pending: Vec::new(),
            has_aici: false,
            aici_logs: Vec::new(),
            aici_sampling: AiciSampling::Regular,
            pending_fork_ids: Vec::new(),
            expected: None,
        }
    }

    pub fn get_len(&self) -> usize {
        self.tokens.len()
    }

    /// Indicate that the generation will soon run for this sequence and thus
    /// all the tokens will have KV computed.
    pub(crate) fn sync_computed_kv(&mut self) {
        self.num_kv_computed = self.get_len();
    }

    fn trim_computed_kv(&mut self, v: usize, seq_mgr: &impl SequenceManager) {
        if self.num_kv_computed != v {
            assert!(self.num_kv_computed > v);
            seq_mgr.trim(self.seq_id, v);
            self.num_kv_computed = v;
        }
    }

    pub(crate) fn clear_computed_kv(&mut self, seq_mgr: &impl SequenceManager) {
        self.trim_computed_kv(0, seq_mgr);
    }

    fn trim_physical_blocks(&mut self, seq_mgr: &impl SequenceManager) {
        self.trim_computed_kv(std::cmp::min(self.num_kv_computed, self.get_len()), seq_mgr);
    }

    pub fn splice_tokens(
        &mut self,
        seq_mgr: &impl SequenceManager,
        backtrack: usize,
        tokens: &[Token],
    ) {
        self.tokens.truncate(self.get_len() - backtrack);
        self.output_ptr = std::cmp::min(self.output_ptr, self.get_len());
        if backtrack > 0 {
            self.output_pending.clear();
            self.output_pending.extend_from_slice(" ↩ ".as_bytes());
        }
        self.trim_physical_blocks(seq_mgr);
        self.append_tokens(tokens);
    }

    pub fn get_gen_len(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }

    pub fn get_token(&self, idx: usize) -> TokenId {
        self.tokens[idx]
    }

    pub(crate) fn fork_as(
        &self,
        seq_mgr: &impl SequenceManager,
        seq_id: SeqId,
        index: usize,
    ) -> Self {
        seq_mgr.copy(self.seq_id, seq_id, self.num_kv_computed);
        Self {
            seq_id,
            index,
            sched_phase: self.sched_phase,
            num_kv_computed: self.num_kv_computed,
            tokens: self.tokens.clone(),
            output_ptr: self.prompt_len,
            prompt_len: self.prompt_len,
            output_pending: Vec::new(),
            has_aici: self.has_aici,
            aici_logs: Vec::new(),
            pending_fork_ids: Vec::new(),
            aici_sampling: AiciSampling::Regular,
            expected: None,
        }
    }

    pub fn append_tokens(&mut self, tokens: &[Token]) {
        self.tokens.extend_from_slice(tokens)
    }

    pub fn finish_reason(&self) -> Option<FinishReason> {
        match self.sched_phase {
            SchedulingPhase::Finished(reason) => Some(reason),
            _ => None,
        }
    }

    pub fn gen_output(&mut self, tok_trie: &TokTrie) -> SeqOutput {
        let new_output_tokens = self.tokens[self.output_ptr..].to_vec();
        let mut buf = std::mem::take(&mut self.output_pending);
        buf.append(&mut tok_trie.decode(&new_output_tokens));
        if buf.len() > 0 {
            let mut ep = buf.len() - 1;
            if buf[ep] >= 0x80 {
                let mut ln = 0;
                // skip continuation bytes (0b10xx_xxxx), but not too many
                while ln < 4 && buf[ep] & 0b1100_0000 == 0b1000_0000 {
                    if ep == 0 {
                        break;
                    }
                    ep -= 1;
                    ln += 1;
                }
                // now buf[ep] is the first byte of the UTF-8 sequence
                // make sure we have enough continuation bytes
                if (buf[ep] & 0b1110_0000 == 0b1100_0000 && ln >= 1)
                    || (buf[ep] & 0b1111_0000 == 0b1110_0000 && ln >= 2)
                    || (ln >= 3)
                {
                    // OK
                } else {
                    // not enough, move the whole UTF-8 sequence to output_pending
                    self.output_pending.extend(buf.drain(ep..));
                }
            }
        }
        self.output_ptr = self.tokens.len();
        let new_text = String::from_utf8_lossy(&buf).to_string();
        SeqOutput {
            seq_id: self.seq_id.to_num(),
            index: self.index,
            new_output_tokens,
            new_text,
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
    pub deadlock_steps: usize,
    pub sampling_params: SamplingParams,
    pub arrival_time: std::time::Instant,
    pub logits_processor: LogitsProcessor,
    pub max_index: usize,
    pub usage: TokenUsage,
}

impl Debug for SequenceGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequenceGroup")
            .field("request_id", &self.request_id)
            .field("seqs", &self.seqs)
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

    pub fn is_suspended(&self) -> bool {
        self.seqs
            .iter()
            .all(|seq| seq.sched_phase == SchedulingPhase::Suspended || seq.is_finished())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqOutput {
    pub seq_id: usize,
    pub index: usize, // within the sequence group
    pub new_output_tokens: Vec<Token>,
    pub new_text: String,
    /// The tokens generated by the model. Doesn't include prompt tokens.
    pub output_tokens: Vec<Token>,
    pub finish_reason: Option<FinishReason>,
    pub aici_logs: Vec<SequenceResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub gen_tokens: usize,
    pub prompt_tokens: usize,
}

impl TokenUsage {
    pub fn total_tokens(&self) -> usize {
        self.gen_tokens + self.prompt_tokens
    }

    pub fn fuel_tokens(&self) -> usize {
        2 * self.gen_tokens + self.prompt_tokens
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestOutput {
    pub request_id: String,
    pub usage: TokenUsage,
    pub seq_outputs: Vec<SeqOutput>,
    pub is_final: bool,
}

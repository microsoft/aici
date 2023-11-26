use std::{fmt::Debug, marker::PhantomData};

use anyhow::Result;
use candle::Tensor;

use crate::{block::{BlockRef, LogicalTokenBlock}, config::SamplingParams};

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
pub enum SequenceStatus {
    Waiting,
    Running,
    Swapped,
    Finished(FinishReason),
}

#[derive(Debug, Clone, Copy)]
pub enum SeqPhase {
    Prompt,
    Fixed(usize),
    Gen,
}

pub struct Sequence {
    pub seq_id: SeqId,
    pub status: SequenceStatus,
    pub phase: SeqPhase,
    pub tokens: Vec<Token>,
    pub prompt_len: usize,
    pub(crate) phys_blocks: Vec<BlockRef>,
    pub(crate) logical_blocks: Vec<LogicalTokenBlock>,
    _marker: PhantomData<u32>,
    block_size: usize,
}

impl Debug for Sequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequence")
            .field("seq_id", &self.seq_id)
            .field("status", &self.status)
            .field("phase", &self.phase)
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
            status: SequenceStatus::Waiting,
            phase: SeqPhase::Prompt,
            tokens: Vec::new(),
            prompt_len,
            phys_blocks: Vec::new(),
            logical_blocks: Vec::new(),
            block_size,
            _marker: PhantomData,
        };
        seq._append_tokens_to_blocks(tokens);
        seq
    }

    pub(crate) fn fork_as(&self, seq_id: SeqId) -> Self {
        let mut seq = Self {
            seq_id,
            status: self.status,
            phase: self.phase,
            tokens: self.tokens.clone(),
            prompt_len: self.prompt_len,
            phys_blocks: self.phys_blocks.iter().map(|x| x.fork()).collect(),
            logical_blocks: self.logical_blocks.clone(),
            block_size: self.block_size,
            _marker: PhantomData,
        };
        seq._append_tokens_to_blocks(&self.tokens);
        seq
    }

    fn _append_logical_block(&mut self) {
        let block = LogicalTokenBlock::new(self.logical_blocks.len(), self.block_size);
        self.logical_blocks.push(block);
    }

    fn _append_tokens_to_blocks(&mut self, token_ids: &[Token]) {
        let mut cursor = 0;
        self.tokens.extend_from_slice(token_ids);
        while cursor < token_ids.len() {
            if self.logical_blocks.is_empty() {
                self._append_logical_block();
            }

            let last_block = self.logical_blocks.last_mut().unwrap();
            if last_block.is_full() {
                self._append_logical_block();
                continue;
            }

            let num_empty_slots = last_block.get_num_empty_slots();
            let end = std::cmp::min(cursor + num_empty_slots, token_ids.len());
            last_block.append_tokens(&token_ids[cursor..end]);
            cursor = end;
        }
    }

    pub fn append_token_id(&mut self, token_id: Token) {
        self._append_tokens_to_blocks(&[token_id]);
    }
}

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
            let q_len = match seq.phase {
                SeqPhase::Prompt => seq_len,
                SeqPhase::Fixed(len) => len,
                SeqPhase::Gen => 1,
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

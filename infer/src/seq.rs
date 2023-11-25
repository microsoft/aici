use anyhow::Result;
use candle::Tensor;

pub type Token = u32;
pub type SeqId = u32;

pub enum SeqPhase {
    Prompt,
    Fixed(usize),
    Gen,
}

pub struct Sequance {
    pub seq_id: SeqId,
    pub phase: SeqPhase,
    pub tokens: Vec<Token>,
    pub prompt_len: usize,
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
    pub causal: bool,
}

impl BatchInfo {
    pub fn from_seqs(seqs: &[Sequance], device: &candle::Device) -> Result<Self> {
        let mut k_ptr = 0u32;
        let mut q_ptr = 0u32;
        let mut positions = Vec::new();
        let mut seqlens_q = vec![q_ptr];
        let mut seqlens_k = vec![k_ptr];
        let mut tokens = Vec::new();
        let mut max_seqlen_q = 0;
        let mut max_seqlen_k = 0;
        let mut causal = false;
        for seq in seqs {
            let seq_len = seq.tokens.len();
            let k_len = seq_len;
            let q_len = match seq.phase {
                SeqPhase::Prompt => seq_len,
                SeqPhase::Fixed(len) => len,
                SeqPhase::Gen => 1,
            };
            if q_len > 1 {
                causal = true;
            }
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
            causal,
        })
    }
}

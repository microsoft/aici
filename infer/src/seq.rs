use candle::Tensor;

pub type Token = u32;
pub type SeqId = u32;

pub enum SeqPhase {
    Prompt,
    Gen,
}

pub struct Sequance {
    pub seq_id: SeqId,
    pub phase: SeqPhase,
    pub tokens: Vec<Token>,
}

pub struct BatchInfo {
    pub seqlens_q: Tensor,
    pub seqlens_k: Tensor,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
}

use crate::{
    attn::{linear, varlen_attn, RmsNorm, RotaryEmbedding},
    config::{ModelConfig, ModelType},
    engine::RllmModel,
    seq::BatchInfo,
    DType, Device, Tensor,
};
use anyhow::Result;
use serde::Deserialize;
use std::rc::Rc;
use tch::nn::{self, Module, Path};

/// MixFormer model.
/// https://huggingface.co/microsoft/phi-1_5
/// https://arxiv.org/abs/2309.05463
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PhiConfig {
    pub(crate) vocab_size: usize,
    pub(crate) n_positions: usize,
    pub(crate) n_embd: usize,
    pub(crate) n_layer: usize,
    pub(crate) n_inner: Option<usize>,
    pub(crate) n_head: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) activation_function: String,
    pub(crate) layer_norm_epsilon: f64,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) max_position_embeddings: usize,
    // pub(crate) pad_vocab_size_multiple: usize,
}

impl PhiConfig {
    pub fn into_config(self, dtype: DType, device: Device) -> ModelConfig {
        ModelConfig {
            model_type: ModelType::Phi,
            hidden_size: self.n_embd,
            intermediate_size: self.n_inner.unwrap_or(4 * self.n_embd),
            vocab_size: self.vocab_size,
            num_hidden_layers: self.n_layer,
            num_attention_heads: self.n_head,
            num_key_value_heads: self.n_head,
            rms_norm_eps: self.layer_norm_epsilon,
            rope_theta: 10000.0,
            max_sequence_length: self.max_position_embeddings,
            head_dim: self.n_embd / self.n_layer,
            dtype,
            device,
        }
    }
}

#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl MLP {
    fn new(cfg: &ModelConfig, vb: Path) -> Self {
        let n_inner = cfg.intermediate_size;
        let fc1 = linear(cfg.hidden_size, n_inner, &vb / "fc1");
        let fc2 = linear(n_inner, cfg.hidden_size, &vb / "fc2");
        Self { fc1, fc2 }
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1).gelu("tanh").apply(&self.fc2)
    }
}

#[derive(Debug)]
struct CausalLMHead {
    ln: RmsNorm,
    linear: nn::Linear,
}

impl CausalLMHead {
    fn new(cfg: &ModelConfig, vb: Path) -> Self {
        let ln = RmsNorm::from_cfg(&vb / "ln", cfg);
        let linear = linear(cfg.hidden_size, cfg.vocab_size, &vb / "linear");
        Self { ln, linear }
    }
}

impl Module for CausalLMHead {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.ln).apply(&self.linear)
    }
}

#[derive(Debug)]
struct MHA {
    wqkv: nn::Linear,
    out_proj: nn::Linear,
    rotary_emb: RotaryEmbedding,
    config: Rc<ModelConfig>,
    block_idx: usize,
}

impl MHA {
    fn new(cfg: &Rc<ModelConfig>, block_idx: usize, vb: Path) -> Self {
        let op_size = cfg.hidden_size;
        let wqkv = linear(cfg.hidden_size, 3 * op_size, &vb / "Wqkv");
        let out_proj = linear(op_size, cfg.hidden_size, &vb / "out_proj");
        let rotary_emb = RotaryEmbedding::new(cfg);
        Self {
            wqkv,
            out_proj,
            rotary_emb,
            config: cfg.clone(),
            block_idx,
        }
    }

    fn forward(&self, xs: &Tensor, batch_info: &mut BatchInfo) -> Tensor {
        let (b_size, seq_len, _hidden_size) = xs.size3().unwrap();

        let ((q, k), v) = {
            let qkv = self.wqkv.forward(xs).reshape(&[
                b_size,
                seq_len,
                3,
                -1,
                self.config.head_dim as i64,
            ]);

            let mut qkv = qkv.chunk(3, -1);
            let v = qkv.pop().unwrap();

            (
                self.rotary_emb
                    .apply(&batch_info.positions, &qkv[0], &qkv[1]),
                v,
            )
        };

        let y = varlen_attn(&self.config, q, k, v, batch_info, self.block_idx);
        self.out_proj.forward(&y)
    }
}

#[derive(Debug)]
struct ParallelBlock {
    ln: RmsNorm,
    mixer: MHA,
    mlp: MLP,
}

impl ParallelBlock {
    fn new(cfg: &Rc<ModelConfig>, vb: Path, block_idx: usize) -> Self {
        let ln = RmsNorm::from_cfg(&vb / "ln", cfg);
        let mixer = MHA::new(cfg, block_idx, &vb / "mixer");
        let mlp = MLP::new(cfg, &vb / "mlp");
        Self { ln, mixer, mlp }
    }

    fn forward(&self, xs: &Tensor, batch_info: &mut BatchInfo) -> Tensor {
        let residual = xs;
        let xs = xs.apply(&self.ln);
        let attn_outputs = self.mixer.forward(&xs, batch_info);
        let feed_forward_hidden_states = self.mlp.forward(&xs);
        attn_outputs + feed_forward_hidden_states + residual
    }
}

#[derive(Debug)]
pub struct MixFormerSequentialForCausalLM {
    embedding: nn::Embedding,
    blocks: Vec<ParallelBlock>,
    head: CausalLMHead,
}

impl MixFormerSequentialForCausalLM {
    pub fn new(cfg: &Rc<ModelConfig>, vb: Path) -> Self {
        let vb = vb / "layers";
        let embedding = nn::embedding(
            &vb / "wte",
            cfg.vocab_size as i64,
            cfg.hidden_size as i64,
            Default::default(),
        );
        let mut blocks = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let block = ParallelBlock::new(cfg, &vb / (i + 1), i);
            blocks.push(block)
        }
        let head = CausalLMHead::new(cfg, &vb / (cfg.num_hidden_layers + 1));
        Self {
            embedding,
            blocks,
            head,
        }
    }
}

impl RllmModel for MixFormerSequentialForCausalLM {
    fn forward(&self, batch_info: &mut BatchInfo) -> Result<Tensor> {
        let seq_len = batch_info.tokens.numel() as i64;
        let mut xs = self.embedding.forward(&batch_info.tokens);
        for block in self.blocks.iter() {
            xs = block.forward(&xs, batch_info);
        }
        Ok(xs
            .narrow(1, seq_len - 1, 1)
            .apply(&self.head)
            .squeeze_dim(1))
    }
}

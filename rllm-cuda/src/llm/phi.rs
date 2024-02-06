use super::{
    config::{CommonModelConfig, ModelConfig, ModelType, RllmModelConfig},
    layer_norm, linear, varlen_attn, RotaryEmbedding,
};
use crate::paged::BatchInfo;
use serde::Deserialize;
use std::rc::Rc;
use tch::{
    nn::{self, Module, Path},
    IndexOp, Tensor,
};

use super::tmodel::TModelInner;

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
    pub(crate) torch_dtype: String,
}

impl RllmModelConfig for PhiConfig {
    fn into_config(self, common: CommonModelConfig) -> ModelConfig {
        let mut meta = common.meta.clone();
        meta.vocab_size = self.vocab_size;
        meta.tok_vocab_size = self.vocab_size;
        meta.max_sequence_length = self.n_positions;
        ModelConfig {
            model_type: ModelType::Phi,
            meta: common.meta,
            hidden_size: self.n_embd,
            intermediate_size: self.n_inner.unwrap_or(4 * self.n_embd),
            num_hidden_layers: self.n_layer,
            num_attention_heads: self.n_head,
            num_key_value_heads: self.n_head,
            layer_norm_eps: self.layer_norm_epsilon,
            rope_theta: 10000.0,
            head_dim: self.n_embd / self.n_head,
            rotary_dim: self.rotary_dim,
            dtype: ModelConfig::dtype_from_str(common.dtype, &self.torch_dtype),
            device: common.device,
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
    ln: nn::LayerNorm,
    linear: nn::Linear,
}

impl CausalLMHead {
    fn new(cfg: &ModelConfig, vb: Path) -> Self {
        let ln = layer_norm(&vb / "ln", cfg);
        let linear = linear(cfg.hidden_size, cfg.meta.vocab_size, &vb / "linear");
        Self { ln, linear }
    }
}

impl Module for CausalLMHead {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = self.ln.forward(xs);
        let xs = self.linear.forward(&xs);
        xs
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
        let (seq_len, _hidden_size) = xs.size2().unwrap();

        let ((q, k), v) = {
            let qkv = self
                .wqkv
                .forward(xs)
                .reshape(&[seq_len, 3, -1, self.config.head_dim as i64]);

            let mut qkv = qkv.chunk(3, 1);
            let v = qkv.pop().unwrap();

            (
                self.rotary_emb
                    .forward(&batch_info.positions, &qkv[0], &qkv[1]),
                v.squeeze_dim(1),
            )
        };

        let y = varlen_attn(&self.config, q, k, v, batch_info, self.block_idx);

        self.out_proj.forward(&y)
    }
}

#[derive(Debug)]
struct ParallelBlock {
    ln: nn::LayerNorm,
    mixer: MHA,
    mlp: MLP,
}

impl ParallelBlock {
    fn new(cfg: &Rc<ModelConfig>, mut vb: Path, block_idx: usize) -> Self {
        let ln = layer_norm(&vb / "ln", cfg);
        let mixer = MHA::new(cfg, block_idx, &vb / "mixer");
        let mlp = MLP::new(cfg, &vb / "mlp");
        // this optimizes memory usage
        vb.set_kind(cfg.dtype);
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
    config: Rc<ModelConfig>,
}

impl MixFormerSequentialForCausalLM {
    pub fn new(cfg: &Rc<ModelConfig>, vb0: Path) -> Self {
        let vb = &vb0 / "transformer";
        let embedding = nn::embedding(
            &vb / "embd" / "wte",
            cfg.meta.vocab_size as i64,
            cfg.hidden_size as i64,
            Default::default(),
        );
        let mut blocks = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let block = ParallelBlock::new(cfg, &vb / "h" / i, i);
            blocks.push(block)
        }
        let head = CausalLMHead::new(cfg, &vb0 / "lm_head");
        Self {
            embedding,
            blocks,
            head,
            config: cfg.clone(),
        }
    }
}

impl TModelInner for MixFormerSequentialForCausalLM {
    fn forward(&self, batch_info: &mut BatchInfo) -> Tensor {
        let mut xs = self.embedding.forward(&batch_info.tokens);
        for block in self.blocks.iter() {
            xs = block.forward(&xs, batch_info);
        }
        let r = self.head.forward(&xs);

        // it should approximately match...
        let tok_size = self.config.meta.tok_vocab_size as i64;
        if r.size()[1] < tok_size || r.size()[1] > tok_size + 1000 {
            panic!(
                "unexpected logits size: {:?} ({}/{})",
                r.size(),
                tok_size,
                self.config.meta.vocab_size
            );
        }

        let r = r.i((.., 0..tok_size));
        batch_info.extract_positions(&r)
    }
}

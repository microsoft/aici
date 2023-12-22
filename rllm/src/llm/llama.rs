// based on https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/llama.rs

use crate::{
    config::{CommonModelConfig, ModelConfig, ModelType},
    engine::{RllmModel, RllmModelConfig},
    llm::{linear_no_bias, varlen_attn, RmsNorm, RotaryEmbedding},
    paged::BatchInfo,
    Tensor,
};
use anyhow::Result;
use serde::Deserialize;
use std::rc::Rc;
use tch::{
    nn::{self, Module, Path},
    IndexOp,
};

#[derive(Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize, // TODO - is this max seq len?
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub torch_dtype: String,
}

fn default_rope() -> f32 {
    10_000.0
}

impl RllmModelConfig for LlamaConfig {
    fn into_config(self, common: CommonModelConfig) -> ModelConfig {
        let head_dim = self.hidden_size / self.num_attention_heads;
        ModelConfig {
            model_type: ModelType::Llama,
            meta: common.meta,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            tok_vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            layer_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_sequence_length: self.max_position_embeddings,
            head_dim,
            rotary_dim: head_dim,
            dtype: ModelConfig::dtype_from_str(common.dtype, &self.torch_dtype),
            device: common.device,
        }
    }
}

struct CausalSelfAttention {
    q_proj: Option<nn::Linear>,
    k_proj: Option<nn::Linear>,
    v_proj: Option<nn::Linear>,
    // after .finalize()
    qkv_proj: Option<Tensor>,
    o_proj: nn::Linear,
    config: Rc<ModelConfig>,
    rotary: RotaryEmbedding,
}

impl CausalSelfAttention {
    fn forward(&self, x: &Tensor, batch_info: &mut BatchInfo, block_idx: usize) -> Tensor {
        let (b_sz, seq_len, hidden_size) = x.size3().unwrap();
        assert!(b_sz == 1);

        batch_info.log_tensor("x", &x);

        let kv_heads = self.config.num_key_value_heads as i64;
        let q_heads = self.config.num_attention_heads as i64;

        let qkv = x
            .linear(&self.qkv_proj.as_ref().unwrap(), None::<&Tensor>)
            .reshape(&[seq_len, q_heads + 2 * kv_heads, self.config.head_dim as i64]);
        let q = qkv.i((.., 0..q_heads, ..)).reshape(&[seq_len, -1]);
        let k = qkv
            .i((.., q_heads..(kv_heads + q_heads), ..))
            .reshape(&[seq_len, -1]);
        let v = qkv
            .i((.., (kv_heads + q_heads).., ..))
            .reshape(&[seq_len, -1]);

        let (q, k) = self.rotary.forward(&batch_info.positions, &q, &k);

        let v = v.reshape(&[
            seq_len,
            self.config.num_key_value_heads as i64,
            self.config.head_dim as i64,
        ]);

        let y = varlen_attn(&self.config, q, k, v, batch_info, block_idx);

        let y = y.reshape(&[b_sz, seq_len, hidden_size]);
        let y = self.o_proj.forward(&y);

        batch_info.log_tensor("yp", &y);

        y
    }

    fn load(vb: Path, rotary: &RotaryEmbedding, cfg: &Rc<ModelConfig>) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = Some(linear_no_bias(size_in, size_q, &vb / "q_proj"));
        let k_proj = Some(linear_no_bias(size_in, size_kv, &vb / "k_proj"));
        let v_proj = Some(linear_no_bias(size_in, size_kv, &vb / "v_proj"));
        let o_proj = linear_no_bias(size_q, size_in, &vb / "o_proj");
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            qkv_proj: None,
            o_proj,
            config: cfg.clone(),
            rotary: rotary.clone(),
        })
    }
}

struct Mlp {
    c_fc: Option<Tensor>,
    c_fc1: Option<nn::Linear>,
    c_fc2: Option<nn::Linear>,
    c_proj: nn::Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor, batch_info: &BatchInfo) -> Tensor {
        let m = x.linear(&self.c_fc.as_ref().unwrap(), None::<&Tensor>);
        let sz = m.size()[2];
        let m1 = m.i((.., .., 0..(sz / 2)));
        let m2 = m.i((.., .., (sz / 2)..));
        batch_info.log_tensor("m1", &m1);
        batch_info.log_tensor("m2", &m2);
        let si = m1.silu();
        batch_info.log_tensor("si", &m2);
        let x = si * &m2;
        self.c_proj.forward(&x)
    }

    fn load(vb: Path, cfg: &ModelConfig) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = Some(linear_no_bias(h_size, i_size, &vb / "gate_proj"));
        let c_fc2 = Some(linear_no_bias(h_size, i_size, &vb / "up_proj"));
        let c_proj = linear_no_bias(i_size, h_size, &vb / "down_proj");
        Ok(Self {
            c_fc1,
            c_fc2,
            c_fc: None,
            c_proj,
        })
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn forward(&self, x: &Tensor, batch_info: &mut BatchInfo, block_idx: usize) -> Tensor {
        let residual = x;
        let x = self.rms_1.forward(x);
        let x = self.attn.forward(&x, batch_info, block_idx) + residual;
        let residual = &x;
        batch_info.log_tensor("x0", &x);
        let x = self.rms_2.forward(&x);
        batch_info.log_tensor("x1", &x);
        let x = self.mlp.forward(&x, batch_info);
        batch_info.log_tensor("x2", &x);
        let x = x + residual;
        batch_info.log_tensor("x3", &x);
        x
    }

    fn load(mut vb: Path, rotary: &RotaryEmbedding, cfg: &Rc<ModelConfig>) -> Result<Self> {
        let attn = CausalSelfAttention::load(&vb / "self_attn", rotary, cfg)?;
        let mlp = Mlp::load(&vb / "mlp", cfg)?;
        let rms_1 = RmsNorm::from_cfg(&vb / "input_layernorm", cfg);
        let rms_2 = RmsNorm::from_cfg(&vb / "post_attention_layernorm", cfg);
        // this optimizes memory usage
        vb.set_kind(cfg.dtype);
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }
}

pub struct Llama {
    wte: nn::Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: nn::Linear,
}

impl RllmModel for Llama {
    fn forward(&self, batch_info: &mut BatchInfo) -> Tensor {
        let mut x = self.wte.forward(&batch_info.tokens).unsqueeze(0);
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, batch_info, block_idx);
        }
        let x0 = self.ln_f.forward(&x);
        // println!("x: {}", x0);
        let x = batch_info.extract_positions(&x0.squeeze_dim(0));
        let logits = self.lm_head.forward(&x);
        logits
    }

    fn finalize(&mut self) {
        for block in self.blocks.iter_mut() {
            // combine q, k, v projections into one tensor for performance
            block.attn.qkv_proj = Some(
                Tensor::cat(
                    &[
                        &block.attn.q_proj.as_ref().unwrap().ws,
                        &block.attn.k_proj.as_ref().unwrap().ws,
                        &block.attn.v_proj.as_ref().unwrap().ws,
                    ],
                    0,
                )
                .contiguous(),
            );

            // free memory:
            block.attn.q_proj = None;
            block.attn.k_proj = None;
            block.attn.v_proj = None;

            block.mlp.c_fc = Some(
                Tensor::cat(
                    &[
                        &block.mlp.c_fc1.as_ref().unwrap().ws,
                        &block.mlp.c_fc2.as_ref().unwrap().ws,
                    ],
                    0,
                )
                .contiguous(),
            );

            block.mlp.c_fc1 = None;
            block.mlp.c_fc2 = None;
        }
    }
}

impl Llama {
    pub fn load(vs: Path, cfg: &Rc<ModelConfig>) -> Result<Self> {
        let rotary = RotaryEmbedding::new(cfg);

        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, &vs / "lm_head");

        let wte = nn::embedding(
            &vs / "model" / "embed_tokens",
            cfg.vocab_size as i64,
            cfg.hidden_size as i64,
            Default::default(),
        );

        let ln_f = RmsNorm::from_cfg(&vs / "model" / "norm", cfg);

        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(&vs / "model" / "layers" / i, &rotary, cfg).unwrap())
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }
}

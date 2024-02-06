pub mod config;
pub mod kernels;
pub mod llama;
pub mod loader;
pub mod phi;
pub mod refkernels;
pub mod seqid;
pub mod tmodel;
pub mod util;

use self::config::ModelConfig;
use crate::{
    llm::util::{check_all_close, check_all_close_attn},
    paged::BatchInfo,
};
use std::rc::Rc;
use tch::{
    nn::{self, Module, Path},
    IndexOp, Tensor,
};

// note that this doesn't work for phi-2 - it seems particularly numerically unstable
const CHECK: bool = false;

pub type DType = tch::Kind;

#[derive(Debug)]
pub struct RotaryEmbedding {
    config: Rc<ModelConfig>,
    cos_sin: Tensor,
}

impl Clone for RotaryEmbedding {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            cos_sin: self.cos_sin.shallow_clone(),
        }
    }
}

impl RotaryEmbedding {
    pub fn new(config: &Rc<ModelConfig>) -> Self {
        // pre-compute freqs_cis
        let rotary_dim = config.rotary_dim;
        let theta: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / rotary_dim as f32))
            .collect();
        let theta = Tensor::from_slice(theta.as_slice()).to(config.device);
        let len = config.meta.max_sequence_length as i64;
        let idx_theta = Tensor::arange(len, (DType::Float, config.device))
            .reshape(&[len, 1])
            .matmul(&theta.reshape(&[1, theta.numel() as i64]));
        let cos = idx_theta.cos().to_kind(config.dtype);
        let sin = idx_theta.sin().to_kind(config.dtype);
        let cos_sin = Tensor::cat(&[&cos, &sin], -1).contiguous();
        Self {
            config: config.clone(),
            cos_sin,
        }
    }

    pub fn forward(
        &self,
        positions: &Tensor, // [num_tokens]
        q: &Tensor,         // [num_tokens, num_heads * head_size]
        k: &Tensor,         // [num_tokens, num_kv_heads * head_size]
    ) -> (Tensor, Tensor) {
        // println!("q: {q:?}");
        // println!("k: {k:?}");
        let mut q = q.reshape(&[
            -1,
            (self.config.num_attention_heads * self.config.head_dim) as i64,
        ]);
        let mut k = k.reshape(&[
            -1,
            (self.config.num_key_value_heads * self.config.head_dim) as i64,
        ]);
        let num_tokens = q.size()[0];
        assert!(num_tokens == k.size()[0]);

        // println!("q: {q:?}");
        // println!("k: {k:?}");
        // println!("c: {:?}", &self.cache.cos_sin);

        if CHECK {
            let mut qq = q.copy();
            let mut kk = k.copy();
            kernels::rotary_embedding(
                &positions,
                &mut q,
                &mut k,
                self.config.head_dim,
                &self.cos_sin,
                true,
            );
            refkernels::rotary_embedding(
                &positions,
                &mut qq,
                &mut kk,
                self.config.head_dim,
                &self.cos_sin,
                true,
            );
            check_all_close(&q, &qq, 1e-5);
            check_all_close(&k, &kk, 1e-5);
        } else {
            kernels::rotary_embedding(
                &positions,
                &mut q,
                &mut k,
                self.config.head_dim,
                &self.cos_sin,
                true,
            );
        }

        let q = q.reshape(&[
            num_tokens,
            self.config.num_attention_heads as i64,
            self.config.head_dim as i64,
        ]);
        let k = k.reshape(&[
            num_tokens,
            self.config.num_key_value_heads as i64,
            self.config.head_dim as i64,
        ]);

        (q, k)
    }
}

fn save_attn(
    _config: &ModelConfig,
    k: &Tensor,
    v: &Tensor,
    batch_info: &mut BatchInfo,
    block_idx: usize,
) {
    let (mut key_cache, mut value_cache) = batch_info.kv_cache.get(block_idx);

    assert!(v.size() == k.size());

    // first, stuff the query-sized key/value into the cache
    if CHECK {
        let mut kk = key_cache.copy();
        let mut vv = value_cache.copy();
        kernels::reshape_and_cache(
            k,
            v,
            &mut key_cache,
            &mut value_cache,
            &batch_info.slot_mapping,
        );
        refkernels::reshape_and_cache(k, v, &mut kk, &mut vv, &batch_info.slot_mapping);
        check_all_close(&key_cache, &kk, 1e-5);
        check_all_close(&value_cache, &vv, 1e-5);
    } else {
        kernels::reshape_and_cache(
            k,
            v,
            &mut key_cache,
            &mut value_cache,
            &batch_info.slot_mapping,
        );
    }
}

fn compute_varlen_attn(
    config: &ModelConfig,
    q: &Tensor,
    batch_info: &mut BatchInfo,
    block_idx: usize,
) -> Tensor {
    let (key_cache, value_cache) = batch_info.kv_cache.get(block_idx);

    if q.size()[0] == 0 {
        return Tensor::empty(&[0, config.hidden_size as i64], (q.kind(), q.device()));
    }

    // then, extend key/value and fill them from cache
    let mut k = Tensor::empty(
        &[
            batch_info.gather_mapping.size()[0],
            config.num_key_value_heads as i64,
            config.head_dim as i64,
        ],
        (q.kind(), q.device()),
    );

    let mut v = k.empty_like();
    kernels::gather_cached_kv(
        &mut k,
        &mut v,
        &key_cache,
        &value_cache,
        &batch_info.gather_mapping,
    );

    if CHECK {
        let mut kk = k.empty_like();
        let mut vv = v.empty_like();

        refkernels::gather_cached_kv(
            &mut kk,
            &mut vv,
            &key_cache,
            &value_cache,
            &batch_info.gather_mapping,
        );
        check_all_close(&k, &kk, 1e-5);
        check_all_close(&v, &vv, 1e-5);
    }

    let k = repeat_kv(config, k);
    let v = repeat_kv(config, v);

    let y = {
        batch_info.log_tensor("q", &q);
        batch_info.log_tensor("k", &k);
        batch_info.log_tensor("v", &v);

        // flash-attn expects (seq_len, nheads, head_dim)
        let softmax_scale = 1f32 / (config.head_dim as f32).sqrt();

        let causal = true;

        let y = if config.dtype == DType::BFloat16 || config.dtype == DType::Half {
            let y = kernels::varlen_attn(
                &q,
                &k,
                &v,
                &batch_info.seqlens_q,
                &batch_info.seqlens_k,
                batch_info.max_seqlen_q,
                batch_info.max_seqlen_k,
                softmax_scale,
                causal,
            );

            if CHECK {
                let y2 = refkernels::varlen_attn(
                    &q,
                    &k,
                    &v,
                    &batch_info.seqlens_q,
                    &batch_info.seqlens_k,
                    batch_info.max_seqlen_q,
                    batch_info.max_seqlen_k,
                    softmax_scale,
                    causal,
                );
                check_all_close_attn(&y, &y2);
            }

            y
        } else {
            refkernels::varlen_attn(
                &q,
                &k,
                &v,
                &batch_info.seqlens_q,
                &batch_info.seqlens_k,
                batch_info.max_seqlen_q,
                batch_info.max_seqlen_k,
                softmax_scale,
                causal,
            )
        };

        y
    };

    batch_info.log_tensor("y", &v);

    let y = y.reshape(&[-1, config.hidden_size as i64]);

    y
}

#[cfg(feature = "cuda")]
fn compute_paged_attn(
    config: &ModelConfig,
    q: &Tensor,
    y: &Tensor,
    batch_info: &mut BatchInfo,
    block_idx: usize,
) -> Tensor {
    use tch_cuda::paged_attention_v1;

    if q.size()[0] == 0 {
        return y.shallow_clone();
    }

    let mut out = Tensor::empty_like(q);
    let (key_cache, value_cache) = batch_info.kv_cache.get(block_idx);

    let softmax_scale = 1f32 / (config.head_dim as f32).sqrt();

    paged_attention_v1(
        &mut out,
        &q,
        &key_cache,
        &value_cache,
        config.num_key_value_heads,
        softmax_scale,
        &batch_info.paged_block_tables,
        &batch_info.paged_context_lens,
        batch_info.paged_block_size,
        batch_info.paged_max_context_len,
        None,
    );

    let out = out.reshape(&[-1, config.hidden_size as i64]);

    Tensor::cat(&[y, &out], 0)
}

pub fn varlen_attn(
    config: &ModelConfig,
    q: Tensor, // [num_tokens, num_heads, head_size]
    k: Tensor, // [num_tokens, num_heads, head_size]
    v: Tensor, // [num_tokens, num_heads, head_size]
    batch_info: &mut BatchInfo,
    block_idx: usize,
) -> Tensor // [num_tokens, num_heads * head_size]
{
    // println!("varlen_attn: q: {q:?} k: {k:?} v: {v:?}");
    // doesn't hold for GQA:
    // assert!(q.size() == k.size());
    // assert!(v.size() == k.size());

    save_attn(config, &k, &v, batch_info, block_idx);

    let y = compute_varlen_attn(
        config,
        &q.i((0..batch_info.q_multi, .., ..)),
        batch_info,
        block_idx,
    );

    #[cfg(not(feature = "cuda"))]
    assert!(q.i((batch_info.q_multi.., .., ..)).numel() == 0);

    #[cfg(feature = "cuda")]
    let y = compute_paged_attn(
        config,
        &q.i((batch_info.q_multi.., .., ..)),
        &y,
        batch_info,
        block_idx,
    );

    y
}

// x is [seq_len, num_heads, head_dim]
fn repeat_kv(config: &ModelConfig, x: Tensor) -> Tensor {
    let n_rep = config.num_attention_heads / config.num_key_value_heads;
    if n_rep == 1 {
        x
    } else {
        let dim = 1;
        x.repeat_interleave_self_int(n_rep as i64, Some(dim), None)
    }
}

#[derive(Debug)]
pub struct RmsNorm {
    scale: Tensor,
    size: i64,
    eps: f64,
}

impl RmsNorm {
    pub fn from_cfg(vs: nn::Path, config: &ModelConfig) -> Self {
        Self::new(vs, config.hidden_size, Some(config.layer_norm_eps))
    }

    pub fn new(vs: nn::Path, size: usize, eps: Option<f64>) -> Self {
        let scale = vs.zeros("weight", &[size as i64]);
        Self {
            scale,
            size: size as i64,
            eps: eps.unwrap_or(1e-5),
        }
    }
}

impl Module for RmsNorm {
    // TODO use kernels::rms_norm
    fn forward(&self, xs: &Tensor) -> Tensor {
        let k = xs.kind();
        let xs = xs.to_kind(DType::Float);
        let variance = (&xs * &xs).mean_dim(-1, true, xs.kind());
        let xs_normed = xs * (variance + self.eps).rsqrt();
        let scale = self.scale.reshape([1, 1, self.size]);
        scale * xs_normed.to_kind(k)
    }
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: Path) -> nn::Linear {
    let c = nn::LinearConfig {
        bias: false,
        ..Default::default()
    };
    nn::linear(vb, in_dim as i64, out_dim as i64, c)
}

pub fn linear(in_dim: usize, out_dim: usize, vs: Path) -> nn::Linear {
    nn::linear(
        vs,
        in_dim as i64,
        out_dim as i64,
        nn::LinearConfig::default(),
    )
}

pub fn layer_norm(vs: nn::Path, config: &ModelConfig) -> nn::LayerNorm {
    nn::layer_norm(
        vs,
        vec![config.hidden_size as i64],
        nn::LayerNormConfig {
            eps: config.layer_norm_eps,
            ..nn::LayerNormConfig::default()
        },
    )
}

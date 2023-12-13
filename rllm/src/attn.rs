use crate::{config::ModelConfig, kernels, seq::BatchInfo};
use crate::{util::to_vec1, DType, IndexOp, Tensor};
use anyhow::Result;
use std::rc::Rc;
use tch::nn::{self, Module, Path};

pub fn naive_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let seqlens_q = to_vec1::<i32>(seqlens_q);
    let seqlens_k = to_vec1::<i32>(seqlens_k);
    assert!(seqlens_q.len() == seqlens_k.len());
    let batch_size = seqlens_k.len() - 1;

    let softmax_scale = softmax_scale as f64;

    // flash-attn expects (seq_len, nheads, head_dim)

    let mut attns = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let ptr_q = seqlens_q[i] as i64;
        let ptr_k = seqlens_k[i] as i64;
        let len_q = seqlens_q[i + 1] as i64 - ptr_q;
        let len_k = seqlens_k[i + 1] as i64 - ptr_k;

        assert!(len_q <= max_seqlen_q as i64);
        assert!(len_k <= max_seqlen_k as i64);

        let q = q.i((ptr_q..ptr_q + len_q, .., ..)).transpose(0, 1);
        let k = k.i((ptr_k..ptr_k + len_k, .., ..)).transpose(0, 1);
        let v = v.i((ptr_k..ptr_k + len_k, .., ..)).transpose(0, 1);

        let mut attn = q.contiguous().matmul(&k.transpose(-2, -1).contiguous());
        attn = attn * softmax_scale;
        if causal {
            let mask: Vec<_> = (0..len_q)
                .flat_map(|i| {
                    (0..len_k).map(move |j| {
                        if i + (len_k - len_q) >= j {
                            0f32
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect();
            let mask = Tensor::from_slice(&mask)
                .to(q.device())
                .reshape(&[len_q, len_k])
                .to_kind(attn.kind());

            // println!("mask: {mask}");
            // TODO broadcast?
            attn = attn + mask;
        }

        attn = attn.softmax(-1, attn.kind());
        // Convert to contiguous as matmul doesn't support strided vs for now.
        attn = attn.matmul(&v.contiguous());
        attn = attn.transpose(0, 1);

        attns.push(attn);
    }

    let attn = Tensor::cat(&attns, 0);
    Ok(attn)
}

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
        // precompute freqs_cis
        let rotary_dim = config.hidden_size / config.num_attention_heads;
        let theta: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / rotary_dim as f32))
            .collect();
        let theta = Tensor::from_slice(theta.as_slice()).to(config.device);
        let len = config.max_sequence_length as i64;
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

    pub fn apply(
        &self,
        positions: &Tensor, // [num_tokens]
        q: &Tensor,         // [num_tokens, num_heads * head_size]
        k: &Tensor,         // [num_tokens, num_kv_heads * head_size]
    ) -> (Tensor, Tensor) {
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

        kernels::rotary_embedding(
            &positions,
            &mut q,
            &mut k,
            self.config.head_dim,
            &self.cos_sin,
        );

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

pub fn varlen_attn(
    config: &ModelConfig,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    batch_info: &mut BatchInfo,
    block_idx: usize,
) -> Tensor {
    let trace = false;

    let (key_cache, value_cache) = &mut batch_info.kv_cache[block_idx];

    // first, stuff the query-sized key/value into the cache
    kernels::reshape_and_cache(&k, &v, key_cache, value_cache, &batch_info.slot_mapping);

    // then, extend key/value and fill them from cache
    let mut k = Tensor::empty(
        &[
            batch_info.gather_mapping.size()[0],
            config.num_key_value_heads as i64,
            config.head_dim as i64,
        ],
        (k.kind(), k.device()),
    );

    let mut v = k.empty_like();
    kernels::gather_cached_kv(
        &mut k,
        &mut v,
        key_cache,
        value_cache,
        &batch_info.gather_mapping,
    );

    let k = repeat_kv(config, k);
    let v = repeat_kv(config, v);

    if trace {
        println!("q2: {q:?}\n{q}");
        println!("k2: {k:?}\n{k}");
        println!("v2: {v:?}\n{v}");
    }

    let y = {
        batch_info.log_tensor("q", &q);
        batch_info.log_tensor("k", &k);
        batch_info.log_tensor("v", &v);

        // flash-attn expects (seq_len, nheads, head_dim)
        let softmax_scale = 1f32 / (config.head_dim as f32).sqrt();
        if trace {
            println!("Q {q:?} K {k:?} V {v:?}");
        }
        let causal = true;
        let y = kernels::flash_attn_varlen(
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

        y
    };

    batch_info.log_tensor("y", &v);

    if trace {
        println!("y: {y:?}\n{y}");
    }

    y
}

fn repeat_kv(config: &ModelConfig, x: Tensor) -> Tensor {
    let n_rep = config.num_attention_heads / config.num_key_value_heads;
    if n_rep == 1 {
        x
    } else {
        // let (b_sz, n_kv_head, seq_len, head_dim) = x.size4().unwrap();
        // let _x = x
        //     .unsqueeze(2)
        //     .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))
        //     .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim));
        todo!("dims are wrong")
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
        Self::new(vs, config.hidden_size, Some(config.rms_norm_eps))
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
    fn forward(&self, xs: &Tensor) -> Tensor {
        let norm_xs = (xs * xs).mean_dim(-1, true, xs.kind());
        let xs_normed = xs * (norm_xs + self.eps).rsqrt();
        let scale = self.scale.reshape([1, 1, self.size]);
        scale * xs_normed
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

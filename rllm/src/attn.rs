use crate::{util::to_vec1, DType, Device, IndexOp, Tensor};
use anyhow::Result;
// use candle_core::Result;
// use candle_nn::{linear_no_bias, Embedding, Linear, Module, RmsNorm, VarBuilder};
use serde::Deserialize;
use tch::nn::{self, Module, Path};

use crate::{config::ModelConfig, get_trace, kernels, seq::BatchInfo};


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

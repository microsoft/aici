use tch::{Device, IndexOp, Tensor};

#[cfg(feature = "cuda")]
pub use tch_flash_attn::*;

use crate::util::to_vec1;

/// Convert a vector of lengths into a tensor of offsets, as expected by flash attn.
pub fn to_offsets(seqlens: &[usize], device: Device) -> (usize, Tensor) {
    let mut offsets = Vec::with_capacity(seqlens.len() + 1);
    let mut offset = 0;
    let mut max = 0;
    for len in seqlens {
        max = std::cmp::max(*len, max);
        offsets.push(offset as i32);
        offset += len;
    }
    offsets.push(offset as i32);
    (max, Tensor::from_slice(offsets.as_slice()).to(device))
}

#[cfg(not(feature = "cuda"))]
pub use crate::kernels::reference_gather_cached_kv as gather_cached_kv;
#[cfg(not(feature = "cuda"))]
pub use crate::kernels::reference_reshape_and_cache as reshape_and_cache;
#[cfg(not(feature = "cuda"))]
pub use crate::kernels::reference_varlen_attn as flash_attn_varlen;

pub fn reference_reshape_and_cache(
    key: &Tensor,             // [num_tokens, num_heads, head_size]
    value: &Tensor,           // [num_tokens, num_heads, head_size]
    key_cache: &mut Tensor,   // [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache: &mut Tensor, // [num_blocks, num_heads, head_size, block_size]
    slot_mapping: &Tensor,    // [num_tokens], int
) {
    let (_num_blocks, num_heads, head_size_x, block_size, x) = key_cache.size5().unwrap();

    let slot_mapping = to_vec1::<i32>(slot_mapping);
    let key_reshaped = key.reshape(&[-1, num_heads, head_size_x, x]);

    for (idx, slot) in slot_mapping.iter().enumerate() {
        let slot = *slot as i64;
        let idx = idx as i64;
        let block_idx = slot / block_size;
        let block_off = slot % block_size;

        key_cache
            .i((block_idx, .., .., block_off, ..))
            .copy_(&key_reshaped.i(idx));
        value_cache
            .i((block_idx, .., .., block_off))
            .copy_(&value.i(idx));
    }
}

pub fn reference_gather_cached_kv(
    key: &mut Tensor,      // [num_tokens, num_heads, head_size]
    value: &mut Tensor,    // [num_tokens, num_heads, head_size]
    key_cache: &Tensor,    // [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache: &Tensor,  // [num_blocks, num_heads, head_size, block_size]
    slot_mapping: &Tensor, // [num_tokens], int
) {
    let (_num_blocks, num_heads, head_size_x, block_size, x) = key_cache.size5().unwrap();

    let slot_mapping = to_vec1::<i32>(slot_mapping);
    let key_reshaped = key.view([-1, num_heads, head_size_x, x]);

    for (idx, slot) in slot_mapping.iter().enumerate() {
        let slot = *slot as i64;
        let idx = idx as i64;
        let block_idx = slot / block_size;
        let block_off = slot % block_size;

        key_reshaped
            .i(idx)
            .copy_(&key_cache.i((block_idx, .., .., block_off, ..)));
        value
            .i(idx)
            .copy_(&value_cache.i((block_idx, .., .., block_off)));
    }
}

pub fn reference_varlen_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Tensor {
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
    attn
}

pub fn reference_rotary_embedding(
    positions: &Tensor,  // [num_tokens]
    query0: &mut Tensor, // [num_tokens, num_heads * head_size]
    key0: &mut Tensor,   // [num_tokens, num_kv_heads * head_size]
    head_size: usize,
    cos_sin_cache: &Tensor, // [max_position, rot_dim]
    is_neox: bool,
) {
    let head_size = head_size as i64;
    let rot_dim = cos_sin_cache.size()[1];
    let num_tokens = query0.size()[0];

    assert!(rot_dim >= head_size); // otherwise we need to limit to first rot_dim positions

    let query = query0.view([num_tokens, -1, head_size]);
    let key = key0.view([num_tokens, -1, head_size]);

    let query_rot = query.i((.., .., 0..rot_dim));
    let key_rot = key.i((.., .., 0..rot_dim));

    let cos_sin = cos_sin_cache.i((positions, ..)).to(query.device());
    let mut cos_sin = cos_sin.chunk(2, -1);
    let mut sin = cos_sin.pop().unwrap();
    let mut cos = cos_sin.pop().unwrap();

    let rotate_fn = if is_neox {
        cos = cos.repeat(&[1, 1, 2]).unsqueeze(-2);
        sin = sin.repeat(&[1, 1, 2]).unsqueeze(-2);
        |x: &Tensor| {
            let off = x.size()[2] / 2;
            let x1 = x.i((.., .., 0..off));
            let x2 = x.i((.., .., off..));
            Tensor::cat(&[-x2, x1], -1)
        }
    } else {
        // cos = cos.repeat_interleave_self_int(2, -1, None).unsqueeze(-2);
        // sin = sin.repeat_interleave_self_int(2, -1, None).unsqueeze(-2);
        todo!()
    };

    let query_rot = &query_rot * &cos + rotate_fn(&query_rot) * &sin;
    let key_rot = &key_rot * &cos + rotate_fn(&key_rot) * &sin;


    // println!("query_rot: {query_rot:?}");
    // println!("key_rot: {key_rot:?}");
    // println!("query0: {query0:?}");
    // println!("key0: {key0:?}");

    query0.copy_(&query_rot.reshape(query0.size()));
    key0.copy_(&key_rot.reshape(key0.size()));
}

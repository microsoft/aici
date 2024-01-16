use crate::util::{check_all_close_attn, to_vec1};
use crate::HashMap;
use tch::{IndexOp, Kind, Tensor};

pub fn reshape_and_cache(
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

pub fn gather_cached_kv(
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

pub fn varlen_attn(
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

    let (_batch_size_q, num_heads, head_dim) = q.size3().unwrap();

    assert!(causal == true);

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

        assert!(q.size() == [num_heads, len_q, head_dim]);
        assert!(k.size() == [num_heads, len_k, head_dim]);
        assert!(v.size() == [num_heads, len_k, head_dim]);

        let attn_bias = Tensor::zeros(&[len_q, len_k], (q.kind(), q.device()));
        let mask = Tensor::ones(&[len_q, len_q], (Kind::Bool, q.device()))
            .tril(0)
            .logical_not();
        let _ = attn_bias
            .i((.., len_k - len_q..))
            .masked_fill_(&mask, f64::NEG_INFINITY);

        let attn0 = Tensor::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            Some(&attn_bias),
            0.0,
            false,
            softmax_scale,
        )
        .reshape(&[num_heads, len_q, head_dim])
        .transpose(0, 1);

        // println!("attn0: {attn0:?}");

        if false {
            let attn_cpu = Tensor::scaled_dot_product_attention(
                &q.to_dtype_layout((tch::Kind::Float, tch::Device::Cpu), false, true),
                &k.to_dtype_layout((tch::Kind::Float, tch::Device::Cpu), false, true),
                &v.to_dtype_layout((tch::Kind::Float, tch::Device::Cpu), false, true),
                Some(attn_bias.to_dtype_layout((tch::Kind::Float, tch::Device::Cpu), false, true)),
                0.0,
                false,
                softmax_scale,
            )
            .to_dtype_layout((q.kind(), q.device()), false, true)
            .reshape(&[num_heads, len_q, head_dim])
            .transpose(0, 1);

            check_all_close_attn(&attn_cpu, &attn0);
        }

        assert!(!attn0.max().double_value(&[]).is_nan());

        attns.push(attn0);
    }

    let attn = Tensor::cat(&attns, 0);
    attn
}

pub fn rotary_embedding(
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

    let mut query_rot = &query_rot * &cos + rotate_fn(&query_rot) * &sin;
    let mut key_rot = &key_rot * &cos + rotate_fn(&key_rot) * &sin;

    query_rot = query_rot.squeeze_dim(0);
    key_rot = key_rot.squeeze_dim(0);

    if head_size > rot_dim {
        // println!("query_rot: {query_rot:?}");
        // println!("key_rot: {key_rot:?}");
        // println!("query: {query:?}");
        // println!("key: {key:?}");
        query_rot = Tensor::cat(&[query_rot, query.i((.., .., rot_dim..))], -1);
        key_rot = Tensor::cat(&[key_rot, key.i((.., .., rot_dim..))], -1);
    }

    // println!("query_rot: {query_rot:?}");
    // println!("key_rot: {key_rot:?}");
    // println!("query0: {query0:?}");
    // println!("key0: {key0:?}");

    query0.copy_(&query_rot.reshape(query0.size()));
    key0.copy_(&key_rot.reshape(key0.size()));
}

pub fn copy_blocks(
    _key_caches: &mut Vec<Tensor>,
    _value_caches: &mut Vec<Tensor>,
    _block_mapping: &HashMap<usize, Vec<usize>>,
) {
    todo!()
}

pub fn swap_blocks(
    _src: &Tensor,
    _dst: &Tensor,
    _block_mapping: &HashMap<usize, usize>,
    // _stream: &CudaStream,
) {
    todo!()
}

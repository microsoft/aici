use core::panic;

use candle::{
    cuda_backend::{
        cudarc::driver::{CudaSlice, CudaView, DevicePtr},
        CudaDType,
    },
    Layout, Storage, Tensor,
};
use half::bf16;

extern "C" {
    fn rotary_embedding_bf16(
        positions: *const i64,      // [num_tokens]
        query: *mut bf16,           // [num_tokens, num_heads, head_size]
        key: *mut bf16,             // [num_tokens, num_kv_heads, head_size]
        cos_sin_cache: *const bf16, // [max_position, 2, rot_dim // 2]
        num_tokens: i32,
        rot_dim: i32,
        query_stride: i32,
        key_stride: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_size: i32,
    );
}

fn is_bf16(t: &Tensor) -> bool {
    match t.dtype() {
        candle::DType::BF16 => true,
        _ => false,
    }
}

fn as_cont_cuda_slice<'a, T: CudaDType + 'a>(
    storage: &'a Storage,
    layout: &Layout,
) -> CudaView<'a, T> {
    match storage {
        Storage::Cuda(c) => match layout.contiguous_offsets() {
            Some((o1, o2)) => c.as_cuda_slice::<T>().unwrap().slice(o1..o2),
            None => panic!("not contiguous"),
        },
        _ => panic!("not cuda"),
    }
}

fn as_cuda_slice<'a, T: CudaDType + 'a>(storage: &'a Storage, layout: &Layout) -> CudaView<'a, T> {
    match storage {
        Storage::Cuda(c) => c
            .as_cuda_slice::<T>()
            .unwrap()
            .slice(layout.start_offset()..),
        _ => panic!("not cuda"),
    }
}

pub fn rotary_embedding(
    positions: &Tensor, // [num_tokens]
    query: &mut Tensor, // [num_tokens, num_heads * head_size]
    key: &mut Tensor,   // [num_tokens, num_kv_heads * head_size]
    head_size: usize,
    cos_sin_cache: &Tensor, // [max_position, rot_dim]
) {
    assert!(positions.dtype() == candle::DType::I64);
    assert!(is_bf16(query));
    assert!(is_bf16(key));
    assert!(is_bf16(cos_sin_cache));

    let num_tokens = positions.dims1().unwrap();
    let (_nt0, num_heads_x_head_size) = query.dims2().unwrap();
    let (_nt1, num_kv_heads_x_head_size) = key.dims2().unwrap();
    assert!(num_tokens == _nt0);
    assert!(num_tokens == _nt1);
    let (_max_pos, rot_dim) = cos_sin_cache.dims2().unwrap();
    let num_heads = num_heads_x_head_size / head_size;
    let num_kv_heads = num_kv_heads_x_head_size / head_size;
    assert!(num_heads * head_size == num_heads_x_head_size);
    assert!(num_kv_heads * head_size == num_kv_heads_x_head_size);

    let (pos_stor, pos_layout) = positions.storage_and_layout();
    let pos_buf = as_cont_cuda_slice::<i64>(&pos_stor, &pos_layout);
    let pos_ptr = *pos_buf.device_ptr() as *const i64;

    let (q_stor, q_l) = query.storage_and_layout();
    let query_stride = q_l.stride()[0];
    let query_buf = as_cuda_slice::<bf16>(&q_stor, &q_l);
    let query_ptr = *query_buf.device_ptr() as *mut bf16;

    let (k_stor, k_l) = key.storage_and_layout();
    let key_stride = k_l.stride()[0];
    let key_buf = as_cuda_slice::<bf16>(&k_stor, &k_l);
    let key_ptr = *key_buf.device_ptr() as *mut bf16;

    let (cos_sin_stor, cos_sin_layout) = cos_sin_cache.storage_and_layout();
    let cos_sin_buf = as_cont_cuda_slice::<bf16>(&cos_sin_stor, &cos_sin_layout);
    let cos_sin_ptr = *cos_sin_buf.device_ptr() as *const bf16;

    unsafe {
        rotary_embedding_bf16(
            pos_ptr,
            query_ptr,
            key_ptr,
            cos_sin_ptr,
            num_tokens as i32,
            rot_dim as i32,
            query_stride as i32,
            key_stride as i32,
            num_heads as i32,
            num_kv_heads as i32,
            head_size as i32,
        );
    }
}

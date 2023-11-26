use core::panic;
use std::collections::HashMap;

use candle::{
    cuda_backend::{
        cudarc::driver::{CudaStream, CudaView, DevicePtr},
        CudaDType, CudaStorageSlice,
    },
    Device, Layout, Storage, Tensor,
};
use half::{bf16, f16};

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

    fn copy_blocks_bf16(
        key_cache_ptrs: *mut i64,
        value_cache_ptrs: *mut i64,
        block_mapping: *const i32,
        numel_per_block: i32,
        num_pairs: i32,
        num_layers: i32,
    );

    fn gather_scatter_inner_bf16(
        key: *mut bf16,           // [num_tokens, num_heads, head_size]
        value: *mut bf16,         // [num_tokens, num_heads, head_size]
        key_cache: *mut bf16,     // [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache: *mut bf16,   // [num_blocks, num_heads, head_size, block_size]
        slot_mapping: *const i32, // [num_tokens]
        key_stride: i32,
        value_stride: i32,
        num_heads: i32,
        head_size: i32,
        block_size: i32,
        x: i32,
        num_tokens: i32,
        op: i32,
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

fn as_cuda_ptr<'a, T: CudaDType + 'a>(storage: &'a Storage, layout: &Layout) -> i64 {
    let slice = as_cuda_slice::<T>(storage, layout);
    let ptr = slice.device_ptr();
    *ptr as i64
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

    // println!("{:?}", query.layout());
    // println!("q: {:?} k: {:?}", query.dims(), key.dims());
    // println!(
    //     "rotary_embedding: num_tokens={} num_heads={} num_kv_heads={} head_size={} rot_dim={}",
    //     num_tokens, num_heads, num_kv_heads, head_size, rot_dim);

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

    // println!("key_stride={} query_stride={}", key_stride, query_stride);

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
        sync(key.device());
    }
}

// key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
// value_cache,   // [num_blocks, num_heads, head_size, block_size]

fn check_cont_bf16(t: &Tensor) {
    assert!(t.device().is_cuda());
    assert!(is_bf16(t));
    assert!(t.layout().is_contiguous());
}

fn sync(d: &Device) {
    match d {
        Device::Cuda(_c) => {
            //   c.synchronize().unwrap()
        }
        _ => panic!("not cuda"),
    }
}

pub fn copy_blocks(
    key_caches: &Vec<&Tensor>,
    value_caches: &Vec<&Tensor>,
    block_mapping: &HashMap<usize, Vec<usize>>,
) {
    let num_layers = key_caches.len();
    assert_eq!(num_layers, value_caches.len());
    if num_layers == 0 {
        return;
    }
    let device = key_caches[0].device();
    assert!(device.is_cuda());

    let (_num_blocks, num_heads, head_size, block_size) = key_caches[0].dims4().unwrap();
    let numel_per_block = (num_heads * head_size * block_size) as i32;

    let tsize = key_caches[0].elem_count();

    let key_cache_ptrs: Vec<i64> = key_caches.iter().map(|t| to_cuda_ptr(*t)).collect();
    let value_cache_ptrs: Vec<i64> = value_caches.iter().map(|t| to_cuda_ptr(*t)).collect();

    for layer_idx in 0..(2 * num_layers) {
        let e = if layer_idx < num_layers {
            &key_caches[layer_idx]
        } else {
            &value_caches[layer_idx - num_layers]
        };
        assert!(e.device().same_device(device));
        assert_eq!(e.elem_count(), tsize);
        check_cont_bf16(e);
    }

    let mut block_mapping_vec = Vec::new();
    for (&src_block_number, dst_block_numbers) in block_mapping {
        for &dst_block_number in dst_block_numbers {
            block_mapping_vec.push(src_block_number as u32);
            block_mapping_vec.push(dst_block_number as u32);
        }
    }
    let num_pairs = block_mapping_vec.len() / 2;

    let key_cache_ptrs_tensor = Tensor::new(key_cache_ptrs, device).unwrap();
    let value_cache_ptrs_tensor = Tensor::new(value_cache_ptrs, device).unwrap();
    let block_mapping_tensor = Tensor::new(block_mapping_vec, device).unwrap();

    unsafe {
        copy_blocks_bf16(
            to_cuda_ptr(&key_cache_ptrs_tensor) as _,
            to_cuda_ptr(&value_cache_ptrs_tensor) as _,
            to_cuda_ptr(&block_mapping_tensor) as _,
            numel_per_block,
            num_pairs as i32,
            num_layers as i32,
        );
        sync(device);
    }
}

fn to_cuda_ptr_inner(storage: &Storage, layout: &Layout) -> i64 {
    match storage {
        Storage::Cuda(c) => match c.slice {
            CudaStorageSlice::U8(_) => as_cuda_ptr::<u8>(&storage, &layout),
            CudaStorageSlice::U32(_) => as_cuda_ptr::<u32>(&storage, &layout),
            CudaStorageSlice::I64(_) => as_cuda_ptr::<i64>(&storage, &layout),
            CudaStorageSlice::BF16(_) => as_cuda_ptr::<bf16>(&storage, &layout),
            CudaStorageSlice::F16(_) => as_cuda_ptr::<f16>(&storage, &layout),
            CudaStorageSlice::F32(_) => as_cuda_ptr::<f32>(&storage, &layout),
            CudaStorageSlice::F64(_) => as_cuda_ptr::<f64>(&storage, &layout),
        },
        _ => panic!("not cuda"),
    }
}

fn to_cuda_ptr(t: &Tensor) -> i64 {
    let (storage, layout) = t.storage_and_layout();
    to_cuda_ptr_inner(&storage, &layout)
}

fn gather_scatter_inner(
    key: &Tensor,          // [num_tokens, num_heads, head_size]
    value: &Tensor,        // [num_tokens, num_heads, head_size]
    key_cache: &Tensor,    // [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache: &Tensor,  // [num_blocks, num_heads, head_size, block_size]
    slot_mapping: &Tensor, // [num_tokens]
    op: i32,
) {
    let (num_tokens, num_heads, head_size) = key.dims3().unwrap();
    let (_num_blocks, _num_heads, _head_size_x, block_size, x) = key_cache.dims5().unwrap();

    assert_eq!(num_heads, _num_heads);
    assert_eq!(head_size, _head_size_x * x);

    let key_stride = key.layout().stride()[0];
    let value_stride = value.layout().stride()[0];

    check_cont_bf16(key);
    check_cont_bf16(value);
    check_cont_bf16(key_cache);
    check_cont_bf16(value_cache);

    unsafe {
        gather_scatter_inner_bf16(
            to_cuda_ptr(key) as _,
            to_cuda_ptr(value) as _,
            to_cuda_ptr(key_cache) as _,
            to_cuda_ptr(value_cache) as _,
            to_cuda_ptr(slot_mapping) as _,
            key_stride as i32,
            value_stride as i32,
            num_heads as i32,
            head_size as i32,
            block_size as i32,
            x as i32,
            num_tokens as i32,
            op,
        );
        sync(key.device());
    }
}

pub fn reshape_and_cache(
    key: &Tensor,             // [num_tokens, num_heads, head_size]
    value: &Tensor,           // [num_tokens, num_heads, head_size]
    key_cache: &mut Tensor,   // [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache: &mut Tensor, // [num_blocks, num_heads, head_size, block_size]
    slot_mapping: &Tensor,    // [num_tokens]
) {
    gather_scatter_inner(key, value, key_cache, value_cache, slot_mapping, 0);
}

pub fn gather_cached_kv(
    key: &mut Tensor,      // [num_tokens, num_heads, head_size]
    value: &mut Tensor,    // [num_tokens, num_heads, head_size]
    key_cache: &Tensor,    // [num_blocks, num_heads, head_size/x, block_size, x]
    value_cache: &Tensor,  // [num_blocks, num_heads, head_size, block_size]
    slot_mapping: &Tensor, // [num_tokens]
) {
    gather_scatter_inner(key, value, key_cache, value_cache, slot_mapping, 1);
}

pub fn swap_blocks(
    _src: &Tensor,
    _dst: &Tensor,
    _block_mapping: &HashMap<usize, usize>,
    _stream: &CudaStream,
) {
    todo!()
}

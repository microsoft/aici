// based on https://github.com/vllm-project/vllm/blob/b9fe4616f98b77b4b9458bce203aa6544cb31ef2/csrc/cache_kernels.cu


#include <cuda_bf16.h>
#include <assert.h>
#include <stdio.h>

// void swap_blocks(...) -> implement in Rust
  
// Grid: (num_layers, num_pairs)
template<typename scalar_t>
__global__ void copy_blocks_kernel(
  int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int* __restrict__ block_mapping,
  const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int src_block_number = block_mapping[2 * pair_idx];
  int dst_block_number = block_mapping[2 * pair_idx + 1];

  const int src_block_offset = src_block_number * numel_per_block;
  const int dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int src_offset = src_block_offset + i;
    int dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int src_offset = src_block_offset + i;
    int dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

extern "C" void copy_blocks_bf16(
  int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int* __restrict__ block_mapping,
  const int numel_per_block,
  const int num_pairs,
  const int num_layers)
{
  dim3 grid(num_layers, num_pairs);
  dim3 block(std::min(1024, numel_per_block));
  const cudaStream_t stream = 0;
  copy_blocks_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
    key_cache_ptrs,
    value_cache_ptrs,
    block_mapping,
    numel_per_block);
    
}


template<typename scalar_t>
__global__ void reshape_and_cache_kernel(
  const scalar_t* __restrict__ key,     // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,   // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  scalar_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
  const int* __restrict__ slot_mapping, // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
  const int token_idx = blockIdx.x;
  const int slot_idx = slot_mapping[token_idx];
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int src_key_idx = token_idx * key_stride + i;
    const int src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                            + head_idx * (head_size / x) * block_size * x
                            + x_idx * block_size * x
                            + block_offset * x
                            + x_offset;
    const int tgt_value_idx = block_idx * num_heads * head_size * block_size
                              + head_idx * head_size * block_size
                              + head_offset * block_size
                              + block_offset;
    key_cache[tgt_key_idx] = __ldg(&key[src_key_idx]);
    value_cache[tgt_value_idx] = __ldg(&value[src_value_idx]);
  }
}

#if 0
void reshape_and_cache(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping)  // [num_tokens]
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    key.scalar_type(),
    "reshape_and_cache_kernel",
    [&] {
      vllm::reshape_and_cache_kernel<scalar_t><<<grid, block, 0, stream>>>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int>(),
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        x);
    });
}

#endif

// Grid: (num_blocks, block_size).
template<typename scalar_t>
__global__ void gather_cached_kv_kernel(
  scalar_t* __restrict__ key,             // [num_tokens, [stride], num_heads, head_size]
  scalar_t* __restrict__ value,           // [num_tokens, [stride], num_heads, head_size]
  const scalar_t* __restrict__ key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
  const int* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x) {
    const int token_idx = blockIdx.x;
    const int slot_idx = slot_mapping[token_idx];
    const int block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    const int num_tokens = num_heads * head_size;
    for (int i = threadIdx.x; i < num_tokens; i += blockDim.x) {
      const int tgt_key_idx = token_idx * key_stride + i;
      const int tgt_value_idx = token_idx * value_stride + i;
  
      const int head_idx = i / head_size;
      const int head_offset = i % head_size;
      const int x_idx = head_offset / x;  // the offset of the [head_size/x] dimension
      const int x_offset = head_offset % x;
  
      const int src_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                              + head_idx * (head_size / x) * block_size * x
                              + x_idx * block_size * x
                              + block_offset * x
                              + x_offset;
      const int src_value_idx = block_idx * num_heads * head_size * block_size
                                + head_idx * head_size * block_size
                                + head_offset * block_size
                                + block_offset;

      key[tgt_key_idx] = __ldg(&key_cache[src_key_idx]);
      value[tgt_value_idx] = __ldg(&value_cache[src_value_idx]);
    }
}

template <typename scalar_t>
__global__ void gather_cached_kv_kernel_optimized(
    scalar_t *__restrict__ key,             // [num_tokens, [stride], num_heads, head_size]
    scalar_t *__restrict__ value,           // [num_tokens, [stride], num_heads, head_size]
    const scalar_t *__restrict__ key_cache, // [num_blocks, num_heads, head_size/x, block_size, x]
    const scalar_t *__restrict__ value_cache, // [num_blocks, num_heads, head_size, block_size]
    const int *__restrict__ slot_mapping,   // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x)
{
    const int token_idx = blockIdx.x;
    const int slot_idx = slot_mapping[token_idx];
    const int block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    const int dim = num_heads * head_size;
    assert(dim % 4 == 0);  // this is true for known use cases
    const int unroll_factor = 4;
    const int unrolled_dim = dim / unroll_factor;

    for (int i = threadIdx.x; i < unrolled_dim; i += blockDim.x)
    {
        int tgt_key_indices[unroll_factor];
        int tgt_value_indices[unroll_factor];
        int src_key_indices[unroll_factor];
        int src_value_indices[unroll_factor];
        scalar_t keys_to_store[unroll_factor];
        scalar_t values_to_store[unroll_factor];

        #pragma unroll
        for (int j = 0; j < unroll_factor; ++j)
        {
            int index = i + j * unrolled_dim;

            const int tgt_key_idx = token_idx * key_stride + index;
            const int tgt_value_idx = token_idx * value_stride + index;

            const int head_idx = index / head_size;
            const int head_offset = index % head_size;
            const int x_idx = head_offset / x;
            const int x_offset = head_offset % x;

            const int src_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                    + head_idx * (head_size / x) * block_size * x
                                    + x_idx * block_size * x
                                    + block_offset * x
                                    + x_offset;
            const int src_value_idx = block_idx * num_heads * head_size * block_size
                                      + head_idx * head_size * block_size
                                      + head_offset * block_size
                                      + block_offset;

            tgt_key_indices[j] = tgt_key_idx;
            tgt_value_indices[j] = tgt_value_idx;
            src_key_indices[j] = src_key_idx;
            src_value_indices[j] = src_value_idx;

            keys_to_store[j] = __ldg(&key_cache[src_key_idx]);
            values_to_store[j] = __ldg(&value_cache[src_value_idx]);
        }

        #pragma unroll
        for (int j = 0; j < unroll_factor; ++j)
        {
            key[tgt_key_indices[j]] = keys_to_store[j];
            value[tgt_value_indices[j]] = values_to_store[j];
        }
    }
}


#if 0
void gather_cached_kv(
  torch::Tensor& key,           // [out] [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [out] [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,     // [in]  [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,   // [in]  [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping)  // [in]  [num_tokens]
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    key.scalar_type(),
    "gather_cached_kv_kernel_optimized",
    [&] {
      vllm::gather_cached_kv_kernel_optimized<scalar_t><<<grid, block, 0, stream>>>(
        key.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        key_cache.data_ptr<scalar_t>(),
        value_cache.data_ptr<scalar_t>(),
        slot_mapping.data_ptr<int>(),
        key_stride,
        value_stride,
        num_heads,
        head_size,
        block_size,
        x);
    });
}
#endif


#define scalar_t __nv_bfloat16


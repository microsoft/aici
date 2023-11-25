// adapted from vllm pos_encoding_kernels.cu

#include <cuda_bf16.h>

template<typename scalar_t, bool IS_NEOX>
inline __device__ void apply_rotary_embedding(
  scalar_t* __restrict__ arr,
  const scalar_t* __restrict__ cos_ptr,
  const scalar_t* __restrict__ sin_ptr,
  int rot_offset,
  int embed_dim)
{
  int x_index, y_index;
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = __ldg(cos_ptr + x_index);
    sin = __ldg(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = __ldg(cos_ptr + x_index / 2);
    sin = __ldg(sin_ptr + x_index / 2);
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];
  arr[x_index] = x * cos - y * sin;
  arr[y_index] = y * cos + x * sin;
}

template<typename scalar_t, bool IS_NEOX>
__global__ void rotary_embedding_kernel(
  const int64_t* __restrict__ positions,        // [num_tokens]
  scalar_t* __restrict__ query,                 // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key,                   // [num_tokens, num_kv_heads, head_size]
  const scalar_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int rot_dim,
  const int query_stride,
  const int key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size) {
  // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * query_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(query + token_head, cos_ptr,
                                              sin_ptr, rot_offset, embed_dim);
  }

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int token_head = token_idx * key_stride + head_idx * head_size;
    const int rot_offset = i % embed_dim;
    apply_rotary_embedding<scalar_t, IS_NEOX>(key + token_head, cos_ptr,
                                              sin_ptr, rot_offset, embed_dim);
  }
}

#define scalar_t __nv_bfloat16

extern "C" void rotary_embedding_bf16(
  const int64_t* __restrict__ positions,        // [num_tokens]
  scalar_t* __restrict__ query,                 // [num_tokens, num_heads, head_size]
  scalar_t* __restrict__ key,                   // [num_tokens, num_kv_heads, head_size]
  const scalar_t* __restrict__ cos_sin_cache,   // [max_position, 2, rot_dim // 2]
  const int32_t num_tokens,
  const int32_t rot_dim,
  const int32_t query_stride,
  const int32_t key_stride,
  const int32_t num_heads,
  const int32_t num_kv_heads,
  const int32_t head_size)
{

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim / 2, 512));
  const cudaStream_t stream = 0; // Use the default stream.
  // const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
    positions, query, key, cos_sin_cache, rot_dim, query_stride, key_stride, num_heads, num_kv_heads, head_size);
}

#if 0

void rotary_embedding(
  torch::Tensor& positions,         // [num_tokens]
  torch::Tensor& query,             // [num_tokens, num_heads * head_size]
  torch::Tensor& key,               // [num_tokens, num_kv_heads * head_size]
  int head_size,
  torch::Tensor& cos_sin_cache,     // [max_position, rot_dim]
  bool is_neox) {
  int num_tokens = query.size(0);
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(1) / head_size;
  int num_kv_heads = key.size(1) / head_size;
  int query_stride = query.stride(0);
  int key_stride = key.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim / 2, 512));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    query.scalar_type(),
    "rotary_embedding",
    [&] {
      if (is_neox) {
        vllm::rotary_embedding_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          rot_dim,
          query_stride,
          key_stride,
          num_heads,
          num_kv_heads,
          head_size);
      } else {
        vllm::rotary_embedding_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          rot_dim,
          query_stride,
          key_stride,
          num_heads,
          num_kv_heads,
          head_size);
      }
    });
}

#endif
#include "vllm/cache.h"
#include "vllm/cuda_utils.h"
#include "vllm/ops.h"

typedef torch::Tensor *tensor;

#define PROTECT(call)                                                          \
  try {                                                                        \
    call;                                                                      \
    return nullptr;                                                            \
  } catch (const std::exception &e) {                                          \
    return strdup(e.what());                                                   \
  }

extern "C" {

char *paged_attention_v1_C(tensor out, tensor query, tensor key_cache,
                           tensor value_cache, int num_kv_heads, float scale,
                           tensor block_tables, tensor context_lens,
                           int block_size, int max_context_len,
                           tensor alibi_slopes) {
  c10::optional<torch::Tensor> alibi;
  if (alibi_slopes != nullptr) {
    alibi = *alibi_slopes;
  }
  PROTECT(paged_attention_v1(*out, *query, *key_cache, *value_cache,
                             num_kv_heads, scale, *block_tables, *context_lens,
                             block_size, max_context_len, alibi));
}

char *paged_attention_v2_C(tensor out, tensor exp_sums, tensor max_logits,
                           tensor tmp_out, tensor query, tensor key_cache,
                           tensor value_cache, int num_kv_heads, float scale,
                           tensor block_tables, tensor context_lens,
                           int block_size, int max_context_len,
                           tensor alibi_slopes) {
  c10::optional<torch::Tensor> alibi;
  if (alibi_slopes != nullptr) {
    alibi = *alibi_slopes;
  }
  PROTECT(paged_attention_v2(*out, *exp_sums, *max_logits, *tmp_out, *query,
                             *key_cache, *value_cache, num_kv_heads, scale,
                             *block_tables, *context_lens, block_size,
                             max_context_len, alibi));
}

char *rms_norm_C(tensor out, tensor input, tensor weight, float epsilon) {
  PROTECT(rms_norm(*out, *input, *weight, epsilon));
}

char *fused_add_rms_norm_C(tensor input, tensor residual, tensor weight,
                           float epsilon) {
  PROTECT(fused_add_rms_norm(*input, *residual, *weight, epsilon));
}

char *rotary_embedding_C(tensor positions, tensor query, tensor key,
                         int head_size, tensor cos_sin_cache, bool is_neox) {
  PROTECT(rotary_embedding(*positions, *query, *key, head_size, *cos_sin_cache,
                           is_neox));
}

char *silu_and_mul_C(tensor out, tensor input) {
  PROTECT(silu_and_mul(*out, *input));
}

char *gelu_new_C(tensor out, tensor input) { PROTECT(gelu_new(*out, *input)); }

char *gelu_fast_C(tensor out, tensor input) {
  PROTECT(gelu_fast(*out, *input));
}

char *reshape_and_cache_C(tensor key, tensor value, tensor key_cache,
                          tensor value_cache, tensor slot_mapping) {
  PROTECT(
      reshape_and_cache(*key, *value, *key_cache, *value_cache, *slot_mapping));
}

char *gather_cached_kv_C(tensor key, tensor value, tensor key_cache,
                         tensor value_cache, tensor slot_mapping) {
  PROTECT(
      gather_cached_kv(*key, *value, *key_cache, *value_cache, *slot_mapping));
}

char *copy_blocks_2_C(tensor key_cache_ptrs_tensor,
                      tensor value_cache_ptrs_tensor,
                      tensor block_mapping_tensor, tensor key0) {
  PROTECT(copy_blocks_2(*key_cache_ptrs_tensor, *value_cache_ptrs_tensor,
                        *block_mapping_tensor, *key0));
}
}
// END GENERATED SECTION
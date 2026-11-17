#include "Kernels.hpp"

namespace kernels {

void paged_attention(const float* q, const float* new_k, const float* new_v,
    PagedIterator<float> cached_k, PagedIterator<float> cached_v, float* output,
    size_t tokens, size_t query_heads, size_t kv_heads, size_t head_dim,
    size_t query_offset);

    
    
} // namespace kernels

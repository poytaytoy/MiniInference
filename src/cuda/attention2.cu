namespace kernels {
void attention(const float* q, const float* k, const float* v, float* output,
    size_t tokens, size_t query_heads, size_t kv_heads, size_t head_dim, size_t query_offset) {
    if (tokens == 0) return;
    if (!q || !k || !v || !output || kv_heads == 0 || query_heads == 0 ||
        query_heads % kv_heads != 0 || head_dim == 0 || head_dim > 256 ||
        tokens > static_cast<size_t>(std::numeric_limits<int>::max()) / query_heads ||
        tokens > std::numeric_limits<size_t>::max() / query_heads / head_dim / sizeof(float))
        throw std::invalid_argument("Attention requires valid buffers, divisible Q/KV heads and head_dim in [1, 256]");
    const size_t max_cached_tokens = std::numeric_limits<size_t>::max() / kv_heads / head_dim / sizeof(float);
    if (tokens > max_cached_tokens || query_offset > max_cached_tokens - tokens)
        throw std::invalid_argument("Attention cache dimensions overflow");

    attention_kernel<<<tokens * query_heads, attention_threads>>>(
        q, k, v, output, query_heads, kv_heads, head_dim, query_offset);
    detail::check_launch();
}
} // namespace kernels

#pragma once
#include "Weight.hpp"
#include "PagingCache.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace kernels::detail {

inline void check_cuda(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
    }
}

inline constexpr unsigned threads = 256;

inline int get_sm_count() {
    int device;
    auto error = cudaGetDevice(&device);
    check_cuda(error, "CUDA device query failed");

    int count;
    error = cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
    check_cuda(error, "CUDA SM count query failed");
    return count;
}

inline void check_launch() {
    const auto error = cudaGetLastError();
    check_cuda(error, "CUDA kernel launch failed");
}

}

namespace kernels {

class CudaStream {
public:
    CudaStream();
    ~CudaStream();
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    cudaStream_t get() const;
    void synchronize() const;
    bool ready() const;
private:
    cudaStream_t stream_{nullptr};
};

void rms_norm(const float* input, const Weight& scale, float* output,
    size_t rows, size_t width, float epsilon);

size_t matmul_workspace_bytes(size_t input_elements);

void matmul(const float* input, const Weight& weight, float* output, size_t rows,
    DeviceBuffer& workspace, cudaStream_t stream = nullptr);

void matmul(const float* input, const Weight& weight, float* output, size_t rows,
    DeviceBuffer& workspace, const CudaStream& stream);

void rope(float* input, size_t tokens, size_t heads, size_t head_dim,
    size_t position_offset, float frequency_base);

void attention(const float* q, PagedIterator<float> k, PagedIterator<float> v, float* output,
    size_t tokens, size_t query_heads, size_t kv_heads, size_t head_dim,
    size_t query_offset = 0);

void residual_add(float* state, const float* branch, size_t count);

void silu_multiply(const float* gate, const float* up, float* output, size_t count);

void dequantize_embeddings(const void* packed, float* output, size_t rows,
    size_t width, size_t row_bytes, ggml_type type);

// One block per row; returns one vocabulary index per contiguous logits row.
std::vector<uint32_t> argmax(const float* logits, size_t vocabulary_size, size_t rows);
uint32_t argmax(const float* logits, size_t vocabulary_size);

}

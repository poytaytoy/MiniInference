#include "Kernels.hpp"
#include <utility>

namespace {
void check_cuda(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
    }
}
}

namespace kernels {

CudaStream::CudaStream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "CUDA stream error");
}

CudaStream::~CudaStream() {
    if (stream_) cudaStreamDestroy(stream_);
}

CudaStream::CudaStream(CudaStream&& other) noexcept
    : stream_(std::exchange(other.stream_, nullptr)) {}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        if (stream_) cudaStreamDestroy(stream_);
        stream_ = std::exchange(other.stream_, nullptr);
    }
    return *this;
}

cudaStream_t CudaStream::get() const {
    if (!stream_) throw std::logic_error("CUDA stream was moved from");
    return stream_;
}

void CudaStream::synchronize() const {
    check_cuda(cudaStreamSynchronize(get()), "CUDA stream error");
}

bool CudaStream::ready() const {
    const auto error = cudaStreamQuery(get());
    if (error == cudaErrorNotReady) return false;
    check_cuda(error, "CUDA stream error");
    return true;
}

} // namespace kernels

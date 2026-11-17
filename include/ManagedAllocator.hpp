#pragma once
#include <cuda_runtime.h>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>

template<typename T>
struct ManagedAllocator {
    static inline void check_cuda(cudaError_t error, const char* message) {
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
        }
    }

    using value_type = T;
    using is_always_equal = std::true_type;
    ManagedAllocator() = default;
    template<typename U>
    ManagedAllocator(const ManagedAllocator<U>&) noexcept {}

    T* allocate(std::size_t count) {
        if (!count) return nullptr;
        void* ptr = nullptr;
        const auto error = cudaMallocManaged(&ptr, count * sizeof(T));
        if (error == cudaErrorMemoryAllocation) throw std::bad_alloc();
        check_cuda(error, "cudaMallocManaged failed");
        return static_cast<T*>(ptr);
    }
    void deallocate(T* ptr, std::size_t) noexcept { if (ptr) cudaFree(ptr); }
    template<typename U>
    bool operator==(const ManagedAllocator<U>&) const noexcept { return true; }
    template<typename U>
    bool operator!=(const ManagedAllocator<U>&) const noexcept { return false; }
};

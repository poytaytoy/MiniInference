#pragma once

#include "Weight.hpp"
#include "ManagedAllocator.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

template<typename T>
class PagingCache{
    static inline void check_cuda(cudaError_t error, const char* message) {
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
        }
    }

    struct Page{
        DeviceBufferView buffer;
    };

    std::vector<std::unique_ptr<Page>> pages;
    const std::size_t page_size;
    std::size_t num_of_pages;
    DeviceBuffer buffer;
    std::mutex free_pages_mutex;
    std::queue<Page*> free_pages;

    public:

    // Sizes are element counts; Engine uses one token's K/V row per page.
    PagingCache(std::size_t allocation_needed, std::size_t page_size):
        page_size{(assert(page_size > 0), page_size)},
        num_of_pages{allocation_needed / page_size},
        buffer{num_of_pages * page_size * sizeof(T)}{
            assert(num_of_pages > 0);

            pages.reserve(num_of_pages);
            T* ptr = static_cast<T*>(buffer.data);
            for (std::size_t i = 0; i < num_of_pages; ++i){
                auto page = std::make_unique<Page>(Page{
                    DeviceBufferView{ptr + i * page_size, page_size * sizeof(T)}
                });
                pages.push_back(std::move(page));
                free_pages.push(pages.back().get());
            }
        }

    class PagedMemory{
        PagingCache* owner;
        std::vector<Page*> ptr;
        std::vector<T*, ManagedAllocator<T*>> device_page_table;
        T* const* page_table_data{nullptr};
        std::size_t page_size;
        std::size_t capacity;
        std::size_t utilization{0};

        private:
            PagedMemory(PagingCache* owner, std::vector<Page*>&& pages, std::size_t total_size)
                : owner{owner}, page_size{owner->page_size}, capacity{total_size} {
                device_page_table.reserve(pages.size());
                for (auto* page : pages)
                    device_page_table.push_back(static_cast<T*>(page->buffer.data));
                page_table_data = device_page_table.data();
                ptr = std::move(pages);
            }

            __device__ T* element_ptr(std::size_t index) const {
                return page_table_data[index / page_size] + index % page_size;
            }

            void free_pages(){
                if (!owner || ptr.empty()) return;
                std::scoped_lock lock{owner->free_pages_mutex};
                for (auto page: ptr){
                    owner->free_pages.push(page);
                }
                ptr.clear();
            }
        public:

        // only moving is permitted because each PagedMemory has unique ownership of its pages
        PagedMemory(const PagedMemory& other) = delete;
        PagedMemory& operator=(const PagedMemory& other) = delete;

        PagedMemory(PagedMemory&& other) noexcept
            : owner{std::exchange(other.owner, nullptr)},
              ptr{std::exchange(other.ptr, std::vector<Page*>{})},
              device_page_table{std::move(other.device_page_table)},
              page_table_data{std::exchange(other.page_table_data, nullptr)},
              page_size{other.page_size},
              capacity{std::exchange(other.capacity, 0)},
              utilization{std::exchange(other.utilization, 0)} {}

        PagedMemory& operator=(PagedMemory&& other){
            if (this == &other) return *this;
            free_pages();
            owner = std::exchange(other.owner, nullptr);
            ptr = std::exchange(other.ptr, std::vector<Page*>{});
            device_page_table = std::move(other.device_page_table);
            page_table_data = std::exchange(other.page_table_data, nullptr);
            page_size = other.page_size;
            capacity = std::exchange(other.capacity, 0);
            utilization = std::exchange(other.utilization, 0);

            return *this;
        };

        class Iterator {
            T* const* pages;
            std::size_t page_size;
            std::size_t index;

        public:
            // Host construction captures only device-accessible state for kernel arguments.
            explicit Iterator(const PagedMemory* p, std::size_t index = 0)
                : pages(p->page_table_data),
                  page_size(p->page_size), index(index) {}

            __device__ T& operator*() const {
                return (*this)[0];
            }

            __device__ T& operator[](std::size_t offset) const {
                const auto position = index + offset;
                return pages[position / page_size][position % page_size];
            }

            __device__ Iterator& operator++() {
                ++index;
                return *this;
            }

            __device__ Iterator& operator+=(std::size_t offset) {
                index += offset;
                return *this;
            }

            __device__ bool operator!=(const Iterator& other) const {
                return pages != other.pages || page_size != other.page_size || index != other.index;
            }
        };

        ~PagedMemory(){
            free_pages();
        }

        __device__ T& operator[](std::size_t index){
            return *element_ptr(index);
        }

        __device__ const T& operator[](std::size_t index) const{
            return *element_ptr(index);
        }

        std::vector<T> copy_to_cpu(std::size_t count) const {
            std::vector<T> result(count);
            for (std::size_t offset = 0; offset < count; offset += page_size) {

                const auto remaining = count - offset;
                const auto elements = remaining < page_size ? remaining : page_size;
                const auto error = cudaMemcpy(result.data() + offset,
                    ptr[offset / page_size]->buffer.data, elements * sizeof(T),
                    cudaMemcpyDeviceToHost);
                check_cuda(error, "KV cache copy failed");
            }
            return result;
        }

        void append_entry(DeviceBufferView entry, std::size_t offset){
            const auto* data = reinterpret_cast<const std::byte*>(entry.data);
            if (offset > capacity || entry.bytes > (capacity - offset) * sizeof(T)) {
                throw std::bad_alloc();
            }

            std::size_t current_page = offset / page_size;
            std::size_t page_offset = (offset % page_size) * sizeof(T);
            std::size_t bytes_copied = 0;

            while (bytes_copied < entry.bytes) {
                const std::size_t allocation = std::min(page_size * sizeof(T) - page_offset, entry.bytes - bytes_copied);
                auto* destination = reinterpret_cast<std::byte*>(ptr[current_page]->buffer.data);
                const auto error = cudaMemcpy(destination + page_offset, data + bytes_copied,
                    allocation, cudaMemcpyDeviceToDevice);
                check_cuda(error, "KV cache append failed");
                bytes_copied += allocation;
                ++current_page;
                page_offset = 0;
            }
            utilization = std::max(utilization, offset + entry.bytes / sizeof(T));
        }

        Iterator begin() const{
            return Iterator(this);
        }

        Iterator end() const{
            return Iterator(this, capacity);
        }

        friend class PagingCache;
    };

    PagedMemory allocate_memory(std::size_t allocation_needed){
        std::size_t pages_need = allocation_needed / page_size + (allocation_needed % page_size != 0);
        std::vector<Page*> storage;
        storage.reserve(pages_need);

        std::scoped_lock lock{free_pages_mutex};
        while (storage.size() < pages_need && !free_pages.empty()) {
            storage.push_back(free_pages.front());
            free_pages.pop();
        }

        std::size_t memory_given = storage.size() * page_size;

        try {
            return PagedMemory(this, std::move(storage), memory_given);
        } catch (...) {
            // Return pages if constructing the page table fails.
            for (auto* page : storage) free_pages.push(page);
            throw;
        }
    };

    std::optional<typename PagedMemory::Iterator> request_memory(PagedMemory& host, std::size_t allocation_size){
        assert(host.owner == this);

        const auto space_remaining = host.capacity - host.utilization;
        if (allocation_size > space_remaining) {
            const auto additional_size = allocation_size - space_remaining;
            const auto pages_needed = additional_size / page_size + (additional_size % page_size != 0);
            std::scoped_lock lock{free_pages_mutex};
            if (free_pages.size() < pages_needed)
                return std::nullopt;

            for (std::size_t i = 0; i < pages_needed; ++i) {
                auto* page = free_pages.front();
                host.ptr.push_back(page);
                host.device_page_table.push_back(static_cast<T*>(page->buffer.data));
                free_pages.pop();
            }

            host.page_table_data = host.device_page_table.data();
            host.capacity += pages_needed * page_size;
        }

        typename PagedMemory::Iterator result(&host, host.utilization);

        return result;
    }

    friend class PagedMemory;
};

template<typename T>
using PagedMemory = typename PagingCache<T>::PagedMemory;

template<typename T>
using PagedIterator = typename PagingCache<T>::PagedMemory::Iterator;

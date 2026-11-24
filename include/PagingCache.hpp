#pragma once

#include "Weight.hpp"
#include "ManagedAllocator.hpp"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

template<typename T>
class PagingCache{
    static constexpr std::size_t TOKENS_PER_PAGE = 16;

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
    std::vector<Page*> free_pages;

    public:

    static std::size_t bytes_for_context(std::size_t tokens, std::size_t layer_count, std::size_t kv_width) {
        const auto pages_per_layer = (tokens + TOKENS_PER_PAGE - 1) / TOKENS_PER_PAGE;
        return pages_per_layer * layer_count * TOKENS_PER_PAGE * kv_width * 2 * sizeof(T);
    }

    PagingCache(std::size_t kv_width, std::size_t budget_bytes):
        page_size{TOKENS_PER_PAGE * kv_width},
        num_of_pages{budget_bytes / (2 * page_size * sizeof(T))},
        buffer{num_of_pages * 2 * page_size * sizeof(T)}{
            if (num_of_pages == 0) {
                throw std::runtime_error("Insufficient free GPU memory for KV cache pool");
            }
            pages.reserve(num_of_pages);
            free_pages.reserve(num_of_pages);
            T* ptr = static_cast<T*>(buffer.data);
            for (std::size_t i = 0; i < num_of_pages; ++i){
                auto page = std::make_unique<Page>(Page{
                    DeviceBufferView{ptr + i * 2 * page_size, 2 * page_size * sizeof(T)}
                });
                pages.push_back(std::move(page));
                free_pages.push_back(pages.back().get());
            }
        }

    class PagedMemory{
        PagingCache* owner;
        std::vector<Page*> ptr;
        std::vector<T*, ManagedAllocator<T*>> device_page_table;
        T* const* page_table_data{nullptr};
        std::size_t page_size;
        std::size_t capacity{0};

        private:
            void attach(Page* page) {
                ptr.push_back(page);
                device_page_table.push_back(static_cast<T*>(page->buffer.data));
                capacity += page_size;
            }

            void free_pages(){
                if (!owner || ptr.empty()) return;
                std::scoped_lock lock{owner->free_pages_mutex};
                for (auto page: ptr){
                    owner->free_pages.push_back(page);
                }
                ptr.clear();
            }
        public:

        explicit PagedMemory(PagingCache& pool) : owner{&pool}, page_size{pool.page_size} {}

        // only moving is permitted because each PagedMemory has unique ownership of its pages
        PagedMemory(const PagedMemory& other) = delete;
        PagedMemory& operator=(const PagedMemory& other) = delete;

        PagedMemory(PagedMemory&& other) noexcept
            : owner{std::exchange(other.owner, nullptr)},
              ptr{std::exchange(other.ptr, std::vector<Page*>{})},
              device_page_table{std::move(other.device_page_table)},
              page_table_data{std::exchange(other.page_table_data, nullptr)},
              page_size{other.page_size},
              capacity{std::exchange(other.capacity, 0)} {}

        PagedMemory& operator=(PagedMemory&& other){
            if (this == &other) return *this;
            free_pages();
            owner = std::exchange(other.owner, nullptr);
            ptr = std::exchange(other.ptr, std::vector<Page*>{});
            device_page_table = std::move(other.device_page_table);
            page_table_data = std::exchange(other.page_table_data, nullptr);
            page_size = other.page_size;
            capacity = std::exchange(other.capacity, 0);

            return *this;
        };

        class Iterator {
            T* const* pages;
            std::size_t page_size;
            std::size_t index;
            std::size_t plane_offset;

        public:
            // Host construction captures only device-accessible state for kernel arguments.
            explicit Iterator(const PagedMemory* p, std::size_t plane_offset)
                : pages(p->page_table_data),
                  page_size(p->page_size), index(0), plane_offset(plane_offset) {}

            __device__ T& operator*() const {
                return (*this)[0];
            }

            __device__ T& operator[](std::size_t offset) const {
                const auto position = index + offset;
                return pages[position / page_size][plane_offset + position % page_size];
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
                return pages != other.pages || page_size != other.page_size ||
                    index != other.index || plane_offset != other.plane_offset;
            }
        };

        ~PagedMemory(){
            free_pages();
        }

        void append_entry(DeviceBufferView keys, DeviceBufferView values, std::size_t offset){
            const auto* key_data = reinterpret_cast<const std::byte*>(keys.data);
            const auto* value_data = reinterpret_cast<const std::byte*>(values.data);
            if (offset > capacity || keys.bytes > (capacity - offset) * sizeof(T)) {
                throw std::bad_alloc();
            }

            std::size_t current_page = offset / page_size;
            std::size_t page_offset = (offset % page_size) * sizeof(T);
            std::size_t bytes_copied = 0;

            while (bytes_copied < keys.bytes) {
                const std::size_t allocation = std::min(page_size * sizeof(T) - page_offset, keys.bytes - bytes_copied);
                auto* destination = reinterpret_cast<std::byte*>(ptr[current_page]->buffer.data);
                check_cuda(cudaMemcpy(destination + page_offset, key_data + bytes_copied,
                    allocation, cudaMemcpyDeviceToDevice), "Key cache append failed");
                check_cuda(cudaMemcpy(destination + page_size * sizeof(T) + page_offset,
                    value_data + bytes_copied, allocation, cudaMemcpyDeviceToDevice), "Value cache append failed");
                bytes_copied += allocation;
                ++current_page;
                page_offset = 0;
            }
        }

        Iterator key_begin() const {
            return Iterator(this, 0);
        }

        Iterator value_begin() const {
            return Iterator(this, page_size);
        }

        friend class PagingCache;
    };

    void request_memory(std::span<PagedMemory> layers, std::size_t tokens) {
        std::scoped_lock lock{free_pages_mutex};
        const auto required_pages = (tokens + TOKENS_PER_PAGE - 1) / TOKENS_PER_PAGE;

        std::size_t pages_needed = 0;
        for (const auto& layer : layers) {
            pages_needed += required_pages - std::min(required_pages, layer.ptr.size());
        }
        if (free_pages.size() < pages_needed) {
            throw std::bad_alloc{};
        }

        for (auto& layer : layers) {
            layer.ptr.reserve(required_pages);
            layer.device_page_table.reserve(required_pages);
            layer.page_table_data = layer.device_page_table.data();
        }

        for (auto& layer : layers) {
            while (layer.ptr.size() < required_pages) {
                layer.attach(free_pages.back());
                free_pages.pop_back();
            }
        }
    }

    friend class PagedMemory;
};

template<typename T>
using PagedMemory = typename PagingCache<T>::PagedMemory;

template<typename T>
using PagedIterator = typename PagingCache<T>::PagedMemory::Iterator;

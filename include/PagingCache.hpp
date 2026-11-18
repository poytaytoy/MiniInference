#pragma once

#include "Weight.hpp"
#include "ManagedAllocator.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
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
        std::string cache_key{};
        std::size_t layer = 0;
        std::shared_ptr<const std::vector<uint32_t>> cached_tokens;
        std::size_t prefix_size = 0;
        typename std::list<Page*>::iterator free_position;

        static std::string hash_key(std::span<const uint32_t> tokens) {
            const std::string_view bytes{reinterpret_cast<const char*>(tokens.data()), tokens.size_bytes()};
            return std::to_string(std::hash<std::string_view>{}(bytes));
        }

        void make_hash_key(std::span<const uint32_t> tokens) {
            cache_key = hash_key(tokens);
        }
    };

    class FreePageQueue {
        std::list<Page*> free_pages;
        std::unordered_multimap<std::string, Page*> cached_pages;

    public:
        // The pool holds free_pages_mutex across queue operations.
        std::size_t size() const {
            return free_pages.size();
        }

        Page* find(const std::string& key, std::span<const uint32_t> prefix, std::size_t layer) const {
            const auto [first, last] = cached_pages.equal_range(key);
            const auto entry = std::find_if(first, last, [&](const auto& cached) {
                const auto* page = cached.second;
                return page->layer == layer && page->prefix_size == prefix.size() &&
                    std::equal(prefix.begin(), prefix.end(), page->cached_tokens->begin());
            });
            return entry == last ? nullptr : entry->second;
        }

        void push(Page* page) {
            page->free_position = free_pages.insert(free_pages.end(), page);
            if (!page->cache_key.empty()) {
                try {
                    cached_pages.emplace(page->cache_key, page);
                } catch (const std::bad_alloc&) {
                    page->cache_key.clear();
                    page->cached_tokens.reset();
                }
            }
        }

        void take(Page* page) {
            free_pages.erase(page->free_position);
            const auto [first, last] = cached_pages.equal_range(page->cache_key);
            const auto entry = std::find_if(first, last,
                [page](const auto& cached) { return cached.second == page; });
            if (entry != last) {
                cached_pages.erase(entry);
            }
            page->cache_key.clear();
            page->cached_tokens.reset();
        }

        Page* pop() {
            auto* page = free_pages.front();
            take(page);
            return page;
        }
    };

    std::vector<std::unique_ptr<Page>> pages;
    const std::size_t page_size;
    std::size_t num_of_pages;
    DeviceBuffer buffer;
    std::mutex free_pages_mutex;
    FreePageQueue free_pages;

    public:

    PagingCache(std::size_t kv_width, std::size_t budget_bytes):
        page_size{TOKENS_PER_PAGE * kv_width},
        num_of_pages{budget_bytes / (2 * page_size * sizeof(T))},
        buffer{num_of_pages * 2 * page_size * sizeof(T)}{
            if (num_of_pages == 0) {
                throw std::runtime_error("Insufficient free GPU memory for KV cache pool");
            }
            pages.reserve(num_of_pages);
            T* ptr = static_cast<T*>(buffer.data);
            for (std::size_t i = 0; i < num_of_pages; ++i){
                auto page = std::make_unique<Page>(Page{
                    DeviceBufferView{ptr + i * 2 * page_size, 2 * page_size * sizeof(T)}
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
        std::size_t capacity{0};
        std::size_t utilization{0};

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
                    owner->free_pages.push(page);
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

        void set_cache_keys(const std::shared_ptr<const std::vector<uint32_t>>& tokens, std::size_t layer) {
            const auto full_pages = std::min(tokens->size() / TOKENS_PER_PAGE, utilization / page_size);
            for (std::size_t i = 0; i < full_pages; ++i) {
                auto* page = ptr[i];
                const auto prefix = std::span<const uint32_t>{*tokens}.first((i + 1) * TOKENS_PER_PAGE);
                page->make_hash_key(prefix);
                page->layer = layer;
                page->cached_tokens = tokens;
                page->prefix_size = prefix.size();
            }
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
            utilization = std::max(utilization, offset + keys.bytes / sizeof(T));
        }

        Iterator key_begin() const {
            return Iterator(this, 0);
        }

        Iterator value_begin() const {
            return Iterator(this, page_size);
        }

        friend class PagingCache;
    };

    std::optional<std::size_t> request_memory(std::span<PagedMemory> layers,
        std::span<const uint32_t> tokens, std::size_t cached_tokens, std::size_t budget) {
        std::scoped_lock lock{free_pages_mutex};

        // Collect complete chunks before removing any pages from the queue.
        std::vector<Page*> matches;
        if (cached_tokens == 0 && layers.front().ptr.empty()) {
            for (std::size_t end = TOKENS_PER_PAGE; end < tokens.size(); end += TOKENS_PER_PAGE) {
                const auto prefix = tokens.first(end);
                const auto key = Page::hash_key(prefix);
                const auto start = matches.size();
                for (std::size_t layer = 0; layer < layers.size(); ++layer) {
                    auto* page = free_pages.find(key, prefix, layer);
                    if (!page) break;
                    matches.push_back(page);
                }
                if (matches.size() - start != layers.size()) {
                    matches.resize(start);
                    break;
                }
            }
        }

        const auto reused_pages = matches.size() / layers.size();
        const auto prefix_tokens = cached_tokens + reused_pages * TOKENS_PER_PAGE;
        const auto count = std::min(budget, tokens.size() - prefix_tokens);
        const auto required_pages = (prefix_tokens + count + TOKENS_PER_PAGE - 1) / TOKENS_PER_PAGE;

        std::size_t pages_needed = 0;
        for (const auto& layer : layers) {
            pages_needed += required_pages - std::min(required_pages, layer.ptr.size());
        }
        if (free_pages.size() < pages_needed) return std::nullopt;

        for (auto& layer : layers) {
            layer.ptr.reserve(required_pages);
            layer.device_page_table.reserve(required_pages);
            layer.page_table_data = layer.device_page_table.data();
        }

        for (std::size_t i = 0; i < matches.size(); ++i) {
            auto* page = matches[i];
            free_pages.take(page);
            auto& layer = layers[i % layers.size()];
            layer.attach(page);
            layer.utilization += page_size;
        }
        for (auto& layer : layers) {
            while (layer.ptr.size() < required_pages) {
                layer.attach(free_pages.pop());
            }
        }
        return prefix_tokens;
    }

    friend class PagedMemory;
};

template<typename T>
using PagedMemory = typename PagingCache<T>::PagedMemory;

template<typename T>
using PagedIterator = typename PagingCache<T>::PagedMemory::Iterator;

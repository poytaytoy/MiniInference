#pragma once

#include "Weight.hpp"
#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>

template<typename T> 
class PagingCache{ 
    struct Page{
        DeviceBufferRef buffer;
    };

    std::vector<std::unique_ptr<Page>> pages; 
    const std::size_t page_size;
    std::size_t num_of_pages;
    DeviceBuffer buffer;     
    std::mutex free_pages_mutex;
    std::queue<Page*> free_pages;

    public: 

    PagingCache(std::size_t allocation_needed, std::size_t page_size):
        page_size{(assert(page_size > 0), page_size)},
        num_of_pages{allocation_needed / page_size},
        buffer{num_of_pages * page_size * sizeof(T)}{
            assert(allocation_needed > 0); 
            assert(num_of_pages > 0);

            std::scoped_lock lock{free_pages_mutex};

            pages.reserve(num_of_pages);
            T* ptr = static_cast<T*>(buffer.data);
            for (std::size_t i = 0; i < num_of_pages; ++i){
                auto page = std::make_unique<Page>(Page{
                    DeviceBufferRef{ptr + i * page_size, page_size * sizeof(T)}
                });
                pages.push_back(std::move(page));
                free_pages.push(pages.back().get());
            }
        }
    
    class PagedMemory{ 
        PagingCache* owner; 
        std::vector<Page*> ptr; 
        DeviceBuffer device_page_table{0};
        std::size_t page_size;
        std::size_t capacity;  
        std::size_t utilization; 

        private: 
            PagedMemory(PagingCache* owner, std::vector<Page*>&& pages, std::size_t total_size)
                : owner{owner}, page_size{owner->page_size}, capacity{total_size} {
                std::vector<T*> data_ptrs;
                data_ptrs.reserve(pages.size());

                for (auto* page : pages){
                    data_ptrs.push_back(static_cast<T*>(page->buffer.data));
                }
                    
                device_page_table = DeviceBuffer(data_ptrs.data(), data_ptrs.size() * sizeof(T*));
                ptr = std::move(pages);
            }

            __device__ T* element_ptr(std::size_t index) const {
                auto* pages = static_cast<T* const*>(device_page_table.data);
                return pages[index / page_size] + index % page_size;
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
              page_size{other.page_size},
              capacity{std::exchange(other.capacity, 0)} {}

        PagedMemory& operator=(PagedMemory&& other){
            if (this == &other) return *this;
            free_pages();
            owner = std::exchange(other.owner, nullptr);
            ptr = std::exchange(other.ptr, std::vector<Page*>{});
            device_page_table = std::move(other.device_page_table);
            page_size = other.page_size;
            capacity = std::exchange(other.capacity, 0);
            
            return *this; 
        };

        class Iterator {
            T* const* pages;
            std::size_t page_size;
            std::size_t index;

        public:
            // Host construction captures only device-accessible state for kernel arguments.
            explicit Iterator(const PagedMemory* p, std::size_t index = 0)
                : pages(static_cast<T* const*>(p->device_page_table.data)),
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

        Iterator begin() const{
            return Iterator(this);
        }

        Iterator end() const{
            return Iterator(this, capacity);
        }
        
        friend class PagingCache; 
    };

    // returns a bunch of paged memory does not gurantee. it does not gurantee you will get the size requested 
    PagedMemory allocate_memory(std::size_t allocation_needed){
        std::size_t pages_need = (allocation_needed + page_size - 1)/page_size;   
        std::vector<Page*> storage; 
        storage.reserve(pages_need);

        std::scoped_lock lock{free_pages_mutex}; 
        for (size_t i = 0; i < pages_need; i++){
            if (!free_pages.empty()){
                storage.push_back(free_pages.front()); 
                free_pages.pop(); 
            }
        }

        std::size_t memory_given = storage.size() * page_size;

        try {
            return PagedMemory(this, std::move(storage), memory_given);
        } catch (...) {
            // The constructor takes ownership only after the upload succeeds.
            for (auto* page : storage) free_pages.push(page);
            throw;
        }
    }; 

    // Should return nothing if the memory request cannot be fullfilled. 
    std::optional<typename PagingCache<T>::PagedMemory::Iterator> requset_memory(PagedMemory& host, std::size_t allocation_size){
        std::size_t space_remaining = host.capacity - host.utilization; 

        if (space_remaining < allocation_size){ 
            std::scoped_lock lock{free_pages_mutex}; 
            if (!free_pages.empty()){
                host.storage.push_back(free_pages.front()); 
                free_pages.pop(); 
            } else { 
                return std::nullopt; 
            }
        }
        
        return host + allocation_size; 
    }

    friend class PagedMemory; 
};

template<typename T>
using PagedMemory = PagingCache<T>::PagedMemory; 

template<typename T>
using PagedReference = PagingCache<T>::PagedMemory::Iterator; 

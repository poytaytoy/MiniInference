#pragma once
#include "ChatFormatter.hpp"
#include "PagingCache.hpp"
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <vector>

struct ChatSession {
    struct CacheEntry {
        std::unique_ptr<PagedMemory<float>> key;
        std::unique_ptr<PagedMemory<float>> value;
        std::vector<float> key_tokens;
        std::vector<float> value_tokens;
        
        std::size_t gpu_allocated_key_tokens{0};
        std::size_t gpu_allocated_value_tokens{0};

        CacheEntry(PagingCache<float>& pool)
            : key(std::make_unique<PagedMemory<float>>(pool.allocate_memory(0))),
              value(std::make_unique<PagedMemory<float>>(pool.allocate_memory(0))) {}
    };

    std::vector<ChatMessage> messages;
    std::vector<uint32_t> tokens;
    std::unordered_map<std::string, CacheEntry> cache;
    std::size_t kv_width = 0;
    std::size_t cached_tokens = 0;
    std::size_t cache_capacity = 2048;

    CacheEntry* update_cache(const std::string& name, PagingCache<float>& pool,
        std::size_t new_tokens) {
        auto [it, inserted] = cache.try_emplace(name, pool);
        auto& entry = it->second;
        auto unavailable = [&]() -> CacheEntry* {
            if (inserted) cache.erase(it);
            return nullptr;
        };
        if (!entry.key || !entry.value) return nullptr;
        const auto required_tokens = cached_tokens + new_tokens;
        if (required_tokens > cache_capacity)
            throw std::out_of_range("KV cache capacity exceeded");

        if (required_tokens > entry.gpu_allocated_key_tokens) {
            if (!pool.request_memory(*entry.key, (required_tokens - entry.gpu_allocated_key_tokens) * kv_width))
                return unavailable();
            entry.gpu_allocated_key_tokens = required_tokens;
        }
        
        if (required_tokens > entry.gpu_allocated_value_tokens) {
            if (!pool.request_memory(*entry.value, (required_tokens - entry.gpu_allocated_value_tokens) * kv_width))
                return unavailable();
            entry.gpu_allocated_value_tokens = required_tokens;
        }
        return &entry;
    }

    PagedIterator<float> get_k(const std::string& name){
        return cache.at(name).key->begin(); 
    }

    PagedIterator<float> get_v(const std::string& name){
        return cache.at(name).value->begin(); 
    }

    // Offload completed cache rows, then return their GPU pages to the pool.
    void offload_cache() {
        for (auto& [name, entry] : cache) {
            if (!entry.key || !entry.value) continue;
            auto keys = entry.key->copy_to_cpu(cached_tokens * kv_width);
            auto values = entry.value->copy_to_cpu(cached_tokens * kv_width);
            entry.key_tokens = std::move(keys);
            entry.value_tokens = std::move(values);
            entry.key.reset();
            entry.value.reset();
            entry.gpu_allocated_key_tokens = 0;
            entry.gpu_allocated_value_tokens = 0;
        }
    }

    ChatSession() = default;
    ~ChatSession() = default;
    ChatSession(const ChatSession&) = delete;
    ChatSession& operator=(const ChatSession&) = delete;
    ChatSession(ChatSession&&) noexcept = default;
    ChatSession& operator=(ChatSession&&) noexcept = default;
};

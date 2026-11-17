#pragma once
#include "ChatMessage.hpp"
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

    std::size_t pending_tokens() const {
        return tokens.size() - cached_tokens;
    }

    CacheEntry* update_cache(const std::string& name, PagingCache<float>& pool,
        std::size_t new_tokens) {
        auto [it, inserted] = cache.try_emplace(name, pool);
        auto& entry = it->second;
        auto unavailable = [&]() -> CacheEntry* {
            if (inserted) cache.erase(it);
            return nullptr;
        };
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

    PagedMemory<float>& get_k(const std::string& name){
        return *cache.at(name).key; 
    }

    PagedMemory<float>& get_v(const std::string& name){
        return *cache.at(name).value; 
    }

    ChatSession() = default;
    ~ChatSession() = default;
    ChatSession(const ChatSession&) = delete;
    ChatSession& operator=(const ChatSession&) = delete;
    ChatSession(ChatSession&&) noexcept = default;
    ChatSession& operator=(ChatSession&&) noexcept = default;
};

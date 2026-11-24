#pragma once
#include "ChatMessage.hpp"
#include "PagingCache.hpp"
#include <algorithm>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <vector>

struct ChatSession {
    static constexpr std::size_t DEFAULT_CONTEXT_LIMIT = 2048;

    std::vector<ChatMessage> messages;
    std::vector<uint32_t> tokens;
    std::vector<PagedMemory<float>> cache;
    std::size_t cached_tokens = 0;
    std::size_t cache_capacity = DEFAULT_CONTEXT_LIMIT;
    bool prefill_complete = false;

    std::size_t pending_tokens() const {
        return tokens.size() - cached_tokens;
    }

    std::size_t update_cache(PagingCache<float>& pool, std::size_t layer_count,
        std::size_t budget) {
        if (tokens.size() > cache_capacity) {
            throw std::out_of_range("KV cache capacity exceeded");
        }
        if (cache.empty()) {
            cache.reserve(layer_count);
            for (std::size_t layer = 0; layer < layer_count; ++layer) {
                cache.emplace_back(pool);
            }
        }

        const auto count = std::min(budget, pending_tokens());
        try {
            pool.request_memory(cache, cached_tokens + count);
        } catch (const std::bad_alloc&) {
            cache.clear();
            cached_tokens = 0;
            prefill_complete = false;
            throw std::runtime_error("Server ran out of KV cache memory while processing this request");
        }
        return count;
    }

    ChatSession() = default;
    ChatSession(const ChatSession&) = delete;
    ChatSession& operator=(const ChatSession&) = delete;
    ChatSession(ChatSession&&) noexcept = default;
    ChatSession& operator=(ChatSession&&) noexcept = default;
};

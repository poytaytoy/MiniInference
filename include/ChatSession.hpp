#pragma once
#include "ChatMessage.hpp"
#include "PagingCache.hpp"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <vector>

struct ChatSession {
    static constexpr std::size_t DEFAULT_CONTEXT_LIMIT = 2048;

    std::vector<ChatMessage> messages;
    std::vector<uint32_t> tokens;
    std::vector<PagedMemory<float>> cache;
    std::size_t cached_tokens = 0;
    std::size_t cache_capacity = DEFAULT_CONTEXT_LIMIT;

    std::size_t pending_tokens() const {
        return tokens.size() - cached_tokens;
    }

    std::optional<std::size_t> update_cache(PagingCache<float>& pool, std::size_t layer_count,
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

        const auto prefix = pool.request_memory(cache, tokens, cached_tokens, budget);
        if (!prefix) return std::nullopt;

        cached_tokens = *prefix;
        return std::min(budget, pending_tokens());
    }

    ChatSession() = default;
    ~ChatSession() {
        if (cached_tokens == 0 || cache.empty()) return;
        try {
            const auto end = tokens.begin() + std::min(cached_tokens, tokens.size());
            const auto prefix = std::make_shared<const std::vector<uint32_t>>(tokens.begin(), end);
            for (std::size_t layer = 0; layer < cache.size(); ++layer) {
                cache[layer].set_cache_keys(prefix, layer);
            }
        } catch (const std::bad_alloc&) {
            // Cache keys are optional. The page owners still return their pages.
        }
    }
    ChatSession(const ChatSession&) = delete;
    ChatSession& operator=(const ChatSession&) = delete;
    ChatSession(ChatSession&&) noexcept = default;
    ChatSession& operator=(ChatSession&&) noexcept = default;
};

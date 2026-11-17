#pragma once
#include "EmbedLookup.hpp"
#include "GGUF.hpp"
#include "ChatSession.hpp"
#include "Weight.hpp"
#include <random>
#include <utility>

class Engine {
public:
    static Engine& start(size_t max_batch_tokens = 64, const std::filesystem::path& file = {});
    std::string run(ChatSession& session,
        bool enable_thinking = false, size_t max_new_tokens = 512,
        bool stream_output = true, bool stop_at_eos = true);

    size_t max_batch_tokens() const { return max_batch_tokens_; }

    uint32_t inference(ChatSession& session);
    using BatchEntry = std::pair<ChatSession*, size_t>;
    
    std::vector<uint32_t> simulate_inference(const std::vector<BatchEntry>& batch);
    std::vector<uint32_t> inference(const std::vector<BatchEntry>& batch);

    void append_token(ChatSession& session, uint32_t token);

    
private:
    friend class TaskManager;

    static const char* const model_path;

    ~Engine() = default;

    explicit Engine(const std::filesystem::path& file = model_path, size_t max_batch_tokens = 64);

    Engine(const Engine& other) = delete;
    Engine(Engine&& other) = delete;
    Engine& operator=(const Engine& other) = delete;
    Engine& operator=(Engine&& other) = delete;  

    const size_t max_batch_tokens_;
    DeviceBuffer token_buffer{size_t{0}};
    DeviceBuffer embedding_rows{size_t{0}};
    DeviceBuffer normalized{size_t{0}};
    DeviceBuffer q{size_t{0}};
    DeviceBuffer k{size_t{0}};
    DeviceBuffer k_normalized{size_t{0}};
    DeviceBuffer v{size_t{0}};
    DeviceBuffer q_normalized{size_t{0}};
    DeviceBuffer attended{size_t{0}};
    DeviceBuffer branch{size_t{0}};
    DeviceBuffer gate{size_t{0}};
    DeviceBuffer up{size_t{0}};
    DeviceBuffer activated{size_t{0}};
    DeviceBuffer logits{size_t{0}};

    GGUF gguf;
    EmbedLookup embedlookup;
    const size_t width;
    const size_t heads;
    const size_t kv_heads;
    const size_t head_dim;
    const size_t ffn_width;
    const size_t layer_count;
    const float epsilon;
    const float rope_base;
    
    // Shared by all layers and sessions; outlives their paged allocations.
    std::unique_ptr<PagingCache<float>> kv_cache_pool;
    std::map<std::string, Weight> weights;
    uint32_t eos_token;
    std::mt19937 random{std::random_device{}()}; 
};

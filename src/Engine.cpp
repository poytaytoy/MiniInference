#include "Engine.hpp"
#include "Kernels.hpp"
#include <algorithm>
#include <format>
#include <stdexcept>
#include <cassert>
#include <unordered_set>

namespace {
void check_cuda(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
    }
}
}

const char* const Engine::model_path = INFERENCE_MODEL_PATH;

/*static*/ Engine& Engine::start(size_t max_batch_tokens, const std::filesystem::path& file){
    static Engine engine(file.empty() ? std::filesystem::path(model_path) : file, max_batch_tokens);

    return engine; 
};

Engine::Engine(const std::filesystem::path& file, size_t max_batch_tokens)
    : max_batch_tokens_(max_batch_tokens), gguf(file, false), embedlookup(gguf),
      width(embedlookup.get_embedding_size()),
      heads(gguf.get_attention_head_count().value()),
      kv_heads(gguf.get_value<uint32_t>("qwen3.attention.head_count_kv").value()),
      head_dim(gguf.get_value<uint32_t>("qwen3.attention.key_length").value()),
      ffn_width(gguf.get_value<uint32_t>("qwen3.feed_forward_length").value()),
      layer_count(gguf.get_layer_count().value()),
      epsilon(gguf.get_value<float>("qwen3.attention.layer_norm_rms_epsilon").value()),
      rope_base(gguf.get_value<float>("qwen3.rope.freq_base").value()) {
    
    for (const auto& tensor : gguf.get_tensors())
        weights.try_emplace(tensor.name, gguf, tensor.name);
    
    assert(max_batch_tokens_ > 0);
    
    token_buffer = DeviceBuffer(max_batch_tokens_ * width * sizeof(float));
    embedding_rows = DeviceBuffer(max_batch_tokens_ * embedlookup.get_row_bytes());
    normalized = DeviceBuffer(max_batch_tokens_ * width * sizeof(float));
    q = DeviceBuffer(max_batch_tokens_ * heads * head_dim * sizeof(float));
    k = DeviceBuffer(max_batch_tokens_ * kv_heads * head_dim * sizeof(float));
    k_normalized = DeviceBuffer(max_batch_tokens_ * kv_heads * head_dim * sizeof(float));
    v = DeviceBuffer(max_batch_tokens_ * kv_heads * head_dim * sizeof(float));
    q_normalized = DeviceBuffer(max_batch_tokens_ * heads * head_dim * sizeof(float));
    attended = DeviceBuffer(max_batch_tokens_ * heads * head_dim * sizeof(float));
    attention_sequences = DeviceBuffer(layer_count * max_batch_tokens_ * sizeof(kernels::AttentionSequence));
    attention_row_map = DeviceBuffer(max_batch_tokens_ * sizeof(size_t));
    branch = DeviceBuffer(max_batch_tokens_ * width * sizeof(float));
    gate = DeviceBuffer(max_batch_tokens_ * ffn_width * sizeof(float));
    up = DeviceBuffer(max_batch_tokens_ * ffn_width * sizeof(float));
    activated = DeviceBuffer(max_batch_tokens_ * ffn_width * sizeof(float));
    logits = DeviceBuffer(max_batch_tokens_ * embedlookup.get_vocabulary_size() * sizeof(float));
    const auto max_matmul_width = std::max({width, heads * head_dim, ffn_width});
    matmul_workspace = DeviceBuffer(kernels::matmul_workspace_bytes(max_batch_tokens_ * max_matmul_width));

    const size_t kv_width = kv_heads * head_dim;
    size_t free_bytes = 0, total_bytes = 0;
    
    const auto error = cudaMemGetInfo(&free_bytes, &total_bytes);
    check_cuda(error, "GPU memory query failed");

    const auto minimum_budget = 2 * PagingCache<float>::bytes_for_context(
        ChatSession::DEFAULT_CONTEXT_LIMIT, layer_count, kv_width);
    if (free_bytes < minimum_budget) {
        throw std::runtime_error("Insufficient free GPU memory to fit two full contexts in the KV cache");
    }
    const auto budget = std::max(free_bytes - free_bytes / 4, minimum_budget);
    kv_cache_pool = std::make_unique<PagingCache<float>>(kv_width, budget);

    const auto eos = gguf.get_eos_token_id();
    
    if (!eos || *eos >= gguf.get_tokens().size())
        throw std::invalid_argument("Model requires a valid EOS token");
    
    eos_token = static_cast<uint32_t>(*eos);
}

std::vector<uint32_t> Engine::simulate_inference(const std::vector<BatchEntry>& batch) {
    std::vector<uint32_t> predictions;
    for (const auto& entry : batch) {
        for (size_t i = 0; i < entry.second; ++i) {
            if (std::uniform_int_distribution<int>(0, 63)(random) == 0) {
                predictions.push_back(eos_token);
            } else {
                const auto token = std::uniform_int_distribution<uint32_t>(0, embedlookup.get_vocabulary_size() - 2)(random);
                predictions.push_back(token >= eos_token ? token + 1 : token);
            }
        }
    }
    return predictions;
}

std::vector<uint32_t> Engine::inference(const std::vector<BatchEntry>& batch) {
#if INFERENCE_SIMULATED
    return simulate_inference(batch);
#else
    using namespace kernels;
    std::vector<size_t> row_starts;
    std::vector<size_t> row_to_sequence;
    std::vector<size_t> prediction_rows;
    std::vector<uint32_t> pending;
    std::unordered_set<ChatSession*> seen;
    size_t tokens = 0;
    for (const auto& [session, count] : batch) {
        row_to_sequence.insert(row_to_sequence.end(), count, row_starts.size());
        row_starts.push_back(tokens);
        tokens += count;
        if (count == session->pending_tokens()) {
            prediction_rows.push_back(tokens - 1);
        }
        pending.insert(pending.end(), session->tokens.begin() + session->cached_tokens,session->tokens.begin() + session->cached_tokens + count);
    }
    if (batch.empty()) return {};

    // Page tables already exist for every layer, so upload the metadata once per batch.
    std::vector<AttentionSequence> sequences;
    sequences.reserve(layer_count * batch.size());
    for (size_t layer = 0; layer < layer_count; ++layer) {
        for (size_t entry = 0; entry < batch.size(); ++entry) {
            const auto* session = batch[entry].first;
            const auto& cache = session->cache[layer];
            sequences.push_back({cache.key_begin(), cache.value_begin(),
                row_starts[entry], session->cached_tokens});
        }
    }
    check_cuda(cudaMemcpy(attention_sequences.data, sequences.data(),
        sequences.size() * sizeof(AttentionSequence), cudaMemcpyHostToDevice), "Attention metadata upload failed");
    check_cuda(cudaMemcpy(attention_row_map.data, row_to_sequence.data(),
        row_to_sequence.size() * sizeof(size_t), cudaMemcpyHostToDevice), "Attention row map upload failed");

    embedlookup.lookup(pending, token_buffer, embedding_rows);
    auto fp = [](DeviceBuffer& buffer) { return static_cast<float*>(buffer.data); };

    for (size_t i = 0; i < layer_count; ++i) {
        const auto prefix = std::format("blk.{}.", i);
        auto weight = [&](const char* name) -> const Weight& { return weights.at(prefix + name); };

        rms_norm(fp(token_buffer), weight("attn_norm.weight"), fp(normalized), tokens, width, epsilon);
        matmul(fp(normalized), weight("attn_q.weight"), fp(q), tokens, matmul_workspace);
        matmul(fp(normalized), weight("attn_k.weight"), fp(k), tokens, matmul_workspace);
        matmul(fp(normalized), weight("attn_v.weight"), fp(v), tokens, matmul_workspace);

        rms_norm(fp(q), weight("attn_q_norm.weight"), fp(q_normalized), tokens * heads, head_dim, epsilon);
        rms_norm(fp(k), weight("attn_k_norm.weight"), fp(k_normalized), tokens * kv_heads, head_dim, epsilon);
        
        for (size_t entry = 0; entry < batch.size(); ++entry) {
            const auto& [session, count] = batch[entry];
            const auto row = row_starts[entry];
            const auto offset = session->cached_tokens;
            const auto q_offset = row * heads * head_dim;
            const auto kv_offset = row * kv_heads * head_dim;
            rope(fp(q_normalized) + q_offset, count, heads, head_dim, offset, rope_base);
            rope(fp(k_normalized) + kv_offset, count, kv_heads, head_dim, offset, rope_base);
            
            auto& cache = session->cache[i];
            const auto cache_offset = offset * kv_heads * head_dim;
            const auto entry_bytes = count * kv_heads * head_dim * sizeof(float);
            cache.append_entry(DeviceBufferView{fp(k_normalized) + kv_offset, entry_bytes},
                DeviceBufferView{fp(v) + kv_offset, entry_bytes}, cache_offset);
        }
        attention(fp(q_normalized), fp(attended),
            static_cast<const AttentionSequence*>(attention_sequences.data) + i * batch.size(),
            static_cast<const size_t*>(attention_row_map.data), tokens, heads, kv_heads, head_dim);

        matmul(fp(attended), weight("attn_output.weight"), fp(branch), tokens, matmul_workspace);
        residual_add(fp(token_buffer), fp(branch), tokens * width);

        rms_norm(fp(token_buffer), weight("ffn_norm.weight"), fp(normalized), tokens, width, epsilon);
        matmul(fp(normalized), weight("ffn_gate.weight"), fp(gate), tokens, matmul_workspace);
        matmul(fp(normalized), weight("ffn_up.weight"), fp(up), tokens, matmul_workspace);
        silu_multiply(fp(gate), fp(up), fp(activated), tokens * ffn_width);
        matmul(fp(activated), weight("ffn_down.weight"), fp(branch), tokens, matmul_workspace);
        residual_add(fp(token_buffer), fp(branch), tokens * width);
    }

    // generates predictions only on decode batch entries 
    const auto prediction_count = prediction_rows.size();
    if (prediction_count == 0) {
        check_cuda(cudaStreamSynchronize(nullptr), "Prefill batch synchronization failed");
        return std::vector<uint32_t>(tokens);
    }

    const float* output_rows = fp(token_buffer);
    if (prediction_count != tokens) {
        // The branch buffer is free after the final residual add.
        auto* gathered = fp(branch);
        for (size_t i = 0; i < prediction_count; ++i) {
            check_cuda(cudaMemcpyAsync(gathered + i * width, output_rows + prediction_rows[i] * width,
                width * sizeof(float), cudaMemcpyDeviceToDevice, nullptr), "Prediction row copy failed");
        }
        output_rows = gathered;
    }

    rms_norm(output_rows, weights.at("output_norm.weight"), fp(normalized),
        prediction_count, width, epsilon);
    matmul(fp(normalized), weights.at("token_embd.weight"), fp(logits), prediction_count, matmul_workspace);
    auto predictions = argmax(fp(logits), embedlookup.get_vocabulary_size(), prediction_count);
    if (prediction_count == tokens) {
        return predictions;
    }

    // TaskManager reads only the last row of each completed entry.
    std::vector<uint32_t> results(tokens);
    for (size_t i = 0; i < prediction_count; ++i) {
        results[prediction_rows[i]] = predictions[i];
    }
    return results;
#endif
}

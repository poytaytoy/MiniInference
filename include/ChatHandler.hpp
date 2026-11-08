#pragma once
#include "GGUF.hpp"
#include "ChatMessage.hpp"
#include "Tokenizer.hpp"
#include <functional>
#include <mutex>
#include <minja/minja.hpp>

struct ChatSession;
class TaskManager;

// Formats conversation input and drives token or full-response generation.
class ChatHandler {
public:
    explicit ChatHandler(const GGUF& model);
    std::string render(const std::vector<ChatMessage>& messages,
        bool enable_thinking = false, bool add_generation_prompt = true) const;
    struct Response {
        std::string text;
        size_t prompt_tokens;
        size_t completion_tokens;
        std::string finish_reason;
    };

    // Calls for the same session must be serialized by the caller.
    void prepare(const std::shared_ptr<ChatSession>& session,
        std::vector<ChatMessage> messages, bool enable_thinking = false);
    // TaskManager appends the predicted token to the session's token history.
    uint32_t next_token(const std::shared_ptr<ChatSession>& session, TaskManager& manager);
    Response generate(const std::shared_ptr<ChatSession>& session, TaskManager& manager,
        size_t max_tokens = 512,
        const std::function<bool(const std::string&)>& emit = {});

private:
    Tokenizer tokenizer_;
    uint32_t eos_id_;
    std::mutex tokenizer_mutex_;
    std::shared_ptr<minja::TemplateNode> template_;
    std::string bos_token_, eos_token_;
};

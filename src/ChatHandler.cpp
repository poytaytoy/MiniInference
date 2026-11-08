#include "ChatHandler.hpp"
#include "ChatSession.hpp"
#include "TaskManager.hpp"
#include <algorithm>
#include <stdexcept>

ChatHandler::ChatHandler(const GGUF& model)
    : tokenizer_(model), eos_id_(model.get_eos_token_id().value()) {
    const auto source = model.get_value<std::string>("tokenizer.chat_template");
    if (!source || source->empty())
        throw std::invalid_argument("GGUF has no string tokenizer.chat_template");
    const auto& tokens = model.get_tokens();
    auto token = [&](std::optional<uint32_t> id) -> std::string {
        if (!id) return {};
        if (*id >= tokens.size()) throw std::invalid_argument("Invalid chat-template token ID");
        return tokens[*id];
    };
    bos_token_ = token(model.get_bos_token_id());
    eos_token_ = token(model.get_eos_token_id());
    template_ = minja::Parser::parse(*source, {
        /* trim_blocks */ true, /* lstrip_blocks */ true, /* keep_trailing_newline */ false});
}

std::string ChatHandler::render(const std::vector<ChatMessage>& messages,
    bool enable_thinking, bool add_generation_prompt) const {
    if (messages.empty()) throw std::invalid_argument("Chat requires at least one message");
    using Json = nlohmann::ordered_json;
    auto input = Json::array();
    for (const auto& message : messages) {
        if (message.role != "system" && message.role != "user" && message.role != "assistant")
            throw std::invalid_argument("Basic chat system, user and assistant roles");
        input.push_back({{"role", message.role}, {"content", message.content}});
    }
    auto context = minja::Context::make(minja::Value(Json{
        {"messages", input}, {"tools", Json::array()},
        {"add_generation_prompt", add_generation_prompt},
        {"enable_thinking", enable_thinking},
        {"bos_token", bos_token_}, {"eos_token", eos_token_}
    }));
    return template_->render(context);
}

void ChatHandler::prepare(const std::shared_ptr<ChatSession>& session,
    std::vector<ChatMessage> messages, bool enable_thinking) {
    std::vector<uint32_t> input;
    {
        std::scoped_lock lock{tokenizer_mutex_};
        input = tokenizer_.encode(render(messages, enable_thinking), true);
    }
    if (input.empty() || input.size() > session->cache_capacity) {
        throw std::invalid_argument("Prompt exceeds session token capacity");
    }
    const auto prefix = std::min(session->cached_tokens, input.size());
    size_t matched = 0;
    while (matched < prefix && session->tokens[matched] == input[matched]) {
        ++matched;
    }
    session->cached_tokens = matched;
    session->tokens = std::move(input);
    session->messages = std::move(messages);
}

uint32_t ChatHandler::next_token(const std::shared_ptr<ChatSession>& session,
    TaskManager& manager) {
    auto output = std::make_shared<TaskOutput>();
    manager.submit(session, output);
    std::unique_lock lock{output->mutex};
    output->ready.wait(lock, [&] { return output->finished; });
    if (output->error) {
        std::rethrow_exception(output->error);
    }
    return output->token;
}

ChatHandler::Response ChatHandler::generate(const std::shared_ptr<ChatSession>& session,
    TaskManager& manager, size_t max_tokens,
    const std::function<bool(const std::string&)>& emit) {
    Response response{"", session->tokens.size(), 0, "length"};
    std::vector<uint32_t> generated;
    while (generated.size() < max_tokens && session->tokens.size() <= session->cache_capacity) {
        const auto token = next_token(session, manager);
        generated.push_back(token);
        if (token == eos_id_) {
            response.finish_reason = "stop";
            break;
        }
        if (emit && !emit(tokenizer_.decode(token))) {
            break;
        }
    }
    response.text = tokenizer_.decode(generated);
    response.completion_tokens = generated.size();
    session->messages.push_back({"assistant", response.text});
    return response;
}

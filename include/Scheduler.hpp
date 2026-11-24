#pragma once

#include "Engine.hpp"
#include <deque>
#include <exception>
#include <memory>
#include <optional>
#include <vector>

struct TaskOutput;

class Scheduler {
public:
    struct Task {
        std::shared_ptr<ChatSession> session;
        std::shared_ptr<TaskOutput> output;
    };

    struct Result {
        std::shared_ptr<TaskOutput> output;
        std::optional<uint32_t> token;
        std::exception_ptr error;
    };

    using Queue = std::deque<Task>;

    explicit Scheduler(Engine& engine) : engine(engine) {}
    bool has_work() const { return !decode.empty() || !prefill.empty(); }
    std::vector<Result> run_batch(Queue& incoming);
    void drain(Queue& pending);

private:
    Engine& engine;
    Queue decode;
    Queue prefill;
};

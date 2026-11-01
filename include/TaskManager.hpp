#pragma once

#include "Engine.hpp"
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct TaskOutput {
    // Readers must hold this mutex when accessing token, finished, or error.
    std::mutex mutex;
    std::condition_variable ready;
    uint32_t token{0};
    bool finished{false};
    std::exception_ptr error;
};

class TaskManager {
public:
    explicit TaskManager(Engine& engine);
    ~TaskManager();

    TaskManager(const TaskManager&) = delete;
    TaskManager& operator=(const TaskManager&) = delete;
    void submit(std::shared_ptr<ChatSession> session, std::shared_ptr<TaskOutput> output);

private:
    struct Task {
        std::shared_ptr<ChatSession> session;
        std::shared_ptr<TaskOutput> output;
    };

    size_t wait_for_tasks();
    Task pop();
    void push_back(Task& task);
    void wait_for_retry();
    void process();
    void finish(Task& task, std::exception_ptr error = {});
    void finish_pending(std::exception_ptr error);

    Engine& engine;
    std::mutex queue_mutex;

    std::condition_variable available;
    std::deque<Task> queue;
    
    bool stopping{false};
    std::thread worker;
};

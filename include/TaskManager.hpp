#pragma once

#include "Scheduler.hpp"
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>

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
    void process();
    void finish(const Scheduler::Result& result);

    Scheduler scheduler;
    std::mutex queue_mutex;

    std::condition_variable available;
    Scheduler::Queue queue;
    
    bool stopping{false};
    std::thread worker;
};

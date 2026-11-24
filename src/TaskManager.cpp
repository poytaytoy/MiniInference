#include "TaskManager.hpp"
#include <stdexcept>
#include <utility>

TaskManager::TaskManager(Engine& engine)
    : scheduler(engine) {
    worker = std::thread(&TaskManager::process, this);
}

TaskManager::~TaskManager() {
    {
        std::scoped_lock lock{queue_mutex};
        stopping = true;
    }
    available.notify_one();
    worker.join();
}

void TaskManager::submit(std::shared_ptr<ChatSession> session,
    std::shared_ptr<TaskOutput> output) {
    {
        std::scoped_lock lock{queue_mutex};
        if (stopping) {
            throw std::runtime_error("TaskManager is stopped");
        }
        queue.push_back({std::move(session), std::move(output)});
    }
    available.notify_one();
}

void TaskManager::finish(const Scheduler::Result& result) {
    {
        std::scoped_lock lock{result.output->mutex};
        if (result.token) {
            result.output->token = *result.token;
        }
        result.output->error = result.error;
        result.output->finished = true;
    }
    result.output->ready.notify_all();
}

void TaskManager::process() {
    while (true) {
        Scheduler::Queue incoming;
        {
            std::unique_lock lock{queue_mutex};
            available.wait(lock, [&] { return stopping || !queue.empty() || scheduler.has_work(); });
            incoming.swap(queue);
            if (stopping) {
                lock.unlock();
                scheduler.drain(incoming);
                const auto error = std::make_exception_ptr(std::runtime_error("TaskManager stopped"));
                for (const auto& task : incoming) {
                    finish({task.output, {}, error});
                }
                return;
            }
        }
        for (const auto& result : scheduler.run_batch(incoming)) {
            finish(result);
        }
    }
}

#include "TaskManager.hpp"
#include <algorithm>
#include <chrono>
#include <format>
#include <stdexcept>
#include <utility>

TaskManager::TaskManager(Engine& engine)
    : engine(engine) {
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

void TaskManager::finish(Task& task, std::exception_ptr error) {
    {
        std::scoped_lock lock{task.output->mutex};
        task.output->error = error;
        task.output->finished = true;
    }
    task.output->ready.notify_all();
}

void TaskManager::finish_pending(std::exception_ptr error) {
    std::deque<Task> pending;
    {
        std::scoped_lock lock{queue_mutex};
        stopping = true;
        pending.swap(queue);
    }
    for (auto& task : pending) {
        finish(task, error);
    }
}

size_t TaskManager::wait_for_tasks() {
    std::unique_lock lock{queue_mutex};
    available.wait(lock, [&] { return stopping || !queue.empty(); });
    return stopping ? 0 : queue.size();
}

TaskManager::Task TaskManager::pop() {
    std::scoped_lock lock{queue_mutex};
    auto task = std::move(queue.front());
    queue.pop_front();
    return task;
}

void TaskManager::push_back(Task& task) {
    {
        std::scoped_lock lock{queue_mutex};
        if (!stopping) {
            queue.push_back(std::move(task));
            return;
        }
    }
    finish(task, std::make_exception_ptr(std::runtime_error("TaskManager stopped")));
}

void TaskManager::wait_for_retry() {
    std::unique_lock lock{queue_mutex};
    if (!stopping) {
        available.wait_for(lock, std::chrono::milliseconds(10));
    }
}

void TaskManager::process() {
    const auto stopped = std::make_exception_ptr(std::runtime_error("TaskManager stopped"));
    while (true) {
        std::vector<Task> tasks;
        std::vector<Engine::BatchEntry> batch;
        const auto turns = wait_for_tasks();
        if (turns == 0) break;

        size_t budget = engine.max_batch_tokens();
        tasks.reserve(turns);
        batch.reserve(turns);
        for (size_t turn = 0; turn < turns && budget > 0; ++turn) {
            auto task = pop();
            const auto pending = task.session->pending_tokens();
            if (pending == 0) {
                finish(task);
                continue;
            }
            std::size_t count;
            try {
                const auto reserved = task.session->update_cache(*engine.kv_cache_pool, engine.layer_count, budget);
                if (!reserved) {
                    throw std::bad_alloc{};
                }
                count = *reserved;
            } catch (const std::bad_alloc&) {
                push_back(task);
                continue;
            } catch (...) {
                finish(task, std::current_exception());
                continue;
            }
            batch.emplace_back(task.session.get(), count);
            tasks.push_back(std::move(task));
            budget -= count;
        }

        if (batch.empty()) {
            wait_for_retry();
            continue;
        }

        std::vector<uint32_t> predictions;
        try {
            predictions = engine.inference(batch);
        } catch (...) {
            for (auto& task : tasks) {
                finish(task, std::current_exception());
            }
            continue;
        }

        size_t row = 0;
        for (size_t i = 0; i < tasks.size(); ++i) {
            auto& task = tasks[i];
            auto& session = *task.session;
            const auto count = batch[i].second;
            row += count;
            session.cached_tokens += count;
            if (session.pending_tokens() == 0) {
                const auto token = predictions[row - 1];
                session.tokens.push_back(token);
                {
                    std::scoped_lock lock{task.output->mutex};
                    task.output->token = token;
                }
                finish(task);
            } else {
                push_back(task);
            }
        }
    }
    finish_pending(stopped);
}

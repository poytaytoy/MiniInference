#include "Scheduler.hpp"

std::vector<Scheduler::Result> Scheduler::run_batch(Queue& incoming) {
    for (auto& task : incoming) {
        auto& queue = task.session->prefill_complete ? decode : prefill;
        queue.push_back(std::move(task));
    }

    auto budget = engine.max_batch_tokens();
    std::vector<Engine::BatchEntry> batch;
    std::vector<Task> selected;
    std::vector<Result> results;
    batch.reserve(budget);
    selected.reserve(budget);

    for (auto* queue : {&decode, &prefill}) {
        while (budget > 0 && !queue->empty()) {
            auto task = std::move(queue->front());
            queue->pop_front();
            if (task.session->pending_tokens() == 0) {
                results.push_back({std::move(task.output), {}, {}});
                continue;
            }
            try {
                const auto count = task.session->update_cache(
                    *engine.kv_cache_pool, engine.layer_count, budget);
                batch.emplace_back(task.session.get(), count);
                selected.push_back(std::move(task));
                budget -= count;
            } catch (...) {
                results.push_back({std::move(task.output), {}, std::current_exception()});
            }
        }
    }

    if (batch.empty()) return results;
    std::vector<uint32_t> predictions;
    try {
        predictions = engine.inference(batch);
    } catch (...) {
        const auto error = std::current_exception();
        for (auto& task : selected) {
            results.push_back({std::move(task.output), {}, error});
        }
        return results;
    }

    std::size_t row = 0;
    for (std::size_t i = 0; i < selected.size(); ++i) {
        auto& task = selected[i];
        auto& session = *task.session;
        const auto count = batch[i].second;
        row += count;
        session.cached_tokens += count;
        if (session.pending_tokens() != 0) {
            prefill.push_front(std::move(task));
            continue;
        }
        const auto token = predictions[row - 1];
        session.tokens.push_back(token);
        session.prefill_complete = true;
        results.push_back({std::move(task.output), token, {}});
    }
    return results;
}

void Scheduler::drain(Queue& pending) {
    for (auto* queue : {&decode, &prefill}) {
        while (!queue->empty()) {
            pending.push_back(std::move(queue->front()));
            queue->pop_front();
        }
    }
}

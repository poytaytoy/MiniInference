#include "Engine.hpp"
#include "TaskManager.hpp"
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class Runtime {
public:
    // One Runtime/TaskManager per process; Python owns conversation history.
    Runtime(const std::string& model_path, size_t batch_tokens)
        : metadata(model_path, false), engine(Engine::start(batch_tokens, model_path)),
          manager(std::make_unique<TaskManager>(engine)) {}

    uint32_t next_token(const std::shared_ptr<ChatSession>& session) {
        if (!manager) {
            throw std::runtime_error("Runtime is closed");
        }
        auto output = std::make_shared<TaskOutput>();
        manager->submit(session, output);
        std::unique_lock lock{output->mutex};
        output->ready.wait(lock, [&] { return output->finished; });
        if (output->error) {
            std::rethrow_exception(output->error);
        }
        return output->token;
    }

    void close() { manager.reset(); }

    GGUF metadata;
    Engine& engine;
    std::unique_ptr<TaskManager> manager;
};

PYBIND11_MODULE(_inference_engine, module) {
    module.attr("simulated") = bool(INFERENCE_SIMULATED);
    py::class_<ChatSession, std::shared_ptr<ChatSession>>(module, "ChatSession")
        .def(py::init<>())
        .def("prepare_tokens", [](ChatSession& session, std::vector<uint32_t> tokens) {
            if (tokens.empty() || tokens.size() > session.cache_capacity) {
                throw std::invalid_argument("Prompt exceeds session token capacity");
            }
            const auto prefix = std::min(session.cached_tokens, tokens.size());
            size_t matched = 0;
            while (matched < prefix && session.tokens[matched] == tokens[matched]) {
                ++matched;
            }
            session.cached_tokens = matched;
            session.tokens = std::move(tokens);
        })
        .def_property_readonly("token_count", [](const ChatSession& session) { return session.tokens.size(); })
        .def_readonly("cached_tokens", &ChatSession::cached_tokens)
        .def_readwrite("cache_capacity", &ChatSession::cache_capacity)
        .def("pending_tokens", &ChatSession::pending_tokens)
        .def("offload_cache", &ChatSession::offload_cache, py::call_guard<py::gil_scoped_release>());

    py::class_<Runtime>(module, "Runtime")
        .def(py::init<const std::string&, size_t>(), py::arg("model_path"), py::arg("batch_tokens") = 64,
            py::call_guard<py::gil_scoped_release>())
        .def("next_token", &Runtime::next_token, py::call_guard<py::gil_scoped_release>())
        .def("close", &Runtime::close, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("chat_template", [](const Runtime& runtime) {
            return runtime.metadata.get_value<std::string>("tokenizer.chat_template").value();
        })
        .def_property_readonly("eos_token_id", [](const Runtime& runtime) {
            return runtime.metadata.get_eos_token_id().value();
        })
        .def_property_readonly("eos_token", [](const Runtime& runtime) {
            return runtime.metadata.get_tokens()[runtime.metadata.get_eos_token_id().value()];
        })
        .def_property_readonly("bos_token", [](const Runtime& runtime) {
            const auto id = runtime.metadata.get_bos_token_id();
            return id ? runtime.metadata.get_tokens()[*id] : std::string{};
        });
}

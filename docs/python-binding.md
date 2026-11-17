# Python binding

The Python binding connects the Python serving layer to the C++/CUDA engine.
It uses pybind11 and builds into the native module
`inference_engine._inference_engine`.

The binding is implemented in [bindings/python.cpp](../bindings/python.cpp).

## Usage example

This example shows the native binding with Python tokenization.

```python
from transformers import PreTrainedTokenizerFast
from inference_engine._inference_engine import ChatSession, Runtime

runtime = Runtime("model.gguf", batch_tokens=64)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer.json",
    chat_template=runtime.chat_template,
    bos_token=runtime.bos_token or None,
    eos_token=runtime.eos_token,
)
session = ChatSession()
try:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello!"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    session.prepare_tokens(prompt)
    generated = []
    for _ in range(32):
        if session.token_count >= session.cache_capacity:
            break
        token = runtime.next_token(session)  # Also appends the token to the session.
        if token == runtime.eos_token_id:
            break
        generated.append(token)
    print(tokenizer.decode(generated, skip_special_tokens=True))
finally:
    del session  # Release session GPU pages before closing the runtime.
    runtime.close()
```

Use a tokenizer JSON matching the GGUF model. `prepare_tokens` takes the full
prompt token sequence and retains any matching cached prefix:

```python
session = ChatSession()
session.cache_capacity = 4096
session.prepare_tokens([101, 102, 103])  # Illustrative token IDs.
print(session.token_count)     # 3
print(session.cached_tokens)   # 0: no inference has run yet.
print(session.pending_tokens())  # 3
```

## Responsibilities

| Layer | Responsibilities |
| --- | --- |
| Python server | HTTP endpoints, streaming responses, request limits, and session storage. |
| Python chat handler | Conversation history, chat templates, tokenization, decoding, and the generation loop. |
| Native binding | Expose the runtime and sessions, submit token requests, and return results or errors. |
| C++ engine | Task scheduling, token batching, model weights, CUDA operations, and KV cache storage. |

The request flow is:

```text
HTTP request
  -> Python formats and tokenizes the conversation
  -> ChatSession receives prompt token IDs
  -> Runtime.next_token submits work to TaskManager
  -> Engine processes tokens
  -> C++ appends and returns the predicted token ID
  -> Python decodes tokens and returns or streams text
```

## Runtime

`Runtime(model_path, batch_tokens=64)` reads GGUF metadata, obtains the
process-wide engine, and creates a task manager with a worker thread.
`batch_tokens` controls the engine's token budget per batch.

The intended setup is one runtime and task manager per process. `Engine::start`
uses a static engine, so creating another runtime does not create an independent
engine with a different model or batch size.

| Member | Behavior |
| --- | --- |
| `next_token(session)` | Submits the session, waits for completion, and returns one token ID. The task manager processes pending prompt tokens before returning a prediction. |
| `close()` | Destroys the task manager and joins its worker thread. It does not destroy the static engine. Later token requests raise `RuntimeError`. |
| `chat_template` | Reads the model's chat template from GGUF metadata. |
| `eos_token_id` | Returns the end-of-sequence token ID. |
| `eos_token` | Returns the end-of-sequence token string. |
| `bos_token` | Returns the beginning-of-sequence token string, or an empty string if absent. |

Runtime construction, token requests, and closing release Python's Global
Interpreter Lock (GIL). A token request still blocks its calling thread, but
other Python threads can continue running. Errors from the worker are rethrown
through the binding as Python exceptions.

## ChatSession

`ChatSession()` stores token IDs and native KV cache state.

| Member | Behavior |
| --- | --- |
| `prepare_tokens(tokens)` | Replaces the token sequence and retains the cached prefix that matches the new sequence. Rejects empty prompts and prompts exceeding capacity. |
| `token_count` | Read-only length of the current token sequence, including appended predictions. |
| `cached_tokens` | Read-only number of tokens already processed into the cache. |
| `cache_capacity` | Writable session capacity, initially 2048 tokens. |
| `pending_tokens()` | Returns `token_count - cached_tokens`. |

## Python serving layer

[chat_handler.py](../python/inference_engine/chat_handler.py) uses Transformers'
`PreTrainedTokenizerFast` with the tokenizer JSON file and metadata exposed by
the runtime. It applies the chat template, prepares token IDs, requests tokens
until EOS or a generation limit, and decodes the output incrementally.

Token requests run through `asyncio.to_thread`, keeping the HTTP event loop
available. If generation is cancelled while native work is running, Python
waits for that work before releasing the session.

[server.py](../python/inference_engine/server.py) exposes `/v1/models` and
`/v1/chat/completions`. Chat responses support complete JSON responses and SSE
streaming. Each request creates a session for generation. For multi-turn chat,
send the full conversation history in `messages` on each request. The server
runs with one worker process.

## Prediction mode

`INFERENCE_SIMULATED` defaults to `OFF` in CMake and is explicitly disabled by
the Python package build configuration, selecting the model inference path.
The native module exposes the selected mode as the boolean `simulated`.

When simulation is enabled, the engine returns random token IDs with occasional EOS tokens.
It still loads the GGUF model, allocates GPU buffers, and uses the task manager
and cache allocation path. It therefore still needs model files and working
CUDA support.

The non-simulated inference path appends new keys and values to the session’s
paged cache, then runs attention through paged iterators.

## Build

The package uses scikit-build-core to drive CMake and pybind11 to compile the
extension. CMake installs the extension into the `inference_engine` package.

From the repository root:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Building requires a CUDA-capable C++20 toolchain, CUDA toolkit, OpenMP, CMake,
Ninja, and Python development headers. The Python build dependencies are
installed automatically. If `nvcc` is outside `PATH`, supply its location:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc uv pip install -e .
```

Building only `inference_core` produces a static C++ library. It does not build
the Python extension or make the Python server runnable.

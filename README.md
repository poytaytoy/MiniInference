# Inference Engine

A C++/CUDA inference engine for Qwen3 with an OpenAI-compatible Python API.

Requires Python 3.10+, uv, CMake 3.25+, Ninja, a CUDA-capable C++20 compiler,
OpenMP, the CUDA toolkit, and an NVIDIA GPU.

## Run

```sh
git submodule update --init third_party/ggml third_party/json
bash assets/scripts/download_model.sh
uv venv .venv
source .venv/bin/activate
uv pip install -e .
inference-server --batch-tokens 64 --port 8080
```

Use `--model` and `--tokenizer` to select different model files.

## Connect

    const auto minimum_budget = 2 * PagingCache<float>::bytes_for_context(
        ChatSession::DEFAULT_CONTEXT_LIMIT, layer_count, kv_width);
    if (free_bytes < minimum_budget) {
        throw std::runtime_error("Insufficient free GPU memory to fit two full contexts in the KV cache");
    }
Use the OpenAI Python SDK or another OpenAI-compatible client library with
`http://127.0.0.1:8080/v1` as its base URL. Both streaming and complete responses
are supported. Interactive API documentation is available at
`http://127.0.0.1:8080/docs`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Generation performance

Using an NVIDIA GeForce RTX 4060 Laptop GPU and
`Qwen3-4B-Q4_K_M.gguf`. The results are as follows: 

Prompt: "Explain what a GPU does in a few sentences." (23 prompt tokens).

| Run | First token | Generated tokens/s afterwards |
| --- | ---: | ---: |
| 1 | 178.202 ms | 62.105 |
| 2 | 190.730 ms | 60.751 |
| 3 | 190.578 ms | 62.115 |
| Average | 186.504 ms | 61.650 |

## TODO

- Implement paged attention.
- Implement CUDA Graphs.

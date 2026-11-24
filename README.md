# Inference Engine

A C++/CUDA inference engine for Qwen3 with an OpenAI-compatible Python API.

## Benchmark results

Measured on an NVIDIA GeForce RTX 4060 Laptop GPU with
`Qwen3-4B-Q4_K_M.gguf`, a 64-token batch budget, and Nsight tracing enabled.
The server reported 140 input tokens and 64 output tokens per request. All
40 requests completed in each case with zero errors.

| Concurrent requests | Run duration | Average time to first token | Average decode tokens/s per request | Aggregate output tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 93.8 s | 1.17 s | 53.9 | 27.3 |
| 8 | 76.9 s | 3.74 s | 5.6 | 33.3 |

Per-request decode speed measures the tokens after the first output token.
Aggregate throughput includes time spent waiting for the first token.

## Run the server

Requires Python 3.10+, uv, CMake 3.25+, Ninja, a CUDA-capable C++20 compiler,
OpenMP, the CUDA toolkit, and an NVIDIA GPU.

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

## Run the benchmark

The `benchmark/` directory is a standalone uv project. It uses Python 3.12+
and keeps GuideLLM in its own environment.

With the inference server running on port 8080:

```sh
uv run --project benchmark python benchmark/benchmark.py
```

To let the benchmark launch the installed server and capture an Nsight Systems
trace, stop any server on port 8080 and run:

```sh
uv run --project benchmark python benchmark/benchmark.py --profile
```

The benchmark sends 40 requests at concurrency 1, then 40 at concurrency 8
with the same synthetic prompt and output lengths. Use `--concurrency` to
change the second run, or set it to 1 to run only the baseline. The benchmark
prints its output directory under `benchmark/results/`. Both cases get separate
reports and `--profile` adds a shared Nsight trace.

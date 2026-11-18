# HTTP throughput benchmark

With the engine running on port 8080, run this from the repository root.

```sh
python benchmark/throughput.py
```

GuideLLM sends 40 streaming chat requests with up to eight in flight. Each
request uses a synthetic prompt targeting 128 input tokens and requests up to
64 output tokens. The server applies the chat template and may stop at EOS.
GuideLLM reports request throughput, output tokens per second, and streaming
latency. The inference engine does the GPU work.

To set up the benchmark environment on a fresh checkout:

```sh
uv venv --python 3.12 benchmark/.venv
uv pip install --python benchmark/.venv/bin/python guidellm --torch-backend=cpu
```

GuideLLM uses the `Qwen/Qwen3-4B` tokenizer and may download its files on the
first run.

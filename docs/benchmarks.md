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
Each run saves its JSON and CSV reports under `benchmark/results/<run>/`.

To measure one request with about 140 input tokens after chat formatting:

```sh
python benchmark/throughput.py --concurrency 1 --requests 1 --input-tokens 128 --max-tokens 64
```

To set up the benchmark environment on a fresh checkout:

```sh
uv venv --python 3.12 benchmark/.venv
uv pip install --python benchmark/.venv/bin/python guidellm --torch-backend=cpu
```

GuideLLM uses the `Qwen/Qwen3-4B` tokenizer and may download its files on the
first run.

## Capture a CUDA trace

With port 8080 free, add `--profile` to start the installed engine under
Nsight Systems.

```sh
python benchmark/throughput.py --profile --concurrency 8 --requests 40
```

The launcher stops its server after the run. It saves the GuideLLM JSON and
CSV, server log, and `trace.nsys-rep` together under
`benchmark/results/<run>/` and prints the command to open the trace in the UI.

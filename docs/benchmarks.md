# HTTP benchmarks

Start the inference server, then run these commands from the repository root.
The scripts use the Python standard library and default to
`http://127.0.0.1:8080`.

```sh
# Streaming response latency and time to first visible text
python benchmark/latency.py --requests 20 --max-tokens 64

# Complete JSON response latency
python benchmark/latency.py --non-streaming --requests 20

# Aggregate completion tokens/s and requests/s at four-way concurrency
python benchmark/throughput.py --requests 40 --concurrency 4
```

Set `--url` to benchmark another server address. Set `--prompt` to use a
longer prompt, and add `--unique-prompts` to vary the early token prefix
between requests. The default workload repeats the same prompt and allows
prefix reuse. A useful comparison uses the same long base prompt in both
modes. The unique request ID changes tokenization, so compare the prompt
token counts reported by the latency script as well.

Time to first text starts at HTTP request submission and ends at the first
nonempty content event. It includes server scheduling, prompt processing,
and detokenization. Streaming response latency ends at `[DONE]`. Throughput
uses actual completion token counts over the full timed batch, including
prefill and HTTP overhead.

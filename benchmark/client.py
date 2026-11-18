"""Shared HTTP client and reporting for the benchmark scripts."""

import argparse
from dataclasses import dataclass
import json
import math
from statistics import mean
from time import perf_counter
from urllib.request import Request, urlopen
from uuid import uuid4


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parser(description):
    options = argparse.ArgumentParser(description=description)
    options.add_argument("--url", default="http://127.0.0.1:8080")
    options.add_argument("--prompt", default="Explain what a GPU does in a few sentences.")
    options.add_argument("--requests", type=positive_int, default=10)
    options.add_argument("--max-tokens", type=positive_int, default=64)
    options.add_argument("--timeout", type=positive_int, default=120)
    options.add_argument("--unique-prompts", action="store_true",
                         help="prepend a unique request ID to vary the token prefix")
    return options


@dataclass
class Result:
    seconds: float
    prompt_tokens: int
    completion_tokens: int
    first_text_seconds: float | None


def request(args, *, stream):
    prompt = args.prompt
    if args.unique_prompts:
        prompt = f"Request ID {uuid4().hex}. {prompt}"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    endpoint = args.url.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    http_request = Request(
        f"{endpoint}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    first_text = None
    usage = None
    start = perf_counter()
    with urlopen(http_request, timeout=args.timeout) as response:
        if stream:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                received = perf_counter()
                if "error" in event:
                    raise RuntimeError(event["error"]["message"])
                if first_text is None and any(
                    choice["delta"].get("content") for choice in event.get("choices", [])
                ):
                    first_text = received - start
                if event.get("usage") is not None:
                    usage = event["usage"]
            else:
                raise RuntimeError("stream ended before [DONE]")
        else:
            usage = json.load(response)["usage"]
        elapsed = perf_counter() - start

    if usage is None:
        raise RuntimeError("server did not return token usage")
    return Result(elapsed, usage["prompt_tokens"], usage["completion_tokens"], first_text)


def print_latency(label, seconds):
    samples = sorted(seconds)
    if not samples:
        print(f"{label}: no samples")
        return
    p95 = samples[math.ceil(0.95 * len(samples)) - 1]
    print(f"{label}: mean {mean(samples) * 1000:.1f} ms, p95 {p95 * 1000:.1f} ms"
          f" ({len(samples)} samples)")

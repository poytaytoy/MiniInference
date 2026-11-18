"""Run the GuideLLM benchmark against the local inference server."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from tempfile import mkdtemp
from time import monotonic, sleep
from urllib.request import urlopen


BENCHMARK_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    benchmark_env = BENCHMARK_DIR / ".venv"
    if Path(sys.prefix).resolve() != benchmark_env:
        python = benchmark_env / "bin" / "python"
        os.execv(python, [str(python), *sys.argv])

from guidellm.backends.openai.request_handlers import (
    ChatCompletionsRequestHandler,
    OpenAIRequestHandlerFactory,
)
from guidellm.cli import cli


class TextOnlyChatHandler(ChatCompletionsRequestHandler):
    def format(self, *args, **kwargs):
        request = super().format(*args, **kwargs)
        for message in request.body["messages"]:
            content = message["content"]
            if isinstance(content, list):
                message["content"] = "".join(part["text"] for part in content)
        return request


OpenAIRequestHandlerFactory.registry["/v1/chat/completions"] = TextOnlyChatHandler


def make_output_dir():
    results = BENCHMARK_DIR / "results"
    results.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(mkdtemp(prefix=f"{timestamp}-", dir=results))
    print(f"Results directory: {output_dir}", flush=True)
    return output_dir


def run_benchmark(args, output_dir):
    cli([
        "run",
        "--backend",
        "kind=openai_http,target=http://127.0.0.1:8080,"
        "request_format=/v1/chat/completions,validate_backend=false",
        "--tokenizer", "kind=huggingface_auto,model=Qwen/Qwen3-4B",
        "--data", f"kind=synthetic_text,prompt_tokens={args.input_tokens},output_tokens={args.max_tokens}",
        "--profile", f"kind=concurrent,streams={args.concurrency}",
        "--constraint", f"kind=max_requests,count={args.requests}",
        "--output", f"kind=json,path={output_dir / 'benchmarks.json'}",
        "--output", f"kind=csv,path={output_dir / 'benchmarks.csv'}",
    ], standalone_mode=False)


def wait_for_server(server, log_path):
    deadline = monotonic() + 90
    while monotonic() < deadline:
        if server.poll() not in (None, 0):
            break
        try:
            with urlopen("http://127.0.0.1:8080/v1/models", timeout=2):
                return
        except OSError:
            sleep(0.5)
    raise SystemExit(f"Inference server did not start. See {log_path}")


def profile_benchmark(args):
    with socket.socket() as port_check:
        port_check.settimeout(1)
        if port_check.connect_ex(("127.0.0.1", 8080)) == 0:
            raise SystemExit("Port 8080 is already in use. Stop the existing server before profiling.")

    output_dir = make_output_dir()
    session = f"qwen-{output_dir.name}"
    trace = output_dir / "trace.nsys-rep"
    log_path = output_dir / "server.log"
    repo = BENCHMARK_DIR.parent

    with log_path.open("w") as log:
        server = subprocess.Popen([
            "nsys", "launch", f"--session-new={session}", "--trace=cuda",
            str(repo / ".venv/bin/inference-server"), "--port", "8080",
        ], cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_for_server(server, log_path)
            subprocess.run([
                "nsys", "start", f"--session={session}", "--sample=none",
                "--cpuctxsw=none", f"--output={trace.with_suffix('')}",
            ], check=True)
            try:
                run_benchmark(args, output_dir)
            finally:
                subprocess.run(["nsys", "stop", f"--session={session}"], check=True)
        finally:
            subprocess.run(["nsys", "shutdown", f"--session={session}", "--kill=sigterm"])
            server.wait(timeout=30)

    if not trace.is_file():
        raise SystemExit(f"Nsight Systems did not produce a trace. See {log_path}")
    print(f"Open trace: {shlex.join(['nsys-ui', str(trace)])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--profile", action="store_true", help="start the server under Nsight Systems")
    args = parser.parse_args()

    if args.profile:
        profile_benchmark(args)
    else:
        run_benchmark(args, make_output_dir())


if __name__ == "__main__":
    main()

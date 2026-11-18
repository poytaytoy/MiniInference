"""Run the eight-user GuideLLM benchmark against the local inference server."""

import os
from pathlib import Path
import sys


if __name__ == "__main__":
    benchmark_env = Path(__file__).resolve().parent / ".venv"
    if Path(sys.prefix).resolve() != benchmark_env:
        python = benchmark_env / "bin" / "python"
        os.execv(python, [str(python), __file__])

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

if __name__ == "__main__":
    cli([
        "run",
        "--backend",
        "kind=openai_http,target=http://127.0.0.1:8080,"
        "request_format=/v1/chat/completions,validate_backend=false",
        "--tokenizer", "kind=huggingface_auto,model=Qwen/Qwen3-4B",
        "--data", "kind=synthetic_text,prompt_tokens=128,output_tokens=64",
        "--profile", "kind=concurrent,streams=8",
        "--constraint", "kind=max_requests,count=40",
    ])

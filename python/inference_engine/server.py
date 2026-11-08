"""OpenAI-style HTTP/SSE serving with optional in-memory session IDs."""

import argparse
import asyncio
from contextlib import aclosing, asynccontextmanager
import json
import os
from pathlib import Path
import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from ._inference_engine import Runtime, simulated
from .chat_handler import ChatHandler, SessionState


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    model: str = "local-model"
    messages: list[Message] = Field(min_length=1)
    session_id: str | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int = Field(default=512, ge=0)
    max_completion_tokens: int | None = Field(default=None, ge=0)
    enable_thinking: bool = False
    n: Literal[1] = 1


@asynccontextmanager
async def lifespan(app):
    model_path = Path(os.environ.get("INFERENCE_MODEL_PATH", "assets/models/Qwen3-4B-Q4_K_M.gguf")).resolve()
    tokenizer_path = Path(os.environ.get("INFERENCE_TOKENIZER_PATH", str(model_path.parent / "tokenizer.json")))
    runtime = await asyncio.to_thread(Runtime, str(model_path), int(os.environ.get("INFERENCE_BATCH_TOKENS", "64")))
    try:
        app.state.handler = ChatHandler(runtime, tokenizer_path)
        app.state.model_name = model_path.stem
        app.state.sessions = {}
        yield
    finally:
        # Destroy session allocations while the Engine's pool still exists.
        app.state.sessions = {}
        await asyncio.to_thread(runtime.close)


app = FastAPI(lifespan=lifespan)


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [
        {"id": app.state.model_name, "object": "model", "owned_by": "local"}
    ]}


def sse(data):
    return "data: " + (data if isinstance(data, str) else json.dumps(data)) + "\n\n"


@app.post("/v1/chat/completions")
async def completions(request: ChatRequest):
    handler = app.state.handler
    session_id = request.session_id or "session-" + uuid4().hex
    state = app.state.sessions.setdefault(session_id, SessionState())
    completion_id = "chatcmpl-" + uuid4().hex
    created = int(time.time())
    model_name = app.state.model_name
    limit = request.max_tokens if request.max_completion_tokens is None else request.max_completion_tokens
    headers = {"X-Session-ID": session_id}

    def prepare():
        # session_id is an extension: supplied messages append new turns to its history.
        messages = list(state.messages) if request.session_id else []
        messages.extend(message.model_dump() for message in request.messages)
        return handler.prepare(state, messages, request.enable_thinking)

    def chunk(delta, finish_reason=None):
        return {
            "id": completion_id, "object": "chat.completion.chunk", "created": created,
            "model": model_name, "session_id": session_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    if request.stream:
        async def events():
            async with state.lock:
                try:
                    result = prepare()
                    yield sse(chunk({"role": "assistant", "content": ""}))
                    async with aclosing(handler.generate(state, result, limit)) as generation:
                        async for text in generation:
                            yield sse(chunk({"content": text}))
                    yield sse(chunk({}, result.finish_reason))
                    if request.stream_options and request.stream_options.include_usage:
                        usage = chunk({})
                        usage.update(choices=[], usage=result.usage)
                        yield sse(usage)
                except Exception as error:
                    yield sse({"error": {"message": str(error), "type": "server_error"}})
            yield sse("[DONE]")

        return StreamingResponse(events(), media_type="text/event-stream", headers={
            **headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })

    async with state.lock:
        try:
            result = prepare()
            async with aclosing(handler.generate(state, result, limit)) as generation:
                async for _ in generation:
                    pass
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except Exception as error:
            return JSONResponse(status_code=500, content={
                "error": {"message": str(error), "type": "server_error"}
            })
        return JSONResponse(headers=headers, content={
            "id": completion_id, "object": "chat.completion", "created": created,
            "model": model_name, "session_id": session_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text},
                         "finish_reason": result.finish_reason}],
            "usage": result.usage,
        })


def main():
    parser = argparse.ArgumentParser(description="Serve the local inference engine")
    parser.add_argument("--model", default=os.environ.get("INFERENCE_MODEL_PATH", "assets/models/Qwen3-4B-Q4_K_M.gguf"))
    parser.add_argument("--tokenizer", default=os.environ.get("INFERENCE_TOKENIZER_PATH"))
    parser.add_argument("--batch-tokens", type=int, default=64)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    os.environ["INFERENCE_MODEL_PATH"] = str(Path(args.model).resolve())
    os.environ["INFERENCE_BATCH_TOKENS"] = str(args.batch_tokens)
    if args.tokenizer:
        os.environ["INFERENCE_TOKENIZER_PATH"] = str(Path(args.tokenizer).resolve())
    print("Simulated predictions" if simulated else "Model inference")
    # One server process owns one Engine and TaskManager.
    uvicorn.run(app, host=args.host, port=args.port, workers=1)

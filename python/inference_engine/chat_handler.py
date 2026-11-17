"""Conversation preparation and generation, independent of HTTP."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizerFast
from transformers.generation.streamers import TextStreamer

from ._inference_engine import ChatSession, Runtime


@dataclass
class Generation:
    prompt_tokens: int
    text: str = ""
    completion_tokens: int = 0
    finish_reason: str = "length"

    @property
    def usage(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


class _TextChunks(TextStreamer):
    """Use Transformers' incremental decoding without printing to stdout."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.chunks = []

    def on_finalized_text(self, text, stream_end=False):
        if text:
            self.chunks.append(text)


class ChatHandler:
    def __init__(self, runtime: Runtime, tokenizer_file: Path):
        self.runtime = runtime
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            chat_template=runtime.chat_template,
            bos_token=runtime.bos_token or None,
            eos_token=runtime.eos_token,
        )

    def prepare(self, session: ChatSession, messages, enable_thinking=False):
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        session.prepare_tokens(tokens)
        return Generation(prompt_tokens=len(tokens))

    async def next_token(self, session: ChatSession):
        pending = asyncio.create_task(asyncio.to_thread(self.runtime.next_token, session))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            # A C++ task cannot be cancelled mid-inference. Wait before releasing the session.
            await pending
            raise

    async def generate(self, session: ChatSession, result: Generation, max_tokens=512):
        """Yield decoded text chunks for the request's session."""
        generated = []
        streamer = _TextChunks(self.tokenizer)
        try:
            while len(generated) < max_tokens and session.token_count <= session.cache_capacity:
                token = await self.next_token(session)
                generated.append(token)
                result.completion_tokens = len(generated)
                if token == self.runtime.eos_token_id:
                    result.finish_reason = "stop"
                    break
                streamer.put(np.array([token]))
                for text in streamer.chunks:
                    yield text
                streamer.chunks.clear()
            streamer.end()
            for text in streamer.chunks:
                yield text
        finally:
            result.text = self.tokenizer.decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

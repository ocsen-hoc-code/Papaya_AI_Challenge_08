"""Deterministic backend for tests: returns scripted responses, needs no GPU or model."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PIL import Image

from medextract.backends.base import GenerationResult, VLMBackend, VLMError

# A responder maps (images, prompt) to a response text, or raises to simulate a failure.
Responder = Callable[[Sequence[Image.Image], str], str]


@dataclass
class FakeCall:
    images: list[Image.Image]
    prompt: str
    system: str | None


class FakeBackend(VLMBackend):
    """Scripted backend.

    ``responses`` can be a list of strings (returned in order, then the last one
    repeats) or a callable ``(images, prompt) -> str`` for prompt-dependent replies.
    A response that is an ``Exception`` instance is raised instead of returned.
    """

    name = "fake"

    def __init__(
        self,
        responses: Sequence[str | Exception] | Responder,
        *,
        delay_s: float = 0.0,
        model_id: str = "fake-model",
    ) -> None:
        self._responses = responses
        self._delay_s = delay_s
        self.model_id = model_id
        self.calls: list[FakeCall] = []
        self.max_in_flight = 0
        self._in_flight = 0

    async def generate(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult:
        self.calls.append(FakeCall(list(images), prompt, system))
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self._delay_s:
                await asyncio.sleep(self._delay_s)
            if callable(self._responses):
                reply: str | Exception = self._responses(images, prompt)
            else:
                if not self._responses:
                    raise VLMError("FakeBackend has no scripted responses")
                index = min(len(self.calls) - 1, len(self._responses) - 1)
                reply = self._responses[index]
            if isinstance(reply, Exception):
                raise reply
            return GenerationResult(text=reply, generation_tokens=len(reply.split()))
        finally:
            self._in_flight -= 1

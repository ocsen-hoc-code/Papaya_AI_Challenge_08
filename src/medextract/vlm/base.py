"""Backend-agnostic interface to a vision-language model.

Every backend (MLX, Transformers on CUDA/CPU, the fake used in tests) implements
``VLMBackend``. The rest of the pipeline only talks to this interface, so it never
needs to know which hardware it runs on.

Local models run on one accelerator and are not safe to call from several threads, so
``ThreadedBackend`` funnels every call through a single dedicated worker thread. The
asyncio event loop stays free for I/O, rendering and post-processing meanwhile.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class GenerationResult:
    text: str
    # Log-probability of each generated token, when the backend exposes it.
    token_logprobs: list[float] | None = None
    prompt_tokens: int | None = None
    generation_tokens: int | None = None
    elapsed_s: float = 0.0
    extra: dict = field(default_factory=dict)


class VLMError(RuntimeError):
    """Raised when a backend fails to load or to generate."""


class VLMBackend(ABC):
    """Async interface implemented by every model backend."""

    name: str = "base"  # backend kind: "mlx", "cuda", "cpu", "fake"
    model_id: str = ""

    async def load(self) -> None:  # noqa: B027 - optional hook
        """Load weights. Called once before the first ``generate``."""

    @abstractmethod
    async def generate(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Run one multimodal generation and return the decoded text."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""

    def describe(self) -> dict:
        return {"backend": self.name, "model_id": self.model_id}

    async def __aenter__(self) -> VLMBackend:
        await self.load()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()


class ThreadedBackend(VLMBackend):
    """Base for local backends: runs blocking model code on one dedicated thread.

    Subclasses implement ``_load_sync`` and ``_generate_sync``. Calls are serialized,
    because the underlying model can only serve one generation at a time.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"vlm-{self.name}")
        self._loaded = False
        self._load_lock = asyncio.Lock()

    @abstractmethod
    def _load_sync(self) -> None: ...

    @abstractmethod
    def _generate_sync(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult: ...

    async def load(self) -> None:
        async with self._load_lock:
            if self._loaded:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._load_sync)
            self._loaded = True

    async def generate(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> GenerationResult:
        await self.load()
        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        result = await loop.run_in_executor(
            self._executor, self._generate_sync, list(images), prompt, system, max_tokens, temperature
        )
        if not result.elapsed_s:
            result.elapsed_s = time.perf_counter() - start
        return result

    async def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

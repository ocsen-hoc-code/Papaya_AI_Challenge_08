"""Apple Silicon backend: runs an MLX-quantized Qwen3-VL in-process via ``mlx-vlm``.

All MLX work (loading and generation) happens on the single worker thread owned by
``ThreadedBackend``, because MLX streams are thread-local.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from medextract.backends.base import GenerationResult, ThreadedBackend, VLMError

DEFAULT_MODEL_ID = "mlx-community/Qwen3-VL-8B-Instruct-4bit"


class MLXBackend(ThreadedBackend):
    name = "mlx"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None
        self._processor = None
        super().__init__()

    def _load_sync(self) -> None:
        try:
            from mlx_vlm import load
        except ImportError as exc:  # pragma: no cover - depends on the platform
            hint = 'Install the Apple Silicon extra: pip install -e ".[mlx]"'
            raise VLMError(f"mlx-vlm is not installed. {hint}") from exc
        try:
            self._model, self._processor = load(self.model_id)
        except Exception as exc:  # noqa: BLE001 - surface any load failure uniformly
            raise VLMError(f"Failed to load MLX model {self.model_id!r}: {exc}") from exc

    def _generate_sync(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        from mlx_vlm import apply_chat_template, stream_generate

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        formatted = apply_chat_template(self._processor, self._model.config, messages, num_images=len(images))

        pieces: list[str] = []
        logprobs: list[float] = []
        last = None
        try:
            for chunk in stream_generate(
                self._model,
                self._processor,
                formatted,
                image=list(images) or None,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                pieces.append(chunk.text)
                last = chunk
                if chunk.logprobs is not None and chunk.token is not None:
                    logprobs.append(_chosen_logprob(chunk.logprobs, chunk.token))
        except Exception as exc:  # noqa: BLE001
            raise VLMError(f"MLX generation failed: {exc}") from exc
        finally:
            _clear_mlx_cache()

        return GenerationResult(
            text="".join(pieces),
            token_logprobs=logprobs or None,
            prompt_tokens=getattr(last, "prompt_tokens", None),
            generation_tokens=getattr(last, "generation_tokens", None),
            extra={
                # Text of each generated token, aligned with token_logprobs, so callers can
                # map a field's characters back to the tokens that produced it.
                "token_texts": pieces,
                "prompt_tps": getattr(last, "prompt_tps", None),
                "generation_tps": getattr(last, "generation_tps", None),
                "peak_memory_gb": getattr(last, "peak_memory", None),
                "finish_reason": getattr(last, "finish_reason", None),
            },
        )


def _chosen_logprob(logprobs, token: int) -> float:
    """``stream_generate`` yields the log-softmax over the vocabulary; pick the chosen token."""
    size = getattr(logprobs, "size", 1)
    if size > 1:
        return float(logprobs.reshape(-1)[token].item())
    return float(logprobs.item() if hasattr(logprobs, "item") else logprobs)


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx

        clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if clear:
            clear()
    except Exception:  # noqa: BLE001 - best effort only
        pass

"""Qwen3-VL through Hugging Face transformers, for NVIDIA GPUs (4-bit NF4) and plain CPUs (bf16).

Created by ``backends.factory.create_backend``. It reuses ``ThreadedBackend`` from
``backends/base.py``, so all model work runs on one worker thread. Heavy imports (torch,
transformers) happen only inside methods. The CUDA path has only mocked tests; it has not run
on a real GPU.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Literal

from PIL import Image

from medextract.backends.base import GenerationResult, ThreadedBackend, VLMError

ImageDevice = Literal["cuda", "cpu"]


def build_messages(images: Sequence[Image.Image], prompt: str, system: str | None = None) -> list[dict[str, Any]]:
    """Build chat messages: optional system turn, then one user turn with images followed by text."""
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    content: list[dict[str, Any]] = []
    for image in images:
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        content.append({"type": "image", "image": rgb})
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})
    return messages


# The lookups below are module-level functions so tests can replace them. Patching attributes on
# the lazily loaded ``transformers`` package is not reliable.


def _auto_processor_class() -> Any:
    from transformers import AutoProcessor

    return AutoProcessor


def _qwen_model_class() -> Any:
    """Return the transformers class for Qwen3-VL (falls back to the Auto class)."""
    import transformers

    cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if cls is None:
        cls = transformers.AutoModelForImageTextToText
    return cls


def _bnb_config_class() -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig


class TransformersBackend(ThreadedBackend):
    """Runs Qwen3-VL with transformers.

    ``device="cuda"`` loads the model in 4-bit NF4 with bitsandbytes (``device_map="auto"``).
    ``device="cpu"`` loads it in bfloat16 (float32 when the CPU lacks bf16 support) and needs no
    bitsandbytes.
    """

    def __init__(
        self,
        model_id: str,
        device: ImageDevice,
        *,
        load_in_4bit: bool | None = None,
    ) -> None:
        if device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu'; got {device!r}")
        if load_in_4bit is None:
            load_in_4bit = device == "cuda"
        if load_in_4bit and device != "cuda":
            raise ValueError("4-bit loading requires device='cuda' (bitsandbytes is CUDA-only)")
        self.name = device
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = load_in_4bit
        self._processor: Any = None
        self._model: Any = None
        super().__init__()

    # --- ThreadedBackend hooks (run on the worker thread) ----------------------------------

    def _load_sync(self) -> None:
        try:
            import torch
            import transformers  # noqa: F401 - checked here so a missing package gets the hint
        except ImportError as exc:
            raise VLMError(
                f"transformers backend needs torch and transformers: {exc}. "
                f"Install with pip install -e '.[{self.device}]'"
            ) from exc

        try:
            processor_cls = _auto_processor_class()
            model_cls = _qwen_model_class()
            self._processor = processor_cls.from_pretrained(self.model_id)
            if self.load_in_4bit:
                quantization = _bnb_config_class()(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                kwargs: dict[str, Any] = {"quantization_config": quantization, "device_map": "auto"}
            else:
                kwargs = {"dtype": _cpu_dtype(torch)}
            self._model = model_cls.from_pretrained(self.model_id, **kwargs)
            self._model.eval()
        except Exception as exc:
            self._processor = None
            self._model = None
            hint = ""
            if self.load_in_4bit:
                hint = " (4-bit loading needs bitsandbytes and a CUDA GPU: pip install -e '.[cuda]')"
            raise VLMError(f"failed to load {self.model_id!r} on {self.device}: {exc}{hint}") from exc

    def _generate_sync(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> GenerationResult:
        if self._model is None or self._processor is None:
            raise VLMError("model is not loaded; call load() first")
        import torch

        messages = build_messages(images, prompt, system)
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": sample,
            "output_scores": True,
            "return_dict_in_generate": True,
        }
        if sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output = self._model.generate(**inputs, **gen_kwargs)

        sequences = output.sequences
        new_ids = sequences[0][prompt_tokens:]
        text = self._decode(new_ids)
        generation_tokens = int(new_ids.shape[-1])
        logprobs = self._token_logprobs(sequences, output.scores, generation_tokens)
        return GenerationResult(
            text=text,
            token_logprobs=logprobs,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
        )

    def _release_sync(self) -> None:
        self._model = None
        self._processor = None
        if self.device == "cuda":
            import torch

            torch.cuda.empty_cache()

    # --- helpers ---------------------------------------------------------------------------

    def _decode(self, token_ids: Any) -> str:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def _token_logprobs(self, sequences: Any, scores: Any, generation_tokens: int) -> list[float] | None:
        """Per-token log-probabilities of the generated tokens, or None if the model cannot report them."""
        compute = getattr(self._model, "compute_transition_scores", None)
        if compute is None or scores is None:
            return None
        try:
            transition = compute(sequences, scores, normalize_logits=True)
            values = transition[0][:generation_tokens].float().tolist()
        except Exception:  # best effort: missing logprobs must not fail the generation
            return None
        return [float(v) for v in values]

    # --- async API ---------------------------------------------------------------------------

    async def close(self) -> None:
        if self._loaded:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._release_sync)
            self._loaded = False
        await super().close()


def _cpu_dtype(torch: Any) -> Any:
    """bfloat16 when the CPU supports a bf16 matmul, otherwise float32."""
    try:
        probe = torch.zeros(2, 2, dtype=torch.bfloat16)
        _ = probe @ probe
        return torch.bfloat16
    except (RuntimeError, NotImplementedError):
        return torch.float32

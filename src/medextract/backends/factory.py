"""Hardware detection and backend selection.

Order for ``device="auto"``: MLX (Apple Silicon with ``mlx`` installed), then CUDA (torch with
a visible GPU), then CPU. Heavy libraries (mlx, torch, transformers) are imported lazily, only
inside the functions that need them, so importing this module stays cheap.

Facts for reviewers: callers are ``create_backend`` (used by the pipeline and the CLI).
No other hardware-detection code exists in the repo.
"""

from __future__ import annotations

import importlib.util
import platform
from typing import Literal

from medextract.backends.base import VLMBackend, VLMError
from medextract.config import VALID_DEVICES, Settings

BackendKind = Literal["mlx", "cuda", "cpu"]

DEFAULT_MODELS: dict[BackendKind, str] = {
    "mlx": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
    "cuda": "Qwen/Qwen3-VL-8B-Instruct",
    "cpu": "Qwen/Qwen3-VL-8B-Instruct",
}

_MLX_HINT = "install the [mlx] extra on Apple Silicon (pip install -e '.[mlx]')"
_CUDA_HINT = "install the [cuda] extra on a machine with an NVIDIA GPU (pip install -e '.[cuda]')"


def mlx_available() -> bool:
    """True on macOS running on arm64 when the ``mlx`` package is installed."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    return importlib.util.find_spec("mlx") is not None


def cuda_available() -> bool:
    """True when torch is importable and reports a usable CUDA device."""
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch  # lazy: torch is heavy and optional
    except Exception:  # broken torch install counts as "no CUDA"
        return False
    return bool(torch.cuda.is_available())


def detect_backend(preferred: str = "auto") -> BackendKind:
    """Pick a backend. An explicit choice that is unavailable raises ``VLMError`` with a hint."""
    if preferred not in VALID_DEVICES:
        raise ValueError(f"device must be one of {', '.join(VALID_DEVICES)}; got {preferred!r}")
    if preferred == "auto":
        if mlx_available():
            return "mlx"
        if cuda_available():
            return "cuda"
        return "cpu"
    if preferred == "mlx":
        if not mlx_available():
            raise VLMError(f"MLX backend requested but unavailable: it needs macOS on Apple Silicon; {_MLX_HINT}")
        return "mlx"
    if preferred == "cuda":
        if not cuda_available():
            raise VLMError(f"CUDA backend requested but no CUDA GPU is visible to torch; {_CUDA_HINT}")
        return "cuda"
    return "cpu"


def resolve_model_id(kind: BackendKind, model_id: str | None) -> str:
    """Return the configured model id, or the default model for ``kind``."""
    if kind not in DEFAULT_MODELS:
        raise ValueError(f"unknown backend kind {kind!r}")
    if model_id and model_id.strip():
        return model_id.strip()
    return DEFAULT_MODELS[kind]


def create_backend(settings: Settings) -> VLMBackend:
    """Instantiate the backend chosen by ``settings.device`` with the resolved model id."""
    kind = detect_backend(settings.device)
    model_id = resolve_model_id(kind, settings.model_id)
    if kind == "mlx":
        from medextract.backends.mlx import MLXBackend  # lazy: needs mlx

        return MLXBackend(model_id=model_id)
    from medextract.backends.transformers import TransformersBackend  # lazy: needs torch

    device: Literal["cuda", "cpu"] = "cuda" if kind == "cuda" else "cpu"
    return TransformersBackend(model_id=model_id, device=device)

"""Runtime settings, read from ``MEDEXTRACT_*`` environment variables with optional overrides.

Facts for reviewers: this is a new module with no existing duplicate. ``Settings`` is the
only configuration object; ``device.py`` and ``create_backend`` consume it. Invalid values
raise ``ValueError`` that names the offending environment variable (or argument).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

VALID_DEVICES = ("auto", "mlx", "cuda", "cpu")


@dataclass(frozen=True)
class Settings:
    device: str = "auto"  # auto | mlx | cuda | cpu   (env MEDEXTRACT_DEVICE)
    model_id: str | None = None  # None -> default for the chosen backend (env MEDEXTRACT_MODEL_ID)
    max_image_side: int = 1600  # env MEDEXTRACT_MAX_IMAGE_SIDE
    pdf_dpi: int = 200  # env MEDEXTRACT_PDF_DPI
    max_tokens: int = 2048  # env MEDEXTRACT_MAX_TOKENS
    timeout_s: float = 600.0  # per model call, env MEDEXTRACT_TIMEOUT_S
    concurrency: int = 4  # documents in flight, env MEDEXTRACT_CONCURRENCY

    def __post_init__(self) -> None:
        if self.device not in VALID_DEVICES:
            raise ValueError(f"device must be one of {', '.join(VALID_DEVICES)}; got {self.device!r}")
        if self.model_id is not None and not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string or None")
        _check_positive_int("max_image_side", self.max_image_side)
        _check_positive_int("pdf_dpi", self.pdf_dpi)
        _check_positive_int("max_tokens", self.max_tokens)
        _check_positive_int("concurrency", self.concurrency)
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)):
            raise ValueError(f"timeout_s must be a number; got {self.timeout_s!r}")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be a finite number greater than 0; got {self.timeout_s!r}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: Any) -> Settings:
        """Build settings from the environment, then apply keyword overrides.

        ``env`` defaults to ``os.environ``. Overrides whose value is ``None`` are ignored, so
        CLI options left unset do not clobber the environment or the defaults.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        values: dict[str, Any] = {}
        for field_name, env_key, parse in _ENV_FIELDS:
            raw = source.get(env_key)
            if raw is None or not raw.strip():
                continue
            try:
                values[field_name] = parse(raw.strip())
            except ValueError as exc:
                raise ValueError(f"{env_key}: {exc}") from None
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        return cls(**values)


def _check_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1; got {value}")


def _parse_int(raw: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"expected an integer, got {raw!r}") from None


def _parse_float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"expected a number, got {raw!r}") from None


def _parse_device(raw: str) -> str:
    return raw.lower()


def _parse_model_id(raw: str) -> str | None:
    return raw or None


# (field name, environment variable, parser). Parsing errors are re-raised with the env var name.
_ENV_FIELDS: list[tuple[str, str, Callable[[str], Any]]] = [
    ("device", "MEDEXTRACT_DEVICE", _parse_device),
    ("model_id", "MEDEXTRACT_MODEL_ID", _parse_model_id),
    ("max_image_side", "MEDEXTRACT_MAX_IMAGE_SIDE", _parse_int),
    ("pdf_dpi", "MEDEXTRACT_PDF_DPI", _parse_int),
    ("max_tokens", "MEDEXTRACT_MAX_TOKENS", _parse_int),
    ("timeout_s", "MEDEXTRACT_TIMEOUT_S", _parse_float),
    ("concurrency", "MEDEXTRACT_CONCURRENCY", _parse_int),
]

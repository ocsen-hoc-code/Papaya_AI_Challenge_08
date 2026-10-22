"""Tolerant parsing of VLM output into JSON objects, with a validated retry.

Small local models often wrap JSON in ``<think>`` blocks, markdown fences or prose, leave
trailing commas, or stop mid-object when they hit the token limit. ``extract_json`` copes
with all of these. ``generate_structured`` adds Pydantic validation and retries once (by
default) with the validation error fed back to the model.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from typing import Any, TypeVar

import json_repair
from PIL import Image
from pydantic import BaseModel, ValidationError

from medextract.backends.base import GenerationResult, VLMBackend

ModelT = TypeVar("ModelT", bound=BaseModel)

# How much of a bad model output to quote in an error message.
_BAD_OUTPUT_PREVIEW_CHARS = 300

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"`{3}[a-zA-Z0-9_-]*[ \t]*\n?")


class JSONParseError(ValueError):
    """Model output could not be turned into the requested JSON structure."""


def _strip_reasoning(text: str) -> str:
    """Drop ``<think>`` blocks, including a bare ``</think>`` left by chat templates."""
    text = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in text.lower():
        # The opening tag was part of the prompt template; keep only what follows the close.
        text = re.split(r"</think>", text, maxsplit=1, flags=re.IGNORECASE)[1]
    # An unclosed opening tag (output cut off inside the reasoning) is just dropped.
    return re.sub(r"<think>", "", text, flags=re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any]:
    """Return the outermost JSON object in ``text``.

    Strips reasoning blocks, code fences and surrounding prose. Valid JSON is parsed
    strictly first. Otherwise the object is repaired with ``json_repair``, which handles
    truncation, trailing commas and single quotes. Raises ``JSONParseError`` when there
    is no object or the top-level value is not an object.
    """
    if not isinstance(text, str):
        raise JSONParseError(f"expected text output, got {type(text).__name__}")
    cleaned = _FENCE_RE.sub("", _strip_reasoning(text))

    start = cleaned.find("{")
    if start == -1:
        raise JSONParseError("no JSON object found in model output")

    # Strict decode of the first complete object: ignores whatever follows it.
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned, start)
    except json.JSONDecodeError:
        value = None

    if value is None:
        try:
            value = json_repair.repair_json(cleaned[start:], return_objects=True)
        except Exception as exc:  # json_repair is best effort; report its failure uniformly
            raise JSONParseError(f"could not repair JSON: {exc}") from exc

    if not isinstance(value, dict):
        raise JSONParseError(f"top-level JSON value is {type(value).__name__}, expected an object")
    return value


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors()[:5]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return "; ".join(parts)


async def _call_backend(
    backend: VLMBackend,
    images: Sequence[Image.Image],
    prompt: str,
    system: str | None,
    max_tokens: int,
    timeout_s: float | None,
) -> GenerationResult:
    call = backend.generate(images, prompt, system=system, max_tokens=max_tokens)
    if timeout_s is None:
        return await call
    # Timeouts are not retried: a model that stalled once will usually stall again.
    return await asyncio.wait_for(call, timeout_s)


async def generate_structured(
    backend: VLMBackend,
    images: Sequence[Image.Image],
    prompt: str,
    model: type[ModelT],
    *,
    system: str | None = None,
    max_tokens: int = 2048,
    retries: int = 1,
    timeout_s: float | None = None,
) -> tuple[ModelT, GenerationResult, int]:
    """Generate and parse a ``model`` instance, retrying when the output is invalid.

    Returns ``(parsed, last_generation, attempts_used)``. Each retry sends the original
    prompt followed by the latest error, so the model can correct its answer. Raises
    ``JSONParseError`` once all attempts fail, and propagates backend errors and
    ``asyncio.TimeoutError`` (when ``timeout_s`` is set) without retrying.
    """
    total_attempts = max(0, retries) + 1
    current_prompt = prompt
    last_error = ""
    last_result: GenerationResult | None = None

    for attempt in range(1, total_attempts + 1):
        result = await _call_backend(backend, images, current_prompt, system, max_tokens, timeout_s)
        last_result = result
        try:
            data = extract_json(result.text)
            return model.model_validate(data), result, attempt
        except JSONParseError as exc:
            last_error = str(exc)
        except ValidationError as exc:
            last_error = _format_validation_error(exc)

        current_prompt = (
            f"{prompt}\n\nYour previous answer was invalid: {last_error}. Return ONLY valid JSON matching the schema."
        )

    bad_output = last_result.text if last_result is not None else ""
    preview = bad_output[:_BAD_OUTPUT_PREVIEW_CHARS]
    raise JSONParseError(
        f"could not get valid {model.__name__} output after {total_attempts} attempt(s): "
        f"{last_error}. Output starts with: {preview!r}"
    )

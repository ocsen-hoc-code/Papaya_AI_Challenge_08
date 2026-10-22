"""Per-field and classification confidence.

The model's own confidence is weak evidence: it is usually high even when it is wrong. The
score is therefore built mostly from what can be checked: the grounding against the document
text and the normalization outcome. Token probabilities are a mild extra signal.

Field policy (``field_confidence``); ``b`` is the model confidence (default 0.7):

1. Empty value (None, "", [], all-empty dict): ``min(b, 0.1)`` with ``b`` = 0.05 when the model
   gave none. Null fields are always low.
2. Lift for an exact match: ``b' = b + 0.5 * max(0, 0.98 - b)``. Otherwise ``b' = b``.
3. Grounding factor ``g``:
   exact 1.0 (uses step 2), fuzzy 0.9, partial 0.75,
   no_reference (image, no text layer) ``0.6 + 0.25 * grounding.score`` (0.6..0.85),
   not_found 1.0 but capped in step 6.
4. ``c = b' * g``.
5. Token probability, when given: ``c *= 0.5 + 0.5 * token_prob`` (a mild penalty).
   Ambiguous normalization (e.g. a day/month swap): ``c *= 0.7``.
6. Caps: not_found and invalid normalization (``normalized_valid`` False) give ``c <= 0.3``.
7. Round to 2 decimals and clamp to [0, 1].

Classification (``classification_confidence``): normalize the four type scores to
probabilities ``p``. The winner is ``p1`` and the margin is ``p1 - p2`` over the runner-up. The
confidence is ``p1 * (0.5 + 0.5 * margin)``: a clear winner stays high, a near tie falls.
"""

from __future__ import annotations

import math
import re
from typing import Any

from medextract.backends.base import GenerationResult
from medextract.postprocessing.grounding import Grounding, is_empty
from medextract.schemas import DOCUMENT_TYPES, FieldKind

DEFAULT_MODEL_CONFIDENCE = 0.7
NULL_CAP = 0.1
NOT_FOUND_CAP = 0.3
INVALID_CAP = 0.3
AMBIGUOUS_FACTOR = 0.7
EXACT_CEILING = 0.98
GROUNDING_FACTOR = {"exact": 1.0, "fuzzy": 0.9, "partial": 0.75}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


# --- Token probabilities ---------------------------------------------------------------


def span_token_prob(gen: GenerationResult, start: int, end: int) -> float | None:
    """Lowest probability among the tokens that overlap characters ``[start, end)`` of ``gen.text``.

    Needs ``gen.extra["token_texts"]`` (one decoded piece per token, aligned with
    ``gen.token_logprobs``). Returns None when that is missing or the pieces cannot be
    aligned with the text.
    """
    pieces = gen.extra.get("token_texts") if gen.extra else None
    logprobs = gen.token_logprobs
    if not pieces or not logprobs or len(pieces) != len(logprobs) or end <= start:
        return None
    text = gen.text
    cursor = 0
    probs: list[float] = []
    for piece, logprob in zip(pieces, logprobs, strict=True):
        if not piece:
            continue
        idx = text.find(piece, cursor)
        if idx < 0:
            return None
        piece_end = idx + len(piece)
        cursor = piece_end
        if idx < end and piece_end > start:
            probs.append(math.exp(min(logprob, 0.0)))
    return min(probs) if probs else None


_VALUE_WRAPPED_RE = r'"{key}"\s*:\s*\{{'
_VALUE_KEY_RE = r'"value"\s*:\s*'


def _value_extent(text: str, pos: int) -> tuple[int, int] | None:
    """Span of the JSON value that starts at or after ``pos`` (quotes excluded for strings)."""
    n = len(text)
    i = pos
    while i < n and text[i].isspace():
        i += 1
    if i >= n:
        return None
    ch = text[i]
    if ch == '"':
        j = i + 1
        while j < n:
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == '"':
                return i + 1, j
            j += 1
        return None
    if ch in "{[":
        depth = 0
        in_string = False
        j = i
        while j < n:
            c = text[j]
            if in_string:
                if c == "\\":
                    j += 2
                    continue
                if c == '"':
                    in_string = False
            elif c == '"':
                in_string = True
            elif c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    return i, j + 1
            j += 1
        return None
    # Bare literal: number, true, false or null.
    j = i
    while j < n and text[j] not in ",}]" and not text[j].isspace():
        j += 1
    return (i, j) if j > i else None


def find_value_span(text: str, field_name: str) -> tuple[int, int] | None:
    """Locate the value of ``field_name`` in the raw JSON text the model generated.

    For ``"date": {"value": "14 Mar 2024", ...}`` this returns the span of ``14 Mar 2024``
    (without the quotes). Objects and arrays return their full bracketed span. A field
    written without the ``{"value": ...}`` wrapper is also accepted. Returns None when the
    field is not found.
    """
    key = re.escape(field_name)
    wrapped = re.search(_VALUE_WRAPPED_RE.format(key=key), text)
    if wrapped is not None:
        inner = re.compile(_VALUE_KEY_RE).search(text, wrapped.end())
        # The "value" key must belong to this field's object, not to a later one.
        if inner is not None and "}" not in text[wrapped.end() : inner.start()]:
            return _value_extent(text, inner.end())
    direct = re.search(rf'"{key}"\s*:\s*', text)
    if direct is None or text[direct.end() : direct.end() + 1] == "{":
        return None
    return _value_extent(text, direct.end())


# --- Field confidence ------------------------------------------------------------------


def field_confidence(
    *,
    kind: FieldKind,
    value: Any,
    model_confidence: float | None,
    grounding: Grounding,
    normalized_valid: bool = True,
    ambiguous: bool = False,
    token_prob: float | None = None,
) -> float:
    """Final confidence in [0, 1] for one field. See the module docstring for the formula.

    ``kind`` is accepted so that kind-specific policy can be added later; the current
    policy does not depend on it.
    """
    if is_empty(value):
        base = 0.05 if model_confidence is None else _clamp(model_confidence)
        return round(min(base, NULL_CAP), 2)

    base = DEFAULT_MODEL_CONFIDENCE if model_confidence is None else _clamp(model_confidence)
    status = grounding.status
    if status == "exact":
        base = base + 0.5 * max(0.0, EXACT_CEILING - base)
        factor = 1.0
    elif status in GROUNDING_FACTOR:
        factor = GROUNDING_FACTOR[status]
    elif status == "no_reference":
        factor = 0.6 + 0.25 * _clamp(grounding.score)
    else:  # not_found: the cap below does the work
        factor = 1.0

    conf = base * factor
    if token_prob is not None:
        conf *= 0.5 + 0.5 * _clamp(token_prob)
    if ambiguous:
        conf *= AMBIGUOUS_FACTOR
    if status == "not_found":
        conf = min(conf, NOT_FOUND_CAP)
    if not normalized_valid:
        conf = min(conf, INVALID_CAP)
    return round(_clamp(conf), 2)


# --- Classification --------------------------------------------------------------------


def _as_score(raw: Any) -> float:
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score) or score < 0:
        return 0.0
    return score


def classification_confidence(scores: dict[str, float]) -> tuple[str | None, float]:
    """Return ``(best document type, confidence)`` from raw per-type scores.

    Only the four known types count; unknown keys are ignored and negative scores are
    clamped to zero. Empty or all-zero input gives ``(None, 0.0)``.
    """
    values = {t: _as_score(scores.get(t, 0.0)) for t in DOCUMENT_TYPES}
    total = sum(values.values())
    if total <= 0:
        return None, 0.0
    ranked = sorted(DOCUMENT_TYPES, key=lambda t: values[t] / total, reverse=True)
    best = ranked[0]
    p1 = values[best] / total
    p2 = values[ranked[1]] / total
    margin = max(0.0, p1 - p2)
    return best, round(p1 * (0.5 + 0.5 * margin), 3)


# Fields that define each document type. A real document of that type shows them; a
# misclassified document usually does not.
KEY_FIELDS: dict[str, tuple[str, ...]] = {
    "receipt": ("items", "grand_total"),
    "discharge_summary": ("admission_date", "discharge_date", "diagnosis"),
    "lab_report": ("tests",),
    "prescription": ("medications",),
}


def document_confidence(type_confidence: float, document_type: str, field_confidences: dict[str, float]) -> float:
    """Final document-type confidence: the model's type score, corroborated by the content.

    ``support`` is the mean confidence of the type's key fields (0 when a key field is
    missing), so a document whose content does not look like the chosen type loses
    confidence even when the classifier was sure::

        confidence = type_confidence * (0.5 + 0.5 * support)
    """
    keys = KEY_FIELDS.get(document_type, ())
    if not keys:
        return round(max(0.0, min(1.0, type_confidence)), 3)
    support = sum(max(0.0, min(1.0, field_confidences.get(k, 0.0))) for k in keys) / len(keys)
    return round(max(0.0, min(1.0, type_confidence * (0.5 + 0.5 * support))), 3)

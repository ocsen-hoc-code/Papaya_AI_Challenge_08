"""Check extracted values against the document's own text ("no hallucinated data").

The VLM copies each value as printed and quotes the source snippet (``evidence``). This
module measures how well the document supports each value:

* With a usable text layer (PDFs), the value is searched in the text.

  - Strings: exact substring -> ``exact``; rapidfuzz ``partial_ratio`` >= 90 -> ``fuzzy``;
    otherwise ``not_found``. Long text values are checked sentence by sentence.
  - Amounts and quantities: some number token in the text reads as the same value
    (``1.500.000``, ``1,500.00``, ``1.500,50``, Thai digits) -> ``exact``; else ``not_found``.
  - Dates: the quoted evidence is searched and must agree on day, month and year (the
    Buddhist year, year + 543, also counts). Without evidence the parts only have to appear
    somewhere in the text, which is deliberately lenient -> ``partial``.
  - Lists and structured values: each item's main string and numeric sub-fields are
    grounded separately. The score is the mean; the status is ``partial`` when some fail.

* Without a text layer (images), nothing can be checked exactly. The quoted evidence is
  compared with the value: consistent -> score ~0.6, missing -> ~0.4, inconsistent -> ~0.2.
  The status is ``no_reference``.

* A null value gets ``no_reference`` with score 0.0.

``confidence.py`` turns the result into the final per-field confidence.
"""

from __future__ import annotations

import datetime
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from rapidfuzz import fuzz

from medextract.schemas import FieldKind

GroundingStatus = Literal["exact", "fuzzy", "partial", "not_found", "no_reference"]

FUZZY_THRESHOLD = 90  # partial_ratio on folded text, 0..100
EVIDENCE_THRESHOLD = 85  # value vs quoted evidence when there is no text layer
MIN_FUZZY_LEN = 4  # shorter strings can only match exactly
SENTENCE_MIN_LEN = 150  # text values longer than this are grounded sentence by sentence
SENTENCE_MIN_CHARS = 8

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_CID_RE = re.compile(r"\(cid:\d+\)")
_NUMBER_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*")
_NUMERIC_STRING_RE = re.compile(r"\d+(?:[.,]\d+)*")
_INT_TOKEN_RE = re.compile(r"\d+")
_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_SENTENCE_SPLIT_RE = re.compile(r"[.;!?\n]+")
_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_SCALAR_KINDS = frozenset({FieldKind.TEXT, FieldKind.DATE, FieldKind.AMOUNT})

# For structured kinds: (main string key, numeric sub-field keys) of each item.
_ITEM_FIELDS: dict[FieldKind, tuple[str, tuple[str, ...]]] = {
    FieldKind.RECEIPT_ITEMS: ("description", ("quantity", "unit_price", "total")),
    FieldKind.LAB_TESTS: ("test_name", ("result",)),
    FieldKind.MEDICATIONS: ("name", ("quantity",)),
}

# A leaf is one checkable piece of a value: ("text", str), ("number", float) or ("date", raw).
Leaf = tuple[str, Any]


@dataclass
class Grounding:
    status: GroundingStatus
    score: float  # 0..1: how well the document text supports the value
    detail: str | None = None


# --- Text layer quality ----------------------------------------------------------------


def text_layer_usable(text: str | None) -> bool:
    """Return False when the text layer is missing, empty or garbled.

    Garbled means: many U+FFFD replacement characters, many ``(cid:N)`` font tokens, or
    mostly symbols instead of letters, digits and combining marks.
    """
    if not text:
        return False
    body = "".join(ch for ch in text if not ch.isspace())
    if len(body) < 3:
        return False
    if text.count("�") > 0.05 * len(body):
        return False
    cid_chars = sum(len(m.group()) for m in _CID_RE.finditer(text))
    if cid_chars > 0.1 * len(body):
        return False
    # L* letters, M* combining marks (Thai vowels and tone marks), N* numbers.
    wordlike = sum(1 for ch in body if unicodedata.category(ch)[0] in "LMN")
    return wordlike / len(body) >= 0.3


# --- Normalization helpers -------------------------------------------------------------


def _fold(text: str) -> str:
    """NFC, Thai digits to ASCII, casefold, collapsed whitespace."""
    text = unicodedata.normalize("NFC", text).translate(_THAI_DIGITS).casefold()
    return " ".join(text.split())


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):
        return all(is_empty(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(is_empty(v) for v in value)
    return False


def _read_number(token: str) -> float:
    """Read a digit token that may contain '.' or ',' separators.

    Groups of exactly three digits after one kind of separator are thousands groups
    (``1.500.000``, ``15,750``). Otherwise the last separator is the decimal mark
    (``1,500.00``, ``1.500,50``, ``12.5``).
    """
    groups = re.split(r"[.,]", token)
    seps = set(re.findall(r"[.,]", token))
    if len(groups) == 1:
        return float(token)
    if len(seps) == 1 and all(len(g) == 3 for g in groups[1:]):
        return float("".join(groups))
    return float("".join(groups[:-1]) + "." + groups[-1])


def _numbers_in(folded: str) -> list[float]:
    return [_read_number(tok) for tok in _NUMBER_TOKEN_RE.findall(folded)]


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        folded = _fold(value)
        if _NUMERIC_STRING_RE.fullmatch(folded):
            return _read_number(folded)
    return None


def _date_parts(value: Any) -> tuple[int, int, int] | None:
    """(year, month, day) of an ISO date string or a date object, else None."""
    if isinstance(value, datetime.date):
        return value.year, value.month, value.day
    if isinstance(value, str):
        match = _ISO_DATE_RE.fullmatch(value.strip())
        if match:
            return int(match[1]), int(match[2]), int(match[3])
    return None


def _month_name_in(month: int, folded: str) -> bool:
    if not 1 <= month <= 12:
        return False
    name = _MONTHS[month - 1]
    return re.search(rf"\b(?:{name}|{name[:3]})\b", folded) is not None


def _date_components(parts: tuple[int, int, int], folded: str) -> tuple[bool, bool, bool]:
    """Whether (year, day, month) appear in the folded text. Year may be Buddhist (+543)."""
    year, month, day = parts
    ints = {int(t) for t in _INT_TOKEN_RE.findall(folded)}
    year_ok = year in ints or year + 543 in ints
    day_ok = day in ints
    month_ok = month in ints or _month_name_in(month, folded)
    return year_ok, day_ok, month_ok


def _best_ratio(needle: str, haystack: str) -> float:
    if len(needle) < MIN_FUZZY_LEN:
        return 0.0
    return float(fuzz.partial_ratio(needle, haystack))


def _aggregate(results: list[Grounding]) -> Grounding:
    """Combine several leaf groundings: mean score; status from the mix of outcomes."""
    if len(results) == 1:
        return results[0]
    score = round(sum(r.score for r in results) / len(results), 3)
    failed = [r for r in results if r.status == "not_found"]
    if len(failed) == len(results):
        status: GroundingStatus = "not_found"
    elif failed:
        status = "partial"
    elif all(r.status == "exact" for r in results):
        status = "exact"
    elif all(r.status in ("exact", "fuzzy") for r in results):
        status = "fuzzy"
    else:
        status = "partial"
    detail = f"{len(results) - len(failed)}/{len(results)} parts grounded"
    if failed:
        detail += "; missing: " + "; ".join(r.detail or "?" for r in failed[:3])
    return Grounding(status, score, detail)


# --- Leaves ----------------------------------------------------------------------------


def _item_leaves(item: Any, main_key: str, numeric_keys: tuple[str, ...]) -> list[Leaf]:
    if is_empty(item):
        return []
    if not isinstance(item, dict):
        return [("text", str(item))]
    leaves: list[Leaf] = []
    if not is_empty(item.get(main_key)):
        leaves.append(("text", str(item[main_key])))
    for key in numeric_keys:
        raw = item.get(key)
        if is_empty(raw):
            continue
        number = _to_number(raw)
        leaves.append(("number", number) if number is not None else ("text", str(raw)))
    return leaves


def _leaves(kind: FieldKind, value: Any) -> list[Leaf]:
    """Split a non-empty value into the pieces that can be grounded one by one."""
    if kind is FieldKind.TEXT:
        return [("text", str(value))]
    if kind is FieldKind.DATE:
        return [("date", value)]
    if kind is FieldKind.AMOUNT:
        number = _to_number(value)
        return [("number", number)] if number is not None else [("text", str(value))]
    if kind is FieldKind.TEXT_LIST:
        if isinstance(value, (list, tuple)):
            items = list(value)
        else:
            items = [value]
        return [("text", str(v)) for v in items if not is_empty(v)]
    if kind is FieldKind.DIAGNOSIS:
        if isinstance(value, dict):
            secondary = value.get("secondary")
            if isinstance(secondary, str):
                secondary = [secondary]
            parts = [value.get("primary"), *(secondary or [])]
        else:
            parts = [value]
        return [("text", str(p)) for p in parts if not is_empty(p)]
    if kind in _ITEM_FIELDS:
        main_key, numeric_keys = _ITEM_FIELDS[kind]
        items = list(value) if isinstance(value, (list, tuple)) else [value]
        leaves: list[Leaf] = []
        for item in items:
            leaves.extend(_item_leaves(item, main_key, numeric_keys))
        return leaves
    return [("text", str(value))]


# --- Text-layer grounding --------------------------------------------------------------


def _ground_short(value: str, ref: str, evidence: str | None) -> Grounding:
    """Ground one folded string against the folded document text."""
    if value in ref:
        return Grounding("exact", 1.0, "found verbatim in text")
    ratio = _best_ratio(value, ref)
    if ratio >= FUZZY_THRESHOLD:
        return Grounding("fuzzy", round(ratio / 100, 3), f"fuzzy match {ratio:.0f}/100")
    if evidence:
        ev = _fold(evidence)
        ev_in_text = ev in ref
        if ev_in_text or _best_ratio(ev, ref) >= FUZZY_THRESHOLD:
            score = 0.7 if ev_in_text else 0.6
            return Grounding("partial", score, "value not in text, but the quoted evidence is")
    # Low but graded: a near miss is still less wrong than an unrelated string.
    return Grounding(
        "not_found",
        round(0.05 + 0.2 * ratio / 100, 3),
        f"{value!r} not found in text (best match {ratio:.0f}/100)",
    )


def _ground_text(value: str, ref: str, evidence: str | None) -> Grounding:
    folded = _fold(value)
    if not folded:
        return Grounding("no_reference", 0.0, "empty value")
    if folded in ref:
        return Grounding("exact", 1.0, "found verbatim in text")
    if len(folded) > SENTENCE_MIN_LEN:
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(folded)]
        sentences = [s for s in sentences if len(s) >= SENTENCE_MIN_CHARS]
        if len(sentences) > 1:
            return _aggregate([_ground_short(s, ref, None) for s in sentences])
    return _ground_short(folded, ref, evidence)


def _ground_number(value: Any, ref: str) -> Grounding:
    target = _to_number(value)
    if target is None:
        return _ground_text(str(value), ref, None)
    if any(abs(n - target) < 0.005 for n in _numbers_in(ref)):
        return Grounding("exact", 1.0, f"number {target} found in text")
    return Grounding("not_found", 0.1, f"no number in text reads as {target}")


def _ground_date(value: Any, ref: str, evidence: str | None) -> Grounding:
    parts = _date_parts(value)
    if parts is None:
        return _ground_text(str(value), ref, evidence)
    ev = _fold(evidence) if evidence and evidence.strip() else ""
    if ev:
        ev_ratio = 100.0 if ev in ref else _best_ratio(ev, ref)
        if ev_ratio >= FUZZY_THRESHOLD:
            if all(_date_components(parts, ev)):
                verbatim = ev in ref
                return Grounding(
                    "exact" if verbatim else "fuzzy",
                    0.95 if verbatim else 0.85,
                    "quoted date found in text and agrees with the value",
                )
            return Grounding("partial", 0.5, "quoted date found in text but its day/month/year differ")
    found = sum(_date_components(parts, ref))
    if found == 3:
        return Grounding("partial", 0.55, "day, month and year appear in text (no matching quote)")
    if found == 2:
        return Grounding("partial", 0.35, "only some date parts appear in text")
    return Grounding("not_found", 0.1, "date parts not found in text")


def _ground_leaf(leaf: Leaf, ref: str, evidence: str | None) -> Grounding:
    kind, raw = leaf
    if kind == "number":
        return _ground_number(raw, ref)
    if kind == "date":
        return _ground_date(raw, ref, evidence)
    return _ground_text(str(raw), ref, evidence)


# --- Image path (no text layer) --------------------------------------------------------


def _leaf_consistent(leaf: Leaf, ev: str) -> bool:
    """Whether the quoted evidence (folded) supports one leaf of the value."""
    kind, raw = leaf
    if kind == "number":
        return any(abs(n - raw) < 0.005 for n in _numbers_in(ev))
    if kind == "date":
        parts = _date_parts(raw)
        if parts is not None:
            return all(_date_components(parts, ev))
    text = _fold(str(raw))
    if not text:
        return False
    return text in ev or fuzz.partial_ratio(text, ev) >= EVIDENCE_THRESHOLD


def _ground_without_text(kind: FieldKind, value: Any, evidence: str | None) -> Grounding:
    if not evidence or not evidence.strip():
        return Grounding("no_reference", 0.4, "no text layer and no quoted evidence")
    ev = _fold(evidence)
    leaves = _leaves(kind, value)
    checks = [_leaf_consistent(leaf, ev) for leaf in leaves]
    if not checks:
        return Grounding("no_reference", 0.4, "no text layer; nothing to compare with the evidence")
    support = sum(checks) / len(checks)
    score = round(0.2 + 0.4 * support, 3)
    if support == 1:
        detail = "no text layer; quoted evidence is consistent with the value"
    elif support == 0:
        detail = "no text layer; quoted evidence does not match the value"
    else:
        detail = "no text layer; quoted evidence partly matches the value"
    return Grounding("no_reference", score, detail)


# --- Public API ------------------------------------------------------------------------


def ground_field(kind: FieldKind, value: Any, evidence: str | None, text_layer: str | None) -> Grounding:
    """Measure how well the document supports ``value`` for a field of the given ``kind``.

    ``evidence`` is the snippet the model quoted for the field (may be None).
    ``text_layer`` is the PDF's embedded text (None for images).
    """
    if is_empty(value):
        return Grounding("no_reference", 0.0, "no value to ground")
    if not text_layer_usable(text_layer):
        return _ground_without_text(kind, value, evidence)
    leaves = _leaves(kind, value)
    if not leaves:
        return Grounding("no_reference", 0.3, "structured value without groundable sub-fields")
    ref = _fold(text_layer or "")
    if kind in _SCALAR_KINDS:
        return _ground_leaf(leaves[0], ref, evidence)
    return _aggregate([_ground_leaf(leaf, ref, None) for leaf in leaves])

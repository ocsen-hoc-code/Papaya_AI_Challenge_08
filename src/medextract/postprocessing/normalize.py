"""Normalize raw VLM values into canonical forms.

The VLM returns values as printed on the document: strings in English, Vietnamese or Thai,
Thai digits, local date and money formats, or bare JSON numbers. This module turns them into
canonical values: ISO dates, floats for money and quantities, and lists or dicts for the
structured fields.

Every public function returns a ``Normalized`` and never raises. Anything that cannot be parsed
comes back as ``value=None`` with ``valid=False`` and a human-readable ``issue``. A missing raw
value (``None``, empty string, or a placeholder such as ``"N/A"``) is not an error: it comes back
as ``value=None`` with ``valid=True``.
"""

from __future__ import annotations

import datetime as _dt
import functools
import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from medextract.schemas import ITEM_MODELS, Diagnosis, FieldKind

Language = Literal["en", "vi", "th"] | None

_LANGUAGES = ("en", "vi", "th")
_MISSING_TOKENS = frozenset({"", "-", "--", "—", "–", "...", "n/a", "n.a.", "na", "none", "null", "nil"})


@dataclass
class Normalized:
    """Outcome of normalizing one raw value."""

    value: Any
    valid: bool = True  # False if a non-empty raw value could not be parsed or is impossible
    ambiguous: bool = False  # e.g. 03/04/2024 without a language hint (dd/mm vs mm/dd)
    issue: str | None = None  # human-readable reason when invalid or ambiguous


def _invalid(issue: str) -> Normalized:
    return Normalized(None, valid=False, issue=issue)


def _never_raise(func: Callable[..., Normalized]) -> Callable[..., Normalized]:
    """Last-resort guard: an unexpected exception becomes an invalid result."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Normalized:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # normalization must never raise on model output
            return _invalid(f"could not normalize value: {exc}")

    return wrapper


# --- Shared helpers -------------------------------------------------------------------

_THAI_LETTER_RE = re.compile(r"[ก-ฺเ-๎]")
_THAI_ANY_RE = re.compile(r"[฀-๿]")
_VI_LOWER_SPECIAL = frozenset("ăâđêôơư")


def _is_vi_letter(ch: str) -> bool:
    low = ch.lower()
    return low in _VI_LOWER_SPECIAL or 0x1EA0 <= ord(low) <= 0x1EF9


def detect_language(text: str | None) -> str | None:
    """Guess the language of a text: "th", "vi", "en" or None.

    Thai wins when it is the majority of the letters. Otherwise any Vietnamese diacritic means
    "vi". Otherwise mostly-ASCII letters mean "en". Anything else gives None.
    """
    if not text or not isinstance(text, str):
        return None
    thai = ascii_letters = other_letters = vi = 0
    for ch in text:
        if _THAI_LETTER_RE.match(ch):
            thai += 1
        elif ch.isalpha():
            if ch.isascii():
                ascii_letters += 1
            else:
                other_letters += 1
                if _is_vi_letter(ch):
                    vi += 1
    letters = thai + ascii_letters + other_letters
    if letters == 0:
        return None
    if thai * 2 > letters:
        return "th"
    if vi:
        return "vi"
    latin = ascii_letters + other_letters
    if latin and ascii_letters * 2 >= latin:
        return "en"
    return None


def _ascii_digits(text: str) -> str:
    """Replace any Unicode decimal digit (Thai ๐-๙, full-width, ...) with its ASCII form."""
    return "".join(str(unicodedata.decimal(ch)) if ch.isdecimal() else ch for ch in text)


def _is_missing(text: str) -> bool:
    return text.strip().lower() in _MISSING_TOKENS


_WRAPPERS = {"(": ")", "[": "]", '"': '"', "'": "'", "“": "”"}


def _unwrap(text: str) -> str:
    """Remove brackets or quotes that enclose the whole value, e.g. "( Dr. A )" -> "Dr. A".

    Signature lines often print names in brackets. Only a single pair that wraps the entire
    string is removed, so values like "Diabetes (E11)" keep their brackets.
    """
    while len(text) >= 2 and _WRAPPERS.get(text[0]) == text[-1]:
        inner = text[1:-1]
        if text[0] == "(" and ("(" in inner or ")" in inner):
            break
        text = inner.strip()
    return text


def _text(raw: Any) -> Normalized:
    """Plain text, kept verbatim apart from surrounding whitespace and enclosing brackets."""
    if raw is None:
        return Normalized(None)
    if isinstance(raw, bool):
        return _invalid(f"expected text, got {raw!r}")
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and not math.isfinite(raw):
            return _invalid(f"expected text, got {raw!r}")
        return Normalized(str(raw))
    if isinstance(raw, str):
        stripped = _unwrap(raw.strip())
        return Normalized(None if _is_missing(stripped) else stripped)
    if isinstance(raw, (list, tuple)):
        parts: list[str] = []
        valid = True
        issues: list[str] = []
        for entry in raw:
            sub = _text(entry)
            if not sub.valid:
                valid = False
                issues.append(sub.issue or "unsupported entry")
            elif sub.value:
                parts.append(sub.value)
        return Normalized(", ".join(parts) or None, valid=valid, issue="; ".join(issues) or None)
    return _invalid(f"expected text, got {type(raw).__name__}")


_BULLET_RE = re.compile(r"^\s*(?:[-*•·]+\s*|\d+[.)]\s+)")
_LIST_SPLIT_RE = re.compile(r"\n|;|•|\||,")


def _text_list(raw: Any) -> Normalized:
    """List of non-empty strings, from a list or from a delimited string."""
    if raw is None:
        return Normalized(None)
    if isinstance(raw, (list, tuple)):
        items: list[str] = []
        valid = True
        issues: list[str] = []
        for entry in raw:
            sub = _text(entry)
            if not sub.valid:
                valid = False
                issues.append(sub.issue or "unsupported entry")
            elif sub.value:
                items.append(sub.value)
        return Normalized(items, valid=valid, issue="; ".join(issues) or None)
    if isinstance(raw, (str, int, float)) and not isinstance(raw, bool):
        text = _text(raw).value
        if text is None:
            return Normalized(None)
        pieces = [_BULLET_RE.sub("", piece).strip() for piece in _LIST_SPLIT_RE.split(text)]
        return Normalized([piece for piece in pieces if piece])
    return _invalid(f"expected a list of text, got {type(raw).__name__}")


# --- Dates -----------------------------------------------------------------------------

_MONTH_NAMES: tuple[tuple[str, int], ...] = (
    # English (full names and abbreviations)
    ("january", 1),
    ("jan", 1),
    ("february", 2),
    ("feb", 2),
    ("march", 3),
    ("mar", 3),
    ("april", 4),
    ("apr", 4),
    ("may", 5),
    ("june", 6),
    ("jun", 6),
    ("july", 7),
    ("jul", 7),
    ("august", 8),
    ("aug", 8),
    ("september", 9),
    ("sept", 9),
    ("sep", 9),
    ("october", 10),
    ("oct", 10),
    ("november", 11),
    ("nov", 11),
    ("december", 12),
    ("dec", 12),
    # Thai (full names and the usual abbreviations, which carry dots)
    ("มกราคม", 1),
    ("ม.ค.", 1),
    ("กุมภาพันธ์", 2),
    ("ก.พ.", 2),
    ("มีนาคม", 3),
    ("มี.ค.", 3),
    ("เมษายน", 4),
    ("เม.ย.", 4),
    ("พฤษภาคม", 5),
    ("พ.ค.", 5),
    ("มิถุนายน", 6),
    ("มิ.ย.", 6),
    ("กรกฎาคม", 7),
    ("ก.ค.", 7),
    ("สิงหาคม", 8),
    ("ส.ค.", 8),
    ("กันยายน", 9),
    ("ก.ย.", 9),
    ("ตุลาคม", 10),
    ("ต.ค.", 10),
    ("พฤศจิกายน", 11),
    ("พ.ย.", 11),
    ("ธันวาคม", 12),
    ("ธ.ค.", 12),
)


def _month_regex(key: str) -> re.Pattern[str]:
    body = re.escape(key).replace(r"\.", r"\.?")  # dots in abbreviations are optional
    if key.isascii():
        return re.compile(rf"(?<![A-Za-z]){body}(?![A-Za-z])")
    return re.compile(rf"(?<![A-Za-z฀-๿]){body}(?![A-Za-z฀-๿])")


_MONTH_PATTERNS: tuple[tuple[re.Pattern[str], int, int], ...] = tuple(
    (_month_regex(name), month, len(name)) for name, month in _MONTH_NAMES
)

_TIME_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:\s*(?:[ap]\.?m\.?|น\.))?(?:\s*(?:z|[+-]\d{2}:?\d{2}))?")
_ORDINAL_RE = re.compile(r"(?<=\d)(?:st|nd|rd|th)\b")
_VI_FILLER_RE = re.compile(r"(?<![^\W\d_])(?:ngày|ngay|tháng|thang|năm|nam)(?![^\W\d_])")
_TH_FILLERS = ("วันที่", "เดือน", "ปี", "พ.ศ.", "ค.ศ.", "พ.ศ", "ค.ศ")

_NUM_STRICT_RE = re.compile(r"(?<!\d)(\d{1,4})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{4}|\d{2})(?!\d)")
_NUM_LOOSE_RE = re.compile(r"(?<!\d)(\d{1,4})[\s,]+(\d{1,2})[\s,]+(\d{4}|\d{2})(?!\d)")
_NUM_COMPACT_RE = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)")


def _clean_date_text(text: str) -> str:
    """Lower-case the text, ASCII-fy digits and drop times and filler words."""
    t = _ascii_digits(text).strip().lower()
    t = re.sub(r"(?<=\d)t(?=\d)", " ", t)  # ISO "2024-03-15T10:30" separator
    t = _TIME_RE.sub(" ", t)
    t = _ORDINAL_RE.sub(" ", t)
    t = _VI_FILLER_RE.sub(" ", t)
    for word in _TH_FILLERS:
        t = t.replace(word, " ")
    return t


def _find_month(text: str) -> tuple[int, int, int] | None:
    """Return (month, start, end) of the earliest month name, preferring the longest match."""
    best: tuple[int, int, int, int] | None = None  # (start, -length, month, end)
    for pattern, month, _length in _MONTH_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        candidate = (match.start(), -(match.end() - match.start()), month, match.end())
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None
    return best[2], best[0], best[3]


def _year_value(token: str, thai_ctx: bool) -> int | None:
    """Full AD year from a 2- or 4-digit token. Buddhist-era years (> 2400) lose 543 years.

    In a Thai context a 2-digit year is a Buddhist-era year (67 -> BE 2567 -> AD 2024).
    Elsewhere it is read as 20yy.
    """
    if not token.isdigit():
        return None
    if len(token) <= 2:
        yy = int(token)
        return 1957 + yy if thai_ctx else 2000 + yy
    if len(token) != 4:
        return None
    year = int(token)
    if year > 2400:
        year -= 543
    return year


def _safe_date(year: int | None, month: int, day: int) -> _dt.date | None:
    if year is None or not 1900 <= year <= 2100:
        return None
    try:
        return _dt.date(year, month, day)
    except ValueError:
        return None


def _parse_named_month(text: str, thai_ctx: bool, found: tuple[int, int, int]) -> tuple[_dt.date | None, str | None]:
    month, start, end = found
    numbers = re.findall(r"\d+", text[:start]) + re.findall(r"\d+", text[end:])
    if len(numbers) != 2:
        return None, "could not find both a day and a year next to the month name"
    first, second = numbers
    if len(first) >= 3 and len(second) < 3:
        year_tok, day_tok = first, second
    elif len(second) >= 3 and len(first) < 3:
        year_tok, day_tok = second, first
    else:  # positional: "15 Mar 24", "Mar 15, 2024", "15 มีนาคม 2567"
        day_tok, year_tok = first, second
    day = _safe_date(_year_value(year_tok, thai_ctx), month, int(day_tok)) if day_tok.isdigit() else None
    if day is None:
        return None, "not a valid calendar date"
    return day, None


def _parse_numeric_date(text: str, language: str | None, thai_ctx: bool) -> tuple[_dt.date | None, bool, str | None]:
    match = _NUM_STRICT_RE.search(text) or _NUM_LOOSE_RE.search(text) or _NUM_COMPACT_RE.search(text)
    if match is None:
        return None, False, "no day, month and year found"
    first, second, third = match.groups()
    if len(first) == 4:  # year first: YYYY-MM-DD, YYYYMMDD
        day = _safe_date(_year_value(first, thai_ctx), int(second), int(third))
        return (day, False, None) if day else (None, False, "not a valid calendar date")

    year = _year_value(third, thai_ctx)
    day_first = _safe_date(year, int(second), int(first))  # dd/mm/yyyy
    month_first = _safe_date(year, int(first), int(second))  # mm/dd/yyyy
    if day_first and month_first and day_first != month_first:
        if language in ("vi", "th"):
            return day_first, False, None  # dd/mm is the local convention
        issue = (
            f"ambiguous date {match.group(0)!r}: read as {day_first.isoformat()} (day first); "
            f"month first would give {month_first.isoformat()}"
        )
        return day_first, True, issue
    if day_first:
        return day_first, False, None
    if month_first:  # only one reading is a real calendar date, e.g. 03/15/2024
        return month_first, False, None
    return None, False, "not a valid calendar date"


def _parse_date_text(text: str, language: str | None, thai_ctx: bool) -> tuple[_dt.date | None, bool, str | None]:
    cleaned = _clean_date_text(text)
    found = _find_month(cleaned)
    if found is not None:
        day, issue = _parse_named_month(cleaned, thai_ctx, found)
        return day, False, issue
    return _parse_numeric_date(cleaned, language, thai_ctx)


@_never_raise
def normalize_date(raw: Any, language: str | None = None) -> Normalized:
    """Normalize a date to ISO ``YYYY-MM-DD``.

    Day-first is the default. Only the month-name forms and numeric dates with a unique valid
    reading are unambiguous. For ``vi`` and ``th`` the dd/mm reading is the local convention, so
    it is taken without a flag. For ``en`` or an unknown language, a numeric date that reads
    differently day-first and month-first is returned day-first with ``ambiguous=True``. A date
    where only one reading is a real calendar date uses that reading.
    """
    if raw is None:
        return Normalized(None)
    if isinstance(raw, _dt.datetime):
        return Normalized(raw.date().isoformat())
    if isinstance(raw, _dt.date):
        return Normalized(raw.isoformat())
    if isinstance(raw, (bool, list, tuple, dict, set)):
        return _invalid(f"expected a date, got {type(raw).__name__}")
    text = str(raw)
    if _is_missing(text):
        return Normalized(None)
    lang = language if language in _LANGUAGES else detect_language(text)
    thai_ctx = lang == "th" or bool(_THAI_ANY_RE.search(text))
    day, ambiguous, issue = _parse_date_text(text, lang, thai_ctx)
    if day is None:
        return _invalid(f"could not parse {text!r} as a date" if issue is None else f"{text!r}: {issue}")
    return Normalized(day.isoformat(), valid=True, ambiguous=ambiguous, issue=issue if ambiguous else None)


# --- Numbers and amounts ---------------------------------------------------------------

_CURRENCY_RE = re.compile(r"us\$|vnđ|vnd|đồng|₫|đ|usd|thb|baht|บาท|฿|\$|€|£|¥", re.IGNORECASE)


def _parse_grouped(text: str) -> float | None:
    """Parse digits with optional '.' or ',' separators into a float.

    Rules, applied to unsigned text:
    * both '.' and ',' present: the last one is the decimal mark, the other groups thousands
      (groups of exactly 3 digits). ``1.500,50`` and ``1,500.50`` both give 1500.5.
    * one separator, repeated: thousands separators (groups of 3). ``1.500.000`` -> 1500000.
    * one separator, once: exactly 3 digits after it is a thousands separator (``1,500`` ->
      1500, ``15.750`` -> 15750), unless the integer part is 0 (``0.500`` -> 0.5). Any other
      digit count is a decimal mark (``1,5`` -> 1.5, ``1.50`` -> 1.5).
    """
    if not re.fullmatch(r"\d+(?:[.,]\d+)*", text):
        return None
    seps = {ch for ch in text if ch in ".,"}
    if not seps:
        return float(text)
    if len(seps) == 2:
        decimal = text[max(text.rfind("."), text.rfind(","))]
        thousands = "," if decimal == "." else "."
        head, _, tail = text.rpartition(decimal)
        if decimal in head:
            return None
        groups = head.split(thousands)
        if not 1 <= len(groups[0]) <= 3 or not all(len(group) == 3 for group in groups[1:]):
            return None
        return float("".join(groups) + "." + tail)
    separator = seps.pop()
    parts = text.split(separator)
    if len(parts) > 2:
        if 1 <= len(parts[0]) <= 3 and all(len(part) == 3 for part in parts[1:]):
            return float("".join(parts))
        return None
    int_part, frac = parts
    if len(frac) == 3 and int_part not in ("", "0"):
        return float(int_part + frac)
    return float(f"{int_part or '0'}.{frac}")


def _parse_money(text: str) -> float | None:
    """Parse money text with currency marks, spaces and sign/parentheses. None if unparseable."""
    s = _ascii_digits(text).lower()
    s = _CURRENCY_RE.sub(" ", s)
    s = re.sub(r"\s+", "", s)
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1]
    if s and s[0] in "+-−":
        negative = negative or s[0] in "-−"
        s = s[1:]
    if s and s[-1] in "-−":
        negative = True
        s = s[:-1]
    value = _parse_grouped(s)
    if value is None:
        return None
    return -value if negative else value


@_never_raise
def normalize_amount(raw: Any) -> Normalized:
    """Normalize money to a float. The sign is kept so that validation can flag negatives."""
    if raw is None:
        return Normalized(None)
    if isinstance(raw, bool):
        return _invalid(f"expected an amount, got {raw!r}")
    if isinstance(raw, (int, float)):
        if not math.isfinite(raw):
            return _invalid(f"expected an amount, got {raw!r}")
        return Normalized(float(raw))
    if isinstance(raw, str):
        if _is_missing(raw):
            return Normalized(None)
        value = _parse_money(raw)
        if value is None:
            return _invalid(f"could not parse {raw!r} as an amount")
        return Normalized(value)
    return _invalid(f"expected an amount, got {type(raw).__name__}")


_FRACTION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_FIRST_NUMBER_RE = re.compile(r"([-+−]?)(\d+(?:[.,]\d+)*)")


@_never_raise
def normalize_number(raw: Any) -> Normalized:
    """Normalize a quantity: the first number in the text, e.g. ``"2 tablets"`` -> 2.0."""
    if raw is None:
        return Normalized(None)
    if isinstance(raw, bool):
        return _invalid(f"expected a number, got {raw!r}")
    if isinstance(raw, (int, float)):
        if not math.isfinite(raw):
            return _invalid(f"expected a number, got {raw!r}")
        return Normalized(float(raw))
    if not isinstance(raw, str):
        return _invalid(f"expected a number, got {type(raw).__name__}")
    if _is_missing(raw):
        return Normalized(None)
    text = _ascii_digits(raw)
    fraction = _FRACTION_RE.search(text)
    if fraction is not None:
        numerator, denominator = int(fraction.group(1)), int(fraction.group(2))
        if denominator:
            return Normalized(numerator / denominator)
    match = _FIRST_NUMBER_RE.search(text)
    value = _parse_grouped(match.group(2)) if match else None
    if match is None or value is None:
        return _invalid(f"could not parse {raw!r} as a number")
    return Normalized(-value if match.group(1) in ("-", "−") else value)


# --- Structured sub-values -------------------------------------------------------------

_FLAG_WORDS: dict[str, str] = {
    "high": "high",
    "h": "high",
    "hi": "high",
    "hh": "high",
    "up": "high",
    "↑": "high",
    "cao": "high",
    "สูง": "high",
    "low": "low",
    "l": "low",
    "lo": "low",
    "ll": "low",
    "down": "low",
    "↓": "low",
    "thấp": "low",
    "ต่ำ": "low",
    "normal": "normal",
    "n": "normal",
    "nl": "normal",
    "bình thường": "normal",
    "ปกติ": "normal",
}


def _normalize_flag(raw: Any) -> Normalized:
    if raw is None:
        return Normalized(None)
    if not isinstance(raw, str):
        return _invalid(f"expected a lab flag, got {type(raw).__name__}")
    key = " ".join(re.sub(r"[*()\[\]]", "", raw).strip().lower().split())
    if _is_missing(key):
        return Normalized(None)
    if key in _FLAG_WORDS:
        return Normalized(_FLAG_WORDS[key])
    return _invalid(f"unrecognized lab flag {raw!r}")


_BARE_NUMBER_RE = re.compile(r"^[-+−]?\d+(?:[.,]\d+)*$")


def _lab_result(raw: Any) -> Normalized:
    """Lab result: a bare number becomes a float, anything else stays as printed.

    "7.8" -> 7.8, "5,6" (decimal comma) -> 5.6, "๑๒.๕" -> 12.5. Results with qualifiers or
    words ("< 0.5", "7.8 mmol/L", "Positive") are kept verbatim as strings.
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return normalize_number(raw)
    text = _text(raw)
    if isinstance(text.value, str) and _BARE_NUMBER_RE.match(_ascii_digits(text.value)):
        number = normalize_number(text.value)
        if number.valid and number.value is not None:
            return number
    return text


_ITEM_SUBFIELDS: dict[FieldKind, dict[str, Callable[[Any], Normalized]]] = {
    FieldKind.RECEIPT_ITEMS: {
        "description": _text,
        "quantity": normalize_number,
        "unit_price": normalize_amount,
        "total": normalize_amount,
    },
    FieldKind.LAB_TESTS: {
        "test_name": _text,
        "result": _lab_result,
        "unit": _text,
        "reference_range": _text,
        "flag": _normalize_flag,
    },
    FieldKind.MEDICATIONS: {
        "name": _text,
        "dosage": _text,
        "frequency": _text,
        "duration": _text,
        "quantity": normalize_number,
    },
}
# Each sub-model must describe exactly the subfields we normalize.
for _kind, _subs in _ITEM_SUBFIELDS.items():
    if set(_subs) != set(ITEM_MODELS[_kind].model_fields):
        raise RuntimeError(f"subfields of {_kind} do not match ITEM_MODELS")


def _normalize_items(kind: FieldKind, raw: Any) -> Normalized:
    """List of structured items (receipt items, lab tests, medications) as plain dicts."""
    if raw is None:
        return Normalized(None)
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return _invalid(f"expected a list, got {type(raw).__name__}")
    model = ITEM_MODELS[kind]
    subfields = _ITEM_SUBFIELDS[kind]
    items: list[dict[str, Any]] = []
    issues: list[str] = []
    valid = True
    for index, entry in enumerate(raw):
        if entry is None or (isinstance(entry, str) and _is_missing(entry)):
            continue
        if not isinstance(entry, dict):
            valid = False
            issues.append(f"item {index}: expected an object, got {type(entry).__name__}")
            continue
        values: dict[str, Any] = {}
        for name, normalize in subfields.items():
            result = normalize(entry.get(name))
            values[name] = result.value
            if not result.valid:
                valid = False
                issues.append(f"item {index} {name}: {result.issue}")
        if all(value is None for value in values.values()):
            continue
        items.append(model.model_validate(values).model_dump())
    return Normalized(items, valid=valid, issue="; ".join(issues) or None)


def _normalize_diagnosis(raw: Any) -> Normalized:
    """Diagnosis from a dict {primary, secondary}, a list (first = primary) or a string."""
    if raw is None:
        return Normalized(None)
    valid = True
    issues: list[str] = []
    if isinstance(raw, dict):
        primary_result = _text(raw.get("primary"))
        secondary_result = _text_list(raw.get("secondary"))
        primary = primary_result.value
        secondary = secondary_result.value or []
        for sub in (primary_result, secondary_result):
            if not sub.valid:
                valid = False
                issues.append(sub.issue or "invalid diagnosis")
    elif isinstance(raw, (list, tuple)):
        listed = _text_list(raw)
        items = listed.value or []
        primary = items[0] if items else None
        secondary = items[1:]
        valid, issues = listed.valid, ([listed.issue] if listed.issue else [])
    else:
        primary_result = _text(raw)
        primary, secondary = primary_result.value, []
        valid, issues = primary_result.valid, ([primary_result.issue] if primary_result.issue else [])
    if primary is None and not secondary:
        return Normalized(None, valid=valid, issue="; ".join(issues) or None)
    value = Diagnosis(primary=primary, secondary=secondary).model_dump()
    return Normalized(value, valid=valid, issue="; ".join(issues) or None)


# --- Dispatch --------------------------------------------------------------------------


@_never_raise
def normalize_value(kind: FieldKind, raw: Any, language: str | None = None) -> Normalized:
    """Normalize one raw field value according to its ``FieldKind``."""
    kind = FieldKind(kind)
    if kind is FieldKind.TEXT:
        return _text(raw)
    if kind is FieldKind.DATE:
        return normalize_date(raw, language)
    if kind is FieldKind.AMOUNT:
        return normalize_amount(raw)
    if kind is FieldKind.TEXT_LIST:
        return _text_list(raw)
    if kind is FieldKind.DIAGNOSIS:
        return _normalize_diagnosis(raw)
    if kind in ITEM_MODELS:
        return _normalize_items(kind, raw)
    return _invalid(f"unsupported field kind {kind!r}")


__all__ = [
    "Language",
    "Normalized",
    "detect_language",
    "normalize_amount",
    "normalize_date",
    "normalize_number",
    "normalize_value",
]

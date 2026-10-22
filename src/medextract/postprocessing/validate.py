"""Deterministic validation of extracted fields.

Runs after normalization, so values are already canonical: ISO date strings, floats for money,
lists of dicts for structured fields. The rules are plain code, never delegated to the model:

* dates: unparseable or impossible (``invalid_date``), ambiguous (``ambiguous_date``), in the
  future (``future_date``), discharge before admission (``date_order``);
* amounts: must be positive (``non_positive_amount``), unparseable (``invalid_amount``);
* receipt lines: quantity above zero (``invalid_quantity``), quantity x unit price close to the
  line total (``line_total_mismatch``), item totals close to the grand total (``total_mismatch``,
  ``incomplete_items``);
* lab flags: the reported flag must agree with the flag recomputed from the reference range
  (``lab_flag_mismatch``);
* required fields that are empty (``missing_required``).

Every rule works on whatever shape it receives and skips values of the wrong type, so odd model
output produces fewer checks, never an exception.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from collections.abc import Mapping
from typing import Any, Literal

from medextract.postprocessing.normalize import Normalized, normalize_amount
from medextract.schemas import FIELD_SPECS, FieldKind, FieldValue, ValidationIssue

LabFlagValue = Literal["normal", "high", "low"]

TOTAL_TOLERANCE_PERCENT = 5.0  # item totals vs grand_total
LINE_TOLERANCE_PERCENT = 1.0  # quantity x unit_price vs line total

_NUMBER = r"[-+]?\d+(?:[.,]\d+)?"
_NUMBER_FINDER = re.compile(r"\d+(?:[.,]\d+)?")
_BOUND_RE = re.compile(rf"\s*(<=|>=|≤|≥|<|>)\s*({_NUMBER})")
_RANGE_RE = re.compile(rf"\s*({_NUMBER})\s*(?:-|–|—|to)\s*({_NUMBER})")
_RESULT_RE = re.compile(r"\s*[-+]?\d[\d.,]*\s*")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

_REQUIRED_BY_TYPE: dict[str, tuple[str, ...]] = {
    "receipt": ("grand_total",),
    "discharge_summary": ("discharge_date",),
    "lab_report": ("tests",),
    "prescription": ("medications",),
}


def _today() -> _dt.date:
    """Current date. A separate function so tests can pin it."""
    return _dt.date.today()


def _number(value: Any) -> float | None:
    """A finite int or float (not bool), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _value_of(entry: Any) -> Any:
    """Field value from a FieldValue (or a plain {"value": ...} mapping)."""
    if isinstance(entry, FieldValue):
        return entry.value
    if isinstance(entry, Mapping) and "value" in entry:
        return entry["value"]
    return entry


def _parse_iso_date(value: Any) -> _dt.date | None:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str) and _ISO_DATE_RE.fullmatch(value):
        try:
            return _dt.date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _result_number(result: Any) -> float | None:
    """A numeric lab result, or None when the result is not purely numeric."""
    if isinstance(result, bool):
        return None
    if isinstance(result, (int, float)):
        return _number(result)
    if isinstance(result, str) and _RESULT_RE.fullmatch(result):
        return normalize_amount(result.strip()).value
    return None


def _parse_reference_range(reference_range: Any) -> tuple[float | None, float | None, bool, bool] | None:
    """Parse "3.5-5.0", "3.5 – 5.0", "< 200", "<=5", "> 40", "≥ 60" (optionally with units).

    Returns (low, high, low_inclusive, high_inclusive), with None for an open side, or None when
    the text is not a single range or bound.
    """
    if not isinstance(reference_range, str):
        return None
    text = reference_range.strip()
    numbers_in_text = len(_NUMBER_FINDER.findall(text))
    range_match = _RANGE_RE.match(text)
    if range_match and numbers_in_text == 2:
        low = normalize_amount(range_match.group(1)).value
        high = normalize_amount(range_match.group(2)).value
        if low is None or high is None or low > high:
            return None
        return low, high, True, True
    bound_match = _BOUND_RE.match(text)
    if bound_match and numbers_in_text == 1:
        operator, bound_text = bound_match.group(1), bound_match.group(2)
        bound = normalize_amount(bound_text).value
        if bound is None:
            return None
        if operator in ("<", "<=", "≤"):
            return None, bound, True, operator != "<"
        return bound, None, operator != ">", True
    return None


def compute_lab_flag(result: Any, reference_range: str | None) -> LabFlagValue | None:
    """Recompute a lab flag from the numeric result and its reference range.

    Returns "low" below the range, "high" above it, "normal" inside it (bounds inclusive, unless
    the range is written with a strict "<" or ">"). Returns None when the result is not numeric
    or the reference range cannot be parsed.
    """
    try:
        value = _result_number(result)
        if value is None:
            return None
        bounds = _parse_reference_range(reference_range)
        if bounds is None:
            return None
        low, high, low_inclusive, high_inclusive = bounds
        if low is not None and (value < low or (value == low and not low_inclusive)):
            return "low"
        if high is not None and (value > high or (value == high and not high_inclusive)):
            return "high"
        return "normal"
    except Exception:  # validation must never raise on odd model output
        return None


# --- Rules ------------------------------------------------------------------------------


def _check_date(name: str, value: Any, result: Any) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if result is not None and getattr(result, "valid", True) is False:
        detail = getattr(result, "issue", None)
        message = f"{name} could not be read as a valid date" + (f" ({detail})" if detail else "")
        return [ValidationIssue(field=name, code="invalid_date", message=message)]
    if result is not None and getattr(result, "ambiguous", False) is True:
        detail = getattr(result, "issue", None) or "day and month could be swapped"
        issues.append(
            ValidationIssue(
                field=name,
                code="ambiguous_date",
                message=f"{name} is ambiguous: {detail}",
                severity="warning",
            )
        )
    if value is None:
        return issues
    parsed = _parse_iso_date(value)
    if parsed is None:
        if isinstance(value, str):
            issues.append(
                ValidationIssue(
                    field=name,
                    code="invalid_date",
                    message=f"{name} value {value!r} is not a valid date (expected YYYY-MM-DD)",
                )
            )
        return issues  # other types: nothing to check
    if parsed > _today():
        issues.append(
            ValidationIssue(
                field=name,
                code="future_date",
                message=f"{name} {parsed.isoformat()} is after today ({_today().isoformat()})",
                severity="warning",
            )
        )
    return issues


def _check_date_order(values: Mapping[str, Any]) -> list[ValidationIssue]:
    admission = _parse_iso_date(values.get("admission_date"))
    discharge = _parse_iso_date(values.get("discharge_date"))
    if admission is None or discharge is None or discharge >= admission:
        return []
    return [
        ValidationIssue(
            field="discharge_date",
            code="date_order",
            message=(f"discharge_date {discharge.isoformat()} is before admission_date {admission.isoformat()}"),
        )
    ]


def _check_amount(name: str, value: Any, result: Any) -> list[ValidationIssue]:
    if result is not None and getattr(result, "valid", True) is False:
        detail = getattr(result, "issue", None)
        message = f"{name} is not a valid amount" + (f" ({detail})" if detail else "")
        return [ValidationIssue(field=name, code="invalid_amount", message=message)]
    number = _number(value)
    if number is not None and number <= 0:
        return [
            ValidationIssue(
                field=name,
                code="non_positive_amount",
                message=f"{name} must be a positive amount, got {number:g}",
            )
        ]
    return []


def _line_total(item: Mapping[str, Any]) -> float | None:
    """The line total as printed, or quantity x unit_price when the total is missing."""
    total = _number(item.get("total"))
    if total is not None:
        return total
    quantity, unit_price = _number(item.get("quantity")), _number(item.get("unit_price"))
    if quantity is not None and unit_price is not None:
        return quantity * unit_price
    return None


def _check_receipt_items(name: str, items: Any, result: Any) -> list[ValidationIssue]:
    if result is not None and getattr(result, "valid", True) is False:
        detail = getattr(result, "issue", None)
        message = f"{name} contains invalid values" + (f" ({detail})" if detail else "")
        return [ValidationIssue(field=name, code="invalid_value", message=message)]
    if not isinstance(items, list):
        return []

    issues: list[ValidationIssue] = []
    dict_items = [(index, item) for index, item in enumerate(items, start=1) if isinstance(item, Mapping)]
    for index, item in dict_items:
        quantity = _number(item.get("quantity"))
        unit_price = _number(item.get("unit_price"))
        total = _number(item.get("total"))
        if quantity is not None and quantity <= 0:
            issues.append(
                ValidationIssue(
                    field=name,
                    code="invalid_quantity",
                    message=f"item {index} quantity must be greater than 0, got {quantity:g}",
                )
            )
        for label, amount in (("unit_price", unit_price), ("total", total)):
            if amount is not None and amount <= 0:
                issues.append(
                    ValidationIssue(
                        field=name,
                        code="non_positive_amount",
                        message=f"item {index} {label} must be a positive amount, got {amount:g}",
                    )
                )
        if quantity is not None and unit_price is not None and total is not None:
            expected = quantity * unit_price
            base = abs(total) if total else abs(expected)
            if base and abs(expected - total) > base * LINE_TOLERANCE_PERCENT / 100:
                issues.append(
                    ValidationIssue(
                        field=name,
                        code="line_total_mismatch",
                        message=(
                            f"item {index}: quantity {quantity:g} x unit_price {unit_price:g} = "
                            f"{expected:g}, but line total is {total:g}"
                        ),
                        severity="warning",
                    )
                )
    return issues


def _check_item_total(name: str, items: Any, grand_total: Any) -> list[ValidationIssue]:
    """Compare the sum of item totals with grand_total."""
    if not isinstance(items, list):
        return []
    rows = [item for item in items if isinstance(item, Mapping)]
    line_totals = [_line_total(item) for item in rows]
    known = [total for total in line_totals if total is not None]
    if not known:
        return []  # nothing to compare against
    missing = len(line_totals) - len(known)
    issues: list[ValidationIssue] = []
    if missing:
        issues.append(
            ValidationIssue(
                field=name,
                code="incomplete_items",
                message=(
                    f"{missing} of {len(line_totals)} items have neither a total nor a quantity x "
                    "unit_price, so the item total check is incomplete"
                ),
                severity="warning",
            )
        )
        return issues
    grand = _number(grand_total)
    if grand is None or grand <= 0:
        return issues  # missing or non-positive grand_total is reported elsewhere
    items_sum = sum(known)
    difference_percent = abs(items_sum - grand) / grand * 100
    if difference_percent > TOTAL_TOLERANCE_PERCENT:
        issues.append(
            ValidationIssue(
                field="grand_total",
                code="total_mismatch",
                message=(
                    f"item totals sum to {items_sum:,.2f} but grand_total is {grand:,.2f} "
                    f"({difference_percent:.1f}% difference, limit {TOTAL_TOLERANCE_PERCENT:g}%)"
                ),
            )
        )
    return issues


def _check_lab_tests(name: str, tests: Any, result: Any) -> list[ValidationIssue]:
    if result is not None and getattr(result, "valid", True) is False:
        detail = getattr(result, "issue", None)
        message = f"{name} contains invalid values" + (f" ({detail})" if detail else "")
        return [ValidationIssue(field=name, code="invalid_value", message=message)]
    if not isinstance(tests, list):
        return []
    issues: list[ValidationIssue] = []
    for index, test in enumerate(tests, start=1):
        if not isinstance(test, Mapping):
            continue
        reported = test.get("flag")
        if not isinstance(reported, str):
            continue
        computed = compute_lab_flag(test.get("result"), test.get("reference_range"))
        if computed is None or reported.strip().lower() == computed:
            continue
        label = test.get("test_name") or f"test {index}"
        issues.append(
            ValidationIssue(
                field=name,
                code="lab_flag_mismatch",
                message=(
                    f"{label}: reported flag {reported!r} but result {test.get('result')!r} with "
                    f"reference range {test.get('reference_range')!r} is {computed}"
                ),
                severity="warning",
            )
        )
    return issues


def _check_invalid_other(name: str, result: Any) -> list[ValidationIssue]:
    if result is not None and getattr(result, "valid", True) is False:
        detail = getattr(result, "issue", None)
        message = f"{name} contains invalid values" + (f" ({detail})" if detail else "")
        return [ValidationIssue(field=name, code="invalid_value", message=message)]
    return []


def _check_missing(
    document_type: str, values: Mapping[str, Any], normalization: Mapping[str, Any]
) -> list[ValidationIssue]:
    required = ("patient_name", *_REQUIRED_BY_TYPE.get(document_type, ()))
    issues: list[ValidationIssue] = []
    for name in required:
        if name not in FIELD_SPECS[document_type]:
            continue
        value = values.get(name)
        if value is not None and not (isinstance(value, (list, str)) and not value):
            continue
        result = normalization.get(name)
        if result is not None and getattr(result, "valid", True) is False:
            continue  # already reported as invalid
        issues.append(
            ValidationIssue(
                field=name,
                code="missing_required",
                message=f"{name} is missing",
                severity="warning",
            )
        )
    return issues


def validate_fields(
    document_type: str,
    fields: dict[str, FieldValue],
    normalization: dict[str, Normalized] | None = None,
) -> list[ValidationIssue]:
    """Run every validation rule for ``document_type`` over normalized ``fields``.

    ``normalization`` maps field names to their ``Normalized`` outcome. It is used to report raw
    values that could not be parsed or were ambiguous. Unknown document types, and fields that
    are absent, are handled without raising.
    """
    if not isinstance(document_type, str) or document_type not in FIELD_SPECS:
        return []
    specs = FIELD_SPECS[document_type]
    field_map: Mapping[str, Any] = fields if isinstance(fields, Mapping) else {}
    norm: Mapping[str, Any] = normalization if isinstance(normalization, Mapping) else {}
    values = {name: _value_of(field_map.get(name)) for name in specs}
    issues: list[ValidationIssue] = []

    for name, kind in specs.items():
        result = norm.get(name)
        if kind is FieldKind.DATE:
            issues.extend(_check_date(name, values[name], result))
        elif kind is FieldKind.AMOUNT:
            issues.extend(_check_amount(name, values[name], result))
        elif kind is FieldKind.RECEIPT_ITEMS:
            issues.extend(_check_receipt_items(name, values[name], result))
        elif kind is FieldKind.LAB_TESTS:
            issues.extend(_check_lab_tests(name, values[name], result))
        else:
            issues.extend(_check_invalid_other(name, result))

    issues.extend(_check_date_order(values))
    if "items" in specs and "grand_total" in specs:
        issues.extend(_check_item_total("items", values["items"], values["grand_total"]))
    issues.extend(_check_missing(document_type, values, norm))
    return issues


__all__ = ["compute_lab_flag", "validate_fields"]

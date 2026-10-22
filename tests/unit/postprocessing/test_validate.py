import datetime as dt

import pytest

import medextract.postprocessing.validate as validate_module
from medextract.postprocessing.normalize import Normalized
from medextract.postprocessing.validate import compute_lab_flag, validate_fields
from medextract.schemas import FieldValue, ValidationIssue

TODAY = dt.date(2024, 6, 1)


@pytest.fixture(autouse=True)
def pin_today(monkeypatch):
    monkeypatch.setattr(validate_module, "_today", lambda: TODAY)


def fields(**values):
    return {name: FieldValue(value=value, confidence=0.9) for name, value in values.items()}


def codes(issues):
    return [issue.code for issue in issues]


def only(issues, code):
    matches = [issue for issue in issues if issue.code == code]
    assert len(matches) == 1, issues
    return matches[0]


def receipt(items, grand_total=None, **extra):
    base = {"hospital_name": "Bangkok Hospital", "patient_name": "John Doe", "items": items}
    base["grand_total"] = grand_total
    base.update(extra)
    return fields(**base)


# --- compute_lab_flag -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "reference", "expected"),
    [
        ("5.6", "3.5-5.0", "high"),
        ("3.2", "3.5-5.0", "low"),
        ("4.2", "3.5-5.0", "normal"),
        ("3.5", "3.5-5.0", "normal"),  # bounds are inclusive
        ("4", "3.5 – 5.0", "normal"),  # en dash
        ("4", "3.5 — 5.0 mmol/L", "normal"),  # trailing units
        ("0.6", "0 - 0.5 mg/dL", "high"),
        ("199", "< 200", "normal"),
        ("200", "< 200", "high"),  # strict bound
        ("200", "<=200", "normal"),
        ("201", "<=200", "high"),
        ("41", "> 40", "normal"),
        ("40", "> 40", "low"),
        ("61", "≥ 60", "normal"),
        ("59", "≥ 60", "low"),
        (7, "3-5", "high"),  # numbers, not strings
        ("1,000", "500-2,000", "normal"),
    ],
)
def test_compute_lab_flag_from_range(result, reference, expected):
    assert compute_lab_flag(result, reference) == expected


@pytest.mark.parametrize(
    ("result", "reference"),
    [
        ("positive", "3.5-5.0"),  # non-numeric result
        ("", "3.5-5.0"),
        (None, "3.5-5.0"),
        (True, "3.5-5.0"),
        ("5.0", None),  # no reference range
        ("5.0", "see table"),  # unparseable reference
        ("5.0", "M 13.5-17.5 F 12-15"),  # two ranges: ambiguous, not guessed
        ("5.0", "5-3"),  # inverted range
    ],
)
def test_compute_lab_flag_returns_none_when_not_computable(result, reference):
    assert compute_lab_flag(result, reference) is None


# --- Dates ------------------------------------------------------------------------------


def test_invalid_date_from_normalization_is_reported():
    normalization = {"date": Normalized(None, valid=False, issue="'31/02/2024': not a valid calendar date")}
    issues = validate_fields("receipt", fields(date=None, patient_name="A"), normalization)
    issue = only(issues, "invalid_date")
    assert issue.field == "date"
    assert issue.severity == "error"
    assert "31/02/2024" in issue.message


def test_valid_date_is_not_reported_as_invalid():
    normalization = {"date": Normalized("2024-03-15")}
    issues = validate_fields("receipt", fields(date="2024-03-15", patient_name="A"), normalization)
    assert "invalid_date" not in codes(issues)


def test_non_iso_date_value_is_invalid_date():
    issues = validate_fields("receipt", fields(date="15/03/2024", patient_name="A"))
    assert only(issues, "invalid_date").field == "date"


def test_ambiguous_date_is_a_warning():
    normalization = {"date": Normalized("2024-04-03", ambiguous=True, issue="ambiguous date '03/04/2024'")}
    issues = validate_fields("receipt", fields(date="2024-04-03", patient_name="A"), normalization)
    issue = only(issues, "ambiguous_date")
    assert issue.severity == "warning"


def test_future_date_is_a_warning():
    issues = validate_fields("receipt", fields(date="2030-01-01", patient_name="A"))
    issue = only(issues, "future_date")
    assert issue.severity == "warning"
    assert "2030-01-01" in issue.message


def test_past_date_is_not_future():
    issues = validate_fields("receipt", fields(date="2024-05-31", patient_name="A"))
    assert "future_date" not in codes(issues)


def test_discharge_before_admission_is_an_error():
    issues = validate_fields(
        "discharge_summary",
        fields(admission_date="2024-03-10", discharge_date="2024-03-05", patient_name="A"),
    )
    issue = only(issues, "date_order")
    assert issue.field == "discharge_date"
    assert issue.severity == "error"


def test_discharge_on_or_after_admission_is_fine():
    same_day = validate_fields(
        "discharge_summary",
        fields(admission_date="2024-03-10", discharge_date="2024-03-10", patient_name="A"),
    )
    later = validate_fields(
        "discharge_summary",
        fields(admission_date="2024-03-10", discharge_date="2024-03-12", patient_name="A"),
    )
    assert "date_order" not in codes(same_day)
    assert "date_order" not in codes(later)


# --- Amounts ----------------------------------------------------------------------------


def test_invalid_amount_from_normalization_is_reported():
    normalization = {"grand_total": Normalized(None, valid=False, issue="could not parse 'abc' as an amount")}
    issues = validate_fields("receipt", fields(grand_total=None, patient_name="A"), normalization)
    issue = only(issues, "invalid_amount")
    assert issue.field == "grand_total"


@pytest.mark.parametrize("amount", [0, -200.0])
def test_non_positive_grand_total_is_an_error(amount):
    issues = validate_fields("receipt", receipt([], grand_total=amount))
    issue = only(issues, "non_positive_amount")
    assert issue.field == "grand_total"
    assert issue.severity == "error"


def test_positive_grand_total_is_fine():
    issues = validate_fields("receipt", receipt([], grand_total=15750.0))
    assert "non_positive_amount" not in codes(issues)


def test_non_positive_item_price_is_an_error():
    items = [{"description": "Gauze", "quantity": 1, "unit_price": 0, "total": 0}]
    issues = validate_fields("receipt", receipt(items, grand_total=100.0))
    assert "non_positive_amount" in codes(issues)


def test_zero_quantity_is_invalid():
    items = [{"description": "Gauze", "quantity": 0, "unit_price": 50, "total": 0}]
    issues = validate_fields("receipt", receipt(items, grand_total=100.0))
    assert only(issues, "invalid_quantity").field == "items"


def test_positive_quantity_is_not_flagged():
    items = [{"description": "Gauze", "quantity": 2, "unit_price": 50, "total": 100}]
    issues = validate_fields("receipt", receipt(items, grand_total=100.0))
    assert "invalid_quantity" not in codes(issues)


# --- Receipt line totals ----------------------------------------------------------------


def test_line_total_mismatch_is_a_warning():
    items = [{"description": "Paracetamol", "quantity": 2, "unit_price": 100, "total": 210}]
    issues = validate_fields("receipt", receipt(items, grand_total=210.0))
    issue = only(issues, "line_total_mismatch")
    assert issue.severity == "warning"
    assert "200" in issue.message and "210" in issue.message


def test_line_total_within_one_percent_is_fine():
    items = [{"description": "Paracetamol", "quantity": 2, "unit_price": 100, "total": 201}]
    issues = validate_fields("receipt", receipt(items, grand_total=201.0))
    assert "line_total_mismatch" not in codes(issues)


def test_line_total_mismatch_skipped_when_a_part_is_missing():
    items = [{"description": "Paracetamol", "quantity": 2, "unit_price": None, "total": 999}]
    issues = validate_fields("receipt", receipt(items, grand_total=999.0))
    assert "line_total_mismatch" not in codes(issues)


# --- Receipt grand total ----------------------------------------------------------------


def _items_totalling(total):
    return [{"description": "Item", "quantity": 1, "unit_price": total, "total": total}]


def test_total_mismatch_message_has_both_numbers_and_percentage():
    issues = validate_fields("receipt", receipt(_items_totalling(100.0), grand_total=150.0))
    issue = only(issues, "total_mismatch")
    assert issue.field == "grand_total"
    assert issue.severity == "error"
    assert "100.00" in issue.message and "150.00" in issue.message and "%" in issue.message


def test_total_mismatch_at_4_9_percent_is_ok():
    issues = validate_fields("receipt", receipt(_items_totalling(104.9), grand_total=100.0))
    assert "total_mismatch" not in codes(issues)


def test_total_mismatch_at_5_1_percent_is_flagged():
    issues = validate_fields("receipt", receipt(_items_totalling(105.1), grand_total=100.0))
    assert "total_mismatch" in codes(issues)


def test_total_mismatch_uses_quantity_times_unit_price_when_total_missing():
    items = [
        {"description": "A", "quantity": 2, "unit_price": 50, "total": None},
        {"description": "B", "quantity": 1, "unit_price": 30, "total": 30},
    ]
    assert "total_mismatch" not in codes(validate_fields("receipt", receipt(items, grand_total=130.0)))
    assert "total_mismatch" in codes(validate_fields("receipt", receipt(items, grand_total=200.0)))


def test_incomplete_items_are_reported_and_total_check_skipped():
    items = [
        {"description": "A", "quantity": None, "unit_price": None, "total": 100},
        {"description": "B", "quantity": None, "unit_price": None, "total": None},
    ]
    issues = validate_fields("receipt", receipt(items, grand_total=500.0))
    assert only(issues, "incomplete_items").severity == "warning"
    assert "total_mismatch" not in codes(issues)


def test_total_check_needs_grand_total_and_an_item_total():
    no_grand = validate_fields("receipt", receipt(_items_totalling(100.0), grand_total=None))
    no_items = validate_fields("receipt", receipt([], grand_total=100.0))
    assert "total_mismatch" not in codes(no_grand)
    assert "total_mismatch" not in codes(no_items)


def test_missing_grand_total_is_required():
    issues = validate_fields("receipt", receipt(_items_totalling(100.0), grand_total=None))
    issue = only(issues, "missing_required")
    assert issue.field == "grand_total"
    assert issue.severity == "warning"


# --- Lab flags --------------------------------------------------------------------------


def _lab_tests(*tests):
    return fields(lab_name="Central Lab", patient_name="A", date="2024-03-15", tests=list(tests))


def test_lab_flag_mismatch_is_a_warning():
    tests = [
        {
            "test_name": "WBC",
            "result": "5.6",
            "unit": "10^9/L",
            "reference_range": "3.5-5.0",
            "flag": "normal",
        }
    ]
    issues = validate_fields("lab_report", _lab_tests(*tests))
    issue = only(issues, "lab_flag_mismatch")
    assert issue.severity == "warning"
    assert "WBC" in issue.message and "high" in issue.message


def test_consistent_lab_flag_is_not_reported():
    tests = [{"test_name": "WBC", "result": "5.6", "reference_range": "3.5-5.0", "flag": "high"}]
    assert "lab_flag_mismatch" not in codes(validate_fields("lab_report", _lab_tests(*tests)))


def test_lab_flag_is_not_checked_without_reported_flag_or_range():
    tests = [
        {"test_name": "A", "result": "5.6", "reference_range": "3.5-5.0", "flag": None},
        {"test_name": "B", "result": "5.6", "reference_range": None, "flag": "normal"},
        {"test_name": "C", "result": "positive", "reference_range": "3.5-5.0", "flag": "high"},
    ]
    assert "lab_flag_mismatch" not in codes(validate_fields("lab_report", _lab_tests(*tests)))


def test_lab_tests_required():
    issues = validate_fields("lab_report", fields(lab_name="X", patient_name="A", date=None, tests=None))
    assert only(issues, "missing_required").field == "tests"


# --- Required fields --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("document_type", "values", "missing"),
    [
        ("receipt", {"grand_total": 100.0}, "patient_name"),
        ("discharge_summary", {"patient_name": "A", "discharge_date": None}, "discharge_date"),
        ("lab_report", {"patient_name": "A", "tests": []}, "tests"),
        ("prescription", {"patient_name": "A", "medications": None}, "medications"),
    ],
)
def test_required_fields_reported_when_empty(document_type, values, missing):
    issues = validate_fields(document_type, fields(**values))
    assert missing in {issue.field for issue in issues if issue.code == "missing_required"}


def test_patient_name_required_for_every_type():
    for document_type in ("receipt", "discharge_summary", "lab_report", "prescription"):
        issues = validate_fields(document_type, fields(patient_name=None))
        assert "patient_name" in {i.field for i in issues if i.code == "missing_required"}


def test_required_field_not_reported_when_invalid_already():
    normalization = {"discharge_date": Normalized(None, valid=False, issue="impossible")}
    issues = validate_fields("discharge_summary", fields(patient_name="A", discharge_date=None), normalization)
    assert "missing_required" not in {i.field for i in issues if i.code == "missing_required"}
    assert "invalid_date" in codes(issues)


def test_complete_prescription_has_no_issues():
    medications = [{"name": "Amoxicillin", "dosage": "500 mg", "frequency": "TID", "quantity": 21.0}]
    issues = validate_fields(
        "prescription",
        fields(doctor_name="Dr X", patient_name="A", date="2024-03-15", medications=medications),
    )
    assert issues == []


# --- Structured values from normalization -----------------------------------------------


def test_invalid_structured_value_is_reported_as_invalid_value():
    normalization = {"items": Normalized([], valid=False, issue="item 1 unit_price: could not parse 'abc'")}
    issues = validate_fields("receipt", receipt([], grand_total=100.0), normalization)
    issue = only(issues, "invalid_value")
    assert issue.field == "items"
    assert "unit_price" in issue.message


# --- Robustness -------------------------------------------------------------------------


def test_unknown_document_type_gives_no_issues():
    assert validate_fields("unknown", fields(patient_name=None)) == []
    assert validate_fields(None, {}) == []


def test_wrong_types_do_not_raise():
    odd = fields(
        date=12345,
        grand_total="a lot",
        items="not a list",
        patient_name=["x"],
    )
    issues = validate_fields("receipt", odd)
    assert all(isinstance(issue, ValidationIssue) for issue in issues)

    odd_items = receipt([None, "junk", {"quantity": "two", "unit_price": [1], "total": {}}], grand_total=True)
    assert all(isinstance(issue, ValidationIssue) for issue in validate_fields("receipt", odd_items))

    assert validate_fields("lab_report", _lab_tests("junk", {"flag": 3, "result": None})) is not None


def test_missing_fields_and_odd_normalization_do_not_raise():
    assert isinstance(validate_fields("receipt", {}), list)
    assert isinstance(validate_fields("receipt", fields(patient_name="A"), {"date": "not a result"}), list)
    assert isinstance(validate_fields("receipt", fields(patient_name="A"), None), list)

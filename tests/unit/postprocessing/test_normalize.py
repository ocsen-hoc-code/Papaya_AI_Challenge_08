import datetime as dt

import pytest

from medextract.postprocessing.normalize import (
    Normalized,
    detect_language,
    normalize_amount,
    normalize_date,
    normalize_number,
    normalize_value,
)
from medextract.schemas import FieldKind

# --- Language detection ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ใบเสร็จรับเงิน โรงพยาบาลกรุงเทพ", "th"),
        ("Bệnh viện Chợ Rẫy - Hóa đơn", "vi"),
        ("Bangkok Hospital receipt", "en"),
        ("Nguyễn", "vi"),
        ("12345 --- 6789", None),
        ("", None),
        (None, None),
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text) == expected


# --- Dates -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "2024-03-15",
        "15/03/2024",
        "15-03-2024",
        "15.03.2024",
        "15/03/24",
        "15 Mar 2024",
        "March 15, 2024",
        "15th March 2024",
        "Mar 15 24",
        "2024-03-15T10:30:00",
        "2024-03-15 10:30",
        "15/03/2024 10:30 AM",
    ],
)
def test_iso_numeric_and_english_dates(raw):
    result = normalize_date(raw)
    assert result.value == "2024-03-15"
    assert result.valid and not result.ambiguous


def test_datetime_objects_are_accepted():
    assert normalize_date(dt.datetime(2024, 3, 15, 9, 0)).value == "2024-03-15"
    assert normalize_date(dt.date(2024, 3, 15)).value == "2024-03-15"


@pytest.mark.parametrize(
    "raw",
    [
        "ngày 15 tháng 03 năm 2024",
        "15 tháng 3, 2024",
        "15/03/2024",
    ],
)
def test_vietnamese_dates(raw):
    result = normalize_date(raw, "vi")
    assert result.value == "2024-03-15"
    assert result.valid and not result.ambiguous


@pytest.mark.parametrize(
    "raw",
    [
        "15 มีนาคม 2567",
        "15 มี.ค. 67",
        "๑๕ มีนาคม ๒๕๖๗",
        "วันที่ 15 มี.ค. 67",
        "15 มีนาคม พ.ศ. 2567",
        "15/03/2567",
    ],
)
def test_thai_dates_with_buddhist_era_and_thai_digits(raw):
    result = normalize_date(raw)
    assert result.value == "2024-03-15"
    assert result.valid and not result.ambiguous


def test_thai_month_name_with_two_digit_be_year_without_language_hint():
    assert normalize_date("15 มี.ค. 67").value == "2024-03-15"


def test_two_digit_year_outside_thai_context_is_ad():
    assert normalize_date("15 Mar 24").value == "2024-03-15"
    assert normalize_date("15/03/24", "en").value == "2024-03-15"


def test_buddhist_era_year_in_iso_style_is_converted():
    assert normalize_date("2567-03-15").value == "2024-03-15"


def test_ambiguous_numeric_date_without_language_is_flagged_and_day_first():
    result = normalize_date("03/04/2024")
    assert result.value == "2024-04-03"
    assert result.valid
    assert result.ambiguous
    assert result.issue and "03/04/2024" in result.issue


def test_ambiguous_numeric_date_in_english_is_flagged():
    result = normalize_date("03/04/2024", "en")
    assert result.value == "2024-04-03"
    assert result.ambiguous


def test_vietnamese_and_thai_numeric_dates_are_day_first_without_flag():
    assert normalize_date("03/04/2024", "vi") == Normalized("2024-04-03")
    assert normalize_date("03/04/2024", "th") == Normalized("2024-04-03")


def test_unique_reading_is_used_and_not_ambiguous():
    # 15 cannot be a month, so only the month-first reading is a real date.
    result = normalize_date("03/15/2024")
    assert result.value == "2024-03-15"
    assert not result.ambiguous
    # Same day and month: both readings agree.
    assert normalize_date("05/05/2024").ambiguous is False


@pytest.mark.parametrize("raw", ["31/02/2024", "30 Feb 2024", "2023-02-29", "32/01/2024"])
def test_impossible_dates_are_invalid(raw):
    result = normalize_date(raw)
    assert result.valid is False
    assert result.value is None
    assert result.issue


@pytest.mark.parametrize("raw", ["not a date", "abc", "15/03", "March 2024", "10:30", "x/y/z"])
def test_unparseable_dates_return_none_without_raising(raw):
    result = normalize_date(raw)
    assert result.value is None
    assert result.valid is False


@pytest.mark.parametrize("raw", [None, "", "   ", "N/A", "-"])
def test_missing_dates_are_valid_none(raw):
    result = normalize_date(raw)
    assert result.value is None
    assert result.valid is True


def test_non_string_garbage_does_not_raise():
    for raw in (["15/03/2024"], {"d": 1}, True, object()):
        result = normalize_date(raw)
        assert result.value is None
        assert result.valid is False


# --- Amounts ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.500.000", 1_500_000.0),
        ("1.500.000 ₫", 1_500_000.0),
        ("1,500,000 VND", 1_500_000.0),
        ("1,500.50", 1500.50),
        ("1.500,50", 1500.50),
        ("฿1,250.00", 1250.0),
        ("1,250 บาท", 1250.0),
        ("$15,750.00", 15750.0),
        ("15750", 15750.0),
        ("USD 20", 20.0),
        ("1,5", 1.5),
        ("1.50", 1.5),
        ("1,500", 1500.0),
        ("0.500", 0.5),
        ("๑,๒๕๐.๕๐", 1250.5),
        ("15750.00 THB", 15750.0),
    ],
)
def test_amount_formats(raw, expected):
    result = normalize_amount(raw)
    assert result.valid
    assert result.value == pytest.approx(expected)


def test_amount_numbers_pass_through():
    assert normalize_amount(175).value == 175.0
    assert normalize_amount(12.5).value == 12.5


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("-200", -200.0), ("(200)", -200.0), ("200-", -200.0), ("-1.500.000 ₫", -1_500_000.0)],
)
def test_negative_amounts_keep_their_sign(raw, expected):
    result = normalize_amount(raw)
    assert result.valid
    assert result.value == expected


@pytest.mark.parametrize("raw", ["abc", "1.50.00", "12k", "₫", "1,2,3.4.5", True, float("nan")])
def test_unparseable_amounts_are_invalid(raw):
    result = normalize_amount(raw)
    assert result.value is None
    assert result.valid is False


def test_missing_amount_is_valid_none():
    assert normalize_amount(None) == Normalized(None)
    assert normalize_amount("N/A") == Normalized(None)


# --- Numbers (quantities) ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2", 2.0),
        ("2 tablets", 2.0),
        ("2 viên", 2.0),
        ("๒ ครั้ง", 2.0),
        ("x2", 2.0),
        ("1,5 ml", 1.5),
        ("1/2 tablet", 0.5),
        (3, 3.0),
        (2.5, 2.5),
    ],
)
def test_number_formats(raw, expected):
    result = normalize_number(raw)
    assert result.valid
    assert result.value == pytest.approx(expected)


def test_number_without_digits_is_invalid_and_missing_is_none():
    assert normalize_number("two").valid is False
    assert normalize_number("two").value is None
    assert normalize_number(None) == Normalized(None)
    assert normalize_number("") == Normalized(None)


# --- normalize_value dispatch ----------------------------------------------------------


def test_text_is_kept_verbatim_and_stripped():
    assert normalize_value(FieldKind.TEXT, "  Bangkok Hospital  ").value == "Bangkok Hospital"
    assert normalize_value(FieldKind.TEXT, "โรงพยาบาลกรุงเทพ").value == "โรงพยาบาลกรุงเทพ"
    assert normalize_value(FieldKind.TEXT, "   ").value is None
    assert normalize_value(FieldKind.TEXT, None).value is None


def test_date_and_amount_kinds_dispatch():
    assert normalize_value(FieldKind.DATE, "15/03/2024").value == "2024-03-15"
    assert normalize_value(FieldKind.AMOUNT, "1.500.000 ₫").value == 1_500_000.0


def test_date_dispatch_passes_language_through():
    assert normalize_value(FieldKind.DATE, "03/04/2024", "vi").ambiguous is False
    assert normalize_value(FieldKind.DATE, "03/04/2024").ambiguous is True


def test_text_list_from_list_and_delimited_string():
    assert normalize_value(FieldKind.TEXT_LIST, ["X-ray", " ", "CT scan"]).value == ["X-ray", "CT scan"]
    assert normalize_value(FieldKind.TEXT_LIST, "- X-ray\n- CT scan").value == ["X-ray", "CT scan"]
    assert normalize_value(FieldKind.TEXT_LIST, "1. Appendectomy; 2. Drainage").value == [
        "Appendectomy",
        "Drainage",
    ]
    assert normalize_value(FieldKind.TEXT_LIST, []).value == []
    assert normalize_value(FieldKind.TEXT_LIST, None).value is None
    assert normalize_value(FieldKind.TEXT_LIST, "").value is None


def test_diagnosis_from_dict_string_and_missing():
    result = normalize_value(FieldKind.DIAGNOSIS, {"primary": "Pneumonia", "secondary": ["HTN", "DM"]})
    assert result.value == {"primary": "Pneumonia", "secondary": ["HTN", "DM"]}
    assert normalize_value(FieldKind.DIAGNOSIS, "Pneumonia").value == {
        "primary": "Pneumonia",
        "secondary": [],
    }
    assert normalize_value(FieldKind.DIAGNOSIS, {"primary": None, "secondary": []}).value is None
    assert normalize_value(FieldKind.DIAGNOSIS, None).value is None


def test_receipt_items_normalize_numeric_subfields():
    raw = [
        {"description": "Paracetamol", "quantity": "2 tablets", "unit_price": "1.500", "total": "3.000 ₫"},
        {"description": "Gauze", "quantity": 1, "unit_price": 250.0, "total": "250"},
    ]
    result = normalize_value(FieldKind.RECEIPT_ITEMS, raw)
    assert result.valid
    assert result.value == [
        {"description": "Paracetamol", "quantity": 2.0, "unit_price": 1500.0, "total": 3000.0},
        {"description": "Gauze", "quantity": 1.0, "unit_price": 250.0, "total": 250.0},
    ]


def test_receipt_item_with_bad_subvalue_is_invalid_but_keeps_other_values():
    raw = [
        {"description": "Paracetamol", "quantity": 2, "unit_price": "abc", "total": "3.000"},
    ]
    result = normalize_value(FieldKind.RECEIPT_ITEMS, raw)
    assert result.valid is False
    assert result.issue and "item 0" in result.issue and "unit_price" in result.issue
    assert result.value == [{"description": "Paracetamol", "quantity": 2.0, "unit_price": None, "total": 3000.0}]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("7.8", 7.8),
        ("5,6", 5.6),  # Vietnamese decimal comma
        ("๑๒.๕", 12.5),  # Thai digits
        ("-1.2", -1.2),
        (95, 95.0),
        (4.1, 4.1),
        ("< 0.5", "< 0.5"),  # qualifier: kept as printed
        ("7.8 mmol/L", "7.8 mmol/L"),
        ("Positive", "Positive"),
        ("Âm tính", "Âm tính"),
        (None, None),
    ],
)
def test_lab_result_numbers_become_floats(raw, expected):
    result = normalize_value(FieldKind.LAB_TESTS, [{"test_name": "X", "result": raw}])
    value = result.value[0]["result"]
    assert value == expected
    if isinstance(expected, float):
        assert isinstance(value, float)


def test_lab_tests_flags_and_numeric_results():
    raw = [
        {"test_name": "Hemoglobin", "result": 11.2, "unit": "g/dL", "reference_range": "12-16", "flag": "L"},
        {"test_name": "WBC", "result": "12.5", "unit": "10^9/L", "reference_range": "4-10", "flag": "↑"},
        {
            "test_name": "Glucose",
            "result": "95",
            "unit": "mg/dL",
            "reference_range": "70-99",
            "flag": "Normal",
        },
    ]
    result = normalize_value(FieldKind.LAB_TESTS, raw)
    assert result.valid
    assert [test["result"] for test in result.value] == [11.2, 12.5, 95.0]
    assert [test["flag"] for test in result.value] == ["low", "high", "normal"]


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("H", "high"),
        ("L", "low"),
        ("N", "normal"),
        ("cao", "high"),
        ("thấp", "low"),
        ("Bình thường", "normal"),
        ("สูง", "high"),
        ("ต่ำ", "low"),
        ("ปกติ", "normal"),
        ("↓", "low"),
        ("*H", "high"),
        ("", None),
    ],
)
def test_lab_flag_mapping(flag, expected):
    result = normalize_value(FieldKind.LAB_TESTS, [{"test_name": "X", "flag": flag}])
    assert result.valid
    assert result.value[0]["flag"] == expected


def test_unknown_lab_flag_is_invalid():
    result = normalize_value(FieldKind.LAB_TESTS, [{"test_name": "X", "flag": "critical-ish"}])
    assert result.valid is False
    assert result.value[0]["flag"] is None


def test_medications_normalize_quantity():
    raw = [
        {
            "name": "Amoxicillin",
            "dosage": "500 mg",
            "frequency": "TID",
            "duration": "7 days",
            "quantity": "21 capsules",
        }
    ]
    result = normalize_value(FieldKind.MEDICATIONS, raw)
    assert result.valid
    assert result.value == [
        {
            "name": "Amoxicillin",
            "dosage": "500 mg",
            "frequency": "TID",
            "duration": "7 days",
            "quantity": 21.0,
        }
    ]


def test_structured_list_skips_junk_entries_and_flags_them():
    result = normalize_value(FieldKind.MEDICATIONS, [{"name": "Aspirin"}, "junk", None])
    assert result.valid is False
    assert result.value[0]["name"] == "Aspirin"
    assert len(result.value) == 1


def test_structured_missing_and_wrong_shape():
    assert normalize_value(FieldKind.LAB_TESTS, None).value is None
    assert normalize_value(FieldKind.LAB_TESTS, []).value == []
    assert normalize_value(FieldKind.RECEIPT_ITEMS, "not a list").valid is False
    single = normalize_value(FieldKind.RECEIPT_ITEMS, {"description": "Gauze", "total": "250"})
    assert single.value == [{"description": "Gauze", "quantity": None, "unit_price": None, "total": 250.0}]


def test_normalize_value_never_raises_on_odd_input():
    odd_inputs = [object(), {"a": object()}, [object()], float("inf"), True, b"bytes"]
    for kind in FieldKind:
        for raw in odd_inputs:
            result = normalize_value(kind, raw)
            assert isinstance(result, Normalized)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("( นพ.วีระพงษ์ แสงทอง )", "นพ.วีระพงษ์ แสงทอง"),
        ('"Dr. Jane Roe"', "Dr. Jane Roe"),
        ("[BS. Trần An]", "BS. Trần An"),
        ("Diabetes (E11)", "Diabetes (E11)"),
        ("(E11) Diabetes (type 2)", "(E11) Diabetes (type 2)"),
    ],
)
def test_text_unwraps_enclosing_brackets_only(raw, expected):
    assert normalize_value(FieldKind.TEXT, raw).value == expected

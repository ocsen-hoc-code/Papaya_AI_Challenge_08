from medextract.postprocessing.grounding import Grounding, ground_field, text_layer_usable
from medextract.schemas import FieldKind

RECEIPT_TEXT = """Bangkok Hospital
Patient: John Doe
Date: 14 Mar 2024
Appendectomy consultation 1 x 1,500.00
Room fee 2 x 750.00 = 1,500.00
Grand total ๑๕,๗๕๐.๐๐ THB
Chẩn đoán: viêm ruột thừa"""

VI_TEXT = "Bệnh viện Chợ Rẫy\nTổng cộng: 1.500.000 VND\nNgày 14/03/2024"


def test_fabricated_text_value_absent_from_document_is_not_found():
    g = ground_field(FieldKind.TEXT, "Siriraj Dental Clinic", None, RECEIPT_TEXT)
    assert g.status == "not_found"
    assert g.score < 0.3


def test_fabricated_amount_absent_from_document_is_not_found():
    g = ground_field(FieldKind.AMOUNT, 9999.0, None, RECEIPT_TEXT)
    assert g.status == "not_found"
    assert g.score < 0.3


def test_exact_substring_hit():
    g = ground_field(FieldKind.TEXT, "Bangkok Hospital", None, RECEIPT_TEXT)
    assert g.status == "exact"
    assert g.score == 1.0


def test_fuzzy_hit_for_small_typo():
    g = ground_field(FieldKind.TEXT, "Bangkok Hosptial", None, RECEIPT_TEXT)
    assert g.status == "fuzzy"
    assert 0.9 <= g.score < 1.0


def test_text_match_ignores_case_and_whitespace():
    g = ground_field(FieldKind.TEXT, "  bangkok   HOSPITAL ", None, RECEIPT_TEXT)
    assert g.status == "exact"


def test_thai_digits_in_text_match_ascii_amount():
    g = ground_field(FieldKind.AMOUNT, 15750.0, None, RECEIPT_TEXT)
    assert g.status == "exact"
    assert g.score == 1.0


def test_vietnamese_thousands_separators_match_amount():
    assert ground_field(FieldKind.AMOUNT, 1500000.0, None, VI_TEXT).status == "exact"


def test_english_and_european_number_formats_match_amount():
    assert ground_field(FieldKind.AMOUNT, 1500.0, None, RECEIPT_TEXT).status == "exact"
    assert ground_field(FieldKind.AMOUNT, 1500.5, None, "Total 1.500,50 EUR").status == "exact"
    assert ground_field(FieldKind.AMOUNT, 1500.5, None, "Total 1,500.50 USD").status == "exact"


def test_thousands_separator_is_not_read_as_decimal():
    # "1.500.000" must not ground an amount of 1500.0.
    g = ground_field(FieldKind.AMOUNT, 1500.0, None, VI_TEXT)
    assert g.status == "not_found"


def test_date_grounded_through_quoted_evidence():
    g = ground_field(FieldKind.DATE, "2024-03-14", "14 Mar 2024", RECEIPT_TEXT)
    assert g.status == "exact"
    assert g.score >= 0.9


def test_date_with_wrong_day_in_evidence_is_only_partial():
    g = ground_field(FieldKind.DATE, "2024-03-15", "14 Mar 2024", RECEIPT_TEXT)
    assert g.status == "partial"
    assert g.score < 0.9


def test_buddhist_era_year_counts_for_date():
    text = "วันที่ 14/03/2567 โรงพยาบาล"
    g = ground_field(FieldKind.DATE, "2024-03-14", None, text)
    assert g.status == "partial"
    assert g.score >= 0.5


def test_date_absent_from_document_is_not_found():
    g = ground_field(FieldKind.DATE, "2019-07-02", None, RECEIPT_TEXT)
    assert g.status == "not_found"
    assert g.score < 0.3


def test_quoted_evidence_rescues_value_that_is_paraphrased():
    g = ground_field(FieldKind.TEXT, "Appendectomy surgery", "Appendectomy consultation", RECEIPT_TEXT)
    assert g.status == "partial"


def test_value_that_is_a_substring_of_the_text_is_exact():
    assert ground_field(FieldKind.TEXT, "Appendectomy", None, RECEIPT_TEXT).status == "exact"


def test_text_layer_usable_detects_garbled_text():
    assert not text_layer_usable(None)
    assert not text_layer_usable("")
    assert not text_layer_usable("   \n ")
    assert not text_layer_usable("(cid:12)(cid:45)(cid:3)(cid:88)" * 4)
    assert not text_layer_usable("�" * 10 + "ab")
    assert not text_layer_usable("|||---***!!!###$$$%%%^^^&&&")
    assert text_layer_usable(RECEIPT_TEXT)
    assert text_layer_usable(VI_TEXT)
    # Thai text uses many combining marks; they must not count as garbage.
    assert text_layer_usable("ใบเสร็จรับเงิน โรงพยาบาลกรุงเทพ ยอดรวม ๑๕,๗๕๐.๐๐ บาท")


def test_garbled_text_layer_falls_back_to_image_path():
    g = ground_field(FieldKind.TEXT, "Bangkok Hospital", "Bangkok Hospital", "(cid:1)" * 20)
    assert g.status == "no_reference"
    assert g.score == 0.6


def test_image_without_evidence_scores_low_but_not_zero():
    g = ground_field(FieldKind.TEXT, "Bangkok Hospital", None, None)
    assert g.status == "no_reference"
    assert g.score == 0.4


def test_image_with_consistent_evidence_scores_higher():
    g = ground_field(FieldKind.TEXT, "Bangkok Hospital", "Bangkok Hospital", None)
    assert g.status == "no_reference"
    assert g.score == 0.6


def test_image_with_inconsistent_evidence_scores_lowest():
    g = ground_field(FieldKind.AMOUNT, 15750.0, "Grand total 9,999.00", None)
    assert g.status == "no_reference"
    assert g.score == 0.2


def test_image_amount_and_date_checked_against_evidence():
    assert ground_field(FieldKind.AMOUNT, 15750.0, "Grand total 15,750.00", None).score == 0.6
    assert ground_field(FieldKind.DATE, "2024-03-14", "14 Mar 2024", None).score == 0.6
    assert ground_field(FieldKind.DATE, "2024-03-14", "02 Jul 2019", None).score == 0.2


def test_null_values_have_no_reference():
    for value in (None, "", "  ", [], {}, [None]):
        g = ground_field(FieldKind.TEXT, value, None, RECEIPT_TEXT)
        assert g.status == "no_reference"
        assert g.score == 0.0


def test_text_list_partial_when_some_items_fail():
    value = ["Appendectomy consultation", "Heart transplant"]
    g = ground_field(FieldKind.TEXT_LIST, value, None, RECEIPT_TEXT)
    assert g.status == "partial"
    assert 0.3 < g.score < 0.9


def test_text_list_all_found_is_exact():
    g = ground_field(FieldKind.TEXT_LIST, ["Bangkok Hospital", "John Doe"], None, RECEIPT_TEXT)
    assert g.status == "exact"
    assert g.score == 1.0


def test_text_list_all_missing_is_not_found():
    g = ground_field(FieldKind.TEXT_LIST, ["Heart transplant", "Cardiac bypass"], None, RECEIPT_TEXT)
    assert g.status == "not_found"


def test_receipt_items_grounds_description_and_numbers():
    items = [
        {"description": "Appendectomy consultation", "quantity": 1, "total": 1500.0},
        {"description": "Ghost item", "quantity": 1, "total": 99.0},
    ]
    g = ground_field(FieldKind.RECEIPT_ITEMS, items, None, RECEIPT_TEXT)
    assert g.status == "partial"
    assert "ghost item" in (g.detail or "")


def test_receipt_items_fully_supported_is_high():
    items = [{"description": "Room fee", "quantity": 2, "unit_price": 750.0, "total": 1500.0}]
    g = ground_field(FieldKind.RECEIPT_ITEMS, items, None, RECEIPT_TEXT)
    assert g.status in ("exact", "fuzzy")
    assert g.score >= 0.9


def test_lab_test_numeric_result_is_grounded():
    text = "Glucose 5.4 mmol/L\nHemoglobin 13.2 g/dL"
    tests = [{"test_name": "Glucose", "result": 5.4}, {"test_name": "Hemoglobin", "result": 99.9}]
    g = ground_field(FieldKind.LAB_TESTS, tests, None, text)
    assert g.status == "partial"


def test_diagnosis_grounds_primary_and_secondary():
    value = {"primary": "Acute appendicitis", "secondary": ["Hypertension"]}
    g = ground_field(FieldKind.DIAGNOSIS, value, None, "Diagnosis: Acute appendicitis.\nPlan: rest")
    assert g.status == "partial"
    assert g.score < 1.0


def test_long_text_value_is_checked_sentence_by_sentence():
    instructions = (
        "Rest for two weeks and avoid heavy lifting. Keep the wound clean and dry at all times. "
        "Return to the emergency department if fever above 38 degrees develops."
    )
    text = instructions.replace("Keep the wound clean and dry at all times.", "Wound care daily.")
    g = ground_field(FieldKind.TEXT, instructions, None, text)
    assert g.status == "partial"
    assert g.score > 0.3


def test_grounding_detail_defaults_to_none():
    assert Grounding("exact", 1.0).detail is None

from pathlib import Path

import pytest

from medextract.extraction.prompts import (
    MAX_TEXT_HINT_CHARS,
    SYSTEM_PROMPT,
    build_classify_prompt,
    build_extract_prompt,
)
from medextract.schemas import DOCUMENT_TYPES, FIELD_SPECS, ITEM_MODELS, FieldKind

DOCUMENT_TESTS_DIR = Path(__file__).resolve().parents[3] / "document-tests"


def _all_field_names() -> set[str]:
    return {name for fields in FIELD_SPECS.values() for name in fields}


@pytest.mark.parametrize("document_type", DOCUMENT_TYPES)
def test_extract_prompt_mentions_every_field_of_the_type(document_type):
    prompt = build_extract_prompt(document_type)
    for name in FIELD_SPECS[document_type]:
        assert f'"{name}"' in prompt, name


@pytest.mark.parametrize("document_type", DOCUMENT_TYPES)
def test_extract_prompt_only_mentions_its_own_type(document_type):
    prompt = build_extract_prompt(document_type)
    for other in DOCUMENT_TYPES:
        if other != document_type:
            assert other not in prompt
    own = set(FIELD_SPECS[document_type])
    for name in _all_field_names() - own:
        assert f'"{name}"' not in prompt, name


def test_extract_prompt_snapshot_for_receipt_items():
    prompt = build_extract_prompt("receipt")
    line = next(line for line in prompt.splitlines() if line.startswith('- "items"'))
    assert "array of objects, one per printed line item" in line
    assert "Keys: description, quantity, unit_price, total." in line
    assert '"grand_total": amount as a string' in prompt
    assert prompt.count("\n- ") == len(FIELD_SPECS["receipt"])


def test_extract_prompt_lists_item_keys_and_flags_for_lab_tests():
    prompt = build_extract_prompt("lab_report")
    keys = ", ".join(ITEM_MODELS[FieldKind.LAB_TESTS].model_fields)
    assert f"Keys: {keys}." in prompt
    assert "one of normal, high, low" in prompt


def test_extract_prompt_describes_the_diagnosis_shape():
    prompt = build_extract_prompt("discharge_summary")
    assert "Keys: primary, secondary." in prompt


def test_extract_prompt_contains_the_response_shape():
    prompt = build_extract_prompt("prescription")
    assert '"fields": {"<field name>": {"value": ...' in prompt
    assert "Return ONLY a JSON object" in prompt


def test_extract_prompt_language_hint():
    assert "mainly in Thai" in build_extract_prompt("receipt", language="th")
    assert "mainly in" not in build_extract_prompt("receipt")
    assert "mainly in" not in build_extract_prompt("receipt", language="other")


def test_extract_prompt_rejects_unknown_document_type():
    with pytest.raises(ValueError, match="unknown document_type"):
        build_extract_prompt("invoice")


def test_text_hint_is_included_with_the_authority_note():
    prompt = build_extract_prompt("lab_report", text_hint="Hemoglobin 13.5 g/dL")
    assert "Hemoglobin 13.5 g/dL" in prompt
    assert "the image is the source of truth" in prompt
    assert "Do not add fields or values" in prompt


def test_text_hint_is_truncated_with_a_marker():
    long_text = "a" * (MAX_TEXT_HINT_CHARS + 500)
    prompt = build_extract_prompt("receipt", text_hint=long_text)
    assert "a" * MAX_TEXT_HINT_CHARS in prompt
    assert "a" * (MAX_TEXT_HINT_CHARS + 1) not in prompt
    assert "[... text truncated ...]" in prompt


@pytest.mark.parametrize("hint", [None, "", "   \n  "])
def test_empty_text_hint_adds_no_section(hint):
    assert "machine-extracted" not in build_extract_prompt("receipt", text_hint=hint).lower()
    assert "machine-extracted" not in build_classify_prompt(hint).lower()


def test_classify_prompt_defines_every_type_and_the_output_shape():
    prompt = build_classify_prompt()
    for document_type in DOCUMENT_TYPES:
        assert f'"{document_type}": 0.0' in prompt
        assert f"- {document_type}:" in prompt
    assert '"language": "en|vi|th|other"' in prompt
    assert '"reason"' in prompt
    assert "medicines or prices without being a prescription or a receipt" in prompt


def test_classify_prompt_with_text_hint():
    prompt = build_classify_prompt("Total 1.500.000")
    assert "<text>\nTotal 1.500.000\n</text>" in prompt


def test_system_prompt_holds_the_shared_anti_hallucination_rules():
    assert "Never guess" in SYSTEM_PROMPT
    assert 'value" to null' in SYSTEM_PROMPT
    assert "Do not translate or transliterate" in SYSTEM_PROMPT
    assert "0.0-0.1" in SYSTEM_PROMPT
    assert "JSON object only" in SYSTEM_PROMPT


def test_prompts_do_not_contain_test_document_names():
    if not DOCUMENT_TESTS_DIR.is_dir():
        pytest.skip("document-tests/ not present")
    names = [path.name for path in DOCUMENT_TESTS_DIR.iterdir() if path.is_file()]
    assert names, "expected fixture files in document-tests/"
    prompts = [SYSTEM_PROMPT, build_classify_prompt("sample"), build_classify_prompt()]
    prompts += [build_extract_prompt(t, language="vi", text_hint="sample") for t in DOCUMENT_TYPES]
    for prompt in prompts:
        for name in names:
            assert name not in prompt
            assert Path(name).stem not in prompt

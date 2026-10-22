import json

import pytest
from pydantic import ValidationError

from medextract.schemas import (
    DOCUMENT_TYPES,
    FIELD_SPECS,
    Diagnosis,
    ExtractionResult,
    FieldValue,
    LabTest,
    RawExtraction,
    ReceiptItem,
    ValidationIssue,
)


def test_field_specs_cover_every_document_type_and_spec_fields():
    assert set(FIELD_SPECS) == set(DOCUMENT_TYPES)
    assert list(FIELD_SPECS["receipt"]) == [
        "hospital_name",
        "patient_name",
        "date",
        "items",
        "grand_total",
        "payment_method",
    ]
    assert "diagnosis" in FIELD_SPECS["discharge_summary"]
    assert "tests" in FIELD_SPECS["lab_report"]
    assert "medications" in FIELD_SPECS["prescription"]


def test_result_serializes_to_spec_format():
    result = ExtractionResult(
        document_type="receipt",
        confidence=0.95,
        fields={
            "hospital_name": FieldValue(value="Example Hospital", confidence=0.98),
            "grand_total": FieldValue(value=15750.0, confidence=0.97),
            "payment_method": FieldValue(value=None, confidence=0.05),
        },
        validation_errors=[],
        meta={"source": "x.pdf"},
    )
    data = result.to_json_dict(include_meta=False)
    assert set(data) == {"document_type", "confidence", "fields", "validation_errors"}
    assert data["fields"]["grand_total"] == {"value": 15750.0, "confidence": 0.97}
    assert data["fields"]["payment_method"]["value"] is None
    json.dumps(data)  # must be JSON serializable
    assert "meta" in result.to_json_dict(include_meta=True)
    assert "meta" not in result.to_json_dict()  # hidden by default


def test_confidence_must_be_in_unit_interval():
    with pytest.raises(ValidationError):
        FieldValue(value="x", confidence=1.2)
    with pytest.raises(ValidationError):
        ExtractionResult(document_type="receipt", confidence=-0.1)


def test_unknown_document_type_rejected():
    with pytest.raises(ValidationError):
        ExtractionResult(document_type="invoice", confidence=0.5)


def test_raw_models_ignore_unknown_keys_and_accept_nulls():
    raw = RawExtraction.model_validate(
        {"fields": {"date": {"value": None, "confidence": 0.1, "evidence": None, "junk": 1}}, "x": 2}
    )
    assert raw.fields["date"].value is None
    item = ReceiptItem.model_validate({"description": "X-ray", "quantity": 1, "total": 500, "extra": "?"})
    assert item.unit_price is None
    assert Diagnosis().secondary == []
    assert LabTest(flag="high").flag == "high"
    with pytest.raises(ValidationError):
        LabTest(flag="critical")


def test_raw_extraction_wraps_bare_values():
    raw = RawExtraction.model_validate(
        {
            "fields": {
                "items": [{"description": "X-ray", "total": "500"}],
                "diagnosis": {"primary": "Flu", "secondary": []},
                "grand_total": "500",
                "date": {"value": "2024-01-01", "confidence": 0.9},
            }
        }
    )
    assert raw.fields["items"].value == [{"description": "X-ray", "total": "500"}]
    assert raw.fields["items"].confidence is None
    assert raw.fields["diagnosis"].value == {"primary": "Flu", "secondary": []}
    assert raw.fields["grand_total"].value == "500"
    assert raw.fields["date"].confidence == 0.9


def test_validation_issue_defaults_to_error():
    issue = ValidationIssue(field="grand_total", code="total_mismatch", message="off by 7%")
    assert issue.severity == "error"

"""Data contracts shared by every stage of the pipeline.

Two families of models live here:

* Raw models (``RawField``, ``RawExtraction``, ``RawClassification``) describe what the
  VLM is asked to return. They are permissive because model output is untrusted.
* Output models (``FieldValue``, ``ExtractionResult``) describe the pipeline's final JSON,
  which follows the format in ``AI_Challenge_08.md``.

``FIELD_SPECS`` is the single source of truth for which fields each document type has
and what kind of value each field holds. Prompts, normalization and validation are all
driven by it, so adding a field means editing only this table.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DocumentType = Literal["receipt", "discharge_summary", "lab_report", "prescription"]
DOCUMENT_TYPES: tuple[DocumentType, ...] = ("receipt", "discharge_summary", "lab_report", "prescription")


class FieldKind(str, Enum):
    """How a field's value is shaped, normalized and validated."""

    TEXT = "text"  # free text, kept verbatim
    DATE = "date"  # normalized to ISO YYYY-MM-DD
    AMOUNT = "amount"  # money, normalized to a positive float
    TEXT_LIST = "text_list"  # list of strings
    DIAGNOSIS = "diagnosis"  # {"primary": str | None, "secondary": [str]}
    RECEIPT_ITEMS = "receipt_items"  # list of ReceiptItem
    LAB_TESTS = "lab_tests"  # list of LabTest
    MEDICATIONS = "medications"  # list of Medication


FIELD_SPECS: dict[DocumentType, dict[str, FieldKind]] = {
    "receipt": {
        "hospital_name": FieldKind.TEXT,
        "patient_name": FieldKind.TEXT,
        "date": FieldKind.DATE,
        "items": FieldKind.RECEIPT_ITEMS,
        "grand_total": FieldKind.AMOUNT,
        "payment_method": FieldKind.TEXT,
    },
    "discharge_summary": {
        "hospital_name": FieldKind.TEXT,
        "patient_name": FieldKind.TEXT,
        "admission_date": FieldKind.DATE,
        "discharge_date": FieldKind.DATE,
        "diagnosis": FieldKind.DIAGNOSIS,
        "procedures_performed": FieldKind.TEXT_LIST,
        "attending_physician": FieldKind.TEXT,
        "discharge_instructions": FieldKind.TEXT,
    },
    "lab_report": {
        "lab_name": FieldKind.TEXT,
        "patient_name": FieldKind.TEXT,
        "date": FieldKind.DATE,
        "tests": FieldKind.LAB_TESTS,
    },
    "prescription": {
        "doctor_name": FieldKind.TEXT,
        "patient_name": FieldKind.TEXT,
        "date": FieldKind.DATE,
        "medications": FieldKind.MEDICATIONS,
    },
}


# --- Structured sub-values -------------------------------------------------------------


class _Loose(BaseModel):
    """Base for sub-values parsed from model output: ignore unknown keys."""

    model_config = ConfigDict(extra="ignore")


class ReceiptItem(_Loose):
    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    total: float | None = None


LabFlag = Literal["normal", "high", "low"]


class LabTest(_Loose):
    test_name: str | None = None
    result: str | float | None = None
    unit: str | None = None
    reference_range: str | None = None
    flag: LabFlag | None = None


class Medication(_Loose):
    name: str | None = None
    dosage: str | None = None
    frequency: str | None = None
    duration: str | None = None
    quantity: str | float | None = None


class Diagnosis(_Loose):
    primary: str | None = None
    secondary: list[str] = Field(default_factory=list)


# Which sub-model describes one element of each structured kind.
ITEM_MODELS: dict[FieldKind, type[_Loose]] = {
    FieldKind.RECEIPT_ITEMS: ReceiptItem,
    FieldKind.LAB_TESTS: LabTest,
    FieldKind.MEDICATIONS: Medication,
}


# --- Raw VLM responses -----------------------------------------------------------------


class RawField(_Loose):
    """One field as returned by the VLM, before normalization and scoring."""

    value: Any = None
    confidence: float | None = None
    evidence: str | None = None


class RawExtraction(_Loose):
    fields: dict[str, RawField] = Field(default_factory=dict)

    @field_validator("fields", mode="before")
    @classmethod
    def _wrap_bare_values(cls, fields: Any) -> Any:
        """Accept fields the model returned without the {"value", ...} wrapper.

        Small models often answer list or object fields directly, e.g. ``"items": [...]`` or
        ``"diagnosis": {"primary": ...}``. Such values are wrapped with no self-reported
        confidence, so scoring relies on grounding instead.
        """
        if not isinstance(fields, dict):
            return fields
        wrapped = {}
        for name, raw in fields.items():
            is_wrapped = isinstance(raw, dict) and ("value" in raw or "confidence" in raw or "evidence" in raw)
            wrapped[name] = raw if is_wrapped else {"value": raw}
        return wrapped


class RawClassification(_Loose):
    scores: dict[str, float] = Field(default_factory=dict)
    language: str | None = None
    reason: str | None = None


# --- Final output ----------------------------------------------------------------------


class FieldValue(BaseModel):
    value: Any = None
    confidence: float = Field(ge=0.0, le=1.0)


class ValidationIssue(BaseModel):
    field: str | None = None
    code: str
    message: str
    severity: Literal["error", "warning"] = "error"


class ExtractionResult(BaseModel):
    """Final pipeline output for one document (format from AI_Challenge_08.md)."""

    document_type: DocumentType | None
    confidence: float = Field(ge=0.0, le=1.0)
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    validation_errors: list[ValidationIssue] = Field(default_factory=list)
    # Diagnostics (source file, backend, timings, grounding). Not part of the spec format,
    # so it is left out of the JSON output unless explicitly requested.
    meta: dict[str, Any] | None = None

    def to_json_dict(self, include_meta: bool = False) -> dict[str, Any]:
        """JSON-ready dict in the spec format; ``include_meta=True`` adds the diagnostic block."""
        data = self.model_dump(mode="json")
        if not include_meta or self.meta is None:
            data.pop("meta", None)
        return data

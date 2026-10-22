"""Prompt builders for document classification and field extraction.

``SYSTEM_PROMPT`` holds every shared rule (no guessing, verbatim text, confidence
anchors, JSON-only output) and is meant to be sent as the system message. The user
prompts carry only the task, the per-type schema and the optional text hint, which keeps
them short for the small VLM.

The extraction schema is generated from ``FIELD_SPECS`` and the item models in
``schemas.py``, so the prompts cannot drift from the normalization and validation code.
"""

from __future__ import annotations

from typing import cast, get_args

from medextract.schemas import DOCUMENT_TYPES, FIELD_SPECS, ITEM_MODELS, Diagnosis, DocumentType, FieldKind, LabFlag

# Longest embedded PDF text layer passed to the model, in characters.
MAX_TEXT_HINT_CHARS = 6000

SYSTEM_PROMPT = """You read medical documents (receipts, discharge summaries, lab reports and prescriptions) \
in English, Vietnamese and Thai. The document is shown as an image. Answer with JSON only.

Rules:
1. Report only what is visible in the document. Never guess, infer or fill in typical values,
   and never invent a field.
2. If a field is not visible, set "value" to null, "confidence" to 0.0-0.1 and "evidence" to null.
3. Copy names and text exactly as printed, in their original language and script.
   Do not translate or transliterate.
   Each value comes from the place its own label points to. Never swap, reorder or correct
   values, even when they look inconsistent (for example dates out of order or totals that do
   not add up). Consistency is checked later, so report what is printed.
   Person names (patient, doctor, physician): the name with its printed title only (for example
   "Dr.", "BS.", "นพ."), without degrees, specialties, license or registration numbers or brackets.
   Fill a person field only if the document names someone in that role. A director, pharmacist,
   cashier or other signer is not the attending physician or the prescribing doctor.
4. Dates: copy them exactly as printed. Do not convert calendars (for example Buddhist-era
   years) or reformat them.
5. Do not calculate, add up or correct amounts. Report only numbers that are printed.
6. List entries: include only the rows that actually appear in the document.
7. "evidence" is the shortest snippet of the document, copied character by character, that
   supports the value, including the label printed next to it when there is one
   (for example "Discharge date: 12/04/2024"), or null.
8. Confidence reflects how clearly the value is visible:
   - 0.95-1.0: clearly printed and unambiguous.
   - 0.7-0.9: legible, but small, partly hidden or needed interpretation.
   - 0.3-0.6: hard to read, handwritten, faded or ambiguous.
   - 0.0-0.1: not present (the value must be null).
   Give different scores to different fields. Do not give every field the same score.
9. Output one JSON object only: no markdown, no code fences, no commentary."""

_TYPE_DEFINITIONS = """Document types:
- receipt: a bill or payment receipt that lists itemized charges (descriptions, quantities, prices) \
with a total and often a payment method.
- discharge_summary: a hospital discharge or inpatient summary with admission and discharge dates, \
diagnoses, procedures and discharge instructions.
- lab_report: laboratory results: test names with measured values, units and reference ranges, \
often with normal/high/low flags.
- prescription: a prescriber's medication order listing medicines with dosage, frequency, duration \
and/or quantity, usually signed by the doctor.

Decide by the purpose of the whole document, not by single words. A document can mention medicines \
or prices without being a prescription or a receipt (a discharge summary may list discharge \
medications; a receipt may list drugs as line items).
Key cues: itemized charges with a total -> receipt; admission/discharge dates with a diagnosis -> \
discharge_summary; test results with units or reference ranges -> lab_report; medication list with \
dosage and frequency signed by a doctor -> prescription."""

_KIND_HINTS: dict[FieldKind, str] = {
    FieldKind.TEXT: "text, verbatim in its original language and script; null if not visible.",
    FieldKind.DATE: (
        "date, copied exactly as printed (same digits, separators, month names and calendar year). "
        "Do not convert or reformat."
    ),
    FieldKind.AMOUNT: (
        'amount as a string, exactly as printed, with its separators and symbols (for example "1.500.000"). '
        "Do not convert or compute."
    ),
    FieldKind.TEXT_LIST: "array of strings, one per printed entry, verbatim.",
    FieldKind.DIAGNOSIS: (
        'object {"primary": text or null, "secondary": [text, ...]}, verbatim. '
        "Use [] for secondary when none are printed."
    ),
    FieldKind.RECEIPT_ITEMS: "array of objects, one per printed line item.",
    FieldKind.LAB_TESTS: "array of objects, one per printed test row.",
    FieldKind.MEDICATIONS: "array of objects, one per printed medicine.",
}

_LANGUAGE_NAMES = {"en": "English", "vi": "Vietnamese", "th": "Thai"}

_TEXT_HINT_HEADER = (
    "Machine-extracted text from the same document (may be incomplete or out of order; "
    "the image is the source of truth). Use it only to read characters more accurately. "
    "Do not add fields or values that are not visible in the image."
)


def _text_hint_section(text_hint: str | None) -> str:
    """Return the text-hint block, or an empty string when there is no usable text."""
    if text_hint is None or not text_hint.strip():
        return ""
    text = text_hint.strip()
    if len(text) > MAX_TEXT_HINT_CHARS:
        text = text[:MAX_TEXT_HINT_CHARS] + "\n[... text truncated ...]"
    return f"\n\n{_TEXT_HINT_HEADER}\n<text>\n{text}\n</text>"


def _describe_field(kind: FieldKind) -> str:
    """Describe the value shape of one field kind, using the item and sub-value models."""
    if kind is FieldKind.DIAGNOSIS:
        keys = ", ".join(Diagnosis.model_fields)
        return f"{_KIND_HINTS[kind]} Keys: {keys}."
    if kind in ITEM_MODELS:
        keys = ", ".join(ITEM_MODELS[kind].model_fields)
        text = f"{_KIND_HINTS[kind]} Keys: {keys}. Values as printed; use null for a missing key."
        if kind is FieldKind.LAB_TESTS:
            flags = ", ".join(get_args(LabFlag))
            text += f" flag: one of {flags}, only when printed or clearly marked; otherwise null."
        return text
    return _KIND_HINTS[kind]


def _field_schema(document_type: DocumentType) -> str:
    lines = [f'- "{name}": {_describe_field(kind)}' for name, kind in FIELD_SPECS[document_type].items()]
    return "\n".join(lines)


def build_classify_prompt(text_hint: str | None = None) -> str:
    """Prompt asking for a probability per document type, the language and a short reason."""
    return (
        "Classify the document in the image.\n\n"
        f"{_TYPE_DEFINITIONS}\n\n"
        "Return ONLY a JSON object with this shape. The scores must be one probability per type and "
        "sum to about 1.0. "
        "Use the full range: a clear document should score above 0.9 for its type.\n"
        '{"scores": {"receipt": 0.0, "discharge_summary": 0.0, "lab_report": 0.0, "prescription": 0.0}, '
        '"language": "en|vi|th|other", "reason": "<one short sentence naming the cue used, max 20 words>"}'
        f"{_text_hint_section(text_hint)}"
    )


def build_extract_prompt(
    document_type: str,
    *,
    language: str | None = None,
    text_hint: str | None = None,
) -> str:
    """Prompt asking for every field of ``document_type`` with its value, confidence and evidence.

    Raises ``ValueError`` for a document type that is not in ``FIELD_SPECS``.
    """
    if document_type not in DOCUMENT_TYPES:
        expected = ", ".join(DOCUMENT_TYPES)
        raise ValueError(f"unknown document_type {document_type!r}; expected one of {expected}")
    doc_type = cast(DocumentType, document_type)

    header = f"Extract the fields of this {doc_type} from the document image."
    language_name = _LANGUAGE_NAMES.get(language or "")
    if language_name:
        header += f" The document is mainly in {language_name}; keep all text in its original script."

    return (
        f"{header}\n\n"
        'Fields (use these exact names; every field must appear in "fields"):\n'
        f"{_field_schema(doc_type)}\n\n"
        'Return ONLY a JSON object with this shape, where "value" follows the field type above:\n'
        '{"fields": {"<field name>": {"value": ..., "confidence": 0.0, '
        '"evidence": "<verbatim snippet or null>"}}}'
        f"{_text_hint_section(text_hint)}"
    )

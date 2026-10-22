"""Label checks that catch a value attached to the wrong field.

The extraction prompt asks for ``evidence`` that includes the label printed next to each
value. A small model sometimes reads the value correctly but assigns it to the wrong field,
for example when it misreads a Thai label. These checks use the printed labels, in English,
Vietnamese and Thai, to:

* swap admission/discharge dates when their evidence carries the other field's label;
* reject a person field (attending physician, prescribing doctor) when the label next to the
  name shows a different role, such as a hospital director, pharmacist or cashier.

Every change is reported as a warning, so it is visible in ``validation_errors``.
"""

from __future__ import annotations

import unicodedata

from medextract.schemas import RawField, ValidationIssue

ADMISSION_LABELS = (
    "admission",
    "admitted",
    "admit date",
    "date of admission",
    "vào viện",
    "nhập viện",
    "ngày vào",
    "รับไว้",
    "วันที่รับ",
    "แรกรับ",
    "วันรับ",
)
DISCHARGE_LABELS = (
    "discharge",
    "discharged",
    "ra viện",
    "xuất viện",
    "จำหน่าย",
    "วันที่ออก",
)
# Roles that are not the treating or prescribing doctor.
OTHER_ROLE_LABELS = (
    "director",
    "giám đốc",
    "ผู้อำนวยการ",
    "pharmacist",
    "dược sĩ",
    "เภสัชกร",
    "cashier",
    "thu ngân",
    "แคชเชียร์",
    "accountant",
    "kế toán",
)
PERSON_ROLE_FIELDS = ("attending_physician", "doctor_name")
# How many text-layer lines above a name are searched for its label (signature blocks).
LABEL_LOOKBACK_LINES = 3


def _fold(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _has_any(text: str | None, labels: tuple[str, ...]) -> bool:
    if not text:
        return False
    folded = _fold(text)
    return any(label in folded for label in labels)


def _date_role(evidence: str | None) -> str | None:
    """'admission', 'discharge' or None (no label, or both labels in one snippet)."""
    admission = _has_any(evidence, ADMISSION_LABELS)
    discharge = _has_any(evidence, DISCHARGE_LABELS)
    if admission == discharge:
        return None
    return "admission" if admission else "discharge"


def reconcile_date_labels(fields: dict[str, RawField]) -> list[ValidationIssue]:
    """Swap admission/discharge dates in place when their labels show they were crossed."""
    admission, discharge = fields.get("admission_date"), fields.get("discharge_date")
    if admission is None or discharge is None or admission.value is None or discharge.value is None:
        return []
    adm_role, dis_role = _date_role(admission.evidence), _date_role(discharge.evidence)
    crossed = adm_role == "discharge" or dis_role == "admission"
    own_label_seen = adm_role == "admission" or dis_role == "discharge"
    if not crossed or own_label_seen:
        return []
    fields["admission_date"], fields["discharge_date"] = discharge, admission
    return [
        ValidationIssue(
            field="admission_date",
            code="fields_swapped_by_label",
            message=(
                "Admission and discharge dates were swapped to match their printed labels "
                f"(admission evidence: {discharge.evidence!r}; discharge evidence: {admission.evidence!r})"
            ),
            severity="warning",
        )
    ]


def _label_context(value: str, evidence: str | None, text_layer: str | None) -> str:
    """Evidence plus the text-layer lines just above the first line that contains the value."""
    parts = [evidence or ""]
    if text_layer:
        lines = text_layer.splitlines()
        target = _fold(value)
        for index, line in enumerate(lines):
            if target and target in _fold(line):
                parts.extend(lines[max(0, index - LABEL_LOOKBACK_LINES) : index + 1])
                break
    return "\n".join(parts)


def check_person_roles(fields: dict[str, RawField], text_layer: str | None) -> list[ValidationIssue]:
    """Null out person fields whose printed label shows a different role (in place)."""
    issues: list[ValidationIssue] = []
    for name in PERSON_ROLE_FIELDS:
        field = fields.get(name)
        if field is None or not isinstance(field.value, str) or not field.value.strip():
            continue
        context = _label_context(field.value, field.evidence, text_layer)
        if _has_any(context, OTHER_ROLE_LABELS):
            issues.append(
                ValidationIssue(
                    field=name,
                    code="role_mismatch",
                    message=(
                        f"{field.value!r} is labelled with a different role in the document; the field was set to null"
                    ),
                    severity="warning",
                )
            )
            fields[name] = RawField(value=None, confidence=0.05, evidence=None)
    return issues

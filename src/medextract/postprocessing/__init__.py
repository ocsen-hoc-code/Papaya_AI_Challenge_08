"""Deterministic steps after the model call, in pipeline order.

1. ``roles``       - fix values attached to the wrong field, using printed labels
2. ``normalize``   - ISO dates, numbers as floats, verbatim text
3. ``grounding``   - check values against the PDF text layer
4. ``confidence``  - per-field and document confidence
5. ``validate``    - dates, amounts, totals, lab flags
"""

from medextract.postprocessing.confidence import (
    classification_confidence,
    document_confidence,
    field_confidence,
    find_value_span,
    span_token_prob,
)
from medextract.postprocessing.grounding import Grounding, ground_field, text_layer_usable
from medextract.postprocessing.normalize import Normalized, detect_language, normalize_value
from medextract.postprocessing.roles import check_person_roles, reconcile_date_labels
from medextract.postprocessing.validate import compute_lab_flag, validate_fields

__all__ = [
    "Grounding",
    "Normalized",
    "check_person_roles",
    "classification_confidence",
    "compute_lab_flag",
    "detect_language",
    "document_confidence",
    "field_confidence",
    "find_value_span",
    "ground_field",
    "normalize_value",
    "reconcile_date_labels",
    "span_token_prob",
    "text_layer_usable",
    "validate_fields",
]

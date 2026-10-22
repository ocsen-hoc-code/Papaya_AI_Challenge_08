"""End-to-end tests with the real model. Run with: .venv/bin/pytest -m slow tests/test_integration.py

Skipped unless a usable backend is available (MLX on Apple Silicon, or MEDEXTRACT_DEVICE set).
Assertions check structure and a few unambiguous values, not exact model wording.
"""

import os
from pathlib import Path

import pytest

from medextract.backends.factory import mlx_available
from medextract.pipeline import Pipeline
from medextract.schemas import FIELD_SPECS

DOCS = Path(__file__).resolve().parents[2] / "document-tests"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (mlx_available() or os.environ.get("MEDEXTRACT_DEVICE")),
        reason="needs Apple Silicon (MLX) or an explicit MEDEXTRACT_DEVICE",
    ),
]


@pytest.mark.parametrize(
    ("filename", "expected_type"),
    [("09_prescription_en.pdf", "prescription"), ("07_lab_report_en.pdf", "lab_report")],
)
async def test_real_document(filename, expected_type):
    path = DOCS / filename
    if not path.exists():
        pytest.skip(f"{filename} not present")
    async with Pipeline.from_env() as pipeline:
        result = await pipeline.process_document(path)

    assert result.document_type == expected_type, result.validation_errors
    assert result.confidence >= 0.6
    assert set(result.fields) == set(FIELD_SPECS[expected_type])
    assert result.fields["patient_name"].value
    list_field = "medications" if expected_type == "prescription" else "tests"
    assert len(result.fields[list_field].value) >= 1
    assert all(0.0 <= f.confidence <= 1.0 for f in result.fields.values())

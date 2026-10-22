"""Pipeline and CLI tests with a scripted FakeBackend (no GPU, no model download)."""

import asyncio
import json

import pymupdf
import pytest

from medextract import cli
from medextract.backends.fake import FakeBackend
from medextract.config import Settings
from medextract.pipeline import Pipeline

RECEIPT_TEXT = [
    "City Care Hospital",
    "Patient: Jane Roe",
    "Date: 05/02/2024",
    "Consultation 1 500.00 500.00",
    "X-ray 2 250.00 500.00",
    "Grand total 1,000.00",
    "Paid by: Credit card",
]

CLASSIFY_RECEIPT = json.dumps(
    {
        "scores": {"receipt": 0.9, "discharge_summary": 0.04, "lab_report": 0.03, "prescription": 0.03},
        "language": "en",
        "reason": "itemized charges with a total",
    }
)


def _field(value, confidence=0.95, evidence=None):
    return {"value": value, "confidence": confidence, "evidence": evidence}


def receipt_extraction(**overrides):
    fields = {
        "hospital_name": _field("City Care Hospital", evidence="City Care Hospital"),
        "patient_name": _field("Jane Roe", evidence="Patient: Jane Roe"),
        "date": _field("05/02/2024", evidence="Date: 05/02/2024"),
        "items": _field(
            [
                {"description": "Consultation", "quantity": "1", "unit_price": "500.00", "total": "500.00"},
                {"description": "X-ray", "quantity": "2", "unit_price": "250.00", "total": "500.00"},
            ]
        ),
        "grand_total": _field("1,000.00", evidence="Grand total 1,000.00"),
        "payment_method": _field("Credit card", evidence="Paid by: Credit card"),
    }
    fields.update(overrides)
    return json.dumps({"fields": fields})


def make_pdf(path, lines):
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(lines):
        page.insert_text((72, 72 + 18 * i), line, fontsize=11)
    doc.save(path)
    return path


def scripted(extraction_text, classify_text=CLASSIFY_RECEIPT):
    def respond(images, prompt):
        return classify_text if "Document types:" in prompt else extraction_text

    return respond


def pipeline_with(responder, **settings):
    return Pipeline(FakeBackend(responder), Settings(**settings))


async def test_single_receipt_end_to_end(tmp_path):
    pdf = make_pdf(tmp_path / "receipt.pdf", RECEIPT_TEXT)
    async with pipeline_with(scripted(receipt_extraction())) as pipeline:
        result = await pipeline.process_document(pdf)

    data = result.to_json_dict(include_meta=False)
    assert set(data) == {"document_type", "confidence", "fields", "validation_errors"}
    assert result.document_type == "receipt"
    assert result.confidence > 0.6
    f = result.fields
    assert f["hospital_name"].value == "City Care Hospital"
    assert f["date"].value == "2024-02-05"  # day-first
    assert f["grand_total"].value == pytest.approx(1000.0)
    assert len(f["items"].value) == 2
    assert all(0.0 <= fv.confidence <= 1.0 for fv in f.values())
    assert not [i for i in result.validation_errors if i.code == "total_mismatch"]
    assert result.meta["timings"]["total_s"] >= 0
    assert result.meta["has_text_layer"] is True


async def test_missing_field_is_null_with_low_confidence(tmp_path):
    pdf = make_pdf(tmp_path / "receipt.pdf", RECEIPT_TEXT)
    extraction = receipt_extraction(payment_method=_field(None, confidence=0.0))
    async with pipeline_with(scripted(extraction)) as pipeline:
        result = await pipeline.process_document(pdf)
    assert result.fields["payment_method"].value is None
    assert result.fields["payment_method"].confidence <= 0.1


async def test_invented_value_gets_low_confidence_and_warning(tmp_path):
    pdf = make_pdf(tmp_path / "receipt.pdf", RECEIPT_TEXT)
    extraction = receipt_extraction(patient_name=_field("Johnathan Smithers", confidence=0.97))
    async with pipeline_with(scripted(extraction)) as pipeline:
        result = await pipeline.process_document(pdf)
    patient = result.fields["patient_name"]
    assert patient.confidence <= 0.3
    assert patient.confidence < result.fields["hospital_name"].confidence
    assert any(i.code == "ungrounded_value" and i.field == "patient_name" for i in result.validation_errors)


async def test_total_mismatch_is_flagged(tmp_path):
    pdf = make_pdf(tmp_path / "receipt.pdf", [*RECEIPT_TEXT[:-2], "Grand total 1,200.00"])
    extraction = receipt_extraction(grand_total=_field("1,200.00", evidence="Grand total 1,200.00"))
    async with pipeline_with(scripted(extraction)) as pipeline:
        result = await pipeline.process_document(pdf)
    assert any(i.code == "total_mismatch" for i in result.validation_errors)


async def test_batch_isolates_failures_and_keeps_order(tmp_path):
    good = make_pdf(tmp_path / "a.pdf", RECEIPT_TEXT)
    corrupt = tmp_path / "b.pdf"
    corrupt.write_bytes(b"%PDF-1.4 this is not really a pdf")
    missing = tmp_path / "c.pdf"
    async with pipeline_with(scripted(receipt_extraction())) as pipeline:
        results = await pipeline.process_batch([good, corrupt, missing])

    assert [r.meta["source"] for r in results] == ["a.pdf", "b.pdf", "c.pdf"]
    assert results[0].document_type == "receipt"
    for failed in results[1:]:
        assert failed.document_type is None
        assert failed.confidence == 0.0
        assert failed.validation_errors[0].code == "ingest_error"


async def test_invalid_model_output_is_reported_not_raised(tmp_path):
    pdf = make_pdf(tmp_path / "receipt.pdf", RECEIPT_TEXT)
    async with pipeline_with(scripted("I cannot read this document.")) as pipeline:
        result = await pipeline.process_document(pdf)
    assert result.document_type is None
    assert result.validation_errors[0].code == "invalid_model_output"


async def test_timeout_is_reported(tmp_path):
    pdf = make_pdf(tmp_path / "receipt.pdf", RECEIPT_TEXT)
    pipeline = Pipeline(FakeBackend(scripted(receipt_extraction()), delay_s=0.5), Settings(timeout_s=0.05))
    async with pipeline:
        result = await pipeline.process_document(pdf)
    assert result.validation_errors[0].code == "timeout"


async def test_unclassifiable_document(tmp_path):
    pdf = make_pdf(tmp_path / "x.pdf", ["Hello"])
    classify = json.dumps({"scores": {"invoice": 1.0}, "language": "en"})
    async with pipeline_with(scripted(receipt_extraction(), classify_text=classify)) as pipeline:
        result = await pipeline.process_document(pdf)
    assert result.document_type is None
    assert result.validation_errors[0].code == "unclassified"


@pytest.mark.parametrize(("concurrency", "expect_overlap"), [(1, False), (3, True)])
async def test_concurrency_limit_is_respected(tmp_path, concurrency, expect_overlap):
    paths = [make_pdf(tmp_path / f"d{i}.pdf", RECEIPT_TEXT) for i in range(4)]
    backend = FakeBackend(scripted(receipt_extraction()), delay_s=0.02)
    async with Pipeline(backend, Settings(concurrency=concurrency)) as pipeline:
        results = await pipeline.process_batch(paths)
    assert len(results) == 4
    assert (backend.max_in_flight > 1) is expect_overlap
    assert backend.max_in_flight <= concurrency


async def test_on_result_callback_and_stream(tmp_path):
    paths = [make_pdf(tmp_path / f"d{i}.pdf", RECEIPT_TEXT) for i in range(3)]
    seen = []

    async def on_result(result):
        await asyncio.sleep(0)
        seen.append(result.meta["source"])

    async with pipeline_with(scripted(receipt_extraction())) as pipeline:
        await pipeline.process_batch(paths, on_result=on_result)
        streamed = [r.meta["source"] async for r in pipeline.stream(paths)]
    assert sorted(seen) == sorted(streamed) == ["d0.pdf", "d1.pdf", "d2.pdf"]


def test_cli_run_writes_results_and_summary(tmp_path, monkeypatch, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    make_pdf(docs / "r1.pdf", RECEIPT_TEXT)
    (docs / "notes.txt").write_text("ignored")
    out = tmp_path / "out"

    monkeypatch.setattr(
        Pipeline, "from_env", classmethod(lambda cls, **kw: cls(FakeBackend(scripted(receipt_extraction()))))
    )
    exit_code = cli.main(["run", str(docs), "-o", str(out)])

    assert exit_code == 0
    result = json.loads((out / "r1.json").read_text(encoding="utf-8"))
    assert result["document_type"] == "receipt"
    assert "meta" not in result  # hidden by default
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [row["source"] for row in summary] == ["r1.pdf"]
    assert "r1.pdf" in capsys.readouterr().err


@pytest.mark.parametrize("with_meta", [False, True])
def test_cli_meta_flag(tmp_path, monkeypatch, capsys, with_meta):
    pdf = make_pdf(tmp_path / "r1.pdf", RECEIPT_TEXT)
    out = tmp_path / "out"
    monkeypatch.setattr(
        Pipeline, "from_env", classmethod(lambda cls, **kw: cls(FakeBackend(scripted(receipt_extraction()))))
    )
    flags = ["--meta"] if with_meta else []

    # File output.
    assert cli.main(["run", str(pdf), "-o", str(out), *flags]) == 0
    saved = json.loads((out / "r1.json").read_text(encoding="utf-8"))
    assert ("meta" in saved) is with_meta
    if with_meta:
        assert saved["meta"]["source"] == "r1.pdf"
        assert "timings" in saved["meta"]

    # Single file printed to stdout.
    capsys.readouterr()
    assert cli.main(["run", str(pdf), *flags]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert ("meta" in printed) is with_meta


async def test_document_confidence_drops_when_key_fields_are_missing(tmp_path):
    pdf = make_pdf(tmp_path / "receipt.pdf", RECEIPT_TEXT)
    no_content = receipt_extraction(items=_field(None, confidence=0.0), grand_total=_field(None, confidence=0.0))
    async with pipeline_with(scripted(receipt_extraction())) as pipeline:
        good = await pipeline.process_document(pdf)
    async with pipeline_with(scripted(no_content)) as pipeline:
        weak = await pipeline.process_document(pdf)
    assert good.confidence > 0.8
    assert weak.confidence < 0.6 < good.confidence
    assert weak.meta["classifier_confidence"] == good.meta["classifier_confidence"]
    assert any(i.code == "low_classification_confidence" for i in weak.validation_errors)

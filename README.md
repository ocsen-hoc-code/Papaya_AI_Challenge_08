# medextract: medical document extractor

An async pipeline that takes a medical document (PDF or image) and returns structured JSON. Each result has the
document type, the extracted fields with per-field confidence scores, and validation errors. It runs the
**Qwen3-VL-8B-Instruct** vision-language model locally, in-process. It uses the Apple GPU (MLX), an NVIDIA GPU (4-bit) or
the CPU, depending on what the machine has.

Built for AI Challenge 08 (`AI_Challenge_08.md`). It handles receipts, discharge summaries, lab reports and prescriptions
in **English, Vietnamese and Thai**.

```json
{
  "document_type": "receipt",
  "confidence": 0.95,
  "fields": {
    "hospital_name": { "value": "Riverside General Hospital", "confidence": 0.98 },
    "date":          { "value": "2024-03-14", "confidence": 0.96 },
    "grand_total":   { "value": 931.5, "confidence": 0.97 },
    "payment_method":{ "value": null, "confidence": 0.05 }
  },
  "validation_errors": []
}
```

## Install

Python 3.10+. Pick the extra for your hardware:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[mlx,dev]"    # Apple Silicon (Metal GPU), recommended on Macs
.venv/bin/pip install -e ".[cuda,dev]"   # NVIDIA GPU, 4-bit NF4 via bitsandbytes
.venv/bin/pip install -e ".[cpu,dev]"    # CPU only (slow; the 8B model needs ~17 GB RAM in bf16)
```

The model weights download from Hugging Face on first use: ~5 GB for MLX, ~17 GB for Transformers.

## Usage

```bash
.venv/bin/medextract info                                    # which backend/model will be used
.venv/bin/medextract run document-tests/ -o results/         # a whole folder -> results/<name>.json + summary.json
.venv/bin/medextract run document-tests/01_receipt_en.pdf    # one file -> JSON on stdout
.venv/bin/medextract run docs/ -o out/ --concurrency 2 --device cpu --model Qwen/Qwen3-VL-2B-Instruct
.venv/bin/python scripts/eval.py                             # score results/ against ground_truth/
```

From Python:

```python
import asyncio
from medextract.pipeline import Pipeline

async def main():
    async with Pipeline.from_env() as pipeline:          # auto-selects MLX -> CUDA -> CPU
        result = await pipeline.process_document("x.pdf")
        print(result.to_json_dict(include_meta=False))
        results = await pipeline.process_batch(["a.pdf", "b.png"], concurrency=4)
        async for r in pipeline.stream(["c.jpg", "d.pdf"]):   # results as they finish
            ...

asyncio.run(main())
```

Settings come from environment variables, and CLI flags override them:

| Variable | Default | Meaning |
|---|---|---|
| `MEDEXTRACT_DEVICE` | `auto` | `auto` \| `mlx` \| `cuda` \| `cpu` |
| `MEDEXTRACT_MODEL_ID` | per backend | e.g. `mlx-community/Qwen3-VL-8B-Instruct-4bit`, `Qwen/Qwen3-VL-8B-Instruct` |
| `MEDEXTRACT_CONCURRENCY` | `4` | documents in flight |
| `MEDEXTRACT_MAX_IMAGE_SIDE` | `1600` | longest image side sent to the model (px) |
| `MEDEXTRACT_PDF_DPI` | `200` | PDF render resolution |
| `MEDEXTRACT_MAX_TOKENS` | `2048` | generation limit per call |
| `MEDEXTRACT_TIMEOUT_S` | `600` | timeout per model call |

## How it works

```
file ─► ingest ─────────► classify ─────► extract ───────► normalize ─► ground ─► score ─► validate ─► JSON
        (PDF render,      (VLM, small     (VLM, per-type   (ISO dates,  (vs PDF   (conf.)  (dates,
         text layer)       image, 4        prompt from      amounts,     text                totals, flags)
                           type scores)    FIELD_SPECS)     Thai/VN)     layer)
```

- **Async.** `asyncio` keeps up to `concurrency` documents in flight; a semaphore bounds memory. All model calls go
  through **one dedicated worker thread**, because a local GPU serves one generation at a time and MLX is not
  thread-safe. Loading, PDF rendering and post-processing of other documents overlap with inference. A failing document
  produces a result with `validation_errors` (`ingest_error`, `timeout`, `invalid_model_output`, …) and never stops the
  batch.
- **Backends.** `vlm/base.py` defines `VLMBackend`. `device.py` picks `MLXBackend` (Apple Silicon), then
  `TransformersBackend` on CUDA (4-bit NF4), then the CPU (bf16). `FakeBackend` drives the unit tests without a model.
- **The model transcribes; code decides.** Prompts ask for values exactly as printed. Calendar conversion
  (Buddhist era), number formats (`1.500.000`, `1,500.00`, Thai digits), totals and lab-flag checks happen in
  deterministic code.
- **No hallucination.** Missing fields are `null` with confidence ≤ 0.1. Every value is checked against the PDF text
  layer when there is one; a value that can't be found is capped at 0.3 confidence and flagged `ungrounded_value`.
- **Meaningful confidence.** The score combines the model's own confidence, grounding against the text layer,
  normalization problems (invalid → ≤ 0.3, ambiguous date ×0.7) and token probabilities.

See **[docs/PROMPTING.md](docs/PROMPTING.md)** for the prompt engineering approach.

### Validation rules

| Code | Severity | Rule |
|---|---|---|
| `invalid_date` | error | a printed date could not be parsed or is impossible (e.g. 31/02) |
| `date_order` | error | discharge date before admission date |
| `invalid_amount`, `non_positive_amount`, `invalid_quantity` | error | amounts and quantities must be positive numbers |
| `total_mismatch` | error | sum of item totals differs from `grand_total` by more than 5% |
| `line_total_mismatch` | warning | quantity × unit price differs from the line total |
| `lab_flag_mismatch` | warning | printed flag disagrees with the reference range |
| `ambiguous_date`, `future_date`, `missing_required`, `incomplete_items` | warning | |
| `ungrounded_value` | warning | value not found in the document's text layer |

Lab tests without a printed flag get one derived from the reference range, marked `"flag_source": "derived"`.

## Project layout

```
src/medextract/   pipeline.py  cli.py  config.py  device.py  ingest.py  schemas.py
                  normalize.py validate.py grounding.py confidence.py parsing.py roles.py
                  prompts/     vlm/{base,mlx_backend,transformers_backend,fake}.py
tests/            unit tests (FakeBackend) + test_integration.py (real model, -m slow)
document-tests/   the 10 test documents
ground_truth/     hand-labelled answers for the 10 documents
results/          pipeline output for the 10 documents + eval_report.md
scripts/          eval.py, smoke_mlx.py
tasks/            task board (todo / pending / done)
```

## Tests

```bash
.venv/bin/pytest -q                                   # unit tests, no model needed
.venv/bin/pytest tests/test_normalize.py -q           # one file
.venv/bin/pytest tests/test_validate.py -k total -q   # one test by name
.venv/bin/pytest -m slow -q                           # real-model tests (downloads/uses the model)
.venv/bin/ruff check src tests scripts
```

## Results on the 10 test documents

Outputs are in `results/` (one JSON per document, `summary.json`, and `eval_report.md` from `scripts/eval.py`, which
compares them with the hand-labelled `ground_truth/`). Run on an Apple M3 Pro (18 GB) with
`mlx-community/Qwen3-VL-8B-Instruct-4bit`, ~2.4 min per document (two model calls each).

| Metric | Result |
|---|---|
| Classification | **10/10** correct |
| Field accuracy (58 fields) | **98.3%** (one partial match, see below) |
| Hallucinated fields (value where the truth is null) | **0** |
| Validation errors caught | **2/2**: `total_mismatch` in `02_receipt_vi` (items 2,746,000 vs total 2,938,000, 6.5%) and `date_order` in `06_discharge_summary_th` (discharge 12/04/2567 before admission 15/04/2567) |
| Mean confidence: PDF fields / image fields / absent fields | 0.93 / 0.66 / 0.05 |

How the prompts and post-processing were tuned (each change is general, not tied to a test file):

| Run | Field accuracy | Hallucinated | Validation caught | Change |
|---|---|---|---|---|
| 1 | 89.7% | 1 | 1/2 | baseline |
| 2 | 93.1% | 1 | 1/2 | prompt: never swap/correct values; person names without degrees or registration numbers; a person field only for that role; evidence includes the printed label. Text values lose enclosing brackets. |
| 3 | **98.3%** | **0** | **2/2** | `roles.py`: printed labels in the evidence/text layer swap back crossed admission/discharge dates and reject a director/pharmacist named as the attending physician (both reported as warnings) |

Remaining difference: in `05_discharge_summary_vi` the "treatment" line lists the surgery plus "kháng sinh, giảm đau,
truyền dịch" (antibiotics, analgesia, IV fluids). The pipeline reports both as procedures; the ground truth keeps only
the surgery. This is a judgement call rather than a misread.

The confidence scores follow how reliable the source is: values grounded in a PDF text layer score ~0.93–0.98, values
read only from an image (no text to check against) ~0.5–0.73, and absent fields 0.05. Classification confidence is 1.0 for
all 10 documents, because the model gave near-certain type scores on these clear examples.

## Limitations

- **The CUDA path has not run on a real NVIDIA GPU.** It has unit tests with mocked `transformers`/`bitsandbytes`. The CPU
  path was verified for real with `Qwen/Qwen3-VL-2B-Instruct`; the 8B model on CPU needs ~17 GB RAM and is slow.
- Images (scans, photos) have no text layer, so their confidence relies on the model and its evidence snippets. It is
  lower and less discriminative than for born-digital PDFs.
- English dates like `05/02/2024` are read day-first and flagged `ambiguous_date`.
- One page image per call works best; long multi-page PDFs are capped at 10 pages.

## Timeline

Estimated up front: **7–8 hours** for one developer (spec: 4–6 h for a cloud-API solution; the local multi-backend
model, async pipeline and evaluation add to that).

| Phase | Tasks (`tasks/`) | Estimate |
|---|---|---|
| Setup and model smoke test | T01–T02 | 0.5 h |
| Schemas, ingest, normalization | T03–T05 | 0.75 h |
| Backends, device detection, prompts, JSON parsing | T06–T11 | 2.75 h |
| Async pipeline and CLI | T12–T13 | 1 h |
| Grounding, confidence, validation | T14–T16 | 1 h |
| Ground truth, evaluation, tuning, docs | T17–T22 | 2 h |

Actual: about **3 hours of wall-clock time**, using Claude Code with parallel sub-agents for independent tasks (ingest,
normalization/validation, backends, prompts/parsing, ground truth, grounding/confidence). Roughly 1 hour of that was
spent waiting on the three full model runs over the 10 documents.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repo holds a take-home challenge: **AI Challenge 08, Medical Document Extractor**. The full spec is in `AI_Challenge_08.md`; read it before you change anything. The implementation is the Python package `medextract` (`src/medextract/`); `README.md` covers usage and `docs/PROMPTING.md` the prompt design.

## Commands

The venv is `.venv/` (not committed). Create it with `./setup.sh` (finds a Python 3.10+, detects MLX/CUDA/CPU, installs, runs the unit tests); `make help` lists shortcuts. macOS system `python3` is 3.9 and too old.

```bash
.venv/bin/pytest -q                                    # unit tests (FakeBackend, no model, ~5 s)
.venv/bin/pytest tests/unit/postprocessing/test_validate.py -q            # one file
.venv/bin/pytest tests/unit/postprocessing/test_validate.py -k total -q   # one test by name
.venv/bin/pytest -m slow -q                            # real-model tests (slow; deselected by default)
.venv/bin/ruff check src tests scripts                 # lint (line length 120)
.venv/bin/medextract info                              # detected backend/model
.venv/bin/medextract run document-tests/ -o results/   # full run, ~2-3 min per document on an M3 Pro
.venv/bin/python scripts/eval.py                       # score results/ vs ground_truth/ -> results/eval_report.md
```

Only one process should use the 8B model at a time on an 18 GB machine (~7 GB peak each).

## Architecture

- `pipeline.py` orchestrates: `ingest` (PDF render + text layer) → classify (VLM, 1024 px image) → extract (VLM, per-type prompt) → `assemble_result`, which normalizes, grounds, scores confidence and validates each field. `process_batch` bounds in-flight documents with a semaphore. A document-level failure becomes a result with a `validation_errors` code and never raises.
- Layout: `src/medextract/` has `pipeline.py`, `config.py`, `schemas.py`, `cli.py` at the top, plus one subpackage per layer: `ingestion/` (loader), `backends/` (base, factory, mlx, transformers, fake), `extraction/` (prompts, parsing), `postprocessing/` (roles, normalize, grounding, confidence, validate). Imports are absolute (`from medextract.postprocessing.normalize import ...`). Tests mirror it under `tests/unit/`; real-model tests live in `tests/integration/`.
- `schemas.FIELD_SPECS` is the single source of truth for fields per type. Prompts (`extraction/prompts.py`), normalization and validation are all driven by it, so add a field there first.
- `backends/base.py`: `VLMBackend` interface; `ThreadedBackend` runs all model calls on one dedicated thread (MLX is not thread-safe). `backends/factory.create_backend` picks MLX → CUDA → CPU. Never load the MLX and Transformers backends in the same process (mlx-vlm patches processors process-wide).
- The model only transcribes (dates and amounts as printed). Conversion lives in `postprocessing/normalize.py`, rules in `validate.py`, hallucination checks against the PDF text layer in `grounding.py`, and the confidence formula in `confidence.py`.
- `RawExtraction` accepts bare field values (the model often omits the `{"value": ...}` wrapper for lists).

## Task tracking (`tasks/`)

The work is split into tasks `tasks/T01.md`–`T22.md`. Each one has a description, acceptance criteria, dependencies and a `status:` (`todo` / `pending` / `done`) in its frontmatter. `tasks/BOARD.md` is generated, so don't edit it by hand.

- `python3 tasks/tasks.py next` lists tasks you can start now (their dependencies are done).
- Run `python3 tasks/tasks.py set <ID> pending` when you start a task, and `set <ID> done` only when every acceptance criterion in its file is ticked.
- Write findings and decisions in the task's `## Notes` section.

## Language

Write everything in **English**: code, comments, docstrings, log and CLI messages, commit messages, task files and docs. The only exceptions are the source documents themselves and values extracted from them, which stay verbatim in Vietnamese or Thai.

## Agreed design decisions

- Python with the `Qwen3-VL-8B-Instruct` vision model, run in-process. There is no API server (FastAPI was dropped); the entry points are the CLI and the `Pipeline` class.
- Async pipeline: `asyncio` with bounded queues. All model calls go through a single dedicated thread, while ingest and post-processing run concurrently.
- The backend is chosen automatically: MLX on Apple Silicon (`mlx-community/Qwen3-VL-8B-Instruct-4bit`), then CUDA (`transformers` with bitsandbytes 4-bit), then CPU. Override with `MEDEXTRACT_DEVICE` and `MEDEXTRACT_MODEL_ID`.
- Test data is `document-tests/`. Don't move or rename it. Ground-truth labels go in `ground_truth/` and outputs go in `results/`.

## What is being built

The pipeline takes a medical document (PDF or image) and returns structured JSON. It uses a vision-capable LLM to:

1. **Classify** the document as one of `receipt`, `discharge_summary`, `lab_report` or `prescription`, with a confidence score.
2. **Extract** the fields for that type (the field lists per type are in the spec).
3. **Validate** the result: dates must be valid, amounts must be positive, and item totals must sum to `grand_total` (flag it if the mismatch is over 5%). Report problems in `validation_errors`.

Output shape (see the spec for the full example). Every field is wrapped as `{ "value": ..., "confidence": 0.0–1.0 }`:

```json
{ "document_type": "...", "confidence": 0.95, "fields": { "<name>": { "value": ..., "confidence": ... } }, "validation_errors": [] }
```

## Test documents (`document-tests/`)

There are 10 fixtures. The filename prefix gives the expected type and the suffix gives the language:

| Type | Files |
|---|---|
| receipt | `01_receipt_en.pdf`, `02_receipt_vi.pdf`, `03_receipt_th.pdf` |
| discharge_summary | `04_discharge_summary_en.pdf`, `05_discharge_summary_vi.pdf`, `06_discharge_summary_th.png` |
| lab_report | `07_lab_report_en.pdf`, `08_lab_report_vi.jpg` |
| prescription | `09_prescription_en.pdf`, `10_prescription_vi.png` |

What this means for the code:
- Inputs come as both PDF and raster images (PNG/JPG). PDFs must be rendered or converted to something the LLM's vision input accepts.
- The documents are in **English, Vietnamese and Thai**. Prompts and parsing must handle non-Latin scripts and local date and number formats (for example, Thai Buddhist-era years and Vietnamese thousands separators like `1.500.000`). Normalize values to standard forms (ISO dates, numeric amounts).

## Graded requirements to keep in mind

- **No hallucination.** If a field isn't visible in the document, return `value: null` with low confidence. Never fill it with a plausible guess.
- **Meaningful confidence.** Scores must differ by field and drop for uncertain or illegible fields, not sit uniformly high.
- **Reusable pipeline.** Never hardcode logic to these specific test files; the classification must not rely on filenames.
- Validation should be deterministic code that runs after the LLM output, not left to the LLM.

## Deliverables (from the spec)

- The 10 test documents plus their extraction results, committed to the repo.
- A short writeup of the prompt engineering approach.
- A timeline estimate.
- The project pushed to GitHub. The default branch is `main`.

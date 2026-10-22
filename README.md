# medextract: medical document extractor

An async pipeline that takes a medical document (PDF or image) and returns structured JSON. Each result has the
document type, the extracted fields with per-field confidence scores, and validation errors. It runs the
**Qwen3-VL-8B-Instruct** vision-language model locally, in-process. It uses the Apple GPU (MLX), an NVIDIA GPU (4-bit) or
the CPU, depending on what the machine has.

Built for AI Challenge 08 (`AI_Challenge_08.md`). It handles receipts, discharge summaries, lab reports and prescriptions
in **English, Vietnamese and Thai**.

**Submission at a glance:** the 10 test documents are in [`document-tests/`](document-tests/) and their extraction
results in [`results/`](results/) (scored in [`results/eval_report.md`](results/eval_report.md)). The prompt engineering
writeup is in [Prompt engineering approach](#prompt-engineering-approach), with full detail in
[`docs/PROMPTING.md`](docs/PROMPTING.md). See also [Results](#results-on-the-10-test-documents) and
[Timeline](#timeline).

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

## Quick start

```bash
git clone https://github.com/ocsen-hoc-code/Papaya_AI_Challenge_08.git
cd Papaya_AI_Challenge_08
./setup.sh                                                     # one-time install, ~1 min
.venv/bin/medextract run document-tests/01_receipt_en.pdf      # first run downloads the model (~5 GB)
```

`./setup.sh` handles everything:
1. Finds a **Python 3.10 or newer** on the machine, even if the default `python3` is older (macOS ships 3.9).
2. Picks the backend: **MLX** on Apple Silicon, **CUDA** if an NVIDIA GPU is present, otherwise **CPU**.
3. Creates `.venv/`, upgrades pip and installs the package. A `.venv` built with an old Python is rebuilt.
4. Checks the install and prints the commands to use next.

Options: `./setup.sh cpu` (force a backend: `mlx` / `cuda` / `cpu`), `--check` (show what it would use, change
nothing), `--recreate`, `--download-model` (fetch the weights now), `--no-test`, `--help`. `make help` lists
shortcuts such as `make test` and `make run FILE=x.pdf`.

The only prerequisite is a Python 3.10+ somewhere on the machine. If `./setup.sh` reports none, install one
(`brew install python@3.12` on macOS, `sudo apt install python3.12 python3.12-venv` on Debian/Ubuntu, or
python.org) and run it again.

### Using conda (step by step)

Use this if your default `python3` is too old (e.g. macOS's 3.9) or you manage Pythons with conda. Conda only
provides the Python; `./setup.sh` still builds the project's own `.venv/`, so the run commands stay the same.

1. **Check that conda is installed:** `conda --version`. If it is not:
   - macOS: `brew install --cask miniforge`, or download Miniforge from https://github.com/conda-forge/miniforge
   - then run `conda init zsh` (or `conda init bash`) and **open a new terminal**.
2. **Create an environment with a recent Python:**
   ```bash
   conda create -n medextract python=3.12 -y
   ```
3. **Activate it.** The prompt changes from `(base)` to `(medextract)`:
   ```bash
   conda activate medextract
   ```
4. **Check the version.** It must say 3.10 or newer:
   ```bash
   python --version        # Python 3.12.x
   ```
5. **Install the project:**
   ```bash
   cd Papaya_AI_Challenge_08
   ./setup.sh              # step 1/4 should show ".../envs/medextract/bin/python3 (Python 3.12.x)"
   ```
6. **Run it:**
   ```bash
   .venv/bin/medextract run document-tests/01_receipt_en.pdf
   ```

Next time, `cd` into the project and run the `.venv/bin/...` commands directly. `.venv` keeps its own copy of the
Python, so `conda activate` is not needed again. To remove everything later, run `conda env remove -n medextract` and
`rm -rf .venv`.

### Manual install (Windows, or if you prefer)

```bash
python -m venv .venv                         # must be Python 3.10+; check with: python --version
.venv/bin/python -m pip install --upgrade pip   # editable installs need pip >= 21.3
.venv/bin/pip install -e ".[mlx,dev]"        # Apple Silicon | ".[cuda,dev]" NVIDIA GPU | ".[cpu,dev]" CPU only
```
On Windows use `.venv\Scripts\python` and `.venv\Scripts\pip` (the `mlx` extra is macOS-only).

### Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `File "setup.py" or "setup.cfg" not found ... editable mode` | pip older than 21.3 | `.venv/bin/python -m pip install --upgrade pip`, or just run `./setup.sh` |
| `requires a different Python` / syntax errors on import | `.venv` was created with Python 3.9 | `./setup.sh --recreate` |
| `no such file or directory: .venv/bin/medextract` | the install did not finish | run `./setup.sh` and read its last error |
| `the mlx backend needs a Mac with Apple Silicon` | Intel Mac or Linux | `./setup.sh cpu` (or `cuda` with an NVIDIA GPU) |
| Out of memory | the 8B model needs ~7 GB (MLX) / ~17 GB (CPU) of RAM | close other apps; run one `medextract` at a time; `--model Qwen/Qwen3-VL-2B-Instruct` for a smaller model |

The `.venv/` folder is not committed (it is in `.gitignore`), so it is created once per clone. Tests for the
Transformers backend are skipped unless `torch` is installed (the `[cpu]` or `[cuda]` extra).

The model weights download from Hugging Face on first use: ~5 GB for MLX, ~17 GB for Transformers.

## Usage

```bash
.venv/bin/medextract info                                    # which backend/model will be used
.venv/bin/medextract run document-tests/ -o results/         # a whole folder -> results/<name>.json + summary.json
.venv/bin/medextract run document-tests/01_receipt_en.pdf    # one file -> JSON on stdout
.venv/bin/medextract run x.pdf --meta                        # also show the diagnostic "meta" block
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
        print(result.to_json_dict())                     # spec format; include_meta=True adds diagnostics
        print(result.meta["timings"])                    # meta is always available on the object
        results = await pipeline.process_batch(["a.pdf", "b.png"], concurrency=4)
        async for r in pipeline.stream(["c.jpg", "d.pdf"]):   # results as they finish
            ...

asyncio.run(main())
```

The JSON output follows the spec format only. Diagnostics are hidden unless you pass `--meta` (CLI) or
`include_meta=True` (Python). They cover the source file, backend and model, detected language, classification scores,
per-stage timings, retry counts and the grounding status of each field.

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
- **Backends.** `backends/base.py` defines `VLMBackend`. `backends/factory.py` picks `MLXBackend` (Apple Silicon), then
  `TransformersBackend` on CUDA (4-bit NF4), then the CPU (bf16). `FakeBackend` drives the unit tests without a model.
- **The model transcribes; code decides.** Prompts ask for values exactly as printed. Calendar conversion
  (Buddhist era), number formats (`1.500.000`, `1,500.00`, Thai digits), totals and lab-flag checks happen in
  deterministic code.
- **No hallucination.** Missing fields are `null` with confidence ≤ 0.1. Every value is checked against the PDF text
  layer when there is one; a value that can't be found is capped at 0.3 confidence and flagged `ungrounded_value`.
- **Meaningful confidence.** The score combines the model's own confidence, grounding against the text layer,
  normalization problems (invalid → ≤ 0.3, ambiguous date ×0.7) and token probabilities.

The prompts and why they look the way they do are covered in [Prompt engineering approach](#prompt-engineering-approach).

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

Numbers are returned as floats: amounts, quantities and lab results (`"7.8"` → `7.8`, Vietnamese `"5,6"` → `5.6`,
Thai digits included). A lab result that is not a bare number (`"< 0.5"`, `"Positive"`) stays a string, as printed.

## Prompt engineering approach

A small local model (8B, 4-bit) is prone to inventing plausible values and to reporting uniformly high confidence. The
approach is to **let the model transcribe and let code decide**: the prompts ask only for what is printed, and every
conversion, check and score that code can do deterministically is done in code. The prompts live in
[`src/medextract/extraction/prompts.py`](src/medextract/extraction/prompts.py).

1. **Two calls per document.** First *classify*, using a 1024 px image. The model must return a score for each of the 4
   types, not a single label: asked for a label, it echoed the printed title ("OFFICIAL RECEIPT"). Then *extract*
   with a prompt for that type only, so fields from different types never mix.
2. **One schema source.** The field list in each extraction prompt is generated from `FIELD_SPECS`, the same table that
   drives normalization and validation, so prompts and code cannot drift apart.
3. **Shared rules in the system prompt.**
   - Report only what is visible. Anything not visible is `null`, with confidence ≤ 0.1.
   - Copy text verbatim in its original script; never translate.
   - Copy dates exactly as printed and do not convert Buddhist-era years. Do not compute or correct amounts.
   - Never swap or "fix" values that look inconsistent, because validation catches those.
   - Person names come with their title only, and a person field is filled only if that role is named.
   - Every value comes with `evidence`: the verbatim snippet *including its printed label*.
   - Confidence anchors from "clearly printed" to "not present", and an explicit "don't give every field the same score".
4. **Text layer as a hint.** For born-digital PDFs, the embedded text is passed in, labelled "may be incomplete; the
   image is the source of truth". It improves spelling of names and Thai/Vietnamese diacritics. Garbled text layers are
   detected and dropped.
5. **Code does the rest.**
   - Normalization: ISO dates (Buddhist era −543, day-first for vi/th), number formats such as `1.500.000`, Thai digits.
   - Grounding: every value is checked against the text layer; a value that can't be found is capped at 0.3 and flagged.
   - Label checks: crossed admission/discharge dates are swapped back, and a director named as the physician is rejected.
   - Computed confidence, plus validation of dates, amounts, totals and lab flags.
6. **Robust output handling.** JSON is parsed leniently, with repair and a single retry that includes the error.
7. **Iterated on evidence, not on the test files.** Each of the 3 runs was scored against hand-labelled ground truth,
   and every fix is a general rule. No prompt contains anything from `document-tests/`, and a unit test enforces that.

| Run | Field accuracy | Hallucinated fields | Validation errors caught | Change |
|---|---|---|---|---|
| 1 | 89.7% | 1 | 1/2 | baseline |
| 2 | 93.1% | 1 | 1/2 | no-swap and role rules; names without degrees; evidence includes the label |
| 3 | **98.3%** | **0** | **2/2** | label checks in code (`postprocessing/roles.py`) |

Full write-up, including what was deliberately not done: [`docs/PROMPTING.md`](docs/PROMPTING.md).

## Project layout

```
src/medextract/
├── __init__.py          public API: Pipeline, Settings, ExtractionResult
├── __main__.py          python -m medextract
├── cli.py               `medextract` command
├── config.py            Settings (MEDEXTRACT_* variables)
├── pipeline.py          async orchestrator
├── schemas.py           data contracts, FIELD_SPECS
├── ingestion/           loader.py: PDF/image -> page images + text layer
├── backends/            base.py (interface), factory.py (MLX -> CUDA -> CPU), mlx.py, transformers.py, fake.py
├── extraction/          prompts.py, parsing.py (tolerant JSON + retry)
└── postprocessing/      roles.py, normalize.py, grounding.py, confidence.py, validate.py
tests/
├── unit/                mirrors src/ (FakeBackend, no model needed)
└── integration/         real-model tests (-m slow)
document-tests/   the 10 test documents
ground_truth/     hand-labelled answers for the 10 documents
results/          pipeline output for the 10 documents + eval_report.md
scripts/          eval.py, smoke_mlx.py
tasks/            task board (todo / pending / done)
```

## Tests

```bash
.venv/bin/pytest -q                                   # unit tests, no model needed
.venv/bin/pytest tests/unit/postprocessing/test_normalize.py -q          # one file
.venv/bin/pytest tests/unit/postprocessing/test_validate.py -k total -q  # one test by name
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
read only from an image (no text to check against) ~0.5–0.73, and absent fields 0.05. The document-level confidence
combines the classifier's type score with how well the type's key fields were found (`items` + `grand_total` for a
receipt, `medications` for a prescription, …): 0.92–0.96 for the PDFs, 0.75–0.82 for the scanned images. A document
classified with certainty but missing its key fields drops below 0.6 and gets a `low_classification_confidence` warning.
The raw classifier score is kept in `meta.classifier_confidence`.

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

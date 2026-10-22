# Prompt engineering approach

The extractor runs a small local vision-language model: **Qwen3-VL-8B-Instruct, 4-bit**. A small model is more likely to
drift, invent plausible values and report high confidence for everything. So the design gives the model as little
to decide as possible and moves every decision that code can make out of the prompt.

## 1. Two calls per document: classify, then extract

| Call | Image | Output | Why |
|---|---|---|---|
| Classify | page downscaled to 1024 px (~600 tokens) | `{"scores": {4 types}, "language", "reason"}` | Layout is enough to tell the type. A smaller image makes prefill ~3x cheaper. |
| Extract | page at 1600 px (~1800 tokens) | `{"fields": {name: {"value", "confidence", "evidence"}}}` | One prompt per type with only that type's fields, so the model never mixes schemas. |

Classification asks for a **score for every type** instead of one label. The smoke test showed why: asked for a single
label, the model answered with the printed title ("OFFICIAL RECEIPT"). Scores turn into a classifier confidence in
code: `p1 × (0.5 + 0.5 × (p1 − p2))`, so a near tie gives a low confidence even when the top score is high. On clear
documents the model is near-certain (1.0), so the final document confidence is also checked against the content:
`classifier × (0.5 + 0.5 × support)`, where `support` is the mean confidence of the type's key fields (e.g. `items` and
`grand_total` for a receipt). A "receipt" with no line items or total is not believable, whatever the classifier said.
The prompt defines each type by purpose and lists the cues that tell them apart. It also says a document can mention
medicines or prices without being a prescription or a receipt; discharge summaries often list discharge medication.

## 2. One source of truth for the schema

The per-type field list in the extraction prompt is **generated from `FIELD_SPECS`** in `schemas.py`, the same table that
drives normalization and validation. The list-item keys come from the Pydantic item models. A field cannot exist in the
prompt but not in the code, or the other way round. Each field gets a one-line hint for its kind (text, date, amount,
list of rows, diagnosis object).

## 3. Shared rules live in the system prompt

`SYSTEM_PROMPT` holds every rule that applies to all calls, so the per-call prompts stay short:

1. Report only what is visible; never guess, infer or use typical values.
2. Not visible: `value: null`, confidence 0.0–0.1, `evidence: null`.
3. Copy text **verbatim, in the original script**: no translation, no transliteration (Vietnamese diacritics and
   Thai script are kept).
4. **Copy dates exactly as printed** and do not convert calendars.
5. **Do not calculate or correct amounts.** Report only printed numbers.
6. List rows only if they are printed.
7. `evidence` is the shortest verbatim snippet that supports the value.
8. Confidence anchors (0.95+ clear / 0.7–0.9 legible but small or interpreted / 0.3–0.6 hard to read or handwritten /
   ≤ 0.1 absent) and an explicit "do not give every field the same score".
9. JSON only.

Rules 4 and 5 matter most. Converting a Buddhist-era year, parsing `1.500.000` or totalling line items is
easy for code and error-prone for an 8B model. Asking the model to only transcribe keeps it in the job it does
best, and lets code apply the rules the same way every time:

- `normalize.py`: ISO dates (Buddhist era −543, day-first for vi/th, ambiguity flag for en), Vietnamese/English/Thai
  number formats, Thai digits, lab flags in three languages.
- `validate.py`: valid dates, discharge ≥ admission, positive amounts, quantity × unit price, line items vs grand total
  (> 5% → `total_mismatch`), lab flags vs reference range.

## 4. Text layer as a reading aid, not a source

Born-digital PDFs carry an embedded text layer. It is passed to the model as a hint, truncated to 6000 characters. The
hint is labelled *machine-extracted, may be incomplete or out of order; the image is the source of truth; do not add
fields from it*. It helps the model spell names and read small digits correctly, especially Thai and Vietnamese
diacritics. A garbled text layer (replacement characters, `(cid:N)` tokens, mostly symbols) is detected and not used.
Scanned images have no text layer and rely on the image alone.

## 5. Confidence is computed, not trusted

The model's self-reported confidence is usually high, and on clean print almost every token has probability ~1.0
(measured in T02). Using either alone would give uniformly high scores. The final per-field confidence
(`confidence.py`) combines four signals:

1. **Model confidence** (default 0.7 if missing) as the starting point.
2. **Grounding** (`grounding.py`), the main signal. The value is searched for in the text layer: exact substring, fuzzy
   match ≥ 90, number token with the same value (`1.500.000` = `1,500,000` = Thai digits), or date parts (including the
   Buddhist-era year). A value that cannot be found is **capped at 0.3** and flagged `ungrounded_value`, which is how
   hallucinations surface. Images have no reference, so the score depends on whether the model's own `evidence` agrees
   with its value (0.6 consistent, 0.4 missing, 0.2 inconsistent).
3. **Normalization outcome**: unparseable or impossible values are capped at 0.3; ambiguous dates ×0.7.
4. **Token probability** of the value's tokens (min over its span), as a mild factor `0.5 + 0.5·p`.

Null values always get ≤ 0.1.

## 6. Labels as a cross-check (`roles.py`)

Evaluation showed the model can read a value correctly but attach it to the wrong field. On a Thai image it misread
the discharge label (จำหน่าย) and swapped admission and discharge. On a Vietnamese PDF it reported the hospital
director, the only named signer, as the attending physician. Two prompt rules (never swap or correct values; fill
a person field only for that role) were not enough on their own. So the prompt now asks for `evidence` **including the
printed label**, and code checks it with multilingual role keywords:

- If a date's evidence carries the other field's label (admission vs discharge) and neither carries its own,
  the two dates are swapped back, with a `fields_swapped_by_label` warning.
- If the label next to a person's name (in the evidence, or up to 3 text-layer lines above the name in a signature
  block) is a different role (director, pharmacist, cashier), the field becomes `null`, with a `role_mismatch`
  warning that names the rejected value.

These rules are keyed on labels, not on any test document, and every change shows up in `validation_errors`.

## 7. Robust JSON

`parsing.py` strips `<think>` blocks, code fences and prose. It decodes the first JSON object, falls back to
`json_repair` for truncated or slightly malformed output, and validates with Pydantic. If that still fails, it retries
**once** with the error appended ("Your previous answer was invalid: … Return ONLY valid JSON"). Timeouts are not
retried.

## 8. What was deliberately not done

- **No few-shot examples taken from the test documents.** They would bias the model towards those layouts and break the
  "reusable pipeline" requirement. Prompts contain no names, hospitals or values from `document-tests/`, and a test
  enforces this.
- **No single "extract everything" call.** A combined schema for four types roughly doubles the prompt and invites
  fields from the wrong type.
- **No translation.** Values stay verbatim; downstream systems can translate if needed.

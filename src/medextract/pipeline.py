"""Async extraction pipeline: document file -> classified, extracted, scored, validated JSON.

Per document the stages are::

    ingest (worker thread) -> classify (VLM) -> extract (VLM) -> post-process -> result

``process_batch`` keeps up to ``concurrency`` documents in flight. Model calls are
serialized by the backend's single worker thread, so while one document occupies the
accelerator, the others are being loaded, rendered or post-processed. A failure in one
document never stops the batch: that document gets a result whose ``validation_errors``
explain what went wrong.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from medextract.backends.base import GenerationResult, VLMBackend, VLMError
from medextract.config import Settings
from medextract.extraction.parsing import JSONParseError, generate_structured
from medextract.extraction.prompts import SYSTEM_PROMPT, build_classify_prompt, build_extract_prompt
from medextract.ingestion.loader import DocumentInput, IngestError, aload_document
from medextract.postprocessing.confidence import (
    classification_confidence,
    document_confidence,
    field_confidence,
    find_value_span,
    span_token_prob,
)
from medextract.postprocessing.grounding import ground_field, text_layer_usable
from medextract.postprocessing.normalize import Normalized, detect_language, normalize_value
from medextract.postprocessing.roles import check_person_roles, reconcile_date_labels
from medextract.postprocessing.validate import compute_lab_flag, validate_fields
from medextract.schemas import (
    FIELD_SPECS,
    DocumentType,
    ExtractionResult,
    FieldValue,
    RawClassification,
    RawExtraction,
    RawField,
    ValidationIssue,
)

logger = logging.getLogger("medextract")

# Classification needs the layout, not small print, so it uses a smaller image (fewer tokens).
CLASSIFY_MAX_SIDE = 1024
CLASSIFY_MAX_TOKENS = 256
# Below this classification confidence a warning is attached to the result.
LOW_CLASSIFICATION_CONFIDENCE = 0.6

ResultCallback = Callable[[ExtractionResult], Awaitable[None] | None]


class Pipeline:
    """Runs documents through a ``VLMBackend``. Use as an async context manager."""

    def __init__(self, backend: VLMBackend, settings: Settings | None = None) -> None:
        self.backend = backend
        self.settings = settings or Settings()

    @classmethod
    def from_env(cls, **overrides: Any) -> Pipeline:
        """Build settings from ``MEDEXTRACT_*`` variables and pick the best available backend."""
        from medextract.backends.factory import create_backend

        settings = Settings.from_env(**overrides)
        return cls(create_backend(settings), settings)

    async def __aenter__(self) -> Pipeline:
        await self.backend.load()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.backend.close()

    # --- public API --------------------------------------------------------------------

    async def process_document(self, path: str | Path) -> ExtractionResult:
        """Process one document. Never raises for document-level problems."""
        path = Path(path)
        timings: dict[str, float] = {}
        meta: dict[str, Any] = {"source": path.name, **self.backend.describe()}
        started = time.perf_counter()
        try:
            result = await self._process(path, timings, meta)
        except IngestError as exc:
            result = _failure(meta, "ingest_error", f"Could not read the document: {exc}")
        except (asyncio.TimeoutError, TimeoutError):
            result = _failure(meta, "timeout", f"Model call exceeded {self.settings.timeout_s:.0f}s")
        except JSONParseError as exc:
            result = _failure(meta, "invalid_model_output", str(exc))
        except VLMError as exc:
            result = _failure(meta, "model_error", str(exc))
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop a batch
            logger.exception("Unexpected error while processing %s", path.name)
            result = _failure(meta, "pipeline_error", f"{type(exc).__name__}: {exc}")
        timings["total_s"] = round(time.perf_counter() - started, 2)
        # The result holds its own copy of meta, so attach the final timings to it directly.
        result.meta = {**(result.meta or {}), "timings": timings}
        logger.info("%s: %s in %.1fs", path.name, result.document_type, timings["total_s"])
        return result

    async def process_batch(
        self,
        paths: Iterable[str | Path],
        *,
        concurrency: int | None = None,
        on_result: ResultCallback | None = None,
    ) -> list[ExtractionResult]:
        """Process many documents concurrently. Results keep the input order."""
        paths = list(paths)
        results: list[ExtractionResult | None] = [None] * len(paths)
        async for index, result in self._iter_results(paths, concurrency):
            results[index] = result
            if on_result is not None:
                maybe = on_result(result)
                if asyncio.iscoroutine(maybe):
                    await maybe
        return [r for r in results if r is not None]

    async def stream(
        self, paths: Iterable[str | Path], *, concurrency: int | None = None
    ) -> AsyncIterator[ExtractionResult]:
        """Yield results as documents finish (completion order)."""
        async for _, result in self._iter_results(list(paths), concurrency):
            yield result

    # --- internals ---------------------------------------------------------------------

    async def _iter_results(
        self, paths: list[str | Path], concurrency: int | None
    ) -> AsyncIterator[tuple[int, ExtractionResult]]:
        limit = asyncio.Semaphore(max(1, concurrency or self.settings.concurrency))

        async def run(index: int, path: str | Path) -> tuple[int, ExtractionResult]:
            async with limit:  # bounds memory: at most `concurrency` documents loaded at once
                return index, await self.process_document(path)

        tasks = [asyncio.create_task(run(i, p)) for i, p in enumerate(paths)]
        try:
            for next_done in asyncio.as_completed(tasks):
                yield await next_done
        finally:
            for task in tasks:
                task.cancel()

    async def _process(self, path: Path, timings: dict[str, float], meta: dict[str, Any]) -> ExtractionResult:
        s = self.settings

        t = time.perf_counter()
        doc = await aload_document(path, max_side=s.max_image_side, dpi=s.pdf_dpi)
        timings["ingest_s"] = round(time.perf_counter() - t, 2)
        text_layer = doc.text_layer if text_layer_usable(doc.text_layer) else None
        meta.update(pages=doc.page_count, has_text_layer=text_layer is not None)

        # 1. Classify.
        t = time.perf_counter()
        classification, _, attempts = await generate_structured(
            self.backend,
            _shrink(doc.images, CLASSIFY_MAX_SIDE),
            build_classify_prompt(text_layer),
            RawClassification,
            system=SYSTEM_PROMPT,
            max_tokens=CLASSIFY_MAX_TOKENS,
            timeout_s=s.timeout_s,
        )
        timings["classify_s"] = round(time.perf_counter() - t, 2)
        doc_type, type_confidence = classification_confidence(classification.scores)
        language = _language(classification.language, text_layer)
        meta.update(classification_scores=classification.scores, language=language, classify_attempts=attempts)
        if doc_type is None:
            return _failure(meta, "unclassified", "The model did not assign any known document type")

        # 2. Extract.
        t = time.perf_counter()
        extraction, generation, attempts = await generate_structured(
            self.backend,
            doc.images,
            build_extract_prompt(doc_type, language=language, text_hint=text_layer),
            RawExtraction,
            system=SYSTEM_PROMPT,
            max_tokens=s.max_tokens,
            timeout_s=s.timeout_s,
        )
        timings["extract_s"] = round(time.perf_counter() - t, 2)
        meta.update(extract_attempts=attempts, generated_tokens=generation.generation_tokens)

        # 3. Normalize, ground, score, validate.
        t = time.perf_counter()
        result = assemble_result(
            doc,
            doc_type,
            type_confidence,
            extraction,
            generation,
            language=language,
            text_layer=text_layer,
            meta=meta,
        )
        timings["postprocess_s"] = round(time.perf_counter() - t, 3)
        return result


def assemble_result(
    doc: DocumentInput,
    doc_type: DocumentType,
    type_confidence: float,
    extraction: RawExtraction,
    generation: GenerationResult,
    *,
    language: str | None,
    text_layer: str | None,
    meta: dict[str, Any],
) -> ExtractionResult:
    """Turn the raw model extraction into the final, scored and validated result."""
    fields: dict[str, FieldValue] = {}
    normalization: dict[str, Normalized] = {}
    grounding_report: dict[str, str] = {}

    # Fix values the model attached to the wrong field, using the printed labels.
    raw_fields = dict(extraction.fields)
    issues: list[ValidationIssue] = reconcile_date_labels(raw_fields) + check_person_roles(raw_fields, text_layer)

    for name, kind in FIELD_SPECS[doc_type].items():
        raw = raw_fields.get(name) or RawField()
        norm = normalize_value(kind, raw.value, language)
        normalization[name] = norm
        grounding = ground_field(kind, norm.value, raw.evidence, text_layer)
        grounding_report[name] = grounding.status

        span = find_value_span(generation.text, name)
        token_prob = span_token_prob(generation, *span) if span else None
        confidence = field_confidence(
            kind=kind,
            value=norm.value,
            model_confidence=raw.confidence,
            grounding=grounding,
            normalized_valid=norm.valid,
            ambiguous=norm.ambiguous,
            token_prob=token_prob,
        )
        fields[name] = FieldValue(value=norm.value, confidence=confidence)

        if grounding.status == "not_found" and not _is_empty(norm.value):
            issues.append(
                ValidationIssue(
                    field=name,
                    code="ungrounded_value",
                    message="Value was not found in the document's text layer; it may be misread or invented",
                    severity="warning",
                )
            )

    issues = validate_fields(doc_type, fields, normalization) + issues
    if doc_type == "lab_report":
        _fill_missing_lab_flags(fields)

    confidence = document_confidence(type_confidence, doc_type, {n: f.confidence for n, f in fields.items()})
    if confidence < LOW_CLASSIFICATION_CONFIDENCE:
        issues.append(
            ValidationIssue(
                code="low_classification_confidence",
                message=(
                    f"Document type '{doc_type}' assigned with low confidence ({confidence:.2f}): "
                    "the classifier was unsure or the type's key fields were not found"
                ),
                severity="warning",
            )
        )

    meta["grounding"] = grounding_report
    meta["classifier_confidence"] = type_confidence
    return ExtractionResult(
        document_type=doc_type,
        confidence=confidence,
        fields=fields,
        validation_errors=issues,
        meta=meta,
    )


def _fill_missing_lab_flags(fields: dict[str, FieldValue]) -> None:
    """Derive normal/high/low from the printed reference range when no flag is printed.

    This runs after validation, so ``lab_flag_mismatch`` only compares printed flags.
    Each row records where its flag came from in ``flag_source`` ("printed" or "derived").
    """
    tests = fields.get("tests")
    if tests is None or not isinstance(tests.value, list):
        return
    for row in tests.value:
        if not isinstance(row, dict):
            continue
        if row.get("flag") is not None:
            row["flag_source"] = "printed"
            continue
        derived = compute_lab_flag(row.get("result"), row.get("reference_range"))
        if derived is not None:
            row["flag"] = derived
            row["flag_source"] = "derived"


def _failure(meta: dict[str, Any], code: str, message: str) -> ExtractionResult:
    logger.warning("%s: %s - %s", meta.get("source"), code, message)
    return ExtractionResult(
        document_type=None,
        confidence=0.0,
        fields={},
        validation_errors=[ValidationIssue(code=code, message=message)],
        meta=meta,
    )


def _language(model_language: str | None, text_layer: str | None) -> str | None:
    from_text = detect_language(text_layer) if text_layer else None
    if from_text:
        return from_text  # the text layer is more reliable than the model's guess
    if model_language in ("en", "vi", "th"):
        return model_language
    return None


def _shrink(images: list[Image.Image], max_side: int) -> list[Image.Image]:
    shrunk = []
    for image in images:
        if max(image.size) > max_side:
            image = image.copy()
            image.thumbnail((max_side, max_side), Image.LANCZOS)
        shrunk.append(image)
    return shrunk


def _is_empty(value: Any) -> bool:
    return value is None or value == [] or value == "" or value == {}

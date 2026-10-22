"""Command line interface.

medextract run <file-or-dir>... [-o results/] [--concurrency N] [--device auto|mlx|cuda|cpu]
medextract info
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from medextract.config import Settings
from medextract.schemas import ExtractionResult

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

logger = logging.getLogger("medextract")


def collect_paths(inputs: list[str]) -> list[Path]:
    """Expand directories into their supported files (sorted); keep explicit files as given."""
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(
                sorted(
                    p
                    for p in path.iterdir()
                    if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_SUFFIXES
                )
            )
        else:
            paths.append(path)  # missing files are reported per document by the pipeline
    return paths


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summary_row(result: ExtractionResult) -> dict:
    meta = result.meta or {}
    errors = [i for i in result.validation_errors if i.severity == "error"]
    warnings = [i for i in result.validation_errors if i.severity == "warning"]
    return {
        "source": meta.get("source"),
        "document_type": result.document_type,
        "confidence": result.confidence,
        "errors": [i.code for i in errors],
        "warnings": [i.code for i in warnings],
        "seconds": (meta.get("timings") or {}).get("total_s"),
    }


def _print_summary(rows: list[dict]) -> None:
    print(f"\n{'source':<32} {'type':<18} {'conf':>5} {'time':>6}  issues", file=sys.stderr)
    for row in rows:
        issues = ", ".join(row["errors"] + [f"{w} (warn)" for w in row["warnings"]]) or "-"
        seconds = f"{row['seconds']:.1f}s" if row["seconds"] is not None else "-"
        print(
            f"{row['source'] or '?':<32} {row['document_type'] or 'FAILED':<18} "
            f"{row['confidence']:>5.2f} {seconds:>6}  {issues}",
            file=sys.stderr,
        )


async def _run(args: argparse.Namespace) -> int:
    from medextract.pipeline import Pipeline

    paths = collect_paths(args.inputs)
    if not paths:
        print("No supported documents found.", file=sys.stderr)
        return 2

    out_dir = Path(args.output) if args.output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    include_meta = args.meta

    def save(result: ExtractionResult) -> None:
        if out_dir is None:
            return
        source = (result.meta or {}).get("source") or "document"
        _write_json(out_dir / f"{Path(source).stem}.json", result.to_json_dict(include_meta=include_meta))

    pipeline = Pipeline.from_env(device=args.device, model_id=args.model, concurrency=args.concurrency)
    logger.info("Backend: %s (%s), %d document(s)", pipeline.backend.name, pipeline.backend.model_id, len(paths))
    async with pipeline:
        results = await pipeline.process_batch(paths, on_result=save)

    rows = [_summary_row(r) for r in results]
    if out_dir:
        _write_json(out_dir / "summary.json", rows)
    else:
        payload = [r.to_json_dict(include_meta=include_meta) for r in results]
        print(json.dumps(payload[0] if len(payload) == 1 else payload, ensure_ascii=False, indent=2))
    _print_summary(rows)
    return 1 if any(r.document_type is None for r in results) else 0


def _info(args: argparse.Namespace) -> int:
    from medextract.backends.factory import cuda_available, detect_backend, mlx_available, resolve_model_id

    settings = Settings.from_env(device=args.device, model_id=args.model)
    kind = detect_backend(settings.device)
    info = {
        "selected_backend": kind,
        "model_id": resolve_model_id(kind, settings.model_id),
        "mlx_available": mlx_available(),
        "cuda_available": cuda_available(),
        "settings": dict(vars(settings)),
    }
    print(json.dumps(info, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medextract", description="Extract structured data from medical documents.")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", choices=["auto", "mlx", "cuda", "cpu"], help="override MEDEXTRACT_DEVICE")
    common.add_argument("--model", help="override MEDEXTRACT_MODEL_ID")

    run = sub.add_parser("run", parents=[common], help="process files or directories")
    run.add_argument("inputs", nargs="+", help="document files and/or directories")
    run.add_argument("-o", "--output", help="directory for <name>.json results and summary.json")
    run.add_argument("--concurrency", type=int, help="documents in flight (default 4)")
    run.add_argument(
        "--meta",
        action="store_true",
        help="include the diagnostic 'meta' block (source, backend, timings, grounding); hidden by default",
    )

    sub.add_parser("info", parents=[common], help="show the detected backend and settings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    try:
        if args.command == "info":
            return _info(args)
        return asyncio.run(_run(args))
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

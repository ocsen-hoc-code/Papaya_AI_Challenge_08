"""Compare pipeline results with hand-labelled ground truth.

Usage: .venv/bin/python scripts/eval.py [--results results/] [--truth ground_truth/]

Reports classification accuracy, per-field accuracy, hallucinations (value returned where
the truth is null), whether confidence separates right from wrong fields, and whether the
validation issues that the ground truth implies were raised. Writes
``<results>/eval_report.md`` and ``<results>/eval_report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from rapidfuzz import fuzz

from medextract.postprocessing.normalize import normalize_number
from medextract.postprocessing.validate import compute_lab_flag, validate_fields
from medextract.schemas import FIELD_SPECS, FieldKind, FieldValue

TEXT_MATCH = 90  # rapidfuzz ratio (0-100) needed for a text value to count as correct
FIELD_CORRECT = 0.9  # a structured field counts as correct at this score or above
ROW_KEYS = {
    FieldKind.RECEIPT_ITEMS: "description",
    FieldKind.LAB_TESTS: "test_name",
    FieldKind.MEDICATIONS: "name",
}
WRONG_OUTCOMES = ("wrong", "partial", "missed", "hallucinated")


@dataclass
class FieldScore:
    source: str
    field: str
    kind: str
    score: float  # 0..1
    outcome: str  # correct | partial | wrong | missed | hallucinated | correct_null
    confidence: float | None
    truth: Any
    predicted: Any


# --- value comparison ------------------------------------------------------------------


def _norm_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value)).casefold()
    return " ".join(text.split())


def text_score(truth: Any, pred: Any) -> float:
    if truth is None or pred is None:
        return 1.0 if truth is None and pred is None else 0.0
    return fuzz.ratio(_norm_text(truth), _norm_text(pred)) / 100.0


def number_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return normalize_number(value).value if value is not None else None


def scalar_score(truth: Any, pred: Any) -> float:
    """Numbers compare numerically (1% tolerance); everything else fuzzily as text."""
    if truth is None and pred is None:
        return 1.0
    if truth is None or pred is None:
        return 0.0
    t, p = number_or_none(truth), number_or_none(pred)
    if t is not None and p is not None and not (isinstance(truth, str) and isinstance(pred, str)):
        return 1.0 if abs(t - p) <= 0.01 * max(1.0, abs(t)) else 0.0
    score = text_score(truth, pred)
    return 1.0 if score * 100 >= TEXT_MATCH else score


def list_score(truth: list, pred: list, row_score) -> float:
    """Greedy one-to-one alignment; F1-style so missing and extra rows both cost."""
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0
    remaining = list(range(len(pred)))
    matched = 0.0
    for t_row in truth:
        if not remaining:
            break
        best = max(remaining, key=lambda j: row_score(t_row, pred[j]))
        matched += row_score(t_row, pred[best])
        remaining.remove(best)
    precision, recall = matched / len(pred), matched / len(truth)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def expected_lab_flag(row: dict) -> str | None:
    """Truth flag: as printed, else derived from the printed range (the pipeline derives too)."""
    return row.get("flag") or compute_lab_flag(row.get("result"), row.get("reference_range"))


def make_row_scorer(kind: FieldKind):
    key = ROW_KEYS[kind]

    def score(t_row: dict, p_row: dict) -> float:
        if not isinstance(p_row, dict):
            return 0.0
        if text_score(t_row.get(key), p_row.get(key)) < 0.6:
            return 0.0  # not the same row
        parts = []
        for sub, t_val in t_row.items():
            if kind is FieldKind.LAB_TESTS and sub == "flag":
                parts.append(1.0 if expected_lab_flag(t_row) == p_row.get("flag") else 0.0)
            else:
                parts.append(scalar_score(t_val, p_row.get(sub)))
        return mean(parts)

    return score


def value_score(kind: FieldKind, truth: Any, pred: Any) -> float:
    if kind is FieldKind.DATE:
        return 1.0 if truth == pred else 0.0
    if kind is FieldKind.AMOUNT:
        return scalar_score(truth, pred)
    if kind is FieldKind.TEXT:
        score = text_score(truth, pred)
        return 1.0 if score * 100 >= TEXT_MATCH else score
    if kind is FieldKind.TEXT_LIST:
        return list_score(truth or [], pred or [], text_score)
    if kind is FieldKind.DIAGNOSIS:
        truth, pred = truth or {}, pred or {}
        primary = scalar_score(truth.get("primary"), pred.get("primary"))
        secondary = list_score(truth.get("secondary") or [], pred.get("secondary") or [], text_score)
        return 0.7 * primary + 0.3 * secondary
    if kind in ROW_KEYS:
        return list_score(truth or [], pred or [], make_row_scorer(kind))
    return scalar_score(truth, pred)


def is_empty(value: Any) -> bool:
    return value is None or value == [] or value == {} or value == ""


# --- evaluation ------------------------------------------------------------------------


def evaluate(results_dir: Path, truth_dir: Path) -> dict:
    docs, fields, validation = [], [], []
    for truth_path in sorted(truth_dir.glob("*.json")):
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        result_path = results_dir / f"{truth_path.stem}.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        source = truth.get("source", truth_path.stem)
        doc_type = truth["document_type"]
        predicted_type = result.get("document_type") if result else None
        docs.append(
            {
                "source": source,
                "truth_type": doc_type,
                "predicted_type": predicted_type,
                "type_correct": predicted_type == doc_type,
                "type_confidence": result.get("confidence") if result else None,
            }
        )

        pred_fields = (result or {}).get("fields", {}) if predicted_type == doc_type else {}
        for name, kind in FIELD_SPECS[doc_type].items():
            t_val = truth["fields"].get(name)
            p = pred_fields.get(name) or {}
            p_val, conf = p.get("value"), p.get("confidence")
            if is_empty(t_val):
                score = 1.0 if is_empty(p_val) else 0.0
                outcome = "correct_null" if score else "hallucinated"
            elif is_empty(p_val):
                score, outcome = 0.0, "missed"
            else:
                score = value_score(kind, t_val, p_val)
                outcome = "correct" if score >= FIELD_CORRECT else ("partial" if score >= 0.5 else "wrong")
            fields.append(FieldScore(source, name, kind.value, round(score, 3), outcome, conf, t_val, p_val))

        # Validation: errors the truth implies vs. errors the pipeline raised.
        truth_values = {n: FieldValue(value=v, confidence=1.0) for n, v in truth["fields"].items()}
        expected = {i.code for i in validate_fields(doc_type, truth_values) if i.severity == "error"}
        raised = {i["code"] for i in (result or {}).get("validation_errors", []) if i.get("severity") == "error"}
        validation.append(
            {
                "source": source,
                "expected": sorted(expected),
                "raised": sorted(raised),
                "caught": sorted(expected & raised),
                "missed": sorted(expected - raised),
                "extra": sorted(raised - expected),
            }
        )
    return {"documents": docs, "fields": [asdict(f) for f in fields], "validation": validation}


def summarize(report: dict) -> dict:
    docs, fields, validation = report["documents"], report["fields"], report["validation"]
    scored = [f for f in fields if f["outcome"] != "correct_null"]
    right = [
        f["confidence"] for f in fields if f["outcome"] in ("correct", "correct_null") and f["confidence"] is not None
    ]
    wrong = [f["confidence"] for f in fields if f["outcome"] in WRONG_OUTCOMES and f["confidence"] is not None]
    confidences = [f["confidence"] for f in fields if f["confidence"] is not None]
    return {
        "documents": len(docs),
        "classification_accuracy": round(mean(d["type_correct"] for d in docs), 3) if docs else 0.0,
        "field_accuracy": round(mean(f["outcome"] in ("correct", "correct_null") for f in fields), 3),
        "mean_field_score": round(mean(f["score"] for f in fields), 3),
        "non_null_field_accuracy": round(mean(f["outcome"] == "correct" for f in scored), 3) if scored else None,
        "hallucinated_fields": sum(f["outcome"] == "hallucinated" for f in fields),
        "truth_null_fields": sum(is_empty(f["truth"]) for f in fields),
        "missed_fields": sum(f["outcome"] == "missed" for f in fields),
        "mean_confidence_correct": round(mean(right), 3) if right else None,
        "mean_confidence_incorrect": round(mean(wrong), 3) if wrong else None,
        "confidence_spread": round(max(confidences) - min(confidences), 3) if confidences else None,
        "distinct_confidence_values": len({round(c, 2) for c in confidences}),
        "validation_expected": sum(len(v["expected"]) for v in validation),
        "validation_caught": sum(len(v["caught"]) for v in validation),
        "validation_extra": sum(len(v["extra"]) for v in validation),
    }


def to_markdown(report: dict, summary: dict) -> str:
    lines = ["# Evaluation report", "", "## Summary", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary.items()]
    lines += ["", "## Classification", "", "| Document | Truth | Predicted | Confidence |", "|---|---|---|---|"]
    for d in report["documents"]:
        mark = "✅" if d["type_correct"] else "❌"
        lines.append(f"| {d['source']} | {d['truth_type']} | {mark} {d['predicted_type']} | {d['type_confidence']} |")
    lines += ["", "## Fields", "", "| Document | Field | Outcome | Score | Confidence |", "|---|---|---|---|---|"]
    for f in report["fields"]:
        lines.append(f"| {f['source']} | {f['field']} | {f['outcome']} | {f['score']} | {f['confidence']} |")
    lines += [
        "",
        "## Validation",
        "",
        "| Document | Expected errors | Raised errors | Missed | Extra |",
        "|---|---|---|---|---|",
    ]
    for v in report["validation"]:
        cells = [", ".join(v[k]) or "-" for k in ("expected", "raised", "missed", "extra")]
        lines.append(f"| {v['source']} | " + " | ".join(cells) + " |")
    problems = [f for f in report["fields"] if f["outcome"] in WRONG_OUTCOMES]
    if problems:
        lines += ["", "## Field problems (truth vs predicted)", ""]
        for f in problems:
            truth = json.dumps(f["truth"], ensure_ascii=False)[:300]
            pred = json.dumps(f["predicted"], ensure_ascii=False)[:300]
            lines.append(f"- **{f['source']} / {f['field']}** ({f['outcome']}, conf {f['confidence']})")
            lines += [f"  - truth: `{truth}`", f"  - predicted: `{pred}`"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results")
    parser.add_argument("--truth", default="ground_truth")
    args = parser.parse_args(argv)
    results_dir, truth_dir = Path(args.results), Path(args.truth)
    if not any(truth_dir.glob("*.json")):
        print(f"No ground truth files in {truth_dir}", file=sys.stderr)
        return 2

    results_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate(results_dir, truth_dir)
    summary = summarize(report)
    (results_dir / "eval_report.json").write_text(
        json.dumps({"summary": summary, **report}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "eval_report.md").write_text(to_markdown(report, summary), encoding="utf-8")
    width = max(len(k) for k in summary)
    for key, value in summary.items():
        print(f"{key:<{width}}  {value}")
    print(f"\nWrote {results_dir / 'eval_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

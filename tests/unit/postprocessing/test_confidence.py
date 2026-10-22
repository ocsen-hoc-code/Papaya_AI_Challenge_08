import math

import pytest

from medextract.backends.base import GenerationResult
from medextract.postprocessing.confidence import (
    classification_confidence,
    field_confidence,
    find_value_span,
    span_token_prob,
)
from medextract.postprocessing.grounding import Grounding
from medextract.schemas import FieldKind


def _conf(status: str, score: float = 0.0, **kwargs) -> float:
    base = dict(
        kind=FieldKind.TEXT,
        value="Bangkok Hospital",
        model_confidence=0.95,
        grounding=Grounding(status, score),
    )
    base.update(kwargs)
    return field_confidence(**base)


def test_certain_grounded_field_beats_fuzzy_beats_image_beats_not_found_beats_null():
    certain = _conf("exact", 1.0)
    fuzzy = _conf("fuzzy", 0.93)
    image_only = _conf("no_reference", 0.4)
    not_found = _conf("not_found", 0.1)
    null = _conf("no_reference", 0.0, value=None)
    assert certain > fuzzy > image_only > not_found > null


def test_confidence_differs_between_certain_and_uncertain_fields():
    certain = _conf("exact", 1.0)
    uncertain = _conf("partial", 0.5, model_confidence=0.95)
    assert certain - uncertain >= 0.15
    assert certain >= 0.9
    assert uncertain <= 0.75


def test_not_found_is_capped_even_with_high_model_confidence():
    assert _conf("not_found", 0.1, model_confidence=1.0) <= 0.3


def test_null_value_is_low_regardless_of_model_confidence():
    assert _conf("exact", 1.0, value=None, model_confidence=0.99) <= 0.1
    assert _conf("exact", 1.0, value=[], model_confidence=None) <= 0.1
    assert _conf("exact", 1.0, value={"primary": None, "secondary": []}, model_confidence=None) <= 0.1


def test_invalid_normalization_is_capped():
    valid = _conf("exact", 1.0)
    invalid = _conf("exact", 1.0, normalized_valid=False)
    assert invalid <= 0.3
    assert invalid < valid


def test_ambiguous_normalization_lowers_confidence():
    plain = _conf("exact", 1.0, value="2024-03-04")
    ambiguous = _conf("exact", 1.0, value="2024-03-04", ambiguous=True)
    assert ambiguous < plain
    assert ambiguous == pytest.approx(plain * 0.7, abs=0.02)


def test_low_token_probability_lowers_confidence_mildly():
    base = _conf("exact", 1.0, token_prob=None)
    weak = _conf("exact", 1.0, token_prob=0.1)
    strong = _conf("exact", 1.0, token_prob=0.99)
    assert weak < strong <= base + 0.01
    assert weak > base * 0.5


def test_missing_model_confidence_defaults_to_moderate_value():
    value = _conf("partial", 0.6, model_confidence=None)
    assert value == pytest.approx(0.7 * 0.75, abs=0.01)


def test_result_is_always_within_unit_interval():
    assert 0.0 <= _conf("exact", 1.0, model_confidence=5.0) <= 1.0
    assert 0.0 <= _conf("exact", 1.0, model_confidence=-3.0) <= 1.0
    assert 0.0 <= _conf("not_found", 0.0, model_confidence=-3.0) <= 1.0


def test_exact_match_lifts_confidence_toward_ceiling():
    assert _conf("exact", 1.0, model_confidence=0.7) > 0.7
    assert _conf("exact", 1.0, model_confidence=0.7) <= 0.98


# --- Token probabilities ---------------------------------------------------------------


def _gen_with_pieces(text: str, pieces: list[str], probs: list[float]) -> GenerationResult:
    assert "".join(pieces) == text
    return GenerationResult(
        text=text,
        token_logprobs=[math.log(p) for p in probs],
        extra={"token_texts": pieces},
    )


_JSON = '{"date": {"value": "14 Mar 2024", "confidence": 0.9}}'
_PIECES = ['{"', "date", '": {"', "value", '": "', "14", " Mar", " 2024", '", "', "confidence", '": 0.9}}']
_PROBS = [0.99, 0.99, 0.99, 0.99, 0.99, 0.40, 0.90, 0.95, 0.99, 0.99, 0.99]


def test_span_token_prob_is_minimum_over_overlapping_tokens():
    gen = _gen_with_pieces(_JSON, _PIECES, _PROBS)
    span = find_value_span(_JSON, "date")
    assert span is not None
    assert _JSON[span[0] : span[1]] == "14 Mar 2024"
    assert span_token_prob(gen, *span) == pytest.approx(0.40, abs=1e-6)


def test_span_token_prob_ignores_tokens_outside_span():
    gen = _gen_with_pieces(_JSON, _PIECES, _PROBS)
    start = _JSON.index("confidence")
    assert span_token_prob(gen, start, start + len("confidence")) == pytest.approx(0.99, abs=1e-6)


def test_span_token_prob_none_without_token_texts():
    gen = GenerationResult(text=_JSON, token_logprobs=[math.log(p) for p in _PROBS])
    assert span_token_prob(gen, 0, 5) is None


def test_span_token_prob_none_without_logprobs_or_on_length_mismatch():
    assert span_token_prob(GenerationResult(text=_JSON, extra={"token_texts": _PIECES}), 0, 5) is None
    bad = GenerationResult(text=_JSON, token_logprobs=[0.0], extra={"token_texts": _PIECES})
    assert span_token_prob(bad, 0, 5) is None


def test_span_token_prob_none_for_empty_span():
    gen = _gen_with_pieces(_JSON, _PIECES, _PROBS)
    assert span_token_prob(gen, 5, 5) is None


# --- Value spans -----------------------------------------------------------------------


def test_find_value_span_returns_string_contents_without_quotes():
    text = '{"fields": {"date": {"value": "14 Mar 2024", "confidence": 0.9}}}'
    start, end = find_value_span(text, "date")
    assert text[start:end] == "14 Mar 2024"


def test_find_value_span_tolerates_whitespace_and_newlines():
    text = '{\n  "date" :{\n     "value" :\n   "2024-03-14"\n  }\n}'
    start, end = find_value_span(text, "date")
    assert text[start:end] == "2024-03-14"


def test_find_value_span_handles_numbers_and_null():
    text = '{"grand_total": {"value": 15750.0, "confidence": 0.9}, "payment_method": {"value": null}}'
    s, e = find_value_span(text, "grand_total")
    assert text[s:e] == "15750.0"
    s, e = find_value_span(text, "payment_method")
    assert text[s:e] == "null"


def test_find_value_span_covers_structured_values():
    text = '{"items": {"value": [{"description": "Room", "total": 1}], "confidence": 0.8}}'
    s, e = find_value_span(text, "items")
    assert text[s:e] == '[{"description": "Room", "total": 1}]'


def test_find_value_span_handles_escaped_quotes_in_strings():
    text = r'{"hospital_name": {"value": "St \"Mary\" Hospital", "confidence": 0.9}}'
    s, e = find_value_span(text, "hospital_name")
    assert text[s:e] == r"St \"Mary\" Hospital"


def test_find_value_span_is_none_for_missing_field():
    assert find_value_span('{"date": {"value": "x"}}', "patient_name") is None


def test_find_value_span_does_not_match_a_longer_key():
    text = '{"admission_date": {"value": "2024-01-01"}, "date": {"value": "2024-03-14"}}'
    s, e = find_value_span(text, "date")
    assert text[s:e] == "2024-03-14"


# --- Classification --------------------------------------------------------------------


def test_clear_winner_gets_high_confidence():
    doc_type, conf = classification_confidence(
        {"receipt": 0.95, "discharge_summary": 0.02, "lab_report": 0.02, "prescription": 0.01}
    )
    assert doc_type == "receipt"
    assert conf > 0.85


def test_near_tie_gets_low_confidence():
    doc_type, conf = classification_confidence({"receipt": 0.5, "prescription": 0.48})
    assert doc_type == "receipt"
    assert conf < 0.3


def test_near_tie_is_less_confident_than_clear_winner():
    _, tie = classification_confidence({"receipt": 0.5, "prescription": 0.48})
    _, clear = classification_confidence({"receipt": 0.9, "prescription": 0.05})
    assert clear > tie


def test_unknown_keys_are_ignored():
    assert classification_confidence({"invoice": 100.0, "receipt": 0.1}) == ("receipt", 1.0)


def test_negative_scores_are_clamped_to_zero():
    assert classification_confidence({"receipt": -5.0, "lab_report": 1.0}) == ("lab_report", 1.0)


def test_empty_or_zero_scores_give_no_type():
    assert classification_confidence({}) == (None, 0.0)
    assert classification_confidence({"receipt": 0.0}) == (None, 0.0)
    assert classification_confidence({"invoice": 3.0}) == (None, 0.0)


def test_document_confidence_corroborated_by_key_fields():
    from medextract.postprocessing.confidence import document_confidence

    full = document_confidence(1.0, "receipt", {"items": 0.95, "grand_total": 0.97})
    image = document_confidence(1.0, "receipt", {"items": 0.6, "grand_total": 0.65})
    missing = document_confidence(1.0, "receipt", {"items": 0.05, "grand_total": 0.05})
    assert full > image > missing
    assert full >= 0.95
    assert missing <= 0.55
    assert document_confidence(0.6, "lab_report", {"tests": 0.9}) < full
    assert document_confidence(0.8, "unknown_type", {}) == 0.8
    assert 0.0 <= document_confidence(1.5, "prescription", {"medications": 2.0}) <= 1.0

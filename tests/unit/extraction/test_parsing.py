import asyncio

import pytest
from PIL import Image
from pydantic import BaseModel

from medextract.backends.fake import FakeBackend
from medextract.extraction.parsing import JSONParseError, extract_json, generate_structured


class Sample(BaseModel):
    name: str
    count: int


VALID = '{"name": "alpha", "count": 3}'


def _img() -> list[Image.Image]:
    return [Image.new("RGB", (8, 8), "white")]


# --- extract_json ----------------------------------------------------------------------


def test_extract_json_plain_object():
    assert extract_json(VALID) == {"name": "alpha", "count": 3}


def test_extract_json_strips_markdown_fences():
    text = f"```json\n{VALID}\n```"
    assert extract_json(text) == {"name": "alpha", "count": 3}
    assert extract_json(f"```\n{VALID}\n```") == {"name": "alpha", "count": 3}


def test_extract_json_ignores_prose_around_the_object():
    text = f"Sure, here is the result: {VALID}\nLet me know if you need anything else."
    assert extract_json(text) == {"name": "alpha", "count": 3}


def test_extract_json_skips_braces_inside_prose_before_the_object():
    text = f"Use the {{placeholder}} format. {VALID}"
    assert extract_json(text) == {"name": "alpha", "count": 3}


def test_extract_json_removes_think_block():
    text = f"<think>The user wants {{a list}} of things, maybe {{x}}.</think>\n{VALID}"
    assert extract_json(text) == {"name": "alpha", "count": 3}


def test_extract_json_handles_bare_closing_think_tag():
    text = f"reasoning goes here without the opening tag</think>\n```json\n{VALID}\n```"
    assert extract_json(text) == {"name": "alpha", "count": 3}


def test_extract_json_handles_unclosed_think_tag_before_json():
    text = f"<think>still thinking\n{VALID}"
    assert extract_json(text) == {"name": "alpha", "count": 3}


def test_extract_json_repairs_trailing_commas():
    assert extract_json('{"name": "alpha", "count": 3,}') == {"name": "alpha", "count": 3}
    assert extract_json('{"items": [1, 2, 3,], "ok": true,}') == {"items": [1, 2, 3], "ok": True}


def test_extract_json_repairs_single_quotes():
    assert extract_json("{'name': 'alpha', 'count': 3}") == {"name": "alpha", "count": 3}


def test_extract_json_repairs_truncated_output():
    text = '{"fields": {"hospital_name": {"value": "Clinic A", "confidence": 0.9}, "patient_name": {"value": "Na'
    data = extract_json(text)
    assert data["fields"]["hospital_name"] == {"value": "Clinic A", "confidence": 0.9}
    assert "patient_name" in data["fields"]


def test_extract_json_repairs_object_truncated_after_nested_object():
    text = '{"fields": {"a": {"value": 1}}, "b": {"value": '
    data = extract_json(text)
    assert data["fields"] == {"a": {"value": 1}}


def test_extract_json_keeps_unicode_text():
    text = '```json\n{"hospital_name": {"value": "Bệnh viện Đa khoa"}}\n```'
    assert extract_json(text)["hospital_name"]["value"] == "Bệnh viện Đa khoa"


@pytest.mark.parametrize("text", ["", "no braces here", "[1, 2, 3]", "   "])
def test_extract_json_rejects_text_without_an_object(text):
    with pytest.raises(JSONParseError):
        extract_json(text)


def test_extract_json_rejects_non_string_input():
    with pytest.raises(JSONParseError):
        extract_json(None)  # type: ignore[arg-type]


# --- generate_structured ---------------------------------------------------------------


async def test_generate_structured_returns_parsed_model_on_first_attempt():
    backend = FakeBackend([f"```json\n{VALID}\n```"])
    parsed, result, attempts = await generate_structured(backend, _img(), "extract", Sample)
    assert parsed == Sample(name="alpha", count=3)
    assert result.text.startswith("```json")
    assert attempts == 1
    assert len(backend.calls) == 1


async def test_generate_structured_passes_system_and_max_tokens_to_backend():
    backend = FakeBackend([VALID])
    await generate_structured(backend, _img(), "extract", Sample, system="SYS", max_tokens=99)
    assert backend.calls[0].system == "SYS"


async def test_generate_structured_retries_with_the_error_in_the_prompt():
    backend = FakeBackend(["this is not json at all", VALID])
    parsed, _, attempts = await generate_structured(backend, _img(), "extract the card", Sample)
    assert parsed.count == 3
    assert attempts == 2
    assert len(backend.calls) == 2
    retry_prompt = backend.calls[1].prompt
    assert retry_prompt.startswith("extract the card")
    assert "Your previous answer was invalid: no JSON object found in model output." in retry_prompt
    assert "Return ONLY valid JSON matching the schema." in retry_prompt


async def test_generate_structured_retries_on_pydantic_validation_error():
    backend = FakeBackend(['{"name": "alpha", "count": "many"}', VALID])
    parsed, _, attempts = await generate_structured(backend, _img(), "extract", Sample)
    assert parsed == Sample(name="alpha", count=3)
    assert attempts == 2
    assert "count" in backend.calls[1].prompt
    assert "valid integer" in backend.calls[1].prompt


async def test_generate_structured_always_invalid_raises_with_last_error_and_preview():
    bad = "x" * 500 + " not json"
    backend = FakeBackend([bad])
    with pytest.raises(JSONParseError) as info:
        await generate_structured(backend, _img(), "extract", Sample, retries=2)
    assert len(backend.calls) == 3
    message = str(info.value)
    assert "no JSON object found" in message
    assert "3 attempt(s)" in message
    assert "x" * 300 in message
    assert "x" * 301 not in message


async def test_generate_structured_zero_retries_fails_after_one_attempt():
    backend = FakeBackend(["nope", VALID])
    with pytest.raises(JSONParseError):
        await generate_structured(backend, _img(), "extract", Sample, retries=0)
    assert len(backend.calls) == 1


async def test_generate_structured_timeout_raises_without_retry():
    backend = FakeBackend([VALID], delay_s=0.5)
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await generate_structured(backend, _img(), "extract", Sample, timeout_s=0.01)
    assert len(backend.calls) == 1


async def test_generate_structured_without_timeout_waits_for_slow_backend():
    backend = FakeBackend([VALID], delay_s=0.01)
    parsed, _, attempts = await generate_structured(backend, _img(), "extract", Sample, timeout_s=None)
    assert parsed.name == "alpha"
    assert attempts == 1

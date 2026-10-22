"""TransformersBackend tests.

Unit tests replace the transformers classes with fakes (via monkeypatch) so they never download
a model. They cover message building, decoding only the new tokens, logprob extraction and the
CUDA-vs-CPU load options. The ``slow`` test runs the real 2B model on CPU:

    .venv/bin/pytest -m slow tests/unit/backends/test_transformers.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from medextract.backends import transformers as tb  # noqa: E402
from medextract.backends.base import GenerationResult, VLMError  # noqa: E402
from medextract.backends.transformers import (  # noqa: E402
    TransformersBackend,
    _cpu_dtype,
    build_messages,
)

EOS = 151645
PROMPT_IDS = (1, 2, 3)
NEW_IDS = (100, 101)


def _img(color="white", size=(8, 8), mode="RGB"):
    return Image.new(mode, size, color)


# --- fakes ---------------------------------------------------------------------------------


class FakeTokenizer:
    """Decodes ids as dash-joined numbers, dropping EOS, so tests can see exactly what was decoded."""

    def decode(self, token_ids, skip_special_tokens=True):
        ids = [int(t) for t in token_ids]
        if skip_special_tokens:
            ids = [i for i in ids if i != EOS]
        return "-".join(str(i) for i in ids)


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self, prompt_ids=PROMPT_IDS):
        self.tokenizer = FakeTokenizer()
        self.prompt_ids = list(prompt_ids)
        self.chat_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        ids = torch.tensor([self.prompt_ids])
        return FakeBatch(input_ids=ids, attention_mask=torch.ones_like(ids))


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, new_ids=(*NEW_IDS, EOS), logprobs=(-0.1, -0.5, -0.01)):
        self.new_ids = list(new_ids)
        self.logprobs = list(logprobs)
        self.generate_calls = []
        self.transition_calls = []

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask=None, **kwargs):
        self.generate_calls.append(kwargs)
        prompt = input_ids[0].tolist()
        full = torch.tensor([prompt + self.new_ids])
        scores = tuple(torch.zeros(1, 10) for _ in self.new_ids)
        return SimpleNamespace(sequences=full, scores=scores)

    def compute_transition_scores(self, sequences, scores, normalize_logits=True):
        self.transition_calls.append(normalize_logits)
        return torch.tensor([self.logprobs])


class FakeModelNoScores(FakeModel):
    compute_transition_scores = None  # the backend must cope with a model that cannot report logprobs


class FakeModelRaisingScores(FakeModel):
    def compute_transition_scores(self, *args, **kwargs):
        raise RuntimeError("scores unavailable")


class _Recorder:
    """Stands in for a transformers config class; records the keyword arguments it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_hf(monkeypatch):
    """Replace the processor, model and BitsAndBytesConfig lookups with recording fakes."""
    state = SimpleNamespace(
        processor=FakeProcessor(),
        model=FakeModel(),
        processor_ids=[],
        from_pretrained_calls=[],
        bnb_configs=[],
        error=None,
    )

    class AutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            state.processor_ids.append(model_id)
            return state.processor

    class Qwen3VLForConditionalGeneration:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            state.from_pretrained_calls.append((model_id, kwargs))
            if state.error is not None:
                raise state.error
            return state.model

    class BitsAndBytesConfig(_Recorder):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            state.bnb_configs.append(self)

    monkeypatch.setattr(tb, "_auto_processor_class", lambda: AutoProcessor)
    monkeypatch.setattr(tb, "_qwen_model_class", lambda: Qwen3VLForConditionalGeneration)
    monkeypatch.setattr(tb, "_bnb_config_class", lambda: BitsAndBytesConfig)
    return state


# --- message building ----------------------------------------------------------------------


def test_build_messages_with_system_and_images():
    messages = build_messages([_img(), _img()], "Extract the fields.", system="You are precise.")
    assert messages[0] == {"role": "system", "content": "You are precise."}
    user = messages[1]
    assert user["role"] == "user"
    assert [part["type"] for part in user["content"]] == ["image", "image", "text"]
    assert user["content"][-1] == {"type": "text", "text": "Extract the fields."}
    assert all(isinstance(part["image"], Image.Image) for part in user["content"][:-1])


def test_build_messages_without_system_and_converts_images_to_rgb():
    messages = build_messages([_img(mode="L")], "Hi")
    assert len(messages) == 1 and messages[0]["role"] == "user"
    image = messages[0]["content"][0]["image"]
    assert image.mode == "RGB"


# --- construction ----------------------------------------------------------------------------


def test_constructor_defaults_follow_device():
    cuda = TransformersBackend("m/model", "cuda")
    cpu = TransformersBackend("m/model", "cpu")
    assert (cuda.name, cuda.load_in_4bit) == ("cuda", True)
    assert (cpu.name, cpu.load_in_4bit) == ("cpu", False)
    assert cuda.describe() == {"backend": "cuda", "model_id": "m/model"}


def test_invalid_device_and_cpu_4bit_are_rejected():
    with pytest.raises(ValueError, match="'cuda' or 'cpu'"):
        TransformersBackend("m", "mlx")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires device='cuda'"):
        TransformersBackend("m", "cpu", load_in_4bit=True)


# --- loading: CUDA vs CPU options ---------------------------------------------------------------


async def test_cuda_load_requests_nf4_4bit_quantization(fake_hf):
    backend = TransformersBackend("Qwen/Qwen3-VL-8B-Instruct", "cuda")
    await backend.load()
    await backend.close()

    assert fake_hf.processor_ids == ["Qwen/Qwen3-VL-8B-Instruct"]
    model_id, kwargs = fake_hf.from_pretrained_calls[0]
    assert model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert kwargs["device_map"] == "auto"
    assert kwargs["quantization_config"] is fake_hf.bnb_configs[0]
    assert "dtype" not in kwargs
    config = fake_hf.bnb_configs[0].kwargs
    assert config == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
    }


async def test_cpu_load_uses_bf16_and_no_quantization(fake_hf):
    backend = TransformersBackend("Qwen/Qwen3-VL-2B-Instruct", "cpu")
    await backend.load()
    await backend.close()

    _, kwargs = fake_hf.from_pretrained_calls[0]
    assert "quantization_config" not in kwargs
    assert "device_map" not in kwargs
    assert kwargs["dtype"] == _cpu_dtype(torch)
    assert fake_hf.bnb_configs == []  # bitsandbytes is never touched on CPU


def test_cpu_dtype_is_bf16_on_this_machine():
    assert _cpu_dtype(torch) == torch.bfloat16


async def test_load_failure_is_wrapped_in_vlm_error(fake_hf):
    fake_hf.error = OSError("network down")
    backend = TransformersBackend("Qwen/Qwen3-VL-2B-Instruct", "cpu")
    with pytest.raises(VLMError, match="Qwen/Qwen3-VL-2B-Instruct.*network down"):
        await backend.load()
    await backend.close()


def test_generate_sync_before_load_raises_vlm_error():
    backend = TransformersBackend("m", "cpu")
    with pytest.raises(VLMError, match="not loaded"):
        backend._generate_sync([], "x", None, 8, 0.0)
    backend._executor.shutdown(wait=False)


# --- generation ----------------------------------------------------------------------------------


async def test_generate_decodes_only_new_tokens_and_reports_counts(fake_hf):
    backend = TransformersBackend("m", "cpu")
    result = await backend.generate([_img()], "Read it.", max_tokens=64)
    await backend.close()

    assert isinstance(result, GenerationResult)
    assert result.text == "100-101"  # prompt ids 1,2,3 are not decoded; EOS is dropped
    assert result.prompt_tokens == len(PROMPT_IDS)
    assert result.generation_tokens == 3  # two text tokens plus EOS
    assert result.token_logprobs == pytest.approx([-0.1, -0.5, -0.01])
    assert result.elapsed_s > 0

    gen = fake_hf.model.generate_calls[0]
    assert gen["max_new_tokens"] == 64
    assert gen["do_sample"] is False
    assert gen["output_scores"] is True
    assert gen["return_dict_in_generate"] is True
    assert "temperature" not in gen
    assert fake_hf.model.transition_calls == [True]


async def test_sampling_only_when_temperature_is_positive(fake_hf):
    backend = TransformersBackend("m", "cpu")
    await backend.generate([], "x", temperature=0.7)
    await backend.close()
    gen = fake_hf.model.generate_calls[0]
    assert gen["do_sample"] is True
    assert gen["temperature"] == 0.7


async def test_chat_template_gets_system_images_and_generation_prompt(fake_hf):
    backend = TransformersBackend("m", "cpu")
    await backend.generate([_img()], "Describe.", system="Sys.")
    await backend.close()

    messages, kwargs = fake_hf.processor.chat_calls[0]
    assert messages[0] == {"role": "system", "content": "Sys."}
    assert messages[1]["content"][0]["type"] == "image"
    assert messages[1]["content"][1] == {"type": "text", "text": "Describe."}
    assert kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }


async def test_logprobs_are_none_when_model_cannot_report_them(fake_hf):
    fake_hf.model = FakeModelNoScores()
    backend = TransformersBackend("m", "cpu")
    result = await backend.generate([], "x")
    await backend.close()
    assert result.token_logprobs is None
    assert result.text == "100-101"


async def test_logprobs_are_none_when_scoring_raises(fake_hf):
    fake_hf.model = FakeModelRaisingScores()
    backend = TransformersBackend("m", "cpu")
    result = await backend.generate([], "x")
    await backend.close()
    assert result.token_logprobs is None
    assert result.generation_tokens == 3


async def test_close_releases_model_on_worker_thread(fake_hf):
    backend = TransformersBackend("m", "cpu")
    await backend.load()
    assert backend._model is not None
    await backend.close()
    assert backend._model is None and backend._processor is None
    assert backend._loaded is False


async def test_generation_returns_text_through_threaded_backend(fake_hf):
    # End-to-end through the async API (ThreadedBackend), still with fakes.
    async with TransformersBackend("m", "cpu") as backend:
        result = await backend.generate([_img()], "Read it.")
    assert result.text == "100-101"


# --- real model (slow) -------------------------------------------------------------------------


def _text_image(text: str) -> Image.Image:
    image = Image.new("RGB", (320, 96), "white")
    ImageDraw.Draw(image).text((16, 36), text, fill="black")
    return image


@pytest.mark.slow
async def test_cpu_path_runs_real_qwen3vl_2b_model():
    # mlx-vlm registers its own processors process-wide; once it has been imported (e.g. by the
    # MLX integration tests in the same pytest run), transformers would hand back MLX arrays.
    # A real deployment uses one backend per process, so run this test in a fresh interpreter.
    if "mlx_vlm" in sys.modules and not os.environ.get("MEDEXTRACT_ISOLATED_TEST"):
        test_id = f"{__file__}::test_cpu_path_runs_real_qwen3vl_2b_model"
        env = {**os.environ, "MEDEXTRACT_ISOLATED_TEST": "1"}
        cmd = [sys.executable, "-m", "pytest", "-m", "slow", "-q", "-p", "no:warnings", test_id]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
        return

    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    backend = TransformersBackend(model_id, "cpu")

    start = time.perf_counter()
    await backend.load()
    load_s = time.perf_counter() - start

    start = time.perf_counter()
    result = await backend.generate(
        [_text_image("HELLO 42")],
        "What text is written in this image? Answer briefly.",
        max_tokens=32,
    )
    gen_s = time.perf_counter() - start
    await backend.close()

    print(
        f"\n[slow] load={load_s:.1f}s generate={gen_s:.1f}s "
        f"tokens={result.generation_tokens} prompt_tokens={result.prompt_tokens} "
        f"tok/s={(result.generation_tokens or 0) / gen_s:.2f} text={result.text!r}"
    )
    assert result.text.strip()
    assert result.generation_tokens is not None and 0 < result.generation_tokens <= 32
    assert result.token_logprobs is not None
    assert len(result.token_logprobs) == result.generation_tokens
    assert all(lp <= 1e-6 for lp in result.token_logprobs)

"""Hardware detection, backend selection and settings parsing.

Hardware is simulated by patching ``platform.system``/``platform.machine``, ``find_spec`` and a
fake ``torch`` module, so these tests run on any machine and never import mlx or torch.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import platform
import subprocess
import sys
import types

import pytest

from medextract.backends.base import VLMError
from medextract.backends.factory import (
    DEFAULT_MODELS,
    create_backend,
    cuda_available,
    detect_backend,
    mlx_available,
    resolve_model_id,
)
from medextract.config import Settings

# --- simulated hardware ------------------------------------------------------------------

MAC_ARM = {"system": "Darwin", "machine": "arm64"}
LINUX_NVIDIA = {"system": "Linux", "machine": "x86_64"}


def _simulate(monkeypatch, *, system: str, machine: str, packages: set[str], cuda: bool = False):
    """Pretend to be ``system``/``machine`` with ``packages`` importable and CUDA ``cuda``."""
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)

    def fake_find_spec(name, *args, **kwargs):
        if name in packages:
            return importlib.machinery.ModuleSpec(name, None)
        return None

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    if "torch" in packages:
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        monkeypatch.setitem(sys.modules, "torch", torch)
    else:
        monkeypatch.delitem(sys.modules, "torch", raising=False)


def _simulate_mac_mlx(monkeypatch):
    _simulate(monkeypatch, **MAC_ARM, packages={"mlx", "torch", "transformers"}, cuda=False)


def _simulate_nvidia(monkeypatch):
    _simulate(monkeypatch, **LINUX_NVIDIA, packages={"torch", "transformers"}, cuda=True)


def _simulate_cpu_only(monkeypatch):
    _simulate(monkeypatch, **LINUX_NVIDIA, packages={"torch", "transformers"}, cuda=False)


# --- the three hardware cases --------------------------------------------------------------


def test_auto_picks_mlx_on_apple_silicon(monkeypatch):
    _simulate_mac_mlx(monkeypatch)
    assert detect_backend("auto") == "mlx"


def test_auto_picks_cuda_when_gpu_visible(monkeypatch):
    _simulate_nvidia(monkeypatch)
    assert detect_backend("auto") == "cuda"


def test_auto_falls_back_to_cpu(monkeypatch):
    _simulate_cpu_only(monkeypatch)
    assert detect_backend("auto") == "cpu"


def test_auto_prefers_mlx_over_cuda_when_both_look_available(monkeypatch):
    _simulate(monkeypatch, **MAC_ARM, packages={"mlx", "torch"}, cuda=True)
    assert detect_backend("auto") == "mlx"


def test_mlx_not_available_on_intel_mac_or_without_package(monkeypatch):
    _simulate(monkeypatch, system="Darwin", machine="x86_64", packages={"mlx"})
    assert mlx_available() is False
    _simulate(monkeypatch, system="Darwin", machine="arm64", packages=set())
    assert mlx_available() is False
    _simulate(monkeypatch, system="Linux", machine="arm64", packages={"mlx"})
    assert mlx_available() is False


def test_cuda_not_available_without_torch(monkeypatch):
    _simulate(monkeypatch, **LINUX_NVIDIA, packages=set())
    assert cuda_available() is False
    assert detect_backend("auto") == "cpu"


def test_cuda_not_available_when_torch_reports_no_gpu(monkeypatch):
    _simulate(monkeypatch, **LINUX_NVIDIA, packages={"torch"}, cuda=False)
    assert cuda_available() is False


def test_cuda_not_available_when_torch_import_is_broken(monkeypatch):
    _simulate(monkeypatch, **LINUX_NVIDIA, packages={"torch"})
    monkeypatch.setitem(sys.modules, "torch", None)  # import raises ImportError
    assert cuda_available() is False


# --- explicit choices ----------------------------------------------------------------------


@pytest.mark.parametrize("preferred", ["mlx", "cuda", "cpu"])
def test_explicit_choice_is_honoured_when_available(monkeypatch, preferred):
    _simulate(monkeypatch, **MAC_ARM, packages={"mlx", "torch"}, cuda=True)
    assert detect_backend(preferred) == preferred


def test_explicit_mlx_unavailable_raises_with_hint(monkeypatch):
    _simulate_nvidia(monkeypatch)
    with pytest.raises(VLMError, match=r"\[mlx\] extra"):
        detect_backend("mlx")


def test_explicit_cuda_unavailable_raises_with_hint(monkeypatch):
    _simulate_cpu_only(monkeypatch)
    with pytest.raises(VLMError, match=r"\[cuda\] extra"):
        detect_backend("cuda")


def test_explicit_cpu_is_always_selectable(monkeypatch):
    _simulate(monkeypatch, **MAC_ARM, packages=set())
    assert detect_backend("cpu") == "cpu"


def test_unknown_device_name_is_rejected():
    with pytest.raises(ValueError, match="device must be one of"):
        detect_backend("tpu")


# --- model ids -----------------------------------------------------------------------------


def test_default_models_per_backend():
    assert DEFAULT_MODELS["mlx"] == "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    assert DEFAULT_MODELS["cuda"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert DEFAULT_MODELS["cpu"] == "Qwen/Qwen3-VL-8B-Instruct"


@pytest.mark.parametrize("kind", ["mlx", "cuda", "cpu"])
def test_resolve_model_id_uses_default_when_unset(kind):
    assert resolve_model_id(kind, None) == DEFAULT_MODELS[kind]
    assert resolve_model_id(kind, "   ") == DEFAULT_MODELS[kind]


def test_resolve_model_id_honours_override():
    assert resolve_model_id("cpu", "Qwen/Qwen3-VL-2B-Instruct") == "Qwen/Qwen3-VL-2B-Instruct"


# --- create_backend (backend classes are replaced by recording stubs) -----------------------


class _RecordingBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_stub_backends(monkeypatch):
    mlx_module = types.ModuleType("medextract.backends.mlx")
    mlx_module.MLXBackend = _RecordingBackend
    tf_module = types.ModuleType("medextract.backends.transformers")
    tf_module.TransformersBackend = _RecordingBackend
    monkeypatch.setitem(sys.modules, "medextract.backends.mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "medextract.backends.transformers", tf_module)


def test_create_backend_mlx_on_mac(monkeypatch):
    _install_stub_backends(monkeypatch)
    _simulate_mac_mlx(monkeypatch)
    backend = create_backend(Settings())
    assert isinstance(backend, _RecordingBackend)
    assert backend.kwargs == {"model_id": DEFAULT_MODELS["mlx"]}


def test_create_backend_cuda_on_nvidia(monkeypatch):
    _install_stub_backends(monkeypatch)
    _simulate_nvidia(monkeypatch)
    backend = create_backend(Settings(model_id="my/model"))
    assert backend.kwargs == {"model_id": "my/model", "device": "cuda"}


def test_create_backend_cpu_on_cpu_only(monkeypatch):
    _install_stub_backends(monkeypatch)
    _simulate_cpu_only(monkeypatch)
    backend = create_backend(Settings())
    assert backend.kwargs == {"model_id": DEFAULT_MODELS["cpu"], "device": "cpu"}


def test_create_backend_explicit_device_overrides_detection(monkeypatch):
    _install_stub_backends(monkeypatch)
    _simulate_mac_mlx(monkeypatch)
    backend = create_backend(Settings(device="cpu"))
    assert backend.kwargs == {"model_id": DEFAULT_MODELS["cpu"], "device": "cpu"}


def test_create_backend_explicit_unavailable_device_raises(monkeypatch):
    _install_stub_backends(monkeypatch)
    _simulate_cpu_only(monkeypatch)
    with pytest.raises(VLMError):
        create_backend(Settings(device="mlx"))


# --- Settings.from_env ---------------------------------------------------------------------


def test_settings_defaults_without_env():
    settings = Settings.from_env({})
    assert settings == Settings()
    assert settings.device == "auto"
    assert settings.model_id is None
    assert settings.max_image_side == 1600
    assert settings.pdf_dpi == 200
    assert settings.max_tokens == 2048
    assert settings.timeout_s == 600.0
    assert settings.concurrency == 4


def test_settings_parse_all_env_variables():
    env = {
        "MEDEXTRACT_DEVICE": "CUDA",
        "MEDEXTRACT_MODEL_ID": " Qwen/Qwen3-VL-2B-Instruct ",
        "MEDEXTRACT_MAX_IMAGE_SIDE": "1024",
        "MEDEXTRACT_PDF_DPI": "150",
        "MEDEXTRACT_MAX_TOKENS": "512",
        "MEDEXTRACT_TIMEOUT_S": "12.5",
        "MEDEXTRACT_CONCURRENCY": "2",
    }
    settings = Settings.from_env(env)
    assert settings.device == "cuda"
    assert settings.model_id == "Qwen/Qwen3-VL-2B-Instruct"
    assert settings.max_image_side == 1024
    assert settings.pdf_dpi == 150
    assert settings.max_tokens == 512
    assert settings.timeout_s == 12.5
    assert settings.concurrency == 2


def test_empty_env_values_mean_unset():
    settings = Settings.from_env({"MEDEXTRACT_MODEL_ID": "", "MEDEXTRACT_PDF_DPI": "  "})
    assert settings.model_id is None
    assert settings.pdf_dpi == 200


def test_overrides_win_over_env_and_none_overrides_are_ignored():
    env = {"MEDEXTRACT_CONCURRENCY": "3", "MEDEXTRACT_DEVICE": "cpu"}
    settings = Settings.from_env(env, concurrency=7, device=None, model_id="x/y")
    assert settings.concurrency == 7
    assert settings.device == "cpu"
    assert settings.model_id == "x/y"


def test_from_env_defaults_to_os_environ(monkeypatch):
    monkeypatch.setenv("MEDEXTRACT_MAX_TOKENS", "77")
    assert Settings.from_env().max_tokens == 77


@pytest.mark.parametrize(
    ("key", "value", "fragment"),
    [
        ("MEDEXTRACT_PDF_DPI", "abc", "MEDEXTRACT_PDF_DPI: expected an integer"),
        ("MEDEXTRACT_TIMEOUT_S", "fast", "MEDEXTRACT_TIMEOUT_S: expected a number"),
        ("MEDEXTRACT_MAX_TOKENS", "0", "max_tokens must be at least 1"),
        ("MEDEXTRACT_CONCURRENCY", "-2", "concurrency must be at least 1"),
        ("MEDEXTRACT_TIMEOUT_S", "0", "timeout_s must be a finite number greater than 0"),
        ("MEDEXTRACT_TIMEOUT_S", "nan", "timeout_s must be a finite number"),
        ("MEDEXTRACT_DEVICE", "tpu", "device must be one of auto, mlx, cuda, cpu"),
    ],
)
def test_invalid_env_values_raise_value_error(key, value, fragment):
    with pytest.raises(ValueError, match=fragment):
        Settings.from_env({key: value})


def test_settings_constructor_validates_directly():
    with pytest.raises(ValueError):
        Settings(max_image_side=0)
    with pytest.raises(ValueError):
        Settings(model_id="   ")
    with pytest.raises(ValueError):
        Settings(max_tokens=True)  # bools are not integers here


def test_importing_device_module_does_not_load_heavy_libraries():
    # Run in a fresh interpreter so this test process's own imports do not interfere.
    code = (
        "import sys, medextract.backends.factory, medextract.config\n"
        "heavy = [m for m in ('mlx', 'torch', 'transformers') if m in sys.modules]\n"
        "sys.exit(1 if heavy else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

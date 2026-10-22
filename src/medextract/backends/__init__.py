"""Vision-language model backends.

``base`` defines the ``VLMBackend`` interface; ``factory`` picks MLX (Apple Silicon), then CUDA,
then CPU. ``mlx`` and ``transformers`` are imported lazily by the factory, so importing this
package never loads mlx or torch.
"""

from medextract.backends.base import GenerationResult, ThreadedBackend, VLMBackend, VLMError
from medextract.backends.factory import DEFAULT_MODELS, create_backend, detect_backend, resolve_model_id
from medextract.backends.fake import FakeBackend

__all__ = [
    "DEFAULT_MODELS",
    "FakeBackend",
    "GenerationResult",
    "ThreadedBackend",
    "VLMBackend",
    "VLMError",
    "create_backend",
    "detect_backend",
    "resolve_model_id",
]

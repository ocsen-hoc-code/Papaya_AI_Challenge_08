"""medextract: async extraction of structured data from medical documents with a local VLM.

Package layout (one responsibility per layer)::

    medextract/
    ├── pipeline.py        orchestrates the stages below (async)
    ├── config.py          Settings (MEDEXTRACT_* environment variables)
    ├── schemas.py         data contracts and FIELD_SPECS
    ├── cli.py             `medextract` command
    ├── ingestion/         PDF/image -> page images + text layer
    ├── backends/          vision-language model backends (MLX, Transformers CUDA/CPU, fake)
    ├── extraction/        prompts and tolerant JSON parsing
    └── postprocessing/    normalize, label checks, grounding, confidence, validation

The public API is re-exported here and imported lazily, so ``import medextract`` stays cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

__all__ = ["ExtractionResult", "Pipeline", "Settings", "__version__"]

_LAZY = {
    "Pipeline": "medextract.pipeline",
    "Settings": "medextract.config",
    "ExtractionResult": "medextract.schemas",
}

if TYPE_CHECKING:
    from medextract.config import Settings
    from medextract.pipeline import Pipeline
    from medextract.schemas import ExtractionResult


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'medextract' has no attribute {name!r}")

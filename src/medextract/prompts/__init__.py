"""Prompts sent to the vision-language model. See ``builders`` for the design notes."""

from .builders import SYSTEM_PROMPT, build_classify_prompt, build_extract_prompt

__all__ = ["SYSTEM_PROMPT", "build_classify_prompt", "build_extract_prompt"]

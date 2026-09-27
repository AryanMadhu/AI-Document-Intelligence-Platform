"""LLM providers and factory.

Re-exports the lightweight interface, error type, and factory from
:mod:`src.llm.base`. This intentionally does NOT import any concrete provider
module (and therefore never imports the Google SDK) at package import time —
the factory imports the concrete implementation locally when selected.
"""

from src.llm.base import LLMError, LLMProvider, build_llm_provider

__all__ = ["LLMError", "LLMProvider", "build_llm_provider"]

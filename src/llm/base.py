"""LLM provider interface and Config-driven factory.

This module defines the ``LLMProvider`` abstraction — the swappable seam that
Requirement 15 mandates for the answer-generation model — plus the
``build_llm_provider`` factory that selects a concrete implementation from a
:class:`~src.config.Config` value (never from hard-coded application logic).

It mirrors the structure of :mod:`src.embeddings.base`.

Security notes (R16.4):
    * The factory reports an unknown provider or an unavailable Stretch
      provider BY the offending selector string only. No credential value is
      ever included in an ``LLMError`` message.

Credential enforcement (design choice):
    Enforcement of ``GEMINI_API_KEY`` is **lazy** — it happens inside the
    provider on first use (``GeminiLLMProvider._get_client`` /
    ``generate``), NOT in this factory. This keeps ``build_llm_provider``
    free of any SDK import or credential read, so merely constructing the
    provider (e.g. in tests) never requires network or a real key. A missing
    key surfaces as a named, value-free :class:`LLMError` the first time a
    generation is attempted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import Config


class LLMError(Exception):
    """Raised when LLM generation fails or the factory is misconfigured.

    Messages are always value-free: they may name a provider selector string
    or a required credential key, but they never contain a credential's value
    or a raw provider payload that could carry sensitive data (R16.4).
    """


class LLMProvider(ABC):
    """Generates a grounded answer from a prompt. Swappable via ``Config``.

    Concrete implementations expose a stable ``provider_name`` and
    ``model_name`` and implement ``generate``. No domain-specific (e.g.
    banking) logic lives in any provider (R15.5) — they are thin adapters over
    the underlying model SDK.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable provider identifier.

        Returns:
            A stable string id, e.g. ``'gemini'``.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable LLM model identifier.

        Returns:
            The model id, e.g. ``'gemini-flash-latest'``.
        """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the model's completion for the prompt.

        Args:
            prompt: The fully-assembled prompt to send to the model.

        Returns:
            The model's text completion.

        Raises:
            LLMError: If the underlying model fails to produce a completion.
        """


# Provider selectors that are recognized but deferred to Stretch/Future scope.
# They map to a concrete implementation behind the same interface later; in the
# MVP the factory recognizes them and fails clearly rather than treating them
# as unknown.
_STRETCH_PROVIDERS = frozenset({"openai", "anthropic", "ollama"})


def build_llm_provider(config: Config) -> LLMProvider:
    """Build the active :class:`LLMProvider` from configuration.

    Provider selection is driven entirely by ``config.llm_provider`` — it is
    never hard-coded in application logic. Switching providers is a
    configuration (``.env``) change, not a source change (R15.3-4).

    The concrete implementation module is imported *locally*, inside the branch
    that selects it, so that merely building a differently-configured provider
    (or importing this module) does not drag in the Google SDK.

    Credential enforcement is lazy (see module docstring): a missing
    ``GEMINI_API_KEY`` surfaces from the provider on first generation, as a
    named, value-free :class:`LLMError`, not from this factory.

    Args:
        config: The active configuration. ``config.llm_provider`` selects the
            implementation and ``config.llm_model`` supplies the model id.

    Returns:
        A concrete ``LLMProvider`` instance.

    Raises:
        LLMError: If ``config.llm_provider`` names a Stretch provider not yet
            available in the MVP, or an unknown provider. The message includes
            the offending selector string but never any credential value
            (R16.4).
    """
    provider_key = config.llm_provider

    if provider_key == "gemini":
        # Local import so the Google SDK loads only when Gemini is selected.
        from src.llm.gemini_provider import GeminiLLMProvider

        return GeminiLLMProvider(model_name=config.llm_model, config=config)

    if provider_key in _STRETCH_PROVIDERS:
        # Recognized selector, but the concrete impl is Stretch/Future.
        raise LLMError(
            f"LLM provider {provider_key!r} is a Stretch/future provider "
            "and is not available in the MVP."
        )

    # Unknown/unsupported provider: name the offending selector, no secret.
    raise LLMError(f"Unsupported LLM provider: {provider_key!r}")

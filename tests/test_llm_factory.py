"""Fast, deterministic unit tests for the LLM factory and Gemini provider.

These tests never touch the network or consume any quota. The Gemini client is
replaced with a fake via ``_get_client`` monkeypatching, so the suite runs with
no network access and without importing the Google SDK.

Credential enforcement is LAZY (in the provider, not the factory): a missing
``GEMINI_API_KEY`` surfaces on the first ``generate`` / ``_get_client`` call,
so the missing-credential test triggers that path explicitly.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config import Config
from src.llm.base import (
    LLMError,
    LLMProvider,
    build_llm_provider,
)
from src.llm.gemini_provider import GeminiLLMProvider

# A sentinel secret used to prove no key value ever leaks into a repr/message.
_SAMPLE_SECRET = "SECRET-TEST-KEY-DO-NOT-LEAK"


class _FakeResponse:
    """Stand-in for a genai generate_content response."""

    def __init__(self, text) -> None:
        self.text = text


class _FakeModels:
    """Stand-in for ``client.models`` returning a canned response."""

    def __init__(self, text) -> None:
        self._text = text

    def generate_content(self, model, contents):
        return _FakeResponse(self._text)


class _FakeClient:
    """Fake genai.Client whose ``models.generate_content`` returns canned text."""

    def __init__(self, text="grounded answer") -> None:
        self.models = _FakeModels(text)


class _RaisingModels:
    """Fake ``client.models`` whose ``generate_content`` always raises.

    The raised exception embeds a sample secret to prove the provider does not
    propagate the underlying payload into its own error message.
    """

    def generate_content(self, model, contents):
        raise RuntimeError(f"upstream failure containing {_SAMPLE_SECRET}")


class _RaisingClient:
    """Fake client whose generation always raises (with an embedded secret)."""

    def __init__(self) -> None:
        self.models = _RaisingModels()


def _config(**overrides) -> Config:
    """Build a default Config, applying any field overrides (no env/.env load)."""
    return dataclasses.replace(Config(), **overrides)


def test_factory_gemini_returns_gemini_provider() -> None:
    """llm_provider='gemini' maps to a GeminiLLMProvider with the config model."""
    config = _config(
        llm_provider="gemini",
        llm_model="gemini-flash-latest",
        gemini_api_key=_SAMPLE_SECRET,
    )

    provider = build_llm_provider(config)

    assert isinstance(provider, GeminiLLMProvider)
    assert provider.provider_name == "gemini"
    assert provider.model_name == "gemini-flash-latest"
    # Construction must not have created a client (lazy).
    assert provider._client is None


def test_missing_gemini_key_raises_named_value_free_error() -> None:
    """A missing GEMINI_API_KEY surfaces lazily as a named, value-free error."""
    # Default config: gemini_api_key is None.
    config = _config(llm_provider="gemini")
    provider = build_llm_provider(config)

    # Enforcement is lazy: building is fine; generation triggers the check.
    with pytest.raises(LLMError) as exc_info:
        provider.generate("hello")

    message = str(exc_info.value)
    assert "GEMINI_API_KEY" in message
    # No secret leaked (there is none set, but the message must stay value-free).
    assert _SAMPLE_SECRET not in message


@pytest.mark.parametrize("stretch_key", ["openai", "anthropic", "ollama"])
def test_factory_stretch_providers_raise_not_available(stretch_key) -> None:
    """Stretch selectors raise LLMError naming the provider as MVP-unavailable."""
    config = _config(llm_provider=stretch_key)

    with pytest.raises(LLMError) as exc_info:
        build_llm_provider(config)

    message = str(exc_info.value)
    assert stretch_key in message
    assert "MVP" in message


def test_factory_unknown_provider_raises_named_error() -> None:
    """An unknown provider string raises LLMError naming the string (R16.4)."""
    config = _config(llm_provider="does-not-exist")

    with pytest.raises(LLMError) as exc_info:
        build_llm_provider(config)

    message = str(exc_info.value)
    assert "does-not-exist" in message
    assert "API_KEY" not in message.upper()


def test_generate_happy_path_without_network(monkeypatch) -> None:
    """generate returns the model text, using a fake client (no network)."""
    provider = GeminiLLMProvider(model_name="gemini-flash-latest")
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient("grounded answer"))

    assert provider.generate("prompt") == "grounded answer"


def test_generate_failure_wraps_and_hides_secret(monkeypatch) -> None:
    """A failing generate_content surfaces as LLMError with no leaked secret."""
    provider = GeminiLLMProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: _RaisingClient())

    with pytest.raises(LLMError) as exc_info:
        provider.generate("prompt")

    message = str(exc_info.value)
    assert _SAMPLE_SECRET not in message


def test_generate_none_text_raises(monkeypatch) -> None:
    """A response whose text is None raises LLMError."""
    provider = GeminiLLMProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient(None))

    with pytest.raises(LLMError):
        provider.generate("prompt")


def test_llm_provider_is_abstract() -> None:
    """LLMProvider cannot be instantiated directly."""
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_provider_repr_does_not_leak_secret() -> None:
    """The provider repr never contains the configured API key value (R16.4)."""
    config = _config(llm_provider="gemini", gemini_api_key=_SAMPLE_SECRET)
    provider = build_llm_provider(config)

    assert _SAMPLE_SECRET not in repr(provider)

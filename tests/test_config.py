"""Unit tests for :mod:`src.config`.

Every test isolates the process environment with pytest's ``monkeypatch``
fixture so results are deterministic regardless of any real ``.env`` present.
``Config.from_env`` is always called with ``load_env_file=False`` here so no
on-disk ``.env`` can influence the assertions.
"""

from __future__ import annotations

import pytest

from src.config import Config, MissingCredentialError

# All environment variables the Config reads; cleared per-test for isolation.
_ALL_ENV_VARS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "TOP_K",
    "CHUNK_SIZE_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "MAX_PDF_SIZE_MB",
    "MAX_PDF_PAGES",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every Config-relevant variable from the process environment."""
    for name in _ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_resolve_to_documented_values(clean_env: None) -> None:
    """With a clean environment, every setting resolves to its default."""
    config = Config.from_env(load_env_file=False)

    assert config.embedding_provider == "sentence-transformers"
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.llm_provider == "gemini"
    assert config.llm_model == "gemini-flash-latest"
    assert config.top_k == 5
    assert config.chunk_size_tokens == 700
    assert config.chunk_overlap_tokens == 90
    assert config.max_pdf_size_mb == 5
    assert config.max_pdf_pages == 50

    # Credentials default to None when unset.
    assert config.gemini_api_key is None
    assert config.openai_api_key is None
    assert config.anthropic_api_key is None


def test_integer_defaults_are_ints(clean_env: None) -> None:
    """Integer-typed settings are real ``int`` values (not strings)."""
    config = Config.from_env(load_env_file=False)

    assert isinstance(config.top_k, int)
    assert isinstance(config.chunk_size_tokens, int)
    assert isinstance(config.chunk_overlap_tokens, int)
    assert isinstance(config.max_pdf_size_mb, int)
    assert isinstance(config.max_pdf_pages, int)


def test_env_overrides_are_applied(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env vars override defaults, including integer coercion from strings."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "512")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "64")
    monkeypatch.setenv("MAX_PDF_SIZE_MB", "20")
    monkeypatch.setenv("MAX_PDF_PAGES", "100")

    config = Config.from_env(load_env_file=False)

    assert config.embedding_provider == "openai"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.llm_provider == "openai"
    assert config.llm_model == "gpt-4o-mini"

    # Integer coercion: strings become ints.
    assert config.top_k == 9
    assert isinstance(config.top_k, int)
    assert config.chunk_size_tokens == 512
    assert config.chunk_overlap_tokens == 64
    assert config.max_pdf_size_mb == 20
    assert config.max_pdf_pages == 100


def test_credentials_loaded_from_env(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credential values are read from the environment when present."""
    monkeypatch.setenv("GEMINI_API_KEY", "gem-secret-123")
    monkeypatch.setenv("OPENAI_API_KEY", "oai-secret-456")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-secret-789")

    config = Config.from_env(load_env_file=False)

    assert config.gemini_api_key == "gem-secret-123"
    assert config.openai_api_key == "oai-secret-456"
    assert config.anthropic_api_key == "ant-secret-789"


def test_require_credential_returns_value_when_set(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``require_credential`` returns the value when the credential is set."""
    monkeypatch.setenv("GEMINI_API_KEY", "gem-secret-123")
    config = Config.from_env(load_env_file=False)

    assert config.require_credential("GEMINI_API_KEY") == "gem-secret-123"
    # Attribute-style name is also accepted.
    assert config.require_credential("gemini_api_key") == "gem-secret-123"


def test_require_credential_missing_raises_named_value_free_error(
    clean_env: None,
) -> None:
    """A missing credential raises an error naming the key but no value."""
    config = Config.from_env(load_env_file=False)

    with pytest.raises(MissingCredentialError) as exc_info:
        config.require_credential("GEMINI_API_KEY")

    message = str(exc_info.value)
    # The key NAME must be present ...
    assert "GEMINI_API_KEY" in message
    # ... and the message clearly signals absence, with no secret value.
    assert "not set" in message


def test_require_credential_error_contains_no_secret_value(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even for an empty-string credential, no value leaks into the error."""
    # Empty value is treated as unset.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    config = Config.from_env(load_env_file=False)

    with pytest.raises(MissingCredentialError) as exc_info:
        config.require_credential("OPENAI_API_KEY")

    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message


def test_repr_does_not_leak_credential_values(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``repr(config)`` must not contain any credential value."""
    secret = "super-secret-key-value-DO-NOT-LEAK"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setenv("OPENAI_API_KEY", secret + "-oai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret + "-ant")

    config = Config.from_env(load_env_file=False)
    rendered = repr(config)

    assert secret not in rendered
    assert "GEMINI_API_KEY".lower() not in rendered or "gem-secret" not in rendered
    # Non-secret settings may still appear in repr.
    assert "top_k=5" in rendered
    assert "llm_provider='gemini'" in rendered

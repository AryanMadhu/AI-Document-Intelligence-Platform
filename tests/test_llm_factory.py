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
    LLMAuthError,
    LLMError,
    LLMProvider,
    LLMQuotaExceededError,
    LLMServiceBusyError,
    build_llm_provider,
)
from src.llm import gemini_provider as gemini_module
from src.llm.gemini_provider import (
    GeminiLLMProvider,
    _is_auth_error,
    _is_quota_exhausted,
    _is_thinking_unsupported,
    _is_timeout,
    _is_transient,
)

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

    def generate_content(self, model, contents, config=None):
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

    def generate_content(self, model, contents, config=None):
        raise RuntimeError(f"upstream failure containing {_SAMPLE_SECRET}")


class _RaisingClient:
    """Fake client whose generation always raises (with an embedded secret)."""

    def __init__(self) -> None:
        self.models = _RaisingModels()


class _RecordingModels:
    """Fake ``client.models`` that RECORDS the ``config`` it received.

    Returns a normal canned response so the generate path completes; the test
    inspects ``received_config`` to assert what the provider passed.
    """

    def __init__(self, text="ok") -> None:
        self._text = text
        self.received_config = None

    def generate_content(self, model, contents, config=None):
        self.received_config = config
        return _FakeResponse(self._text)


class _RecordingClient:
    """Fake client exposing a _RecordingModels so tests can inspect config."""

    def __init__(self, text="ok") -> None:
        self.models = _RecordingModels(text)


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


# ---------------------------------------------------------------------------
# Bounded retry + transient-overload handling (SERVICE_BUSY).
#
# All tests monkeypatch ``time.sleep`` to a no-op so the backoff adds no real
# delay and the suite stays fast.
# ---------------------------------------------------------------------------


class _TransientError(Exception):
    """A fake SDK error that classifies as TRANSIENT.

    Carries a numeric ``code`` (503) and a string ``status`` ("UNAVAILABLE"),
    and its text also contains an unavailability token — mirroring the shape of
    a real Gemini 503 UNAVAILABLE.
    """

    def __init__(self, message: str = "503 UNAVAILABLE: try again later") -> None:
        super().__init__(message)
        self.code = 503
        self.status = "UNAVAILABLE"


class _CountingModels:
    """Fake ``client.models`` that counts calls and raises/returns per a script.

    Each entry in ``script`` is either an Exception instance (raised) or a str
    (returned as the response text). ``calls`` records how many times
    ``generate_content`` was invoked.
    """

    def __init__(self, script) -> None:
        self._script = list(script)
        self.calls = 0

    def generate_content(self, model, contents, config=None):
        item = self._script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class _CountingClient:
    """Fake client wrapping a _CountingModels built from a script."""

    def __init__(self, script) -> None:
        self.models = _CountingModels(script)


class _ThinkingUnsupportedError(Exception):
    """A fake SDK error signalling a model rejects ``thinking_config``.

    Mirrors the real shape: a 400 / INVALID_ARGUMENT whose message names the
    unsupported ``thinking_config`` field. Its text is value-free.
    """

    def __init__(
        self,
        message: str = "400 INVALID_ARGUMENT: thinking_config is not supported for this model",
    ) -> None:
        super().__init__(message)
        self.code = 400
        self.status = "INVALID_ARGUMENT"


class _ConfigRecordingModels:
    """Fake ``client.models`` that COUNTS calls, RECORDS each received config,
    and raises/returns per a script.

    Each entry in ``script`` is either an Exception instance (raised) or a str
    (returned as the response text). ``configs`` records the ``config`` object
    passed on each invocation so tests can assert the fallback downgraded it.
    """

    def __init__(self, script) -> None:
        self._script = list(script)
        self.calls = 0
        self.configs = []

    def generate_content(self, model, contents, config=None):
        self.configs.append(config)
        item = self._script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class _ConfigRecordingClient:
    """Fake client wrapping a _ConfigRecordingModels built from a script."""

    def __init__(self, script) -> None:
        self.models = _ConfigRecordingModels(script)


def test_transient_error_retried_then_raises_busy(monkeypatch) -> None:
    """A persistently-transient error is retried up to 2 attempts then busy-raises."""
    monkeypatch.setattr(gemini_module.time, "sleep", lambda _s: None)
    client = _CountingClient(
        [_TransientError(), _TransientError()]
    )
    provider = GeminiLLMProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    with pytest.raises(LLMServiceBusyError):
        provider.generate("prompt")

    # Both bounded attempts were made.
    assert client.models.calls == 2


def test_transient_error_succeeds_on_second_attempt(monkeypatch) -> None:
    """A transient first attempt that then succeeds returns the text (2 calls)."""
    monkeypatch.setattr(gemini_module.time, "sleep", lambda _s: None)
    client = _CountingClient([_TransientError(), "grounded answer"])
    provider = GeminiLLMProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    assert provider.generate("prompt") == "grounded answer"
    assert client.models.calls == 2


def test_non_transient_error_fails_fast_as_llm_error(monkeypatch) -> None:
    """A NON-transient error raises LLMError (not busy) after exactly 1 attempt."""
    monkeypatch.setattr(gemini_module.time, "sleep", lambda _s: None)
    client = _CountingClient([RuntimeError("boom")])
    provider = GeminiLLMProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    with pytest.raises(LLMError) as exc_info:
        provider.generate("prompt")

    # It must be a plain LLMError, NOT the busy subclass, and must not retry.
    assert not isinstance(exc_info.value, LLMServiceBusyError)
    assert client.models.calls == 1


class TestIsTransient:
    """Unit checks for the private transient classifier."""

    def test_status_code_503_is_transient(self):
        exc = _TransientError()
        assert _is_transient(exc) is True

    def test_name_unavailable_is_transient(self):
        class ServiceUnavailable(Exception):
            pass

        assert _is_transient(ServiceUnavailable("down")) is True

    def test_overloaded_text_is_transient(self):
        assert _is_transient(Exception("The model is overloaded")) is True

    def test_rate_limit_text_is_transient(self):
        assert _is_transient(Exception("429 rate limit exceeded")) is True

    def test_plain_runtime_error_is_not_transient(self):
        assert _is_transient(RuntimeError("boom")) is False

    def test_value_error_is_not_transient(self):
        assert _is_transient(ValueError("bad input")) is False

    def test_bare_digits_in_message_are_not_transient(self):
        # Contains the "500" substring (a token count) but no real transient
        # signal — must NOT be misclassified as transient after the fix.
        exc = Exception("model gemini-1.5 returned 500000 tokens")
        assert _is_transient(exc) is False

    def test_status_line_digits_are_not_transient(self):
        # Contains the "502" substring (inside a request id) and an HTTP "200"
        # line — no real transient signal — must NOT be transient after the fix.
        exc = RuntimeError("HTTP/1.1 200 OK, request_id=abc502def")
        assert _is_transient(exc) is False


# ---------------------------------------------------------------------------
# Per-call config (AFC disabled + http timeout) and timeout -> non-retried busy.
#
# A live test proved the ``generate_content`` call could take 75-331s — driven
# by the SDK's Automatic Function Calling round-trips and the absence of any
# per-call timeout — far exceeding the UI's 60s read timeout. The fix passes a
# config that disables AFC and caps a single call at 45s, and treats a timeout
# as a NON-retried busy condition (the 45s cap already consumed the budget).
# ---------------------------------------------------------------------------


class _DeadlineExceeded(Exception):
    """A fake SDK error whose CLASS NAME alone signals a timeout.

    Carries an optional 504 ``code`` to mirror a real gateway-timeout shape.
    """

    def __init__(self, message: str = "deadline_exceeded") -> None:
        super().__init__(message)
        self.code = 504


def test_generate_passes_afc_disabled_and_timeout_config(monkeypatch) -> None:
    """generate passes a config with AFC disabled and a 45000ms http timeout."""
    client = _RecordingClient("ok")
    provider = GeminiLLMProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    assert provider.generate("prompt") == "ok"

    cfg = client.models.received_config
    assert cfg is not None
    # AFC disabled so the plain-text prompt makes no function-calling round-trips.
    assert cfg.automatic_function_calling.disable is True
    # A single call is bounded to 45s (45000 ms) — under the UI's 60s read timeout.
    assert cfg.http_options.timeout == 45000


def test_generate_passes_thinking_disabled_config(monkeypatch) -> None:
    """generate's primary config disables the model's thinking budget (==0).

    The thinking budget is the measured cause of ~tens of seconds of latency on
    even trivial prompts, so the primary (success-path) config must carry
    ``thinking_config.thinking_budget == 0`` alongside the AFC-disabled +
    45000ms timeout settings already covered above.
    """
    client = _RecordingClient("ok")
    provider = GeminiLLMProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    assert provider.generate("prompt") == "ok"

    cfg = client.models.received_config
    assert cfg is not None
    # Thinking disabled (0 == DISABLED) to remove the latency it introduced.
    assert cfg.thinking_config is not None
    assert cfg.thinking_config.thinking_budget == 0
    # The latency-related settings remain intact on the same primary config.
    assert cfg.automatic_function_calling.disable is True
    assert cfg.http_options.timeout == 45000


def test_timeout_error_is_busy_without_retry(monkeypatch) -> None:
    """A timeout-like error raises LLMServiceBusyError after EXACTLY 1 attempt."""
    monkeypatch.setattr(gemini_module.time, "sleep", lambda _s: None)
    client = _CountingClient([_DeadlineExceeded("request deadline_exceeded")])
    provider = GeminiLLMProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    with pytest.raises(LLMServiceBusyError):
        provider.generate("prompt")

    # No retry: the 45s cap already consumed the budget.
    assert client.models.calls == 1


class TestIsTimeout:
    """Unit checks for the private timeout classifier."""

    def test_deadline_message_is_timeout(self):
        assert _is_timeout(Exception("504 deadline exceeded")) is True

    def test_timed_out_message_is_timeout(self):
        assert _is_timeout(RuntimeError("request timed out")) is True

    def test_code_504_is_timeout(self):
        class _Err(Exception):
            def __init__(self):
                super().__init__("gateway timeout")
                self.code = 504

        assert _is_timeout(_Err()) is True

    def test_plain_503_transient_is_not_timeout(self):
        # 503 is transient (fast retry) but NOT a timeout — must be False so it
        # still gets the quick retry rather than the no-retry busy branch.
        assert _is_timeout(_TransientError()) is False

    def test_plain_runtime_error_is_not_timeout(self):
        assert _is_timeout(RuntimeError("boom")) is False


# ---------------------------------------------------------------------------
# Thinking disabled for latency + one-time graceful fallback.
#
# The primary per-call config disables the model's "thinking" token budget
# (thinking_budget=0) because it added ~tens of seconds of latency even on
# trivial prompts. Some models reject ``thinking_config``; when they do, the
# provider downgrades ONCE to an identical config WITHOUT thinking_config. That
# downgrade is a config change, NOT a backoff: it must not sleep and must not
# consume the bounded transient-retry budget.
# ---------------------------------------------------------------------------


def test_thinking_unsupported_falls_back_without_thinking(monkeypatch) -> None:
    """A thinking-unsupported first call downgrades ONCE and then succeeds.

    The fake's FIRST call raises a thinking-unsupported error; its SECOND call
    succeeds. generate() must return the text after exactly 2 calls, and the
    SECOND (fallback) config must have ``thinking_config is None``.
    """
    monkeypatch.setattr(gemini_module.time, "sleep", lambda _s: None)
    client = _ConfigRecordingClient(
        [_ThinkingUnsupportedError(), "grounded answer"]
    )
    provider = GeminiLLMProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    assert provider.generate("prompt") == "grounded answer"

    # Exactly one primary attempt + one fallback attempt.
    assert client.models.calls == 2
    first_cfg, second_cfg = client.models.configs
    # The primary call carried the thinking budget (disabled, ==0).
    assert first_cfg.thinking_config is not None
    assert first_cfg.thinking_config.thinking_budget == 0
    # The fallback downgraded to a config WITHOUT thinking_config.
    assert second_cfg.thinking_config is None
    # The fallback keeps the other latency settings intact.
    assert second_cfg.automatic_function_calling.disable is True
    assert second_cfg.http_options.timeout == 45000


def test_thinking_fallback_does_not_consume_transient_budget(monkeypatch) -> None:
    """A thinking downgrade that then hits a FAST transient still gets a retry.

    Script: primary call raises thinking-unsupported (attempt 1 downgrade) ->
    the inline fallback raises a FAST transient -> that transient is classified
    normally, so the bounded retry budget is still available for a 2nd attempt,
    which succeeds. The downgrade itself must NOT have counted as a transient
    attempt.
    """
    monkeypatch.setattr(gemini_module.time, "sleep", lambda _s: None)
    client = _ConfigRecordingClient(
        [_ThinkingUnsupportedError(), _TransientError(), "grounded answer"]
    )
    provider = GeminiLLMProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    assert provider.generate("prompt") == "grounded answer"

    # 1 primary (thinking) + 1 fallback (transient) + 1 retry (success) = 3.
    assert client.models.calls == 3
    first_cfg, second_cfg, third_cfg = client.models.configs
    # Attempt 1 primary: thinking enabled-budget (disabled, ==0).
    assert first_cfg.thinking_config is not None
    # Inline fallback after downgrade: no thinking_config.
    assert second_cfg.thinking_config is None
    # Attempt 2 (the bounded retry) uses the primary config again.
    assert third_cfg.thinking_config is not None
    assert third_cfg.thinking_config.thinking_budget == 0


def test_thinking_unsupported_model_omits_thinking_config(monkeypatch) -> None:
    """An allowlisted model omits thinking_config on its ONLY call (no fallback).

    gemini-3.5-flash-lite is in _THINKING_UNSUPPORTED_MODELS, so its primary
    config must be the no-thinking config: exactly 1 call, whose config has
    ``thinking_config is None`` while keeping AFC disabled + 45000ms timeout.
    """
    client = _ConfigRecordingClient(["grounded answer"])
    provider = GeminiLLMProvider(model_name="gemini-3.5-flash-lite")
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    assert provider.generate("prompt") == "grounded answer"

    # Exactly ONE call — the allowlisted model never needs the fallback.
    assert client.models.calls == 1
    (only_cfg,) = client.models.configs
    # The primary config for an allowlisted model carries NO thinking_config.
    assert only_cfg.thinking_config is None
    # The other latency settings remain intact on that same primary config.
    assert only_cfg.automatic_function_calling.disable is True
    assert only_cfg.http_options.timeout == 45000


def test_default_model_still_sends_thinking_config(monkeypatch) -> None:
    """A non-allowlisted model still sends thinking_config (budget == 0).

    gemini-flash-latest is NOT in _THINKING_UNSUPPORTED_MODELS, so its primary
    call must carry ``thinking_config.thinking_budget == 0`` — unchanged
    thinking-disabled behavior.
    """
    client = _ConfigRecordingClient(["grounded answer"])
    provider = GeminiLLMProvider(model_name="gemini-flash-latest")
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    assert provider.generate("prompt") == "grounded answer"

    assert client.models.calls == 1
    (only_cfg,) = client.models.configs
    # Other models keep thinking disabled VIA thinking_config (budget == 0).
    assert only_cfg.thinking_config is not None
    assert only_cfg.thinking_config.thinking_budget == 0
    assert only_cfg.automatic_function_calling.disable is True
    assert only_cfg.http_options.timeout == 45000


def test_unrelated_400_still_llm_error(monkeypatch) -> None:
    """A non-allowlisted model's generic 400 (no "thinking") -> LLMError fast.

    Proves the generic 400 handler was NOT broadened: an INVALID_ARGUMENT that
    never mentions "thinking" and is not an allowlisted model is surfaced as a
    plain LLMError (NOT a busy/downgrade) after EXACTLY 1 call.
    """
    monkeypatch.setattr(gemini_module.time, "sleep", lambda _s: None)

    class _GenericInvalidArgument(Exception):
        def __init__(self) -> None:
            super().__init__("400 INVALID_ARGUMENT: unsupported field 'foo'")
            self.code = 400
            self.status = "INVALID_ARGUMENT"

    client = _CountingClient([_GenericInvalidArgument()])
    provider = GeminiLLMProvider(model_name="gemini-flash-latest")
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    with pytest.raises(LLMError) as exc_info:
        provider.generate("prompt")

    # Plain LLMError, NOT the busy subclass, and NOT silently downgraded.
    assert not isinstance(exc_info.value, LLMServiceBusyError)
    # Exactly one call: no downgrade, no retry.
    assert client.models.calls == 1
    assert _SAMPLE_SECRET not in str(exc_info.value)


class TestIsThinkingUnsupported:
    """Unit checks for the private thinking-unsupported classifier."""

    def test_message_names_thinking_is_true(self):
        assert _is_thinking_unsupported(
            Exception("thinking_config is not supported")
        ) is True

    def test_400_with_thinking_in_message_is_true(self):
        class _Err(Exception):
            def __init__(self):
                super().__init__("INVALID_ARGUMENT: thinking field rejected")
                self.code = 400

        assert _is_thinking_unsupported(_Err()) is True

    def test_plain_503_transient_is_not_thinking(self):
        # A 503 transient must stay a normal fast-transient, NOT a downgrade.
        assert _is_thinking_unsupported(_TransientError()) is False

    def test_plain_runtime_error_is_not_thinking(self):
        assert _is_thinking_unsupported(RuntimeError("boom")) is False

    def test_400_without_thinking_mention_is_not_thinking(self):
        # A generic 400 that never mentions "thinking" (neither in its class
        # name nor its message) must NOT be classified as a downgrade.
        class _InvalidArgument(Exception):
            def __init__(self):
                super().__init__("400 INVALID_ARGUMENT: bad request")
                self.code = 400
                self.status = "INVALID_ARGUMENT"

        assert _is_thinking_unsupported(_InvalidArgument()) is False


# ---------------------------------------------------------------------------
# Quota exhausted + auth/permission -> distinct, NON-retried errors.
#
# These mirror the REAL google-genai SDK shape: an exception carrying an integer
# ``code`` and a string ``status`` (e.g. "RESOURCE_EXHAUSTED",
# "PERMISSION_DENIED", "UNAUTHENTICATED", "INVALID_ARGUMENT"). They must be
# classified BEFORE the transient check so a 429 goes to quota (no retry) and a
# 401/403 goes to auth (no retry) — neither consumes the bounded retry budget.
# All tests monkeypatch ``time.sleep`` so no real delay is incurred.
# ---------------------------------------------------------------------------


class _QuotaError(Exception):
    """Fake SDK error for an exhausted quota (429 / RESOURCE_EXHAUSTED)."""

    def __init__(
        self,
        message: str = "429 RESOURCE_EXHAUSTED. {'error': {'code': 429}}",
        code: int = 429,
        status: str = "RESOURCE_EXHAUSTED",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class _AuthError(Exception):
    """Fake SDK error for an auth/permission failure (401/403, etc.)."""

    def __init__(
        self,
        message: str = "403 PERMISSION_DENIED. {'error': {'code': 403}}",
        code: int = 403,
        status: str = "PERMISSION_DENIED",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def test_quota_exhausted_not_retried(monkeypatch) -> None:
    """A 429 RESOURCE_EXHAUSTED raises LLMQuotaExceededError after EXACTLY 1 call."""
    monkeypatch.setattr(gemini_module.time, "sleep", lambda _s: None)
    client = _CountingClient([_QuotaError()])
    provider = GeminiLLMProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    with pytest.raises(LLMQuotaExceededError) as exc_info:
        provider.generate("prompt")

    # Quota is intercepted BEFORE the transient check -> NOT the busy subclass.
    assert not isinstance(exc_info.value, LLMServiceBusyError)
    # No retry: exactly one call.
    assert client.models.calls == 1
    # Value-free message.
    assert _SAMPLE_SECRET not in str(exc_info.value)


def test_auth_error_not_retried(monkeypatch) -> None:
    """Auth failures raise LLMAuthError after EXACTLY 1 call (no retry)."""
    monkeypatch.setattr(gemini_module.time, "sleep", lambda _s: None)

    # 403 PERMISSION_DENIED.
    client_403 = _CountingClient([_AuthError()])
    provider_403 = GeminiLLMProvider()
    monkeypatch.setattr(provider_403, "_get_client", lambda: client_403)
    with pytest.raises(LLMAuthError):
        provider_403.generate("prompt")
    assert client_403.models.calls == 1

    # 400 with an "API key not valid" message (API-key-related 400 -> auth).
    client_400 = _CountingClient(
        [_AuthError("400 INVALID_ARGUMENT. API key not valid. Please pass a valid API key.",
                    code=400, status="INVALID_ARGUMENT")]
    )
    provider_400 = GeminiLLMProvider()
    monkeypatch.setattr(provider_400, "_get_client", lambda: client_400)
    with pytest.raises(LLMAuthError):
        provider_400.generate("prompt")
    assert client_400.models.calls == 1

    # 401 UNAUTHENTICATED -> auth.
    client_401 = _CountingClient(
        [_AuthError("401 UNAUTHENTICATED.", code=401, status="UNAUTHENTICATED")]
    )
    provider_401 = GeminiLLMProvider()
    monkeypatch.setattr(provider_401, "_get_client", lambda: client_401)
    with pytest.raises(LLMAuthError):
        provider_401.generate("prompt")
    assert client_401.models.calls == 1


def test_transient_503_still_retried(monkeypatch) -> None:
    """Regression: a 503 UNAVAILABLE still retries to 2 attempts then busy-raises."""
    monkeypatch.setattr(gemini_module.time, "sleep", lambda _s: None)
    client = _CountingClient([_TransientError(), _TransientError()])
    provider = GeminiLLMProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    with pytest.raises(LLMServiceBusyError):
        provider.generate("prompt")

    # The 503 path is unchanged: both bounded attempts were made.
    assert client.models.calls == 2


def test_thinking_400_still_downgrades_not_auth(monkeypatch) -> None:
    """A thinking 400 INVALID_ARGUMENT still downgrades (NOT auth)."""
    monkeypatch.setattr(gemini_module.time, "sleep", lambda _s: None)
    # First call raises the thinking-unsupported 400; the fallback succeeds.
    client = _ConfigRecordingClient([_ThinkingUnsupportedError(), "grounded answer"])
    provider = GeminiLLMProvider()
    monkeypatch.setattr(provider, "_get_client", lambda: client)

    # The thinking 400 must NOT be read as auth.
    assert _is_auth_error(_ThinkingUnsupportedError()) is False
    # The existing downgrade still works end-to-end.
    assert provider.generate("prompt") == "grounded answer"
    assert client.models.calls == 2
    first_cfg, second_cfg = client.models.configs
    assert first_cfg.thinking_config is not None
    assert second_cfg.thinking_config is None


class TestIsQuotaExhausted:
    """Unit checks for the private quota classifier."""

    def test_code_429_is_quota(self):
        assert _is_quota_exhausted(_QuotaError()) is True

    def test_resource_exhausted_status_is_quota(self):
        class _Err(Exception):
            def __init__(self):
                super().__init__("limit")
                self.status = "RESOURCE_EXHAUSTED"

        assert _is_quota_exhausted(_Err()) is True

    def test_quota_text_is_quota(self):
        assert _is_quota_exhausted(Exception("exceeded your quota")) is True

    def test_plain_503_transient_is_not_quota(self):
        assert _is_quota_exhausted(_TransientError()) is False

    def test_plain_runtime_error_is_not_quota(self):
        assert _is_quota_exhausted(RuntimeError("boom")) is False


class TestIsAuthError:
    """Unit checks for the private auth classifier."""

    def test_403_permission_denied_is_auth(self):
        assert _is_auth_error(_AuthError()) is True

    def test_401_unauthenticated_is_auth(self):
        exc = _AuthError("401 UNAUTHENTICATED.", code=401, status="UNAUTHENTICATED")
        assert _is_auth_error(exc) is True

    def test_400_api_key_not_valid_is_auth(self):
        exc = _AuthError(
            "400 INVALID_ARGUMENT. API key not valid.",
            code=400,
            status="INVALID_ARGUMENT",
        )
        assert _is_auth_error(exc) is True

    def test_429_quota_is_not_auth(self):
        assert _is_auth_error(_QuotaError()) is False

    def test_503_transient_is_not_auth(self):
        assert _is_auth_error(_TransientError()) is False

    def test_400_thinking_is_not_auth(self):
        assert _is_auth_error(_ThinkingUnsupportedError()) is False

    def test_plain_runtime_error_is_not_auth(self):
        assert _is_auth_error(RuntimeError("boom")) is False


def test_quota_and_auth_mutually_exclusive() -> None:
    """A 429 is quota-not-auth, and a 403 is auth-not-quota."""
    quota = _QuotaError()
    auth = _AuthError()
    assert _is_quota_exhausted(quota) is True
    assert _is_auth_error(quota) is False
    assert _is_auth_error(auth) is True
    assert _is_quota_exhausted(auth) is False

"""Google Gemini LLM provider (MVP default).

``GeminiLLMProvider`` generates completions with Google Gemini (default model
``gemini-flash-latest``). It is the MVP default LLM provider: its free tier
keeps the deployed demo at zero cost while the :class:`~src.llm.base.LLMProvider`
interface remains the boundary that lets the LLM be swapped later without
redesign.

Implemented against the maintained ``google-genai`` SDK (``from google import
genai``), NOT the legacy ``google-generativeai`` package.

Lazy loading (mirrors ``SentenceTransformersEmbeddingProvider``):
    * ``from google import genai`` is imported INSIDE :meth:`_get_client`, not
      at module top, so importing this module or constructing the provider
      imports no SDK.
    * The ``genai.Client`` is created on first use and cached.

Credential enforcement is lazy: ``GEMINI_API_KEY`` is read from ``Config`` (via
:meth:`Config.require_credential`) inside :meth:`_get_client`, so a missing key
surfaces as a named, value-free :class:`LLMError` on first generation — never at
construction time.

Latency note (model "thinking"):
    * The per-call config DISABLES the model's "thinking" token budget
      (``thinking_config=ThinkingConfig(thinking_budget=0)``). A live test showed
      thinking added ~tens of seconds of latency even for trivial prompts. Some
      models reject ``thinking_config``. Models known to reject it are listed in
      ``_THINKING_UNSUPPORTED_MODELS`` and OMIT ``thinking_config`` from the
      start (their primary config is the no-thinking one). For any OTHER model
      that unexpectedly rejects it, the provider still has a ONE-TIME graceful
      fallback to an identical config WITHOUT ``thinking_config`` as a safety net
      (see :meth:`GeminiLLMProvider.generate`). The fallback is a config
      downgrade, not a backoff, so it does NOT sleep and does NOT consume the
      bounded transient-retry budget.

Security notes (R16.4):
    * The API key is NEVER stored as a plain attribute. It is read lazily from
      ``Config`` (which itself keeps credentials out of its ``repr``) each time
      the client is built, then handed straight to ``genai.Client`` and never
      logged, echoed, or embedded in an error message.
    * Failure messages are generic and value-free (they never include a raw
      provider payload that could carry the key or other sensitive data).
"""

from __future__ import annotations

import random
import time
from typing import Any, Optional

from src.config import Config
from src.llm.base import (
    LLMAuthError,
    LLMError,
    LLMProvider,
    LLMQuotaExceededError,
    LLMServiceBusyError,
)


# ---------------------------------------------------------------------------
# Bounded-retry policy (APPROVED).
#
# Gemini intermittently returns a TRANSIENT 503 UNAVAILABLE ("model experiencing
# high demand ... try again later"). A later attempt typically succeeds, so a
# small bounded retry with backoff turns most of these transient blips into a
# successful answer instead of a user-visible failure.
#
# Policy: at most 2 total attempts, with ONE short backoff (~0.5s) + a small
# random jitter BEFORE the 2nd attempt. Worst-case added latency is ~0.5-0.7s —
# far below the UI's 60s read timeout. The budget was REDUCED from the earlier
# 3-attempt / 1s+2s backoff because, under sustained "high demand", that budget
# could approach or exceed the UI read timeout and surface as a TIMEOUT instead
# of a quick result. These are module-level constants so tests can
# monkeypatch/shorten them (and monkeypatch ``time.sleep``) to keep the suite
# fast.
# ---------------------------------------------------------------------------
_RETRY_MAX_ATTEMPTS = 2
_RETRY_BASE_DELAYS = (0.5,)

# Per-call cap passed to the SDK as an HTTP request timeout (MILLISECONDS). A
# live test proved the Gemini ``generate_content`` call could run 75-331s —
# driven by the SDK's Automatic Function Calling (AFC) round-trips and the
# absence of any per-call timeout — which far exceeds the UI's 60s read timeout
# and surfaced as "taking too long" on every query. Bounding a SINGLE call to
# 45s keeps the worst-case well under the UI's 60s read timeout.
_GENERATE_TIMEOUT_MS = 45000  # 45s per-call cap; keeps total under the UI 60s read timeout

_THINKING_BUDGET = 0  # disable model "thinking" tokens: they added ~tens of seconds of latency even for trivial prompts

# Models known NOT to support thinking_config: calling them WITH it returns a
# generic 400 INVALID_ARGUMENT (whose message does not mention "thinking", so
# the text-based _is_thinking_unsupported matcher would NOT catch it). For
# these models we OMIT thinking_config from the start — an explicit, targeted
# rule, not a broad 400 retry. Verified: gemini-3.5-flash-lite (succeeds in
# ~1.2s without thinking_config; 400s with it).
_THINKING_UNSUPPORTED_MODELS = frozenset({"gemini-3.5-flash-lite"})

# HTTP status codes we treat as transient (retryable).
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Lowercased WORD tokens in the exception's class name / str that indicate a
# transient unavailability / overload / rate-limit condition.
#
# Bare numeric tokens ("500"/"502"/"503"/"504"/"429") and the loose phrase
# "try again" were REMOVED to avoid false positives: digits appear in many
# benign error messages (token counts, request ids, byte sizes, HTTP "200 OK"
# lines), so scanning the message text for them misclassified almost everything
# as transient and retried it. Transient HTTP status is now detected ONLY via
# the structured ``code`` / ``status_code`` / ``status`` attributes — never by
# scanning for digit substrings.
_TRANSIENT_TOKENS = (
    "unavailable",
    "overloaded",
    "rate limit",
    "resource_exhausted",
)


def _is_transient(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a TRANSIENT (retryable) LLM failure.

    Classification is intentionally conservative — when unsure, we return
    ``False`` so the call fails fast (no retry) and behaves exactly as before.
    Detection avoids importing the Google SDK at module top; it inspects the
    caught exception directly:

    * a numeric ``code`` / ``status_code`` attribute in
      :data:`_TRANSIENT_STATUS_CODES` (429/500/502/503/504);
    * a string ``status`` attribute naming an unavailability/overload/rate-limit
      condition (e.g. ``"UNAVAILABLE"``, ``"RESOURCE_EXHAUSTED"``);
    * a transient WORD token appearing in the lowercased ``type(exc).__name__``
      + ``str(exc)`` (e.g. "unavailable", "overloaded", "rate limit",
      "resource_exhausted"). Bare numeric substrings are intentionally NOT
      scanned here — they produced false positives — so digits in a benign
      message no longer trigger a retry.

    Args:
        exc: The exception raised by the underlying SDK call.

    Returns:
        ``True`` if the failure is classified transient, else ``False``.
    """
    # Numeric status via code / status_code attributes.
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        # bool is an int subclass; ignore it defensively.
        if isinstance(value, int) and not isinstance(value, bool):
            if value in _TRANSIENT_STATUS_CODES:
                return True

    # String status attribute (e.g. "UNAVAILABLE", "RESOURCE_EXHAUSTED").
    status_attr = getattr(exc, "status", None)
    if isinstance(status_attr, str):
        lowered = status_attr.lower()
        if any(token in lowered for token in _TRANSIENT_TOKENS):
            return True

    # Scan the class name + message text for transient tokens.
    haystack = f"{type(exc).__name__} {exc}".lower()
    if any(token in haystack for token in _TRANSIENT_TOKENS):
        return True

    return False


# Lowercased tokens in the exception's class name / str that indicate a
# CLIENT-SIDE deadline/timeout (the 45s per-call http timeout fired, or an
# upstream 504 Gateway Timeout). These are specific, safe WORDS — never bare
# digits — so they do not misclassify benign messages that merely contain
# numbers. A timeout is treated as a NON-retried busy condition: the 45s cap
# has already consumed the budget, so a retry would risk exceeding the UI's
# 60s read timeout.
_TIMEOUT_TOKENS = (
    "timeout",
    "timed out",
    "deadline",
    "deadline_exceeded",
)


def _is_timeout(exc: BaseException) -> bool:
    """Return True if ``exc`` indicates a CLIENT-SIDE deadline/timeout.

    A timeout is a distinct kind of transient failure that must NOT be retried:
    the per-call 45s http timeout (:data:`_GENERATE_TIMEOUT_MS`) has already
    consumed essentially the whole budget, so a second attempt would risk
    exceeding the UI's 60s read timeout. It is therefore surfaced immediately as
    a busy condition rather than retried.

    Detection is intentionally specific (safe words, never bare digits):

    * a numeric ``code`` / ``status_code`` attribute equal to ``504`` (Gateway
      Timeout / deadline exceeded); or
    * a timeout WORD token ("timeout", "timed out", "deadline",
      "deadline_exceeded") appearing in the lowercased ``type(exc).__name__`` +
      ``str(exc)``.

    This is checked BEFORE :func:`_is_transient`, so a 504 — which is also in
    :data:`_TRANSIENT_STATUS_CODES` — goes to the no-retry busy branch. That is
    intended.

    Args:
        exc: The exception raised by the underlying SDK call.

    Returns:
        ``True`` if the failure is a client-side deadline/timeout, else
        ``False``.
    """
    # Numeric 504 via code / status_code attributes.
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        # bool is an int subclass; ignore it defensively.
        if isinstance(value, int) and not isinstance(value, bool):
            if value == 504:
                return True

    # Scan the class name + message text for specific timeout tokens.
    haystack = f"{type(exc).__name__} {exc}".lower()
    if any(token in haystack for token in _TIMEOUT_TOKENS):
        return True

    return False


# Lowercased tokens that indicate an exhausted usage quota / rate limit. These
# are specific WORDS, never bare digits, so they do not misclassify a benign
# message that merely contains numbers.
_QUOTA_TOKENS = (
    "resource_exhausted",
    "quota",
)


def _is_quota_exhausted(exc: BaseException) -> bool:
    """Return True if ``exc`` signals an EXHAUSTED usage quota / rate limit.

    This is an HTTP 429 / ``RESOURCE_EXHAUSTED`` condition. It is a distinct,
    NON-retried outcome surfaced as :class:`~src.llm.base.LLMQuotaExceededError`
    — and it is checked BEFORE :func:`_is_transient`, so even though 429 is
    still in :data:`_TRANSIENT_STATUS_CODES`, a 429 is intercepted here and
    never retried.

    Detection is specific and conservative:

    * a numeric ``code`` / ``status_code`` attribute equal to ``429``; OR
    * a string ``status`` attribute equal (case-insensitively) to
      ``"RESOURCE_EXHAUSTED"``; OR
    * a quota WORD token ("resource_exhausted", "quota") appearing in the
      lowercased ``type(exc).__name__`` + ``str(exc)``.

    Args:
        exc: The exception raised by the underlying SDK call.

    Returns:
        ``True`` if the failure is an exhausted quota / rate limit, else
        ``False``.
    """
    # Numeric 429 via code / status_code attributes.
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        # bool is an int subclass; ignore it defensively.
        if isinstance(value, int) and not isinstance(value, bool):
            if value == 429:
                return True

    # String status attribute equal to RESOURCE_EXHAUSTED (case-insensitive).
    status_attr = getattr(exc, "status", None)
    if isinstance(status_attr, str) and status_attr.lower() == "resource_exhausted":
        return True

    # Scan the class name + message text for quota tokens.
    haystack = f"{type(exc).__name__} {exc}".lower()
    if any(token in haystack for token in _QUOTA_TOKENS):
        return True

    return False


# Lowercased string ``status`` values that indicate an auth/permission failure.
_AUTH_STATUS_VALUES = frozenset({"unauthenticated", "permission_denied"})

# Lowercased tokens in the class name / str that clearly indicate an
# auth/permission failure. These are specific phrases, never bare digits.
_AUTH_TOKENS = (
    "unauthenticated",
    "permission_denied",
    "api key not valid",
    "api_key_invalid",
    "invalid api key",
    "invalid authentication",
)

# Lowercased tokens that indicate a message clearly concerns the API
# key / credential (used to treat a 400 / INVALID_ARGUMENT as auth ONLY when it
# is credential-related, so a generic 400 or the thinking-config 400 are NOT
# misread as auth).
_CREDENTIAL_TOKENS = (
    "api key",
    "api_key",
    "credential",
)


def _is_auth_error(exc: BaseException) -> bool:
    """Return True if ``exc`` signals an AUTH / permission failure.

    This covers a rejected or unauthorized credential surfaced as
    :class:`~src.llm.base.LLMAuthError`. It is a hard config error, NOT
    transient, so it is NOT retried. It is checked BEFORE :func:`_is_transient`.

    Detection is specific and conservative:

    * a numeric ``code`` / ``status_code`` attribute in ``{401, 403}``; OR
    * a string ``status`` attribute (lowercased) equal to ``"unauthenticated"``
      or ``"permission_denied"``; OR
    * an auth WORD/phrase token ("unauthenticated", "permission_denied",
      "api key not valid", "api_key_invalid", "invalid api key",
      "invalid authentication") appearing in the lowercased
      ``type(exc).__name__`` + ``str(exc)``; OR
    * a 400 / INVALID_ARGUMENT ONLY when the message clearly concerns the API
      key / credential (contains "api key" / "api_key" / "credential") — so a
      generic 400 and the thinking-config 400 are NOT misread as auth.

    Args:
        exc: The exception raised by the underlying SDK call.

    Returns:
        ``True`` if the failure is an auth/permission rejection, else ``False``.
    """
    haystack = f"{type(exc).__name__} {exc}".lower()

    # Numeric 401 / 403 via code / status_code attributes.
    is_400 = False
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        # bool is an int subclass; ignore it defensively.
        if isinstance(value, int) and not isinstance(value, bool):
            if value in (401, 403):
                return True
            if value == 400:
                is_400 = True

    # String status attribute naming an auth/permission failure.
    status_attr = getattr(exc, "status", None)
    if isinstance(status_attr, str):
        lowered = status_attr.lower()
        if lowered in _AUTH_STATUS_VALUES:
            return True
        if lowered == "invalid_argument":
            is_400 = True

    # Explicit auth phrases in the class name / message.
    if any(token in haystack for token in _AUTH_TOKENS):
        return True

    # A 400 / INVALID_ARGUMENT is auth ONLY when it clearly concerns the
    # credential — otherwise a generic 400 (or the thinking-config 400) would
    # be misread as auth.
    if is_400 and any(token in haystack for token in _CREDENTIAL_TOKENS):
        return True

    return False


def _is_thinking_unsupported(exc: BaseException) -> bool:
    """Return True if ``exc`` is a model's rejection of ``thinking_config``.

    The ``google-genai`` SDK warns that setting ``thinking_config`` on a model
    that does not support thinking returns an error (typically a 400 /
    INVALID_ARGUMENT whose message names the unsupported ``thinking`` field).
    When this is detected, :meth:`GeminiLLMProvider.generate` performs a
    ONE-TIME graceful fallback to an identical config WITHOUT ``thinking_config``
    — a config downgrade, NOT a transient retry (it does not sleep and does not
    consume the bounded transient-retry budget).

    Detection is intentionally SPECIFIC so a generic 400 (or any unrelated
    failure) is NOT misread as a thinking downgrade:

    * the word "thinking" appears in the lowercased ``type(exc).__name__`` +
      ``str(exc)`` (the SDK's error names the ``thinking`` field); OR
    * it is a 400 / INVALID_ARGUMENT (numeric ``code`` / ``status_code`` == 400,
      or a ``status`` of ``"INVALID_ARGUMENT"``) whose message mentions
      "thinking".

    Args:
        exc: The exception raised by the underlying SDK call.

    Returns:
        ``True`` if the failure is a thinking-unsupported rejection, else
        ``False``.
    """
    haystack = f"{type(exc).__name__} {exc}".lower()

    # The SDK's thinking rejection names the "thinking" field in its message /
    # error class. That naming alone is a specific, reliable signal.
    if "thinking" in haystack:
        return True

    # Otherwise require BOTH a 400 / INVALID_ARGUMENT shape AND a "thinking"
    # mention — but the mention is already covered above, so a 400 without any
    # "thinking" text is deliberately NOT treated as a downgrade.
    is_invalid_argument = False
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        # bool is an int subclass; ignore it defensively.
        if isinstance(value, int) and not isinstance(value, bool) and value == 400:
            is_invalid_argument = True
            break
    status_attr = getattr(exc, "status", None)
    if isinstance(status_attr, str) and status_attr.lower() == "invalid_argument":
        is_invalid_argument = True

    return is_invalid_argument and "thinking" in haystack


class GeminiLLMProvider(LLMProvider):
    """Generates completions using Google Gemini via the ``google-genai`` SDK.

    The underlying ``genai.Client`` is lazy-loaded and cached on first
    ``generate`` call. No domain-specific logic lives here (R15.5) — it is a
    thin adapter over the SDK.
    """

    def __init__(
        self,
        model_name: str = "gemini-flash-latest",
        config: Optional[Config] = None,
    ) -> None:
        """Initialize the provider without creating a client or reading the key.

        The credential is intentionally NOT read here (lazy enforcement): the
        provider only stores the ``Config`` reference so it can read
        ``GEMINI_API_KEY`` on first use. ``Config`` keeps credentials out of
        its ``repr``, so holding the reference does not risk leaking the key.

        Args:
            model_name: The Gemini model id to use. Defaults to
                ``'gemini-flash-latest'``.
            config: The active configuration, used to read ``GEMINI_API_KEY``
                lazily on first generation. When ``None``, a default
                :class:`Config` (env-only) is used.
        """
        self._model_name = model_name
        self._config: Config = config if config is not None else Config()
        # Cached genai.Client; ``None`` until the first generate call.
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Return the cached ``genai.Client``, creating it on first use.

        ``from google import genai`` is imported HERE (not at module top) so
        importing this module or constructing the provider does not import the
        SDK. The API key is read from ``Config`` here (lazy enforcement) and
        handed straight to the client.

        Returns:
            The cached ``genai.Client`` instance.

        Raises:
            LLMError: If ``GEMINI_API_KEY`` is unset (named, value-free), or if
                the SDK cannot be imported or the client cannot be created.
        """
        if self._client is None:
            # Read the credential lazily; re-raise as a named, value-free error.
            try:
                api_key = self._config.require_credential("GEMINI_API_KEY")
            except Exception as exc:
                # Config's message names GEMINI_API_KEY only, never the value.
                raise LLMError(str(exc)) from exc

            try:
                from google import genai

                self._client = genai.Client(api_key=api_key)
            except LLMError:
                raise
            except Exception as exc:
                # Generic, value-free message — no payload that could leak.
                raise LLMError("Failed to initialize the Gemini client") from exc
        return self._client

    @property
    def provider_name(self) -> str:
        """Return the stable provider id ``'gemini'``."""
        return "gemini"

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        return self._model_name

    def generate(self, prompt: str) -> str:
        """Generate a completion for ``prompt`` using Gemini.

        Args:
            prompt: The fully-assembled prompt to send to the model.

        Returns:
            The model's text completion as a ``str``.

        Raises:
            LLMError: If the credential is missing, the model returns no text,
                or the underlying SDK call fails. Messages are value-free.
        """
        client = self._get_client()

        # Build the per-call configs ONCE and reuse them for every attempt.
        #
        # ``types`` is imported LAZILY here (same pattern as ``from google import
        # genai`` in ``_get_client``) so importing this module stays SDK-free.
        #
        #   * Automatic Function Calling (AFC) is DISABLED: we send a plain text
        #     prompt and expect plain text back — function calling is unused, and
        #     its SDK round-trips were the measured cause of multi-minute latency.
        #   * The http timeout bounds a SINGLE call to 45s so it can never
        #     approach the UI's 60s read timeout.
        #   * The model's "thinking" token budget is DISABLED
        #     (``thinking_config=ThinkingConfig(thinking_budget=0)``): thinking
        #     added ~tens of seconds of latency even for trivial prompts.
        from google.genai import types

        # ``cfg_without_thinking`` is the EXACT config we used before this fix
        # (AFC disabled + 45s http timeout). It is the fallback used only when a
        # model rejects ``thinking_config``.
        cfg_without_thinking = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            http_options=types.HttpOptions(timeout=_GENERATE_TIMEOUT_MS),
        )
        # ``cfg_with_thinking`` is identical but ALSO disables the thinking
        # budget. It is the primary config for normal models; models in
        # _THINKING_UNSUPPORTED_MODELS use ``cfg_without_thinking`` as primary
        # instead (see the primary_cfg selection below).
        cfg_with_thinking = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            http_options=types.HttpOptions(timeout=_GENERATE_TIMEOUT_MS),
            thinking_config=types.ThinkingConfig(thinking_budget=_THINKING_BUDGET),
        )

        # Models in _THINKING_UNSUPPORTED_MODELS must never receive thinking_config
        # (they reject it with a 400). For them the primary config IS the no-thinking
        # config. All other models keep the thinking-disabled config as primary, with
        # the existing one-time _is_thinking_unsupported fallback as a safety net.
        if self._model_name in _THINKING_UNSUPPORTED_MODELS:
            primary_cfg = cfg_without_thinking
        else:
            primary_cfg = cfg_with_thinking

        def _call(cfg: Any) -> Any:
            """Invoke the SDK once with ``cfg``; return the response or raise."""
            return client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=cfg,
            )

        # Bounded retry loop (APPROVED policy). Failures are classified in a
        # fixed order:
        #   0. THINKING-UNSUPPORTED on the primary config -> ONE-TIME graceful
        #      fallback to the no-thinking config. This is a safety net for a
        #      model NOT in _THINKING_UNSUPPORTED_MODELS that unexpectedly
        #      rejects thinking_config (allowlisted models already use the
        #      no-thinking config as primary, so they never hit this). It is a
        #      config DOWNGRADE, not a backoff: it does NOT sleep and does NOT
        #      consume the transient-retry budget. If that fallback call also
        #      raises, its exception is classified normally (below) — so a 429
        #      or auth error from the fallback still routes to quota/auth.
        #   1. QUOTA exhausted (429 / RESOURCE_EXHAUSTED) -> LLMQuotaExceededError
        #      IMMEDIATELY, no retry/sleep. Checked BEFORE _is_transient so a 429
        #      (still in _TRANSIENT_STATUS_CODES) is intercepted as quota, not
        #      retried. The transient set is deliberately left unchanged so the
        #      503 policy is untouched.
        #   2. AUTH/permission rejected -> LLMAuthError IMMEDIATELY, no
        #      retry/sleep. Value-free: names the condition, never the key.
        #   3. TIMEOUT/deadline -> busy IMMEDIATELY, no retry (the 45s cap
        #      already consumed the budget; retrying would risk the UI 60s).
        #   4. NON-transient -> fail fast as LLMError, no retry.
        #   5. FAST transient (e.g. a quick 503/overload) -> bounded retry.
        last_exc: Optional[BaseException] = None
        for attempt in range(1, _RETRY_MAX_ATTEMPTS + 1):
            try:
                # ``primary_cfg`` is model-dependent: the thinking-disabled config
                # for normal models, or the no-thinking config for models in
                # _THINKING_UNSUPPORTED_MODELS (which reject thinking_config).
                response = _call(primary_cfg)
            except Exception as exc:
                # Step 0: a model that rejects thinking_config. Downgrade ONCE,
                # inline, to the no-thinking config. Whatever that fallback call
                # raises (if anything) falls through to the normal timeout /
                # transient / non-transient classification for THIS attempt —
                # the downgrade itself does not count as a transient retry.
                if _is_thinking_unsupported(exc):
                    try:
                        response = _call(cfg_without_thinking)
                    except Exception as fallback_exc:
                        exc = fallback_exc
                    else:
                        # Fallback succeeded; skip classification and return.
                        text = getattr(response, "text", None)
                        if text is None:
                            raise LLMError("Gemini returned no text")
                        return str(text)

                if _is_quota_exhausted(exc):
                    # Exhausted usage quota / rate limit (429 / RESOURCE_EXHAUSTED).
                    # NON-retried: retrying would not help and would only consume
                    # more of an already-exhausted budget. Checked BEFORE
                    # _is_transient so a 429 lands here rather than being retried.
                    # Value-free message: names the condition, never the key or
                    # the provider payload.
                    raise LLMQuotaExceededError(
                        "The usage quota for the language model has been reached."
                    ) from exc

                if _is_auth_error(exc):
                    # Rejected / unauthorized credential (401/403,
                    # PERMISSION_DENIED/UNAUTHENTICATED, or an API-key-related
                    # 400). NON-retried: a rejected credential never succeeds on
                    # retry. Value-free: names the condition only — never the key
                    # or the provider response body.
                    raise LLMAuthError(
                        "The language model credential was rejected."
                    ) from exc

                if _is_timeout(exc):
                    # Client-side deadline/timeout (or an upstream 504). Do NOT
                    # retry — go straight to busy so we never approach the UI's
                    # 60s read timeout. Checked BEFORE _is_transient so a 504
                    # (also transient) lands here intentionally.
                    raise LLMServiceBusyError(
                        "The language model took too long to respond. "
                        "Please try again shortly."
                    ) from exc

                if not _is_transient(exc):
                    # Non-transient: value-free message, no retry. Do not include
                    # the exception's raw text (it could echo request/response
                    # payload). Chain for local debugging only.
                    raise LLMError("Gemini generation failed") from exc

                last_exc = exc
                if attempt < _RETRY_MAX_ATTEMPTS:
                    # Backoff BEFORE the next attempt, with a small jitter.
                    base = _RETRY_BASE_DELAYS[attempt - 1]
                    time.sleep(base * (1 + random.uniform(0, 0.25)))
                    continue

                # Last attempt also transient: surface the distinct busy error.
                raise LLMServiceBusyError(
                    "The language model is temporarily unavailable. "
                    "Please try again shortly."
                ) from exc

            text = getattr(response, "text", None)
            if text is None:
                raise LLMError("Gemini returned no text")
            return str(text)

        # Unreachable: the loop either returns, or raises on the last attempt.
        # Kept as a defensive fallback so the function always has a clear exit.
        raise LLMServiceBusyError(
            "The language model is temporarily unavailable. "
            "Please try again shortly."
        ) from last_exc

    def __repr__(self) -> str:  # pragma: no cover - trivial formatting
        """Return a repr that exposes no secret.

        The API key is never a stored attribute, and ``Config`` omits
        credentials from its own ``repr``, so this repr is safe to log.
        """
        return f"GeminiLLMProvider(model_name={self._model_name!r})"

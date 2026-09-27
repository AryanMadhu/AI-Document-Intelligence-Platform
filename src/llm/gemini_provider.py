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

Security notes (R16.4):
    * The API key is NEVER stored as a plain attribute. It is read lazily from
      ``Config`` (which itself keeps credentials out of its ``repr``) each time
      the client is built, then handed straight to ``genai.Client`` and never
      logged, echoed, or embedded in an error message.
    * Failure messages are generic and value-free (they never include a raw
      provider payload that could carry the key or other sensitive data).
"""

from __future__ import annotations

from typing import Any, Optional

from src.config import Config
from src.llm.base import LLMError, LLMProvider


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
        try:
            response = client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
        except Exception as exc:
            # Value-free: do not include the exception's raw text, which could
            # echo the request/response payload. Chain for local debugging only.
            raise LLMError("Gemini generation failed") from exc

        text = getattr(response, "text", None)
        if text is None:
            raise LLMError("Gemini returned no text")
        return str(text)

    def __repr__(self) -> str:  # pragma: no cover - trivial formatting
        """Return a repr that exposes no secret.

        The API key is never a stored attribute, and ``Config`` omits
        credentials from its own ``repr``, so this repr is safe to log.
        """
        return f"GeminiLLMProvider(model_name={self._model_name!r})"

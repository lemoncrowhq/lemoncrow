"""Background keepalive thread for owned agent sessions.

Provider-specific behaviour:
- **Anthropic**: fires a tiny completion every 4.5 min to prevent the
  5-minute prompt-cache TTL from expiring.
- **Gemini**: refreshes the server-side ``cachedContent`` TTL every 4 min
  (Gemini's minimum TTL is 5 min — refresh before it expires).
- **OpenAI**: no-op — OpenAI automatic prefix caching has no TTL to manage.
- **Others**: no-op by default.
"""

from __future__ import annotations

import logging
import threading

from lemoncrow.pro.capabilities.owned_agent_session.gemini_cache import GeminiContextCache

logger = logging.getLogger(__name__)

_ANTHROPIC_INTERVAL_SECONDS = 270  # 4.5 min — fires before Anthropic's 5-min TTL
_GEMINI_INTERVAL_SECONDS = 240  # 4.0 min — fires before Gemini's 5-min TTL


class KeepaliveThread:
    """Background daemon thread that keeps the provider's cache warm.

    Args:
        model: The provider-prefixed litellm model string (e.g.
            ``"anthropic/claude-opus-4-8"``).
        provider: The provider name from ``OwnedRouteDecision.provider``
            (e.g. ``"anthropic"``, ``"openai"``, ``"google"``).
        gemini_cache: Optional :class:`GeminiContextCache` instance used to
            refresh the Gemini server-side TTL.  Required when *provider* is
            ``"google"`` and Gemini context caching is in use.
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str = "",
        gemini_cache: GeminiContextCache | None = None,
    ) -> None:
        from lemoncrow.pro.capabilities.owned_agent_session.phase_runner import _provider_cache_style

        self._model = model
        self._cache_style = _provider_cache_style(provider)
        self._gemini_cache = gemini_cache
        interval = (
            _ANTHROPIC_INTERVAL_SECONDS
            if self._cache_style == "anthropic"
            else _GEMINI_INTERVAL_SECONDS if self._cache_style == "gemini" else 0.0  # no-op for openai / others
        )
        self._interval = interval
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="lemoncrow-keepalive")

    def start(self) -> None:
        if self._interval > 0:
            self._worker.start()
        # else: no-op provider (OpenAI / unknown) — don't start thread at all

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._ping()
            except Exception:  # noqa: BLE001
                logger.debug("keepalive ping failed (non-fatal)", exc_info=True)

    def _ping(self) -> None:
        if self._cache_style == "anthropic":
            self._ping_anthropic()
        elif self._cache_style == "gemini":
            self._ping_gemini()
        # openai / none: nothing to do

    def _ping_anthropic(self) -> None:
        try:
            from lemoncrow.infra.internal_llm.litellm_client import chat_with_result
        except Exception:  # noqa: BLE001
            return
        try:
            chat_with_result([{"role": "user", "content": "ping"}], model=self._model)
        except Exception:  # noqa: BLE001
            pass

    def _ping_gemini(self) -> None:
        if self._gemini_cache is not None:
            self._gemini_cache.refresh()
        else:
            # No context cache available — fall back to a tiny completion,
            # routed through the infra litellm wrapper to keep the provider SDK
            # confined to src/lemoncrow/infra/internal_llm/.
            try:
                from lemoncrow.infra.internal_llm.litellm_client import chat_with_result
            except Exception:  # noqa: BLE001
                return
            try:
                chat_with_result([{"role": "user", "content": "ping"}], model=self._model)
            except Exception:  # noqa: BLE001
                pass


__all__ = ["KeepaliveThread"]

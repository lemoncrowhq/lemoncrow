"""Gemini Context Caching via REST API.

Creates a server-side context cache for the stable system prompt,
then passes ``cachedContent`` in subsequent litellm completions so the
model reads cached tokens at ~0.25x the cost of fresh input.

Gemini context caching reference:
  https://ai.google.dev/gemini-api/docs/caching

TTL is 5 minutes (the Gemini minimum), refreshed by keepalive before expiry.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_TTL = "300s"  # 5 min — Gemini minimum


def _gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    return key.strip()


def _request(method: str, url: str, body: dict[str, object] | None = None) -> dict[str, object]:
    key = _gemini_api_key()
    if not key:
        raise RuntimeError("No GEMINI_API_KEY or GOOGLE_API_KEY found in environment")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return dict(json.loads(resp.read()))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace") if exc.fp else ""
        raise RuntimeError(f"Gemini cache API {method} {url}: {exc.code} {body_text}") from exc


@dataclass
class GeminiContextCache:
    """Manages one Gemini server-side context cache for a session.

    Usage::

        cache = GeminiContextCache.create(model="gemini-2.0-flash", system_prompt="...")
        # pass cache.name in litellm extra_body
        litellm.completion(..., extra_body={"cachedContent": cache.name})
        # refresh before 5-min TTL expires (keepalive does this)
        cache.refresh()
    """

    name: str  # e.g. "cachedContents/abc123"
    model: str
    _errors: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def create(cls, *, model: str, system_prompt: str, ttl: str = _DEFAULT_TTL) -> GeminiContextCache:
        """Create a server-side context cache for *system_prompt*."""
        from lemoncrow.pro.capabilities.optimization.cache_economics import valid_gemini_ttl

        ttl = valid_gemini_ttl(ttl)
        # Gemini model names in the cache API use "models/" prefix
        api_model = model if model.startswith("models/") else f"models/{model.split('/')[-1]}"
        body: dict[str, object] = {
            "model": api_model,
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "ttl": ttl,
        }
        result = _request("POST", f"{_GEMINI_API_BASE}/cachedContents", body)
        name = str(result.get("name", ""))
        if not name:
            raise RuntimeError(f"Gemini cache creation returned no name: {result}")
        logger.debug("Created Gemini context cache: %s (model=%s)", name, model)
        return cls(name=name, model=model)

    def refresh(self) -> None:
        """Extend TTL to prevent expiry. Called by keepalive thread."""
        try:
            _request("PATCH", f"{_GEMINI_API_BASE}/{self.name}", {"ttl": _DEFAULT_TTL})
            logger.debug("Refreshed Gemini context cache TTL: %s", self.name)
        except Exception as exc:
            logger.debug("Gemini cache refresh failed (non-fatal): %s", exc)

    def delete(self) -> None:
        """Delete the server-side cache (best-effort cleanup)."""
        try:
            _request("DELETE", f"{_GEMINI_API_BASE}/{self.name}")
            logger.debug("Deleted Gemini context cache: %s", self.name)
        except Exception as exc:
            logger.debug("Gemini cache delete failed (non-fatal): %s", exc)


__all__ = ["GeminiContextCache"]

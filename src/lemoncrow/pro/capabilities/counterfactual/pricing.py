"""Vendor price tables and CandidateModel definitions.

See docs/plans/active/commercial-wedge/W2-counterfactual.md for the full spec.

Pricing is sourced from the main ``lemoncrow.core.capabilities.pricing`` module
(LiteLLM model cost catalog + ``pricing.yaml`` overrides), falling back to
the static prices defined in ``_STATIC_FALLBACK_PRICES`` when LiteLLM does
not know a model.  This keeps routing simulation in sync with actual cost
tracking instead of maintaining a separate price table.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from lemoncrow.core.capabilities.pricing import (
    ModelPricing,
    _load_pricing_table,
    get_model_pricing,
)


@dataclass(frozen=True)
class CandidateModel:
    """A vendor+model pair with pricing and capability metadata."""

    vendor: str
    model_id: str
    tier: str  # "cheap" | "high"
    pricing: ModelPricing
    supports_tool_use: bool = True
    output_multiplier: float = 1.0
    context_window: int = 200_000


@dataclass(frozen=True)
class PricingTable:
    """Versioned collection of CandidateModel entries."""

    version: str
    candidates: tuple[CandidateModel, ...]

    def candidates_for_vendor(self, vendor: str) -> tuple[CandidateModel, ...]:
        return tuple(c for c in self.candidates if c.vendor == vendor)


# ---------------------------------------------------------------------------
# Candidate specs: (vendor, model_id, tier, supports_tool_use, output_multiplier,
#                   context_window, fallback_input_usd, fallback_output_usd)
# Prices are looked up from the main pricing module (LiteLLM + pricing.yaml).
# Fallback prices are used only when LiteLLM does not know the model.
# ---------------------------------------------------------------------------

_CANDIDATE_SPECS: tuple[tuple[str, str, str, bool, float, int, float, float], ...] = (
    # Anthropic
    ("anthropic", "claude-haiku-4-5", "cheap", True, 1.0, 200_000, 0.80, 4.00),
    ("anthropic", "claude-sonnet-4-5", "high", True, 1.0, 200_000, 3.00, 15.00),
    ("anthropic", "claude-opus-4-5", "high", True, 1.5, 200_000, 15.00, 75.00),
    ("anthropic", "claude-fable-5", "high", True, 1.0, 1_000_000, 10.00, 50.00),
    # OpenAI
    ("openai", "gpt-4o-mini", "cheap", True, 1.0, 128_000, 0.15, 0.60),
    ("openai", "gpt-4o", "high", True, 1.0, 128_000, 2.50, 10.00),
    # Google
    ("google", "gemini-2.0-flash", "cheap", True, 1.0, 1_000_000, 0.10, 0.40),
    ("google", "gemini-2.0-pro", "high", True, 1.0, 2_000_000, 1.25, 5.00),
    # AWS Bedrock
    ("bedrock", "bedrock/anthropic.claude-haiku-4-5-v1:0", "cheap", True, 1.0, 200_000, 0.80, 4.00),
    ("bedrock", "bedrock/anthropic.claude-sonnet-4-5-v1:0", "high", True, 1.0, 200_000, 3.00, 15.00),
    ("bedrock", "bedrock/us.anthropic.claude-sonnet-4-6", "high", True, 1.0, 200_000, 3.00, 15.00),
    # GCP Vertex AI
    ("vertex", "vertex_ai/gemini-2.0-flash", "cheap", True, 1.0, 1_000_000, 0.075, 0.30),
    ("vertex", "vertex_ai/claude-3-5-sonnet@20241022", "high", True, 1.0, 200_000, 3.00, 15.00),
    # Azure OpenAI
    ("azure", "azure/gpt-4o-mini", "cheap", True, 1.0, 128_000, 0.15, 0.60),
    ("azure", "azure/gpt-4o", "high", True, 1.0, 128_000, 2.50, 10.00),
    # OpenRouter
    ("openrouter", "openrouter/anthropic/claude-haiku-4-5", "cheap", True, 1.0, 200_000, 0.90, 4.50),
    ("openrouter", "openrouter/anthropic/claude-sonnet-4-5", "high", True, 1.0, 200_000, 3.30, 16.50),
    # Groq
    ("groq", "groq/llama-3.3-70b-versatile", "cheap", True, 1.0, 128_000, 0.59, 0.79),
    ("groq", "groq/llama-3.1-8b-instant", "cheap", False, 1.0, 128_000, 0.05, 0.08),
    # Mistral
    ("mistral", "mistral/mistral-large-latest", "high", True, 1.0, 128_000, 2.00, 6.00),
    ("mistral", "mistral/mistral-small-latest", "cheap", True, 1.0, 128_000, 0.20, 0.60),
    # Ollama (local — near-zero cost for routing logic)
    ("ollama", "ollama/llama3.2", "cheap", False, 1.0, 128_000, 0.001, 0.001),
    ("ollama", "ollama/qwen2.5-coder:7b", "cheap", False, 1.0, 128_000, 0.001, 0.001),
    # Together AI
    ("together", "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "cheap", True, 1.0, 128_000, 0.88, 0.88),
    # OpenCode Zen (free tier: zero-cost models, served against the "public"
    # bearer token when the user has no Zen account)
    ("zen", "zen/big-pickle", "high", True, 1.0, 200_000, 0.0, 0.0),
    ("zen", "zen/deepseek-v4-flash-free", "cheap", True, 1.0, 128_000, 0.0, 0.0),
    # Fireworks AI
    (
        "fireworks",
        "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct",
        "cheap",
        True,
        1.0,
        128_000,
        0.90,
        0.90,
    ),
)


def _build_candidates() -> tuple[CandidateModel, ...]:
    """Build candidates, pulling live prices from the main pricing module with static fallback."""
    candidates: list[CandidateModel] = []
    for vendor, model_id, tier, supports_tool_use, output_multiplier, context_window, fb_in, fb_out in _CANDIDATE_SPECS:
        mp = get_model_pricing(model_id)
        if not mp.known:
            mp = ModelPricing(model_id=model_id, input=fb_in, output=fb_out)
        candidates.append(
            CandidateModel(
                vendor=vendor,
                model_id=model_id,
                tier=tier,
                pricing=mp,
                supports_tool_use=supports_tool_use,
                output_multiplier=output_multiplier,
                context_window=context_window,
            )
        )
    return tuple(candidates)


_cached_table: PricingTable | None = None
_cached_source_id: int | None = None
# Serialises the (expensive) first build so a concurrent caller waits for the
# in-flight computation instead of duplicating it -- ``lru_cache`` on the main
# module's ``_load_pricing_table`` does not deduplicate in-flight calls.  The
# MCP server pre-warms this table on a background thread at import
# (``mcp_server._warm_pricing_table``); without the lock a first tool call
# racing that warm would rebuild the whole table on the response path.
_build_lock = threading.Lock()


def load_pricing_table(_version: str | None = None) -> PricingTable:
    """Return the pricing table, building it lazily from the main pricing module.

    The cache is keyed off the identity of the main module's pricing table so
    that ``override_pricing()`` and ``pricing.yaml`` edits (which clear that
    table's cache and produce a fresh dict) force a rebuild here too, instead
    of routing off stale rates for the process lifetime.
    """
    global _cached_table, _cached_source_id
    with _build_lock:
        source_id = id(_load_pricing_table())
        if _cached_table is None or source_id != _cached_source_id:
            _cached_table = PricingTable(version="live", candidates=_build_candidates())
            _cached_source_id = source_id
        return _cached_table


__all__ = ["CandidateModel", "ModelPricing", "PricingTable", "load_pricing_table"]

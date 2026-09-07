"""Prefix-cache eviction economics for model routing."""

from __future__ import annotations

from math import isfinite

from lemoncrow.core.capabilities.pricing import ModelPricing, fallback_cost_usd
from lemoncrow.pro.capabilities.prefix_cache.planner import PrefixCachePlan


def cache_eviction_cost_usd(plan_a: PrefixCachePlan, plan_b: PrefixCachePlan, pricing: ModelPricing) -> float:
    """Estimate the USD cost of switching away from a reusable prefix cache."""
    if plan_a.prefix_hash == plan_b.prefix_hash:
        return 0.0

    tokens = max(0, int(max(plan_a.prefix_tokens, plan_b.prefix_tokens)))
    cache_write_cost = _finite_non_negative(pricing.tokens_to_usd(tokens, "cache_write"))
    if cache_write_cost > 0.0:
        return cache_write_cost

    input_cost = _finite_non_negative(pricing.tokens_to_usd(tokens, "input"))
    if input_cost > 0.0:
        return input_cost

    return _finite_non_negative(round(fallback_cost_usd(input_tokens=tokens), 8))


def _finite_non_negative(value: float) -> float:
    if not isfinite(value) or value < 0.0:
        return 0.0
    return float(value)

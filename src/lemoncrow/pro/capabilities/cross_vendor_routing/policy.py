"""Policy helpers for cross-vendor routing."""

from __future__ import annotations

from lemoncrow.core.capabilities.cross_vendor_routing_contract import RoutePolicyError as RoutePolicyError
from lemoncrow.pro.capabilities.counterfactual.capabilities import classify_turn_kind

from .configuration import RouteConfig


def turn_kind_for_tool(tool_name: str) -> str:
    return classify_turn_kind(tool_name)


def allowed_vendors(
    config: RouteConfig,
    *,
    tool_name: str,
    actual_vendor: str | None,
    configured_vendors: tuple[str, ...],
) -> tuple[str, ...]:
    turn_kind = turn_kind_for_tool(tool_name)
    if turn_kind == "edit" and config.edit_mode == "pin-actual-vendor":
        vendor = (actual_vendor or "").strip().lower()
        if not vendor:
            raise RoutePolicyError("actual_vendor is required for edit routing when edit_mode pins to actual vendor")
        if vendor not in configured_vendors:
            raise RoutePolicyError(f"actual vendor {vendor!r} is not configured for routing")
        return (vendor,)
    return configured_vendors


__all__ = ["RoutePolicyError", "allowed_vendors", "turn_kind_for_tool"]

"""Ordering-related lint rules."""

from __future__ import annotations

from ..compiler import CompiledPrompt
from ..models import Stability
from .common import FindingData, RuleSpec


def _volatile_before_stable(compiled: CompiledPrompt, _previous: CompiledPrompt | None) -> list[FindingData]:
    findings: list[FindingData] = []
    if compiled.prefix_end_index < 0:
        return findings

    for index, block in enumerate(compiled.blocks):
        if index <= compiled.prefix_end_index and block.stability in {
            Stability.TURN,
            Stability.VOLATILE,
        }:
            findings.append(
                {
                    "rule_id": "ordering.volatile-before-stable",
                    "severity": "error",
                    "block_id": block.id,
                    "message": (
                        f"Block {block.id!r} with stability {block.stability.value!r} "
                        f"appears before stable prefix end index {compiled.prefix_end_index}."
                    ),
                    "fix_hint": "Move TURN/VOLATILE blocks after all STATIC/SESSION/BRANCH blocks.",
                }
            )
    return findings


def _session_before_static(compiled: CompiledPrompt, _previous: CompiledPrompt | None) -> list[FindingData]:
    findings: list[FindingData] = []
    seen_session = False
    for block in compiled.blocks:
        if block.stability is Stability.SESSION:
            seen_session = True
        if seen_session and block.stability is Stability.STATIC:
            findings.append(
                {
                    "rule_id": "ordering.session-before-static",
                    "severity": "warn",
                    "block_id": block.id,
                    "message": "A STATIC block appears after a SESSION block in compile order.",
                    "fix_hint": "Keep all STATIC blocks ahead of SESSION blocks.",
                }
            )
    return findings


def _stability_override(compiled: CompiledPrompt, _previous: CompiledPrompt | None) -> list[FindingData]:
    findings: list[FindingData] = []
    for block in compiled.blocks:
        if block.stability_override_reason:
            findings.append(
                {
                    "rule_id": "ordering.unstable-stability-override",
                    "severity": "warn",
                    "block_id": block.id,
                    "message": (
                        f"Block {block.id!r} overrides default stability; reason={block.stability_override_reason!r}."
                    ),
                    "fix_hint": "Confirm this override is required for cache safety.",
                }
            )
    return findings


RULES: list[RuleSpec] = [
    RuleSpec(
        rule_id="ordering.volatile-before-stable",
        severity="error",
        checker=_volatile_before_stable,
    ),
    RuleSpec(
        rule_id="ordering.session-before-static",
        severity="warn",
        checker=_session_before_static,
    ),
    RuleSpec(
        rule_id="ordering.unstable-stability-override",
        severity="warn",
        checker=_stability_override,
    ),
]

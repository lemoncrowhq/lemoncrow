"""Shared rule descriptors for prompt-compilation lint modules."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, TypedDict

from ..compiler import CompiledPrompt

SeverityName = Literal["error", "warn", "info"]


class FindingData(TypedDict):
    rule_id: str
    severity: SeverityName
    block_id: str | None
    message: str
    fix_hint: str | None


RuleChecker = Callable[[CompiledPrompt, CompiledPrompt | None], Iterable[FindingData]]


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    severity: SeverityName
    checker: RuleChecker

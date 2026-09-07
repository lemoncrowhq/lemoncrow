"""Capability wrapper for prompt compilation."""

from __future__ import annotations

from collections.abc import Sequence

from lemoncrow.core.capabilities.telemetry import TelemetrySubstrate

from .compiler import CompiledPrompt, compile_prompt
from .models import PromptBlock


class PromptCompilerCapability:
    """Thin capability wrapper that compiles prompts and emits telemetry."""

    def __init__(self, telemetry: TelemetrySubstrate | None = None) -> None:
        self._telemetry = telemetry

    def compile(
        self,
        blocks: Sequence[PromptBlock],
        *,
        tail_budget_tokens: int | None = None,
    ) -> CompiledPrompt:
        compiled = compile_prompt(blocks, tail_budget_tokens=tail_budget_tokens)
        if self._telemetry is not None:
            self._telemetry.emit(
                "prompt_compilation",
                "stable_prefix_tokens",
                float(compiled.stable_prefix_tokens),
                prefix_end_index=compiled.prefix_end_index,
            )
            self._telemetry.emit(
                "prompt_compilation",
                "dynamic_tail_tokens",
                float(compiled.dynamic_tail_tokens),
            )
        return compiled

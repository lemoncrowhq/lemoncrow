"""LemonCrow Prompt Compiler - cache-safe context assembly.

Public surface for the prompt compiler capability (P0-P8).
Start here; read docs/plans/active/prompt-compiler/index.md for the full plan.

Dependency graph:
  P0 (this module) → P1 (compiler) → P2 (linter), P3 (providers), P5 (trace)
                  → P4 (CLI), P6 (session inspector), P7 (MCP tool), P8 (SDK)
"""

from .capability import PromptCompilerCapability
from .compiler import BudgetTooSmall, CompiledPrompt, compile_prompt
from .models import (
    COUNTEREXAMPLE_METADATA_KEY,
    DEFAULT_STABILITY,
    BlockKind,
    PromptBlock,
    Stability,
)

__all__ = [
    "COUNTEREXAMPLE_METADATA_KEY",
    "DEFAULT_STABILITY",
    "BlockKind",
    "BudgetTooSmall",
    "CompiledPrompt",
    "PromptBlock",
    "PromptCompilerCapability",
    "Stability",
    "compile_prompt",
]

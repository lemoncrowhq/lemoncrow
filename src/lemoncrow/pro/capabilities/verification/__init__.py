"""Layered verification with structured counterexamples (M3).

Runs deterministic checks (lint / typecheck / tests) over the files an agent
touched and turns failures into structured :class:`Counterexample` objects the
next turn can ingest — instead of a binary pass/fail. The verifier is
host-agnostic: it produces counterexamples; the *loop* that feeds them back is
driven by the host (see M5's PostToolUse choreography), not by LemonCrow.
"""

from __future__ import annotations

from .budget import RetryBudget
from .capability import VerifierCapability
from .counterexample import Counterexample

__all__ = ["Counterexample", "RetryBudget", "VerifierCapability"]

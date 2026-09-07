from __future__ import annotations

from dataclasses import dataclass, field

from lemoncrow.core.capabilities.savings_summary import estimate_cost_usd


@dataclass
class PhaseTokens:
    phase: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def cost_usd(self, model_id: str) -> float:
        return estimate_cost_usd(
            model_id=model_id,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
        )

    def naive_cost_usd(self, model_id: str) -> float:
        """Cost as if all tokens were fresh input (no cache)."""
        return estimate_cost_usd(
            model_id=model_id,
            input_tokens=self.input_tokens + self.cache_read_tokens + self.cache_write_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )


@dataclass
class SessionReceipt:
    session_id: str
    provider: str
    model: str
    phases: list[PhaseTokens] = field(default_factory=list)
    turn_count: int = 0

    @property
    def total_input_tokens(self) -> int:
        return sum(p.input_tokens for p in self.phases)

    @property
    def total_output_tokens(self) -> int:
        return sum(p.output_tokens for p in self.phases)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(p.cache_read_tokens for p in self.phases)

    @property
    def total_cache_write_tokens(self) -> int:
        return sum(p.cache_write_tokens for p in self.phases)

    @property
    def total_fresh_tokens(self) -> int:
        return self.total_input_tokens

    @property
    def cache_efficiency_pct(self) -> float:
        total = self.total_cache_read_tokens + self.total_cache_write_tokens + self.total_fresh_tokens
        if total == 0:
            return 0.0
        return round(self.total_cache_read_tokens / total * 100, 1)

    def cost_usd(self) -> float:
        return sum(p.cost_usd(self.model) for p in self.phases)

    def naive_cost_usd(self) -> float:
        return sum(p.naive_cost_usd(self.model) for p in self.phases)

    def savings_usd(self) -> float:
        return max(0.0, self.naive_cost_usd() - self.cost_usd())

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "turn_count": self.turn_count or len(self.phases),
            "phases": [
                {
                    "phase": p.phase,
                    "input_tokens": p.input_tokens,
                    "output_tokens": p.output_tokens,
                    "cache_read_tokens": p.cache_read_tokens,
                    "cache_write_tokens": p.cache_write_tokens,
                    "cost_usd": p.cost_usd(self.model),
                }
                for p in self.phases
            ],
            "totals": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "cache_read_tokens": self.total_cache_read_tokens,
                "cache_write_tokens": self.total_cache_write_tokens,
                "cache_efficiency_pct": self.cache_efficiency_pct,
                "cost_usd": self.cost_usd(),
                "naive_cost_usd": self.naive_cost_usd(),
                "savings_usd": self.savings_usd(),
            },
        }

    def format_receipt(self) -> str:
        lines = [
            f"Session: {self.session_id}",
            f"Provider: {self.provider} / {self.model}",
            f"Turns: {self.turn_count or len(self.phases)}",
            "",
            "Tokens by phase:",
        ]
        for p in self.phases:
            lines.append(
                f"  {p.phase:12s}  input={p.input_tokens:>8,}  cache_read={p.cache_read_tokens:>8,}"
                f"  cache_write={p.cache_write_tokens:>8,}  output={p.output_tokens:>8,}"
                f"  ${p.cost_usd(self.model):.4f}"
            )
        lines += [
            "",
            f"Cache efficiency: {self.cache_efficiency_pct:.1f}%  (target: >60%)",
            f"Cost:     ${self.cost_usd():.4f}",
            f"Naive:    ${self.naive_cost_usd():.4f}  (no cache, per-phase-cold baseline)",
            f"Saved:    ${self.savings_usd():.4f}",
        ]
        return "\n".join(lines)


__all__ = ["PhaseTokens", "SessionReceipt"]

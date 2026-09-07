from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation.paths import default_store_root


def _new_session_id() -> str:
    return f"lemoncrow-run-{uuid.uuid4().hex[:8]}"


@dataclass
class OwnedAgentSession:
    session_id: str
    provider: str
    model: str
    transport: str
    cache_policy: str = "inherit"
    phase_linear: bool = True
    messages: list[dict[str, Any]] = field(default_factory=list)
    current_phase: str = "init"
    current_mode: str = "code"  # never changes the system prompt — only user turn prefix

    @classmethod
    def new(
        cls,
        *,
        provider: str,
        model: str,
        transport: str,
        cache_policy: str = "inherit",
        phase_linear: bool = True,
    ) -> OwnedAgentSession:
        return cls(
            session_id=_new_session_id(),
            provider=provider,
            model=model,
            transport=transport,
            cache_policy=cache_policy,
            phase_linear=phase_linear,
        )

    def add_user_turn(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_turn(self, content: str, *, mark_breakpoint: bool = False) -> None:
        if mark_breakpoint:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )
        else:
            self.messages.append({"role": "assistant", "content": content})

    def jsonl_path(self, root: Path | None = None) -> Path:
        store = root or default_store_root()
        runs_dir = store / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        return runs_dir / f"{self.session_id}.jsonl"

    def save(self, root: Path | None = None) -> Path:
        path = self.jsonl_path(root)
        lines: list[dict[str, Any]] = []
        lines.append(
            {
                "event": "session_meta",
                "session_id": self.session_id,
                "provider": self.provider,
                "model": self.model,
                "transport": self.transport,
                "cache_policy": self.cache_policy,
                "phase_linear": self.phase_linear,
            }
        )
        for msg in self.messages:
            lines.append({"event": "turn", **msg})
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, session_id: str, root: Path | None = None) -> OwnedAgentSession:
        store = root or default_store_root()
        path = store / "runs" / f"{session_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {path}")
        meta: dict[str, Any] = {}
        messages: list[dict[str, Any]] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            obj = json.loads(raw)
            if obj.get("event") == "session_meta":
                meta = obj
            elif obj.get("event") == "turn":
                turn = {k: v for k, v in obj.items() if k != "event"}
                messages.append(turn)
        return cls(
            session_id=meta.get("session_id", session_id),
            provider=str(meta.get("provider", "")),
            model=str(meta.get("model", "")),
            transport=str(meta.get("transport", "")),
            cache_policy=str(meta.get("cache_policy", "inherit")),
            phase_linear=bool(meta.get("phase_linear", True)),
            messages=messages,
        )


__all__ = ["OwnedAgentSession"]

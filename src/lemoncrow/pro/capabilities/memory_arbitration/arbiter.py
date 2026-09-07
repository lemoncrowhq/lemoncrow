"""Mem0-style four-op memory arbitration."""

from __future__ import annotations

import json
import logging
import re

from lemoncrow.core.capabilities.memory_arbitration_contract import (
    ArbitrationDecision as ArbitrationDecision,
)
from lemoncrow.core.capabilities.memory_arbitration_contract import (
    ArbitrationOp as ArbitrationOp,
)
from lemoncrow.core.foundation.memory_models import MemoryBlock
from lemoncrow.infra.embeddings.base import Embedder
from lemoncrow.infra.internal_llm import chat
from lemoncrow.infra.internal_llm.exceptions import InternalLLMError
from lemoncrow.infra.storage.memory_store import MemoryStore

logger = logging.getLogger(__name__)

_log = logging.getLogger(__name__)


def _emit_arbitration_metric(op: str) -> None:
    try:
        from prometheus_client import Counter

        if not hasattr(_emit_arbitration_metric, "counter"):
            _emit_arbitration_metric.counter = Counter(  # type: ignore[attr-defined]
                "lemoncrow_memory_arbitration_total",
                "Memory arbitration decisions by operation",
                ["op"],
            )
        _emit_arbitration_metric.counter.labels(op=op).inc()  # type: ignore[attr-defined]
    except Exception:
        logging.exception("Recovered from broad exception handler")
        logger.warning(
            "Suppressed exception at arbiter.py:41",
            exc_info=True,
        )


def _decision(**kwargs: object) -> ArbitrationDecision:
    decision = ArbitrationDecision.model_validate(kwargs)
    _emit_arbitration_metric(decision.op)
    return decision


def _tokens(text: str) -> set[str]:
    return {item.lower() for item in re.findall(r"[a-zA-Z0-9_]+", text)}


def _similar_blocks(new_fact: MemoryBlock, store: MemoryStore, *, k: int) -> list[MemoryBlock]:
    try:
        blocks = store.list_blocks(new_fact.agent_id, include_tombstoned=False, limit=500)
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return []
    query_tokens = _tokens(new_fact.value + " " + new_fact.label)
    scored: list[tuple[float, MemoryBlock]] = []
    for block in blocks:
        if block.id == new_fact.id:
            continue
        block_tokens = _tokens(block.value + " " + block.label)
        if not block_tokens or not query_tokens:
            score = 0.0
        else:
            score = len(query_tokens & block_tokens) / len(query_tokens | block_tokens)
        scored.append((score, block))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [block for score, block in scored[:k] if score > 0.0]


def arbitrate(
    new_fact: MemoryBlock,
    store: MemoryStore,
    embedder: Embedder,
    *,
    k: int = 5,
) -> ArbitrationDecision:
    """Return ADD, UPDATE, DELETE, or NOOP for a memory write."""
    _ = embedder
    top_k = _similar_blocks(new_fact, store, k=k)
    if not top_k:
        return _decision(op="ADD", reason="no similar memory blocks")

    prompt = {
        "new_fact": new_fact.model_dump(mode="json"),
        "existing": [block.model_dump(mode="json") for block in top_k],
        "ops": ["ADD", "UPDATE", "DELETE", "NOOP"],
    }
    schema = {
        "type": "object",
        "properties": {
            "op": {"enum": ["ADD", "UPDATE", "DELETE", "NOOP"]},
            "target_block_id": {"type": ["string", "null"]},
            "merged_value": {"type": ["string", "null"]},
            "reason": {"type": "string"},
        },
        "required": ["op", "reason"],
    }
    try:
        response = chat(
            [
                {
                    "role": "system",
                    "content": "Decide how to merge a new memory fact. Return strict JSON only.",
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, sort_keys=True, separators=(",", ":")),
                },
            ],
            json_schema=schema,
        )
    except InternalLLMError:
        return _decision(op="ADD", reason="arbitration unavailable")

    if not isinstance(response, dict):
        _log.warning("invalid arbitration response type: %r", response)
        return _decision(op="ADD", reason="invalid arbitration response")
    try:
        decision = ArbitrationDecision.model_validate(response)
        _emit_arbitration_metric(decision.op)
        return decision
    except Exception as exc:
        logging.exception("Recovered from broad exception handler")
        _log.warning("invalid arbitration response: %s", exc)
        return _decision(op="ADD", reason="invalid arbitration response")


__all__ = ["ArbitrationDecision", "arbitrate"]

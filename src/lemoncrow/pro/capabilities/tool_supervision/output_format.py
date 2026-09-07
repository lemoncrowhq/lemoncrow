"""Caller-selectable output encoding for read/search/symbols-style tools (G13).

This ties together the N6 savings gate and the N7 columnar+intern encoding into
a single selector callable from the MCP dispatcher. It is deliberately a thin,
pure function so the gateway stays a dispatcher and the logic is unit-testable.

The ``format`` selector has three values:

* ``auto`` (default) — returns the rendered text unchanged. This is the current,
  byte-compatible behavior: structural encoding is NOT applied here because it
  would alter a tool's default output shape (see the workstream hard
  constraint). Existing tests that parse default output stay green.
* ``json``  — force raw compact JSON of the structured result (no compaction).
* ``compact`` — build the N7 columnar form of the result's row list, then run it
  through the N6 savings gate. The compact form is only emitted when it beats
  the default rendering by the threshold; otherwise the default text is kept,
  so ``compact`` can never inflate a small / low-redundancy payload.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from lemoncrow.pro.capabilities.tool_supervision.compact_output import (
    DEFAULT_SAVINGS_THRESHOLD,
    columnar_encode,
    gate_compact,
)

OutputFormat = Literal["auto", "compact", "json"]

# Tools whose structured result carries a list of homogeneous rows under one of
# these keys. The first key present and holding a non-empty list of dicts is the
# columnar-encoding target.
_ROW_LIST_KEYS = ("matches", "usages", "references", "results", "rows", "hits", "callers", "callees")


def normalize_format(value: Any) -> OutputFormat:
    """Coerce an arbitrary input to a valid format selector (default ``auto``)."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("auto", "compact", "json"):
            return lowered  # type: ignore[return-value]
    return "auto"


def _row_list(result: dict[str, Any]) -> tuple[str, list[dict[str, Any]]] | None:
    """Return (key, rows) for the first row-list field, or None."""
    for key in _ROW_LIST_KEYS:
        value = result.get(key)
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            return key, [dict(item) for item in value]
    return None


def _compact_columnar_text(result: dict[str, Any]) -> str | None:
    """Self-describing compact columnar text for a row-list result, or None.

    The header carries the originating row-list key plus any scalar metadata
    from the result (everything that is not the encoded row list), so the
    encoding is lossless in principle and reversible by the consumer.
    """
    found = _row_list(result)
    if found is None:
        return None
    key, rows = found
    meta = {k: v for k, v in result.items() if k != key}
    payload = {
        "encoding": "columnar",
        "row_key": key,
        "meta": meta,
        "data": columnar_encode(rows),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def apply_output_format(
    *,
    fmt: Any,
    result: Any,
    rendered_text: str,
    threshold: float = DEFAULT_SAVINGS_THRESHOLD,
) -> tuple[str, bool]:
    """Select the output text for a tool result given the *fmt* selector.

    Returns ``(text, used_compact)``. ``used_compact`` is True only when an N7
    columnar form actually cleared the N6 gate and was emitted.

    Default (``auto``) returns *rendered_text* unchanged — byte-compatible with
    today. ``json`` forces raw compact JSON. ``compact`` tries the columnar form
    gated by N6 against *rendered_text*.
    """
    selector = normalize_format(fmt)
    if selector == "auto":
        return rendered_text, False

    if selector == "json":
        if isinstance(result, str):
            return result, False
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str), False

    # selector == "compact"
    if not isinstance(result, dict):
        return rendered_text, False
    compact_text = _compact_columnar_text(result)
    if compact_text is None:
        return rendered_text, False
    gate = gate_compact(rendered_text, compact_text, threshold=threshold)
    return gate.chosen, gate.used_compact


__all__ = [
    "OutputFormat",
    "apply_output_format",
    "normalize_format",
]

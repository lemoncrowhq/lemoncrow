"""Reversible spill store for oversized tool outputs.

The MCP dispatch path bounds a runaway tool result before it reaches the host
(head+tail compaction, then a hard byte ceiling). Historically the bytes the
ceiling drops are *lost*: a shell/sql/read/web_fetch result that overflows the
budget is truncated and the tail is gone, so the agent cannot recover it without
re-running the (often expensive, non-idempotent) tool.

This module writes the full payload to a plain-text file in the shared spill dir
and hands back a path. The agent recovers the content by calling
``read <path>`` — with ``:L1-L200`` line ranges to page through large results —
so no separate retrieval tool is needed. If ``read`` itself targets a spill file
and the result exceeds the wire budget, the dispatch layer skips re-spilling and
lets normal truncation apply instead, so there is no recursive spill chain.

The spill directory is shared with ``native_search`` via ``LEMONCROW_MCP_SPILL_DIR``
(falling back to a temp dir), so a single env var controls where everything lands.

No network, no LLM, deterministic. Best-effort: a write failure returns ``None``
rather than breaking the tool call.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# Bounded retention so the shared spill dir can't grow without limit across a long
# session or many sessions (nothing else ever deletes these files). The sweep runs
# best-effort on each write. Override via env; set either axis to 0 to disable it.
_DEFAULT_SPILL_MAX_FILES = 512
_DEFAULT_SPILL_TTL_SECONDS = 24 * 60 * 60  # 24h


def _spill_dir() -> Path:
    """Resolve the spill directory, mirroring ``native_search._spill_dir``.

    Honors ``LEMONCROW_MCP_SPILL_DIR`` so search spills and tool-output spills
    share one location; otherwise uses ``<tmp>/lemoncrow-spill``.
    """
    configured = os.environ.get("LEMONCROW_MCP_SPILL_DIR")
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        path = Path(tempfile.gettempdir()) / "lemoncrow-spill"
    path.mkdir(parents=True, exist_ok=True)
    # Spill payloads can contain command output, file contents, and SQL results,
    # so keep the directory owner-only. Best-effort: a shared/pre-existing dir we
    # don't own may reject chmod, which is fine.
    with contextlib.suppress(OSError):
        path.chmod(0o700)
    return path


def _retention_limits() -> tuple[int, int]:
    """Return ``(max_files, max_age_seconds)``; either ``<= 0`` disables that axis."""

    def _read(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            return default

    return (
        _read("LEMONCROW_MCP_SPILL_MAX_FILES", _DEFAULT_SPILL_MAX_FILES),
        _read("LEMONCROW_MCP_SPILL_TTL_SECONDS", _DEFAULT_SPILL_TTL_SECONDS),
    )


def _enforce_retention(directory: Path) -> None:
    """Evict old spill artifacts by age then count so the dir stays bounded.

    Best-effort and never raises into the caller: retention is hygiene, not
    Sweeps ``*.txt`` (tool-output spills), ``*.json`` (native-search spills), and
    the raw binary spills ``web_fetch`` produces for PDFs (``*.pdf``) and their
    embedded images (``*.png``, ``*.jpg``, ``*.jpeg``, ``*.bmp``, ``*.tiff``,
    ``*.gif``, ``*.bin``) in the shared spill dir.
    """
    max_files, max_age = _retention_limits()
    if max_files <= 0 and max_age <= 0:
        return
    try:
        entries: list[tuple[float, Path]] = []
        for pattern in ("*.txt", "*.json", "*.pdf", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.gif", "*.bin"):
            for p in directory.glob(pattern):
                try:
                    entries.append((p.stat().st_mtime, p))
                except OSError:
                    continue
    except OSError:
        return
    now = time.time()
    survivors: list[tuple[float, Path]] = []
    for mtime, p in entries:
        if max_age > 0 and (now - mtime) > max_age:
            with contextlib.suppress(OSError):
                p.unlink()
        else:
            survivors.append((mtime, p))
    if max_files > 0 and len(survivors) > max_files:
        survivors.sort(key=lambda item: item[0])  # oldest first
        for _, p in survivors[: len(survivors) - max_files]:
            with contextlib.suppress(OSError):
                p.unlink()


@dataclass(frozen=True)
class SpillRecord:
    """A persisted spill: on-disk path + original byte size."""

    path: Path
    original_bytes: int


def spill(
    content: str | bytes,
    *,
    tool_name: str,
    kind: str = "tool_output",
) -> SpillRecord | None:
    """Persist the full ``content`` and return a referenceable record.

    The payload is written as plain text so the agent can recover it via
    ``read <path>`` (with ``:L1-L200`` ranges to page through large results)
    without a separate retrieval tool. Returns ``None`` on any write failure
    (best-effort; the caller falls back to the prior truncate/compact behavior).

    Args:
        content:   The full (oversized) tool output to preserve.
        tool_name: Included in the filename for provenance.
        kind:      Logical tag encoded in the filename, e.g. ``tool_output`` or
                   ``original``.
    """
    if isinstance(content, bytes):
        # Defensive: this is a shared chokepoint fed by many callers (MCP
        # dispatch text assembled from arbitrary tool results, bash/sql/
        # web_fetch output) that occasionally hand us raw bytes despite the
        # str contract. Normalize once here so `.encode("utf-8")` below and
        # `write_text` operate on str, not bytes.
        content = content.decode("utf-8", errors="replace")
    try:
        directory = _spill_dir()
        # Short name: it is quoted verbatim in every spill footer the model
        # reads, so kind/timestamp (retention uses mtime) are pure token weight.
        del kind
        file_name = f"{tool_name}-{uuid.uuid4().hex[:8]}.txt"
        original_bytes = len(content.encode("utf-8"))
        spill_path = directory / file_name
        # Atomic publish: write to a sibling temp file then rename, so a concurrent
        # read never observes a half-written file. The '.tmp' suffix keeps
        # in-flight writes out of the '*.txt' retention sweep.
        tmp_path = directory / f".{file_name}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(tmp_path, spill_path)
        finally:
            with contextlib.suppress(OSError):
                if tmp_path.exists():
                    tmp_path.unlink()
        _enforce_retention(directory)
        return SpillRecord(path=spill_path, original_bytes=original_bytes)
    except OSError:
        return None


def spill_bytes(
    data: bytes,
    *,
    tool_name: str,
    kind: str = "original",
    suffix: str = ".bin",
) -> SpillRecord | None:
    """Persist raw binary content and return a referenceable record.

    Mirrors :func:`spill` but writes bytes verbatim instead of UTF-8 text, for
    content that can't be represented as text at all -- e.g. the original PDF
    behind a ``web_fetch`` text extraction, so the agent can point a human (or
    a vision-capable step) at the real file when extraction loses charts/tables.
    Returns ``None`` on any write failure (best-effort).
    """
    try:
        directory = _spill_dir()
        file_name = f"{kind}-{tool_name}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}{suffix}"
        spill_path = directory / file_name
        tmp_path = directory / f".{file_name}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            tmp_path.write_bytes(data)
            os.replace(tmp_path, spill_path)
        finally:
            with contextlib.suppress(OSError):
                if tmp_path.exists():
                    tmp_path.unlink()
        _enforce_retention(directory)
        return SpillRecord(path=spill_path, original_bytes=len(data))
    except OSError:
        return None


# Verbs the canonical footer accepts (documented, not enforced, so a future
# verb like "summarized" slots in without any parsing changes elsewhere):
# "shrunk" (spill + head/tail summary), "truncated" (hard byte/char cap),
# "compacted" or "compacted:{method}" (structured compaction, e.g.
# "compacted:dedup").


def spill_notice(
    *,
    verb: str,
    original_chars: int,
    kept_chars: int,
    path: Path | str | None = None,
) -> str:
    """The ONE canonical footer notice for every shrink/spill/truncate/compact event.

    Single source of truth for the notice grammar so the model learns exactly
    one pattern for reading a shrink/spill/truncate notice instead of the
    ~6 divergent ones this replaces. Counts are CHARACTERS, never bytes.

    ``path`` names the on-disk file the full content can be recovered from via
    ``read <path>``. Pass ``None`` when there is no recovery path (the spill
    itself failed, or the cap that triggered this notice has nothing spilled
    behind it) -- the notice then reports a hard truncation with no ``read``
    hint (the ``verb`` argument is unused in that shape: without a recovery
    path the event is, from the model's perspective, just a truncation
    regardless of which mechanism triggered it).
    """
    if path is None:
        return f"[lc: truncated {original_chars}→{kept_chars}; narrow the query for full]"
    return f"[lc: {verb} {original_chars}→{kept_chars}; full: {path}]"


# Clipped-summary marker: spliced in before the footer when max_chars forces
# the summary body itself to be cut further to fit (distinct from the
# original->kept shrink the footer already reports).
_CLIPPED_SUMMARY_MARKER = "\n[… summary clipped; full in spill …]\n"


def summary_with_ref(
    summary: str,
    record: SpillRecord,
    *,
    original_chars: int,
    verb: str = "shrunk",
    max_chars: int | None = None,
) -> str:
    """Compose the host-facing text: summary body + the canonical footer notice.

    The footer's ``kept_chars`` is ``len(summary)`` -- the visible body actually
    shown, not the wire size including the footer's own text (the convention
    every other producer of this footer uses). When ``max_chars`` forces the
    summary to be cut further to fit, the clipped-summary marker is spliced in
    before the footer.
    """

    def _footer(kept_chars: int) -> str:
        return f"\n\n{spill_notice(verb=verb, original_chars=original_chars, kept_chars=kept_chars, path=record.path)}"

    if max_chars is None:
        return f"{summary}{_footer(len(summary))}"
    if max_chars <= 0:
        return ""

    footer = _footer(len(summary))
    if len(footer) >= max_chars:
        # The cap is smaller than the full footer. Prefer the bare path so the
        # spill stays recoverable, falling back to a prefix only when even
        # that can't fit.
        ref = str(record.path)
        return ref if len(ref) <= max_chars else ref[:max_chars]

    available = max_chars - len(footer)
    if len(summary) > available:
        if available <= len(_CLIPPED_SUMMARY_MARKER):
            summary = summary[:available]
        else:
            content_budget = available - len(_CLIPPED_SUMMARY_MARKER)
            head_chars = int(content_budget * 0.7)
            tail_chars = content_budget - head_chars
            summary = f"{summary[:head_chars]}{_CLIPPED_SUMMARY_MARKER}{summary[-tail_chars:]}"
        # kept_chars (len(summary)) only shrank, which can only shorten the
        # footer (fewer digits) -- recomputing keeps ORIG->KEPT accurate.
        footer = _footer(len(summary))
    return f"{summary}{footer}"

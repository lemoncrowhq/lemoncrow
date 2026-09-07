"""SemanticFileMemoryCapability — thin orchestrator over all sub-modules."""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import stat
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from lemoncrow.core.capabilities._optional_runtime import Repo
from lemoncrow.core.capabilities.native_read_baseline import claude_read_baseline_text
from lemoncrow.infra.code_intel.languages import language_for_path
from lemoncrow.pro.capabilities.prompt_compilation.tokens import count_tokens

from .graph_analytics import GraphAnalytics
from .indexer import FileIndex
from .models import FileOutline, SemanticSummary
from .python_ast import analyze_python, stub_function_bodies
from .python_ast import outline as python_outline
from .search import SymbolIndex
from .typescript_ast import analyze_typescript
from .typescript_ast import outline as typescript_outline

_logger = logging.getLogger(__name__)
_DEFAULT_OUTLINE_THRESHOLD = 500

# Hard cap on how many bytes ``smart_read`` will pull into memory for a single
# file. Without it an unconditional ``read_text`` of a multi-GB log, or a read
# of a special file (``/dev/zero``, a FIFO — ``st_size == 0`` so no size signal),
# would OOM or block forever. Regular files under the cap are read whole, so
# normal source files behave exactly as before.
_MAX_READ_BYTES = int(os.environ.get("LEMONCROW_READ_MAX_BYTES", str(8 * 1024 * 1024)))


def _read_source_bounded(file_path: Path) -> tuple[str, bool]:
    """Read *file_path* as text, never materializing more than ``_MAX_READ_BYTES``.

    Returns ``(source, truncated)``. ``truncated`` is True when only a byte
    prefix was read — either the regular file exceeds the cap, or the path is a
    special file (char-special, FIFO, ...) whose size is unknown and must not be
    read unbounded. Decode always uses ``errors="replace"`` so binary/non-UTF-8
    bytes survive. A regular file at or under the cap is read whole, preserving
    identical behavior for normal-sized files.
    """
    try:
        st = file_path.stat()
        is_regular = stat.S_ISREG(st.st_mode)
        size = st.st_size
    except OSError:
        # Stat failed — treat size as unknown and apply the prefix cap.
        is_regular = False
        size = 0

    # A non-regular file reports st_size == 0 even though it may stream forever;
    # treat "unknown size" as oversized and read only a bounded prefix. A regular
    # file at or below the cap is safe to read whole.
    if is_regular and size <= _MAX_READ_BYTES:
        return file_path.read_text(encoding="utf-8", errors="replace"), False

    # Oversized regular file or special file: read at most _MAX_READ_BYTES from
    # the raw fd so gigabytes / endless streams are never pulled into memory.
    chunks: list[bytes] = []
    remaining = _MAX_READ_BYTES
    fd = os.open(file_path, os.O_RDONLY)
    try:
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    # A regular file whose stat said it was small but that we read fully (e.g.
    # special file that happened to end) is only "truncated" if we actually hit
    # the cap with bytes still pending.
    truncated = len(raw) >= _MAX_READ_BYTES and (not is_regular or size > _MAX_READ_BYTES)
    return raw.decode("utf-8", errors="replace"), truncated


def _read_source_guarded(file_path: Path) -> str:
    """Regular-file-guarded bounded read for the cache-free preview helpers.

    Mirrors ``smart_read``'s guard: reject FIFO/socket/special files *before*
    ``_read_source_bounded`` (``os.open`` on a writerless FIFO blocks the syscall
    forever, before the byte cap can apply), then read at most ``_MAX_READ_BYTES``
    so a multi-GB file can't OOM the process.
    """
    try:
        st = file_path.stat()
    except OSError as exc:
        raise FileNotFoundError(f"file not found: {file_path}") from exc
    if stat.S_ISDIR(st.st_mode):
        raise FileNotFoundError(f"path is a directory, not a file: {file_path}")
    if not stat.S_ISREG(st.st_mode):
        raise FileNotFoundError(f"not a regular file (FIFO/socket/special files are unreadable): {file_path}")
    source, _ = _read_source_bounded(file_path)
    return source


def default_outline_threshold() -> int:
    """Outline LOC threshold: ``LEMONCROW_OUTLINE_THRESHOLD`` env override, else 500.

    Files with effective LOC above the threshold are outline-eligible; files at
    or below are always read in full. The 25% savings guard still decides
    whether an eligible file's outline actually ships.
    """
    raw = os.environ.get("LEMONCROW_OUTLINE_THRESHOLD", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            _logger.warning("invalid LEMONCROW_OUTLINE_THRESHOLD=%r, using %d", raw, _DEFAULT_OUTLINE_THRESHOLD)
    return _DEFAULT_OUTLINE_THRESHOLD


@lru_cache(maxsize=1)
def _outline_notice_chars() -> int:
    """Chars the read layer prepends to outline bodies (notice + blank line)."""
    from lemoncrow.pro.capabilities.source_projection import SourceProjection

    return len(SourceProjection.outline().notice or "") + 2


def _outline_saves_enough(outline_text: str, source: str) -> bool:
    """25% savings guard, counting the projection notice the agent actually receives.

    Without the notice overhead a tiny file passes on the bare outline but
    ships *larger* than its own body (and forces a full=true round-trip).
    """
    return len(outline_text) + _outline_notice_chars() <= int(len(source) * 0.75)


class SemanticFileMemoryCapability:
    """
    Semantic file analysis with content-addressed caching.

    Capabilities:
    - Full Python AST extraction (functions, classes, methods, variables,
      decorators, docstrings, complexity, return types)
    - Full TypeScript/JS export/interface/type/enum detection
    - SHA-256 content-addressed cache (reliable across git, Docker, rsync)
    - Cross-file symbol resolution
    - BM25-ranked full-text search over cached summaries
    - Reverse dependency graph for change impact analysis
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._index = FileIndex(self._root)
        self._symbol_index = SymbolIndex(self._index)
        try:
            self._symbol_index._ensure_idf()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    @staticmethod
    def _language_for(path: Path) -> str:
        # Delegate to the canonical registry (DLS-LANG-03/04). Unknown
        # extensions resolve to None at the registry boundary; callers here
        # map that to "text". Shell extensions (.sh/.bash/.zsh) now resolve to
        # "bash", reaching the live tree-sitter grammar.
        lang = language_for_path(path)
        return lang.name if lang is not None else "text"

    # ------------------------------------------------------------------
    # Core summarisation
    # ------------------------------------------------------------------

    @staticmethod
    def _effective_loc(source: str, language: str) -> int:
        """Count effective LOC (exclude blank + comment-only lines)."""
        if language == "python":
            return sum(1 for line in source.splitlines() if line.strip() and not line.lstrip().startswith("#"))

        if language in {"typescript", "javascript"}:
            count = 0
            in_block_comment = False
            for raw in source.splitlines():
                line = raw.strip()
                if not line:
                    continue

                if in_block_comment:
                    end_idx = line.find("*/")
                    if end_idx < 0:
                        continue
                    line = line[end_idx + 2 :].strip()
                    in_block_comment = False
                    if not line:
                        continue

                if line.startswith("//"):
                    continue

                if line.startswith("/*"):
                    end_idx = line.find("*/", 2)
                    if end_idx < 0:
                        in_block_comment = True
                        continue
                    trailing = line[end_idx + 2 :].strip()
                    if not trailing:
                        continue
                    line = trailing

                if line:
                    count += 1
            return count

        return sum(1 for line in source.splitlines() if line.strip())

    @staticmethod
    def _range_starts_past(range_spec: str, total_lines: int) -> bool:
        """True if *range_spec*'s start line is beyond *total_lines*.

        Used on truncated reads to detect requests that fall entirely past the
        byte-capped prefix without letting ``_parse_range_spec`` raise a
        misleading "exceeds file length" error against the prefix length.
        """
        start_match = re.match(r"L?(\d+)", range_spec.strip(), flags=re.IGNORECASE)
        if start_match is None:
            return False
        return int(start_match.group(1)) > total_lines

    @staticmethod
    def _parse_range_bounds(range_spec: str) -> tuple[int, int | None]:
        """Parse a range spec into (start, end) WITHOUT needing the file length.

        ``end`` is None for an open-ended range (``"L42-"``); a bare line
        (``"42"``) yields ``(42, 42)``. Falls back to ``(1, None)`` on a bad spec.
        """
        m = re.match(r"\s*[Ll]?(\d+)\s*([-,:])?\s*[Ll]?(\d*)\s*$", range_spec)
        if m is None:
            return 1, None
        start = max(1, int(m.group(1)))
        if not m.group(2):
            return start, start
        if not m.group(3):
            return start, None
        return start, max(start, int(m.group(3)))

    def _range_read(self, file_path: Path, range_spec: str) -> dict[str, Any]:
        """Read only the requested line range by streaming the file, instead of
        loading + splitting the whole file. tokens_saved is 0: a deliberate slice
        does not 'avoid' a full-file read, so crediting it inflates savings."""
        start, end = self._parse_range_bounds(range_spec)
        kept: list[str] = []
        total_read = 0
        used = 0
        capped = False
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for lineno, line in enumerate(handle, start=1):
                total_read = lineno
                used += len(line)
                if used > _MAX_READ_BYTES:
                    capped = True
                    total_read = lineno - 1
                    break
                if lineno < start:
                    continue
                if end is not None and lineno > end:
                    total_read = lineno - 1
                    break
                kept.append(line.rstrip("\n"))
        result: dict[str, Any] = {
            "path": str(file_path),
            "language": self._language_for(file_path),
            "mode": "range",
            "tokens_saved": 0,
        }
        if not kept:
            result["range"] = range_spec
            result["content"] = ""
            result["truncation_notice"] = (
                f"[scanned first {_MAX_READ_BYTES}B; range past it -- request earlier slice]"
                if capped
                else f"[range past {total_read} lines -- request earlier slice]"
            )
            return result
        actual_range = f"{start}-{start + len(kept) - 1}"
        result["range"] = actual_range
        result["content"] = "\n".join(kept)
        if capped:
            result["truncation_notice"] = f"[stopped at {_MAX_READ_BYTES}B cap -- may be incomplete]"
        return result

    @staticmethod
    def _parse_range_spec(range_spec: str, total_lines: int) -> tuple[int, int]:
        """Parse ranges like 42-118/L42-L118/42,118, plus tolerant open-ended forms."""
        s = range_spec.strip()
        start_match = re.match(r"L?(\d+)", s, flags=re.IGNORECASE)
        if start_match is None:
            raise ValueError("range must start with a line number")

        start = int(start_match.group(1))
        if start < 1:
            raise ValueError("range lines must be >= 1")

        end_match = re.search(r"[-,:]\s*L?(\d+)", s, flags=re.IGNORECASE)
        if end_match is not None:
            end = int(end_match.group(1))
            if end < 1:
                raise ValueError("range lines must be >= 1")
        elif "-" in s:
            # Open-ended or malformed end (e.g., "L259-" / "259-foo") -> EOF.
            end = total_lines
        else:
            # Single-line form (e.g., "L42") -> read one line.
            end = start

        if start > total_lines:
            raise ValueError(f"range start {start} exceeds file length {total_lines}")

        if end < start:
            end = start
        return start, min(end, total_lines)

    @staticmethod
    def _token_savings(baseline_text: str, returned_text: str) -> int:
        full_tokens = max(1, count_tokens(baseline_text))
        returned_tokens = max(1, count_tokens(returned_text))
        return max(0, full_tokens - returned_tokens)

    @staticmethod
    def _outline_for(path: Path, source: str, language: str, *, effective_loc: int) -> FileOutline:
        if language == "python":
            base = python_outline(str(path), source)
        else:
            lang = "javascript" if language == "javascript" else "typescript"
            base = typescript_outline(str(path), source, lang=lang)
        return base.model_copy(update={"loc": effective_loc})

    # Generic outline fallback for languages without a dedicated AST builder.
    # Strategy: keep lines that look structural — column-0 declarations,
    # signature-shaped keywords, Markdown headings, import/use/package lines.
    # Imperfect, but vastly smaller than the full source for typical files.
    _GENERIC_LEADING_KEYWORDS: ClassVar[frozenset[str]] = frozenset(
        {
            # declarations
            "def",
            "class",
            "func",
            "function",
            "fn",
            "struct",
            "enum",
            "trait",
            "impl",
            "type",
            "interface",
            "record",
            "object",
            "protocol",
            "extension",
            "mod",
            "module",
            "package",
            "namespace",
            "actor",
            # imports / linkage
            "import",
            "from",
            "use",
            "using",
            "require",
            "include",
            "export",
            "extern",
            # visibility-leading (the body keyword follows)
            "pub",
            "public",
            "private",
            "protected",
            "internal",
            "open",
            "sealed",
            "abstract",
            "final",
            "static",
            "async",
            # value-decls (Rust/Go/TS const)
            "const",
            "let",
            "var",
            # SQL DDL
            "create",
            "alter",
            "drop",
        }
    )

    @classmethod
    def _generic_outline_text(cls, source: str, language: str) -> str:
        """Extract a structural skeleton from any language. Returns plain text."""
        if language == "markdown":
            # For Markdown: keep heading lines + the first non-blank line under each.
            kept: list[str] = []
            prev_was_heading = False
            for raw in source.splitlines():
                stripped = raw.strip()
                if stripped.startswith("#"):
                    kept.append(raw.rstrip())
                    prev_was_heading = True
                elif prev_was_heading and stripped:
                    kept.append(raw.rstrip())
                    prev_was_heading = False
            return "\n".join(kept)

        kept_lines: list[str] = []
        for raw in source.splitlines():
            stripped = raw.lstrip()
            if not stripped:
                continue
            # Column-0 declarations (most languages)
            if raw == stripped:
                kept_lines.append(raw.rstrip())
                continue
            # Indented but signature-looking
            first_token = stripped.split(None, 1)[0].rstrip(":(<{").lower()
            if first_token in cls._GENERIC_LEADING_KEYWORDS:
                kept_lines.append(raw.rstrip())
        return "\n".join(kept_lines)

    def smart_read(
        self,
        path: str | Path,
        *,
        range_spec: str | None = None,
        expand: bool = False,
        outline_threshold: int | None = None,
    ) -> dict[str, Any]:
        """Read a file and project it down to the cheapest representation that
        preserves meaning.  The full cascade runs on every call:

        1. **Range read** (if ``range_spec`` is set): stream only the requested
           lines — no outline, no projection, just exact bytes.  Returns early.

        2. **Outline** (large files only, ``expand=False``):
           If ``effective_loc > outline_threshold`` (default 500), try, in order:
           - Python AST outline (Python only) — class/function signatures.
           - Tree-sitter outline (Go, Rust, Java, TS, …) — grammar-level
             structural extraction.
           - Generic regex outline — function/class heads for any language.
           Each candidate must save ≥ 25 % over the raw source or it is
           skipped.  Returns mode="outline".

        3. **Minified** (all files, fires when outline did not):
           ``build_minified_projection`` uses tree-sitter to strip docstrings,
           inline comments, and runs of blank lines while keeping every
           executable line intact.  Only applied when a supported language
           grammar is available.  Returns mode="minified".

        4. **Compact** (conservative fallback):
           ``build_compact_projection`` collapses multiple consecutive blank
           lines into one and trims trailing whitespace — language-agnostic.
           Returns mode="compact".

        5. **Full** (last resort):
           No projection saved anything meaningful; raw source is returned.
           Returns mode="full".

        ``expand=True`` skips steps 2-4 and always returns the raw source.
        Steps 3-4 are applied in the MCP layer (``mcp_server._smart_read_single``)
        after this method returns, so the ``mode`` field here reflects only the
        outline/full decision; the final delivery to the model may still be
        minified or compact even when this returns mode="full".
        """
        if outline_threshold is None:
            outline_threshold = default_outline_threshold()
        file_path = Path(path)
        # Only regular files are valid read targets. Rejecting non-regular paths
        # up front keeps FIFOs/named pipes, sockets, and char/block-special files
        # away from _read_source_bounded — os.open() on a FIFO with no writer
        # blocks the syscall itself forever, before the byte cap can apply.
        try:
            st = file_path.stat()
        except OSError as exc:
            raise FileNotFoundError(f"file not found: {file_path}") from exc
        if stat.S_ISDIR(st.st_mode):
            raise FileNotFoundError(f"path is a directory, not a file: {file_path}")
        if not stat.S_ISREG(st.st_mode):
            raise FileNotFoundError(f"not a regular file (FIFO/socket/special files are unreadable): {file_path}")

        if range_spec:
            # A range read is a deliberate slice: stream only the requested lines
            # instead of reading + splitting the whole file, and don't credit it
            # against the full-file baseline (that inflates tokens_saved).
            return self._range_read(file_path, range_spec)

        source, truncated = _read_source_bounded(file_path)
        language = self._language_for(file_path)
        effective_loc = self._effective_loc(source, language)

        cache_hit = self._index.get(file_path) is not None
        # summarize_file does its own unbounded read_text; skip it when the
        # source was capped so an oversized/special file never gets read whole
        # through the cache-miss path.
        if not cache_hit and not truncated:
            self.summarize_file(file_path, cache_enabled=True)

        result: dict[str, Any] = {
            "path": str(file_path),
            "language": language,
            "loc": effective_loc,
            "cache_hit": cache_hit,
        }
        if truncated:
            result["truncated"] = True
            # loc was computed over the byte-capped prefix only, so it is a lower
            # bound on the file's true line count, not an exact value.
            result["loc"] = f">={effective_loc}"
            result["loc_is_lower_bound"] = True
            result["truncation_notice"] = (
                f'[read first {_MAX_READ_BYTES}B -- oversized; narrow range, e.g. range="L1-L400"]'
            )

        mode, payload, outline_payload = self._projection_payload(
            file_path,
            source,
            language,
            effective_loc,
            expand=expand,
            outline_threshold=outline_threshold,
        )
        baseline = claude_read_baseline_text(source)
        if mode == "outline":
            result.update(
                {
                    "mode": "outline",
                    "outline": outline_payload,
                    "tokens_saved": self._token_savings(baseline, payload),
                }
            )
        else:
            result.update(
                {
                    "mode": "full",
                    "content": source,
                    "tokens_saved": self._token_savings(baseline, source),
                }
            )
        return result

    @classmethod
    def _projection_payload(
        cls,
        file_path: Path,
        source: str,
        language: str,
        effective_loc: int,
        *,
        expand: bool,
        outline_threshold: int,
    ) -> tuple[str, str, dict[str, Any] | None]:
        """Mode-selection cascade shared by ``smart_read`` and ``project_preview``.

        Returns ``(mode, payload_text, outline_payload)`` where *payload_text*
        is the exact text shipped to the agent and *outline_payload* is the
        value for the result's ``outline`` key (``None`` in full mode).
        """
        if not expand and effective_loc > outline_threshold:
            # Per-language AST outline (python only — TS/JS go through tree-sitter
            # below). 25% guard: don't ship a fake savings event if the outline is
            # larger than the source (e.g. parse failed and returned an empty
            # FileOutline, or source has invalid syntax).
            if language == "python":
                outline = cls._outline_for(file_path, source, language, effective_loc=effective_loc)
                outline_json = json.dumps(outline.model_dump(mode="json"), ensure_ascii=False)
                if _outline_saves_enough(outline_json, source):
                    return "outline", outline_json, outline.model_dump(mode="json")
                # Guard failed — fall through to tree-sitter / generic / full.

            # Tree-sitter outline for languages with a per-grammar config.
            # Same 25% guard: if the structural extraction doesn't save at
            # least a quarter, fall through rather than ship fake savings.
            from .treesitter_ast import SUPPORTED_LANGUAGES
            from .treesitter_ast import outline_text as ts_outline_text

            if language in SUPPORTED_LANGUAGES:
                ts_text = ts_outline_text(language, source)
                if ts_text and _outline_saves_enough(ts_text, source):
                    return "outline", ts_text, {"kind": "treesitter", "language": language, "text": ts_text}

            # Generic regex-based outline fallback for languages without a
            # tree-sitter config or where the tree-sitter outline didn't earn
            # the 25% bar. Same 25% safety guard.
            if language != "text":
                outline_text = cls._generic_outline_text(source, language)
                if outline_text and _outline_saves_enough(outline_text, source):
                    return "outline", outline_text, {"kind": "generic", "language": language, "text": outline_text}

        return "full", source, None

    @classmethod
    def project_preview(
        cls,
        path: str | Path,
        source: str | None = None,
        *,
        outline_threshold: int | None = None,
    ) -> dict[str, Any]:
        """Cache-free preview of what the ``read`` tool would ship for *path*.

        Runs the real projection cascade (outline → compact whitespace → full)
        without touching the summary cache, mirroring ``smart_read`` plus the
        MCP layer's ``build_compact_projection`` post-pass. Used by
        ``lc project`` so CLI savings numbers match the live read
        pipeline. Token counts use the same tiktoken accounting (baseline =
        Claude's built-in Read approximation).
        """
        if outline_threshold is None:
            outline_threshold = default_outline_threshold()
        file_path = Path(path)
        if source is None:
            source = _read_source_guarded(file_path)
        language = cls._language_for(file_path)
        effective_loc = cls._effective_loc(source, language)
        mode, payload, _ = cls._projection_payload(
            file_path,
            source,
            language,
            effective_loc,
            expand=False,
            outline_threshold=outline_threshold,
        )
        if mode == "full" and source:
            # Mirror the MCP read layer: full-mode bodies enter the agent's
            # context projected — prefer the tree-sitter minified view, falling
            # back to the conservative compact transform (mcp_server._smart_read_single).
            from lemoncrow.pro.capabilities.source_projection import (
                build_compact_projection,
                build_minified_projection,
                language_for_minify,
            )

            minify_lang = language_for_minify(str(file_path))
            minified = build_minified_projection(source, minify_lang) if minify_lang is not None else None
            if minified is not None and minified.applied:
                mode, payload = "minified", minified.content
            else:
                compact = build_compact_projection(source, language)
                if compact.applied:
                    mode, payload = "compact", compact.content
        baseline = claude_read_baseline_text(source)
        raw_tokens = count_tokens(baseline)
        tokens = raw_tokens if payload == baseline else count_tokens(payload)
        return {
            "mode": mode,
            "language": language,
            "loc": effective_loc,
            "text": payload,
            "raw_tokens": raw_tokens,
            "tokens": tokens,
        }

    @classmethod
    def project_modes(
        cls,
        path: str | Path,
        source: str | None = None,
        *,
        outline_threshold: int | None = None,
    ) -> dict[str, Any]:
        """Token cost of every projection mode for *path* plus the cascade winner.

        Unlike ``project_preview`` (winner only), this reports outline / minified
        / compact / full side by side so nothing is hidden. Outline is
        force-evaluated (threshold ignored) so its number is visible even when
        the real cascade skips it; ``available`` is False when a mode can't earn
        its keep.
        """
        from lemoncrow.pro.capabilities.source_projection import (
            build_compact_projection,
            build_minified_projection,
            language_for_minify,
        )

        if outline_threshold is None:
            outline_threshold = default_outline_threshold()
        file_path = Path(path)
        if source is None:
            source = _read_source_guarded(file_path)
        language = cls._language_for(file_path)
        effective_loc = cls._effective_loc(source, language)
        raw_tokens = count_tokens(claude_read_baseline_text(source))

        # Outline forced eligible (threshold=-1) so its savings show even when
        # the real threshold skips it; o_mode != "outline" means it can't earn 25%.
        o_mode, o_text, _ = cls._projection_payload(
            file_path, source, language, effective_loc, expand=False, outline_threshold=-1
        )
        outline_ok = o_mode == "outline"

        minify_lang = language_for_minify(str(file_path))
        minified = build_minified_projection(source, minify_lang) if minify_lang is not None else None
        if minified is not None and minified.applied:
            minified_ok, minified_tokens, minified_text = True, minified.projected_tokens, minified.content
        else:
            minified_ok, minified_tokens, minified_text = False, raw_tokens, None
        compact = build_compact_projection(source, language)

        if effective_loc > outline_threshold and outline_ok:
            winner = "outline"
        elif minified_ok:
            winner = "minified"
        elif compact.applied:
            winner = "compact"
        else:
            winner = "full"

        return {
            "language": language,
            "loc": effective_loc,
            "raw_tokens": raw_tokens,
            "winner": winner,
            "modes": {
                "outline": {
                    "tokens": count_tokens(o_text) if outline_ok else raw_tokens,
                    "text": o_text if outline_ok else None,
                    "available": outline_ok,
                },
                "minified": {"tokens": minified_tokens, "text": minified_text, "available": minified_ok},
                "compact": {
                    "tokens": compact.projected_tokens if compact.applied else raw_tokens,
                    "text": compact.content if compact.applied else None,
                    "available": compact.applied,
                },
                "full": {"tokens": raw_tokens, "text": source, "available": True},
            },
        }

    def summarize_file(
        self,
        path: str | Path,
        *,
        max_lines: int = 120,
        cache_enabled: bool = True,
    ) -> SemanticSummary:
        """Analyse a file and cache the result (by SHA-256 content hash)."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"file not found: {file_path}")

        # Serve from cache if unchanged
        if cache_enabled:
            cached = self._index.get(file_path)
            if cached:
                return self._entry_to_summary(cached)

        source = file_path.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
        language = self._language_for(file_path)

        symbol_details: list[dict[str, Any]] = []
        symbols: list[str] = []
        exports: list[str] = []
        imports_modules: list[str] = []
        dependency_map: list[str] = []
        ast_summary = "ast:unsupported"
        module_docstring = ""
        complexity_score = 0
        git_last_commit = ""
        git_last_author_date = ""

        if language == "python":
            sym_infos, imp_infos, ast_summary, module_docstring, complexity_score = analyze_python(source)
            symbols = [s.name for s in sym_infos]
            exports = [s.name for s in sym_infos if s.is_export and not s.is_private]
            symbol_details = [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "lineno": s.lineno,
                    "signature": s.signature,
                    "is_private": s.is_private,
                    "docstring": s.docstring,
                    "decorators": s.decorators,
                    "type_hint": s.type_hint,
                    "complexity": s.complexity,
                }
                for s in sym_infos
            ]
            imports_modules = sorted({i.module for i in imp_infos})
            # Resolve local imports to file paths. Canonicalise to absolute
            # resolved form so reverse-dep lookups in change_impact match
            # regardless of the path form the caller indexed/queried with.
            base = file_path.parent
            for imp in imp_infos:
                parts = imp.module.split(".")
                for search_base in [base, base.parent]:
                    candidate = search_base / Path(*parts).with_suffix(".py")
                    if candidate.is_file():
                        dependency_map.append(str(candidate.resolve()))
                        break
            dependency_map = list(dict.fromkeys(dependency_map))[:15]
            summary_str = stub_function_bodies(source, max_body_lines=2)
            if len(summary_str.splitlines()) > max_lines:
                summary_str = "\n".join(summary_str.splitlines()[:max_lines]) + "\n... [truncated]"

        elif language in ("typescript", "javascript"):
            sym_infos_ts, imp_infos_ts, ast_summary = analyze_typescript(source)
            symbols = [s.name for s in sym_infos_ts]
            exports = [s.name for s in sym_infos_ts if s.is_export]
            symbol_details = [
                {"name": s.name, "kind": s.kind, "lineno": s.lineno, "signature": s.signature} for s in sym_infos_ts
            ]
            imports_modules = sorted({i.module for i in imp_infos_ts})
            summary_str = "\n".join(lines[:max_lines])
            if len(lines) > max_lines:
                summary_str += "\n... [truncated]"

        else:
            summary_str = "\n".join(lines[:max_lines])
            if len(lines) > max_lines:
                summary_str += "\n... [truncated]"

        # Find linked test files
        test_files = self._find_test_files(file_path)
        git_last_commit, git_last_author_date = "", ""  # skip live git calls; populate via pre-warm only

        payload: dict[str, Any] = {
            "path": str(file_path),
            "language": language,
            "summary": summary_str,
            "symbols": symbols,
            "exports": exports,
            "lines_total": len(lines),
            "ast_summary": ast_summary,
            "symbol_details": symbol_details,
            "imports": imports_modules,
            "dependency_map": dependency_map,
            "test_files": test_files,
            "module_docstring": module_docstring,
            "complexity_score": complexity_score,
            "git_last_commit": git_last_commit,
            "git_last_author_date": git_last_author_date,
        }
        if cache_enabled:
            self._index.put(file_path, payload)
            # put() already stored the content hash; reuse it instead of a
            # second get() (which would re-load the JSON cache and re-hash the
            # file just to read back what we already have).
            payload["content_hash"] = self._index.content_hash(file_path)
        return self._entry_to_summary(payload)

    @staticmethod
    def _entry_to_summary(entry: dict[str, Any]) -> SemanticSummary:
        return SemanticSummary(
            path=str(entry.get("path", "")),
            language=str(entry.get("language", "text")),
            summary=str(entry.get("summary", "")),
            symbols=list(entry.get("symbols", [])),
            exports=list(entry.get("exports", [])),
            lines_total=int(entry.get("lines_total", 0)),
            ast_summary=str(entry.get("ast_summary", "")),
            content_hash=str(entry.get("content_hash", "")),
            symbol_details=list(entry.get("symbol_details", [])),
            imports=list(entry.get("imports", [])),
            dependency_map=list(entry.get("dependency_map", [])),
            test_files=list(entry.get("test_files", [])),
            module_docstring=str(entry.get("module_docstring", "")),
            complexity_score=int(entry.get("complexity_score", 0)),
            git_last_commit=str(entry.get("git_last_commit", "")),
            git_last_author_date=str(entry.get("git_last_author_date", "")),
        )

    def get_cached(self, path: str | Path) -> SemanticSummary | None:
        """Return cached summary if still valid (hash-match), else None."""
        file_path = Path(path)
        if not file_path.is_file():
            return None
        entry = self._index.get(file_path)
        if entry is None:
            return None
        return self._entry_to_summary(entry)

    def module_summary(self, path: str | Path) -> dict[str, Any]:
        """Return a concise dict suitable for CLI display or LLM injection."""
        s = self.get_cached(path) or self.summarize_file(path)
        return {
            "path": s.path,
            "language": s.language,
            "exports": s.exports,
            "symbols": s.symbols[:50],
            "imports": s.imports,
            "dependency_map": s.dependency_map,
            "test_files": s.test_files,
            "lines_total": s.lines_total,
            "ast_summary": s.ast_summary,
            "module_docstring": s.module_docstring,
            "complexity_score": s.complexity_score,
            "content_hash": s.content_hash,
            "git_last_commit": s.git_last_commit,
            "git_last_author_date": s.git_last_author_date,
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def symbol_search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Find symbols matching query across all cached files."""
        return self._symbol_index.resolve_symbol(query)[:limit]

    def semantic_search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """BM25-ranked full-text search over cached file summaries."""
        return self._symbol_index.bm25_search(query, limit=limit)

    def change_impact(self, path: str | Path) -> dict[str, Any]:
        """Estimate blast radius of modifying a file (uses reverse dep graph)."""
        # Ensure file is in cache
        fp = Path(path)
        if fp.is_file() and not self._index.get(fp):
            self.summarize_file(fp)
        # Query the reverse-dep graph by the same canonical (absolute resolved)
        # form dependency_map entries are stored under, so a relative-vs-absolute
        # path form does not silently collapse the importer set to [].
        lookup = str(fp.resolve()) if fp.exists() else str(path)
        return self._symbol_index.change_impact(lookup)

    def graph_analytics(self) -> GraphAnalytics:
        """Return file-graph analytics (blast_radius/dead_code/cycles/coupling)."""
        return GraphAnalytics(self._symbol_index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_test_files(self, file_path: Path) -> list[str]:
        stem = file_path.stem
        if stem.startswith("test_") or stem.endswith("_test"):
            return []
        root = file_path.parent
        for _ in range(4):
            tests_dir = root / "tests"
            if tests_dir.is_dir():
                # Single tree walk instead of two rglob passes. Exact
                # ``test_{stem}.py`` matches rank first, then looser
                # ``*{stem}*test*.py`` matches; short-circuit once the exact
                # bucket alone fills the cap.
                exact_pat = f"test_{stem}.py"
                loose_pat = f"*{stem}*test*.py"
                exact: list[Path] = []
                loose: list[Path] = []
                for p in tests_dir.rglob("*.py"):
                    name = p.name
                    if fnmatch.fnmatchcase(name, exact_pat):
                        exact.append(p)
                        if len(exact) >= 5:
                            break
                    elif fnmatch.fnmatchcase(name, loose_pat):
                        loose.append(p)
                return [str(p) for p in (exact + loose)[:5]]
            root = root.parent
        return []

    def _load(self) -> dict[str, Any]:
        """Return raw index state dict (backward-compat with engine.py)."""
        return self._index._load()

    def _git_metadata(self, file_path: Path) -> tuple[str, str]:
        """Return (last_commit_sha, authored_datetime_iso) for a file."""
        if Repo is None:
            return "", ""
        try:
            repo = Repo(file_path, search_parent_directories=True)
            wtd = repo.working_tree_dir
            assert wtd is not None
            rel_path = str(file_path.resolve().relative_to(wtd))
            commits = list(repo.iter_commits(paths=rel_path, max_count=1))
            if not commits:
                return "", ""
            commit = commits[0]
            return str(commit.hexsha), commit.authored_datetime.isoformat()
        except Exception as exc:
            # InvalidGitRepositoryError is expected when the file is not inside
            # a git repo (e.g. a benchmark snapshot copy). Return empty silently.
            if type(exc).__name__ == "InvalidGitRepositoryError":
                return "", ""
            logging.exception("Recovered from broad exception handler")
            return "", ""

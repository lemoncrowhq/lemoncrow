"""LemonCrow-native combined file search/read capability."""

from __future__ import annotations

import ast
import base64
import bisect
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from lemoncrow.core.capabilities.native_read_baseline import claude_read_baseline_text
from lemoncrow.core.foundation.redaction import redact_tool_output

try:  # Third-party engine supporting a per-call wall-clock `timeout=` on search.
    import regex as _regex_module
except ImportError:  # pragma: no cover - fallback path when `regex` is absent.
    _regex_module = None

SearchOutputMode = Literal[
    "ranked_file_map",
    "file_paths_with_content",
    "file_paths_only",
    "file_paths_with_match_count",
]

MAX_STRUCTURED_OUTPUT_CHARS = 80_000
DEFAULT_CONTEXT_BUDGET_TOKENS = 2_000
INLINE_CHARS_PER_TOKEN = 2
# Max paths emitted by output_mode=file_paths_only before truncation.
_FILE_PATHS_ONLY_CAP = 200
# Wall-clock budget for running a user-supplied content_regex across all
# candidate files. Caps catastrophic-backtracking ReDoS to a bounded hang.
_REGEX_DEADLINE_SECONDS = 5.0
# Per-line input cap fed to the user regex — a single pathological line cannot
# drive backtracking time superlinearly past this bound.
_REGEX_MAX_LINE_CHARS = 20_000
# Tighter per-line cap used only on the stdlib `re` fallback path. `re` has no
# per-call wall-clock hook, so a single catastrophic-backtracking match (e.g.
# `(a+)+$`) runs to completion in C with only a between-lines deadline check.
# Capping the input chars fed to `re` bounds that superlinear blowup; the
# `regex` engine keeps the wider bound because its `timeout=` aborts mid-match.
_REGEX_MAX_LINE_CHARS_RE_FALLBACK = 2_000
# Per-file byte ceiling for reading a candidate's contents during search. Files
# larger than this are skipped (and logged) rather than read in full — a few
# hundred-MB tracked `.log`/`.json`/`.csv`/extensionless dumps must not be
# slurped into memory per search. Env-overridable for repos that need it.
_MAX_SEARCH_FILE_BYTES = int(os.environ.get("LEMONCROW_SEARCH_MAX_FILE_BYTES", str(5 * 1024 * 1024)))
# Upper bound on candidate paths materialized by `_iter_files`. Stops walking a
# pathological subtree once enough candidates are gathered to satisfy any
# caller `file_limit` (default 100) with margin. Skipped for graph modes that
# need the full candidate set (e.g. `#imported-by`).
_MAX_SEARCH_CANDIDATES = int(os.environ.get("LEMONCROW_SEARCH_MAX_CANDIDATES", str(20_000)))
# Fast grep backends — ripgrep is strongly preferred; system grep is the
# fallback. The Python directory-walk path is not used for content_regex
# searches on any Mac/Linux system where one of these is always available.
_RG_BIN: str | None = shutil.which("rg")
_GREP_BIN: str | None = shutil.which("grep")
SKIP_DIRS: frozenset[str] = frozenset(
    {
        # VCS
        ".git",
        # LemonCrow internals
        ".lemoncrow",
        # Python
        ".venv",
        "venv",
        ".tox",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".eggs",
        # JS/TS
        "node_modules",
        ".next",
        ".nuxt",
        ".turbo",
        ".svelte-kit",
        # Build outputs
        "dist",
        "build",
        "out",
        "target",
        # Coverage
        "coverage",
        ".nyc_output",
    }
)
_SKIP_DIRS = SKIP_DIRS  # backwards-compat alias


def _append_rg_skip_globs(cmd: list[str]) -> None:
    """Keep ripgrep's fast path aligned with the Python walker exclusions."""
    for directory in sorted(_SKIP_DIRS):
        cmd.extend(["--glob", f"!**/{directory}/**"])


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_PDF_SUFFIXES = {".pdf"}
_BINARY_SUFFIXES = {
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".jar",
    ".so",
    ".o",
    ".a",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
    ".class",
    ".wasm",
    ".pyc",
    ".pyd",
    ".obj",
    ".lib",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".eot",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".parquet",
    ".npy",
    ".npz",
    ".pkl",
    ".ico",
    ".tiff",
    ".heic",
}
_TEXT_SUFFIXES = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".sql",
    ".sh",
    ".css",
    ".html",
    ".ipynb",
    ".csv",
    ".tsv",
}
_TYPE_ALIASES = {
    "python": ["**/*.py"],
    "py": ["**/*.py"],
    "typescript": ["**/*.ts", "**/*.tsx"],
    "ts": ["**/*.ts", "**/*.tsx"],
    "javascript": ["**/*.js", "**/*.jsx"],
    "js": ["**/*.js", "**/*.jsx"],
    "markdown": ["**/*.md"],
    "md": ["**/*.md"],
    "sql": ["**/*.sql"],
    "json": ["**/*.json"],
    "yaml": ["**/*.yaml", "**/*.yml"],
    "notebook": ["**/*.ipynb"],
    "image": ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.gif", "**/*.webp", "**/*.bmp"],
    "pdf": ["**/*.pdf"],
}


@dataclass(frozen=True)
class PatternSpec:
    pattern: str
    start_line: int | None = None
    end_line: int | None = None
    graph_mode: Literal["imports", "imported_by"] | None = None


@dataclass(frozen=True)
class RankedMatch:
    file: str
    score: float
    match_count: int
    ranges: list[tuple[int, int]]
    symbols: list[str]
    why: str


def _repo_root(repo_root: str | Path | None = None) -> Path:
    return Path(repo_root or Path.cwd()).resolve()


def _safe_resolve(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    resolved = path if path.is_absolute() else root / path
    resolved = resolved.resolve()
    # Reads are not confined to the workspace root — an absolute path to a
    # config file, sibling repo, or system path is legitimate for read tools.
    # Only block relative paths that escape via ".." traversal (almost always
    # unintentional).  Writes remain strictly confined (rich_edit._resolve
    # keeps its own unconditional out-of-workspace check).
    if not path.is_absolute():
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"path escape denied: {raw_path} is outside the workspace root {root} — "
                "use an absolute path to read files outside the workspace"
            ) from exc
    return resolved


def _has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[]{}")


def _parse_pattern(pattern: str) -> PatternSpec:
    graph_mode: Literal["imports", "imported_by"] | None = None
    if pattern.endswith("#imports"):
        pattern = pattern[: -len("#imports")]
        graph_mode = "imports"
    elif pattern.endswith("#imported-by"):
        pattern = pattern[: -len("#imported-by")]
        graph_mode = "imported_by"

    match = re.search(r":L(\d+)(?:-L(\d+))?$", pattern, re.IGNORECASE)
    if not match:
        return PatternSpec(pattern=pattern, graph_mode=graph_mode)
    start_line = int(match.group(1))
    end_line = int(match.group(2) or match.group(1))
    return PatternSpec(
        pattern=pattern[: match.start()],
        start_line=start_line,
        end_line=end_line,
        graph_mode=graph_mode,
    )


def _parse_when(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("T", " ").replace("Z", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _iter_files(
    root: Path, base: Path, patterns: list[PatternSpec], type_alias: str | None
) -> list[tuple[Path, PatternSpec]]:
    expanded = list(patterns)
    if type_alias:
        expanded.extend(PatternSpec(pattern=item) for item in _TYPE_ALIASES.get(type_alias.lower(), []))
    # `#imported-by` resolves which files import a target, so it must scan the
    # whole candidate set — the walk cap would silently drop importers. Every
    # other mode is bounded so a pathological subtree cannot be materialized in
    # full. The cap is generous (default 20k) so normal repos are unaffected.
    capped = not any(spec.graph_mode == "imported_by" for spec in expanded)
    cap = _MAX_SEARCH_CANDIDATES if capped else None

    def _capped(count: int) -> bool:
        if cap is not None and count >= cap:
            logging.info(
                "search candidate walk capped at %d files; some files under %s are "
                "not searched (raise LEMONCROW_SEARCH_MAX_CANDIDATES or narrow the path/glob)",
                cap,
                base,
            )
            return True
        return False

    if not expanded and base.is_file():
        expanded.append(PatternSpec(pattern=str(base.relative_to(root))))
    elif not expanded and base.is_dir():
        # No glob/type specified — walk all files under base
        fallback_spec = PatternSpec(pattern=".")
        found: dict[Path, PatternSpec] = {}
        for item in base.rglob("*"):
            if not item.is_file() or any(part in _SKIP_DIRS for part in item.parts):
                continue
            resolved = item.resolve()
            if not resolved.is_relative_to(root):
                continue
            found.setdefault(resolved, fallback_spec)
            if _capped(len(found)):
                break
        return sorted(found.items(), key=lambda pair: str(pair[0]))

    candidates: dict[Path, PatternSpec] = {}
    for spec in expanded:
        raw = spec.pattern or "."
        if _has_glob(raw):
            # Absolute globs (e.g. "/etc/*") and "../"-escaping globs must not
            # bypass the workspace boundary — resolve every match and require
            # it to live under root before admitting it.
            if Path(raw).is_absolute():
                continue
            for match in base.glob(raw):
                if not match.is_file():
                    continue
                resolved = match.resolve()
                if not resolved.is_relative_to(root):
                    continue
                if any(part in _SKIP_DIRS for part in resolved.parts):
                    continue
                candidates.setdefault(resolved, spec)
                if _capped(len(candidates)):
                    break
            continue

        candidate = _safe_resolve(root, raw)
        if candidate.is_file():
            candidates.setdefault(candidate, spec)
            continue
        if candidate.is_dir():
            for item in candidate.rglob("*"):
                if not item.is_file() or any(part in _SKIP_DIRS for part in item.parts):
                    continue
                resolved = item.resolve()
                if not resolved.is_relative_to(root):
                    continue
                candidates.setdefault(resolved, spec)
                if _capped(len(candidates)):
                    break

    return sorted(candidates.items(), key=lambda pair: str(pair[0]))


def _is_text_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return True
    # Search anything that is not a known binary type: covers code files like
    # .rs/.go/.java/.c and extensionless files (Dockerfile, Makefile, LICENSE).
    return suffix not in _IMAGE_SUFFIXES and suffix not in _PDF_SUFFIXES and suffix not in _BINARY_SUFFIXES


def _read_search_text(path: Path) -> str | None:
    """Read a candidate's text for matching, skipping oversized files.

    `_is_text_file` admits large `.log`/`.json`/`.csv`/extensionless dumps, and
    reading a few hundred-MB tracked file in full per search is a DoS surface.
    Files over ``_MAX_SEARCH_FILE_BYTES`` are skipped (logged so coverage isn't
    silently truncated); returns ``None`` for a skip and on read errors.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > _MAX_SEARCH_FILE_BYTES:
        logging.info(
            "search skipped %s: %d bytes exceeds the %d-byte per-file cap "
            "(raise LEMONCROW_SEARCH_MAX_FILE_BYTES to include it)",
            path,
            size,
            _MAX_SEARCH_FILE_BYTES,
        )
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _over_cap_size(path: Path) -> int | None:
    """Return the file's size in bytes when it exceeds the per-file read cap.

    Lets callers distinguish an oversized-skip (so the omission can be surfaced)
    from a no-match or read error, both of which also yield no rendered output.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    return size if size > _MAX_SEARCH_FILE_BYTES else None


def _skipped_oversized_footer(skipped: list[str]) -> str:
    """One-line footer naming files dropped for exceeding the per-file cap."""
    cap_mb = _MAX_SEARCH_FILE_BYTES / (1024 * 1024)
    shown = skipped[:10]
    listed = ", ".join(shown)
    overflow = len(skipped) - len(shown)
    if overflow > 0:
        listed += f", and {overflow} more"
    return f"[{len(skipped)} skipped >{cap_mb:g}MB: {listed}]"


def _line_window(lines: list[str], line_no: int, before: int, after: int) -> tuple[int, int, list[str]]:
    start = max(1, line_no - before)
    end = min(len(lines), line_no + after)
    return start, end, lines[start - 1 : end]


_CODEY_SUFFIXES = {".py", ".pyi", ".ts", ".tsx", ".js", ".jsx"}
_DOCSTRING_QUOTES = ('"""', "'''")


def _collapse_docstrings(window: list[str], *, threshold: int = 8, keep_head: int = 2) -> list[str]:
    """Collapse long triple-quoted blocks (NumPy-style docstrings / big string
    constants) to their summary + an elision marker, leaving signatures and code
    untouched. The agent asked to read a function, not its 80-line param table:
    keep the opener + first ``keep_head`` content lines, drop the prose bulk.

    Only blocks longer than ``threshold`` lines are collapsed; everything else is
    returned verbatim. Language-agnostic enough for Python/JS triple/backtick-free
    docstrings (handles ``\"\"\"`` and ``'''``).
    """
    out: list[str] = []
    i, n = 0, len(window)
    while i < n:
        line = window[i]
        opener = next((q for q in _DOCSTRING_QUOTES if q in line), None)
        # An opener that does not also close on the same line starts a block.
        if opener is not None and line.count(opener) == 1:
            j = i + 1
            while j < n and opener not in window[j]:
                j += 1
            if j < n and (j - i + 1) > threshold:
                out.append(line)
                out.extend(window[i + 1 : min(i + 1 + keep_head, j)])
                elided = j - (i + keep_head)
                if elided > 0:
                    indent = window[j][: len(window[j]) - len(window[j].lstrip())]
                    out.append(f"{indent}# … {elided} docstring line(s) elided; read the range to expand …")
                out.append(window[j])
                i = j + 1
                continue
        out.append(line)
        i += 1
    return out


def _truncate_line(line: str, max_line_length: int | None) -> str:
    if not max_line_length or max_line_length <= 0:
        return line
    return line if len(line) <= max_line_length else line[:max_line_length] + "..."


def _claude_grep_path_baseline_bytes(rel_path: str) -> int:
    return len(rel_path) + 1


def _claude_read_baseline_bytes(source: str) -> int:
    return len(claude_read_baseline_text(source))


def _spill_dir() -> Path:
    configured = os.environ.get("LEMONCROW_MCP_SPILL_DIR")
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        path = Path(tempfile.gettempdir()) / "lemoncrow-spill"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _spill_response_payload(payload: dict[str, Any]) -> Path:
    spill_path = _spill_dir() / f"search-{int(time.time() * 1000)}.json"
    spill_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return spill_path


def _python_summary(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    rows: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            rows.append(f"{type(node).__name__}: {node.name} @ line {node.lineno}")
    return "\n".join(rows)


def _js_summary(source: str) -> str:
    rows: list[str] = []
    pattern = re.compile(
        r"(?:export\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_$][\w$]*)|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(?[^=]*?=>"
    )
    for idx, line in enumerate(source.splitlines(), start=1):
        match = pattern.search(line)
        if match:
            rows.append(f"symbol: {match.group(1) or match.group(2)} @ line {idx}")
    return "\n".join(rows[:200])


def _summarize(path: Path, source: str) -> str:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _python_summary(source)
    if suffix in {".ts", ".tsx", ".js", ".jsx"}:
        return _js_summary(source)
    return ""


def _compact_notebook(source: str) -> str:
    try:
        notebook = json.loads(source)
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return source[:4000]
    rows: list[str] = []
    for idx, cell in enumerate(notebook.get("cells", [])):
        cell_type = cell.get("cell_type", "cell")
        raw_source = cell.get("source", "")
        text = "".join(raw_source) if isinstance(raw_source, list) else str(raw_source)
        rows.append(f"# cell {idx} ({cell_type})")
        rows.append(text[:2000])
        if cell_type == "code" and cell.get("outputs"):
            rows.append("# outputs: cached/read-only; cleared when source changes")
    return "\n".join(rows)


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return f"[PDF text extraction unavailable: install pypdf to read {path.name}]"
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        logging.exception("Recovered from broad exception handler")
        return f"[PDF text extraction failed: {exc}]"


def _imports_for(path: Path, source: str) -> list[str]:
    imports: list[str] = []
    if path.suffix.lower() == ".py":
        for match in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", source, flags=re.M):
            imports.append(match.group(1) or match.group(2) or "")
    else:
        for match in re.finditer(
            r"(?:from\s+['\"]([^'\"]+)['\"]|import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)|require\(\s*['\"]([^'\"]+)['\"]\s*\))",
            source,
        ):
            imports.append(next(group for group in match.groups() if group))
    return [item for item in imports if item]


def _imported_by_for(root: Path, target: Path, candidates: list[tuple[Path, PatternSpec]]) -> list[str]:
    rel_target = str(target.relative_to(root)) if target.is_relative_to(root) else str(target)
    stem = target.stem
    module = rel_target.removesuffix(".py").replace("/", ".")
    target_js = rel_target.removesuffix(".js")
    imported_by: list[str] = []
    for candidate, _spec in candidates:
        if candidate == target or not candidate.is_file():
            continue
        source = _read_search_text(candidate)
        if source is None:
            continue
        imports = _imports_for(candidate, source)
        hit = False
        for item in imports:
            if item == module or item.endswith(f".{stem}") or item.endswith(f"/{stem}") or item.endswith(target_js):
                hit = True
                break
        if not hit:
            continue
        rel = str(candidate.relative_to(root)) if candidate.is_relative_to(root) else str(candidate)
        imported_by.append(rel)
    return sorted(imported_by)


def _looks_plain_identifier_query(content_regex: str | None) -> bool:
    if not content_regex:
        return False
    if re.search(r"[.^$*+?{}\[\]|()\\]", content_regex):
        return False
    return bool(re.search(r"[A-Za-z0-9]", content_regex))


def _query_variants(content_regex: str | None) -> list[str]:
    if not _looks_plain_identifier_query(content_regex):
        return [content_regex] if content_regex else []
    assert content_regex is not None
    base = content_regex.strip()
    tokens = re.split(r"[\s_-]+", base)
    cleaned = [tok for tok in tokens if tok]
    if not cleaned:
        return [base]
    snake = "_".join(tok.lower() for tok in cleaned)
    kebab = "-".join(tok.lower() for tok in cleaned)
    camel = cleaned[0].lower() + "".join(tok.capitalize() for tok in cleaned[1:])
    pascal = "".join(tok.capitalize() for tok in cleaned)
    variants = [base, snake, kebab, camel, pascal, base.replace(" ", "_"), base.replace(" ", "-")]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _find_symbol_spans(path: Path, source: str) -> list[tuple[int, int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        pattern = r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_]\w*)"
    elif suffix in {".ts", ".tsx", ".js", ".jsx"}:
        pattern = r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_$][\w$]*)"
    else:
        return []
    # `finditer` yields matches in ascending position, so a forward cursor makes
    # total newline counting O(len(source)) instead of O(matches x len(source)).
    # The previous `source.count("\n", 0, match.start())` per match was quadratic
    # in file size and ran for minutes on large generated .ts files.
    starts: list[int] = []
    names: list[str] = []
    line = 1
    cursor = 0
    for match in re.finditer(pattern, source, flags=re.M):
        line += source.count("\n", cursor, match.start())
        cursor = match.start()
        starts.append(line)
        names.append(match.group(1))
    if not starts:
        return []
    total_lines = len(source.splitlines())
    finalized: list[tuple[int, int, str]] = []
    for idx, start in enumerate(starts):
        next_start = starts[idx + 1] if idx + 1 < len(starts) else total_lines + 1
        finalized.append((start, max(start, next_start - 1), names[idx]))
    return finalized


def _match_line_numbers(
    lines: list[str],
    regex: re.Pattern[str] | None,
    content_regex: str | None,
    *,
    include_all_when_no_regex: bool,
    deadline: float | None = None,
) -> list[int]:
    if regex is not None:
        out: list[int] = []
        # The `regex` engine accepts a per-call `timeout=` that raises
        # TimeoutError, giving a true wall-clock bound on a single catastrophic
        # backtracking search. Stdlib `re` has no such hook, so it relies only
        # on the between-iteration deadline check below.
        timed_pattern: Any = regex if _regex_module is not None and isinstance(regex, _regex_module.Pattern) else None
        # On the stdlib `re` fallback (no per-call wall-clock hook) feed each
        # line through a tighter char cap so one catastrophic-backtracking match
        # cannot run to completion in C past a bounded input length.
        max_line_chars = _REGEX_MAX_LINE_CHARS if timed_pattern is not None else _REGEX_MAX_LINE_CHARS_RE_FALLBACK
        for idx, line in enumerate(lines, start=1):
            if deadline is not None and time.monotonic() > deadline:
                break
            window = line[:max_line_chars]
            if timed_pattern is not None and deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    matched = timed_pattern.search(window, timeout=remaining)
                except TimeoutError:
                    logging.warning(
                        "content_regex search hit the %.1fs wall-clock budget; "
                        "returning partial matches (possible catastrophic backtracking)",
                        _REGEX_DEADLINE_SECONDS,
                    )
                    break
            else:
                matched = regex.search(window)
            if matched:
                out.append(idx)
        return out
    if include_all_when_no_regex:
        return list(range(1, len(lines) + 1))
    variants = _query_variants(content_regex)
    if not variants:
        return []
    compiled = [re.compile(re.escape(item), re.I) for item in variants]
    out = []
    for idx, line in enumerate(lines, start=1):
        if any(pat.search(line) for pat in compiled):
            out.append(idx)
    return out


def _merge_ranges(ranges: list[tuple[int, int]], *, gap: int = 3) -> list[tuple[int, int]]:
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda item: (item[0], item[1]))
    merged: list[tuple[int, int]] = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, end))
            continue
        merged.append((start, end))
    return merged


def _symbol_windows(
    path: Path,
    lines: list[str],
    source: str,
    line_nos: list[int],
    *,
    lines_before: int,
    lines_after: int,
) -> tuple[list[tuple[int, int]], list[str]]:
    symbols = _find_symbol_spans(path, source)
    # Spans are sorted by start and non-overlapping (each ends one line before the
    # next begins), so a binary search finds the containing span in O(log n) --
    # the previous inner linear scan was O(line_nos x symbols), quadratic on big
    # files where both grow together.
    symbol_starts = [start for start, _end, _name in symbols]
    windows: list[tuple[int, int]] = []
    symbol_hits: list[str] = []
    for line_no in line_nos:
        matched_symbol = None
        idx = bisect.bisect_right(symbol_starts, line_no) - 1
        if idx >= 0:
            start, end, name = symbols[idx]
            if start <= line_no <= end:
                matched_symbol = (start, end, name)
        if matched_symbol is not None:
            start, end, name = matched_symbol
            windows.append((start, end))
            symbol_hits.append(name)
            continue
        start = max(1, line_no - lines_before)
        end = min(len(lines), line_no + lines_after)
        windows.append((start, end))
    dedup_symbols: list[str] = []
    seen: set[str] = set()
    for name in symbol_hits:
        if name in seen:
            continue
        seen.add(name)
        dedup_symbols.append(name)
    return _merge_ranges(windows), dedup_symbols


def _rg_candidate_files(
    *,
    pattern: str,
    base: Path,
    glob_patterns: list[str],
    ignore_case: bool,
    timeout: float,
) -> list[Path] | None:
    """Return files matching *pattern* via ``rg -l``. None = error / timeout."""
    assert _RG_BIN is not None
    cmd: list[str] = [_RG_BIN, "--files-with-matches", "--no-messages", "--hidden"]
    if ignore_case:
        cmd.append("-i")
    for g in glob_patterns:
        cmd.extend(["--glob", g])
    _append_rg_skip_globs(cmd)
    cmd.extend(["--", pattern, str(base)])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode not in (0, 1):  # 0=found, 1=no match, other=error
        return None
    return [Path(line) for line in proc.stdout.splitlines() if line]


def _grep_candidate_files(
    *,
    pattern: str,
    base: Path,
    glob_patterns: list[str],
    ignore_case: bool,
    timeout: float,
) -> list[Path] | None:
    """Return files matching *pattern* via ``grep -rl``. None = error / timeout."""
    assert _GREP_BIN is not None
    cmd: list[str] = [_GREP_BIN, "-rl", "-H"]
    if ignore_case:
        cmd.append("-i")
    for g in glob_patterns:
        # grep --include matches basenames only; a path-qualified glob like
        # "src/**/*.py" would otherwise never match anything.
        cmd.extend(["--include", Path(g).name])
    cmd.extend(["--", pattern, str(base)])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode not in (0, 1):
        return None
    return [Path(line) for line in proc.stdout.splitlines() if line]


def _rg_line_numbers(
    *,
    pattern: str,
    base: Path,
    glob_patterns: list[str],
    ignore_case: bool,
    timeout: float,
) -> dict[str, list[int]] | None:
    """Return {abs_path: [1-based line nos]} via ``rg -n``. None = error / timeout."""
    assert _RG_BIN is not None
    cmd: list[str] = [
        _RG_BIN,
        "--line-number",
        "--no-heading",
        "--with-filename",
        "--no-messages",
        "--hidden",
        # NUL after the filename so a path containing ':' is parsed unambiguously
        # (plain "path:lineno:content" splitting drops such files silently).
        "--null",
    ]
    if ignore_case:
        cmd.append("-i")
    for g in glob_patterns:
        cmd.extend(["--glob", g])
    _append_rg_skip_globs(cmd)
    cmd.extend(["--", pattern, str(base)])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode not in (0, 1):
        return None
    result: dict[str, list[int]] = {}
    for line in proc.stdout.splitlines():
        # format: /abs/path\0lineno:content  (NUL isolates the path, which may
        # itself contain ':'); the remainder is "lineno:content".
        path, sep, rest = line.partition("\0")
        if not sep:
            continue
        lineno_str = rest.partition(":")[0]
        try:
            lineno = int(lineno_str)
        except ValueError:
            continue
        result.setdefault(path, []).append(lineno)
    return result


def _grep_line_numbers(
    *,
    pattern: str,
    base: Path,
    glob_patterns: list[str],
    ignore_case: bool,
    timeout: float,
) -> dict[str, list[int]] | None:
    """Return {abs_path: [1-based line nos]} via ``grep -rn``. None = error / timeout."""
    assert _GREP_BIN is not None
    # -Z: NUL after the filename so a path containing ':' is parsed unambiguously.
    cmd: list[str] = [_GREP_BIN, "-rn", "-H", "-Z"]
    if ignore_case:
        cmd.append("-i")
    for g in glob_patterns:
        # grep --include matches basenames only; a path-qualified glob like
        # "src/**/*.py" would otherwise never match anything.
        cmd.extend(["--include", Path(g).name])
    cmd.extend(["--", pattern, str(base)])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode not in (0, 1):
        return None
    result: dict[str, list[int]] = {}
    for line in proc.stdout.splitlines():
        # format: path\0lineno:content  (NUL isolates a path that may contain ':')
        path, sep, rest = line.partition("\0")
        if not sep:
            continue
        lineno_str = rest.partition(":")[0]
        try:
            lineno = int(lineno_str)
        except ValueError:
            continue
        result.setdefault(path, []).append(lineno)
    return result


def _render_text_result(
    path: Path,
    root: Path,
    spec: PatternSpec,
    regex: re.Pattern[str] | None,
    content_regex: str | None,
    *,
    output_mode: SearchOutputMode,
    lines_before: int,
    lines_after: int,
    max_line_length: int | None,
    lines_per_file: int | None,
    summary: bool | None,
    if_modified_since: datetime | None,
    deadline: float | None = None,
    badge_provider: Callable[[str, list[str]], str | None] | None = None,
    precomputed_match_lines: list[int] | None = None,
) -> tuple[str | None, int]:
    rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    unchanged = if_modified_since is not None and mtime <= if_modified_since
    if unchanged:
        if output_mode == "file_paths_only":
            return None, 0
        if output_mode == "file_paths_with_match_count":
            return f"{rel}\t0 (unchanged)", 0
        if output_mode == "file_paths_with_content":
            return f"{rel} (unchanged)", 0

    if path.suffix.lower() in _PDF_SUFFIXES:
        source = _extract_pdf(path)
    else:
        read = _read_search_text(path)
        if read is None:
            # Oversized or unreadable: skip rather than slurp the file in full.
            return None, 0
        source = read
    if path.suffix.lower() == ".ipynb":
        source = _compact_notebook(source)

    if spec.graph_mode == "imports":
        imports = _imports_for(path, source)
        return f"{rel}\nimports:\n" + "\n".join(f"- {item}" for item in imports), len(imports)
    if spec.graph_mode == "imported_by":
        # Handled in caller where we have full candidate list context.
        return None, 0

    lines = source.splitlines()
    # A ":Lx-Ly" suffix with no pattern is a range read via grep: return the
    # raw slice. With a pattern, the range instead scopes which matches report.
    if spec.start_line is not None and regex is None and content_regex is None:
        start = spec.start_line
        end = spec.end_line or start
        selected = lines[start - 1 : end]
        body = redact_tool_output("\n".join(_truncate_line(line, max_line_length) for line in selected))
        return f"{rel}:L{start}-L{end}\n{body}", len(selected)

    include_all = regex is None and content_regex is None
    if precomputed_match_lines is not None:
        match_lines = precomputed_match_lines
    else:
        match_lines = _match_line_numbers(
            lines, regex, content_regex, include_all_when_no_regex=include_all, deadline=deadline
        )
    if spec.start_line is not None:
        lo, hi = spec.start_line, spec.end_line or spec.start_line
        match_lines = [n for n in match_lines if lo <= n <= hi]
    if include_all and lines_per_file:
        match_lines = match_lines[: max(0, lines_per_file)]

    if output_mode == "file_paths_only":
        return rel if match_lines or not regex else None, len(match_lines)
    if output_mode == "file_paths_with_match_count":
        return f"{rel}\t{len(match_lines)}", len(match_lines)
    if regex and not match_lines:
        return None, 0

    use_summary = bool(summary)
    # When content_regex is provided, the user explicitly wants line matches.
    # Auto-summary would discard them — skip it.
    if (
        summary is None
        and regex is None
        and path.suffix.lower() in {".py", ".ts", ".tsx", ".js", ".jsx"}
        and len(lines) > 500
    ):
        use_summary = True
    if use_summary:
        outline = _summarize(path, source)
        if outline:
            return f"{rel}\n{redact_tool_output(outline)}", len(match_lines)

    # When a badge provider is supplied, surface the call-graph counts for any
    # symbol definitions the matches land in -- appended to the file header so the
    # relational facts ride along the search the agent already ran.
    header = rel
    if badge_provider is not None and content_regex is not None:
        _windows, matched_symbols = _symbol_windows(
            path, lines, source, match_lines, lines_before=lines_before, lines_after=lines_after
        )
        if matched_symbols:
            badge = badge_provider(rel, matched_symbols)
            if badge:
                header = f"{rel}  ·  {badge}"
    rendered: list[str] = [header]
    emitted = 0
    windows_seen: set[tuple[int, int]] = set()
    for line_no in match_lines:
        if lines_per_file and lines_per_file > 0 and emitted >= lines_per_file:
            break
        start, end, window = _line_window(lines, line_no, lines_before, lines_after)
        if (start, end) in windows_seen:
            continue
        windows_seen.add((start, end))
        # Lean output: an agent reading a function body via grep+context does not
        # need the 80-line NumPy docstring -- collapse it to summary + marker.
        # summary=False (explicit raw) opts out.
        if summary is not False and path.suffix.lower() in _CODEY_SUFFIXES and len(window) > 8:
            window = _collapse_docstrings(window)
        rendered.append(f"@@ {start}-{end}")
        rendered.extend(_truncate_line(line, max_line_length) for line in window)
        emitted += len(window)
    return redact_tool_output("\n".join(rendered)), len(match_lines)


def search_workspace(
    *,
    path: str = ".",
    content_regex: str | None = None,
    file_glob_patterns: list[str] | None = None,
    output_mode: SearchOutputMode = "file_paths_with_content",
    lines_before: int = 0,
    lines_after: int = 0,
    ignore_case: bool = False,
    type: str | None = None,
    file_limit: int | None = None,
    lines_per_file: int | None = 500,
    if_modified_since: str | None = None,
    max_line_length: int | None = 1000,
    multiline: bool = False,
    summary: bool | None = None,
    repo_root: str | Path | None = None,
    cap_chars: int = MAX_STRUCTURED_OUTPUT_CHARS,
    context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS,
    include_metadata: bool = True,
    badge_provider: Callable[[str, list[str]], str | None] | None = None,
) -> dict[str, Any]:
    """Search and read files in one structured response.

    ``badge_provider`` (optional, content mode only) receives ``(rel_path, [symbol
    names matched as definitions])`` and returns a short string appended to that
    file's header -- used to ride call-graph counts along regex matches. It stays
    Index-agnostic here: the caller supplies the lookup.
    """
    # Track naive vs rendered bytes to compute tokens_saved. Naive = grep
    # output bytes (matching lines + context, not full file); rendered = what
    # LemonCrow actually returns after ranking/summarisation. ~4 bytes/token.
    naive_bytes = 0
    # Files dropped for exceeding the per-file read cap, surfaced in the result
    # so a present-but-skipped match is not reported as a false "no match".
    skipped_oversized: list[str] = []
    root = _repo_root(repo_root)
    base_spec = _parse_pattern(path)
    base = _safe_resolve(root, base_spec.pattern or ".")
    # Resolve the base first (which strips any ":Lx-Ly" line-range suffix)
    # so a bare "file.py:L60-L100" path is accepted as a single-file search rather
    # than rejected for having no pattern.
    if not (content_regex or file_glob_patterns or type or base.is_file()):
        return {
            "isError": True,
            "content": [{"type": "text", "text": "Provide content_regex, file_glob_patterns, or type"}],
        }
    specs = [_parse_pattern(item) for item in (file_glob_patterns or [])]
    if not specs and base.is_file():
        specs = [base_spec]

    flags = re.I if ignore_case else 0
    if multiline:
        flags |= re.S | re.M
    if content_regex:
        # Prefer the `regex` engine so `_match_line_numbers` can impose a true
        # per-call wall-clock bound via `.search(text, timeout=...)`; fall back
        # to stdlib `re` (between-iteration deadline only) when it is absent.
        compile_engine = _regex_module or re
        try:
            regex = compile_engine.compile(content_regex, flags)
        except compile_engine.error:
            # Pattern failed as a regex (e.g. unbalanced `(` from a shell grep
            # command where `(` is a literal).  Fall back to a literal-string
            # match so `grep -n "print(" file.py` works as it would in a shell.
            try:
                regex = re.compile(re.escape(content_regex), flags)
            except re.error as exc:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Invalid content_regex: {exc}"}],
                }
    else:
        regex = None
    # Bound total time spent running a user-supplied regex across files so a
    # catastrophic-backtracking pattern cannot hang the worker indefinitely.
    deadline = time.monotonic() + _REGEX_DEADLINE_SECONDS if regex is not None else None
    since = _parse_when(if_modified_since)
    # Fast grep path: delegate content_regex matching to rg/grep subprocess to
    # avoid the ~1200ms _iter_files walk + ~5s Python re-scan on large repos.
    _rg_line_map: dict[str, list[int]] | None = None
    _fast_candidates: list[tuple[Path, PatternSpec]] | None = None
    _fast_eligible = (
        regex is not None
        and not multiline
        and since is None
        and not any(s.graph_mode for s in specs)
        and (_RG_BIN is not None or _GREP_BIN is not None)
    )
    if _fast_eligible:
        assert content_regex is not None
        _use_rg = _RG_BIN is not None
        _fast_globs: list[str] = [s.pattern for s in specs if s.pattern and _has_glob(s.pattern)]
        if not _fast_globs and type is not None:
            _fast_globs = _TYPE_ALIASES.get(type.lower(), [])
        _grep_kw: dict[str, Any] = dict(
            pattern=content_regex,
            base=base,
            glob_patterns=_fast_globs,
            ignore_case=ignore_case,
            timeout=_REGEX_DEADLINE_SECONDS + 2.0,
        )
        if output_mode == "file_paths_only":
            _found_l = _rg_candidate_files(**_grep_kw) if _use_rg else _grep_candidate_files(**_grep_kw)
            if _found_l is not None:
                _fp_list: list[str] = []
                for _p in _found_l:
                    try:
                        _fp_list.append(str(_p.relative_to(root)))
                    except ValueError:
                        _fp_list.append(str(_p))
                _agg_parts = [f"# grep ({len(_fp_list)} files)"]
                if _fp_list:
                    _agg_parts.append("")
                    _agg_parts.extend(_fp_list[:_FILE_PATHS_ONLY_CAP])
                    _ov = len(_fp_list) - _FILE_PATHS_ONLY_CAP
                    if _ov > 0:
                        _agg_parts.append(f"... and {_ov} more")
                _fast_resp: dict[str, Any] = {
                    "content": [{"type": "text", "text": "\n".join(_agg_parts)}],
                    "tokens_saved": 0,
                }
                if include_metadata:
                    _fast_resp["_meta"] = {"fileMatchCount": len(_fp_list), "capChars": cap_chars}
                return _fast_resp
        else:
            _line_fn = _rg_line_numbers if _use_rg else _grep_line_numbers
            _line_map = _line_fn(**_grep_kw)
            if _line_map is not None:
                _rg_line_map = _line_map
                _dummy_spec = specs[0] if specs else base_spec
                _fast_candidates = [(Path(k), _dummy_spec) for k in _rg_line_map]
                deadline = time.monotonic() + _REGEX_DEADLINE_SECONDS  # reset deadline
    candidates = (
        _fast_candidates
        if _fast_candidates is not None
        else _iter_files(root, base if base.is_dir() else root, specs, type)
    )
    limit = file_limit or 100
    blocks: list[dict[str, str]] = []
    total_chars = 0
    effective_cap_chars = cap_chars
    if output_mode == "file_paths_with_content":
        effective_cap_chars = min(cap_chars, max(1000, context_budget_tokens) * 4)
    if output_mode == "ranked_file_map":
        ranked: list[RankedMatch] = []
        total_budget = max(1000, context_budget_tokens)
        for candidate, spec in candidates:
            if len(ranked) >= limit:
                break
            if not _is_text_file(candidate) and candidate.suffix.lower() not in _PDF_SUFFIXES:
                continue
            if since is not None and datetime.fromtimestamp(candidate.stat().st_mtime) <= since:
                continue
            if candidate.suffix.lower() in _PDF_SUFFIXES:
                source = _extract_pdf(candidate)
            else:
                read = _read_search_text(candidate)
                if read is None:
                    if _over_cap_size(candidate) is not None:
                        rel = str(candidate.relative_to(root)) if candidate.is_relative_to(root) else str(candidate)
                        skipped_oversized.append(rel)
                    continue
                source = read
            lines = source.splitlines()
            if spec.graph_mode == "imports":
                imports = _imports_for(candidate, source)
                rel = str(candidate.relative_to(root)) if candidate.is_relative_to(root) else str(candidate)
                naive_bytes += _claude_read_baseline_bytes(source)
                ranked.append(
                    RankedMatch(
                        file=rel,
                        score=1.0,
                        match_count=len(imports),
                        ranges=[],
                        symbols=[],
                        why="imports graph requested",
                    )
                )
                continue
            if spec.graph_mode == "imported_by":
                imported = _imported_by_for(root, candidate, candidates)
                rel = str(candidate.relative_to(root)) if candidate.is_relative_to(root) else str(candidate)
                naive_bytes += _claude_read_baseline_bytes(source)
                ranked.append(
                    RankedMatch(
                        file=rel,
                        score=1.0,
                        match_count=len(imported),
                        ranges=[],
                        symbols=[],
                        why=f"imported by {len(imported)} files",
                    )
                )
                continue

            _ckey = str(candidate)
            _pre = _rg_line_map.get(_ckey) if _rg_line_map else None
            if _pre is not None:
                line_nos = _pre
            else:
                line_nos = _match_line_numbers(
                    lines, regex, content_regex, include_all_when_no_regex=regex is None, deadline=deadline
                )
            if spec.start_line is not None:
                lo, hi = spec.start_line, spec.end_line or spec.start_line
                line_nos = [n for n in line_nos if lo <= n <= hi]
                if not line_nos:
                    continue
            if regex and not line_nos:
                continue
            rel = str(candidate.relative_to(root)) if candidate.is_relative_to(root) else str(candidate)
            # ranked_file_map is closest to Claude Grep/Glob path output, not
            # a full or contextual Read of every matched file.
            naive_bytes += _claude_grep_path_baseline_bytes(rel)
            ranges, symbols = _symbol_windows(
                candidate,
                lines,
                source,
                line_nos,
                lines_before=max(0, lines_before),
                lines_after=max(0, lines_after),
            )
            match_count = len(line_nos)
            score = float(match_count) + (0.15 * len(symbols))
            ranked.append(
                RankedMatch(
                    file=rel,
                    score=score,
                    match_count=match_count,
                    ranges=ranges,
                    symbols=symbols[:6],
                    why="regex matched symbol-aware ranges" if symbols else "regex matched merged line ranges",
                )
            )

        if not ranked:
            payload: dict[str, Any] = {
                "mode": "ranked_file_map",
                "matches": [],
                "next": [],
                "tokens_saved": naive_bytes // 4,
            }
            if skipped_oversized:
                payload["skipped"] = _skipped_oversized_footer(skipped_oversized)
            if include_metadata:
                payload["_meta"] = {"fileMatchCount": 0, "capChars": cap_chars}
            return payload

        ranked.sort(key=lambda item: (-item.score, item.file))
        top_score = max(item.score for item in ranked) or 1.0
        normalized = [replace(item, score=round(item.score / top_score, 3)) for item in ranked]
        selected: list[RankedMatch] = []
        used_tokens = 0
        for idx, item in enumerate(normalized):
            max_ranges = 4 if idx == 0 else 3 if idx == 1 else 2
            reduced_ranges = item.ranges[:max_ranges]
            est = max(30, len(item.file) // 4 + (len(reduced_ranges) * 24) + (len(item.symbols) * 8))
            if selected and used_tokens + est > total_budget:
                break
            used_tokens += est
            selected.append(replace(item, ranges=reduced_ranges))

        handles: dict[str, tuple[str, tuple[int, int] | None]] = {}
        matches_payload: list[dict[str, Any]] = []
        next_actions: list[str] = []
        for idx, item in enumerate(selected, start=1):
            range_text = [f"{start}-{end}" for start, end in item.ranges]
            handle = f"m{idx}"
            handles[handle] = (item.file, item.ranges[0] if item.ranges else None)
            matches_payload.append(
                {
                    "handle": handle,
                    "file": item.file,
                    "score": item.score,
                    "match_count": item.match_count,
                    "ranges": range_text,
                    "symbols": item.symbols,
                    "why": item.why,
                }
            )
            if item.ranges:
                start, end = item.ranges[0]
                next_actions.append(f"read {item.file}#{start}-{end}")
            if item.symbols:
                next_actions.append(f"read {item.file}#{item.symbols[0]}")
        rendered_bytes = sum(
            len(item.get("file", ""))
            + sum(len(r) for r in item.get("ranges", []))
            + sum(len(s) for s in item.get("symbols", []))
            + len(item.get("why", ""))
            for item in matches_payload
        )
        payload = {
            "mode": "ranked_file_map",
            "matches": matches_payload,
            "next": next_actions[: min(12, len(next_actions))],
            "context_budget_tokens": total_budget,
            "handles": {
                k: {"file": v[0], "range": f"{v[1][0]}-{v[1][1]}" if v[1] else None} for k, v in handles.items()
            },
            "tokens_saved": max(0, (naive_bytes - rendered_bytes) // 4),
        }
        if skipped_oversized:
            payload["skipped"] = _skipped_oversized_footer(skipped_oversized)
        if include_metadata:
            payload["_meta"] = {"fileMatchCount": len(matches_payload), "capChars": cap_chars}
        return payload

    file_match_count = 0
    # For file_paths_with_* modes: accumulate into one block instead of
    # emitting individual {"type": "text", "text": "path\tN"} per file.
    mc_hit_lines: list[str] = []  # rendered "path\tN" lines where N > 0
    mc_hit_count: int = 0  # number of files with matches
    mc_zero_count: int = 0  # number of files with 0 matches
    fp_paths: list[str] = []  # accumulated paths for file_paths_only mode
    for candidate, spec in candidates:
        if len(blocks) >= limit:
            break
        suffix = candidate.suffix.lower()
        if suffix in _IMAGE_SUFFIXES and output_mode == "file_paths_with_content" and regex is None:
            # Guard the base64 path: encoding inflates the file ~1.33x in memory
            # and the whole blob rides in the tool result, so an oversized image
            # is reported as skipped rather than slurped + encoded.
            try:
                image_bytes = candidate.stat().st_size
            except OSError:
                continue
            if image_bytes > _MAX_SEARCH_FILE_BYTES:
                rel = str(candidate.relative_to(root)) if candidate.is_relative_to(root) else str(candidate)
                skipped_oversized.append(rel)
                continue
            data = base64.b64encode(candidate.read_bytes()).decode("ascii")
            mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            blocks.append({"type": "image", "data": data, "mimeType": mime})
            file_match_count += 1
            continue
        if not _is_text_file(candidate) and suffix not in _PDF_SUFFIXES:
            continue
        if spec.graph_mode == "imported_by":
            imported = _imported_by_for(root, candidate, candidates)
            rel = str(candidate.relative_to(root)) if candidate.is_relative_to(root) else str(candidate)
            rendered = f"{rel}\nimported-by:\n" + "\n".join(f"- {item}" for item in imported)
            target_source = _read_search_text(candidate)
            if target_source is not None:
                naive_bytes += _claude_read_baseline_bytes(target_source)
            file_match_count += 1
            remaining = effective_cap_chars - total_chars
            if remaining <= 0:
                blocks.append({"type": "text", "text": "[truncated: structured output cap reached]"})
                break
            text = rendered[:remaining]
            if len(rendered) > remaining:
                text += f"\n[truncated: {len(rendered) - remaining} omitted]"
            total_chars += len(text)
            blocks.append({"type": "text", "text": text})
            continue
        _ckey = str(candidate)
        _pre = _rg_line_map.get(_ckey) if _rg_line_map else None
        _file_rendered, _count = _render_text_result(
            candidate,
            root,
            spec,
            regex,
            content_regex,
            output_mode=output_mode,
            lines_before=max(0, lines_before),
            lines_after=max(0, lines_after),
            max_line_length=max_line_length,
            lines_per_file=lines_per_file,
            summary=summary,
            if_modified_since=since,
            deadline=deadline,
            badge_provider=badge_provider,
            precomputed_match_lines=_pre,
        )
        if _file_rendered is None:
            if _over_cap_size(candidate) is not None:
                rel = str(candidate.relative_to(root)) if candidate.is_relative_to(root) else str(candidate)
                skipped_oversized.append(rel)
            continue
        # Naive = grep output = _file_rendered (matched lines + context).
        # Savings = LemonCrow's post-processing reduction (summarisation, cap truncation).
        naive_bytes += len(_file_rendered)
        rendered = _file_rendered
        file_match_count += 1
        if output_mode == "file_paths_with_match_count":
            if _count > 0:
                mc_hit_lines.append(rendered)
                mc_hit_count += 1
            else:
                mc_zero_count += 1
            continue
        if output_mode == "file_paths_only":
            fp_paths.append(rendered)
            continue
        remaining = effective_cap_chars - total_chars
        if remaining <= 0:
            blocks.append({"type": "text", "text": "[truncated: structured output cap reached]"})
            break
        text = rendered[:remaining]
        if len(rendered) > remaining:
            text += f"\n[truncated: {len(rendered) - remaining} omitted]"
        total_chars += len(text)
        blocks.append({"type": "text", "text": text})

    if output_mode == "file_paths_with_match_count":
        # Aggregate into a single text block — one line per hit, suppress zeros.
        agg_parts: list[str] = []
        agg_parts.append(f"# grep ({mc_hit_count} files)")
        if mc_hit_lines:
            agg_parts.append("")
            agg_parts.extend(sorted(mc_hit_lines, key=lambda line: -int(line.rsplit("\t", 1)[-1])))
        text = "\n".join(agg_parts)
        total_chars = len(text)
        blocks.append({"type": "text", "text": text})

    if output_mode == "file_paths_only":
        # Aggregate paths into a single text block — one path per line.
        # Capped: an unbounded path dump (hundreds of files) permanently bloats
        # the agent context; the caller can narrow path or globs to page through.
        agg_parts = [f"# grep ({len(fp_paths)} files)"]
        if fp_paths:
            agg_parts.append("")
            agg_parts.extend(fp_paths[:_FILE_PATHS_ONLY_CAP])
            overflow = len(fp_paths) - _FILE_PATHS_ONLY_CAP
            if overflow > 0:
                agg_parts.append(f"... and {overflow} more")
        text = "\n".join(agg_parts)
        total_chars = len(text)
        blocks.append({"type": "text", "text": text})

    if skipped_oversized:
        # Surface the omission so a present-but-skipped match is not read as a
        # false "no match". Counts toward total_chars like any rendered block.
        footer = _skipped_oversized_footer(skipped_oversized)
        total_chars += len(footer)
        blocks.append({"type": "text", "text": footer})

    response: dict[str, Any] = {
        "content": blocks,
        "tokens_saved": max(0, (naive_bytes - total_chars) // 4),
    }
    if include_metadata:
        response["_meta"] = {"fileMatchCount": file_match_count, "capChars": effective_cap_chars}
    inline_chars_budget = max(1000, context_budget_tokens) * INLINE_CHARS_PER_TOKEN
    if output_mode == "file_paths_with_content" and total_chars > inline_chars_budget:
        spill_payload = {
            "mode": output_mode,
            "content": blocks,
            "_meta": {
                "fileMatchCount": file_match_count,
                "capChars": effective_cap_chars,
                "inlineChars": total_chars,
                "inlineCharsBudget": inline_chars_budget,
            },
        }
        spill_path = _spill_response_payload(spill_payload)
        preview = ""
        if blocks and isinstance(blocks[0], dict) and blocks[0].get("type") == "text":
            preview = str(blocks[0].get("text", ""))[:240]
        response = {
            "content": [
                {
                    "type": "text",
                    "text": f"[lc: spilled; {spill_path}]",
                }
            ],
            "artifact": {
                "path": str(spill_path),
                "format": "json",
                "bytes": spill_path.stat().st_size,
                "preview": preview,
            },
            # Spilled response: the agent reads only the stub, so savings are
            # naive read cost minus the stub preview length.
            "tokens_saved": max(0, (naive_bytes - len(preview)) // 4),
        }
        if include_metadata:
            response["_meta"] = {
                "fileMatchCount": file_match_count,
                "capChars": effective_cap_chars,
                "inlineChars": total_chars,
                "inlineCharsBudget": inline_chars_budget,
                "spilled": True,
            }
    return response


__all__ = ["MAX_STRUCTURED_OUTPUT_CHARS", "SearchOutputMode", "search_workspace"]

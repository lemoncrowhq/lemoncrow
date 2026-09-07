"""Evidence-only post-edit discovery for contract literals removed by an edit.

Renames and deprecations often span independent consumers with no call-graph edge
to the edited site: configuration keys, wire fields, and dict literals are plain
strings, invisible to symbol-level callers/callees/usages. When an edit removes a
quoted literal, surface the remaining occurrences in *other* files so the agent can
inspect parallel consumers while its implementation hypothesis is still revisable.

Detection combines two layers for precision *and* recall: ast-grep (the engine
behind the ``codemod`` tool) matches the literal as a string *node*, so it is precise
for code files -- it never matches the same text inside a larger string, a docstring,
or a comment. A language-agnostic text search (guarded by a structural heuristic)
adds the non-code consumers ast-grep can't parse (config, templates, docs). ast-grep
is authoritative for any code file it covers; text contributes the rest, and is the
sole path when the ast-grep binary is unavailable.

This module extracts the removed literals and module symbols and shapes the evidence.
It never blocks or rolls back an edit, and fails open (returns ``None``) on any error.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lemoncrow.core.foundation.redaction import redact_tool_output

_QUOTED_LITERAL_RE = re.compile(r"'((?:\\.|[^'\\])*)'|\"((?:\\.|[^\"\\])*)\"|`((?:\\.|[^`\\])*)`")

# Decorators whose removal silently strips attributes/methods callers may use, with
# no call-graph edge to surface the breakage. ``functools.lru_cache``/``cache`` add
# ``.cache_clear``/``.cache_info``/``.cache_parameters`` to the wrapped function;
# removing the decorator makes every ``fn.cache_clear()`` elsewhere an AttributeError.
# Generic stdlib contract -- not project-specific. Extend this map (e.g. cached_property)
# as other attribute-providing decorators prove worth surfacing.
_DECORATOR_PROVIDED_ATTRS: dict[str, tuple[str, ...]] = {
    "lru_cache": ("cache_clear", "cache_info", "cache_parameters"),
    "cache": ("cache_clear", "cache_info", "cache_parameters"),
}
# A cache decorator immediately above a (possibly async) def -- captures both.
_CACHE_DECORATED_DEF_RE = re.compile(
    r"@(?:\w+\.)*(?P<deco>" + "|".join(_DECORATOR_PROVIDED_ATTRS) + r")\b[^\n]*\n\s*(?:async\s+)?def\s+(?P<name>\w+)"
)
_NOISY_LITERALS = frozenset(
    {
        "",
        "0",
        "1",
        "false",
        "true",
        "none",
        "null",
        # Common bool synonyms -- too short and ubiquitous to be meaningful contract
        # literals (e.g. env-var coercions like `in {"1", "true", "yes", "on"}`
        # appear in dozens of unrelated places and generate FIXME noise).
        "yes",
        "no",
        "on",
        "off",
    }
)
# A literal found in this many distinct files is ambient vocabulary (e.g. common
# bool synonyms, status words) rather than a contract identifier. Skip it.
_MAX_LITERAL_FILE_SPREAD = 4

_QUOTES = ("'", '"', "`")
_DELIMITERS = frozenset("[]{}():,=")
_MAX_FILE_BYTES = 1_000_000

# Edited-file extension -> ast-grep language name. ast-grep matches per language,
# so detection covers the language(s) of the files actually edited.
_EXT_TO_ASTGREP_LANG = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".kt": "kotlin",
    ".swift": "swift",
}


if TYPE_CHECKING:
    from lemoncrow.core.capabilities.tool_supervision_contract import _TextSearcher


def _quoted_literals(text: str) -> set[str]:
    out: set[str] = set()
    for match in _QUOTED_LITERAL_RE.finditer(text):
        literal = next((group for group in match.groups() if group is not None), "")
        # Escaped values can't be searched literally without language-specific
        # decoding; one-character and very long prose strings are noisy.
        if (
            2 <= len(literal) <= 80
            and "\\" not in literal
            and "\n" not in literal
            and literal.strip().lower() not in _NOISY_LITERALS
        ):
            out.add(literal)
    return out


def literal_replacements(edits: list[dict[str, Any]], *, limit: int = 6) -> dict[str, str | None]:
    """Map each removed quoted literal to its replacement, when a rename is identifiable.

    A literal present in ``old_string`` but not ``new_string`` was removed. A rename is
    only claimed when substituting a removed literal for an added one turns some old
    line into a line that appears verbatim in ``new`` (e.g. ``config['db']`` ->
    ``config['database']``); otherwise the value is ``None`` (removed, no confident
    replacement).

    The test is by substitution, deliberately NOT by positional line pairing: an edit
    that inserts or deletes lines shifts every line below it, so ``zip``-aligning
    old/new lines -- even when the line COUNT is coincidentally equal -- pairs
    unrelated lines and manufactures phantom renames. Requiring the literal to be
    genuinely absent from ``new`` likewise stops a still-present key (unchanged, merely
    shifted next to the churn) from ever being reported as removed or renamed.
    """
    replacements: dict[str, str | None] = {}
    for edit in edits:
        old = edit.get("old_string")
        new = edit.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            continue
        removed = _quoted_literals(old) - _quoted_literals(new)
        added = _quoted_literals(new) - _quoted_literals(old)
        for literal in removed:
            replacements.setdefault(literal, None)
        if not removed or not added:
            continue
        new_line_set = set(new.splitlines())
        sorted_added = sorted(added)
        for old_line in old.splitlines():
            present = _quoted_literals(old_line) & removed
            if len(present) != 1:
                continue
            literal = next(iter(present))
            for quote in _QUOTES:
                token = f"{quote}{literal}{quote}"
                if token not in old_line:
                    continue
                target = next(
                    (r for r in sorted_added if old_line.replace(token, f"{quote}{r}{quote}") in new_line_set),
                    None,
                )
                if target is not None:
                    replacements[literal] = target
                    break
    # Prefer longer literals: more contract-specific, less noisy.
    ordered = sorted(replacements, key=lambda value: (-len(value), value))[:limit]
    return {literal: replacements[literal] for literal in ordered}


def removed_literals(edits: list[dict[str, Any]], *, limit: int = 6) -> list[str]:
    """Return quoted literals removed, rather than merely moved, by *edits*."""
    return list(literal_replacements(edits, limit=limit))


def _removed_cache_decorated_symbols(edits: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(decorator, symbol) pairs whose cache decorator this edit removed.

    Flags only a decorator present above ``def NAME`` in *old* and absent above the
    same ``def NAME`` in *new* -- i.e. genuinely stripped, not merely relocated to a
    new helper (the helper keeps the decorator, so its own name is never flagged).
    """
    out: list[tuple[str, str]] = []
    for edit in edits:
        old = edit.get("old_string")
        new = edit.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            continue
        for match in _CACHE_DECORATED_DEF_RE.finditer(old):
            deco, name = match.group("deco"), match.group("name")
            still = re.search(
                r"@(?:\w+\.)*" + re.escape(deco) + r"\b[^\n]*\n\s*(?:async\s+)?def\s+" + re.escape(name) + r"\b",
                new,
            )
            if not still and (deco, name) not in out:
                out.append((deco, name))
    return out


def decorator_contract_impact(
    edits: list[dict[str, Any]],
    *,
    engine: _TextSearcher | None,
    touched_paths: list[str],
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Sites in untouched files that call a decorator-provided method this edit removed.

    e.g. removing ``@lru_cache`` from ``get_resolver`` breaks ``get_resolver.cache_clear()``
    in another file -- a semantic dependency invisible to literal matching. Deterministic
    (decorator removal is textual; the method names are a fixed stdlib vocabulary) and
    low-noise (fires only when the decorator is removed AND its method is used elsewhere).
    """
    pairs = _removed_cache_decorated_symbols(edits)
    if not pairs or engine is None:
        return []
    touched = set(touched_paths)
    sites: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for deco, name in pairs:
        for attr in _DECORATOR_PROVIDED_ATTRS.get(deco, ()):
            access = f"{name}.{attr}"
            try:
                hits = engine.search_text(access, path=".", limit=20, ignore_case=False)
            except Exception:  # noqa: BLE001 -- evidence-only; never break the edit
                hits = []
            for hit in hits:
                path = getattr(hit, "file_path", None)
                line = getattr(hit, "line", None)
                text = getattr(hit, "text", "") or ""
                if not isinstance(path, str) or path in touched:
                    continue
                if access not in text:  # precise: the actual attribute access, not a name collision
                    continue
                key = (path, int(line) if isinstance(line, int) else -1)
                if key in seen:
                    continue
                seen.add(key)
                sites.append(
                    {
                        "path": f"{path}:L{line}" if isinstance(line, int) else path,
                        "old": f"@{deco} on {name}()",
                        "new": f"{access} no longer exists",
                        # Snippet is raw file content -- mask secrets before it
                        # rides into agent-facing FIXME evidence (same masking
                        # applied to every other live tool output).
                        "snippet": redact_tool_output(text.strip())[:80],
                    }
                )
                if len(sites) >= limit:
                    return sites
    return sites


def _is_test_path(path: str) -> bool:
    name = Path(path).name.lower()
    parts = {part.lower() for part in Path(path).parts}
    return (
        bool(parts & {"test", "tests", "spec", "specs", "__tests__"})
        or name.startswith(("test_", "spec_"))
        or name.endswith(("_test.py", "_spec.rb"))
    )


def _is_structural_occurrence(line: str, literal: str) -> bool:
    """Text-fallback heuristic: True when the quoted literal is used as code, not prose.

    A real contract key sits next to a delimiter (``['db']``, ``.get('db'``,
    ``{'db':``, ``'db':``). The same token in prose sits between words
    (``the 'db' cache backend``) and is dropped as noise. ast-grep does this
    exactly; this approximates it when ast-grep is unavailable.
    """
    for quote in _QUOTES:
        token = f"{quote}{literal}{quote}"
        idx = line.find(token)
        while idx >= 0:
            before = line[idx - 1] if idx > 0 else ""
            after_index = idx + len(token)
            after = line[after_index] if after_index < len(line) else ""
            if before in _DELIMITERS or after in _DELIMITERS:
                return True
            idx = line.find(token, idx + 1)
    return False


def _astgrep_languages(touched_paths: list[str]) -> list[str]:
    languages: list[str] = []
    for path in touched_paths:
        language = _EXT_TO_ASTGREP_LANG.get(Path(path).suffix.lower())
        if language and language not in languages:
            languages.append(language)
    return languages


def _astgrep_patterns(literal: str) -> list[str]:
    # ast-grep string-literal patterns are quote-sensitive: 'x' matches only
    # single-quoted nodes, "x" only double-quoted. Try both and merge.
    return [f"{quote}{literal}{quote}" for quote in ("'", '"') if quote not in literal]


def _literal_queries(literal: str) -> list[str]:
    return [f"{quote}{literal}{quote}" for quote in _QUOTES]


def _line_lookup(repo_root: Path, cache: dict[str, list[str]], path: str, line: int) -> str:
    lines = cache.get(path)
    if lines is None:
        try:
            target = repo_root / path
            lines = (
                []
                if target.stat().st_size > _MAX_FILE_BYTES
                else target.read_text(encoding="utf-8", errors="replace").splitlines()
            )
        except OSError:
            lines = []
        cache[path] = lines
    return lines[line - 1].strip() if 1 <= line <= len(lines) else ""


def _astgrep_detect(
    literals: list[str],
    repo_root: Path,
    touched: set[str],
    *,
    languages: list[str],
    limit: int,
    pattern_builder: Callable[[str], list[str]] = _astgrep_patterns,
) -> dict[str, list[tuple[str, int, str]]] | None:
    """Structural detection via ast-grep. ``None`` means it could not run (caller falls back).

    ``pattern_builder`` maps a candidate to its ast-grep patterns -- quoted-literal
    nodes by default; a symbol detector swaps in the bare identifier.
    """
    if not languages:
        return None
    try:
        from lemoncrow.infra.code_intel.astgrep import AstGrepAdapter, AstGrepToolUnavailable
    except Exception:  # noqa: BLE001
        return None
    adapter = AstGrepAdapter(repo_root)
    line_cache: dict[str, list[str]] = {}
    out: dict[str, list[tuple[str, int, str]]] = {literal: [] for literal in literals}
    ran = False
    for literal in literals:
        seen: set[tuple[str, int]] = set()
        for language in languages:
            for pattern in pattern_builder(literal):
                try:
                    result = adapter.search(pattern=pattern, language=language, limit=limit)
                except AstGrepToolUnavailable:
                    return None  # binary missing -> let caller use the text fallback
                except Exception:  # noqa: BLE001
                    continue  # malformed pattern for this language, etc.
                ran = True
                for match in result.matches:
                    path = match.file_path
                    if not path or path in touched:
                        continue
                    line = match.line + 1  # ast-grep JSON ranges are 0-based; report 1-based
                    key = (path, line)
                    if key in seen:
                        continue
                    seen.add(key)
                    snippet = _line_lookup(repo_root, line_cache, path, line) or (match.snippet or "").strip()
                    out[literal].append((path, line, snippet))
    return out if ran else None


def _text_detect(
    literals: list[str],
    engine: _TextSearcher | None,
    touched: set[str],
    *,
    limit: int,
    query_builder: Callable[[str], list[str]] = _literal_queries,
    gate: Callable[[str, str], bool] = _is_structural_occurrence,
) -> dict[str, list[tuple[str, int, str]]]:
    """Language-agnostic fallback: engine text search + a used-as-code heuristic.

    ``query_builder`` yields the raw search strings for a candidate and ``gate`` keeps
    only hits that use it structurally (as code, not prose) -- quoted-literal by
    default; a symbol detector swaps in identifier variants.
    """
    out: dict[str, list[tuple[str, int, str]]] = {literal: [] for literal in literals}
    if engine is None:
        return out
    for literal in literals:
        seen: set[tuple[str, int]] = set()
        for query in query_builder(literal):
            try:
                hits = engine.search_text(query, limit=limit)
            except Exception:  # noqa: BLE001
                continue
            for hit in hits:
                path = getattr(hit, "file_path", None)
                line = getattr(hit, "line", None)
                text = getattr(hit, "text", "") or ""
                if not isinstance(path, str) or not isinstance(line, int) or path in touched:
                    continue
                if not gate(text, literal):
                    continue
                key = (path, line)
                if key in seen:
                    continue
                seen.add(key)
                out[literal].append((path, line, text.strip()))
    return out


def _combine_matches(
    astgrep_matches: list[tuple[str, int, str]] | None,
    text_matches: list[tuple[str, int, str]],
) -> list[tuple[str, int, str]]:
    """Best of both: ast-grep is authoritative for code files; text adds only the
    non-code files (config, templates, docs) ast-grep can't parse."""
    if astgrep_matches is None:
        return list(text_matches)  # ast-grep unavailable -> pure text recall
    seen = {(path, line) for path, line, _ in astgrep_matches}
    combined = list(astgrep_matches)
    for path, line, snippet in text_matches:
        if Path(path).suffix.lower() in _EXT_TO_ASTGREP_LANG:
            continue  # code file -> ast-grep already covered it precisely
        if (path, line) in seen:
            continue
        seen.add((path, line))
        combined.append((path, line, snippet))
    return combined


def contract_literal_impact(
    edits: list[dict[str, Any]],
    *,
    engine: _TextSearcher | None,
    repo_root: Path,
    touched_paths: list[str],
    max_matches_per_literal: int = 2,
    search_limit: int = 30,
) -> dict[str, Any] | None:
    """Return remaining occurrences of literals removed by *edits*, in untouched files.

    Detection prefers ast-grep (structural, precise); it degrades to *engine* text
    search when ast-grep can't run. Matches inside *touched_paths* are excluded --
    only parallel consumers the agent may have missed are evidence. Returns ``None``
    when no literal was removed or nothing remains elsewhere.
    """
    replacements = literal_replacements(edits)
    if not replacements:
        return None
    touched = set(touched_paths)
    literals = list(replacements)

    # Recall layer: language-agnostic text search (heuristic-filtered) finds
    # candidates anywhere, including non-code config/templates ast-grep can't parse.
    text_by_literal = _text_detect(literals, engine, touched, limit=search_limit)
    # Precision layer: ast-grep over every code language that actually appears
    # (edited files + text candidates). It is authoritative for code files --
    # matching string *nodes* drops the docstring/comment false positives that the
    # text heuristic only approximates away.
    candidate_paths = [match[0] for matches in text_by_literal.values() for match in matches]
    languages = _astgrep_languages(list(touched) + candidate_paths)
    astgrep_by_literal = _astgrep_detect(literals, repo_root, touched, languages=languages, limit=search_limit)

    sites: list[dict[str, Any]] = []
    for literal in literals:
        astgrep_found = astgrep_by_literal.get(literal) if astgrep_by_literal is not None else None
        found = _combine_matches(astgrep_found, text_by_literal.get(literal) or [])
        if not found:
            continue
        # Rarity gate: a literal found in many files is ambient vocabulary (common
        # bool synonyms, generic status words), not a contract -- skip it.
        if len({match[0] for match in found}) > _MAX_LITERAL_FILE_SPREAD:
            continue
        # Production consumers before tests, then stable by location.
        found.sort(key=lambda match: (_is_test_path(match[0]), match[0], match[1]))
        for path, line, snippet in found[:max_matches_per_literal]:
            entry: dict[str, Any] = {
                "path": f"{path}:L{line}",
                "old": literal,
                # Snippet is raw file content -- mask secrets before it rides
                # into agent-facing FIXME evidence (same masking applied to
                # every other live tool output).
                "snippet": redact_tool_output(snippet)[:80],
            }
            if replacements[literal] is not None:
                entry["new"] = replacements[literal]
            sites.append(entry)
    if not sites:
        return None
    return {
        "reason": ("These sites still use the old form you just changed -- update each or say why not."),
        "sites": sites,
    }


# Column-0 (module-level) definitions whose removal breaks importers in other files:
# a function, a class, or an UPPER_SNAKE constant. Indented defs/classes (methods,
# nested helpers) are excluded -- their references are attribute-qualified and far
# noisier than a bare module name.
_MODULE_SYMBOL_DEF_RE = re.compile(
    r"^(?:async\s+)?def\s+(?P<fn>\w+)"
    r"|^class\s+(?P<cls>\w+)"
    r"|^(?P<const>[A-Z_][A-Z0-9_]{2,})\s*(?::[^=\n]+)?=(?!=)",
    re.MULTILINE,
)
# Shorter module names (<4 chars) are too common to reference-match without noise;
# the file-spread gate already drops ambient names -- this just skips their lookups.
_MIN_SYMBOL_LEN = 4


def _defined_module_symbols(text: str) -> set[str]:
    out: set[str] = set()
    for match in _MODULE_SYMBOL_DEF_RE.finditer(text):
        name = match.group("fn") or match.group("cls") or match.group("const")
        if name:
            out.add(name)
    return out


def _removed_module_symbols(edits: list[dict[str, Any]]) -> list[str]:
    """Names of module-level defs/classes/constants defined in *old* but not *new*.

    A removed or renamed module symbol breaks every ``import`` / reference in other
    files, with no surviving definition for the call graph to resolve. Only genuine
    removals are returned: a name still defined in *new* (unchanged, merely shifted)
    is skipped -- the same discipline ``literal_replacements`` applies to strings.
    """
    removed: list[str] = []
    seen: set[str] = set()
    for edit in edits:
        old = edit.get("old_string")
        new = edit.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            continue
        for name in _defined_module_symbols(old) - _defined_module_symbols(new):
            if len(name) >= _MIN_SYMBOL_LEN and name not in seen:
                seen.add(name)
                removed.append(name)
    return removed


def _symbol_tokens(name: str) -> list[str]:
    # A bare identifier: the ast-grep pattern and the text query are the name itself.
    return [name]


def _is_identifier_occurrence(line: str, name: str) -> bool:
    """Text-fallback gate: *name* appears as a bare identifier reference -- not a
    substring of a longer name, an attribute of an unrelated object, or in a comment."""
    if line.lstrip().startswith("#"):
        return False
    return re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", line) is not None


def symbol_contract_impact(
    edits: list[dict[str, Any]],
    *,
    engine: _TextSearcher | None,
    repo_root: Path,
    touched_paths: list[str],
    max_matches_per_symbol: int = 2,
    search_limit: int = 30,
) -> list[dict[str, Any]]:
    """Sites in untouched files still referencing a module-level symbol this edit
    removed or renamed (a def / class / constant).

    The symbol counterpart of ``contract_literal_impact``: a deleted or renamed name
    breaks importers with no surviving definition for the call graph to resolve, so
    detection is structural -- ast-grep matches the identifier as an AST node (not a
    string or comment), with the engine's indexed text search as the language-agnostic
    fallback. Same discipline: exclude touched files, drop ambient names by file
    spread, fail-open. Returns a flat site list (empty when nothing remains elsewhere).
    """
    names = _removed_module_symbols(edits)
    if not names:
        return []
    touched = set(touched_paths)
    text_by_name = _text_detect(
        names,
        engine,
        touched,
        limit=search_limit,
        query_builder=_symbol_tokens,
        gate=_is_identifier_occurrence,
    )
    candidate_paths = [match[0] for matches in text_by_name.values() for match in matches]
    languages = _astgrep_languages(list(touched) + candidate_paths)
    astgrep_by_name = _astgrep_detect(
        names, repo_root, touched, languages=languages, limit=search_limit, pattern_builder=_symbol_tokens
    )
    sites: list[dict[str, Any]] = []
    for name in names:
        astgrep_found = astgrep_by_name.get(name) if astgrep_by_name is not None else None
        found = _combine_matches(astgrep_found, text_by_name.get(name) or [])
        if not found:
            continue
        # Rarity gate: a name referenced across many files is ambient, not a contract.
        if len({match[0] for match in found}) > _MAX_LITERAL_FILE_SPREAD:
            continue
        found.sort(key=lambda match: (_is_test_path(match[0]), match[0], match[1]))
        for path, line, snippet in found[:max_matches_per_symbol]:
            sites.append(
                {
                    "path": f"{path}:L{line}",
                    "old": name,
                    "new": f"{name} no longer defined here",
                    "snippet": redact_tool_output(snippet)[:80],
                }
            )
    return sites


def _split_top_level(param_str: str) -> list[str]:
    """Split a parameter list on top-level commas (commas inside (), [], {} stay put)."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in param_str:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def _has_top_level_default(param: str) -> bool:
    # A default is a top-level '=' -- not one nested in an annotation like
    # ``Callable[[int], int]`` (no '=') or a call default ``field(default=1)``.
    depth = 0
    for char in param:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "=" and depth == 0:
            return True
    return False


def _required_param_names(param_str: str) -> set[str]:
    """Names of parameters a caller MUST pass: positional-or-keyword / keyword-only
    with no default. ``self``/``cls``, ``*args``/``**kwargs`` and the ``*``/``/``
    markers are excluded -- they never impose a new required argument on callers."""
    required: set[str] = set()
    for raw in _split_top_level(param_str):
        param = raw.strip()
        if not param or param.startswith("*") or param == "/":
            continue
        name = re.split(r"[:=]", param, maxsplit=1)[0].strip()
        if name in ("self", "cls") or not name.isidentifier():
            continue
        if not _has_top_level_default(param):
            required.add(name)
    return required


_DEF_SIGNATURE_RE = re.compile(r"(?:^|\n)[ \t]*(?:async[ \t]+)?def[ \t]+(\w+)[ \t]*\(")


def _def_signatures(text: str) -> dict[str, str]:
    """Map each ``def NAME`` to its raw parameter string, balanced across newlines.

    A def whose parameter list is not closed within *text* (a hunk that cuts the
    signature mid-way) is skipped -- never guessed at."""
    out: dict[str, str] = {}
    for match in _DEF_SIGNATURE_RE.finditer(text):
        name = match.group(1)
        idx = match.end()  # just past the '('
        depth = 1
        while idx < len(text) and depth > 0:
            char = text[idx]
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            idx += 1
        if depth == 0:
            out[name] = text[match.end() : idx - 1]
    return out


def _signature_change_params(edits: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map each def present in BOTH old and new to the parameters that BECAME required
    -- newly added without a default, or an existing param that lost its default.
    Either breaks callers that don't pass it. Renamed/removed defs are the job of
    ``symbol_contract_impact``; a benign change (a new *optional* param) yields nothing."""
    out: dict[str, list[str]] = {}
    for edit in edits:
        old = edit.get("old_string")
        new = edit.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            continue
        old_sigs = _def_signatures(old)
        for name, new_params in _def_signatures(new).items():
            if name not in old_sigs or len(name) < _MIN_SYMBOL_LEN:
                continue
            newly_required = _required_param_names(new_params) - _required_param_names(old_sigs[name])
            if newly_required:
                out.setdefault(name, []).extend(sorted(newly_required))
    return out


def _call_patterns(name: str) -> list[str]:
    return [f"{name}($$$)"]  # ast-grep: a call whose callee is the bare identifier


def _is_call_occurrence(line: str, name: str) -> bool:
    """Text-fallback gate: *name* is invoked as a bare call (``name(``) -- not an
    attribute call, a substring, or a comment."""
    if line.lstrip().startswith("#"):
        return False
    return re.search(rf"(?<![\w.]){re.escape(name)}[ \t]*\(", line) is not None


def signature_change_impact(
    edits: list[dict[str, Any]],
    *,
    engine: _TextSearcher | None,
    repo_root: Path,
    touched_paths: list[str],
    max_matches_per_symbol: int = 2,
    search_limit: int = 30,
) -> list[dict[str, Any]]:
    """Call sites in untouched files of a function whose signature gained a required
    parameter -- callers that don't pass it now break.

    This is the one contract change the call graph CAN see (the symbol survives the
    edit, so it stays resolvable); detection stays structural for consistency and
    freshness-independence -- ast-grep matches the call expression, engine text search
    backs it up. Only *required* additions fire (a new optional param is non-breaking),
    so the common signature tweak stays silent. Fail-open; touched files excluded."""
    changed = _signature_change_params(edits)
    if not changed:
        return []
    names = list(changed)
    touched = set(touched_paths)
    text_by_name = _text_detect(
        names, engine, touched, limit=search_limit, query_builder=_symbol_tokens, gate=_is_call_occurrence
    )
    candidate_paths = [match[0] for matches in text_by_name.values() for match in matches]
    languages = _astgrep_languages(list(touched) + candidate_paths)
    astgrep_by_name = _astgrep_detect(
        names, repo_root, touched, languages=languages, limit=search_limit, pattern_builder=_call_patterns
    )
    sites: list[dict[str, Any]] = []
    for name in names:
        astgrep_found = astgrep_by_name.get(name) if astgrep_by_name is not None else None
        found = _combine_matches(astgrep_found, text_by_name.get(name) or [])
        if not found:
            continue
        # Rarity gate: a name called across many files is ambient, not a contract.
        if len({match[0] for match in found}) > _MAX_LITERAL_FILE_SPREAD:
            continue
        found.sort(key=lambda match: (_is_test_path(match[0]), match[0], match[1]))
        required = ", ".join(changed[name])
        for path, line, snippet in found[:max_matches_per_symbol]:
            sites.append(
                {
                    "path": f"{path}:L{line}",
                    "old": f"{name}(...)",
                    "new": f"now requires: {required}",
                    "snippet": redact_tool_output(snippet)[:80],
                }
            )
    return sites


__all__ = [
    "contract_literal_impact",
    "decorator_contract_impact",
    "literal_replacements",
    "removed_literals",
    "signature_change_impact",
    "symbol_contract_impact",
]

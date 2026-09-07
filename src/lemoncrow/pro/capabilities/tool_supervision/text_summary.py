"""Type-aware extractive summarization for the `read` tool's `:summary` suffix.

No LLM required: produces a compact, high-signal gist by sniffing the file's
type (extension first, content second) and dispatching to a format-specific
extractor -- markdown/rst get a heading-tree-with-gists, JSON/YAML/TOML/INI
get a shape report, CSV gets a header + sample rows, logs/spill files reuse
the existing bash-output anomaly/dedup machinery, code files get a docstring
+ symbol inventory, and everything else falls through to a small Luhn-style
extractive prose summarizer. Every path is bounded to `target_chars` and uses
the canonical `[… N …]` inline-marker family for anything it must cut.

Stdlib only, plus `pyyaml` (already a hard LemonCrow dependency) for YAML.
`:summary` is a GIST, deliberately distinct from `:outline` (the deterministic
structural projection) -- this module never calls the outline machinery.
"""

from __future__ import annotations

import configparser
import csv
import io
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from lemoncrow.pro.capabilities.tool_supervision.bash_exec import (
    _ANOMALY_LINE_RE,
    _dedupe_repeated_lines,
    _extract_anomaly_windows,
)
from lemoncrow.pro.capabilities.tool_supervision.compact_output import compress_tool_output

_MARKDOWN_EXT = frozenset({".md", ".markdown", ".mdx", ".rst"})
_JSON_EXT = frozenset({".json"})
_YAML_EXT = frozenset({".yaml", ".yml"})
_TOML_EXT = frozenset({".toml"})
_INI_EXT = frozenset({".ini", ".cfg", ".conf"})
_CSV_EXT = frozenset({".csv", ".tsv"})
_LOG_EXT = frozenset({".log"})
_CODE_EXT = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".mjs",
        ".cjs",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".hpp",
        ".rb",
        ".php",
        ".cs",
        ".swift",
        ".kt",
        ".scala",
        ".sh",
        ".sql",
    }
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
_STOPWORDS = frozenset(
    "a an the and or but if then else for nor so yet of to in on at by with from "
    "into over under again further once here there when where why how all any "
    "both each few more most other some such no not only own same than too very "
    "s t can will just don should now is are was were be been being have has had "
    "do does did this that these those it its as".split()
)


def heuristic_summary(text: str, *, path: str | Path | None = None, target_chars: int = 4096) -> str:
    """Type-aware extractive gist of *text*, bounded to *target_chars*.

    Never raises: a format-specific extractor that fails (invalid JSON, an
    unparsable YAML doc, ...) falls through to the always-safe prose path.
    """
    if not text.strip():
        return ""
    kind = _classify(text, path)
    body = ""
    try:
        if kind == "markdown":
            body = _summarize_markdown(text)
        elif kind == "json":
            body = _summarize_json(text)
        elif kind == "structured":
            body = _summarize_structured(text, path)
        elif kind == "csv":
            body = _summarize_csv(text, path)
        elif kind == "log":
            body = _summarize_log(text, target_chars)
        elif kind == "code":
            body = _summarize_code(text)
    except Exception:  # noqa: BLE001 -- a format-specific extractor must never break :summary
        body = ""
    if not body.strip():
        body = _summarize_prose(text, target_chars)
    return _bound(body, target_chars)


def llm_summary_tier(text: str, *, target_chars: int = 4096) -> tuple[str, str] | None:
    """Best-effort internal-LLM gist tier: ``(body, verb)`` on success, ``None``
    on ANY failure (backend disabled, provider error, empty output, ...) --
    never raises, so the caller falls back to :func:`heuristic_summary`.

    Shared by the `read` tool's ``:summary`` suffix and `web_fetch`'s
    ``summary=true`` so both use the identical model-tier ladder and verb
    grammar (``summarized:{model}``) instead of two divergent copies.
    """
    try:
        from lemoncrow.infra.internal_llm import summarize as _internal_summarize

        llm_text = _internal_summarize(text, max_tokens=max(256, target_chars // 4)).strip()
    except Exception:  # noqa: BLE001 -- LLM failure must never break a summary
        return None
    if not llm_text:
        return None
    body = llm_text[:target_chars]
    verb = f"summarized:{_sanitize_model_label(_internal_llm_model_label())}"
    return body, verb


def _sanitize_model_label(label: str) -> str:
    """Strip spaces/brackets so a model name composes cleanly into a
    ``summarized:{model}`` verb (``spill_notice``'s grammar has no room for either)."""
    return re.sub(r"[\s\[\]]+", "-", label.strip()) or "llm"


def _internal_llm_model_label() -> str:
    """Best-effort model name for the ``summarized:{model}`` verb.

    Reads the SAME env vars each internal-LLM backend resolves its own model
    from (``LEMONCROW_OPENAI_MODEL`` / ``LEMONCROW_OLLAMA_MODEL``); falls back to
    the bare backend name (e.g. ``ollama``) when no model override is set.
    """
    backend = os.environ.get("LEMONCROW_LLM_BACKEND", "none").lower().strip()
    if backend in ("openai", "openai_compatible"):
        model = os.environ.get("LEMONCROW_OPENAI_MODEL", "").strip()
    elif backend == "ollama":
        model = os.environ.get("LEMONCROW_OLLAMA_MODEL", "").strip()
    else:
        model = ""
    if model:
        return model
    return backend if backend not in ("", "none") else "llm"


def _is_spill_path(path: str | Path | None) -> bool:
    if not path:
        return False
    from lemoncrow.pro.capabilities.tool_supervision.tool_output_spill import _spill_dir

    try:
        return Path(path).resolve().parent == _spill_dir().resolve()
    except (OSError, ValueError):
        return False


def _classify(text: str, path: str | Path | None) -> str:
    # Spill files carry provenance in their directory, not a meaningful
    # extension (`tool_output-bash-....txt`) -- treat as log/tool-output
    # unconditionally, ahead of any extension sniff.
    if _is_spill_path(path):
        return "log"
    ext = Path(path).suffix.lower() if path else ""
    if ext in _MARKDOWN_EXT:
        return "markdown"
    if ext in _JSON_EXT:
        return "json"
    if ext in _YAML_EXT or ext in _TOML_EXT or ext in _INI_EXT:
        return "structured"
    if ext in _CSV_EXT:
        return "csv"
    if ext in _LOG_EXT:
        return "log"
    if ext in _CODE_EXT:
        return "code"
    stripped = text.lstrip()
    if stripped[:1] in "{[":
        return "json"
    if re.search(r"(?m)^#{1,6}\s+\S", text):
        return "markdown"
    return "prose"


def _split_sentences(text: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(collapsed) if p.strip()]


def _bound(text: str, target_chars: int) -> str:
    if len(text) <= target_chars:
        return text
    head = int(target_chars * 0.7)
    return compress_tool_output(text, threshold_chars=target_chars, head_chars=head, tail_chars=target_chars - head)


# --------------------------------------------------------------------------- #
# Markdown / rst -- heading tree + first sentence(s) of each section          #
# --------------------------------------------------------------------------- #

_HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*)$")


def _summarize_markdown(text: str) -> str:
    out: list[str] = []
    section: list[str] = []

    def flush() -> None:
        if not section:
            return
        gist = " ".join(_split_sentences(" ".join(section))[:2])
        if gist:
            out.append(f"  {gist}")
        section.clear()

    for line in text.splitlines():
        heading = _HEADING_RE.match(line.rstrip())
        if heading:
            flush()
            out.append(f"{heading.group(1)} {heading.group(2)}")
        elif line.strip():
            section.append(line.strip())
    flush()
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# JSON -- shape report (keys/types, array lengths, depth) + sample entries    #
# --------------------------------------------------------------------------- #


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return type(value).__name__


def _max_depth(value: Any, depth: int = 0) -> int:
    if isinstance(value, dict) and value:
        return max(_max_depth(v, depth + 1) for v in value.values())
    if isinstance(value, list) and value:
        return max(_max_depth(v, depth + 1) for v in value)
    return depth


def _json_shape(data: Any) -> str:
    if isinstance(data, dict):
        keys = list(data.keys())
        shown = ", ".join(f"{k}: {_type_name(data[k])}" for k in keys[:12])
        more = f", …(+{len(keys) - 12} more)" if len(keys) > 12 else ""
        return f"object, {len(keys)} top-level key(s) [{shown}{more}], max depth {_max_depth(data)}"
    if isinstance(data, list):
        elem = _type_name(data[0]) if data else "empty"
        return f"array of {len(data)} item(s) (element type: {elem}), max depth {_max_depth(data)}"
    return f"scalar ({_type_name(data)})"


def _json_samples(data: Any, limit: int = 3) -> list[str]:
    items: list[Any]
    if isinstance(data, list):
        items = data[:limit]
    elif isinstance(data, dict):
        items = list(data.values())[:limit]
    else:
        items = [data]
    rendered: list[str] = []
    for item in items:
        text = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        rendered.append(text[:200] + ("…" if len(text) > 200 else ""))
    return rendered


def _summarize_json(text: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ""
    lines = [f"JSON — {_json_shape(data)}"]
    samples = _json_samples(data)
    if samples:
        lines.append("samples:")
        lines.extend(f"  {s}" for s in samples)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# YAML / TOML / INI -- top-level keys/sections + representative values       #
# --------------------------------------------------------------------------- #


def _short_repr(value: Any, limit: int = 80) -> str:
    if isinstance(value, dict):
        return f"{{…{len(value)} keys…}}"
    if isinstance(value, list):
        return f"[…{len(value)} items…]"
    text = str(value)
    return text[:limit] + ("…" if len(text) > limit else "")


def _mapping_shape(data: dict[Any, Any]) -> str:
    items = list(data.items())
    lines = [f"{len(items)} top-level key(s):"]
    for key, value in items[:20]:
        lines.append(f"  {key}: {_short_repr(value)}")
    if len(items) > 20:
        lines.append(f"  … (+{len(items) - 20} more)")
    return "\n".join(lines)


def _summarize_structured(text: str, path: str | Path | None) -> str:
    ext = Path(path).suffix.lower() if path else ""
    if ext in _TOML_EXT:
        import tomllib

        data: Any = tomllib.loads(text)
        return _mapping_shape(data) if isinstance(data, dict) else _json_shape(data)
    if ext in _INI_EXT:
        parser = configparser.ConfigParser()
        parser.read_string(text)
        lines = [f"[{s}]: {', '.join(parser[s].keys())}" for s in parser.sections()]
        return "\n".join(lines)
    # YAML (default for this tier -- also the fallback for an unrecognised
    # structured extension reaching here).
    import yaml

    data = yaml.safe_load(text)
    if isinstance(data, dict):
        return _mapping_shape(data)
    return _json_shape(data) if data is not None else ""


# --------------------------------------------------------------------------- #
# CSV / TSV -- header, shape, first few data rows                            #
# --------------------------------------------------------------------------- #


def _summarize_csv(text: str, path: str | Path | None) -> str:
    ext = Path(path).suffix.lower() if path else ""
    delimiter = "\t" if ext == ".tsv" else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    if not rows:
        return ""
    header, data_rows = rows[0], rows[1:]
    lines = [
        f"CSV — {len(header)} column(s), {len(data_rows)} data row(s)",
        f"header: {', '.join(header)}",
    ]
    lines.extend(f"  {', '.join(row)}" for row in data_rows[:3])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Logs / tool output / spill files -- reuse the bash-output anomaly machinery #
# --------------------------------------------------------------------------- #

_TRACEBACK_START_RE = re.compile(r"^Traceback \(most recent call last\)", re.IGNORECASE | re.MULTILINE)


def _summarize_log(text: str, target_chars: int) -> str:
    total_lines = text.count("\n") + 1
    deduped, _saved = _dedupe_repeated_lines(text)
    tracebacks = _TRACEBACK_START_RE.findall(deduped)
    anomaly_lines = [ln for ln in deduped.splitlines() if _ANOMALY_LINE_RE.search(ln)]
    shape = f"{total_lines:,} lines"
    if tracebacks:
        shape += f", {len(tracebacks)} traceback(s)"
    elif anomaly_lines:
        shape += f", {len(anomaly_lines)} anomaly line(s)"
    if anomaly_lines:
        shape += f"; last: {anomaly_lines[-1].strip()[:200]}"
    windowed = _extract_anomaly_windows(deduped, target_chars, context=2)
    if windowed is None:
        head = int(target_chars * 0.6)
        windowed = compress_tool_output(
            deduped, threshold_chars=target_chars, head_chars=head, tail_chars=target_chars - head
        )
    return f"{shape}\n\n{windowed}"


# --------------------------------------------------------------------------- #
# Code files -- module docstring + symbol inventory + top first-line docs    #
# --------------------------------------------------------------------------- #

_MODULE_DOCSTRING_RE = re.compile(r'^\s*(?:#[^\n]*\n)*\s*(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')', re.DOTALL)
_CLASS_RE = re.compile(r"^[ \t]*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_DEF_RE = re.compile(r"^[ \t]*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_JS_FUNC_RE = re.compile(r"^[ \t]*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)", re.MULTILINE)
_DEF_DOC_RE = re.compile(
    r'^[ \t]*(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)[^\n]*:[ \t]*\n[ \t]*(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')',
    re.MULTILINE | re.DOTALL,
)
# Escalation marker for a bounded symbol inventory: names the recovery path
# (`:outline`) instead of a bare ellipsis, matching the `[… N chars omitted …]`
# marker family used elsewhere in this module.
_SYMBOL_ESCALATION_MARKER = "[… {n} more symbols; :outline for full structure …]"


def _summarize_code(text: str) -> str:
    out = [f"{text.count(chr(10)) + 1:,} lines"]
    mod_doc = _MODULE_DOCSTRING_RE.match(text)
    if mod_doc:
        doc = (mod_doc.group(1) or mod_doc.group(2) or "").strip()
        first = _split_sentences(doc)[:1]
        if first:
            out.append(first[0])

    classes = _CLASS_RE.findall(text)
    functions = _DEF_RE.findall(text) or _JS_FUNC_RE.findall(text)

    def inventory_clause(label: str, names: list[str], shown_n: int) -> str:
        shown = names[:shown_n]
        suffix = f" {_SYMBOL_ESCALATION_MARKER.format(n=len(names) - shown_n)}" if len(names) > shown_n else ""
        noun = label if len(names) == 1 else f"{label}s"
        return f"{len(names)} {noun}: {', '.join(shown)}{suffix}"

    clauses = []
    if classes:
        clauses.append(inventory_clause("class", classes, 8))
    if functions:
        clauses.append(inventory_clause("function", functions, 12))
    if clauses:
        out.append("defines " + "; ".join(clauses))

    for _kind, name, doc1, doc2 in _DEF_DOC_RE.findall(text)[:5]:
        doc = (doc1 or doc2).strip()
        first = _split_sentences(doc)[:1]
        if first:
            out.append(f"  {name}: {first[0]}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Generic prose fallback -- LexRank extractive scoring (in-house)             #
#
# LexRank = sentence centrality on an ISF-weighted cosine-similarity graph
# (power iteration with damping, i.e. PageRank over sentences). Picks the
# sentences most representative of the whole text rather than Luhn's
# frequency-sum, which over-rewards long keyword-dense sentences. Implemented
# in-house (like BM25) so :summary stays offline and dependency-free -- sumy
# would pull nltk + a runtime punkt download.
# --------------------------------------------------------------------------- #

# O(n^2) sentence-pair similarity is the algorithm; capped at 200 sentences a
# gist never needs more, keeping the worst case ~20k pairs (sub-millisecond).
_LEXRANK_MAX_SENTENCES = 200
_LEXRANK_DAMPING = 0.85
_LEXRANK_ITERATIONS = 20
_LEXRANK_EPSILON = 1e-4


def _sentence_terms(sentence: str) -> Counter[str]:
    return Counter(w for w in _WORD_RE.findall(sentence.lower()) if len(w) > 2 and w not in _STOPWORDS)


def _lexrank_scores(term_counts: list[Counter[str]]) -> list[float]:
    n = len(term_counts)
    # Inverse sentence frequency: shared-but-distinctive terms drive similarity,
    # mirroring TF-IDF cosine without materializing a vocabulary matrix.
    df: Counter[str] = Counter()
    for counts in term_counts:
        df.update(counts.keys())
    isf = {word: math.log(1.0 + n / df[word]) for word in df}
    norms = [math.sqrt(sum((count * isf[word]) ** 2 for word, count in counts.items())) for counts in term_counts]

    def cosine(i: int, j: int) -> float:
        a, b = term_counts[i], term_counts[j]
        if not a or not b:
            return 0.0
        shared = a.keys() & b.keys()
        if not shared:
            return 0.0
        num = sum(a[w] * b[w] * isf[w] * isf[w] for w in shared)
        return num / (norms[i] * norms[j]) if norms[i] and norms[j] else 0.0

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine(i, j)
            sim[i][j] = sim[j][i] = s
    row_sums = [sum(row) or 1.0 for row in sim]
    scores = [1.0 / n] * n
    for _ in range(_LEXRANK_ITERATIONS):
        nxt = [
            (1.0 - _LEXRANK_DAMPING) / n
            + _LEXRANK_DAMPING * sum(sim[j][i] / row_sums[j] * scores[j] for j in range(n) if sim[j][i])
            for i in range(n)
        ]
        delta = sum(abs(a - b) for a, b in zip(nxt, scores, strict=True))
        scores = nxt
        if delta < _LEXRANK_EPSILON:
            break
    return scores


def _summarize_prose(text: str, target_chars: int) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return text[:target_chars]
    if len(sentences) <= 6:
        return " ".join(sentences)
    sentences = sentences[:_LEXRANK_MAX_SENTENCES]

    scores = _lexrank_scores([_sentence_terms(s) for s in sentences])
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    keep_n = max(3, min(len(sentences), target_chars // 120))
    # Duplicated sentences are maximally central by construction -- dedupe on
    # the normalized form so a repeated line cannot fill the whole gist.
    keep: set[int] = set()
    seen: set[str] = set()

    def _try_keep(index: int) -> None:
        normalized = sentences[index].strip().lower()
        if normalized in seen:
            return
        seen.add(normalized)
        keep.add(index)

    _try_keep(0)
    if len(sentences) > 1:
        _try_keep(1)
    for index in ranked:
        if len(keep) >= keep_n:
            break
        _try_keep(index)
    return " ".join(sentences[i] for i in sorted(keep))


__all__ = ["heuristic_summary", "llm_summary_tier"]

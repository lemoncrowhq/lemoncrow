"""Persistent symbol index and token-budgeted retrieval for local code."""

from __future__ import annotations

import ast
import atexit
import concurrent.futures
import contextlib
import fnmatch
import hashlib
import itertools
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, cast, overload

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

from lemoncrow.core.foundation.paths import default_store_root
from lemoncrow.core.foundation.weakref_token import WeakRefToken
from lemoncrow.core.service.telemetry import emit_product_local
from lemoncrow.infra.code_intel.astgrep import (
    AstGrepAdapter,
    AstGrepToolUnavailable,
    PatternMatch,
    PatternRewriteResult,
    PatternSearchResult,
)
from lemoncrow.infra.tree_sitter.tags import Tag, detect_language, extract_tags
from lemoncrow.pro.capabilities.code_context.ann_symbol_index import (
    SymbolAnnIndex,
    ensure_symbol_vector_schema,
)
from lemoncrow.pro.capabilities.code_context.budget import (
    PROTECTED_TOP_RANK,
    BudgetPacker,
)
from lemoncrow.pro.capabilities.code_context.cache import RetrievalCache
from lemoncrow.pro.capabilities.code_context.call_graph import (
    CallGraphDirection,
    CallGraphEdge,
    CallGraphNode,
    CallGraphTraversalResult,
    build_call_graph_payload,
    traverse_call_graph,
)
from lemoncrow.pro.capabilities.code_context.call_graph_centrality import compute_call_graph_centrality
from lemoncrow.pro.capabilities.code_context.diversity import demote_doc_overflow
from lemoncrow.pro.capabilities.code_context.embedding import (
    SearchMode,
    SemanticSearchRanker,
    resolve_search_mode,
    semantic_candidate_limit,
)
from lemoncrow.pro.capabilities.code_context.generated_files import is_generated_path
from lemoncrow.pro.capabilities.code_context.intel_store import ProviderHealth, SymbolIntelStore
from lemoncrow.pro.capabilities.code_context.models import (
    ContextPack,
    CrossLangReference,
    IndexedFileRecord,
    IndexStats,
    RouteRecord,
    SymbolRecord,
    TextMatch,
    UsageReference,
)
from lemoncrow.pro.capabilities.code_context.output_policy import (
    cap_source_by_tokens,
    hard_cap_chars,
    resolve_output_policy,
)
from lemoncrow.pro.capabilities.code_context.ranking import (
    SymbolScoringContext,
    idf_weight,
    score_symbol_row,
)
from lemoncrow.pro.capabilities.code_context.rerank import SearchReranker
from lemoncrow.pro.capabilities.repo_map import build_repo_map
from lemoncrow.pro.capabilities.repo_map.budget import count_tokens, estimate_tokens
from lemoncrow.pro.capabilities.repo_map.graph import iter_source_files, should_skip_relative_path
from lemoncrow.pro.code_intel.cross_lang import CrossLangEdge, CrossLangEdgeStore

# watchdog for OS-native file watching (inotify/FSEvents/ReadDirectoryChangesW).
# Falls back gracefully when the package is not installed.
try:
    from watchdog.events import (
        DirCreatedEvent,
        DirDeletedEvent,
        DirMovedEvent,
        FileClosedEvent,
        FileCreatedEvent,
        FileDeletedEvent,
        FileModifiedEvent,
        FileMovedEvent,
        FileSystemEvent,
        FileSystemEventHandler,
    )
    from watchdog.observers import Observer

    # Write-class events only. Without this filter watchdog watches
    # IN_ALL_EVENTS -- including IN_OPEN / IN_ACCESS / IN_CLOSE_NOWRITE -- so
    # every source file the engine itself READS while rendering results fires
    # kernel events back into the watcher: each one is realpath'd, built into
    # an event object, matched against the full .gitignore pathspec (fnmatch
    # storm), and enough survive to trigger spurious autosync reindex scans.
    # Confirmed via cProfile on a 24-query benchmark pass against a static
    # repo: 50,977 queue_events calls, 684K fnmatch calls, 42K lstat calls and
    # 205 watcher notifications with ZERO actual file writes in the tree. The
    # filter narrows the inotify mask at the KERNEL level (watchdog>=4
    # get_event_mask_from_filter), so read-class events are never generated.
    _WATCHER_EVENT_FILTER = [
        FileCreatedEvent,
        FileDeletedEvent,
        FileModifiedEvent,
        FileMovedEvent,
        FileClosedEvent,  # IN_CLOSE_WRITE: catches writes flushed on close
        DirCreatedEvent,  # keeps recursive watch bookkeeping for new dirs
        DirDeletedEvent,
        DirMovedEvent,
    ]
except ImportError:
    Observer = None  # type: ignore[assignment,misc]
    FileSystemEventHandler = object  # type: ignore[assignment,misc]
    _WATCHER_EVENT_FILTER = []

if TYPE_CHECKING:
    from lemoncrow.pro.capabilities.code_context.search_verdict import ChannelHealth
    from lemoncrow.pro.code_intel.git_history.adapter import DeletedHistorySearchAdapter


def _query_is_natural_language(query: str) -> bool:
    """Return True when the query looks like natural language -> activate semantic.

    Symbol-name queries (short, <= 2 meaningful tokens) skip semantic so lexical
    can dominate.  Multi-word sentence queries activate it so the embedder can
    contribute.  Override via LEMONCROW_SEMANTIC_MODE=always|off|auto (default auto).
    """
    _mode = os.environ.get("LEMONCROW_SEMANTIC_MODE", "auto").strip().lower()
    if _mode == "always":
        return True
    if _mode == "off":
        return False
    # auto: word count heuristic
    stripped = query.strip()
    # Strip common leading keywords ("def ", "class ", "async def ") that appear
    # in definition-gold queries but don't make them natural language.
    for _kw in ("async def ", "def ", "class "):
        if stripped.lower().startswith(_kw):
            stripped = stripped[len(_kw) :]
            break
    words = stripped.split()
    # >= 4 words after stripping the keyword -> treat as natural-language query
    return len(words) >= 4


_MAX_FILE_BYTES = 1_000_000
# Free-tier repo-size cap for the context engine (context_engine is a Pro
# feature at scale -- see licensing/features.py). Generous on purpose: this is
# well past a typical solo/small-team repo, so Free stays "genuinely useful";
# it's a real ceiling only for large monorepos, which is exactly what Pro's
# uncapped large-repo indexing is for.
_FREE_TIER_MAX_FILES = 2_500
logger = logging.getLogger(__name__)


def _is_implicit_tmp_index_blocked(repo_root: Path) -> bool:
    """True when *repo_root* is under /tmp and implicit first-time auto-indexing
    of it has not been explicitly allowed.

    Explicit indexing (the ``lc code index --reindex`` CLI path, used by
    benchmark/eval scripts) is a different call path entirely and is never
    affected by this -- this only gates the *implicit* build a plain tool call
    (e.g. ``code_search``) would otherwise trigger on an unindexed /tmp dir.
    """
    if os.environ.get("LEMONCROW_ALLOW_TMP_AUTOINDEX", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    try:
        resolved = repo_root.resolve()
    except OSError:
        resolved = repo_root
    return resolved == Path("/tmp") or Path("/tmp") in resolved.parents


_DB_LOCKS_GUARD = threading.Lock()
_DB_LOCKS: dict[str, threading.RLock] = {}


def _shared_db_lock(db_path: Path) -> threading.RLock:
    key = str(db_path.resolve())
    with _DB_LOCKS_GUARD:
        lock = _DB_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _DB_LOCKS[key] = lock
        return lock


class _ReusedConnection:
    """Wraps a shared sqlite connection so per-call ``with self._connect()`` and
    ``contextlib.closing(...)`` blocks reuse it instead of opening a new one.
    ``close()`` and ``__exit__`` are no-ops -- the owning ``_reuse_connection``
    scope commits and closes the real connection once. Confined to a single
    thread via the engine's thread-local, matching sqlite's per-thread rule."""

    __slots__ = ("_conn",)
    _conn: sqlite3.Connection

    def __init__(self, conn: sqlite3.Connection) -> None:
        object.__setattr__(self, "_conn", conn)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_conn"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Special-case the slot itself: default copy/pickle reconstruction for
        # slotted objects (copy.copy(), __setstate__) does `setattr(y, "_conn",
        # conn)` on a fresh instance where the slot isn't set yet -- forwarding
        # that to the (not-yet-set) wrapped connection raises AttributeError.
        if name == "_conn":
            object.__setattr__(self, name, value)
            return
        setattr(object.__getattribute__(self, "_conn"), name, value)

    def __enter__(self) -> sqlite3.Connection:
        conn: sqlite3.Connection = object.__getattribute__(self, "_conn")
        return conn

    def __exit__(self, *exc: object) -> Literal[False]:
        # Mirror sqlite3.Connection.__exit__: commit on success / rollback on error
        # so an inner ``with self._connect()`` write block releases its lock promptly
        # instead of holding it open for the whole reuse scope. The connection itself
        # stays open (that's the reuse); only close() is neutralized.
        #
        # Skip the commit/rollback when no transaction is pending (i.e. the inner
        # block was read-only).  ``conn.commit()`` on a connection with no pending
        # changes still acquires the write lock briefly in WAL mode, which can
        # block for the full busy_timeout (~30 s) when another process holds the
        # write lock (e.g. LemonCrow autosync on the same DB).  The outer
        # ``_reuse_connection`` scope handles the final commit/close.
        conn: sqlite3.Connection = object.__getattribute__(self, "_conn")
        if not conn.in_transaction:
            return False
        if exc[0] is None:
            conn.commit()
        else:
            conn.rollback()
        return False

    def close(self) -> None:
        return None


_FTS_TERM_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_PRECISE_SYMBOL_QUERY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
# Code-query leading keywords used by the AND-channel guard in _fts_and_query.
_CODE_LEADING_KW = frozenset({"def", "class", "import", "from", "return", "async", "yield"})
_SINCE_RELATIVE_RE = re.compile(r"^(?P<amount>\d+)(?P<unit>[dwmy])$")
_JS_IMPORT_RE = re.compile(
    r"(?:from\s+['\"]([^'\"]+)['\"]|import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)|require\(\s*['\"]([^'\"]+)['\"]\s*\))"
)
_RUST_MOD_RE = re.compile(r"^\s*(?:pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", re.M)
_GO_IMPORT_RE = re.compile(r"^\s*import\s+(?:\((.*?)\)|\"([^\"]+)\")", re.M | re.S)
_FASTAPI_DECORATOR_RE = re.compile(
    r"^\s*@(?P<router>[A-Za-z_][A-Za-z0-9_]*)\.(?P<verb>get|post|put|patch|delete|options|head|trace|websocket)\(\s*['\"](?P<route>[^'\"]+)['\"]"
)
_FASTAPI_API_ROUTE_RE = re.compile(
    r"^\s*@(?P<router>[A-Za-z_][A-Za-z0-9_]*)\.api_route\(\s*['\"](?P<route>[^'\"]+)['\"](?P<rest>.*)\)\s*$"
)
_FLASK_ROUTE_RE = re.compile(
    r"^\s*@(?P<router>[A-Za-z_][A-Za-z0-9_]*)\.route\(\s*['\"](?P<route>[^'\"]+)['\"](?P<rest>.*)\)\s*$"
)
_FLASK_ADD_URL_RULE_RE = re.compile(
    r"^\s*(?P<router>[A-Za-z_][A-Za-z0-9_]*)\.add_url_rule\(\s*['\"](?P<route>[^'\"]+)['\"](?P<rest>.*)\)\s*$"
)
_DJANGO_PATH_RE = re.compile(
    r"^\s*(?:re_)?path\(\s*['\"](?P<route>[^'\"]+)['\"]\s*,\s*(?P<handler>[A-Za-z_][A-Za-z0-9_\.]*)"
)
_DJANGO_URL_RE = re.compile(
    r"^\s*url\(\s*(?:r)?['\"](?P<route>[^'\"]+)['\"]\s*,\s*(?P<handler>[A-Za-z_][A-Za-z0-9_\.]*)"
)
_EXPRESS_ROUTE_RE = re.compile(
    r"(?P<router>app|router)\.(?P<verb>get|post|put|patch|delete|options|head|all|use)\(\s*[`'\"](?P<route>[^`'\"]+)[`'\"]\s*(?:,\s*(?P<handler>[A-Za-z_$][A-Za-z0-9_$.]*))?"
)
_EXPRESS_ROUTE_CHAIN_RE = re.compile(
    r"(?P<router>app|router)\.route\(\s*[`'\"](?P<route>[^`'\"]+)[`'\"]\s*\)(?P<chain>.+)$"
)
_EXPRESS_CHAIN_METHOD_RE = re.compile(
    r"\.(?P<verb>get|post|put|patch|delete|options|head|all|use)\(\s*(?P<handler>[A-Za-z_$][A-Za-z0-9_$.]*)?"
)
_METHOD_LITERAL_RE = re.compile(r"['\"](?P<method>GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD|TRACE|CONNECT)['\"]", re.I)
_LOCAL_PROVENANCE = "local"
_SEARCH_ESSENTIAL_KEYS = [
    "symbol_id",
    "symbol_name",
    "qualified_name",
    "file_path",
    "kind",
    "start_line",
    "signature",
    "provenance",
]
# Fields stripped from each item for repo-scope searches where they are
# always uniform or unnecessary ("origin" is always "internal"; per-item
# "provenance" is redundant with top-level; "symbol_id" is internal only).
_SEARCH_REPO_STRIP_ITEM_KEYS: frozenset[str] = frozenset({"origin", "provenance", "symbol_id"})
_SEARCH_OPTIONAL_KEYS = [
    "snippet",
    "doc_summary",
    "score",
    "repo_id",
    "content_hash",
    "parent_symbol",
    "start_byte",
    "end_byte",
]
_DELETED_SEARCH_ESSENTIAL_KEYS = [
    *_SEARCH_ESSENTIAL_KEYS,
    "deleted_at",
    "deleted_at_sha",
    "last_author",
]
_DELETED_SEARCH_OPTIONAL_KEYS = [
    "rename_target",
    "rename_note",
    "last_commit_msg",
    "matched_on",
]
_SEARCH_COMPACT_DEFAULT_KEYS = set([*_SEARCH_ESSENTIAL_KEYS, "score", "commit_sha"])
_LINEAGE_INDEX_VERSION = 2
# Bump when source selection or symbol/text extraction semantics change in a way
# an incremental mtime/hash check cannot see for unchanged files.
_CODE_INDEXER_SEMANTICS_VERSION = 2
_LINEAGE_DEFAULT_SCORE_PENALTY = 0.1

# --- File-watcher constants ---
# Minimal set of directories that are *never* source code, regardless of .gitignore.
# The watcher relies primarily on .gitignore (loaded via pathspec) for file filtering;
# this is just a fast-path guard for paths that git doesn't even track.
_WATCHER_HARD_SKIP_DIRS: frozenset[str] = frozenset({".git"})
# Source file extensions to watch, built from the canonical language registry.
# Deferred mid-module (as the prior __import__ did) to avoid an import cycle.
from lemoncrow.infra.code_intel.languages import LANGUAGES as _WATCHER_LANGUAGES  # noqa: E402

_WATCHER_SOURCE_EXTENSIONS: frozenset[str] = frozenset(ext for lang in _WATCHER_LANGUAGES for ext in lang.extensions)
_DELETED_SEARCH_COMPACT_DEFAULT_KEYS = set(
    [*_DELETED_SEARCH_ESSENTIAL_KEYS, "score", "matched_on", "rename_target", "rename_note"]
)
_FILES_ESSENTIAL_KEYS = ["file_path"]
_FILES_OPTIONAL_KEYS = ["top_symbols", "symbol_count", "language"]
_ROUTES_ESSENTIAL_KEYS = ["framework", "method", "route", "file_path", "line", "provenance"]
_ROUTES_OPTIONAL_KEYS = ["handler", "router", "language"]
_SYMBOL_ESSENTIAL_KEYS = ["symbol_name"]
_SYMBOL_OPTIONAL_KEYS = [
    "source",
    "doc_summary",
    "content_hash",
    "parent_symbol",
    "start_byte",
    "end_byte",
    "repo_id",
    "score",
    "cross_lang_refs",
]
_INDEX_ESSENTIAL_KEYS = [
    "repo_id",
    "files_indexed",
    "symbols_indexed",
    "imports_indexed",
    "index_version",
    "provenance",
]
_CONTEXT_ESSENTIAL_KEYS = [
    "task",
    "symbols",
]
_EXPLORE_ESSENTIAL_KEYS = [
    "query",
    "entry_points",
    "files",
    "exact_match",
    "truncated",
    "provenance",
]
# `files` is essential (never dropped by the packer). Budget pressure is handled
# earlier: multi-file explores drop whole files, single-file explores trim source
# sections -- so the renderer always has content to display.
_EXPLORE_OPTIONAL_KEYS = [
    "relationships",
    "additional_relevant_files",
    "skeletonized",
    "skeleton_tokens_saved",
]
# Per-section source caps for tool_explore, measured in TOKENS -- the context
# unit the agent actually pays -- not chars: a char cap over-counts cheap
# line-number prefixes and clips dense bodies right at the tail (where a
# function's actual behavior lives). Base covers the common symbol; an exact
# top match is returned effectively whole (safety-ceilinged at one symbol's
# worth) so the agent never re-reads the very symbol it named.
_EXPLORE_SOURCE_SECTION_MAX_TOKENS = 900
_EXPLORE_SOURCE_SECTION_EXACT_MAX_TOKENS = 5000

# Index-free sibling skeletonization for tool_explore: when an explore result
# pulls >=3 same-kind symbols that share a name affix (e.g. *Embedder, *Resolver),
# the highest-scored member is kept full and the rest render signatures-only.
# Heuristic over the already-selected symbols -- no new index queries.
_SKELETON_KINDS = frozenset({"class", "struct", "interface", "trait", "protocol", "enum", "method", "function"})
_SKELETON_STOPWORDS = frozenset(
    {
        "make",
        "handle",
        "data",
        "base",
        "util",
        "utils",
        "test",
        "tests",
        "impl",
        "main",
        "value",
        "values",
        "name",
        "names",
        "type",
        "types",
        "node",
        "item",
        "items",
        "list",
        "dict",
        "async",
        "await",
        "none",
        "true",
        "false",
        "self",
        "func",
        "call",
        "args",
        "kwargs",
        "init",
        "build",
        "create",
        "update",
        "delete",
        "result",
        "config",
        "client",
        "server",
        "model",
        "models",
        "error",
        "errors",
    }
)
_SKELETON_MIN_FAMILY = 3
_SKELETON_MIN_BODY_LINES = 12
# Family-completion retrieval: surface sibling families that name-ranked search
# misses (FTS tokenization splits camelCase, so 'embedder' finds the base class
# but not 'OpenAIEmbedder'). Bounded substring lookups over the symbol index.
_EXPLORE_FAMILY_PROBE_SYMBOLS = 12
_EXPLORE_FAMILY_TOTAL_CAP = 12
_EXPLORE_FAMILY_PER_FAMILY_CAP = 8
# Explore relevance floor: drop ranked symbols scoring below this fraction of the
# top hit (unless pinned -- exact match, recall anchor, or seed file). Bites only
# when a query has a dominant hit (exact symbol >> lexical sub-token co-matches);
# uniform low-score concept queries leave the floor near zero, keeping everything.
_EXPLORE_SCORE_FLOOR_FRAC = 0.30
# Fused-recall tail: how many next-best fused files (beyond the source cap) to
# expose as extra candidate paths. These carry cross-channel evidence (line-FTS
# body coverage, exact-symbol matches) for concept / NL queries whose gold file
# has no top-scoring symbol hit; the lean view appends them strictly after the
# primary candidates, so recall improves without touching the primary ranking.
_FUSED_RECALL_TAIL = 24
# Deep-recall tail: a second, deeper lexical pass surfaces files whose symbol
# bodies weakly match a concept / NL query but ranked past the primary top-K.
# These files are appended AFTER the primary candidates (monotonic: they never
# reorder the top), so recall improves without touching precision. The limit is
# the depth of the deeper symbol search; the file cap bounds the tail length.
_EXPLORE_DEEP_RECALL_LIMIT = 80
_EXPLORE_DEEP_RECALL_FILES = 24
# Path-quality filters for explore results.
# Hard-remove: minified/vendor artefacts are never useful navigation targets.
_MINIFIED_FILE_RE = re.compile(r"\.min\.(js|css)$", re.IGNORECASE)
_VENDOR_PATH_RE = re.compile(r"(?:^|/)(?:vendor|node_modules|dist|__pycache__)/", re.IGNORECASE)
# Soft-penalise: test/spec files rank below implementation files unless the
# query is explicitly about tests.
_TEST_PATH_RE = re.compile(
    r"(?:^|/)tests?/|/test_[^/]+$|_test\.(?:py|js|ts|rb|go)$|(?:^|/)spec/",
    re.IGNORECASE,
)
_TEST_SCORE_PENALTY = 0.75  # multiply test-file scores by this when query doesn't mention tests
# The IDF-weighted lexical coverage constants + the multi-signal symbol scorer
# (score_symbol_row) live in the compiled-only ranking module
# (code_context/ranking.py, shipped as .so, denied from the public mirror).
# NOTE: a full bench A/B (2026-07-13) measured that scorer at only +0.003 overall
# MRR -- the ~0.67 retrieval quality comes from THIS method's channel/FTS
# architecture (exact-name channels, IDF-pruned discriminative FTS, candidate
# generation), not the re-rank. See ranking.py for the numbers.
# Definitional kinds outrank trivial variables/constants when explore decides which
# files survive the file/budget caps -- a class/function is higher signal than a const.
_DEFINITION_KINDS = frozenset(
    {"class", "struct", "interface", "trait", "protocol", "enum", "function", "method", "type_alias", "namespace"}
)
# Kinds probed by query-driven family completion (the definition families a query names).
_QUERY_PROBE_KINDS = ("class", "function", "method")
# A #define line, tolerating the variable whitespace after '#' that C header-guard
# nesting produces (e.g. kernel style '#  define likely(x) ...' inside #ifndef).
# See call_graph_centrality's macro-callee exclusion.
_MACRO_DEFINE_RE = re.compile(r"^#\s*define\b")


def _explore_skeleton_enabled() -> bool:
    """Whether tool_explore sibling skeletonization is active (env override)."""
    import os

    value = os.environ.get("LEMONCROW_EXPLORE_SKELETON")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


# Callee short-names that resolve to language builtins / ubiquitous container
# methods. As callees they have no navigable definition and only add noise +
# tokens, so they are dropped when no same-language definition is indexed.
_PY_CALLEE_NOISE: frozenset[str] = frozenset(
    {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "compile",
        "complex",
        "delattr",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "eval",
        "exec",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "globals",
        "hasattr",
        "hash",
        "help",
        "hex",
        "id",
        "input",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "locals",
        "map",
        "max",
        "memoryview",
        "min",
        "next",
        "object",
        "oct",
        "open",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "vars",
        "zip",
        "append",
        "extend",
        "insert",
        "pop",
        "remove",
        "clear",
        "copy",
        "get",
        "keys",
        "values",
        "items",
        "update",
        "setdefault",
        "add",
        "discard",
        "join",
        "split",
        "rsplit",
        "splitlines",
        "strip",
        "lstrip",
        "rstrip",
        "replace",
        "startswith",
        "endswith",
        "lower",
        "upper",
        "title",
        "encode",
        "decode",
        "read",
        "write",
        "readline",
        "readlines",
        "close",
        "flush",
        "seek",
        "format_map",
        "index",
        "count",
        "sort",
        "reverse",
        "find",
        "rfind",
        "group",
    }
)

_USAGES_ESSENTIAL_KEYS = ["file_path", "line", "provenance"]
_USAGES_OPTIONAL_KEYS = ["snippet", "caller", "edge_kind", "confidence"]
_PATTERN_ESSENTIAL_KEYS = ["file_path", "line", "snippet"]
_PATTERN_OPTIONAL_KEYS = ["captures", "column", "end_line", "end_column"]
_STATUS_ESSENTIAL_KEYS = [
    "repo_id",
    "repo_root",
    "db_path",
    "index_version",
    "index",
    "cache",
    "providers",
    "provider_freshness",
    "warnings",
    "freshness",
    "autosync",
    "provenance",
]
_CACHE_STATUS_ESSENTIAL_KEYS = [
    "repo_id",
    "index_version",
    "entry_count",
    "entries_by_tool",
    "total_bytes",
    "max_bytes",
]
_CACHE_INVALIDATE_ESSENTIAL_KEYS = ["invalidated_entries"]
_CALL_GRAPH_ESSENTIAL_KEYS = [
    "target",
    "direction",
    "related",
    "related_count",
    "data_status",
    "provenance",
]
_CALL_GRAPH_OPTIONAL_KEYS = [
    "depth",
    "related",
    "related_count",
    "truncated",
    "edges",
    "edge_count",
    "data_status",
    "ambiguity",
    "message",
    "snapshot",
]
_BLAME_ESSENTIAL_KEYS = [
    "symbol_name",
    "file_path",
    "provenance",
]
_BLAME_OPTIONAL_KEYS = [
    "index_sha",
    "head_sha",
    "last_modified",
    "last_commit_summary",
    "hunks",
    "churn",
]
_OVERFLOW_SPILL_MIN_EXCESS_TOKENS = 128
_OVERFLOW_SPILL_MIN_REDUCTION_TOKENS = 256
DeletedHistoryItem = dict[str, Any]
_CACHE_TOOL_ALIASES = {
    "all": None,
    "explore": "code.explore",
    "files": "code.files",
    "status": "code.status",
    "routes": "code.routes",
    "search": "code.search",
    "symbol": "code.symbol",
    "context": "code.context",
    "usages": "code.usages",
    "callers": "code.callers",
    "callees": "code.callees",
    "pattern": "code.pattern",
}
_OPERATION_TOKEN_CAPS = {
    "cache_status": 50,
    "index": 80,
    "search": 800,
    "symbol": 800,
    "pattern": 800,
    "callers": 700,
    "callees": 300,
    "usages": 700,
    "context": 2400,
    "blame": 50,
    "cache_invalidate": 35,
}
# Map internal field names to shortened MCP output names to reduce token bloat.
# Applied post-processing in _short_item_keys().
_FIELD_NAME_SHORTMAP = {
    "file_path": "path",
    "symbol_name": "name",
    "symbol_id": "id",
    "start_line": "line",
    "doc_summary": "doc",
    "deleted_at": "deleted",
    "deleted_at_sha": "deleted_sha",
    "last_author": "author",
    "last_commit_msg": "msg",
    "matched_on": "match",
    "rename_target": "renamed_to",
    "rename_note": "rename",
}


def apply_field_name_shortening(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply field-name shortening to reduce token bloat in MCP responses.
    Maps internal names (file_path, start_line, etc.) to compact forms (path, line, etc.).
    Applies recursively to all nested structures.
    """

    def shorten_dict(item: dict[str, Any]) -> dict[str, Any]:
        """Recursively shorten field names in a dict."""
        result: dict[str, Any] = {}
        for k, v in item.items():
            short_key = _FIELD_NAME_SHORTMAP.get(k, k)
            if isinstance(v, dict):
                result[short_key] = shorten_dict(v)
            elif isinstance(v, list):
                if v and isinstance(v[0], dict):
                    result[short_key] = [shorten_dict(i) if isinstance(i, dict) else i for i in v]
                else:
                    result[short_key] = v
            else:
                result[short_key] = v
        return result

    # Shorten entire payload recursively, but handle snapshot specially
    result = {}
    for k, v in payload.items():
        short_key = _FIELD_NAME_SHORTMAP.get(k, k)
        if isinstance(v, dict):
            # Don't shorten top-level snapshot dict as it has specific structure
            if k != "snapshot":
                result[short_key] = shorten_dict(v)
            else:
                result[short_key] = v
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            result[short_key] = [shorten_dict(item) for item in v]  # type: ignore[assignment]
        else:
            result[short_key] = v
    return result


_SEARCH_SNIPPET_FORCE_COMPACT_LIMIT = 50


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


@dataclass(frozen=True)
class _ExtractedSymbol:
    name: str
    qualified_name: str
    kind: str
    signature: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    parent_symbol: str | None = None
    doc_summary: str | None = None

    # mypyc-compiled instances have no __dict__, so pickle's default restore
    # path calls setattr() per field instead of updating __dict__ directly —
    # which frozen dataclasses reject, breaking cross-process results in
    # _parallel_extract. Route reconstruction through __init__ instead, which
    # already writes fields via object.__setattr__.
    def __reduce__(self) -> tuple[type[_ExtractedSymbol], tuple[object, ...]]:
        return (self.__class__, tuple(getattr(self, f.name) for f in fields(self)))


@dataclass(frozen=True)
class _IndexedReference:
    """A local, index-time reference extracted from Python AST."""

    file_path: str
    symbol_name: str
    line: int
    column: int
    end_column: int
    enclosing_symbol_name: str | None
    enclosing_qualified_name: str | None
    snippet: str

    # See _ExtractedSymbol.__reduce__: mypyc classes have no __dict__, so
    # frozen dataclasses need an explicit __reduce__ to survive
    # ProcessPoolExecutor's pickle round-trip.
    def __reduce__(self) -> tuple[type[_IndexedReference], tuple[object, ...]]:
        return (self.__class__, tuple(getattr(self, f.name) for f in fields(self)))


@dataclass(frozen=True)
class _IndexedCallEdge:
    """A local, index-time function/method call edge extracted from Python AST."""

    caller_symbol_name: str
    caller_qualified_name: str
    caller_file_path: str
    caller_start_line: int
    caller_end_line: int
    callee_name: str
    call_line: int
    call_column: int
    snippet: str

    # See _ExtractedSymbol.__reduce__: mypyc classes have no __dict__, so
    # frozen dataclasses need an explicit __reduce__ to survive
    # ProcessPoolExecutor's pickle round-trip.
    def __reduce__(self) -> tuple[type[_IndexedCallEdge], tuple[object, ...]]:
        return (self.__class__, tuple(getattr(self, f.name) for f in fields(self)))


@dataclass
class _FileIndexData:
    """Pure extraction result for one file — no DB handles."""

    rel: str
    language: str
    content_hash: str
    size_bytes: int
    text_lines: list[tuple[int, str]]
    symbols: list[_ExtractedSymbol]
    symbol_sources: list[str]  # source slices for FTS (parallel to symbols)
    imports: list[tuple[str, str | None]]
    references: list[_IndexedReference]
    call_edges: list[_IndexedCallEdge]
    mtime_ns: int = 0


# Held in the open contract (subclasses a builtin -> can't live in a compiled
# module); re-exported here so `from ...engine import IndexLockTimeout` still works.
from lemoncrow.core.capabilities.code_context_contract import (  # noqa: E402
    IndexLockTimeout as IndexLockTimeout,
)


def _index_lock_timeout_s() -> float:
    """Seconds a blocking index-write-lock acquisition waits before giving up.

    Defaults to 10s (unchanged); override via LEMONCROW_INDEX_LOCK_TIMEOUT_S for
    long prewarm builds that must win the lock before serving tool calls.
    """
    raw = os.environ.get("LEMONCROW_INDEX_LOCK_TIMEOUT_S")
    if not raw:
        return 10.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 10.0


def _repo_id(repo_root: Path) -> str:
    return hashlib.sha256(str(repo_root.resolve()).encode("utf-8")).hexdigest()[:16]


def _sqlite_vector_extension_path() -> str | None:
    """Path to the sqlite_vector loadable extension, or None when the optional
    ``sqliteai-vector`` package is absent. SQLite appends the platform shared-lib
    suffix to the bare 'vector' stem at load time."""
    if _sqlite_vector_ext_memo:
        return _sqlite_vector_ext_memo[0]
    path: str | None
    try:
        import importlib.resources

        path = str(importlib.resources.files("sqlite_vector.binaries") / "vector")
    except (ImportError, OSError):
        path = None
    _sqlite_vector_ext_memo.append(path)
    return path


# Above this many bytes, skip vector_quantize_preload and let the TurboQuant scan
# read the quantized data via mmap (reclaimable, file-backed) instead of pinning it
# in anonymous RAM. ~960 MB for linux's 1.24Mx1536 store stays under the cap.
_SQLITE_VEC_PRELOAD_MAX_BYTES = 2 * 1024**3

# One-element memo: [] = not probed yet, [path|None] = resolved once per process.
_sqlite_vector_ext_memo: list[str | None] = []


def _default_db_path(repo_root: Path) -> Path:
    from lemoncrow.core.foundation.paths import resolve_workspace_store_dir

    return resolve_workspace_store_dir(workspace_root=repo_root) / "code_context.sqlite"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    total = 0
    for line in text.splitlines(keepends=True):
        total += len(line.encode("utf-8"))
        offsets.append(total)
    if not text.endswith(("\n", "\r")):
        offsets.append(total)
    return offsets


def _safe_relpath(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


# Binary/non-text extensions to skip when walking non-source files for the
# Python-text-search fallback.  Kept small and conservative — the walk is
# only reached when rg is unavailable and the FTS index is empty.
_SKIP_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".webp",
        ".svg",
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        ".mkv",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".so",
        ".dylib",
        ".dll",
        ".o",
        ".a",
        ".pyc",
        ".pyo",
        ".class",
    }
)


def _walk_non_source_files(root: Path) -> list[Path]:
    """Yield readable non-source files under *root* (best-effort, shallow)."""
    results: list[Path] = []
    try:
        for entry in sorted(root.rglob("*")):
            if not entry.is_file():
                continue
            if entry.suffix.lower() in _SKIP_EXTENSIONS:
                continue
            # Skip hidden dirs and common junk dirs.
            parts = entry.relative_to(root).parts
            if any(p.startswith(".") or p in {"node_modules", "__pycache__", "vendor", ".git"} for p in parts[:-1]):
                continue
            results.append(entry)
    except OSError:
        pass
    return results


# IDF pruning cap for the FTS OR/prefix channels: a query token present in more
# than this fraction of all indexed symbols (e.g. "get", "name", "field") is
# non-discriminative -- it contributes a huge bm25 posting-list scan while the
# rarer tokens decide the match. Such tokens are dropped from the FTS MATCH (see
# CodeContextEngine._discriminative_fts_terms). ~10% is the elbow where common
# code tokens start dominating scan cost. The absolute floor disables pruning on
# small corpora: a posting list under ~1500 docs scans in a few ms, so there is
# nothing to prune, and a percentage cap on a tiny repo would wrongly drop tokens
# present in only 2-3 symbols (breaking recall on small indexes).
_FTS_COMMON_TERM_DF_FRACTION = 0.10
_FTS_COMMON_TERM_DF_FLOOR = 1500
# Anchor-df cap for the substring/path trigram channels (a MUCH tighter bound than
# the FTS OR/prefix cap above). A token with df over this has large trigram
# postings, so `t.name LIKE '%anchor%'` scans 100s of ms on a big repo to surface a
# few non-token substring matches -- while FTS bm25 already ranked the whole-token
# match in ~15ms. Below it, the substring scan is a few ms and catches partial-token
# substrings the FTS tokenizer splits ('mbedde' in 'Embedder'). Override via env.
_SUBSTRING_ANCHOR_DF_CAP = int(os.environ.get("LEMONCROW_SUBSTRING_ANCHOR_DF_CAP", "500"))
# "Semantic additive only" fusion gate for the hybrid symbol-search path: freeze
# the top-K lexical(+graph) hits so the semantic channel can only surface symbols
# lexical missed -- it never demotes a symbol lexical already ranked in the top-K.
# 0 disables the gate (prior RRF behaviour); see SemanticSearchRanker.reciprocal_rank_fuse.
_SEMANTIC_ADDITIVE_TOP_K = 5
# Cap the FTS OR/prefix query to the rarest few discriminative terms.  The most
# selective tokens carry the match; extra mid-frequency tokens ("data", "set")
# only enlarge the bm25 posting-list scan.  Bounds FTS latency regardless of how
# many tokens a messy multi-clause/regex query produces.  Bounded by TOTAL bm25
# posting-list work (sum of doc-frequencies), rarest-first, rather than a flat
# term count: a small repo keeps every cheap term (no recall loss), while a big
# repo stops before a mid-frequency term would blow the budget.
# Maximum cumulative document-frequency across all OR-query terms.  When a term
# would push the total above this, it is dropped.  Capping at ~500 prevents a
# handful of moderately-common tokens (e.g. "import", df≈1000 in django) from
# forcing FTS5 to score thousands of rows when rarer, more-discriminative terms
# are already present.  The budget never drops ALL terms: the first (rarest) term
# is always kept regardless of its df (see _discriminative_fts_terms).
_FTS_DF_BUDGET = 500

# Thread pool for parallel FTS channel execution in _search_symbols_local.
# SQLite's C extension releases the GIL during query execution, so threads
# achieve true CPU-level parallelism for I/O-bound FTS reads without the
# process-startup overhead or fork-while-threaded deadlock risks of a
# ProcessPoolExecutor.  Each thread opens its own read connection (WAL mode
# makes concurrent readers lock-free).
_SEARCH_CHANNEL_EXECUTOR: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
    max_workers=16,
)

# Persistent thread pool for the three V6 HEF channels (_hef_exact_symbol_candidates,
# _hef_anchor_zoekt_candidates, _hef_line_fts_candidates).  Module-level so threads
# are reused across queries: no spawn/join overhead per call, and the 200ms deadline
# in _fused_explore_hybrid actually releases the caller immediately instead of
# waiting for shutdown(wait=True).
_HEF_CHANNEL_EXECUTOR: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
    max_workers=3, thread_name_prefix="lemoncrow-hef"
)

# Dedicated pool for the per-anchor Zoekt fetches inside
# _hef_anchor_zoekt_candidates. That method already runs ON _HEF_CHANNEL_EXECUTOR,
# so re-submitting its nested _fetch tasks back onto the SAME 3-worker pool
# self-starves under concurrent explores: the parent tasks occupy every worker
# while blocking on nested tasks that can never start, so the channel silently
# returns []. A separate pool lets the nested fetches run without re-entering the
# caller's pool while preserving the per-fetch 0.15s deadline.
_HEF_ANCHOR_EXECUTOR: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
    max_workers=6, thread_name_prefix="lemoncrow-hef-anchor"
)

# Dedicated pool for the semantic-symbol-search timeout wrapper in
# search_symbols()'s hybrid branch (see _SEMANTIC_SYMBOL_DEADLINE_S). Isolated
# from _SEARCH_CHANNEL_EXECUTOR (shared by the 5 lexical FTS/trigram channels)
# so an abandoned/slow semantic load -- possible even with the matrix-loading
# lock below, since a large-repo chunked-streaming query never caches and can
# legitimately run past the deadline -- never competes with lexical search
# capacity for a worker slot.
_SEMANTIC_SYMBOL_EXECUTOR: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="lemoncrow-semsym"
)


# Per-statement wall-clock deadline for a single search channel.  The caller's
# `_fut.result(timeout=8.0)` only abandons the WAIT -- without this, the SQLite
# query keeps running in the worker thread (a leading-wildcard LIKE '%term%'
# full-scans a huge symbol_trigram table for 15-20s on linux-sized repos) and
# piles up behind later queries.  A progress handler that trips at the deadline
# ABORTS the running statement, so a pathological scan self-cancels.  Kept below
# the 8s caller wait so the channel returns [] cleanly before that fires.
_SEARCH_CHANNEL_DEADLINE_S = float(os.environ.get("LEMONCROW_SEARCH_CHANNEL_DEADLINE_S", "2.5"))

# Hard cap on the synchronous semantic-symbol lookup inside search_symbols()'s
# hybrid branch. That lookup (_search_symbols_semantic_local ->
# _search_symbols_semantic_ann) is normally a cached in-memory matmul (<10ms
# warm), but a cache miss at huge-repo scale (linux: ~727k-1.24M vectors) pays
# a real multi-second-to-tens-of-seconds table load/reconstruct -- confirmed via
# a real run: every natural-language query against an isolated linux snapshot
# measured 6.4-10.8s here (flat, not just the first query), and the full
# multi-channel benchmark run saw a 58.8s tail. _maybe_warm_ann_cache /
# prewarm_semantic_matrix reduce how often this is hit but don't guarantee it
# (background warm race, an evicted cache, a freshly started process) -- unlike
# the anchor-file recall channel (_semantic_candidate_files, capped at 500ms via
# the same submit+result(timeout=) pattern below), this call sat directly in the
# critical path with no bound at all. On timeout the abandoned future keeps
# running on the shared executor (warms the cache for the NEXT query) while this
# query falls back to lexical-only ranking -- degraded, not blocked.
_SEMANTIC_SYMBOL_DEADLINE_S = float(os.environ.get("LEMONCROW_SEMANTIC_SYMBOL_DEADLINE_S", "2.0"))

# Benchmark/CI-only: re-raise a semantic-symbol failure (timeout or exception)
# instead of the production default of silently degrading to lexical_hits.
# Graceful degradation is correct in production (one embedder hiccup must never
# take down a user's whole query) but it also means an eval/benchmark run can
# silently score "semantic channel worked" numbers that are actually 100%
# lexical fallback -- exactly the failure mode that hid the real bug behind an
# earlier fix here (see the except-Exception comment below). Off by default;
# set only in benchmark/CI harnesses, never in a real deployment.
_SEMANTIC_SYMBOL_HARD_FAIL = os.environ.get("LEMONCROW_SEMANTIC_HARD_FAIL", "").strip() == "1"

# Temporary diagnostic (LEMONCROW_SEMANTIC_STATS=1): tally how the semantic-
# symbol branch actually resolves under real load -- ok / timeout / exception --
# and log a running count every 20 calls. The timeout/exception branches below
# are otherwise completely silent (by design, for production), so there is no
# way to see from the outside whether "semantic enabled" is actually
# contributing or is silently starved by _SEMANTIC_SYMBOL_DEADLINE_S under
# sustained multi-repo load. Remove once the investigation this supports is
# closed out.
_SEMANTIC_SYMBOL_STATS_ENABLED = os.environ.get("LEMONCROW_SEMANTIC_STATS", "").strip() == "1"
_SEMANTIC_SYMBOL_STATS: dict[str, int] = {"ok": 0, "timeout": 0, "exception": 0}
_SEMANTIC_SYMBOL_STATS_LOCK = threading.Lock()


def _tally_semantic_symbol_outcome(outcome: str) -> None:
    if not _SEMANTIC_SYMBOL_STATS_ENABLED:
        return
    with _SEMANTIC_SYMBOL_STATS_LOCK:
        _SEMANTIC_SYMBOL_STATS[outcome] += 1
        total = sum(_SEMANTIC_SYMBOL_STATS.values())
        if total % 20 == 0:
            logger.warning("semantic-symbol tally after %d calls: %s", total, dict(_SEMANTIC_SYMBOL_STATS))


# Tight deadline for the low-priority recall-supplement channels (substring +
# path trigram scans, base 820-860).  Their cost is set by trigram POSTING size,
# not token df: a common substring like 'include' has huge 3-gram postings
# ('inc','ncl'), so the LIKE scans 300-500ms to surface a few non-token matches
# the FTS bm25 channel already ranked in ~15ms.  A rare anchor's scan finishes in
# <5ms, so this bound only ever fires on the non-discriminative case, where the
# channel adds no recall anyway -- it self-aborts (-> empty) instead of dominating
# p100.  Override via LEMONCROW_SUPPLEMENT_CHANNEL_DEADLINE_S.
_SUPPLEMENT_CHANNEL_DEADLINE_S = float(os.environ.get("LEMONCROW_SUPPLEMENT_CHANNEL_DEADLINE_S", "0.08"))


# Cap for CodeContextEngine._file_bytes_cache (persistent stat-validated file
# bytes). 64 MB comfortably holds the hot source set of a large repo; a single
# file larger than the cap is served uncached.
_FILE_BYTES_CACHE_MAX_BYTES = 64 * 1024 * 1024

# mmap_size ceiling for SQLite read connections (see _apply_pragmas and
# _channel_connection). Both used to hard-code a 256MB mmap window regardless
# of the actual DB size -- harmless for a small repo, but a real gap for a
# large one: django's code_context.sqlite (352MB) and linux_core's (1.1GB)
# both exceed it, so everything past the first 256MB silently fell back to
# buffered reads through a 4-16MB private page cache instead of the shared,
# reclaimable, page-cache-backed mmap. Sizing the window to the file itself
# (floored at the old fixed cap, capped here for safety) fixes that without
# any real memory cost: mmap_size is a virtual-mapping ceiling, not a
# preallocation -- resident memory still only grows with the pages a query
# actually touches, and connections mapping the same file share physical
# pages via the OS page cache. Override via LEMONCROW_SQLITE_MMAP_CEILING_BYTES.
_SQLITE_MMAP_FLOOR_BYTES = 268435456
_SQLITE_MMAP_CEILING_BYTES = int(os.environ.get("LEMONCROW_SQLITE_MMAP_CEILING_BYTES", str(4 * 1024 * 1024 * 1024)))


def _sized_mmap_bytes(path: Path) -> int:
    """mmap_size for `path`, floored/capped by _SQLITE_MMAP_{FLOOR,CEILING}_BYTES."""
    try:
        size = os.stat(path).st_size
    except OSError:
        return _SQLITE_MMAP_FLOOR_BYTES
    return max(_SQLITE_MMAP_FLOOR_BYTES, min(size, _SQLITE_MMAP_CEILING_BYTES))


# cache_size ceiling for SQLite read connections (see _apply_pragmas and
# _channel_connection). mmap_size already covers a full read of the on-disk
# file (see _sized_mmap_bytes), but it does NOT cover everything cache_size
# does: pages sitting in the WAL that haven't been checkpointed into the main
# file yet (relevant whenever autosync/incremental reindex writes while reads
# are in flight), and the fallback when mmap silently fails to apply (some
# filesystems, or a platform where sqlite3.OperationalError is raised and
# suppressed -- see _channel_connection). So it's still worth sizing generously
# on a big machine: 2GB comfortably holds even the largest indexed repo's
# whole code_context.sqlite (linux_core: ~1.1GB) directly in SQLite's own
# cache, independent of mmap. Override via LEMONCROW_SQLITE_CACHE_CEILING_KB (KB).
_SQLITE_CACHE_CEILING_KB = int(os.environ.get("LEMONCROW_SQLITE_CACHE_CEILING_KB", str(2 * 1024 * 1024)))  # 2 GB

# Hard reserve subtracted from available RAM BEFORE any cache_size sizing --
# this much is never spent regardless of how generous the ceiling above is or
# how much RAM looks free. Floored at a fixed amount so a small machine still
# keeps a real safety margin, and scaled as a fraction of TOTAL (not
# available) RAM so the reserve itself doesn't shrink just because something
# else is already using memory. Override LEMONCROW_SQLITE_RAM_RESERVE_BYTES to
# pin an exact byte count instead, or LEMONCROW_SQLITE_RAM_RESERVE_FRACTION to
# change the percentage.
_SQLITE_RAM_RESERVE_FLOOR_BYTES = int(
    os.environ.get("LEMONCROW_SQLITE_RAM_RESERVE_FLOOR_BYTES", str(4 * 1024 * 1024 * 1024))
)  # 4 GB
_SQLITE_RAM_RESERVE_FRACTION = float(
    os.environ.get("LEMONCROW_SQLITE_RAM_RESERVE_FRACTION", "0.10")
)  # 10% of total RAM


def _total_ram_bytes() -> int:
    """Best-effort TOTAL physical RAM. Used for the reserve floor (below), which
    should track the machine's real capacity, not fluctuate with what else is
    using memory right now the way /proc/meminfo's MemAvailable does."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 4 * 1024 * 1024 * 1024


def _available_ram_bytes() -> int:
    """Best-effort available RAM, for scaling the cache_size ceiling to the
    machine instead of a one-size-fits-all constant. Prefers /proc/meminfo's
    MemAvailable (Linux; already accounts for reclaimable page cache); falls
    back to total physical RAM via sysconf, then a conservative 4GB assumption
    when neither is readable (non-Linux, sandboxed, etc.)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return _total_ram_bytes()


_RAM_BUDGET_TTL_S = 30.0
_ram_budget_cache: tuple[float, int] | None = None


def _ram_budget_bytes() -> int:
    """Available RAM minus the hard reserve -- what cache sizing is allowed to
    spend. Never negative (a reserve bigger than what's free just collapses
    every connection to its floor, e.g. a tiny/already-loaded machine)."""
    # Cached for _RAM_BUDGET_TTL_S: this runs on every connection (re)open --
    # per tool call on the main connection -- and reading /proc/meminfo each
    # time is measurable at benchmark rates. MemAvailable drifting for up to
    # 30s only shifts an advisory cache-size hint, never correctness.
    global _ram_budget_cache
    now = time.monotonic()
    cached = _ram_budget_cache
    if cached is not None and now - cached[0] < _RAM_BUDGET_TTL_S:
        return cached[1]
    reserve = max(_SQLITE_RAM_RESERVE_FLOOR_BYTES, int(_total_ram_bytes() * _SQLITE_RAM_RESERVE_FRACTION))
    budget = max(0, _available_ram_bytes() - reserve)
    _ram_budget_cache = (now, budget)
    return budget


# Sizing each connection independently (each just checking "is there budget
# for ME") is unsound: up to this many connections can be open at once in one
# process -- the 16 _SEARCH_CHANNEL_EXECUTOR threads (one pooled connection
# each) plus the one _apply_pragmas/_connect() connection -- and each would
# see the SAME undiminished budget and claim up to it independently, so the
# real worst case is (this many) x (per-connection size), not one size.
# Dividing the budget by this count BEFORE sizing any single connection makes
# the stated worst case (this many connections, all maxed) mathematically
# bounded by the actual budget instead of a multiple of it.
_SQLITE_MAX_POOLED_CONNECTIONS = 17


def _sized_cache_kb(floor_kb: int) -> int:
    """cache_size (KB): floored at the old fixed default, capped at
    _SQLITE_CACHE_CEILING_KB, and never allowed to dip into the RAM reserve
    (_ram_budget_bytes already subtracted it) even with every pooled connection
    maxed out at once (budget is pre-divided across _SQLITE_MAX_POOLED_CONNECTIONS).
    """
    budget_kb = _ram_budget_bytes() // 1024 // _SQLITE_MAX_POOLED_CONNECTIONS
    return max(floor_kb, min(budget_kb, _SQLITE_CACHE_CEILING_KB))


_SEARCH_CONN_LOCAL = threading.local()


# Maximum pooled connections per thread in _channel_connection().  Threads are
# persistent (_SEARCH_CHANNEL_EXECUTOR), so uncapped accumulation across
# different db_paths (e.g. test tmp dirs) blows past the host FD limit — each
# SQLite connection holds ≥3 FDs (main + WAL + SHM).  Production workloads
# alternate between at most 2 paths (main DB + fts.sqlite); 4 gives headroom
# without wasting FDs.
_CHANNEL_CONNECTION_SLOTS_MAX = 4


def _channel_connection(db_path: Path) -> sqlite3.Connection:
    """Thread-local pooled read connection for search channels.

    _SEARCH_CHANNEL_EXECUTOR threads are persistent, so reopening a connection
    per channel call paid connect + pragma cost on every query AND threw away
    SQLite's page cache each time (~8% of query wall time under py-spy). One
    slot per thread: repo-focused workflows (sessions, benchmarks) hit the same
    db_path repeatedly; a different db_path swaps the slot. The (dev, inode)
    key drops the pooled connection when the DB file is replaced on disk
    (workspace re-init), which a path-only key would silently miss.

    Slots are capped at _CHANNEL_CONNECTION_SLOTS_MAX per thread; oldest
    entries are evicted (closed) when the cap is exceeded so that test-time
    accumulation of unique tmp paths never exhausts the FD limit.
    """
    try:
        st = os.stat(db_path)
        key = (str(db_path), st.st_dev, st.st_ino)
    except OSError:
        key = (str(db_path), -1, -1)
    # One slot per (thread, db file): HEF channel threads alternate between the
    # main DB (exact channel) and fts.sqlite (line channel); a single slot per
    # thread would close/reopen a connection on every alternation.
    slots: dict[str, tuple[tuple[str, int, int], sqlite3.Connection]] | None = getattr(
        _SEARCH_CONN_LOCAL, "slots", None
    )
    if slots is None:
        slots = {}
        _SEARCH_CONN_LOCAL.slots = slots
    slot = slots.get(key[0])
    if slot is not None:
        if slot[0] == key:
            return slot[1]
        with contextlib.suppress(sqlite3.Error):
            slot[1].close()
        slots.pop(key[0], None)
    # Evict oldest entries when the cap is exceeded so that unique paths
    # (e.g. one per test tmp dir) don't accumulate without bound.
    while len(slots) >= _CHANNEL_CONNECTION_SLOTS_MAX:
        oldest_key = next(iter(slots))
        with contextlib.suppress(sqlite3.Error):
            slots[oldest_key][1].close()
        slots.pop(oldest_key)
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        timeout=5.0,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    # A meaningful page cache + mmap I/O only pay off now that the connection
    # outlives a single query. Both scale with the machine/file instead of a
    # fixed size: mmap_size to the file on disk (see _sized_mmap_bytes) so a
    # large repo's whole index gets memory-mapped instead of quietly falling
    # back to buffered reads past a fixed 256MB; cache_size to available RAM
    # (see _sized_cache_kb, floored at the old fixed 16MB) so a big-RAM box
    # keeps more hot pages resident in SQLite's own cache.
    conn.execute(f"PRAGMA cache_size = -{_sized_cache_kb(16_000)}")
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute(f"PRAGMA mmap_size = {_sized_mmap_bytes(db_path)}")
    slots[key[0]] = (key, conn)
    return conn


def _run_search_channel(
    db_path: Path, sql: str, params: tuple[Any, ...], deadline_s: float = _SEARCH_CHANNEL_DEADLINE_S
) -> list[dict[str, Any]]:
    """Execute one FTS/trigram search channel on a pooled read connection.

    Called from worker threads in _SEARCH_CHANNEL_EXECUTOR so that all five
    channels of _search_symbols_local run concurrently.  Each worker thread
    keeps one pooled read connection per DB file (WAL mode makes reader-reader
    access lock-free), so connect/pragma cost and a cold SQLite page cache are
    paid once per thread instead of once per channel call.  SQLite's C
    extension releases the GIL during query execution, so threads achieve
    genuine CPU-level parallelism for these I/O-bound FTS reads.
    sqlite3.OperationalError (e.g. FTS edge-case syntax) is swallowed and
    treated as an empty result set so it never crashes the caller.
    Returns dicts (not sqlite3.Row) for consistent dict-based result handling.

    A SQLite progress handler enforces _SEARCH_CHANNEL_DEADLINE_S as a real
    statement timeout: sqlite3.connect(timeout=...) is only a busy-lock wait, so
    without this a slow full-scan (leading-wildcard LIKE on a large trigram
    table) would run for tens of seconds even after the caller has timed out.
    Returning non-zero from the handler raises OperationalError, caught below as
    an empty channel -> graceful degradation instead of a stuck worker thread.
    """
    conn = _channel_connection(db_path)
    _deadline = time.monotonic() + deadline_s
    # Checked every ~50k VM ops: frequent enough to abort a runaway scan within
    # milliseconds of the deadline, rare enough to add no measurable overhead to
    # the common <200ms query.
    conn.set_progress_handler(lambda: 1 if time.monotonic() > _deadline else 0, 50_000)
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.set_progress_handler(None, 0)


def _fts_or_query_from_terms(terms: list[str]) -> str:
    return " OR ".join(f'"{term[:64]}"' for term in terms[:12] if term)


def _fts_prefix_query_from_terms(terms: list[str]) -> str:
    return " OR ".join(f'"{term[:64]}"*' for term in terms[:12] if term)


def _safe_fts_query(query: str) -> str:
    # Quote each term as an FTS5 string literal so natural-language queries whose
    # words happen to be FTS operators (or/and/near/not) are treated as literal
    # terms instead of breaking the MATCH grammar. Terms are [A-Za-z0-9_]+ only,
    # so no embedded-quote escaping is required. Use the cleaned query terms
    # (snake/camel subtokens, code/regex noise dropped) so messy queries
    # (`def foo|def bar`, `^class Baz`) match on real identifiers, not "def".
    return _fts_or_query_from_terms(_query_terms(query))


def _fts_prefix_query(query: str) -> str:
    return _fts_prefix_query_from_terms(_query_terms(query))


def _fts_and_query(query: str) -> str:
    """FTS5 implicit-AND query for code/identifier queries with 2+ distinctive terms.

    Multi-term AND finds symbols whose indexed text contains ALL distinctive terms,
    a stronger hit signal than the OR fallback.  Guards against natural-language
    queries (e.g. "create login token for authenticated user") that would otherwise
    match a symbol's docstring text and blur the lexical/semantic boundary.  Only
    fires when the query contains structural code markers: pipe separators,
    underscores, CamelCase terms, ALL_CAPS identifiers, or a code-keyword prefix.
    Returns empty string when the guard rejects the query or fewer than two
    distinctive terms (length ≥ 4) are present.
    """
    if os.environ.get("ABLATE_B"):  # ablation switch (benchmark attribution only)
        return ""
    terms_raw = _FTS_TERM_RE.findall(query)
    if not terms_raw:
        return ""

    # Natural-language guard: skip the AND channel unless the query has at least one
    # structural code marker.  Without |/_/CamelCase/ALL_CAPS/code-kw, the query is
    # almost certainly a concept description, not a code/identifier search.
    if "|" not in query and "_" not in query:
        has_code_kw = terms_raw[0].lower() in _CODE_LEADING_KW
        has_mixed_case = any(any(c.isupper() for c in t) and any(c.islower() for c in t) for t in terms_raw)
        has_all_caps = any(len(t) >= 4 and t.isupper() for t in terms_raw)
        # n-gram compound tokens (e.g. 'donothingaction', 'preprocessable') are
        # always code-derived and never appear in natural-language docstrings.
        # The n-gram pipeline strips underscores so they won't have |/_ markers;
        # use a minimum length of 12 chars as a reliable compound-token signal.
        has_long_compound = len(terms_raw) >= 2 and len(terms_raw[0]) >= 12
        if not (has_code_kw or has_mixed_case or has_all_caps or has_long_compound):
            return ""

    seen_lower: set[str] = set()
    distinctive: list[str] = []
    for t in terms_raw:
        tl = t.lower()
        if len(t) >= 4 and tl not in seen_lower:
            seen_lower.add(tl)
            distinctive.append(t)
    if len(distinctive) < 2:
        return ""
    # FTS5 implicit AND: space-separated quoted phrases without the OR keyword.
    return " ".join(f'"{t[:64]}"' for t in distinctive[:4])


def _ngram_tokens(name: str) -> list[str]:
    """Stripped-join n-gram tokens for a compound identifier.

    The FTS5 unicode61 tokenizer splits on ``_`` so 'get_timezone' stored in
    the index is silently broken into 'get' + 'timezone' -- the compound form
    is lost.  By joining sub-tokens WITHOUT a separator we get tokens that the
    tokenizer treats as a single unit::

        _get_timezone_name
        → pieces: ['get', 'timezone', 'name']
        → bigrams: 'gettimezone', 'timezonename'
        → full:    'gettimezonename'

    These are highly specific (df ≈ 1-3) so they are cheap to scan and give
    precise BM25 signal.  Used both when indexing symbols and when extracting
    query terms, so the two sides stay in sync.
    """
    pieces: list[str] = []
    for raw in _FTS_TERM_RE.findall(name):
        for camel in _CAMEL_BOUNDARY_RE.split(raw):
            for piece in camel.split("_"):
                p = piece.strip().lower()
                if p:
                    pieces.append(p)
    if len(pieces) < 2:
        return []
    out: list[str] = []
    # Bigrams (consecutive pairs)
    for i in range(len(pieces) - 1):
        out.append(pieces[i] + pieces[i + 1])
    # Full compound (3+ parts only; 2-part already covered by the single bigram)
    if len(pieces) >= 3:
        out.append("".join(pieces))
    return out


def _identifier_terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw in _FTS_TERM_RE.findall(text):
        pieces: list[str] = []
        for camel in _CAMEL_BOUNDARY_RE.split(raw):
            # Split snake_case / dotted pieces too, so `_sqlite_datetime_parse`
            # yields sqlite/datetime/parse (matches a query naming those) instead
            # of only the whole underscore-joined token.
            for piece in camel.split("_"):
                lowered = piece.strip().lower()
                if lowered:
                    pieces.append(lowered)
                    terms.append(lowered)
        # Emit stripped-join bigrams and full compound so that a query for
        # '_get_timezone_name' also searches 'gettimezone timezonename
        # gettimezonename', matching only the one symbol that defines the exact
        # compound name instead of every file that contains 'get'/'timezone'.
        terms.extend(_ngram_tokens(raw))
    return terms


# Query-time morphological stemming.  The index is built once and frozen, so a
# query's inflected words never re-tokenize to the code's canonical identifier
# form: a prose query says "parsing" / "incrementing" / "counts" while the code
# defines "parse" / "increment" / "count".  Exact FTS matching misses these
# outright -- measured on the pytest index, "incrementing" matched 0 lines while
# "increment*" matched the defining file; "counts" matched 2 files vs 28 for
# "count*".  _query_stem returns a stem to search as an FTS prefix (stem*), or
# None when the term is too short or carries no strippable inflection.  It is
# deliberately conservative: a wrong stem only widens recall on a channel that
# bm25 IDF already down-weights, and a missed stem falls back to exact matching.
_STEM_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        {
            "izations",
            "ization",
            "izing",
            "izes",
            "ized",
            "ations",
            "ation",
            "ements",
            "ement",
            "ings",
            "ing",
            "ments",
            "ment",
            "ness",
            "tions",
            "tion",
            "sions",
            "sion",
            "ers",
            "ors",
            "ies",
            "es",
            "ed",
            "ly",
            "s",
        },
        key=len,
        reverse=True,
    )
)
_STEM_VOWELS = frozenset("aeiou")


def _query_stem(term: str) -> str | None:
    """Return an FTS-prefix stem for an inflected query term, or None.

    Strips one common English inflection and collapses a trailing doubled
    consonant (``formatted`` -> ``formatt`` -> ``format``).  Intended to be used
    as a prefix (``stem*``), so an imperfect stem still matches the canonical
    form; the minimum stem length (4) keeps it from degenerating to a wildcard.
    """
    t = term.lower()
    if len(t) < 5 or not t.isalnum():
        return None
    for suffix in _STEM_SUFFIXES:
        if t.endswith(suffix):
            stem = t[: len(t) - len(suffix)]
            if suffix == "ies":
                stem += "y"
            if len(stem) < 4:
                continue
            if len(stem) >= 5 and stem[-1] == stem[-2] and stem[-1] not in _STEM_VOWELS:
                stem = stem[:-1]
            if stem != t and len(stem) >= 4:
                return stem
    return None


def _trigrams(text: str) -> list[str]:
    """Overlapping lowercased 3-char sequences, matching the FTS5 trigram tokenizer
    so the index can serve approximate (typo) lookups -- candidates sharing trigrams
    with the query -- instead of a full-table edit-distance scan (the pg_trgm idea)."""
    s = text.lower()
    return [s[i : i + 3] for i in range(len(s) - 2)] if len(s) >= 3 else []


# Python keywords / regex-ish noise that carry no symbol signal -- dropped from
# QUERY term extraction so messy queries (grep regexes, `def foo`, `^class Bar`)
# don't flood FTS with "def"/"class" or stall on metacharacters. Symbol-name
# tokenization (_identifier_terms) is deliberately left untouched.
_QUERY_STOPWORDS = frozenset(
    {
        "def",
        "class",
        "return",
        "self",
        "cls",
        "import",
        "from",
        "lambda",
        "async",
        "await",
        "yield",
        "pass",
        "raise",
        "with",
        "for",
        "while",
        "if",
        "elif",
        "else",
        "try",
        "except",
        "finally",
        "and",
        "or",
        "not",
        "none",
        "true",
        "false",
        "del",
        "global",
        "nonlocal",
        "assert",
        "break",
        "continue",
        "in",
        "is",
        "as",
        "the",
        "this",
    }
)


def _query_terms(query: str) -> list[str]:
    """Identifier subtokens of a query with keyword/regex noise removed. Falls
    back to the raw identifier terms if filtering would empty the query."""
    if os.environ.get("ABLATE_A"):  # ablation switch (benchmark attribution only)
        return _identifier_terms(query)
    cleaned = [t for t in _identifier_terms(query) if len(t) >= 2 and t not in _QUERY_STOPWORDS]
    return cleaned or _identifier_terms(query)


# Punctuation trimmed off the ends of a raw whitespace token before it is
# probed for existence -- quotes/brackets a caller wrapped around a literal
# aren't part of the substring that actually appears in the repo.
_FALLBACK_TOKEN_STRIP_CHARS = "\"'`,;:!?()[]{}<>"


def _fallback_probe_tokens(query: str) -> list[str]:
    """Whitespace tokens of *query*, longest first, for the existence-gated
    substring fallback in ``tool_search`` (see
    ``CodeContextEngine._existence_gated_fallback_token``).

    Deliberately NOT ``_identifier_terms``: that splits on every internal
    ``_``/camelCase/``.`` boundary, which would shatter a hyphenated literal
    like ``x-flipt-accept-server-version`` into ``x``, ``flipt``, ``accept``...
    -- losing the exact string a caller may have embedded in a descriptive
    query. A plain whitespace split keeps each token intact (hyphens, dots,
    underscores and all) so the existence probe downstream can gate on the
    literal exactly as written, regardless of its shape.

    Stopwords (``_QUERY_STOPWORDS``) and very short tokens are dropped before
    probing -- not because shape disqualifies them, but because probing is a
    DB round trip per token and there's no reason to spend one on "the" or
    "a". Tokens are then sorted longest-first: a long token is far less
    likely to coincidentally exist elsewhere in the repo than a short common
    word, so trying the most specific-looking tokens first minimizes the
    chance that a coincidental hit on an ordinary word (e.g. "parse", which
    is a plausible substring in nearly any codebase) wins the race before the
    genuinely diagnostic literal is even tried. This is a priority ordering
    only -- every token, whatever its shape, is still eligible.
    """
    seen: set[str] = set()
    candidates: list[str] = []
    for raw in query.split():
        token = raw.strip(_FALLBACK_TOKEN_STRIP_CHARS)
        if not (3 <= len(token) <= 100):
            continue
        lowered = token.lower()
        if lowered in _QUERY_STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        candidates.append(token)
    candidates.sort(key=len, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Explore top-5 reranker: feature extraction + linear scoring
# ---------------------------------------------------------------------------
# Features are computed purely from what _tool_explore_impl already returned
# (file path, symbol list, source sections). No DB I/O, no network calls.

_DEFAULT_EXPLORE_RERANKER_MODEL_PATH = Path(__file__).with_name("explore_reranker_model.json")

_ER_FEATURE_NAMES: tuple[str, ...] = (
    "reciprocal_rank",
    "rank_one",
    "path_term_coverage",
    "path_identifier_exact",
    "basename_similarity",
    "symbol_term_coverage",
    "symbol_identifier_exact",
    "source_term_coverage",
    "source_best_line_coverage",
    "test_scope_match",
    "test_scope_mismatch",
    "doc_scope_match",
    "doc_scope_mismatch",
    "path_depth",
    # Semantic cosine of the file's best-matching symbol to the query (0 when the
    # file was not a semantic anchor). The learned reranker was previously blind to
    # the embedding signal -- this is the feature it needs to arbitrate lexical vs
    # semantic. Sourced from the explore payload's per-file ``semantic_score``.
    "semantic_cosine",
    # Cheap path/query interaction features. These avoid source flattening and
    # let the model distinguish filename, directory, regex, prose, and path-shaped
    # searches without changing candidate generation or the result surface.
    "basename_term_coverage",
    "directory_term_coverage",
    "basename_token_jaccard",
    "basename_identifier_exact",
    "path_suffix_exact",
    "extension_match",
    "ordered_path_terms",
    "last_directory_term_coverage",
    "query_identifier_like",
    "query_regex_like",
    "query_prose_like",
    "query_path_like",
    "query_term_count",
)
_ER_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "class",
        "def",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "return",
        "self",
        "the",
        "to",
        "with",
    }
)
_ER_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ER_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_ER_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ER_TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|testing|specs?)(/|$)|(^|/)test_[^/]+$|_test\.[^/]+$",
    re.IGNORECASE,
)
_ER_DOC_PATH_RE = re.compile(
    r"(^|/)(docs?|documentation|examples?|galleries)(/|$)|\.(?:md|rst|ipynb)$",
    re.IGNORECASE,
)
_ER_PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ER_EXTENSION_QUERY_RE = re.compile(r"\.([A-Za-z0-9]{1,8})(?:\b|$)")
_ER_REGEX_QUERY_RE = re.compile(r"[|*()\[\]^$+?\\{}]")
_ER_IDENTIFIER_QUERY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
_ER_DEFINITION_QUERY_RE = re.compile(r"^(?:def|class|async\s+def)\s+[A-Za-z_][A-Za-z0-9_.]*$")

# --- Hybrid explore fusion (v6) ---
_HEF_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_HEF_DEFINITION_RE = re.compile(r"\b(?P<kind>def|class)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
_HEF_QUOTED_RE = re.compile(r"""(?P<quote>["'])(?P<value>.*?)(?P=quote)""")
_HEF_TEST_RE = re.compile(
    r"(^|/)(tests?|testing|specs?)(/|$)|(^|/)test_[^/]+$",
    re.IGNORECASE,
)
_HEF_AUX_RE = re.compile(
    r"(^|/)(docs?(?:-internal)?|documentation|examples?|galleries|benchmarks?|"
    r"frontend|vendor|third_party)(/|$)|\.(?:md|rst|ipynb|json|lock)$",
    re.IGNORECASE,
)
_HEF_STOP: frozenset[str] = frozenset(
    {
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "case",
        "class",
        "continue",
        "def",
        "del",
        "do",
        "else",
        "except",
        "false",
        "finally",
        "for",
        "from",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "none",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "self",
        "super",
        "true",
        "try",
        "while",
        "with",
        "yield",
    }
)
_HEF_PROSE_STOP: frozenset[str] = _HEF_STOP | frozenset(
    {
        "the",
        "this",
        "that",
        "these",
        "those",
        "then",
        "than",
        "into",
        "onto",
        "when",
        "where",
        "which",
        "what",
        "with",
        "without",
        "within",
        "should",
        "could",
        "would",
        "have",
        "has",
        "had",
        "does",
        "did",
        "done",
        "make",
        "using",
        "used",
        "use",
        "value",
        "values",
        "result",
        "results",
        "file",
        "files",
        "code",
        "name",
        "string",
        "object",
        "method",
        "function",
    }
)

_ER_QUERY_TEST_RE = re.compile(
    r"\btests?\b|\btesting\b|\bpytest\b|\bunittest\b|\bspecs?\b|\btest_[A-Za-z0-9_]+",
    re.IGNORECASE,
)
_ER_QUERY_DOC_RE = re.compile(
    r"\bdocs?\b|\bdocumentation\b|\bexamples?\b|\bgallery\b|\breadme\b",
    re.IGNORECASE,
)


def _er_identifier_parts(value: str) -> list[str]:
    parts: list[str] = []
    for raw in re.split(r"[./:_-]+", value):
        for part in _ER_CAMEL_RE.split(raw):
            normalized = part.strip().lower()
            if len(normalized) >= 2 and normalized not in _ER_STOPWORDS:
                parts.append(normalized)
    return parts


def _er_dedupe(values: list[str], limit: int) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        n = v.strip().lower()
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
        if len(out) >= limit:
            break
    return tuple(out)


def _er_query_features(
    query: str,
) -> tuple[tuple[str, ...], tuple[str, ...], bool, bool]:
    identifiers = [
        t
        for t in _ER_IDENTIFIER_RE.findall(query)
        if len(t) >= 3
        and t.lower() not in _ER_STOPWORDS
        and ("_" in t or "." in t or t.isupper() or any(c.isupper() for c in t[1:]))
    ]
    terms: list[str] = []
    for raw in _ER_TOKEN_RE.findall(query):
        terms.extend(_er_identifier_parts(raw))
        n = raw.lower()
        if len(n) >= 3 and n not in _ER_STOPWORDS:
            terms.append(n)
    return (
        _er_dedupe(terms, 20),
        _er_dedupe(identifiers, 12),
        bool(_ER_QUERY_TEST_RE.search(query)),
        bool(_ER_QUERY_DOC_RE.search(query)),
    )


def _er_flatten_text(value: Any, limit: int = 12_000) -> str:
    chunks: list[str] = []
    remaining = limit

    def _visit(item: Any) -> None:
        nonlocal remaining
        if remaining <= 0 or item is None:
            return
        if isinstance(item, str):
            text = item[:remaining]
            chunks.append(text)
            remaining -= len(text)
        elif isinstance(item, dict):
            for k, child in item.items():
                if str(k) in {"content_hash", "symbol_id", "repo_id"}:
                    continue
                _visit(child)
                if remaining <= 0:
                    break
        elif isinstance(item, (list, tuple)):
            for child in item:
                _visit(child)
                if remaining <= 0:
                    break

    _visit(value)
    return "\n".join(chunks)


def _er_coverage(text: str, terms: tuple[str, ...]) -> float:
    if not terms:
        return 0.0
    lowered = text.lower()
    return sum(t in lowered for t in terms) / len(terms)


def _er_char_trigrams(value: str) -> set[str]:
    """Character trigrams for Jaccard similarity (not the FTS tokenizer)."""
    n = re.sub(r"[^a-z0-9]+", "", value.lower())
    if not n:
        return set()
    if len(n) < 3:
        return {n}
    return {n[i : i + 3] for i in range(len(n) - 2)}


def _er_trisim(left: str, right: str) -> float:
    lg, rg = _er_char_trigrams(left), _er_char_trigrams(right)
    if not lg or not rg:
        return 0.0
    return len(lg & rg) / len(lg | rg)


def _er_query_shape(query: str) -> str:
    stripped = query.strip()
    if "/" in stripped or _ER_EXTENSION_QUERY_RE.search(stripped):
        return "path"
    if _ER_REGEX_QUERY_RE.search(stripped):
        return "regex"
    if _ER_DEFINITION_QUERY_RE.fullmatch(stripped) or _ER_IDENTIFIER_QUERY_RE.fullmatch(stripped):
        return "identifier"
    return "prose"


def _er_normalized_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _er_path_query_features(
    query: str,
    file_path: str,
    terms: tuple[str, ...],
    identifiers: tuple[str, ...],
) -> list[float]:
    path_text = file_path.replace("\\", "/").lower()
    path = PurePosixPath(path_text)
    basename = path.stem
    directory = "/".join(path.parts[:-1])
    last_directory = path.parts[-2] if len(path.parts) > 1 else ""

    basename_tokens = set(_ER_PATH_TOKEN_RE.findall(basename))
    query_tokens = set(terms)
    token_union = basename_tokens | query_tokens
    token_jaccard = len(basename_tokens & query_tokens) / len(token_union) if token_union else 0.0
    basename_identifier_exact = max(
        (
            float(_er_normalized_identifier(identifier) == _er_normalized_identifier(basename))
            for identifier in identifiers
        ),
        default=0.0,
    )

    normalized_query_path = query.strip().replace("\\", "/").lower().lstrip("./")
    query_is_path = bool("/" in normalized_query_path or _ER_EXTENSION_QUERY_RE.search(normalized_query_path))
    path_suffix_exact = float(
        bool(normalized_query_path)
        and query_is_path
        and (path_text == normalized_query_path or path_text.endswith(f"/{normalized_query_path}"))
    )
    requested_extensions = {match.group(1).lower() for match in _ER_EXTENSION_QUERY_RE.finditer(query)}
    extension_match = float(bool(requested_extensions) and path.suffix.lstrip(".").lower() in requested_extensions)

    position = -1
    ordered_terms = 0
    for term in terms:
        next_position = path_text.find(term, position + 1)
        if next_position < 0:
            break
        ordered_terms += 1
        position = next_position
    ordered_path_terms = ordered_terms / len(terms) if terms else 0.0

    query_shape = _er_query_shape(query)
    return [
        _er_coverage(basename, terms),
        _er_coverage(directory, terms),
        token_jaccard,
        basename_identifier_exact,
        path_suffix_exact,
        extension_match,
        ordered_path_terms,
        _er_coverage(last_directory, terms),
        float(query_shape == "identifier"),
        float(query_shape == "regex"),
        float(query_shape == "prose"),
        float(query_shape == "path"),
        min(1.0, len(terms) / 8.0),
    ]


def _er_entry_features(
    query: str,
    entry: dict[str, Any],
    rank: int,
    active_feature_indices: Sequence[int] | None = None,
) -> list[float]:
    terms, identifiers, wants_tests, wants_docs = _er_query_features(query)
    raw_path = str(entry.get("file_path") or entry.get("path") or "")
    file_path = raw_path.replace("\\", "/")
    path_text = file_path.lower()
    basename = Path(file_path).stem

    active = frozenset(active_feature_indices) if active_feature_indices is not None else None
    needs_symbols = active is None or bool(active & {5, 6})
    needs_source = active is None or bool(active & {7, 8})
    needs_path_query = active is None or any(index >= 15 for index in active)
    symbol_text = _er_flatten_text(entry.get("symbols")) if needs_symbols else ""
    source_text = _er_flatten_text(entry.get("source_sections")) if needs_source else ""
    source_lines = source_text.splitlines() if needs_source else []
    best_line = max(
        (_er_coverage(ln, terms) for ln in source_lines[:400]),
        default=0.0,
    )

    path_id_exact = max(
        (float(ident in path_text) for ident in identifiers),
        default=0.0,
    )
    sym_id_exact = max(
        (float(ident in symbol_text.lower()) for ident in identifiers),
        default=0.0,
    )
    basename_sim = max(
        (_er_trisim(ident, basename) for ident in identifiers),
        default=0.0,
    )

    is_test = bool(_ER_TEST_PATH_RE.search(file_path))
    is_doc = bool(_ER_DOC_PATH_RE.search(file_path))
    depth = min(1.0, file_path.count("/") / 12.0)
    path_query_features = (
        _er_path_query_features(query, file_path, terms, identifiers) if needs_path_query else [0.0] * 13
    )

    return [
        1.0 / max(1, rank),
        float(rank == 1),
        _er_coverage(path_text, terms),
        path_id_exact,
        basename_sim,
        _er_coverage(symbol_text, terms),
        sym_id_exact,
        _er_coverage(source_text, terms),
        best_line,
        float(wants_tests and is_test),
        float(not wants_tests and is_test),
        float(wants_docs and is_doc),
        float(not wants_docs and is_doc),
        depth,
        float(entry.get("semantic_score") or 0.0),
        *path_query_features,
    ]


def _er_entry_path(entry: dict[str, Any]) -> str:
    """Extract the file path from an explore file entry."""
    return str(entry.get("file_path") or entry.get("path") or "")


def _er_linear_score(weights: list[float], features: list[float]) -> float:
    return sum(w * f for w, f in zip(weights, features, strict=True))


def _validate_er_model(raw: Any) -> dict[str, Any] | None:
    """Validate/normalize an explore reranker model (LambdaMART trees or linear).

    Legacy per-workspace deploys have no ``model_type`` and are treated as linear.
    Models using an older feature-vector prefix remain valid when new features
    are appended to the canonical vector.
    """
    if not isinstance(raw, dict) or not raw.get("enabled"):
        return None
    feature_names = raw.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or feature_names != list(_ER_FEATURE_NAMES[: len(feature_names)])
    ):
        return None
    active = raw.get("active_feature_indices")
    if active is not None:
        if (
            not isinstance(active, list)
            or not active
            or any(not isinstance(index, int) or index < 0 or index >= len(feature_names) for index in active)
        ):
            return None
        raw = {**raw, "active_feature_indices": list(dict.fromkeys(active))}
    model_type = raw.get("model_type")
    if model_type == "lambdamart_trees":
        return raw if isinstance(raw.get("trees"), list) and raw["trees"] else None
    if model_type in ("linear", None):
        if len(raw.get("weights", [])) != len(feature_names):
            return None
        return raw if model_type == "linear" else {**raw, "model_type": "linear"}
    return None


def _er_tree_score(trees: list[dict[str, Any]], features: list[float]) -> float:
    """Sum leaf values across a LambdaMART forest for one candidate.

    Each tree is stored as parallel arrays (feature/threshold/left/right/leaf).
    A node with ``feature == -1`` is a leaf. Decision rule: ``x < threshold``
    takes the ``left`` branch (XGBoost "yes" direction). Constant per-candidate
    offsets (base_score) are omitted — they do not change within-group order.
    Pure-Python and dependency-free so it stays well under the inline budget
    (~40 depth-3 trees over a handful of candidates: a few thousand compares).
    """
    total = 0.0
    for tree in trees:
        feature = tree["feature"]
        threshold = tree["threshold"]
        left = tree["left"]
        right = tree["right"]
        node = 0
        while feature[node] != -1:
            node = left[node] if features[feature[node]] < threshold[node] else right[node]
        total += tree["leaf"][node]
    return total


# Characters that mean a query is NOT a literal path substring: whitespace (a
# multi-term query) and regex/FTS metacharacters (an alternation/pattern query).
# When any are present, `file_path LIKE '%<whole query>%'` can never match a real
# path, so the path channel skips the full-query LIKE and keeps only the cheap
# first-term LIKE.
_NON_PATHY_QUERY_RE = re.compile(r"[\s|*()\[\]^$+?\\=<>{}]")


def _query_is_pathy_literal(query: str) -> bool:
    """True when the whole query could appear as a literal substring of a file
    path (a single token with no regex/multi-term metacharacters)."""
    q = query.strip()
    return bool(q) and _NON_PATHY_QUERY_RE.search(q) is None


def _is_precise_symbol_query(query: str) -> bool:
    return bool(_PRECISE_SYMBOL_QUERY_RE.fullmatch(query.strip()))


def _matches_file_glob(path: str, pattern: str) -> bool:
    normalized_path = path.replace("\\", "/")
    normalized_pattern = pattern.replace("\\", "/")
    pure_path = PurePosixPath(normalized_path)
    if pure_path.match(normalized_pattern):
        return True
    if "**/" in normalized_pattern and pure_path.match(normalized_pattern.replace("**/", "")):
        return True
    if fnmatch.fnmatch(normalized_path, normalized_pattern):
        return True
    regex = re.escape(normalized_pattern)
    regex = regex.replace(r"\*\*/", r"(?:.*/)?")
    regex = regex.replace(r"\*\*", r".*")
    regex = regex.replace(r"\*", r"[^/]*")
    regex = regex.replace(r"\?", r"[^/]")
    if re.fullmatch(regex, normalized_path):
        return True
    return False


def _exact_symbol_hits(hits: list[SymbolRecord], query: str) -> list[SymbolRecord]:
    normalized_query = query.strip()
    normalized_query_lower = normalized_query.lower()
    case_sensitive = [
        hit for hit in hits if hit.symbol_name == normalized_query or hit.qualified_name == normalized_query
    ]
    if case_sensitive:
        return case_sensitive
    return [
        hit
        for hit in hits
        if hit.symbol_name.lower() == normalized_query_lower or hit.qualified_name.lower() == normalized_query_lower
    ]


# A query is "symbol-like" when it is a bare identifier or dotted path (no
# spaces) -- the shape worth an explicit exact-name lookup. Multi-word concept
# queries skip that lookup so they never pay an extra search.
_SYMBOL_QUERY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
# A token looks like a code identifier when it has an internal underscore
# (trim_docstring) or a camelCase boundary (MyClass) -- the shape worth an
# exact symbol-name probe in multi-word queries. Plain English words
# (admindocs, default, role) never match, staying on the anchor/recall path.
_COMPOUND_IDENT_RE = re.compile(r"[A-Za-z0-9]_[A-Za-z0-9]|[a-z][A-Z]")
_REGEX_NOISE_RE = re.compile(r"[\^$\\().*+?\[\]{}|\s]")


def _split_pipe_query(query: str) -> list[str]:
    """Expand a pipe-delimited OR pattern into individual searchable terms.

    Zoekt treats ``|`` as a literal character, not an OR operator, so a query
    like ``"timezone_name|TIME_ZONE|def timezone"`` finds nothing.  This helper
    splits on ``|``, strips leading regex anchors (``^``, ``\\b``, etc.) and
    trailing noise, and returns the distinct non-trivial terms for individual
    searches whose results are then unioned.

    Returns an empty list when the query has no ``|``, or when fewer than two
    meaningful terms survive cleaning (fall back to the original query).
    """
    if "|" not in query:
        return []
    seen: dict[str, None] = {}  # ordered dedup
    for part in query.split("|"):
        # Strip common regex anchors/metacharacters from edges
        stripped = part.strip().lstrip("^").rstrip("$")
        stripped = re.sub(r"\\[bBsSwWdD]", "", stripped).strip()
        # Must contain at least one word character and be >=3 chars
        if len(stripped) < 3 or not re.search(r"[a-zA-Z0-9_]", stripped):
            continue
        seen[stripped] = None
    terms = list(seen)
    return terms if len(terms) >= 2 else []


def _query_implies_test_scope(query: str) -> bool:
    lowered = query.lower()
    return any(token in lowered for token in ("test", "tests", "spec", "pytest", "unittest"))


def _is_test_file_path(file_path: str) -> bool:
    lowered = file_path.lower()
    # Cheap basename via rfind instead of Path(...).name -- this runs once per
    # candidate row in the search hot loop, where pathlib object construction
    # dominates the profile.  '/' is the normalized separator in stored paths.
    name = lowered[lowered.rfind("/") + 1 :]
    return "/test" in lowered or "/tests/" in lowered or name.startswith("test_") or name.endswith("_test.py")


def _parse_since_filter(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError("since must not be empty")
    match = _SINCE_RELATIVE_RE.fullmatch(normalized.lower())
    if match:
        amount = int(match.group("amount"))
        unit = match.group("unit")
        delta = {
            "d": timedelta(days=amount),
            "w": timedelta(weeks=amount),
            "m": timedelta(days=amount * 30),
            "y": timedelta(days=amount * 365),
        }[unit]
        return int((datetime.now(UTC) - delta).timestamp())
    iso_value = normalized.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError:
        try:
            parsed_date = date.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError("since must be an ISO date/datetime or relative duration like 30d") from exc
        parsed = datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp())


def _normalize_touched_by(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("touched_by must not be empty")
    return normalized


def _row_to_symbol(row: Mapping[str, Any]) -> SymbolRecord:
    row_keys = set(row.keys())
    return SymbolRecord(
        symbol_id=str(row["symbol_id"]),
        repo_id=str(row["repo_id"]),
        file_path=str(row["file_path"]),
        language=str(row["language"]),
        symbol_name=str(row["symbol_name"]),
        qualified_name=str(row["qualified_name"]),
        kind=str(row["kind"]),
        signature=str(row["signature"]),
        start_byte=int(row["start_byte"]),
        end_byte=int(row["end_byte"]),
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        parent_symbol=cast(str | None, row["parent_symbol"]),
        doc_summary=cast(str | None, row["doc_summary"]),
        content_hash=str(row["content_hash"]),
        score=float(row["score"]) if "score" in row_keys and row["score"] is not None else None,
    )


def _git_repo_class() -> Any:
    try:
        from git import Repo
    except Exception:  # pragma: no cover - optional dependency fallback
        logging.exception("Recovered from broad exception handler")
        return None
    return Repo


# Minimum address space (MB) a spawn worker needs to re-import the full package
# (interpreter + tree-sitter grammars + gitpython + glibc arenas, which scale
# with core count). Measured: ~2.5 GB OOMs on import, ~4 GB is safe. RLIMIT_AS
# caps *virtual* address space, which runs well ahead of actual RSS, so the
# per-worker cap must never drop below this floor or workers die on startup.
_WORKER_MIN_MB = 4096


def _available_memory_mb() -> int | None:
    """Best-effort memory we may use, in MB: the lesser of host MemAvailable and
    any cgroup memory ceiling. Returns ``None`` when it can't be determined."""
    candidates: list[int] = []
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    candidates.append(int(line.split()[1]) // 1024)  # kB -> MB
                    break
    except Exception:  # non-Linux / unreadable -- fall through
        pass
    for cg in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(cg).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw and raw != "max" and raw.isdigit():
            val = int(raw)
            if 0 < val < (1 << 62):  # cgroup v1 uses a huge sentinel for "unlimited"
                candidates.append(val // (1024 * 1024))
    return min(candidates) if candidates else None


def _memory_capped_index_workers(cpu_workers: int) -> int:
    """Cap an intended worker count by the memory budget for fresh interpreters."""
    avail_mb = _available_memory_mb()
    if avail_mb is None:
        return max(1, cpu_workers)
    mem_workers = max(1, int(avail_mb * 0.8) // _WORKER_MIN_MB)
    return max(1, min(cpu_workers, mem_workers))


def _resolve_index_max_workers() -> int:
    """Worker count for an explicit indexing ProcessPool.

    Honors ``LEMONCROW_INDEX_MAX_WORKERS`` first. Otherwise CLI/manual indexing
    uses all CPUs, capped by available memory: each spawn worker is a fresh
    interpreter that re-imports the full package (~4 GB of address space), so on
    a memory-constrained host one-per-CPU OOM-kills the pool on import. The pool
    is sized so its total address-space budget stays within ~80% of available
    memory (OS- and cgroup-aware).
    """
    override = os.environ.get("LEMONCROW_INDEX_MAX_WORKERS", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    return _memory_capped_index_workers(os.cpu_count() or 1)


def _resolve_autosync_index_max_workers() -> int:
    """Worker count for background autosync indexing.

    Autosync is opportunistic background work, so default it to half the CPU
    budget while leaving explicit ``lc code index`` runs fully parallel.
    """
    override = os.environ.get("LEMONCROW_AUTOSYNC_INDEX_MAX_WORKERS", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    return _memory_capped_index_workers(max(1, (os.cpu_count() or 1) // 2))


def _resolve_serial_extract_threshold() -> int:
    """File count at/below which indexing extracts serially, in-process.

    A spawned ``ProcessPoolExecutor`` pays a fixed ~1-2s spawn+shutdown cost
    (each worker is a fresh interpreter that re-imports the whole package). For
    small repos that overhead dwarfs the millisecond-scale parsing, so serial
    extraction is faster and produces byte-for-byte identical results. Honors
    ``LEMONCROW_INDEX_SERIAL_MAX_FILES`` (set to 0 to always use the pool).
    """
    override = os.environ.get("LEMONCROW_INDEX_SERIAL_MAX_FILES", "").strip()
    if override.isdigit():
        return int(override)
    return 64


# ---------------------------------------------------------------------------
# Shared process pool — one pool for the lifetime of the process so that
# repeated index calls don't each spawn a fresh set of interpreter workers.
# ---------------------------------------------------------------------------

_PROCESS_POOL: concurrent.futures.ProcessPoolExecutor | None = None
_PROCESS_POOL_LOCK = threading.Lock()


def _worker_memory_guard() -> None:
    """Worker-process initializer: cap virtual address space and arm parent-death signal.

    Two protections applied in each forked worker:

    1. **PR_SET_PDEATHSIG** (Linux only) — deliver SIGTERM to this worker the
       moment its multiprocessing parent exits or crashes. Without this,
       workers can be re-parented to the user's systemd instance and keep
       running (and holding ~300 MB PSS each) indefinitely.

    2. **RLIMIT_AS** — cap virtual address space to this worker's share of ~80%
       of available memory, never below the per-worker import floor
       (``_WORKER_MIN_MB``) -- so a worker can't false-OOM just re-importing the
       package, while a pathological parse still can't grow it unbounded.
       Override with ``LEMONCROW_INDEX_WORKER_MAX_MEM_MB`` (0 disables).

    Both are skipped silently where the relevant OS primitives are unavailable.
    """
    # --- 1. Auto-die when parent exits (Linux workers only) --------------------
    try:
        import ctypes as _ctypes

        _libc = _ctypes.CDLL("libc.so.6", use_errno=True)
        _PR_SET_PDEATHSIG = 1
        _SIGTERM = 15
        _libc.prctl(_PR_SET_PDEATHSIG, _SIGTERM, 0, 0, 0)
    except Exception:
        pass

    # --- 2. Cap address space to prevent runaway OOM ---------------------------
    try:
        import resource as _resource

        override = os.environ.get("LEMONCROW_INDEX_WORKER_MAX_MEM_MB", "").strip()
        if override.lstrip("-").isdigit():
            mb = int(override)
        else:
            avail_mb = _available_memory_mb()
            if avail_mb is None:
                return  # can't size safely -> don't cap (a too-low RLIMIT_AS OOMs on import)
            mb = max(_WORKER_MIN_MB, int(avail_mb * 0.8) // _resolve_index_max_workers())
        if mb <= 0:
            return
        limit = mb * 1024 * 1024
        _resource.setrlimit(_resource.RLIMIT_AS, (limit, limit))
    except Exception:
        pass


def _get_index_process_pool() -> concurrent.futures.ProcessPoolExecutor:
    """Return the shared ProcessPoolExecutor, creating it lazily on first use."""
    global _PROCESS_POOL
    if _PROCESS_POOL is not None:
        return _PROCESS_POOL
    with _PROCESS_POOL_LOCK:
        if _PROCESS_POOL is None:
            # Never fork here. The pool is created lazily from inside indexing,
            # after the parent has opened SQLite connections and acquired the
            # index-write lock. Forked workers inherit those descriptors and can
            # deadlock the result queue or keep DB/lock handles alive after the
            # parent exits. A forkserver gives Linux most of fork's startup speed
            # without cloning the indexing process's live state; other platforms
            # use spawn.
            _ctx_name = os.environ.get("LEMONCROW_INDEX_POOL_CONTEXT", "")
            if not _ctx_name:
                import platform

                _ctx_name = "forkserver" if platform.system() == "Linux" else "spawn"
            mp_ctx = multiprocessing.get_context(_ctx_name)
            _pool_kwargs: dict[str, Any] = {
                "max_workers": _resolve_index_max_workers(),
                "mp_context": mp_ctx,
                "initializer": _worker_memory_guard,
            }
            # max_tasks_per_child recycles workers to free AST/string
            # garbage, but is incompatible with the 'fork' start method.
            if _ctx_name != "fork":
                _pool_kwargs["max_tasks_per_child"] = 256
            _PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(**_pool_kwargs)
            atexit.register(_shutdown_index_process_pool)
    return _PROCESS_POOL


def _reset_index_process_pool() -> None:
    """Tear down a broken pool so the next call recreates it."""
    global _PROCESS_POOL
    with _PROCESS_POOL_LOCK:
        if _PROCESS_POOL is not None:
            _PROCESS_POOL.shutdown(wait=False, cancel_futures=True)
            _PROCESS_POOL = None


def _shutdown_index_process_pool() -> None:  # atexit handler
    global _PROCESS_POOL
    if _PROCESS_POOL is not None:
        _PROCESS_POOL.shutdown(wait=False, cancel_futures=True)
        _PROCESS_POOL = None


def _process_one_file(
    repo_root_str: str,
    path_str: str,
    source_bytes: bytes | None = None,
) -> _FileIndexData | None:
    """Worker entry-point for ``ProcessPoolExecutor`` — pure extraction, no DB.

    Standalone module-level function (pickleable) that does all the extraction
    work for a single file in a subprocess.
    """
    repo_root = Path(repo_root_str)
    path = Path(path_str)

    try:
        st = path.stat()
    except OSError:
        return None
    if st.st_size > _MAX_FILE_BYTES:
        return None

    payload = source_bytes if source_bytes is not None else path.read_bytes()
    source = payload.decode("utf-8", errors="replace")
    language = detect_language(path) or "text"
    rel = _safe_relpath(repo_root, path)
    content_hash = _sha256_bytes(payload)

    # ---- pre-parse AST for Python (parsed once, reused below) ----
    py_tree: ast.Module | None = None
    if language == "python":
        try:
            py_tree = ast.parse(source)
        except SyntaxError:
            py_tree = None

    # ---- extract symbols ----
    tag_list: list[Tag] = []
    if language == "python":
        if py_tree is not None:
            extracted = _extract_python_symbols(source, tree=py_tree)
        else:
            extracted = []
    elif language == "markdown":
        from lemoncrow.infra.code_intel.markdown import extract_markdown_symbols

        extracted = [_ExtractedSymbol(**s) for s in extract_markdown_symbols(source)]
    else:
        try:
            tag_list = extract_tags(path)
        except (OSError, SyntaxError):
            tag_list = []
        extracted = _extract_tag_symbols_worker(path, source, language, tags=tag_list)

    # Pre-read symbol source slices for FTS5 (avoids re-reading during write)
    symbol_sources: list[str] = []
    for sym in extracted:
        s = payload[sym.start_byte : sym.end_byte].decode("utf-8", errors="replace")
        symbol_sources.append(s[:20_000])

    # ---- extract imports ----
    imports_list: list[tuple[str, str | None]] = []
    if language == "python":
        if py_tree is not None:
            imports_list.extend(_python_imports_worker(repo_root, path, source, tree=py_tree))
    elif language in {"typescript", "javascript"}:
        imports_list.extend(_javascript_imports_worker(repo_root, path, source))
    elif language == "rust":
        for match in _RUST_MOD_RE.finditer(source):
            raw = match.group(1)
            imports_list.append((raw, _resolve_relative_module_worker(repo_root, path.parent, raw, [".rs"])))
    elif language == "go":
        for match in _GO_IMPORT_RE.finditer(source):
            raw_block = match.group(1) or match.group(2) or ""
            for raw in re.findall(r"\"([^\"]+)\"", raw_block) or [raw_block]:
                imports_list.append((raw, None))
    imports_list = sorted(set((raw, target) for raw, target in imports_list if raw and target != rel))

    # ---- extract references / call edges ----
    # Python: rich AST references + call edges. Every other tree-sitter language:
    # reference rows from the same tag parse used for symbols, so query-time
    # find_references is a pure index lookup (no whole-repo re-parse).
    references: list[_IndexedReference] = []
    call_edges: list[_IndexedCallEdge] = []
    if language == "python" and py_tree is not None:
        references, call_edges = _extract_python_reference_index_worker(rel, source, extracted, tree=py_tree)
    elif tag_list:
        references, call_edges = _extract_tag_reference_index_worker(rel, source, tag_list, extracted)

    return _FileIndexData(
        rel=rel,
        language=language,
        content_hash=content_hash,
        size_bytes=st.st_size,
        text_lines=[(idx, line[:20_000]) for idx, line in enumerate(source.splitlines(), start=1)],
        symbols=extracted,
        symbol_sources=symbol_sources,
        imports=imports_list,
        references=references,
        call_edges=call_edges,
        mtime_ns=st.st_mtime_ns,
    )


def _extract_python_symbols(source: str, tree: ast.Module | None = None) -> list[_ExtractedSymbol]:
    """Extract Python symbols from source (module-level, pickleable).

    If *tree* is provided (pre-parsed AST), it is used instead of parsing *source*.
    """
    if tree is None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
    offsets = _line_offsets(source)
    lines = source.splitlines()
    symbols: list[_ExtractedSymbol] = []

    def line_text(line_no: int) -> str:
        if 1 <= line_no <= len(lines):
            return lines[line_no - 1].strip()
        return ""

    def add_node(node: ast.AST, name: str, kind: str, parent: str | None) -> None:
        start_line = int(getattr(node, "lineno", 1))
        end_line = int(getattr(node, "end_lineno", start_line))
        col = int(getattr(node, "col_offset", 0))
        end_col = int(getattr(node, "end_col_offset", 0))
        start_byte = offsets[max(0, start_line - 1)] + col
        end_byte = offsets[max(0, end_line - 1)] + end_col if end_col else offsets[min(end_line, len(offsets) - 1)]
        qualified = f"{parent}.{name}" if parent else name
        doc = (
            ast.get_docstring(node) if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) else None
        )
        symbols.append(
            _ExtractedSymbol(
                name=name,
                qualified_name=qualified,
                kind=kind,
                signature=line_text(start_line),
                start_byte=start_byte,
                end_byte=max(start_byte, end_byte),
                start_line=start_line,
                end_line=end_line,
                parent_symbol=parent,
                doc_summary=(stripped.splitlines()[0][:200] if doc and (stripped := doc.strip()) else None),
            )
        )

    def walk_body(body: list[ast.stmt], parent: str | None = None) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                add_node(node, node.name, "class", parent)
                walk_body(node.body, node.name if parent is None else f"{parent}.{node.name}")
            elif isinstance(node, ast.AsyncFunctionDef):
                add_node(node, node.name, "method" if parent else "async_function", parent)
            elif isinstance(node, ast.FunctionDef):
                add_node(node, node.name, "method" if parent else "function", parent)
            elif parent is None and isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        add_node(node, target.id, "variable", None)
            elif parent is None and isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                add_node(node, node.target.id, "variable", None)

    walk_body(tree.body)
    return sorted(symbols, key=lambda item: (item.start_line, item.qualified_name))


def _kind_from_signature_worker(signature: str) -> str:
    stripped = signature.lstrip()
    if stripped.startswith("class "):
        return "class"
    if stripped.startswith(("interface ", "type ")):
        return "type"
    if stripped.startswith(("function ", "func ", "fn ")):
        return "function"
    if stripped.startswith(("struct ", "enum ", "trait ")):
        return "class"
    return "variable"


# tree-sitter definition node kind -> symbol kind. Signature-prefix guessing
# (_kind_from_signature_worker) cannot classify keyword-less definitions:
# every C function (`static int submit_bio(...)`) fell through to "variable",
# so _explore_priority's definitional-kind preference buried the real
# definitions under struct-bearing headers. The tags layer knows the exact
# node kind; trust it, and fall back to the signature heuristic only when it
# is absent (regex-tag languages, stale tag caches).
_TAG_NODE_SYMBOL_KINDS: dict[str, str] = {
    # C / C++
    "function_definition": "function",
    "preproc_function_def": "function",  # function-like macro: callable surface
    "preproc_def": "constant",
    "struct_specifier": "struct",
    "union_specifier": "struct",
    "class_specifier": "class",
    "enum_specifier": "enum",
    "type_definition": "type_alias",
    "namespace_definition": "namespace",
    "field_declaration": "variable",
    "declaration": "variable",
    # Go / Java (method_declaration shared)
    "function_declaration": "function",
    "method_declaration": "method",
    "type_declaration": "type_alias",
    "const_declaration": "constant",
    "var_declaration": "variable",
    # Rust
    "function_item": "function",
    "function_signature_item": "function",
    "struct_item": "struct",
    "enum_item": "enum",
    "trait_item": "trait",
    "impl_item": "class",
    "mod_item": "namespace",
    "type_item": "type_alias",
    "const_item": "constant",
    "static_item": "variable",
    "macro_definition": "function",
    # Java
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "class",
    "annotation_type_declaration": "interface",
    "constructor_declaration": "method",
    # Ruby
    "class": "class",
    "module": "namespace",
    "method": "method",
    "singleton_method": "method",
}


def _extract_tag_symbols_worker(
    path: Path, source: str, language: str, tags: list[Tag] | None = None
) -> list[_ExtractedSymbol]:
    del language
    if tags is None:
        try:
            tags = extract_tags(path)
        except (OSError, SyntaxError):
            return []
    tags = [tag for tag in tags if tag.kind == "definition"]
    offsets = _line_offsets(source)
    lines = source.splitlines()
    sorted_tags = sorted(tags, key=lambda tag: (tag.line, tag.name))
    symbols: list[_ExtractedSymbol] = []
    for index, tag in enumerate(sorted_tags):
        start_line = max(1, tag.line)
        # Until-next-tag spans, DELIBERATELY slack: the doc comment above
        # definition N+1 lands in N's body, and the gap text still matches
        # file-level retrieval (measured: exact per-node extents cost linux
        # NL-query MRR by dropping inter-definition comments from symbol FTS).
        # The LAST definition extends to end-of-file -- the same slack -- not
        # to its own start line, which excluded the final function of every
        # tags-language file from symbol FTS entirely.
        next_line = sorted_tags[index + 1].line - 1 if index + 1 < len(sorted_tags) else len(lines)
        end_line = max(start_line, min(next_line, len(lines)))
        start_byte = offsets[start_line - 1] if start_line - 1 < len(offsets) else tag.byte_range[0]
        end_byte = offsets[end_line] if end_line < len(offsets) else tag.byte_range[1]
        signature = lines[start_line - 1].strip() if start_line <= len(lines) else tag.name
        node_kind = getattr(tag, "node_kind", None)
        symbols.append(
            _ExtractedSymbol(
                name=tag.name,
                qualified_name=tag.name,
                kind=_TAG_NODE_SYMBOL_KINDS.get(node_kind or "") or _kind_from_signature_worker(signature),
                signature=signature,
                start_byte=start_byte,
                end_byte=max(start_byte, end_byte),
                start_line=start_line,
                end_line=end_line,
            )
        )
    return symbols


def _extract_tag_reference_index_worker(
    rel: str,
    source: str,
    tags: list[Tag],
    symbols: list[_ExtractedSymbol],
) -> tuple[list[_IndexedReference], list[_IndexedCallEdge]]:
    """Index reference and call tags for any tree-sitter language.

    Mirrors the Python AST reference worker, reusing the tree-sitter tags already
    parsed for symbol extraction, so query-time find_references is a pure index
    lookup instead of re-parsing the whole repo. `call` tags become call_edges
    (caller = the enclosing function-like symbol), feeding call-graph PageRank
    and caller-count popularity for every tags-indexed language -- previously
    Python-only, which left the call graph empty on C/C++/Go/Rust/Java/Ruby.
    """
    lines = source.splitlines()
    _CALLER_KINDS = {"function", "async_function", "method"}

    def containing(line: int, kinds: set[str] | None = None) -> _ExtractedSymbol | None:
        candidates = [
            sym for sym in symbols if sym.start_line <= line <= sym.end_line and (kinds is None or sym.kind in kinds)
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (item.end_line - item.start_line, -item.start_line))[0]

    references: list[_IndexedReference] = []
    call_edges: list[_IndexedCallEdge] = []
    seen: set[tuple[str, int, int]] = set()
    seen_edges: set[tuple[str, int, int, str]] = set()
    for tag in tags:
        if tag.kind == "call":
            line = tag.line
            if line <= 0:
                continue
            caller = containing(line, _CALLER_KINDS)
            if caller is None:
                continue  # top-level call (module init, macro at file scope): no caller
            line_text = lines[line - 1] if 1 <= line <= len(lines) else ""
            column = max(1, line_text.find(tag.name) + 1) if line_text else 1
            edge_key = (caller.qualified_name, line, column, tag.name)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            call_edges.append(
                _IndexedCallEdge(
                    caller_symbol_name=caller.name,
                    caller_qualified_name=caller.qualified_name,
                    caller_file_path=rel,
                    caller_start_line=caller.start_line,
                    caller_end_line=caller.end_line,
                    callee_name=tag.name,
                    call_line=line,
                    call_column=column,
                    snippet=line_text.strip(),
                )
            )
            continue
        if tag.kind != "reference":
            continue
        name = tag.name
        line = tag.line
        if line <= 0:
            continue
        line_text = lines[line - 1] if 1 <= line <= len(lines) else ""
        column = max(1, line_text.find(name) + 1) if line_text else 1
        key = (name, line, column)
        if key in seen:
            continue
        seen.add(key)
        enclosing = containing(line)
        references.append(
            _IndexedReference(
                file_path=rel,
                symbol_name=name,
                line=line,
                column=column,
                end_column=column + len(name) - 1,
                enclosing_symbol_name=enclosing.name if enclosing else None,
                enclosing_qualified_name=enclosing.qualified_name if enclosing else None,
                snippet=line_text.strip(),
            )
        )
    return references, call_edges


def _python_call_name_worker(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _python_call_name_worker(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _python_call_name_worker(node.func)
    return None


class _ReferenceVisitor(ast.NodeVisitor):
    """Module-level visitor (mypyc cannot compile nested classes)."""

    def __init__(
        self,
        rel: str,
        add_reference: Callable[[str, ast.AST], None],
        containing_symbol: Callable[[int], _ExtractedSymbol | None],
        snippet_for: Callable[[int], str],
        seen_edges: set[tuple[str, int, int, str]],
        call_edges: list[_IndexedCallEdge],
    ) -> None:
        self._rel = rel
        self._add_reference = add_reference
        self._containing_symbol = containing_symbol
        self._snippet_for = snippet_for
        self._seen_edges = seen_edges
        self._call_edges = call_edges

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._add_reference(node.id, node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._add_reference(node.attr, node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _python_call_name_worker(node.func)
        caller = self._containing_symbol(int(getattr(node, "lineno", 0) or 0))
        if callee and caller is not None:
            line = int(getattr(node, "lineno", caller.start_line) or caller.start_line)
            column = int(getattr(node, "col_offset", 0) or 0) + 1
            edge_key = (caller.qualified_name, line, column, callee)
            if edge_key not in self._seen_edges:
                self._seen_edges.add(edge_key)
                self._call_edges.append(
                    _IndexedCallEdge(
                        caller_symbol_name=caller.name,
                        caller_qualified_name=caller.qualified_name,
                        caller_file_path=self._rel,
                        caller_start_line=caller.start_line,
                        caller_end_line=caller.end_line,
                        callee_name=callee,
                        call_line=line,
                        call_column=column,
                        snippet=self._snippet_for(line),
                    )
                )
        self.generic_visit(node)


def _extract_python_reference_index_worker(
    rel: str,
    source: str,
    symbols: list[_ExtractedSymbol],
    tree: ast.Module | None = None,
) -> tuple[list[_IndexedReference], list[_IndexedCallEdge]]:
    if tree is None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return [], []

    lines = source.splitlines()
    references: list[_IndexedReference] = []
    call_edges: list[_IndexedCallEdge] = []
    seen_refs: set[tuple[str, int, int, str | None]] = set()
    seen_edges: set[tuple[str, int, int, str]] = set()

    def snippet_for(line: int) -> str:
        return lines[line - 1].strip() if 1 <= line <= len(lines) else ""

    def containing_symbol(line: int) -> _ExtractedSymbol | None:
        candidates = [
            sym
            for sym in symbols
            if sym.start_line <= line <= sym.end_line and sym.kind in {"function", "async_function", "method", "class"}
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (item.end_line - item.start_line, -item.start_line))[0]

    def add_reference(name: str, node: ast.AST) -> None:
        line = int(getattr(node, "lineno", 0) or 0)
        if line <= 0:
            return
        column = int(getattr(node, "col_offset", 0) or 0) + 1
        end_column = int(getattr(node, "end_col_offset", column + len(name) - 1) or (column + len(name) - 1))
        enclosing = containing_symbol(line)
        key = (name, line, column, enclosing.qualified_name if enclosing else None)
        if key in seen_refs:
            return
        seen_refs.add(key)
        references.append(
            _IndexedReference(
                file_path=rel,
                symbol_name=name,
                line=line,
                column=column,
                end_column=max(column, end_column),
                enclosing_symbol_name=enclosing.name if enclosing else None,
                enclosing_qualified_name=enclosing.qualified_name if enclosing else None,
                snippet=snippet_for(line),
            )
        )

    _ReferenceVisitor(rel, add_reference, containing_symbol, snippet_for, seen_edges, call_edges).visit(tree)
    return references, call_edges


@lru_cache(maxsize=8192)
def _resolve_python_module_worker(repo_root: Path, base: Path, module: str) -> str | None:
    parts = module.split(".")
    search_bases: list[Path] = []
    for candidate in [base, *base.parents, repo_root, repo_root / "src"]:
        resolved = candidate.resolve()
        if resolved not in search_bases:
            search_bases.append(resolved)
    for search_base in search_bases:
        candidate = search_base / Path(*parts).with_suffix(".py")
        if candidate.is_file():
            return _safe_relpath(repo_root, candidate)
        package = search_base / Path(*parts) / "__init__.py"
        if package.is_file():
            return _safe_relpath(repo_root, package)
        src_candidate = repo_root / "src" / Path(*parts).with_suffix(".py")
        if src_candidate.is_file():
            return _safe_relpath(repo_root, src_candidate)
        src_package = repo_root / "src" / Path(*parts) / "__init__.py"
        if src_package.is_file():
            return _safe_relpath(repo_root, src_package)
    return None


def _resolve_relative_module_worker(repo_root: Path, base: Path, raw: str, suffixes: list[str]) -> str | None:
    candidate_base = (base / raw).resolve()
    candidates: list[Path] = []
    if candidate_base.suffix:
        candidates.append(candidate_base)
    else:
        candidates.extend(candidate_base.with_suffix(suffix) for suffix in suffixes)
        candidates.extend(candidate_base / f"index{suffix}" for suffix in suffixes)
        candidates.extend(candidate_base / f"mod{suffix}" for suffix in suffixes)
    for candidate in candidates:
        if candidate.is_file():
            return _safe_relpath(repo_root, candidate)
    return None


def _python_imports_worker(
    repo_root: Path, path: Path, source: str, tree: ast.Module | None = None
) -> list[tuple[str, str | None]]:
    if tree is None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
    imports: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, _resolve_python_module_worker(repo_root, path.parent, alias.name)))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.module, _resolve_python_module_worker(repo_root, path.parent, node.module)))
    return imports


def _javascript_imports_worker(repo_root: Path, path: Path, source: str) -> list[tuple[str, str | None]]:
    imports: list[tuple[str, str | None]] = []
    for match in _JS_IMPORT_RE.finditer(source):
        raw = next(group for group in match.groups() if group)
        target = None
        if raw.startswith("."):
            target = _resolve_relative_module_worker(repo_root, path.parent, raw, [".ts", ".tsx", ".js", ".jsx"])
        imports.append((raw, target))
    return imports


class _SourceFileEventHandler(FileSystemEventHandler):
    """Watchdog event handler that schedules a debounced reindex on source-file changes.

    Filters events by file extension and .gitignore rules (loaded recursively).
    Holds a weak reference to the engine so it does not prevent GC.
    """

    def __init__(self, token: Any) -> None:
        super().__init__()
        # ``token`` is a lightweight non-native SimpleNamespace whose ``.engine``
        # strong-refs the (mypyc-native, non-weakreffable) engine. Holding a
        # weakref to the token keeps the watchdog Observer thread from pinning the
        # engine: the engine<->token cycle is reclaimed by the cyclic GC once the
        # engine has no external referents (and is broken on watcher stop).
        self._engine_ref = weakref.ref(token)

    def dispatch(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src_path = os.fsdecode(event.src_path)  # watchdog may hand back bytes
        # Fast path: skip non-source extensions before any other work.
        ext = Path(src_path).suffix.lower()
        if ext not in _WATCHER_SOURCE_EXTENSIONS:
            return
        # Check against .gitignore rules (loaded and cached by the engine).
        token = self._engine_ref()
        engine = token.engine if token is not None else None
        if engine is not None and engine._watcher_path_is_ignored(src_path):
            return
        if engine is not None:
            engine._notify_watcher_event()


def _watcher_load_gitignore_patterns(repo_root: Path) -> Any | None:
    """Load all .gitignore files under *repo_root* and return a pathspec PathSpec.

    Returns None when pathspec is not installed or no .gitignore files are found.
    """
    try:
        import pathspec
    except ImportError:
        return None
    try:
        # Also check the user's global gitignore (~/.config/git/ignore or
        # core.excludesFile) for patterns like .DS_Store, *.pyc, etc.
        global_patterns: list[str] = []
        core_excludes = _git_core_excludes_file()
        if core_excludes:
            try:
                global_patterns.extend(
                    line.strip()
                    for line in Path(core_excludes).read_text("utf-8").splitlines()
                    if line.strip() and not line.startswith("#")
                )
            except OSError:
                pass

        gitignore_files = list(repo_root.rglob(".gitignore"))
        if not gitignore_files and not global_patterns:
            return None

        spec_lines: list[str] = []
        # Global gitignore patterns apply at the repo root.
        for pat in global_patterns:
            spec_lines.append(pat)

        for gi_path in gitignore_files:
            try:
                rel_dir = gi_path.parent.relative_to(repo_root).as_posix()
                for line in gi_path.read_text("utf-8").splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    # PathSpec expects patterns relative to root; prefix with
                    # the directory of the .gitignore file.
                    if rel_dir == ".":
                        spec_lines.append(stripped)
                    else:
                        spec_lines.append(f"/{rel_dir}/{stripped}")
            except (OSError, ValueError):
                continue

        return pathspec.PathSpec.from_lines("gitignore", spec_lines)
    except (OSError, ValueError):
        logger.debug("Failed to load gitignore patterns", exc_info=True)
        return None


def _git_core_excludes_file() -> str | None:
    """Return the path to the user's global gitignore file, or None."""
    try:
        result = subprocess.run(
            ["git", "config", "--global", "--get", "core.excludesFile"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    # Fallback: XDG default location
    xdg = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    candidate = Path(xdg) / "git" / "ignore"
    return str(candidate) if candidate.is_file() else None


def _watcher_check_ignored_fast(path: str, hard_skip_dirs: frozenset[str]) -> bool:
    """Pure-string check against the hard-coded skip directories (fast path)."""
    for part in path.replace("\\", "/").split("/"):
        if part in hard_skip_dirs:
            return True
    return False


# ---------------------------------------------------------------------------
# Hybrid explore fusion (v6) — module-level helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _HefQueryPlan:
    intent: str
    definitions: tuple[tuple[str, str], ...]
    identifiers: tuple[str, ...]
    anchors: tuple[str, ...]
    terms: tuple[str, ...]
    literals: tuple[str, ...]
    wants_tests: bool
    wants_auxiliary: bool


def _hef_is_code_shaped(token: str) -> bool:
    return (
        "_" in token
        or token.isupper()
        or any(ch.isupper() for ch in token[1:])
        or token.startswith("__")
        or token.endswith("__")
    )


def _hef_dedupe(values: list[str], *, limit: int) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = value.strip()
        low = value.lower()
        if not value or low in seen:
            continue
        seen.add(low)
        out.append(value)
        if len(out) >= limit:
            break
    return tuple(out)


def _hef_bare_alternative(segment: str) -> str | None:
    segment = segment.strip()
    segment = re.sub(r"\\[bBAZz]$", "", segment)
    segment = re.sub(r"^\^|\$$", "", segment)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", segment):
        return segment
    return None


def _hef_path_parts(file_path: str) -> set[str]:
    return {part for part in re.split(r"[/._-]+", file_path.lower()) if part}


def _hef_fts_phrase(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


# ── Selective Zoekt gate ──────────────────────────────────────────────────────
# Per-query A/B over the retrieval benchmark showed the *broad* Zoekt channel
# (full-repo trigram, up to 96 files) earns its keep on two query shapes and
# only adds noise elsewhere:
#   * regex/pattern queries -- FTS5 cannot evaluate `a|b`, `.*`, `[xy]`, escaped
#     metacharacters at all, so Zoekt is the *only* engine that can serve them.
#   * multi-word phrase queries -- full-text recall surfaces files the symbol
#     index buries (content/strings/comments, not just symbols).
# On plain single-identifier lookups the broad channel mostly displaces a file
# that lexical+centrality already ranked #1 (the observed 1->2 regressions), so
# we suppress it there and let the *targeted* per-anchor Zoekt channel (which
# does an identifier-scoped lookup) handle the symbol.  Telemetry on every call
# records the decision so the gate can be tuned from real traffic.
_ZOEKT_REGEX_META = re.compile(r"[|()\[\]{}*+?^$\\]")
# Aggregate fire-rate counter, keyed by (decision, reason). Inspectable in-process
# (e.g. benchmark harness) and cheap; reset with _ZOEKT_GATE_COUNTS.clear().
_ZOEKT_GATE_COUNTS: dict[tuple[str, str], int] = {}


def _zoekt_broad_gate(query: str) -> tuple[bool, str]:
    """Return ``(admit, reason)`` for the broad Zoekt channel on *query*."""
    q = query.strip()
    if _ZOEKT_REGEX_META.search(q):
        return True, "regex"
    if len(q.split()) >= 2:
        return True, "multiword"
    return False, "plain_identifier"


# ── Symbol-exact fast path ────────────────────────────────────────────────────
# When a definition/symbol-intent query resolves to a high-confidence exact
# definition in the symbol index, the anchor-Zoekt and line-FTS recall channels
# are pure overhead: profiling shows the line-FTS scan dominates explore's
# critical path (~2/3 of wall clock on definition golds; it is what the 200ms
# fusion deadline waits on), while the exact channel answers in ~1-2ms.  The
# gate decides — from the exact channel's own confidence signals — whether those
# channels can be skipped entirely.  Deterministic by design: skipped channels
# are never submitted, so fusion sees the same inputs on every run rather than
# "whatever happened to finish before the deadline".
# Disable via LEMONCROW_HEF_FAST_PATH=0 (benchmark ablation).

# Intent-aware confidence floor for the top exact hit. The exact channel's
# confidence formula caps at 1.0 for definition intent (kind coverage counts)
# but at 0.70 for bare-identifier symbol intent (no `def`/`class` kinds to
# match), so the two intents need different floors.
_HEF_FAST_PATH_MIN_CONFIDENCE: dict[str, float] = {
    "definition": 0.72,
    "symbol": 0.65,
    # "code" = identifier/alternation queries without a def/class shape. The
    # exact channel can be decisive for these too, but the floor is stricter:
    # content-style golds ("which files CONTAIN this") ride this intent, so a
    # merely-good exact hit must not suppress the line channel.
    "code": 0.80,
}
# Collision guard: the rarest matched name must be defined in at most this many
# files. A collision-heavy name ('save', 'get') matches dozens of files at equal
# confidence — ranking those needs the full multi-channel pipeline.
_HEF_FAST_PATH_MAX_DF = 4
# Ambiguity guard: the top exact hit must be separated from the runner-up by at
# least this much confidence. Same-name definitions in sibling files (e.g. a
# `client` fixture in three conftest.py files) tie at equal confidence — the
# exact channel cannot order them; only the full pipeline's centrality/lexical
# signals can, so ties are never decisive.
_HEF_FAST_PATH_MIN_MARGIN = 0.05
# Fire-rate counter keyed by (decisive, intent) — same pattern as
# _ZOEKT_GATE_COUNTS; inspectable in-process by the benchmark harness.
_HEF_FAST_PATH_COUNTS: dict[tuple[bool, str], int] = {}
# Regex-shaped queries (escapes, groups, char classes, quantifiers — anything
# beyond plain `a|b` alternation) are asking for CONTENT matches; line-FTS is
# precisely the channel that serves those, so they are never decisive.
_HEF_FAST_PATH_REGEX_META = re.compile(r"[()\[\]{}*+?^$\\]")


def _hef_fast_path_enabled() -> bool:
    value = os.environ.get("LEMONCROW_HEF_FAST_PATH")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _hef_fast_path_intents() -> set[str]:
    """Intents eligible for the decisive-exact fast path.

    Default includes "code" (identifier/alternation queries) on top of the
    original definition/symbol pair -- guarded by the stricter 0.80 floor in
    _HEF_FAST_PATH_MIN_CONFIDENCE plus all the existing decisiveness checks
    (name-shaped query, definition hit, df cap, runner-up margin). Override
    with LEMONCROW_HEF_FAST_PATH_INTENTS=definition,symbol to restore the old
    scope for A/B measurement.
    """
    raw = os.environ.get("LEMONCROW_HEF_FAST_PATH_INTENTS")
    if raw is None:
        return {"definition", "symbol", "code"}
    return {part.strip() for part in raw.split(",") if part.strip()}


def _hef_exact_is_decisive(
    query: str,
    plan: _HefQueryPlan,
    exact_files: list[str],
    exact_details: dict[str, dict[str, Any]],
) -> bool:
    """True when the exact-symbol channel alone confidently resolves *plan*.

    Requires the query to be name-shaped (no regex metacharacters beyond `|`
    alternation) and the top hit to (a) actually DEFINE a queried name (not
    merely bind a variable of that name), (b) match on a rare name (df-capped),
    and (c) clear the intent-aware confidence floor.
    """
    if not exact_files:
        return False
    if _HEF_FAST_PATH_REGEX_META.search(query):
        return False
    threshold = _HEF_FAST_PATH_MIN_CONFIDENCE.get(plan.intent)
    if threshold is None:
        return False
    top = exact_details.get(exact_files[0]) or {}
    if not top.get("definition_tokens"):
        return False
    if int(top.get("best_df", 1_000_000)) > _HEF_FAST_PATH_MAX_DF:
        return False
    if len(exact_files) > 1:
        runner_up = exact_details.get(exact_files[1]) or {}
        margin = float(top.get("confidence", 0.0)) - float(runner_up.get("confidence", 0.0))
        if margin < _HEF_FAST_PATH_MIN_MARGIN:
            return False
    return float(top.get("confidence", 0.0)) >= threshold


def _zoekt_gate_enforced() -> bool:
    """Whether the broad-Zoekt EXECUTION gate is active (default ON).

    When active, the broad Zoekt search is SKIPPED for plain single-identifier
    queries (FTS symbols + the targeted per-anchor Zoekt channel carry them).
    A benchmark A/B showed this cuts mean explore latency ~12pct (p95 ~10pct)
    with no MRR loss (slight Hit@1 gain), so it ships enforced. Opt out with
    LEMONCROW_ZOEKT_GATE=0. (The earlier observe-only verdict was for a *fusion*-
    only gate that could not save latency because the search had already run.)
    """
    return os.environ.get("LEMONCROW_ZOEKT_GATE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _zoekt_gate_record(
    query: str, intent: str, decision: bool, reason: str, zoekt_n: int, anchor_n: int, *, enforced: bool
) -> None:
    """Telemeter a Zoekt gate decision (aggregate counter + opt-in per-query log)."""
    key = ("admit" if decision else "suppress", reason)
    _ZOEKT_GATE_COUNTS[key] = _ZOEKT_GATE_COUNTS.get(key, 0) + 1
    if os.environ.get("LEMONCROW_ZOEKT_GATE_LOG"):
        print(
            f"[zoekt-gate] decision={decision} enforced={enforced} reason={reason} intent={intent} "
            f"zoekt_n={zoekt_n} anchor_n={anchor_n} q={query[:60]!r}",
            file=sys.stderr,
            flush=True,
        )


def _close_sqlite_connections(conns: list[sqlite3.Connection], lock: threading.Lock) -> None:
    """Best-effort close of every registered SQLite connection (finalizer target)."""
    with lock:
        while conns:
            conn = conns.pop()
            with contextlib.suppress(Exception):
                conn.close()


class CodeContextEngine:
    """Local code intelligence using tree-sitter tags, SQLite FTS5, rg, and repo-map ranking."""

    def __init__(
        self,
        repo_root: str | Path = ".",
        *,
        db_path: str | Path | None = None,
        autosync_enabled: bool | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.repo_id = _repo_id(self.repo_root)
        self.db_path = Path(db_path).resolve() if db_path is not None else _default_db_path(self.repo_root)
        self._db_lock = _shared_db_lock(self.db_path)
        self._schema_ready = False
        self._cache = RetrievalCache(self.db_path)
        self._budget = BudgetPacker()
        self._semantic_ranker = SemanticSearchRanker(self.repo_root, store_root=default_store_root())
        self._search_reranker = SearchReranker()
        # G4/N5/N16: persistent ANN over per-symbol embeddings. Opt-in via
        # LEMONCROW_ANN_RETRIEVAL; with the flag off this object is never
        # consulted and the semantic path is byte-identical to today.
        self._ann_symbol_index = SymbolAnnIndex(self.repo_id)
        # In-memory cache of the parsed symbol vectors, keyed by
        # (embedder_name, dim, index_version). Loading + JSON-decoding the whole
        # store on every semantic query is the dominant hot-path cost otherwise; a
        # reindex bumps index_version and invalidates this.
        self._ann_vectors_cache: tuple[tuple[str, int, int], list[str], Any] | None = None
        # Single-flight guard for the small-repo "load all + cache" branch of
        # _search_symbols_semantic_ann: without it, N queries arriving while a
        # cold load is in flight each start their OWN redundant multi-GB
        # reload instead of waiting for the one already warming the cache --
        # confirmed via a real run this alone turned a flat ~8s/query cost
        # into a 12.2s mean / 60.5s p100 pileup once a caller-side deadline
        # made concurrent retries possible.
        self._ann_matrix_loading_lock = threading.Lock()
        # Set once _maybe_warm_ann_cache has kicked off its background thread,
        # so _ensure_indexed (which runs on every tool call) doesn't spawn a
        # new one every time -- see _maybe_warm_ann_cache for why this exists.
        self._ann_warm_started: bool = False
        # sqlite-vector TurboQuant in-DB ANN: replaces the numpy matrix scan on
        # large corpora (linux: 1.24Mx1536 = 7.5 GB matrix → OOM). The extension
        # only operates on a connection's *main* schema, so a dedicated per-thread
        # direct connection to vectors.sqlite is used (bare 'symbol_vectors')
        # rather than the engine's attached-vectors connection. Falls back to the
        # numpy path transparently when the extension is unavailable.
        self._sqlite_vec_tls = threading.local()
        self._sqlite_vec_lock = threading.Lock()
        self._sqlite_vec_disabled = False
        # Registry of every per-thread vectors.sqlite connection this engine opens,
        # so disposal can close them all (a thread-local alone only reaches the
        # current thread's connection). Closed via weakref.finalize when the engine
        # is collected / at interpreter exit.
        self._sqlite_vec_conns: list[sqlite3.Connection] = []
        # CodeContextEngine compiles to a mypyc-native class, which has no
        # __weakref__ slot and so cannot be a weakref target. Attach GC finalizers
        # to a lightweight non-native sentinel that lives and dies with the engine.
        # The sentinel holds no strong ref back to the engine, so the engine stays
        # refcount-collectable and cleanup fires promptly on collection -- exactly
        # as ``weakref.finalize(self, ...)`` did. When a finalizer callback strong-
        # refs the engine (e.g. the autosync bound method), it pins the engine (and
        # thus the sentinel) until interpreter exit, matching the prior behaviour.
        self._gc_sentinel = WeakRefToken()
        weakref.finalize(self._gc_sentinel, _close_sqlite_connections, self._sqlite_vec_conns, self._sqlite_vec_lock)
        self.intel_store = SymbolIntelStore(
            cache=self._cache,
            packer=self._budget,
            local_search=self._search_symbols_local,
            local_get_symbol=self._get_symbol_local,
            local_find_references=self._find_references_local,
            local_find_callers=self._find_callers_local,
            local_find_callees=self._find_callees_local,
        )
        self._deleted_history_search_adapter: DeletedHistorySearchAdapter | None = None
        # Autosync disabled for one-shot CLI commands, enabled for services/daemons.
        # LEMONCROW_CODE_AUTOSYNC=0 turns it off when the caller doesn't decide
        # (same contract as LEMONCROW_CODE_FILE_WATCHER): required for serving
        # FROZEN snapshots -- e.g. benchmark indexes -- where the drift check
        # would otherwise stat-walk the tree and kick off a whole-repo reindex
        # subprocess that rewrites the snapshot underneath live queries
        # (observed: a 2h+ single-worker reindex of the linux snapshot pinning
        # a core and holding the WAL hot, collapsing eval throughput to 1.8/s).
        _autosync_env = os.getenv("LEMONCROW_CODE_AUTOSYNC")
        self._autosync_enabled = (
            autosync_enabled
            if autosync_enabled is not None
            else _autosync_env is None or _autosync_env.strip().lower() not in {"0", "false", "no", "off"}
        )
        self._autosync_debounce_ms = self._parse_autosync_debounce(os.getenv("LEMONCROW_CODE_AUTOSYNC_DEBOUNCE_MS"))
        self._autosync_poll_ms = self._parse_autosync_poll_ms(os.getenv("LEMONCROW_CODE_AUTOSYNC_POLL_MS"))
        self._autosync_state = "idle"
        self._autosync_signature: str | None = None
        self._autosync_last_sync_ms = 0
        self._autosync_last_event_at: str | None = None
        self._autosync_pending_events = 0
        self._autosync_reindex_count = 0
        self._autosync_history: list[dict[str, Any]] = []
        # Counts completed tool calls; used to pace the periodic heap trim.
        self._tool_call_count: int = 0
        self._last_heap_trim_ts: float = 0.0
        # (dev, ino) of the three secondary DB files when their schemas were
        # last initialised; skips the per-call DDL when unchanged.
        self._secondary_schemas_key: tuple[tuple[int, int], ...] | None = None
        self._secondary_schema_lock = threading.Lock()
        self._autosync_lock = threading.RLock()
        self._autosync_stop = threading.Event()
        self._autosync_thread: threading.Thread | None = None
        self._lineage_thread: threading.Thread | None = None
        self._lineage_lock = threading.Lock()
        self._index_ready_cached = False
        # Cache the engine_state index_version so a single tool call (which probes
        # it ~once per sub-query) does not reopen the DB and re-query for a value
        # that only changes on reindex. Invalidated in _bump_index_version.
        self._index_version_cached: int | None = None
        # G6/N16: symbol-level call-graph centrality cache, keyed to the index
        # version so a graph mutation (any reindex bumps index_version) forces a
        # recompute and stale rankings are never served. Guarded by its own lock.
        self._centrality_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self._centrality_cache_lock = threading.Lock()
        # Connection reuse: inside a _reuse_connection() scope, _connect() returns a
        # shared per-thread connection (via _ReusedConnection) instead of opening a
        # new one + re-running PRAGMAs for every sub-query.
        self._scoped_conn_tls = threading.local()
        # File bytes cache: inside a _reuse_connection() scope, _read_file_slice()
        # caches raw file bytes so the same file is read from disk only once per
        # tool call even when multiple symbols are selected from it. Eliminates
        # N_symbols x read_bytes() cost (e.g. 5 symbols from engine.py = 5 x 18 MB
        # -> 1 x 18 MB). Cleared on scope exit; no inter-call sharing.
        self._file_cache_tls = threading.local()
        # Persistent stat-validated file bytes cache (rel path -> (mtime_ns,
        # size, bytes)). The per-call TLS cache above dedupes reads WITHIN one
        # tool call; this one carries the hot set (rerank candidate sources,
        # explore source sections) ACROSS calls. Entries are validated against
        # stat() on every access, so served bytes are always identical to disk.
        self._file_bytes_cache: dict[str, tuple[int, int, bytes]] = {}
        self._file_bytes_cache_total = 0
        self._wal_primed = False
        # FTS corpus size for IDF term pruning, keyed by index_version so a reindex
        # auto-invalidates it (count(*) over symbol_fts is ~18ms -- never per query).
        self._fts_doc_count_cache: dict[int, int] = {}
        # Explicit instance attrs (not self.__dict__ lazy-init): CodeContextEngine
        # compiles to a mypyc-native class, which has no __dict__ -- see the
        # __weakref__ comment above. self.__dict__.setdefault(...)/[...] on a
        # compiled instance raises AttributeError at runtime (only caught in pure
        # Python where the fallback __dict__ masks it).
        self._hef_anchor_cache: dict[str, list[str]] = {}
        self._line_fts_total_cache: int | None = None
        self._line_fts_df_cache: dict[str, int] = {}
        self._lineage_rebuild_full = False
        self._lineage_score_penalty: float = float(
            os.getenv("LEMONCROW_LINEAGE_COMMIT_SCORE_PENALTY", str(_LINEAGE_DEFAULT_SCORE_PENALTY))
        )
        # G7: optional churn provider. When set, it maps a candidate set of
        # symbols to a per-symbol churn score in [0, 1]. It is consulted ONLY as
        # a low-priority ranking tiebreaker (see _context_symbol_rank), never as
        # an override of match quality. It defaults to unset so ranking never
        # incurs git/blame cost in the hot path; callers/tests may inject one.
        self._churn_score_provider: Callable[[list[SymbolRecord]], dict[str, float]] | None = None
        # --- File watcher (event-driven via watchdog) ---
        self._watcher_enabled = self._parse_watcher_enabled(os.getenv("LEMONCROW_CODE_FILE_WATCHER"))
        self._watcher_debounce_ms = self._parse_watcher_debounce(os.getenv("LEMONCROW_CODE_WATCHER_DEBOUNCE_MS"))
        # `Observer` is a runtime variable (None when watchdog is absent), not
        # a valid annotation type -- keep the attribute loosely typed.
        self._file_watcher: Any = None
        self._watcher_event_handler: _SourceFileEventHandler | None = None
        # Engine-holding token handed to the watcher handler (see below); created
        # only while the watcher runs so the engine<->token cycle is short-lived.
        self._watcher_token: Any = None
        self._watcher_gitignore_spec: Any = None  # pathspec.PathSpec or None
        self._watcher_gitignore_mtime: float = 0.0  # last load mtime of gitignore files
        self._watcher_gi_mtimes: dict[str, int] = {}
        self._watcher_last_event_ms: int = 0
        # Flip the main DB to WAL NOW, before any background thread (autosync,
        # file watcher) can hold a read transaction on it. In DELETE journal
        # mode a reader overlapping a multi-db write transaction raises an
        # immediate 'database is locked' (mid-transaction BUSY skips the busy
        # handler); at construction time we are provably single-threaded, so
        # the flip cannot race. No-op when already WAL. Failure (e.g. another
        # PROCESS holds the db) falls back to the retrying flip in
        # _apply_pragmas.
        with contextlib.suppress(sqlite3.Error, OSError):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            _wal_conn = sqlite3.connect(self.db_path, timeout=5.0)
            try:
                _row = _wal_conn.execute("PRAGMA journal_mode=WAL").fetchone()
                if _row is not None and str(_row[0]).lower() == "wal":
                    self._wal_primed = True
            finally:
                _wal_conn.close()
        if self._autosync_enabled:
            if self._watcher_enabled and Observer is not None:
                self._start_file_watcher()
            self._start_autosync_worker()

    def index_repo(
        self,
        *,
        include_globs: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        force: bool = True,
        block: bool = True,
        require_lock: bool = False,
        steal: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> IndexStats:
        """Build or refresh the persistent symbol/import index for this repository.

        Args:
            include_globs: Glob patterns to include (default source-code patterns).
            exclude_globs: Glob patterns to exclude.
            force: If True (default), wipe and rebuild the full index. Pass
                ``force=False`` for an incremental update (skip unchanged files).
            block: If True (default), wait for the cross-process index-write lock.
                Pass ``block=False`` to skip indexing (returning the current
                snapshot) when another process is already rebuilding.
            require_lock: If True, raise ``IndexLockTimeout`` when the lock cannot
                be acquired instead of silently returning a stale snapshot. Use
                for explicit, must-succeed builds (e.g. the CLI prewarm).
            steal: Deprecated compatibility flag. A live lock is never bypassed
                because its holder may still own SQLite writer state.
            progress_callback: Optional callback ``fn(current, total)`` called
                after each file is processed during indexing.
        """
        if self._autosync_enabled:
            with self._db_lock, self._autosync_lock:
                with self._index_write_lock(block=block, steal=steal) as acquired:
                    if not acquired:
                        if require_lock:
                            raise IndexLockTimeout(self.db_path)
                        # Another process holds the cross-process index-write lock.
                        # Don't pile on a redundant concurrent rebuild — return the
                        # current on-disk snapshot and let the other writer finish.
                        return self._current_index_stats()
                    return self._index_repo_unsafe(
                        include_globs=include_globs,
                        exclude_globs=exclude_globs,
                        force=force,
                        progress_callback=progress_callback,
                    )
        else:
            # CLI mode: no autosync, skip the autosync lock to avoid contention
            # with background services that have autosync enabled.
            with self._db_lock:
                with self._index_write_lock(block=block, steal=steal) as acquired:
                    if not acquired:
                        if require_lock:
                            raise IndexLockTimeout(self.db_path)
                        return self._current_index_stats()
                    return self._index_repo_unsafe(
                        include_globs=include_globs,
                        exclude_globs=exclude_globs,
                        force=force,
                        progress_callback=progress_callback,
                    )

    @contextlib.contextmanager
    def _index_write_lock(self, *, block: bool, steal: bool = False) -> Iterator[bool]:
        """Serialize index writes across separate processes sharing this DB.

        ``_db_lock`` only guards threads inside one process; multiple ``lc``
        processes (MCP servers, background service, CLI) each hold their own. This
        advisory ``flock`` ensures only one of them rebuilds the index at a time.
        Yields ``True`` when the lock was acquired (always so when ``block`` is
        True) and ``False`` when a non-blocking attempt found another process
        already indexing.

        *steal* is retained for API compatibility but never bypasses a live
        holder. Unlinking an actively flocked path creates a second lock inode
        while the first process may still own SQLite writer state, allowing two
        indexers into the critical section and causing ``database is locked``.
        """
        if fcntl is None:  # pragma: no cover - non-POSIX platforms
            yield True
            return
        lock_path = self.db_path.with_name(self.db_path.name + ".indexlock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        def _open_lock_fd() -> int:
            return os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)

        fd = _open_lock_fd()
        acquired = False
        try:
            if not block:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except OSError:
                    acquired = False
            else:
                # Poll with LOCK_NB so we don't block the process forever.
                # Default 10s: if another LemonCrow process holds the lock (e.g. a
                # running MCP server) we skip indexing rather than hang. Override
                # via LEMONCROW_INDEX_LOCK_TIMEOUT_S for long prewarm builds that
                # must win the lock before serving.
                _LOCK_TIMEOUT = _index_lock_timeout_s()
                _POLL_INTERVAL = 0.5
                deadline = time.monotonic() + _LOCK_TIMEOUT
                logging.info("Waiting for index write lock (another process may be indexing)...")
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        logging.debug("Index write lock acquired")
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            if steal:
                                logging.warning(
                                    "Cannot safely force the index lock: the holder still owns SQLite writer state."
                                )
                            logging.warning(
                                "Index write lock timeout after %.0fs — "
                                "another process is indexing; skipping this run.",
                                _LOCK_TIMEOUT,
                            )
                            break
                        time.sleep(_POLL_INTERVAL)
            yield acquired
        finally:
            if acquired:
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _current_index_stats(self) -> IndexStats:
        snapshot = self._index_snapshot()
        return IndexStats(
            repo_id=self.repo_id,
            repo_root=str(self.repo_root),
            db_path=str(self.db_path),
            files_indexed=int(snapshot["files_indexed"]),
            symbols_indexed=int(snapshot["symbols_indexed"]),
            imports_indexed=int(snapshot["imports_indexed"]),
            index_version=self._current_index_version(),
        )

    def _apply_file_data_batch(
        self,
        conn: sqlite3.Connection,
        results: list[_FileIndexData],
    ) -> None:
        """Batch-insert all extracted data using ``executemany`` (single writer)."""
        # --- files ---
        conn.executemany(
            """
            INSERT INTO files(repo_id, file_path, language, content_hash, size_bytes, mtime_ns, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            ON CONFLICT(repo_id, file_path) DO UPDATE SET
                language = excluded.language,
                content_hash = excluded.content_hash,
                size_bytes = excluded.size_bytes,
                mtime_ns = excluded.mtime_ns,
                indexed_at = excluded.indexed_at
            """,
            [(self.repo_id, d.rel, d.language, d.content_hash, d.size_bytes, d.mtime_ns) for d in results],
        )

        # --- symbols + FTS ---
        symbol_rows: list[
            tuple[
                str,
                str,
                str,
                str,
                str,
                str,
                str,
                str,
                int,
                int,
                int,
                int,
                str | None,
                str | None,
                str,
            ]
        ] = []
        fts_rows: list[tuple[str, str, str, str, str, str]] = []
        trigram_rows: list[tuple[str, str, str, str]] = []  # (symbol_id, name_plain, qualified_name, file_path)
        for d in results:
            for i, sym in enumerate(d.symbols):
                raw_id = f"{self.repo_id}:{d.rel}:{sym.qualified_name}:{sym.start_byte}:{d.content_hash}"
                sid = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]
                symbol_rows.append(
                    (
                        sid,
                        self.repo_id,
                        d.rel,
                        d.language,
                        sym.name,
                        sym.qualified_name,
                        sym.kind,
                        sym.signature,
                        sym.start_byte,
                        sym.end_byte,
                        sym.start_line,
                        sym.end_line,
                        sym.parent_symbol,
                        sym.doc_summary,
                        d.content_hash,
                    )
                )
                # Augment the FTS 'name' field with stripped-join bigrams and
                # the full compound form so queries for '_get_timezone_name'
                # also match the unique token 'gettimezonename', not just
                # the noisy sub-tokens 'get'/'timezone'/'name'.
                _ngrams = _ngram_tokens(sym.name)
                _fts_name = (sym.name + " " + " ".join(_ngrams)) if _ngrams else sym.name
                fts_rows.append((sid, _fts_name, sym.qualified_name, sym.signature, d.rel, d.symbol_sources[i]))
                # Trigram table uses the PLAIN name (no n-gram expansion) — the trigram
                # tokenizer handles substring matching natively, so pre-expansion only wastes
                # space. Signature is omitted: 5x size amplification, rarely unique over name.
                trigram_rows.append((sid, sym.name, sym.qualified_name, d.rel))

        conn.executemany(
            """
            INSERT OR IGNORE INTO symbols(
                symbol_id, repo_id, file_path, language, symbol_name, qualified_name, kind,
                signature, start_byte, end_byte, start_line, end_line, parent_symbol,
                doc_summary, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            symbol_rows,
        )
        conn.executemany(
            "INSERT INTO symbol_fts(symbol_id, name, qualified_name, signature, file_path, source) VALUES (?, ?, ?, ?, ?, ?)",
            fts_rows,
        )
        conn.executemany(
            "INSERT INTO symbol_trigram(symbol_id, name, qualified_name, file_path) VALUES (?, ?, ?, ?)",
            trigram_rows,
        )
        conn.executemany(
            "INSERT INTO file_path_trigram(repo_id, file_path) VALUES (?, ?)",
            [(self.repo_id, d.rel) for d in results],
        )

        # --- line text + FTS ---
        line_rows: list[tuple[str, str, int, str]] = []
        for d in results:
            line_rows.extend((self.repo_id, d.rel, line_no, text) for line_no, text in d.text_lines if text.strip())
        conn.executemany(
            "INSERT INTO file_line_fts(repo_id, file_path, line, text) VALUES (?, ?, ?, ?)",
            line_rows,
        )

        # --- imports ---
        rows: list[tuple[str, str, str, str | None]] = []
        for d in results:
            rows.extend((self.repo_id, d.rel, raw, target) for raw, target in d.imports)
        conn.executemany(
            "INSERT OR IGNORE INTO imports(repo_id, source_file, raw_import, target_file) VALUES (?, ?, ?, ?)",
            rows,
        )

        # --- references ---
        ref_rows: list[tuple[str, str, str, int, int, int, str | None, str | None, str]] = []
        for d in results:
            ref_rows.extend(
                (
                    self.repo_id,
                    r.symbol_name,
                    r.file_path,
                    r.line,
                    r.column,
                    r.end_column,
                    r.enclosing_symbol_name,
                    r.enclosing_qualified_name,
                    r.snippet,
                )
                for r in d.references
            )
        conn.executemany(
            """INSERT OR IGNORE INTO "references"(
                repo_id, symbol_name, file_path, line, column, end_column,
                enclosing_symbol_name, enclosing_qualified_name, snippet
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ref_rows,
        )

        # --- call_edges (snippet column dropped -- never read anywhere) ---
        edge_rows: list[tuple[str, str, str, str, int, int, str, str, int, int]] = []
        for d in results:
            edge_rows.extend(
                (
                    self.repo_id,
                    e.caller_symbol_name,
                    e.caller_qualified_name,
                    e.caller_file_path,
                    e.caller_start_line,
                    e.caller_end_line,
                    e.callee_name,
                    e.callee_name.rsplit(".", 1)[-1],
                    e.call_line,
                    e.call_column,
                )
                for e in d.call_edges
            )
        conn.executemany(
            """INSERT OR IGNORE INTO call_edges(
                repo_id, caller_symbol_name, caller_qualified_name, caller_file_path,
                caller_start_line, caller_end_line, callee_name, callee_short_name,
                call_line, call_column
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            edge_rows,
        )

    def _parallel_extract(
        self,
        files: list[Path],
        *,
        total: int = 0,
        progress_callback: Callable[[int, int], None] | None = None,
        source_bytes_map: dict[str, bytes] | None = None,
    ) -> list[_FileIndexData]:
        """Extract index data from *files* using ``ProcessPoolExecutor``.

        Each file is processed in a subprocess (true CPU parallelism).
        Results are sorted deterministically by relative path.
        *total* is the denominator for *progress_callback* (defaults to ``len(files)``).
        """
        max_workers = _resolve_index_max_workers()
        total_count = total or len(files)

        # Build argument tuples for the pickleable worker function
        args_list: list[tuple[str, str, bytes | None]] = []
        for path in files:
            sb = source_bytes_map.get(str(path)) if source_bytes_map else None
            args_list.append((str(self.repo_root), str(path), sb))

        # Small repos: skip the ProcessPoolExecutor entirely. Spawning fresh
        # interpreters (each re-imports the whole package) and joining the pool
        # costs a fixed ~1-2s that dwarfs the millisecond-scale parsing of a
        # handful of files. Serial in-process extraction is byte-for-byte
        # identical, just without the process churn.
        if max_workers <= 1 or len(args_list) <= _resolve_serial_extract_threshold():
            serial_results: list[_FileIndexData] = []
            for completed, args in enumerate(args_list, start=1):
                data = _process_one_file(*args)
                if data is not None:
                    serial_results.append(data)
                if progress_callback is not None:
                    progress_callback(completed, total_count)
            serial_results.sort(key=lambda r: r.rel)
            return serial_results

        # Use the shared pool (spawn context, single instance per process).
        # Workers never inherit this process's locks or open fds.
        # Chunked submission: keep at most (max_workers x 4) futures in flight
        # so input pickles and result objects don't all accumulate in the parent
        # at once — important when source_bytes_map is provided.
        # On BrokenProcessPool (worker crash/OOM), reset and retry once.
        results: list[_FileIndexData] = []
        for attempt in range(2):
            executor = _get_index_process_pool()
            try:
                backlog = max(8, max_workers * 4)
                args_iter = iter(args_list)
                pending: dict[concurrent.futures.Future[_FileIndexData | None], None] = {}
                completed_count = 0

                for args in itertools.islice(args_iter, backlog):
                    pending[executor.submit(_process_one_file, *args)] = None

                while pending:
                    done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        del pending[future]
                        data = future.result()
                        if data is not None:
                            results.append(data)
                        completed_count += 1
                        if progress_callback is not None:
                            progress_callback(completed_count, total_count)
                        next_args = next(args_iter, None)
                        if next_args is not None:
                            pending[executor.submit(_process_one_file, *next_args)] = None
                break  # success
            except concurrent.futures.process.BrokenProcessPool:
                logging.warning("Index process pool broken — resetting (attempt %d/2)", attempt + 1)
                _reset_index_process_pool()
                results.clear()
                if attempt == 1:
                    raise

        results.sort(key=lambda r: r.rel)
        return results

    def _index_repo_unsafe(
        self,
        *,
        include_globs: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        force: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> IndexStats:
        """Unlocked inner — callers must hold ``self._autosync_lock``."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        all_files = [
            path
            for path in iter_source_files(
                self.repo_root, include_globs=include_globs, progress_callback=progress_callback
            )
            if not self._excluded(path, exclude_globs or [])
        ]
        from lemoncrow.core.capabilities import licensing

        capped = False
        if not licensing.has_feature("context_engine") and len(all_files) > _FREE_TIER_MAX_FILES:
            all_files = sorted(all_files)[:_FREE_TIER_MAX_FILES]
            capped = True
            logger.warning(
                "context_engine: repo exceeds the Free-tier cap of %d files; indexing the first %d only "
                "(LemonCrow Pro removes this cap)",
                _FREE_TIER_MAX_FILES,
                _FREE_TIER_MAX_FILES,
            )
        total = len(all_files)
        if progress_callback is not None:
            progress_callback(0, total)  # Signal: discovery done, real total known

        with self._connect() as conn:
            self._init_schema(conn)
            if not force and self._stored_indexer_semantics_version(conn) != _CODE_INDEXER_SEMANTICS_VERSION:
                force = True

            if force:
                # --- Full rebuild: wipe everything, then parallel-extract + batch-write ---
                conn.execute("DELETE FROM file_line_fts")
                conn.execute("DELETE FROM symbol_fts")
                conn.execute("DELETE FROM symbol_trigram")
                conn.execute("DELETE FROM file_path_trigram")
                conn.execute("DELETE FROM symbols")
                conn.execute("DELETE FROM imports")
                conn.execute('DELETE FROM "references"')
                conn.execute("DELETE FROM call_edges")
                conn.execute("DELETE FROM files")
                # Stale vectors are never overwritten (symbol_id encodes content),
                # so a full rebuild must wipe them too. Created lazily -> guard.
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute("DELETE FROM symbol_vectors")

                results = self._parallel_extract(all_files, total=total, progress_callback=progress_callback)
                self._apply_file_data_batch(conn, results)

                index_version = self._bump_index_version(conn)
                # Refresh planner stats, then build per-symbol embeddings into the
                # persistent vector store -- same index-time work as the incremental
                # branch below, so a full/forced rebuild also produces vectors
                # (no-op unless an embedder is configured). Keeps the query hot path
                # read-only: it embeds only the query and ANN-reads these vectors.
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute("PRAGMA optimize")
                self._build_symbol_embeddings(conn, index_version)
                files_indexed = len(results)
                symbols_indexed = sum(len(r.symbols) for r in results)
                imports_indexed = sum(len(r.imports) for r in results)

            else:
                # --- Incremental: detect changes, then parallel-extract + batch-write ---
                existing = {}
                for row in conn.execute(
                    "SELECT file_path, content_hash, size_bytes, mtime_ns FROM files WHERE repo_id = ?",
                    (self.repo_id,),
                ):
                    existing[str(row["file_path"])] = (
                        str(row["content_hash"]),
                        int(row["size_bytes"]),
                        int(row["mtime_ns"] or 0),
                    )
                line_index_empty = (
                    conn.execute("SELECT 1 FROM file_line_fts WHERE repo_id = ? LIMIT 1", (self.repo_id,)).fetchone()
                    is None
                )

                to_extract: list[tuple[Path, bytes]] = []  # (path, source_bytes)
                current_paths: set[str] = set()

                for path in all_files:
                    rel = _safe_relpath(self.repo_root, path)
                    current_paths.add(rel)
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    if stat.st_size > _MAX_FILE_BYTES:
                        if rel in existing:
                            self._delete_file_index(conn, rel)
                        continue
                    previous = existing.get(rel)
                    # Fast path: a file whose (size, mtime) matches the indexed row
                    # is already current -- skip the read + sha256 entirely. This
                    # keeps the background incremental resync cheap on large repos
                    # instead of O(repo bytes) on every poll.
                    if (
                        previous is not None
                        and not line_index_empty
                        and previous[1] == int(stat.st_size)
                        and previous[2] == int(stat.st_mtime_ns)
                        and previous[2] != 0
                    ):
                        continue
                    source_bytes = path.read_bytes()
                    content_hash = _sha256_bytes(source_bytes)
                    if (
                        previous is not None
                        and not line_index_empty
                        and previous[0] == content_hash
                        and previous[1] == int(stat.st_size)
                    ):
                        # Content identical (e.g. a touch changed only mtime).
                        # Refresh the stored mtime so the next pass fast-skips,
                        # then move on without re-extracting.
                        conn.execute(
                            "UPDATE files SET mtime_ns = ? WHERE repo_id = ? AND file_path = ?",
                            (int(stat.st_mtime_ns), self.repo_id, rel),
                        )
                        continue
                    self._delete_file_index(conn, rel)
                    to_extract.append((path, source_bytes))

                removed_paths = set(existing.keys()) - current_paths
                for rel in sorted(removed_paths):
                    self._delete_file_index(conn, rel)

                if to_extract:
                    paths = [item[0] for item in to_extract]
                    source_map = {str(p): b for p, b in to_extract}
                    results = self._parallel_extract(
                        paths,
                        total=total,
                        progress_callback=progress_callback,
                        source_bytes_map=source_map,
                    )
                    self._apply_file_data_batch(conn, results)
                else:
                    results = []

                if to_extract or removed_paths:
                    index_version = self._bump_index_version(conn)
                    # Refresh query-planner statistics after a (re)index so
                    # call-graph/symbol lookups use the selective composite
                    # indexes. Without stats SQLite mis-picks the repo_id-only
                    # index for caller-keyed call_edges queries and full-scans
                    # the table on every traversal. PRAGMA optimize self-throttles
                    # (cheap on small incremental deltas, full ANALYZE when needed).
                    with contextlib.suppress(sqlite3.OperationalError):
                        conn.execute("PRAGMA optimize")
                    # Build per-symbol embeddings into the persistent vector store as
                    # part of the code index (no-op unless an embedder is configured).
                    # Doing it here keeps the query hot path read-only -- it embeds
                    # only the query and ANN-reads these prebuilt vectors.
                    self._build_symbol_embeddings(conn, index_version)
                else:
                    row = conn.execute("SELECT value FROM engine_state WHERE key = 'index_version'").fetchone()
                    index_version = int(row["value"]) if row is not None else 0
                    # File-level state is unchanged, but the embedder may have just
                    # been enabled/switched with zero symbols embedded yet -- e.g.
                    # LEMONCROW_CODE_EMBEDDER flipped on after the repo was already
                    # indexed without it. Without this, an unchanged repo's "nothing
                    # to extract" fast path would NEVER backfill embeddings, silently
                    # making semantic search a permanent no-op. Two cheap COUNT
                    # queries (no-op cost once caught up) gate the expensive full
                    # symbols scan in _build_symbol_embeddings.
                    if self._semantic_ranker.available:
                        embedder = self._semantic_ranker.embedder
                        dim = int(getattr(embedder, "dim", 0))
                        if dim > 0:
                            total = conn.execute(
                                "SELECT COUNT(*) FROM symbols WHERE repo_id = ?", (self.repo_id,)
                            ).fetchone()[0]
                            try:
                                embedded = conn.execute(
                                    "SELECT COUNT(*) FROM symbol_vectors "
                                    "WHERE repo_id = ? AND embedder_name = ? AND embedding_dim = ?",
                                    (self.repo_id, embedder.name, dim),
                                ).fetchone()[0]
                            except sqlite3.OperationalError:
                                embedded = 0
                            if embedded < total:
                                self._build_symbol_embeddings(conn, index_version)
                                # A query racing ahead of this backfill (very likely --
                                # this runs in the background autosync worker while
                                # queries keep flowing) may have already cached an
                                # EMPTY result under this exact (embedder, dim,
                                # index_version) key in _search_symbols_semantic_ann.
                                # index_version doesn't change here (by design -- see
                                # the comment on existing_stamped_ids), so that stale
                                # cache entry would never naturally invalidate and
                                # semantic search would silently stay a permanent
                                # no-op for the rest of this process's life even
                                # though real vectors now exist. Drop it so the next
                                # query re-reads the DB.
                                self._ann_vectors_cache = None

                files_indexed = len(to_extract)
                symbols_indexed = sum(len(r.symbols) for r in results)
                imports_indexed = sum(len(r.imports) for r in results)

        if force:
            # Compact all DBs after a full rebuild — DELETE + re-insert leaves free pages
            # that inflate file size until VACUUMed (e.g. 87 MB → 31 MB for pylint).
            for _vac_db in (self.db_path, self.intel_db_path, self.vectors_db_path, self.fts_db_path):
                if _vac_db.exists():
                    with contextlib.suppress(Exception):
                        _vc = sqlite3.connect(str(_vac_db))
                        _vc.execute("VACUUM")
                        _vc.close()

        with self._connect() as conn:
            self._init_schema(conn)
            self._stamp_indexer_semantics_version(conn)

        # Compute+persist the centrality map for the new index_version NOW so the
        # O(edges) power iteration is charged to indexing, never to the first
        # query. Loads the persisted map when the version is unchanged (cheap
        # no-op for incremental runs that indexed nothing).
        with contextlib.suppress(Exception):
            self._symbol_centrality_map()

        emit_product_local(
            "code_index_completed",
            repo_id=self.repo_id,
            files_indexed=files_indexed,
            symbols_indexed=symbols_indexed,
        )
        return IndexStats(
            repo_id=self.repo_id,
            repo_root=str(self.repo_root),
            db_path=str(self.db_path),
            files_indexed=files_indexed,
            symbols_indexed=symbols_indexed,
            imports_indexed=imports_indexed,
            index_version=index_version,
            capped=capped,
        )

    def _stored_indexer_semantics_version(self, conn: sqlite3.Connection) -> int | None:
        row = conn.execute("SELECT value FROM engine_state WHERE key = 'indexer_semantics_version'").fetchone()
        if row is None:
            has_indexed_files = conn.execute(
                "SELECT 1 FROM files WHERE repo_id = ? LIMIT 1", (self.repo_id,)
            ).fetchone()
            return None if has_indexed_files is not None else _CODE_INDEXER_SEMANTICS_VERSION
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return None

    def _stamp_indexer_semantics_version(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO engine_state(key, value) VALUES ('indexer_semantics_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(_CODE_INDEXER_SEMANTICS_VERSION),),
        )

    def _delete_file_index(self, conn: sqlite3.Connection, rel: str) -> None:
        conn.execute("DELETE FROM file_line_fts WHERE repo_id = ? AND file_path = ?", (self.repo_id, rel))
        conn.execute("DELETE FROM file_path_trigram WHERE repo_id = ? AND file_path = ?", (self.repo_id, rel))
        conn.execute(
            """
            DELETE FROM symbol_trigram
            WHERE symbol_id IN (
                SELECT symbol_id FROM symbols WHERE repo_id = ? AND file_path = ?
            )
            """,
            (self.repo_id, rel),
        )
        conn.execute(
            """
            DELETE FROM symbol_fts
            WHERE symbol_id IN (
                SELECT symbol_id FROM symbols WHERE repo_id = ? AND file_path = ?
            )
            """,
            (self.repo_id, rel),
        )
        # Prune persisted embeddings for this file's symbols *before* the symbols
        # themselves. symbol_id encodes the file content hash, so an edited or
        # removed file yields fresh ids -- without this the old vectors orphan
        # (never overwritten, never cleaned) and pollute semantic ranking. The
        # vector table is created lazily, so guard against its absence.
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute(
                """
                DELETE FROM symbol_vectors
                WHERE repo_id = ? AND symbol_id IN (
                    SELECT symbol_id FROM symbols WHERE repo_id = ? AND file_path = ?
                )
                """,
                (self.repo_id, self.repo_id, rel),
            )
        conn.execute("DELETE FROM symbols WHERE repo_id = ? AND file_path = ?", (self.repo_id, rel))
        conn.execute("DELETE FROM imports WHERE repo_id = ? AND source_file = ?", (self.repo_id, rel))
        conn.execute('DELETE FROM "references" WHERE repo_id = ? AND file_path = ?', (self.repo_id, rel))
        conn.execute("DELETE FROM call_edges WHERE repo_id = ? AND caller_file_path = ?", (self.repo_id, rel))
        conn.execute("DELETE FROM files WHERE repo_id = ? AND file_path = ?", (self.repo_id, rel))

    def tool_index(
        self,
        *,
        include_globs: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        force: bool = False,  # incremental by default; pass force=True to full-rebuild
        budget_tokens: int = 4000,
    ) -> dict[str, Any]:
        effective_budget_tokens = self._effective_budget_tokens("index", budget_tokens)
        self._sync_symbol_intel()
        stats = self.index_repo(include_globs=include_globs, exclude_globs=exclude_globs, force=force)
        stats_payload = stats.model_dump(mode="json")
        snapshot = self._index_snapshot()
        return self._pack_single_payload(
            {
                "repo_id": stats_payload["repo_id"],
                "index_version": stats_payload["index_version"],
                "files_indexed": snapshot["files_indexed"],
                "symbols_indexed": snapshot["symbols_indexed"],
                "imports_indexed": snapshot["imports_indexed"],
                "provenance": _LOCAL_PROVENANCE,
            },
            budget_tokens=effective_budget_tokens,
            essential_keys=_INDEX_ESSENTIAL_KEYS,
            optional_keys_in_drop_order=[],
        )

    def tool_search(
        self,
        query: str,
        *,
        limit: int = 20,
        mode: SearchMode = "auto",
        intent: Literal["auto", "symbol", "text", "semantic"] = "auto",
        kind: str | None = None,
        language: str | None = None,
        seed_files: list[str] | None = None,
        snippet: Literal["none", "head", "full"] = "none",
        snippet_lines: int = 8,
        file_glob: str | None = None,
        scope: Literal["repo", "external", "deleted"] = "repo",
        since: str | None = None,
        touched_by: str | None = None,
        budget_tokens: int = 4000,
        auto_index: bool = True,
        provenance_filter: str | None = None,
    ) -> dict[str, Any]:
        force_compact_snippet = self._should_force_search_compaction(scope=scope, snippet=snippet, limit=limit)
        effective_snippet: Literal["none", "head", "full"] = "none" if force_compact_snippet else snippet
        effective_snippet_lines = 0 if effective_snippet == "none" else max(1, int(snippet_lines))
        apply_search_cap = scope == "repo" and snippet == "none"
        effective_budget_tokens = (
            self._effective_budget_tokens("search", budget_tokens) if apply_search_cap else max(1, int(budget_tokens))
        )
        if auto_index and scope != "deleted":
            self._ensure_indexed()
        self._sync_symbol_intel()
        resolved_mode = "semantic" if intent == "semantic" else resolve_search_mode(query, mode)
        if resolved_mode in {"semantic", "hybrid"} and not self._semantic_ranker.available:
            # Semantic search requires a configured embedding backend. By default none
            # is set (no external LLM is contacted). If the caller explicitly asked for
            # semantic/hybrid, say so; for an auto-resolved query fall back to lexical.
            if intent == "semantic" or mode in {"semantic", "hybrid"}:
                return {
                    "items": [],
                    "mode": resolved_mode,
                    "semantic_available": False,
                    "provenance": _LOCAL_PROVENANCE,
                    "cache_hit": False,
                    "message": (
                        "Semantic search is not configured. Set LEMONCROW_CODE_EMBEDDER "
                        "(local|openai|letta|ollama) and optionally LEMONCROW_CODE_EMBED_MODEL to enable it."
                    ),
                }
            resolved_mode = "lexical"
        text_substring_query: str | None = query if intent == "text" else None
        if intent == "auto":
            text_substring_query = self._should_use_text_substring_search(
                query,
                mode=resolved_mode,
                scope=scope,
                kind=kind,
                language=language,
                file_glob=file_glob,
                provenance_filter=provenance_filter,
            )
        use_text_substring = text_substring_query is not None
        temporal_scope = scope in {"repo", "deleted"}
        parsed_since = _parse_since_filter(since) if temporal_scope else None
        normalized_touched_by = _normalize_touched_by(touched_by) if temporal_scope else None
        normalized_seed_files = [self._normalize_file_arg(seed) for seed in seed_files or []]
        rerank_limit = self._search_reranker.pre_rerank_limit(limit, mode=resolved_mode, scope=scope)
        cache_args = {
            "query": query,
            "limit": limit,
            "mode": mode,
            "intent": intent,
            "resolved_mode": resolved_mode,
            "kind": kind,
            "language": language,
            "seed_files": normalized_seed_files,
            "snippet": snippet,
            "effective_snippet": effective_snippet,
            "snippet_lines": effective_snippet_lines,
            "file_glob": file_glob,
            "scope": scope,
            "since_ts": parsed_since,
            "touched_by": normalized_touched_by,
            "budget_tokens": effective_budget_tokens,
            "semantic_candidate_limit": semantic_candidate_limit(rerank_limit),
            "rerank_limit": rerank_limit,
            "rerank": self._search_reranker.cache_fingerprint(mode=resolved_mode, scope=scope),
            "provenance_filter": provenance_filter,
            "use_text_substring": use_text_substring,
        }
        hit, cached = self._cache_get("code.search", cache_args)
        if hit and cached is not None:
            return self._mark_cache_hit(cached)

        if text_substring_query is not None:
            text_payload = self._tool_text_substring_search(
                text_substring_query,
                limit=limit,
                file_glob=file_glob,
                budget_tokens=effective_budget_tokens,
                since_ts=parsed_since,
                touched_by=normalized_touched_by,
            )
            self._cache_set("code.search", cache_args, text_payload)
            return text_payload

        if scope == "deleted":
            raw_deleted_items = self.search_symbols(
                query,
                limit=limit,
                mode=resolved_mode,
                kind=kind,
                language=language,
                snippet=effective_snippet,
                snippet_lines=effective_snippet_lines,
                file_glob=file_glob,
                scope="deleted",
                since=since,
                touched_by=touched_by,
                auto_index=False,
            )
            items = [dict(item) for item in raw_deleted_items]
        else:
            raw_items = self.search_symbols(
                query,
                limit=limit,
                mode=resolved_mode,
                kind=kind,
                language=language,
                snippet=effective_snippet,
                snippet_lines=effective_snippet_lines,
                file_glob=file_glob,
                scope=scope,
                since=since,
                touched_by=touched_by,
                auto_index=False,
                provenance_filter=provenance_filter,
            )
            items = [item.model_dump(mode="json", exclude_none=True) for item in raw_items]
        if scope == "repo" and (parsed_since is not None or normalized_touched_by is not None):
            changed_files = self._deleted_history_adapter().changed_files(
                since_ts=parsed_since,
                touched_by=normalized_touched_by,
            )
            items = [item for item in items if str(item.get("file_path") or "") in changed_files]
        items = self._dedupe_search_items(items)
        items = self._prioritize_grounded_search_items(items, seed_files=normalized_seed_files)
        if (
            not items
            and intent == "auto"
            and scope == "repo"
            and resolved_mode == "lexical"
            and kind is None
            and provenance_filter is None
            # Cold/unbuilt index: search_symbols legitimately returns nothing
            # because there is nothing indexed yet, not because the query
            # missed. _tool_text_substring_search's search_text falls through
            # to a live ripgrep/filesystem scan when its own index is empty --
            # exactly the blocking scan the cold-index guard exists to avoid.
            and self.index_ready()
        ):
            # Symbol/semantic search found nothing. Instead of predicting from the
            # query's shape whether it embeds a literal worth a substring search
            # (the old approach -- and the reason this bug class kept recurring one
            # token-shape at a time: snake_case, then camelCase, then...), react to
            # the actual miss: tokenize the raw query and probe candidates for
            # verbatim existence anywhere in the repo, as EITHER a declared symbol
            # or raw line content. Existence, not shape, gates the fallback, so it
            # durably covers shapes never special-cased here -- a kebab-case HTTP
            # header literal (x-flipt-accept-server-version), a quoted string, a
            # UUID, tomorrow's format. A token that isn't a real substring anywhere
            # changes nothing, so ordinary natural-language queries with no
            # embedded literal behave exactly as before this ran.
            fallback_query = self._existence_gated_fallback_token(
                query, kind=kind, language=language, file_glob=file_glob
            )
            if fallback_query is not None:
                text_payload = self._tool_text_substring_search(
                    fallback_query,
                    limit=limit,
                    file_glob=file_glob,
                    budget_tokens=effective_budget_tokens,
                    since_ts=parsed_since,
                    touched_by=normalized_touched_by,
                )
                self._cache_set("code.search", cache_args, text_payload)
                return text_payload
        # Capture aggregate provenance before compaction strips per-item provenance
        # (repo-scope compaction drops "provenance"/"symbol_id" as redundant with the
        # top-level fields), so the routed-provider provenance survives in the payload.
        aggregate_provenance = self._items_provenance(items)
        if effective_snippet == "none":
            items = self._compact_search_items(items, scope=scope)
        essential_keys = _DELETED_SEARCH_ESSENTIAL_KEYS if scope == "deleted" else _SEARCH_ESSENTIAL_KEYS
        optional_keys = _DELETED_SEARCH_OPTIONAL_KEYS if scope == "deleted" else _SEARCH_OPTIONAL_KEYS
        payload = self._pack_items_payload(
            items,
            budget_tokens=effective_budget_tokens,
            essential_keys=essential_keys,
            optional_keys_in_drop_order=optional_keys,
            extra_payload={
                "mode": resolved_mode,
                "snippet": effective_snippet,
                "provenance": aggregate_provenance,
            },
        )
        self._cache_set("code.search", cache_args, payload)
        return payload

    def search_channel_health(self, query: str, mode: str = "auto") -> ChannelHealth:
        """Liveness of optional retrieval channels for a query (verdict stamping).

        Lets the MCP boundary distinguish a *trustworthy* empty (every channel the
        query wanted actually ran) from a *dark* one (a wanted channel was off):

        - ``semantic``: applicable only when the resolved mode wants it
          (semantic/hybrid); ``True`` if an embedder is configured, ``False``
          (dark) if not, ``None`` if the query never wanted it (lexical lookup).
        - ``zoekt``: applicable for repo-scope lexical/hybrid; ``False`` (dark)
          only when it is meant to route but the backend is unhealthy. A
          config-disabled zoekt is ``None`` (not dark) -- FTS, always live,
          covers the lexical channel.
        """
        from lemoncrow.pro.capabilities.code_context.search_verdict import ChannelHealth

        requested = cast("SearchMode", mode if mode in ("auto", "lexical", "semantic", "hybrid") else "auto")
        resolved = resolve_search_mode(query, requested)
        semantic: bool | None = None
        if resolved in {"semantic", "hybrid"}:
            semantic = bool(self._semantic_ranker.available)
        zoekt: bool | None = None
        if resolved != "semantic":
            with contextlib.suppress(Exception):
                from lemoncrow.infra.code_intel.zoekt.adapter import get_zoekt_supervisor

                supervisor = get_zoekt_supervisor(self.repo_root)
                if supervisor.should_route(self.repo_root):
                    zoekt = bool(supervisor.health().ok)
        return ChannelHealth(semantic=semantic, zoekt=zoekt)

    def tool_blame(
        self,
        *,
        query: str | None = None,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        symbol_name: str | None = None,
        file_path: str | None = None,
        include_churn: bool = True,
        budget_tokens: int = 4000,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        if auto_index:
            self._ensure_indexed()
        self._sync_symbol_intel()
        target = self._resolve_symbol_target(
            operation_name="blame",
            query=query,
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            symbol_name=symbol_name,
            file_path=file_path,
            kind=None,
            language=None,
            file_glob=None,
        )
        if target.get("error"):
            return self._pack_single_payload(
                target,
                budget_tokens=budget_tokens,
                essential_keys=["error", "message", "matches", "cache_hit", "provenance"],
                optional_keys_in_drop_order=["provenance_breakdown"],
            )
        head_sha = self._current_head_sha()
        index_sha = str(target.get("index_sha") or head_sha)
        normalized_file_path = str(target["file_path"])
        cache_args = {
            "query": query,
            "symbol_id": symbol_id or target.get("symbol_id"),
            "qualified_name": qualified_name or target.get("qualified_name"),
            "symbol_name": symbol_name or target.get("symbol_name"),
            "file_path": normalized_file_path,
            "include_churn": include_churn,
            "index_sha": index_sha,
            "head_sha": head_sha,
            "budget_tokens": budget_tokens,
        }
        hit, cached = self._cache_get("code.blame", cache_args)
        if hit and cached is not None:
            return self._mark_cache_hit(cached)
        if index_sha != head_sha:
            payload = self._pack_single_payload(
                {
                    "error": "index_stale",
                    "hint": 'run code op="index" first',
                    "symbol_name": str(target["symbol_name"]),
                    "qualified_name": str(target["qualified_name"]),
                    "file_path": normalized_file_path,
                    "freshness": "stale",
                    "index_sha": index_sha,
                    "head_sha": head_sha,
                    "provenance": "blame",
                },
                budget_tokens=budget_tokens,
                essential_keys=[
                    "error",
                    "hint",
                    "symbol_name",
                    "qualified_name",
                    "file_path",
                    "freshness",
                    "provenance",
                ],
                optional_keys_in_drop_order=["index_sha", "head_sha"],
            )
            self._cache_set("code.blame", cache_args, payload)
            return payload
        from lemoncrow.pro.code_intel.git_history.blame import BlameAnnotator
        from lemoncrow.pro.code_intel.git_history.models import BlameRequest

        try:
            annotation = BlameAnnotator(self.repo_root).annotate(
                BlameRequest(
                    file_path=normalized_file_path,
                    line_start=int(target["start_line"]),
                    line_end=int(target["end_line"]),
                    index_sha=index_sha,
                    head_sha=head_sha,
                    include_churn=include_churn,
                )
            )
        except ValueError:
            # The symbol's recorded line span does not map onto a committed HEAD
            # blob (uncommitted/working-tree region or a dirty re-index). Return a
            # structured payload instead of crashing the MCP handler with -32603.
            payload = self._pack_single_payload(
                {
                    "error": "blame_unavailable",
                    "hint": "symbol range is not yet committed; commit then re-index",
                    "symbol_name": str(target["symbol_name"]),
                    "qualified_name": str(target["qualified_name"]),
                    "file_path": normalized_file_path,
                    "line_start": int(target["start_line"]),
                    "line_end": int(target["end_line"]),
                    "provenance": "blame",
                },
                budget_tokens=budget_tokens,
                essential_keys=[
                    "error",
                    "hint",
                    "symbol_name",
                    "qualified_name",
                    "file_path",
                    "provenance",
                ],
                optional_keys_in_drop_order=["line_start", "line_end"],
            )
            self._cache_set("code.blame", cache_args, payload)
            return payload
        latest_commit_ts = max(hunk.commit_time for hunk in annotation.hunks)
        payload_data: dict[str, Any] = {
            "symbol_name": str(target["symbol_name"]),
            "qualified_name": str(target["qualified_name"]),
            "file_path": normalized_file_path,
            "line_start": int(target["start_line"]),
            "line_end": int(target["end_line"]),
            "index_sha": index_sha,
            "head_sha": head_sha,
            "freshness": annotation.freshness,
            "last_modified": datetime.fromtimestamp(latest_commit_ts, tz=UTC).isoformat().replace("+00:00", "Z"),
            "last_author": annotation.last_author,
            "last_commit_sha": annotation.last_commit_sha,
            "last_commit_summary": annotation.last_commit_summary,
            "age_days": annotation.age_days,
            "local_edits": annotation.local_edits,
            "distinct_authors": len({hunk.author_email for hunk in annotation.hunks if hunk.author_email}),
            "hunks": [asdict(hunk) for hunk in annotation.hunks],
            "provenance": "blame",
        }
        if annotation.churn is not None:
            payload_data["churn"] = asdict(annotation.churn)
        payload = self._pack_single_payload(
            payload_data,
            budget_tokens=budget_tokens,
            essential_keys=_BLAME_ESSENTIAL_KEYS,
            optional_keys_in_drop_order=_BLAME_OPTIONAL_KEYS,
        )
        self._cache_set("code.blame", cache_args, payload)
        return payload

    def _symbol_at_line(self, file_path: str, line: int) -> dict[str, Any]:
        """Resolve the innermost symbol whose span contains `line` in `file_path`."""
        normalized = self._normalize_file_arg(file_path)
        with self._connect() as conn:
            self._init_schema(conn)
            row = conn.execute(
                """
                SELECT *, NULL AS score FROM symbols
                WHERE repo_id = ? AND file_path = ? AND start_line <= ? AND end_line >= ?
                ORDER BY start_line DESC
                LIMIT 1
                """,
                (self.repo_id, normalized, line, line),
            ).fetchone()
        if row is None:
            raise LookupError("no symbol at that position")
        symbol_rec = _row_to_symbol(row)
        path = self.repo_root / symbol_rec.file_path
        source = path.read_bytes()[symbol_rec.start_byte : symbol_rec.end_byte].decode("utf-8", errors="replace")
        return {**symbol_rec.model_dump(mode="json"), "source": source}

    def tool_symbol(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
        line: int | None = None,
        budget_tokens: int = 4000,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        effective_budget_tokens = self._effective_budget_tokens("symbol", budget_tokens)
        if auto_index:
            self._ensure_indexed()
        self._sync_symbol_intel()
        normalized_file_path = self._normalize_file_arg(file_path) if file_path else None
        positional_lookup = (
            file_path is not None and line is not None and not symbol_id and not qualified_name and not symbol_name
        )
        cache_args = {
            "symbol_id": symbol_id,
            "qualified_name": qualified_name,
            "symbol_name": symbol_name,
            "file_path": normalized_file_path,
            "line": line if positional_lookup else None,
            "budget_tokens": effective_budget_tokens,
        }
        hit, cached = self._cache_get("code.symbol", cache_args)
        if hit and cached is not None:
            return self._mark_cache_hit(cached)

        raw_symbol = (
            self._symbol_at_line(file_path, line)  # type: ignore[arg-type]
            if positional_lookup
            else self.get_symbol(
                symbol_id=symbol_id,
                qualified_name=qualified_name,
                symbol_name=symbol_name,
                file_path=normalized_file_path,
                auto_index=False,
            )
        )
        payload = self._pack_single_payload(
            self._hydrate_symbol_cross_lang(raw_symbol),
            budget_tokens=effective_budget_tokens,
            essential_keys=_SYMBOL_ESSENTIAL_KEYS,
            optional_keys_in_drop_order=_SYMBOL_OPTIONAL_KEYS,
        )
        self._cache_set("code.symbol", cache_args, payload)
        return payload

    def _hydrate_symbol_cross_lang(self, payload: dict[str, Any]) -> dict[str, Any]:
        symbol_id = str(payload.get("symbol_id") or "")
        symbol_name = str(payload.get("symbol_name") or "")
        if not symbol_id:
            return payload
        refs: list[CrossLangReference] = []
        for edge in self._cross_lang_store().query_by_source_symbol(symbol_id):
            refs.append(self._symbol_cross_lang_ref(edge, direction="outgoing"))
        for edge in self._cross_lang_store().query_by_target_symbol(
            tgt_symbol_id=symbol_id, tgt_symbol_name=symbol_name
        ):
            refs.append(self._symbol_cross_lang_ref(edge, direction="incoming"))
        if not refs:
            return payload
        deduped = list(
            {
                (
                    ref.direction,
                    ref.symbol_id,
                    ref.symbol_name,
                    ref.file_path,
                    ref.line,
                    ref.edge_kind,
                ): ref
                for ref in refs
            }.values()
        )
        deduped.sort(
            key=lambda ref: (
                ref.direction,
                ref.file_path,
                int(ref.line or 0),
                ref.symbol_name,
                ref.symbol_id,
                ref.edge_kind,
            )
        )
        allowed = {key: value for key, value in payload.items() if key in SymbolRecord.model_fields}
        validated = SymbolRecord.model_validate(
            {
                **allowed,
                "cross_lang_refs": [ref.model_dump(mode="json", exclude_none=True) for ref in deduped],
            }
        ).model_dump(mode="json", exclude_none=True)
        if "source" in payload:
            validated["source"] = payload["source"]
        return validated

    def _symbol_cross_lang_ref(
        self,
        edge: CrossLangEdge,
        *,
        direction: Literal["incoming", "outgoing"],
    ) -> CrossLangReference:
        if direction == "incoming":
            return CrossLangReference(
                symbol_id=edge.src_symbol_id,
                symbol_name=edge.src_symbol_name,
                qualified_name=edge.src_qualified_name,
                language=edge.src_language,
                file_path=edge.src_file_path,
                line=edge.src_line,
                direction=direction,
                edge_kind=edge.edge_kind,
                confidence=edge.confidence,
            )
        return CrossLangReference(
            symbol_id=edge.tgt_symbol_id,
            symbol_name=edge.tgt_symbol_name,
            qualified_name=None,
            language=edge.tgt_language,
            file_path=edge.tgt_file_path,
            line=edge.src_line,
            direction=direction,
            edge_kind=edge.edge_kind,
            confidence=edge.confidence,
        )

    def _cross_lang_store(self) -> CrossLangEdgeStore:
        return CrossLangEdgeStore(self.connection)

    def tool_files(
        self,
        *,
        path: str | None = None,
        pattern: str | None = None,
        format: Literal["tree", "flat", "grouped"] = "tree",
        include_metadata: bool = True,
        max_depth: int | None = None,
        budget_tokens: int = 4000,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        if auto_index:
            self._ensure_indexed()
        self._sync_symbol_intel()
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        normalized_path = self._normalize_files_path(path)
        normalized_pattern = (pattern or "").strip() or None
        cache_args = {
            "path": normalized_path,
            "pattern": normalized_pattern,
            "format": format,
            "include_metadata": include_metadata,
            "max_depth": max_depth,
            "budget_tokens": budget_tokens,
        }
        hit, cached = self._cache_get("code.files", cache_args)
        if hit and cached is not None:
            return self._mark_cache_hit(cached)

        items = self._indexed_file_records(path=normalized_path, pattern=normalized_pattern, max_depth=max_depth)
        essential_keys = list(_FILES_ESSENTIAL_KEYS)
        if format == "grouped":
            essential_keys.append("language")
        optional_keys = _FILES_OPTIONAL_KEYS if include_metadata else []
        full_payload = self._build_files_payload(
            items,
            path=normalized_path,
            pattern=normalized_pattern,
            format=format,
            include_metadata=include_metadata,
            truncated=False,
        )
        full_total_tokens = self._compute_total_tokens({**full_payload, "tokens_saved": 0})

        def build_payload(packed_items: list[dict[str, Any]]) -> dict[str, Any]:
            return self._finalize_packed_payload(
                self._build_files_payload(
                    packed_items,
                    path=normalized_path,
                    pattern=normalized_pattern,
                    format=format,
                    include_metadata=include_metadata,
                    truncated=len(packed_items) < len(items),
                ),
                full_total_tokens=full_total_tokens,
            )

        packed = self._fit_items_to_budget(
            items,
            budget_tokens=budget_tokens,
            essential_keys=essential_keys,
            optional_keys_in_drop_order=optional_keys,
            build_payload=build_payload,
        )
        payload = self._maybe_attach_overflow_metadata(
            packed_payload=packed,
            full_payload=full_payload,
            full_total_tokens=full_total_tokens,
            budget_tokens=budget_tokens,
        )
        self._cache_set("code.files", cache_args, payload)
        return payload

    # ----- domain-neutral Retriever conformance --------------------------- #
    # CodeContextEngine is the code vertical of the engine's one-shot
    # retrieval contract (lemoncrow.core.capabilities.retrieval.Retriever):
    # one call returns a budget-packed context payload; follow-up queries
    # are allowed, retry loops are never required.

    @property
    def source_id(self) -> str:
        """Stable corpus identifier (the repo id) for the Retriever protocol."""
        return self.repo_id

    def retrieve(
        self,
        query: str,
        *,
        budget_tokens: int = 2000,
        max_items: int = 8,
        seeds: list[str] | None = None,
    ) -> dict[str, Any]:
        """One-shot retrieval (Retriever protocol) -- delegates to tool_explore."""
        return self.tool_explore(
            query,
            seed_files=seeds,
            max_files=max_items,
            budget_tokens=budget_tokens,
        )

    def tool_explore(
        self,
        query: str,
        *,
        seed_files: list[str] | None = None,
        max_files: int = 6,
        max_symbols: int = 20,
        include_source: bool = True,
        include_relationships: bool = False,
        line_numbers: bool = True,
        skeletonize: bool = True,
        complete_families: bool | None = None,
        depth: int = 1,
        budget_tokens: int = 2000,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        if auto_index:
            self._ensure_indexed()
        self._sync_symbol_intel()
        # SOFT paths= scope (any seed -- single file, directory, or multi-path): the
        # scope is a ranking prioritisation, never a hard boundary. The whole-repo
        # recall channels (HEF + RRF fusion + reference-file expansion) always run so
        # neighbouring files surface, and an explicit scope reserves only the TOP-2
        # spots (ranks #1-#2) for its in-scope files; from #3 on it is a pure ranking
        # race with no seed bias (see _reserve_scoped_entries here + the same reserve
        # cap in _tool_explore_impl's symbol ranking and the MCP lean view).
        _seed_norm = [self._normalize_file_arg(s) for s in seed_files] if seed_files else []
        # The three V6 HEF recall channels (exact / anchor-zoekt / line-FTS) depend
        # ONLY on the parsed query plan -- never on the baseline result.  Launch them
        # up front so they run *concurrently* with _tool_explore_impl (baseline FTS +
        # main Zoekt) instead of in a second serial phase afterward.  This collapses
        # the explore critical path from (Phase A + Phase B) to max(Phase A, Phase B);
        # _fused_explore_hybrid simply collects the already-in-flight futures.
        hef_futures = self._submit_hef_channels(query)
        # One shared connection for the whole explore (search + relationship
        # hydration + packing are reads, plus a one-time centrality persist).
        with self._reuse_connection():
            result, precomputed_zoekt = self._tool_explore_impl(
                query,
                seed_files=seed_files,
                max_files=max_files,
                max_symbols=max_symbols,
                include_source=include_source,
                include_relationships=include_relationships,
                line_numbers=line_numbers,
                skeletonize=skeletonize,
                complete_families=complete_families,
                depth=depth,
                budget_tokens=budget_tokens,
            )
        fused = self._fused_explore_hybrid(
            query, result, max_files=max_files, precomputed_zoekt=precomputed_zoekt, hef_futures=hef_futures
        )
        # Reserve up to the top-2 spots for the in-scope files (recovering any the
        # max_files cut truncated), then let every other file race by rank below.
        if _seed_norm:
            fused = self._reserve_scoped_entries(fused, result, _seed_norm)
        return self._rerank_explore_result(query, fused)

    @staticmethod
    def _explore_entry_path(entry: Any) -> str:
        """File path of an explore file-entry dict (`path`, or legacy `file_path`)."""
        if not isinstance(entry, dict):
            return ""
        return str(entry.get("path") or entry.get("file_path") or "")

    def _reserve_scoped_entries(
        self, fused: dict[str, Any], baseline: dict[str, Any], seed_norm: list[str], *, limit: int = 2
    ) -> dict[str, Any]:
        """Guarantee up to ``limit`` in-scope files survive fusion and lead the list.

        A paths= scope reserves only the top ``limit`` spots (ranks #1-#2) for its
        in-scope files; every other file -- out-of-scope neighbours the whole-repo
        channels surfaced, plus any 3rd+ in-scope file -- races purely on rank below
        them. Fusion can truncate an in-scope file past its max_files cut, so recover
        up to ``limit`` of them from the baseline and hoist them to the head; the
        rest keep fused (score) order. The MCP lean view applies the same
        reserve-``limit`` cap to the agent-visible file ranking.
        """
        files = fused.get("files")
        if not isinstance(files, list) or not seed_norm:
            return fused
        seeds = [s.rstrip("/") for s in seed_norm if s]
        if not seeds:
            return fused

        def in_scope(path: str) -> bool:
            return bool(path) and any(path == s or path.startswith(s + "/") for s in seeds)

        # Per-file entry-point score. Reserve only in-scope files that actually
        # matched (score > 0); a 0-match in-scope file claims no reserved slot, so it
        # never forces itself above a matched neighbour -- mirrors the MCP lean view's
        # epscore>0 gate so the engine (retrieve/eval path) and the agent-visible
        # ranking agree.
        epscore: dict[str, float] = {}
        for e in fused.get("entry_points") or []:
            if not isinstance(e, dict):
                continue
            ep = self._explore_entry_path(e)
            if not ep:
                continue
            try:
                sc = float(e.get("score") or 0.0)
            except (TypeError, ValueError):
                sc = 0.0
            epscore[ep] = max(epscore.get(ep, 0.0), sc)

        def matched(path: str) -> bool:
            return in_scope(path) and epscore.get(path, 0.0) > 0.0

        reserved: list[dict[str, Any]] = []
        reserved_paths: set[str] = set()
        for f in files:
            if len(reserved) >= limit:
                break
            p = self._explore_entry_path(f)
            if matched(p) and p not in reserved_paths:
                reserved.append(f)
                reserved_paths.add(p)
        if len(reserved) < limit:
            for f in baseline.get("files") or []:
                if len(reserved) >= limit:
                    break
                p = self._explore_entry_path(f)
                if matched(p) and p not in reserved_paths:
                    reserved.append(f)
                    reserved_paths.add(p)
        if not reserved:
            return fused
        rest = [f for f in files if self._explore_entry_path(f) not in reserved_paths]
        out = dict(fused)
        out["files"] = [*reserved, *rest]
        return out

    @property
    def explore_reranker_model_path(self) -> Path:
        """Per-workspace path for the trained explore reranker model.

        Each repo deploys its own model next to its index DB.
        ``LEMONCROW_EXPLORE_RERANKER_MODEL`` overrides it for benchmarking / A-B.
        """
        override = os.environ.get("LEMONCROW_EXPLORE_RERANKER_MODEL")
        if override:
            return Path(override).expanduser()
        return self.db_path.parent / "explore_reranker.json"

    def _load_explore_reranker(self) -> dict[str, Any] | None:
        """Load and validate the explore reranker (LambdaMART trees).

        Explicit and per-workspace models take precedence over the packaged
        global model. The validated model is cached on the instance so JSON is
        parsed once per process. Disabled while collecting self-supervised
        training candidates (``LEMONCROW_SELF_SUPERVISED_TRAINING=1``) so the
        trainer observes raw V6 ordering, and via
        ``LEMONCROW_EXPLORE_RERANKER_ENABLED=0``.
        """
        if hasattr(self, "_er_model_loaded"):
            return self._er_model_cache  # type: ignore[attr-defined]
        # Set cache sentinel before the loaded flag so concurrent threads that
        # observe _er_model_loaded=True always find _er_model_cache defined.
        self._er_model_cache: dict[str, Any] | None = None
        self._er_model_loaded: bool = True
        if str(os.environ.get("LEMONCROW_SELF_SUPERVISED_TRAINING") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return None
        if os.environ.get("LEMONCROW_EXPLORE_RERANKER_ENABLED", "1") == "0":
            return None
        # Explicit/per-workspace models override the packaged global fallback;
        # the db-stem-keyed variant avoids /tmp collisions.
        for path in dict.fromkeys(
            [
                self.explore_reranker_model_path,
                self.db_path.with_suffix(".explore_reranker.json"),
                _DEFAULT_EXPLORE_RERANKER_MODEL_PATH,
            ]
        ):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            model = _validate_er_model(raw)
            if model is not None:
                self._er_model_cache = model
                return model
        return None

    def _rerank_explore_result(
        self,
        query: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Rerank the top candidates in an explore payload with the global
        LambdaMART model. Returns *payload* unchanged when no model is
        available or the proposed order is identical.

        Legacy models only move the original top candidate. Models trained with
        ``rerank_lower_when_top_unchanged`` may also improve ranks 2..N while
        preserving a correct rank-1 result.
        """
        raw_entries = payload.get("files")
        if not isinstance(raw_entries, list) or len(raw_entries) < 2:
            return payload

        model = self._load_explore_reranker()
        if model is None:
            return payload

        window = min(int(model.get("window", 5)), len(raw_entries))
        if window < 2:
            return payload
        model_type = model.get("model_type")
        if model_type == "linear":
            weights = [float(w) for w in model["weights"]]
            blend = float(model.get("blend", 1.0))
        else:
            trees = model["trees"]

        active_feature_indices = model.get("active_feature_indices")
        model_feature_count = len(model["feature_names"])
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for rank, entry in enumerate(raw_entries[:window], 1):
            if not isinstance(entry, dict):
                return payload  # unexpected shape; skip reranking
            features = _er_entry_features(query, entry, rank, active_feature_indices)[:model_feature_count]
            if model_type == "linear":
                learned = _er_linear_score(weights, features)
                score = blend * learned + (1.0 - blend) * (1.0 / rank)
            else:
                score = _er_tree_score(trees, features)
            scored.append((score, rank, entry))

        proposed = sorted(scored, key=lambda item: (-item[0], item[1]))
        rerank_lower = bool(model.get("rerank_lower_when_top_unchanged", False))
        if proposed[0][1] == 1 and not rerank_lower:
            return payload  # legacy policy: top file unchanged

        margin = float(model.get("margin", 0.0))
        if proposed[0][1] != 1 and margin > 0.0:
            original_top = next(item for item in scored if item[1] == 1)
            if proposed[0][0] - original_top[0] < margin:
                if not rerank_lower:
                    return payload
                # Keep the original top, but retain useful model ordering below it.
                proposed = [original_top, *(item for item in proposed if item[1] != 1)]

        if [rank for _score, rank, _entry in proposed] == list(range(1, window + 1)):
            return payload

        reranked = [e for _s, _r, e in proposed] + list(raw_entries[window:])
        result = dict(payload)
        result["files"] = reranked
        result["experiment"] = {
            "name": "explore_reranker_v2_lambdamart",
            "base": payload.get("experiment"),
        }
        return result

    # -----------------------------------------------------------------------
    # Hybrid explore fusion (v6) — instance methods
    # -----------------------------------------------------------------------

    def _hef_parse_query(self, query: str) -> _HefQueryPlan:
        definitions = tuple((match.group("kind"), match.group("name")) for match in _HEF_DEFINITION_RE.finditer(query))
        definition_names = [name for _kind, name in definitions]
        identifiers: list[str] = []
        alternatives: list[str] = []
        for segment in query.split("|"):
            match = _HEF_DEFINITION_RE.search(segment)
            if match:
                alternatives.append(match.group("name"))
            else:
                bare = _hef_bare_alternative(segment)
                if bare is not None:
                    alternatives.append(bare)
        for token in _HEF_IDENTIFIER_RE.findall(query):
            low = token.lower()
            if low in _HEF_STOP or len(token) < 3:
                continue
            if _hef_is_code_shaped(token):
                identifiers.append(token)
        normalized = query.strip()
        with contextlib.suppress(Exception):
            if _is_precise_symbol_query(normalized):
                alternatives.append(normalized.rsplit(".", 1)[-1])
        literals = [
            match.group("value").strip() for match in _HEF_QUOTED_RE.finditer(query) if match.group("value").strip()
        ]
        prose_terms: list[str] = []
        with contextlib.suppress(Exception):
            prose_terms = [
                str(term)
                for term in _query_terms(query)
                if len(str(term)) >= 3 and str(term).lower() not in _HEF_PROSE_STOP
            ]
        identifier_tuple = _hef_dedupe([*definition_names, *alternatives, *identifiers], limit=18)
        anchor_tuple = _hef_dedupe([*definition_names, *alternatives, *identifiers], limit=10)
        literal_tuple = _hef_dedupe(literals, limit=8)
        if definitions:
            intent = "definition"
        elif normalized and _is_precise_symbol_query(normalized):
            intent = "symbol"
        elif identifier_tuple or "|" in query:
            intent = "code"
        else:
            intent = "prose"
        if intent in {"definition", "symbol"}:
            term_values = [*identifier_tuple, *literal_tuple]
        elif intent == "code":
            term_values = [*identifier_tuple, *literal_tuple, *prose_terms]
        else:
            term_values = [*literal_tuple, *prose_terms]
        wants_tests = bool(
            re.search(
                r"\btest(?:_|s\b|ing\b)|\bspec(?:_|s\b)|pytest|unittest|" r"tearDown|setUp|TestCase|Tests\b",
                query,
                re.IGNORECASE,
            )
        )
        wants_auxiliary = bool(
            re.search(
                r"\bdocs?|documentation|example|gallery|benchmark|frontend|" r"javascript|typescript|readme\b",
                query,
                re.IGNORECASE,
            )
        )
        return _HefQueryPlan(
            intent=intent,
            definitions=definitions,
            identifiers=identifier_tuple,
            anchors=anchor_tuple,
            terms=_hef_dedupe(term_values, limit=16),
            literals=literal_tuple,
            wants_tests=wants_tests,
            wants_auxiliary=wants_auxiliary,
        )

    def _hef_exact_symbol_candidates(
        self,
        plan: _HefQueryPlan,
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        if not plan.identifiers:
            return [], {}
        tokens = {token.lower(): token for token in plan.identifiers}
        placeholders = ",".join("?" for _ in tokens)
        try:
            with self._connect(readonly=True) as conn:
                rows = conn.execute(
                    f"""
                    WITH matched AS (
                        SELECT file_path,
                               lower(symbol_name) AS token,
                               lower(kind) AS kind
                        FROM symbols
                        WHERE repo_id = ?
                          AND lower(symbol_name) IN ({placeholders})
                    ),
                    frequencies AS (
                        SELECT token, COUNT(DISTINCT file_path) AS df
                        FROM matched
                        GROUP BY token
                    )
                    SELECT matched.file_path,
                           matched.token,
                           matched.kind,
                           frequencies.df
                    FROM matched
                    JOIN frequencies USING (token)
                    ORDER BY matched.file_path, matched.token, matched.kind
                    """,
                    (self.repo_id, *tokens),
                ).fetchall()
        except Exception:
            return [], {}
        expected = {name.lower(): kind for kind, name in plan.definitions}
        per_file: dict[str, dict[str, Any]] = {}
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            file_path = str(row["file_path"] or "")
            token = str(row["token"] or "")
            kind = str(row["kind"] or "").lower()
            df = max(1, int(row["df"] or 1))
            if not file_path or not token:
                continue
            key = (file_path, token, kind)
            if key in seen:
                continue
            seen.add(key)
            expected_kind = expected.get(token)
            kind_match = (expected_kind == "class" and kind == "class") or (
                expected_kind == "def" and kind in {"function", "method", "async_function"}
            )
            item = per_file.setdefault(
                file_path,
                {
                    "tokens": set(),
                    "definition_tokens": set(),
                    "kind_matches": set(),
                    "idf": 0.0,
                    "best_df": df,
                },
            )
            item["tokens"].add(token)
            if kind in {"class", "function", "method", "async_function"}:
                item["definition_tokens"].add(token)
            if kind_match:
                item["kind_matches"].add(token)
            item["idf"] += math.log1p(1.0 + 1.0 / df)
            item["best_df"] = min(int(item["best_df"]), df)
        identifier_count = max(1, len(plan.identifiers))
        definition_count = max(1, len(plan.definitions))
        for item in per_file.values():
            token_coverage = len(item["tokens"]) / identifier_count
            kind_coverage = len(item["kind_matches"]) / definition_count
            definition_coverage = len(item["definition_tokens"]) / identifier_count
            rarity = min(1.0, float(item["idf"]) / identifier_count)
            item["confidence"] = min(
                1.0,
                0.42 * token_coverage + 0.30 * kind_coverage + 0.16 * definition_coverage + 0.12 * rarity,
            )
        ordered = sorted(
            per_file,
            key=lambda p: (
                -float(per_file[p]["confidence"]),
                -len(per_file[p]["kind_matches"]),
                -len(per_file[p]["tokens"]),
                int(per_file[p]["best_df"]),
                p,
            ),
        )
        details = {
            p: {
                "tokens": sorted(per_file[p]["tokens"]),
                "definition_tokens": sorted(per_file[p]["definition_tokens"]),
                "kind_matches": sorted(per_file[p]["kind_matches"]),
                "idf": round(float(per_file[p]["idf"]), 6),
                "best_df": int(per_file[p]["best_df"]),
                "confidence": round(float(per_file[p]["confidence"]), 6),
            }
            for p in ordered
        }
        return ordered[:96], details

    def _hef_anchor_zoekt_candidates(
        self,
        plan: _HefQueryPlan,
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        if not plan.anchors:
            return [], {}
        cache = self._hef_anchor_cache
        uncached_anchors = [a for a in plan.anchors if a not in cache]
        if uncached_anchors:
            # Run all cache-missing anchor searches concurrently — each is an
            # independent Zoekt HTTP call; parallelism collapses N sequential
            # round-trips to one wall-clock slot.  These MUST run on a separate
            # pool (_HEF_ANCHOR_EXECUTOR): this method itself runs on
            # _HEF_CHANNEL_EXECUTOR, so submitting nested tasks back onto that
            # 3-worker pool self-starves under concurrent explores.
            def _fetch(anchor: str) -> list[str]:
                try:
                    return self._zoekt_candidate_files(anchor, path=".", max_files=30)
                except Exception:
                    return []

            anchor_futs = [_HEF_ANCHOR_EXECUTOR.submit(_fetch, a) for a in uncached_anchors]
            for anchor, fut in zip(uncached_anchors, anchor_futs, strict=True):
                try:
                    cache[anchor] = fut.result(timeout=0.15)
                except Exception:
                    cache[anchor] = []
        per_file: dict[str, dict[str, Any]] = {}
        for anchor in plan.anchors:
            files = cache.get(anchor, [])
            for rank, file_path in enumerate(files, 1):
                item = per_file.setdefault(
                    file_path,
                    {
                        "anchors": set(),
                        "rrf": 0.0,
                        "best_rank": rank,
                    },
                )
                item["anchors"].add(anchor.lower())
                item["rrf"] += 1.0 / (8.0 + rank)
                item["best_rank"] = min(int(item["best_rank"]), rank)
        anchor_count = max(1, len(plan.anchors))
        max_rrf = sum(1.0 / 9.0 for _ in plan.anchors)
        for item in per_file.values():
            coverage = len(item["anchors"]) / anchor_count
            normalized_rrf = min(1.0, float(item["rrf"]) / max_rrf)
            item["confidence"] = min(1.0, 0.72 * coverage + 0.28 * normalized_rrf)
        ordered = sorted(
            per_file,
            key=lambda p: (
                -float(per_file[p]["confidence"]),
                -len(per_file[p]["anchors"]),
                int(per_file[p]["best_rank"]),
                p,
            ),
        )
        details = {
            p: {
                "anchors": sorted(per_file[p]["anchors"]),
                "rrf": round(float(per_file[p]["rrf"]), 6),
                "best_rank": int(per_file[p]["best_rank"]),
                "confidence": round(float(per_file[p]["confidence"]), 6),
            }
            for p in ordered
        }
        return ordered[:96], details

    def _hef_line_fts_candidates(
        self,
        plan: _HefQueryPlan,
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        if not plan.terms:
            return [], {}
        terms = [term.lower() for term in plan.terms]
        # Morphological recall: also match each inflected term's stem as an FTS
        # prefix so a prose query ("parsing", "counts") reaches the code's
        # canonical form ("parse", "count").  bm25 IDF keeps common stems from
        # dominating; term_stems maps original term -> stem for coverage below.
        term_stems: dict[str, str] = {}
        for term in terms:
            stem = _query_stem(term)
            if stem and stem not in terms:
                term_stems[term] = stem
        or_parts = [_hef_fts_phrase(term) for term in terms]
        or_parts.extend(f"{stem}*" for stem in dict.fromkeys(term_stems.values()))
        # Build the AND query from real query identifiers only (plan.identifiers),
        # NOT from all plan.terms which may include n-gram compounds like
        # 'gettimezonename' that are generated for the symbol index but never
        # appear as raw tokens in source lines.  Including them in the AND query
        # means the AND always returns 0 rows (the compound isn't in file_line_fts),
        # so and_hit never fires and we lose the best specificity signal.
        # Use identifiers (the actual words/dotted names from the query) for AND.
        and_terms = [t.lower() for t in plan.identifiers] or terms
        and_query = " AND ".join(_hef_fts_phrase(t) for t in and_terms[: min(8, len(and_terms))])
        rows: list[tuple[Any, bool]] = []
        try:
            # Pooled per-thread read connection to fts.sqlite: this method runs
            # on the HEF executor threads where no scoped connection exists, so
            # self._connect() was paying connect + pragma + 3x ATTACH on every
            # call. The line queries only touch file_line_fts, which lives in
            # fts.sqlite -- a single-file pooled connection covers them.
            conn = _channel_connection(self.fts_db_path)
            # FTS5 bm25 top-K has no block-max skipping: cost is linear in each
            # OR term's document frequency (measured: df=82K 'return' on linux
            # = 30ms for ONE term; rare terms = 0.1ms). Ultra-common terms are
            # near-zero IDF -- they barely move bm25 -- so drop any OR part
            # whose df exceeds the cap before FTS walks its posting list. The
            # AND query is left untouched (intersection cost is bounded by its
            # rarest term). Fall back to the unpruned set when everything is
            # common, so recall never collapses to zero.
            df_cap = self._line_fts_df_cap(conn)
            if df_cap > 0 and len(or_parts) > 1:
                kept = [part for part in or_parts if self._line_fts_df(conn, part) <= df_cap]
                if kept:
                    or_parts = kept
            or_query = " OR ".join(or_parts)
            if len(terms) >= 2:
                and_rows = conn.execute(
                    """
                    SELECT file_path, line, text, bm25(file_line_fts) AS rank
                    FROM file_line_fts
                    WHERE file_line_fts MATCH ? AND repo_id = ?
                    ORDER BY rank ASC, file_path ASC, line ASC
                    LIMIT 200
                    """,
                    (and_query, self.repo_id),
                ).fetchall()
                rows.extend((row, True) for row in and_rows)
            or_rows = conn.execute(
                """
                SELECT file_path, line, text, bm25(file_line_fts) AS rank
                FROM file_line_fts
                WHERE file_line_fts MATCH ? AND repo_id = ?
                ORDER BY rank ASC, file_path ASC, line ASC
                LIMIT 600
                """,
                (or_query, self.repo_id),
            ).fetchall()
            rows.extend((row, False) for row in or_rows)
        except Exception:
            return [], {}
        term_set = set(terms)
        literal_set = {literal.lower() for literal in plan.literals}
        per_file: dict[str, dict[str, Any]] = {}
        for row, from_and in rows:
            file_path = str(row["file_path"] or "")
            if not file_path:
                continue
            skip_generated = False
            with contextlib.suppress(Exception):
                skip_generated = is_generated_path(file_path)
            if skip_generated:
                continue
            if _MINIFIED_FILE_RE.search(file_path):
                continue
            if _VENDOR_PATH_RE.search(file_path):
                continue
            text = str(row["text"] or "").lower()
            covered = set()
            for term in term_set:
                if term in text:
                    covered.add(term)
                else:
                    stem = term_stems.get(term)
                    if stem and stem in text:
                        covered.add(term)
            if not covered:
                continue
            line_coverage = len(covered) / max(1, len(term_set))
            literal_hits = sum(1 for literal in literal_set if literal in text)
            item = per_file.setdefault(
                file_path,
                {
                    "covered": set(),
                    "hit_count": 0,
                    "and_hit": False,
                    "best_rank": float(row["rank"] or 0.0),
                    "max_line_coverage": 0.0,
                    "multi_term_lines": 0,
                    "literal_hits": 0,
                },
            )
            item["covered"].update(covered)
            item["hit_count"] += 1
            item["and_hit"] = bool(item["and_hit"] or from_and)
            item["best_rank"] = min(float(item["best_rank"]), float(row["rank"] or 0.0))
            item["max_line_coverage"] = max(float(item["max_line_coverage"]), line_coverage)
            if len(covered) >= 2:
                item["multi_term_lines"] += 1
            item["literal_hits"] += literal_hits
        for file_path, item in per_file.items():
            file_coverage = len(item["covered"]) / max(1, len(term_set))
            repeat_conf = min(1.0, math.log1p(int(item["hit_count"])) / math.log(13.0))
            proximity_conf = min(
                1.0,
                float(item["max_line_coverage"]) + 0.08 * min(int(item["multi_term_lines"]), 4),
            )
            literal_conf = min(1.0, int(item["literal_hits"]) / max(1, len(literal_set))) if literal_set else 0.0
            confidence = (
                0.40 * file_coverage
                + 0.23 * proximity_conf
                + 0.09 * repeat_conf
                + 0.20 * float(bool(item["and_hit"]))
                + 0.08 * literal_conf
            )
            if not plan.wants_tests and _HEF_TEST_RE.search(file_path):
                confidence *= 0.78
            if not plan.wants_auxiliary and _HEF_AUX_RE.search(file_path):
                confidence *= 0.68
            item["file_coverage"] = file_coverage
            item["confidence"] = min(1.0, confidence)
        ordered = sorted(
            per_file,
            key=lambda p: (
                -float(per_file[p]["confidence"]),
                -float(per_file[p]["file_coverage"]),
                -float(per_file[p]["max_line_coverage"]),
                -int(per_file[p]["and_hit"]),
                float(per_file[p]["best_rank"]),
                p,
            ),
        )
        details = {
            p: {
                "covered": sorted(per_file[p]["covered"]),
                "file_coverage": round(float(per_file[p]["file_coverage"]), 6),
                "max_line_coverage": round(float(per_file[p]["max_line_coverage"]), 6),
                "multi_term_lines": int(per_file[p]["multi_term_lines"]),
                "hit_count": int(per_file[p]["hit_count"]),
                "and_hit": bool(per_file[p]["and_hit"]),
                "literal_hits": int(per_file[p]["literal_hits"]),
                "best_rank": round(float(per_file[p]["best_rank"]), 6),
                "confidence": round(float(per_file[p]["confidence"]), 6),
            }
            for p in ordered
        }
        return ordered[:128], details

    def _line_fts_df_cap(self, conn: sqlite3.Connection) -> int:
        """Document-frequency cap for line-FTS OR terms.

        LEMONCROW_LINE_FTS_DF_CAP: explicit integer cap, 0 disables pruning.
        Unset = auto: 2% of this repo's indexed line count, floored at 2000 so
        small repos never prune (their posting lists are cheap anyway).
        """
        raw = os.environ.get("LEMONCROW_LINE_FTS_DF_CAP")
        if raw is not None:
            with contextlib.suppress(ValueError):
                return max(0, int(raw))
            return 0
        cached = self._line_fts_total_cache
        if cached is None:
            try:
                cached = int(
                    conn.execute("SELECT COUNT(*) FROM file_line_fts WHERE repo_id = ?", (self.repo_id,)).fetchone()[0]
                )
            except sqlite3.Error:
                cached = 0
            self._line_fts_total_cache = cached
        return max(2000, cached // 50)

    def _line_fts_df(self, conn: sqlite3.Connection, match_expr: str) -> int:
        """Cached per-term document frequency in file_line_fts.

        A COUNT walk is ~8x cheaper than the bm25-scored walk for the same
        term and is paid once per (engine, term): common words repeat across
        queries, so the cache amortises to ~zero. The cache is not
        index-version-keyed -- df is a pruning heuristic, and drifting counts
        after a reindex only shift which near-zero-IDF terms get dropped.
        """
        cache = self._line_fts_df_cache
        hit = cache.get(match_expr)
        if hit is not None:
            return hit
        try:
            df = int(
                conn.execute(
                    "SELECT COUNT(*) FROM file_line_fts WHERE file_line_fts MATCH ? AND repo_id = ?",
                    (match_expr, self.repo_id),
                ).fetchone()[0]
            )
        except sqlite3.Error:
            df = 0  # unparsable/odd token: keep it (0 <= any cap)
        cache[match_expr] = df
        return df

    def _submit_hef_channels(
        self,
        query: str,
    ) -> tuple[
        _HefQueryPlan, concurrent.futures.Future[Any], concurrent.futures.Future[Any], concurrent.futures.Future[Any]
    ]:
        """Launch the three V6 HEF recall channels on the persistent thread pool.

        Returns the parsed plan plus the three in-flight futures (exact / anchor /
        line).  Called at the very start of ``tool_explore`` so the channels run
        concurrently with the baseline + main-Zoekt phase; ``_fused_explore_hybrid``
        later collects them.  The channels depend only on the parsed query plan, so
        starting them early is purely a latency win (identical results).
        """
        plan = self._hef_parse_query(query)
        # Symbol-exact fast path: for definition/symbol-intent queries, resolve
        # the cheap exact channel inline (~1-2ms indexed lookup) and consult the
        # gate. When the exact hits are decisive, anchor-Zoekt and line-FTS are
        # never submitted — fusion sees them as deterministically-empty channels
        # instead of paying their scan time (the dominant explore cost).
        # Pre-resolved futures keep _fused_explore_hybrid's collect path uniform.
        if _hef_fast_path_enabled() and plan.intent in _hef_fast_path_intents():
            exact = self._hef_exact_symbol_candidates(plan)
            f_exact_done: concurrent.futures.Future[Any] = concurrent.futures.Future()
            f_exact_done.set_result(exact)
            decisive = _hef_exact_is_decisive(query, plan, *exact)
            key = (decisive, plan.intent)
            _HEF_FAST_PATH_COUNTS[key] = _HEF_FAST_PATH_COUNTS.get(key, 0) + 1
            if decisive:
                f_anchor_empty: concurrent.futures.Future[Any] = concurrent.futures.Future()
                f_anchor_empty.set_result(([], {}))
                f_line_empty: concurrent.futures.Future[Any] = concurrent.futures.Future()
                f_line_empty.set_result(([], {}))
                return plan, f_exact_done, f_anchor_empty, f_line_empty
            f_anchor = _HEF_CHANNEL_EXECUTOR.submit(self._hef_anchor_zoekt_candidates, plan)
            f_line = _HEF_CHANNEL_EXECUTOR.submit(self._hef_line_fts_candidates, plan)
            return plan, f_exact_done, f_anchor, f_line
        f_exact = _HEF_CHANNEL_EXECUTOR.submit(self._hef_exact_symbol_candidates, plan)
        f_anchor = _HEF_CHANNEL_EXECUTOR.submit(self._hef_anchor_zoekt_candidates, plan)
        f_line = _HEF_CHANNEL_EXECUTOR.submit(self._hef_line_fts_candidates, plan)
        return plan, f_exact, f_anchor, f_line

    def _fused_explore_hybrid(
        self,
        query: str,
        baseline_payload: dict[str, Any],
        max_files: int,
        precomputed_zoekt: list[str] | None = None,
        hef_futures: (
            tuple[
                _HefQueryPlan,
                concurrent.futures.Future[Any],
                concurrent.futures.Future[Any],
                concurrent.futures.Future[Any],
            ]
            | None
        ) = None,
    ) -> dict[str, Any]:
        """Apply the V6 multi-channel fusion on top of the baseline explore result.

        Adds exact-symbol, anchor-Zoekt, and line-FTS channels to the baseline
        and Zoekt results, fuses them with intent-aware RRF + confidence weights,
        and returns a new payload with the fused file list.

        ``precomputed_zoekt`` is the already-fetched Zoekt file list from
        ``_tool_explore_impl`` (same query, up to 96 files).  When provided it
        is reused directly, skipping a redundant second Zoekt search.
        """
        raw_files = baseline_payload.get("files")
        # Don't short-circuit when baseline is empty: the non-baseline channels
        # (zoekt, exact, anchor, line) can still surface the gold file even when
        # the FTS symbol search returns nothing (e.g. rare grammar-rule functions,
        # module-level constants not indexed as symbols, or deep-module definitions
        # that the centrality-ranked symbol search buries below the result cap).
        # Only bail out if the payload is genuinely malformed (not a list).
        if not isinstance(raw_files, list):
            return baseline_payload

        baseline_entries: dict[str, dict[str, Any]] = {}
        baseline_files: list[str] = []
        for entry in raw_files:
            if not isinstance(entry, dict):
                continue
            file_path = str(entry.get("path") or entry.get("file_path") or "")
            if file_path and file_path not in baseline_entries:
                baseline_entries[file_path] = entry
                baseline_files.append(file_path)

        # Get full Zoekt results (up to 96) independent of baseline truncation.
        # Reuse the list already fetched by _tool_explore_impl when available so
        # we never search the same query twice on the same code_search call.
        if precomputed_zoekt is not None:
            full_zoekt: list[str] = precomputed_zoekt
        else:
            full_zoekt = []
            with contextlib.suppress(Exception):
                full_zoekt = self._zoekt_candidate_files(query, path=".", max_files=96)

        # The three independent V6 channels run on the persistent _HEF_CHANNEL_EXECUTOR
        # (module-level ThreadPool — no spawn/join per query).  When the caller started
        # them up front (the normal tool_explore path) they have already been running
        # concurrently with the baseline phase; otherwise submit them now.  A 200 ms
        # shared deadline releases the caller immediately; slow threads finish in the
        # background and return their slot to the pool.
        if hef_futures is not None:
            plan, _f_exact, _f_anchor, _f_line = hef_futures
        else:
            plan, _f_exact, _f_anchor, _f_line = self._submit_hef_channels(query)
        _done, _ = concurrent.futures.wait([_f_exact, _f_anchor, _f_line], timeout=0.200)
        exact_files, exact_details = _f_exact.result() if _f_exact in _done else ([], {})
        anchor_files, anchor_details = _f_anchor.result() if _f_anchor in _done else ([], {})
        line_files, line_details = _f_line.result() if _f_line in _done else ([], {})

        # Selective gate: the broad Zoekt channel would only feed fusion for
        # query shapes where it helps (regex/pattern, multi-word).  The targeted
        # per-anchor Zoekt channel stays on regardless.  Enforcement is opt-in
        # (see _zoekt_gate_enforced); telemetry records the decision either way.
        broad_decision, gate_reason = _zoekt_broad_gate(query)
        gate_enforced = _zoekt_gate_enforced()
        broad_admit = broad_decision or not gate_enforced
        _zoekt_gate_record(
            query, plan.intent, broad_decision, gate_reason, len(full_zoekt), len(anchor_files), enforced=gate_enforced
        )
        channels: dict[str, list[str]] = {
            "baseline": baseline_files,
            "zoekt": full_zoekt if broad_admit else [],
            "exact": exact_files,
            "anchors": anchor_files,
            "line": line_files,
        }
        rank_weights: dict[str, float] = {
            "definition": {"baseline": 1.0, "zoekt": 0.9, "exact": 1.4, "anchors": 1.1, "line": 1.2},
            "symbol": {"baseline": 1.2, "zoekt": 1.1, "exact": 1.4, "anchors": 1.1, "line": 0.8},
            "code": {"baseline": 0.9, "zoekt": 0.9, "exact": 0.9, "anchors": 1.1, "line": 1.5},
            "prose": {"baseline": 0.8, "zoekt": 1.0, "exact": 0.3, "anchors": 0.4, "line": 1.8},
        }[plan.intent]
        confidence_weights: dict[str, float] = {
            "definition": {"exact": 1.25, "anchors": 0.60, "line": 1.05},
            "symbol": {"exact": 1.10, "anchors": 0.55, "line": 0.55},
            "code": {"exact": 0.70, "anchors": 0.75, "line": 1.20},
            "prose": {"exact": 0.15, "anchors": 0.20, "line": 1.40},
        }[plan.intent]

        scores: dict[str, float] = {}
        channel_ranks: dict[str, dict[str, int]] = {}
        rrf_k = 8.0
        for channel, files in channels.items():
            channel_ranks[channel] = {p: r for r, p in enumerate(files, 1)}
            weight = rank_weights[channel]
            for rank, file_path in enumerate(files, 1):
                scores[file_path] = scores.get(file_path, 0.0) + weight / (rrf_k + rank)

        for file_path, detail in exact_details.items():
            # Only apply the exact channel's additive confidence bonus to a real
            # definition-site hit (kind_matches or definition_tokens non-empty).
            # A match that's neither -- a bare reference/usage of an
            # incidentally-named identifier, not a definition -- still gets its
            # plain RRF contribution from the channel-rank loop above, but no
            # longer also gets this large additive boost, which could swamp a
            # correct baseline/semantic hit that has no equivalent bonus.
            # Confirmed via a real linux trace: a single-token, non-definitional
            # "afs" reference match scored confidence=0.54 -> a 0.378 bonus that
            # beat a semantically-correct file's entire 0.100 RRF score ~5x.
            if not (detail.get("kind_matches") or detail.get("definition_tokens")):
                continue
            scores[file_path] = scores.get(file_path, 0.0) + confidence_weights["exact"] * float(detail["confidence"])
        for file_path, detail in anchor_details.items():
            scores[file_path] = scores.get(file_path, 0.0) + confidence_weights["anchors"] * float(detail["confidence"])
        for file_path, detail in line_details.items():
            scores[file_path] = scores.get(file_path, 0.0) + confidence_weights["line"] * float(detail["confidence"])

        top_sets = {channel: set(files[:32]) for channel, files in channels.items()}
        explicit_names = {name.lower() for _kind, name in plan.definitions}
        identifier_names = {token.lower() for token in plan.identifiers}

        for file_path in list(scores):
            support = sum(file_path in ts for ts in top_sets.values())
            if support >= 2:
                scores[file_path] += 0.08 * (support - 1)
            exact_detail = exact_details.get(file_path, {})
            kind_matches = set(exact_detail.get("kind_matches", ()))
            if explicit_names and kind_matches:
                scores[file_path] += 0.90 * len(explicit_names & kind_matches)
            path_overlap = len(identifier_names & _hef_path_parts(file_path))
            if path_overlap:
                scores[file_path] += 0.035 * path_overlap
            if not plan.wants_tests and _HEF_TEST_RE.search(file_path) and file_path not in baseline_entries:
                scores[file_path] *= 0.78
            if not plan.wants_auxiliary and _HEF_AUX_RE.search(file_path) and file_path not in baseline_entries:
                scores[file_path] *= 0.68
            if file_path.endswith(".pyi") and not re.search(r"\bpyi|stub\b", query, re.I):
                scores[file_path] *= 0.82

        ordered_all = sorted(
            scores,
            key=lambda p: (
                -scores[p],
                channel_ranks["baseline"].get(p, 10_000),
                channel_ranks["zoekt"].get(p, 10_000),
                channel_ranks["exact"].get(p, 10_000),
                channel_ranks["anchors"].get(p, 10_000),
                channel_ranks["line"].get(p, 10_000),
                p,
            ),
        )
        # Filetype diversity: prose-y queries let documentation (.md/.rst)
        # headings monopolize the ranked window. Cap docs to a quarter of the
        # head and demote the excess below it -- a stable permutation, never a
        # drop (demoted docs stay in the recall tail). Skipped when the query
        # itself asks for docs/aux material.
        if not plan.wants_auxiliary:
            ordered_all = demote_doc_overflow(ordered_all, window=min(max_files, len(ordered_all)))
        ordered = ordered_all[:max_files]

        fused_entries: list[dict[str, Any]] = []
        for file_path in ordered:
            existing = baseline_entries.get(file_path)
            if existing is not None:
                fused_entries.append(existing)
            else:
                fused_entries.append(
                    {
                        "path": file_path,
                        "language": "unknown",
                        "symbols": [],
                        "source_sections": [],
                    }
                )

        result = dict(baseline_payload)
        result["files"] = fused_entries
        # Recall tail: the next-best fused files beyond the source cap. They carry
        # real cross-channel evidence -- line-FTS body coverage and exact-symbol
        # matches -- for concept / natural-language queries whose gold file has no
        # top-scoring *symbol* hit, so the symbol-BM25 file ranking (which the lean
        # view uses) buries it below the truncation. Exposed as extra candidate
        # paths that the lean view appends STRICTLY AFTER the primary candidates,
        # so the top result is unchanged (no precision regression) while recall for
        # the harder query shapes improves. Recall-only files (empty source, hence
        # no entry-point score) that fusion ranked high are exactly what this
        # rescues; it is free (the ordering is already computed above).
        result["fused_recall"] = ordered_all[max_files : max_files + _FUSED_RECALL_TAIL]
        result["experiment"] = {
            "name": "fused_explore_hybrid_v6",
            "intent": plan.intent,
            "base": baseline_payload.get("experiment"),
            "zoekt_gate": {
                "broad_admitted": broad_admit,
                "decision": broad_decision,
                "enforced": gate_enforced,
                "reason": gate_reason,
                "zoekt_n": len(full_zoekt),
                "anchor_n": len(anchor_files),
            },
        }
        return result

    @staticmethod
    def _lexical_search_enabled() -> bool:
        """Return False when LEMONCROW_EXPLORE_LEXICAL=0 disables FTS5 lexical search.

        Lets eval benchmarks run pure-zoekt or pure-semantic channels by turning off
        the baseline FTS5 symbol search.  Default enabled.
        """
        return os.environ.get("LEMONCROW_EXPLORE_LEXICAL", "1") != "0"

    @staticmethod
    def _exact_name_fallback_enabled() -> bool:
        """Gate for exact-name fallback search_symbols calls in _tool_explore_impl.

        Tied to _lexical_search_enabled since those fallbacks also run lexical FTS5.
        """
        return CodeContextEngine._lexical_search_enabled()

    def _tool_explore_impl(
        self,
        query: str,
        *,
        seed_files: list[str] | None = None,
        max_files: int = 6,
        max_symbols: int = 20,
        include_source: bool = True,
        include_relationships: bool = False,
        line_numbers: bool = True,
        skeletonize: bool = True,
        complete_families: bool | None = None,
        depth: int = 1,
        budget_tokens: int = 9000,
    ) -> tuple[dict[str, Any], list[str]]:
        effective_skeletonize = skeletonize and _explore_skeleton_enabled()
        effective_complete = (
            bool(complete_families if complete_families is not None else skeletonize) and _explore_skeleton_enabled()
        )

        bounded_max_symbols = max(1, min(max_symbols, 30))
        bounded_max_files = max(1, min(max_files, 8))
        bounded_depth = max(1, depth)
        normalized_seeds = [self._normalize_file_arg(seed) for seed in seed_files or []]
        seed_set = set(normalized_seeds)
        # When the caller passes a single *file* (not directory) as the scope,
        # restrict FTS5 to that file so its symbols surface even when globally-
        # higher-scoring symbols from other files would crowd them out.  Computed
        # here so the cache_args key reflects the scoping behaviour change.
        _single_file_seed = (
            normalized_seeds[0]
            if len(normalized_seeds) == 1 and not (Path(self.repo_root) / normalized_seeds[0]).is_dir()
            else None
        )
        cache_args = {
            "query": query,
            "seed_files": normalized_seeds,
            # v6: soft paths= scope everywhere -- in-scope symbols reserve only the
            # top-2 rank slots (seed_set no longer blanket-sorts / floor-bypasses),
            # and single-file seeds also inject their defs + run reference expansion.
            # Bumped for every seeded shape (single-file vs dir/multi) so the cache
            # reflects the reserve-2 ranking; unseeded (no seed_set) stays unchanged.
            "_scope_v": (6 if _single_file_seed else 2) if seed_set else 1,
            "max_files": bounded_max_files,
            "max_symbols": bounded_max_symbols,
            "include_source": include_source,
            "include_relationships": include_relationships,
            "line_numbers": line_numbers,
            "skeletonize": effective_skeletonize,
            "complete_families": effective_complete,
            "depth": bounded_depth,
            "budget_tokens": budget_tokens,
        }
        hit, cached = self._cache_get("code.explore", cache_args)
        if hit and cached is not None:
            return self._mark_cache_hit(cached), []

        # Pipe-separated queries: pre-compute the extra word set for token-pin
        # ranking.  Actual OR-expansion happens lazily after the main search returns
        # (see the merge block after the parallel phase) so it only runs when the
        # primary FTS AND search yields too few results, avoiding serial overhead on
        # queries where FTS AND already finds the right file.
        _pipe_query_extra_words: frozenset[str] = frozenset()
        _pipe_clean_terms: list[str] = []
        _pipe_extra_symbols: list[SymbolRecord] = []
        if "|" in query:
            _pipe_terms = _split_pipe_query(query)
            if _pipe_terms:
                _pipe_query_extra_words = frozenset(re.sub(r"[^A-Za-z0-9_.]", "", t) for t in _pipe_terms) - {""}
                _pipe_clean_terms = sorted((t for t in _pipe_query_extra_words if len(t) >= 3), key=len, reverse=True)[
                    :5
                ]

        # Winner-pipeline fusion: pull in trigram (zoekt) AND semantic anchors. The
        # symbol FTS ranks named symbols well but misses concept/regex queries where
        # the right file has no lexically-matching symbol name; zoekt's trigram search
        # and the embedding nearest-neighbours surface those files, and seeding a
        # couple of their definitions makes the file survive ranking. Both are
        # additive and graceful (empty when zoekt is off / no embedder configured) --
        # the exact/score/seed ranking below still governs final order.
        #
        # The three pipelines are independent: the FTS5 search channels run in a
        # dedicated process pool (true CPU parallelism without GIL contention), while
        # the Zoekt recall runs on a worker thread whose HTTP I/O releases the GIL.
        # Wall-clock collapses from sum-of-channels to slowest-channel. Opt out with
        # LEMONCROW_EXPLORE_PARALLEL=0.
        anchor_budget = max(bounded_max_files * 2, 12)
        # Execution gate: for plain single-identifier queries the broad Zoekt
        # channel mostly displaces a file lexical+centrality already ranked #1,
        # so when enforced (LEMONCROW_ZOEKT_GATE) we SKIP the broad search entirely
        # — saving its ~HTTP round-trip — and let FTS symbols + the targeted
        # per-anchor Zoekt channel carry the query.
        _broad_admit = (not _zoekt_gate_enforced()) or _zoekt_broad_gate(query)[0]
        # {file_path: best cosine} from the semantic channel; used both as recall
        # anchors AND (the fix) as a ranking signal below so semantic-surfaced
        # files aren't scored ~0 on lexical/centrality and buried.
        _sem_scores: dict[str, float] = {}
        if os.environ.get("LEMONCROW_EXPLORE_PARALLEL", "1") != "0":
            from concurrent.futures import ThreadPoolExecutor

            # Do NOT use "with _pool" here: its __exit__ calls shutdown(wait=True)
            # which blocks until zoekt/semantic threads finish even if we've
            # already timed out and moved on.  shutdown(wait=False) lets slow
            # threads drain in the background (they terminate within 2 s once
            # the HTTP timeout fires in _WEBSERVER_REQUEST_TIMEOUT_SECONDS).
            _pool = ThreadPoolExecutor(max_workers=2)
            _sem_t0 = time.monotonic()  # stamp before submitting so deadline is from T=0
            # Fetch up to 96 files so _fused_explore_hybrid can reuse this
            # list without a redundant second search call on the same query.
            _zk_fetch_limit = max(anchor_budget, 30)
            _zk = _pool.submit(self._zoekt_candidate_files, query, max_files=_zk_fetch_limit) if _broad_admit else None
            _sem = _pool.submit(self._semantic_candidate_files, query, max_files=anchor_budget)
            # Pass _candidate_files=set() to skip the duplicate zoekt call inside
            # search_symbols -- we already have _zk running in the pool above.
            # search_symbols returns immediately (~50-150 ms), freeing the 0.8 s
            # zoekt timeout budget for the _zk future already in flight.
            # LEMONCROW_EXPLORE_LEXICAL=0 skips FTS5 entirely (eval pure-channel mode).
            raw_symbols = (
                self.search_symbols(
                    query,
                    limit=bounded_max_symbols,
                    snippet="none",
                    auto_index=False,
                    _candidate_files=set(),
                )
                if self._lexical_search_enabled()
                else []
            )
            # Collect zoekt results; cap at 800 ms.  By the time search_symbols
            # returns (~100 ms), zoekt has had most of that budget already.
            _zk_list: list[str] = []
            if _zk is not None:
                try:
                    _zk_list = _zk.result(timeout=0.8)
                except (TimeoutError, concurrent.futures.CancelledError):
                    _zk_list = []
            # _zk_list may have up to 96 entries (for the fusion channel).
            # Limit baseline anchors to anchor_budget to match V6 behaviour:
            # the capturing wrapper in the experiment only returned the first
            # anchor_budget files to _tool_explore_impl, keeping baseline scoring
            # calibrated while the fusion channel still sees the full 96.
            _seen_anchors: dict[str, None] = dict.fromkeys(_zk_list[:anchor_budget])
            try:
                # _sem has been running since T=0 in parallel with search_symbols
                # (~100ms) and zoekt (~20-50ms).  Give it a total budget of 500ms
                # from submission: enough for cached-embed + linux TurboQuant ANN
                # (~200ms), while still bounding worst-case interactive latency.
                # Minimum 50ms floor so the common small-repo case (5ms ANN) never
                # waits unnecessarily.
                _sem_remaining = max(0.05, 0.5 - (time.monotonic() - _sem_t0))
                _sem_scores = _sem.result(timeout=_sem_remaining)
                for _f in _sem_scores:
                    _seen_anchors.setdefault(_f, None)
            except (TimeoutError, concurrent.futures.CancelledError):
                pass
            _pool.shutdown(wait=False)  # don't block; slow threads drain in bg
            anchor_candidates = list(_seen_anchors)
        else:
            # LEMONCROW_EXPLORE_LEXICAL=0 skips FTS5 entirely (eval pure-channel mode).
            raw_symbols = (
                self.search_symbols(
                    query,
                    limit=bounded_max_symbols,
                    snippet="none",
                    auto_index=False,
                )
                if self._lexical_search_enabled()
                else []
            )
            # Preserve zoekt's score-ranked order; append semantic-only files at the end.
            # Limit baseline anchors to anchor_budget (same as V6 capturing_zoekt behaviour).
            _zk_list = self._zoekt_candidate_files(query, max_files=max(anchor_budget, 30)) if _broad_admit else []
            _seen_anchors_s: dict[str, None] = dict.fromkeys(_zk_list[:anchor_budget])
            _sem_scores = self._semantic_candidate_files(query, max_files=anchor_budget)
            for _f in _sem_scores:
                _seen_anchors_s.setdefault(_f, None)
            anchor_candidates = list(_seen_anchors_s)
        # Single-file scope: restrict results to that file only.
        # FTS5 indexes the full symbol body (start_byte:end_byte), so concept
        # queries do match body content. However, a body-match in a DIFFERENT
        # file would crowd out the target file's symbols. Instead:
        #   1. Discard FTS5 hits from OTHER files.
        #   2. Inject all definition symbols from the seed file directly so
        #      every definition is visible even when the query has no FTS match.
        #      Injected symbols bypass the score floor via seed_set.
        #   3. Sort injected symbols by name-query token overlap so the most
        #      relevant definitions surface first.
        if _single_file_seed:
            anchor_candidates = [f for f in anchor_candidates if f == _single_file_seed]
            seed_fts_hits = [s for s in raw_symbols if s.file_path == _single_file_seed]
            _qt = set(re.split(r"\W+", query.lower())) - {"", "the", "a", "in", "of"}
            # Score each injected symbol by name-token overlap so the ranking
            # sort (-(score or 0.0), start_line) surfaces query-relevant symbols
            # first rather than just the earliest-in-file ones.
            seed_file_syms = [
                s.model_copy(update={"score": float(sum(1 for t in _qt if t in (s.symbol_name or "").lower())) * 0.001})
                for s in self._symbols_for_files([_single_file_seed], limit=5000)
                if (s.kind or "").lower() in _DEFINITION_KINDS
            ]
            raw_symbols = self._dedupe_symbols(seed_fts_hits + seed_file_syms)
        # Exact-name guard + anchor gate. When the query is itself an indexed symbol
        # name, the definition (+ its family) is the answer, so two things happen:
        # (1) the exact match is pinned to the front (semantic/lexical ranking can
        # otherwise bury it behind cousins or drop it past the cap), and (2) the
        # zoekt/semantic anchor recall is SKIPPED -- for an exact hit those channels
        # only flood the payload with the many files that merely *reference* the
        # name (grep-style over-recall, the dominant explore bloat). Anchors stay on
        # for concept queries (no exact hit), where they are the recall mechanism.
        # Only symbol-like (single-token) queries pay the extra lexical lookup.
        exact_hits = _exact_symbol_hits(raw_symbols, query)
        # Save the strict full-query exact match for the anchor gate below.
        # The multi-word token probe (elif branch) overwrites exact_hits with
        # token-level matches — using those would skip the anchor merge even
        # though the full query is not an indexed symbol name (e.g. a multi-word
        # query whose only exact match is a sub-token like "aggregate_session_stats"
        # inside "def aggregate_session_stats").  The anchor gate should only
        # fire when the query itself is an exact symbol name.
        _anchor_gate_exact_hits = True if exact_hits else False
        # Exact-name fallback probes also run FTS5, so skip when lexical is
        # disabled (LEMONCROW_EXPLORE_LEXICAL=0, eval pure-channel mode).
        if self._exact_name_fallback_enabled():
            # _exact_name_exists gates each probe search below: no indexed name
            # == no possible exact hit, so the 10-15ms search is provably moot.
            if not exact_hits and _SYMBOL_QUERY_RE.match(query.strip()) and self._exact_name_exists(query):
                lexical_hits = self.search_symbols(
                    query,
                    limit=max(bounded_max_symbols, 10),
                    mode="lexical",
                    snippet="none",
                    auto_index=False,
                    _candidate_files=set(),  # zoekt already ran above; skip duplicate
                )
                exact_hits = _exact_symbol_hits(lexical_hits, query)
            elif not exact_hits and " " in query.strip():
                # Multi-word query: probe each compound-identifier token for an
                # exact symbol-name match. FTS5 BM25 tokenises on underscores, so
                # a test class that references sub-tokens many times can outscore
                # the one definition. E.g. 'trim_docstring admindocs' → probe
                # 'trim_docstring' → pin utils.py::trim_docstring to the front so
                # it survives the floor and appears first.
                token_hits: list[SymbolRecord] = []
                seen_ids: set[str] = set()
                for token in query.strip().split():
                    if not _SYMBOL_QUERY_RE.match(token):
                        continue
                    # Strip class prefix: "DataArray.quantile" → probe "quantile".
                    # The compound-identifier guard applies to the probe name so
                    # plain English words are still skipped, but "Variable.quantile"
                    # is no longer dropped because the dot broke the CamelCase check.
                    has_dot = "." in token
                    probe = token.rsplit(".", 1)[-1] if has_dot else token
                    if len(probe) <= 3:
                        continue
                    # For plain (non-dotted) tokens keep the compound-ident filter so
                    # bare English words like "find" or "get" are not probed.
                    if not has_dot and not _COMPOUND_IDENT_RE.search(token):
                        continue
                    if not self._exact_name_exists(probe):
                        continue
                    lhits = self.search_symbols(
                        probe,
                        limit=max(bounded_max_symbols, 10),
                        mode="lexical",
                        snippet="none",
                        auto_index=False,
                        _candidate_files=set(),  # zoekt already ran above; skip duplicate
                    )
                    for r in _exact_symbol_hits(lhits, probe):
                        if r.symbol_id not in seen_ids:
                            seen_ids.add(r.symbol_id)
                            token_hits.append(r)
                exact_hits = token_hits
        exact_ids = {record.symbol_id for record in exact_hits}
        # The top exact hit (exact_hits is pinned strongest-first) is returned
        # effectively whole (see _EXPLORE_SOURCE_SECTION_EXACT_MAX_TOKENS): it is
        # the symbol the agent named, so truncating its tail only forces the
        # re-read code_search exists to avoid.
        _top_exact_id = exact_hits[0].symbol_id if exact_hits else None
        # No extra searches needed for pipe queries: _pipe_query_extra_words is
        # already merged into _query_words for token-pin ranking, and the file-level
        # coverage boost below rewards files that match more pipe terms.
        anchor_ids: set[str] = set()
        if not _anchor_gate_exact_hits:
            seeded_files = {symbol.file_path for symbol in raw_symbols}
            # anchor_candidates is zoekt-score ordered (highest relevance first);
            # do NOT sort alphabetically — that would bury high-signal files.
            anchor_files = [anchor for anchor in anchor_candidates if anchor not in seeded_files]
            if anchor_files:
                anchor_symbols = self._cap_symbols_per_file(
                    [
                        symbol
                        for symbol in self._symbols_for_files(anchor_files[:bounded_max_files], limit=400)
                        if (symbol.kind or "").lower() in _DEFINITION_KINDS
                    ],
                    max_per_file=2,
                )
                anchor_ids = {symbol.symbol_id for symbol in anchor_symbols}
                raw_symbols = self._dedupe_symbols(raw_symbols + anchor_symbols)
        # ── Reference-file expansion ──────────────────────────────────────
        # For concept queries (no exact hit yet), find files that REFERENCE
        # the top-ranked FTS symbols.  Catches the "subclass override" pattern:
        # FTS finds the base-class definition (e.g. base/base.py for
        # timezone_name) but the gold file is the concrete backend override
        # (e.g. mysql/operations.py) which only REFERENCES that symbol.
        # Uses the intel.sqlite 'references' table which is already attached
        # to the shared connection - no extra connection overhead.
        _ref_anchor_ids: set[str] = set()
        # Single-file `paths=` scope is a SOFT prioritisation: reference-file
        # expansion is exactly how neighbouring files that reference the seed's
        # symbols surface from rank #2 onward, so it runs for the seed case too
        # (the seed itself stays pinned to rank #1 -- see tool_explore).
        if not exact_hits:
            _top_sym_names = list(
                dict.fromkeys(
                    r.symbol_name
                    for r in raw_symbols[:24]
                    if r.symbol_name
                    and len(r.symbol_name) >= 5
                    and not r.symbol_name.lower().startswith(("test_", "assert_", "mock_"))
                )
            )[:6]
            if _top_sym_names:
                try:
                    _ref_ph = ",".join("?" * len(_top_sym_names))
                    with self._connect(readonly=True) as _rconn:
                        _ref_file_rows = _rconn.execute(
                            f"SELECT file_path, COUNT(DISTINCT symbol_name) AS cnt "
                            f'FROM "references" '
                            f"WHERE repo_id=? AND symbol_name IN ({_ref_ph}) "
                            f"GROUP BY file_path ORDER BY cnt DESC LIMIT 12",
                            [self.repo_id, *_top_sym_names],
                        ).fetchall()
                    _known_ref_fps = {r.file_path for r in raw_symbols if r.file_path}
                    _new_ref_fps = [row[0] for row in _ref_file_rows if row[0] not in _known_ref_fps][:8]
                    if _new_ref_fps:
                        _ref_syms = self._cap_symbols_per_file(
                            [
                                s
                                for s in self._symbols_for_files(_new_ref_fps, limit=200)
                                if (s.kind or "").lower() in _DEFINITION_KINDS
                            ],
                            max_per_file=2,
                        )
                        _ref_anchor_ids = {s.symbol_id for s in _ref_syms}
                        raw_symbols = self._dedupe_symbols(raw_symbols + _ref_syms)
                except (KeyError, TypeError):
                    pass
        if exact_hits:
            raw_symbols = exact_hits + [record for record in raw_symbols if record.symbol_id not in exact_ids]
        # Token-level exact pinning: any whitespace-delimited word in the query that
        # exactly matches a symbol name is pinned alongside full-query exact hits.
        # FTS5 BM25 misses this for multi-word queries because test files accumulate
        # far more term hits in assertion bodies than the single definition file.
        # Split on whitespace AND pipe (|) so pipe-separated multi-term queries
        # like "_get_timezone_name|get_current_timezone_name" correctly yield each
        # term as a candidate for exact-symbol pinning.
        # Merge original pipe terms so symbols matching any pipe-token are pinned
        # even though the primary FTS search used only the longest identifier.
        _query_words = frozenset(re.split(r"[\s|]+", query.strip())) | _pipe_query_extra_words
        token_exact_ids = {
            r.symbol_id for r in raw_symbols if r.symbol_name in _query_words or r.symbol_name.lower() in _query_words
        }
        # Pipe-query file coverage: count how many DISTINCT pipe terms have an exact
        # symbol-name match in each file.  Files that cover more pipe terms (e.g.
        # `expressions.py` matching both ExpressionWrapper and DurationField) rank
        # ahead of files that cover only one term.  Free: no extra searches needed.
        _file_pipe_coverage: dict[str, int] = {}
        if _pipe_query_extra_words:
            _fp_terms: dict[str, set[str]] = {}
            for _r in raw_symbols:
                _rname = _r.symbol_name or ""
                _rlow = _rname.lower()
                for _w in _pipe_query_extra_words:
                    if _rname == _w or _rlow == _w.lower():
                        _fp = _r.file_path or ""
                        if _fp:
                            _fp_terms.setdefault(_fp, set()).add(_w)
            _file_pipe_coverage = {fp: len(ws) for fp, ws in _fp_terms.items()}
        # ── Semantic score fusion (the fix) ──────────────────────────────────
        # Give every symbol in a semantically-matched file a rank boost
        # proportional to that file's cosine, so a file the embedder ranked #1
        # competes in the final ordering instead of sinking to the tail. Before
        # this, semantic contributed recall only: the file was added to the
        # candidate pool but scored ~0 on lexical/centrality and lost the
        # -(score) sort. Purely additive -- never demotes a strong lexical hit,
        # only lifts semantic-surfaced files. Weight tunable for calibration.
        if _sem_scores:
            _sem_w = float(os.environ.get("LEMONCROW_SEMANTIC_RANK_WEIGHT", "120"))
            raw_symbols = [
                (
                    record.model_copy(update={"score": (record.score or 0.0) + _sem_scores[record.file_path] * _sem_w})
                    if record.file_path in _sem_scores
                    else record
                )
                for record in raw_symbols
            ]
        # Path-quality filter FIRST: hard-remove minified/vendor artefacts and
        # soft-penalise test files BEFORE the ranking sort and the score floor.
        # The penalty must precede the sort or it can never demote a test file:
        # test symbols often score highest (query terms repeat in assertion
        # bodies), and the sort order fixed here also seeds the stable
        # _explore_priority tie-break below. Pinned exact hits and seed files
        # are exempt.
        query_wants_tests = bool(re.search(r"\btest\b|\bspec\b", query, re.IGNORECASE))
        pinned_ids = exact_ids | anchor_ids | token_exact_ids | _ref_anchor_ids
        pre_filtered: list[SymbolRecord] = []
        for record in raw_symbols:
            fp = record.file_path or ""
            if _MINIFIED_FILE_RE.search(fp) or _VENDOR_PATH_RE.search(fp):
                if record.symbol_id not in pinned_ids and fp not in seed_set:
                    continue  # hard remove before floor
            if not query_wants_tests and _TEST_PATH_RE.search(fp):
                if record.symbol_id not in pinned_ids and fp not in seed_set:
                    record = record.model_copy(update={"score": (record.score or 0.0) * _TEST_SCORE_PENALTY})
            pre_filtered.append(record)
        raw_symbols = pre_filtered
        # Seed reservation cap: an explicit paths= scope guarantees its in-scope
        # symbols only the TOP-2 result spots (ranks #1-#2). Pick the 2 strongest
        # in-scope symbols (exact/pipe/score order) to reserve; every other seed
        # symbol races on pure ranking below, exactly like a neighbouring file --
        # no blanket seed-first sort, no unconditional floor bypass.
        _seed_ranked = sorted(
            (r for r in raw_symbols if r.file_path in seed_set),
            key=lambda r: (
                0 if (r.symbol_id in exact_ids or r.symbol_id in token_exact_ids) else 1,
                -_file_pipe_coverage.get(r.file_path or "", 0),
                -(r.score or 0.0),
                r.file_path or "",
                r.start_line,
            ),
        )
        _reserved_seed_ids = {r.symbol_id for r in _seed_ranked[:2]}
        ranked_symbols = sorted(
            raw_symbols,
            key=lambda record: (
                0 if record.symbol_id in _reserved_seed_ids else 1,
                0 if record.symbol_id in exact_ids or record.symbol_id in token_exact_ids else 1,
                -_file_pipe_coverage.get(record.file_path or "", 0),  # more pipe coverage = rank first
                -(record.score or 0.0),
                record.file_path,
                record.start_line,
            ),
        )
        # Relevance floor: when the top hit is strongly dominant (e.g. an exact
        # symbol scoring far above the lexical sub-token co-matches that share a
        # token like "get"/"name"), drop the near-zero tail so a precise query
        # returns the definition, not every file that merely shares a sub-token.
        # Pinned categories are always kept: the exact hit(s), the recall anchors
        # (zoekt/semantic, intentionally low/zero lexical score), and seed files --
        # so uniform low-score concept queries (floor ~ 0) keep everything.
        if ranked_symbols:
            # Idea C: Compute floor from definition-kind symbols in non-test files only.
            # Test files can have inflated BM25 scores from repeated assertions/references,
            # causing the floor to exclude the actual implementation file.
            definition_scores = [
                record.score or 0.0
                for record in ranked_symbols
                if (record.kind or "").lower() in _DEFINITION_KINDS and not _is_test_file_path(record.file_path)
            ]
            if definition_scores:
                top_score = max(definition_scores)
            else:
                # Fallback: no definition symbols in non-test files, use overall max
                top_score = max((record.score or 0.0) for record in ranked_symbols)
            floor = top_score * _EXPLORE_SCORE_FLOOR_FRAC
            if floor > 0:
                ranked_symbols = [
                    record
                    for record in ranked_symbols
                    if record.symbol_id in pinned_ids
                    or record.symbol_id in _reserved_seed_ids
                    or (record.score or 0.0) >= floor
                ]
        # File diversity: cap symbols-per-file before the symbol budget so one
        # over-populated file (e.g. 8 `as_sqlite` overloads in functions.py)
        # cannot starve the other files the query also matches (the ambiguous-name
        # collapse). Exact/seed hits already sort first, so they survive the cap.
        # Single-file scope: the caller explicitly restricted to one file, so there
        # are no other files to protect -- let the full symbol budget apply without
        # the per-file cap that would otherwise truncate to just 3 symbols.
        _per_file_cap = bounded_max_symbols if _single_file_seed else 3
        diverse_ranked = self._cap_symbols_per_file(ranked_symbols, max_per_file=_per_file_cap)
        selected_symbols = diverse_ranked[:bounded_max_symbols]
        # Ensure reference-expansion symbols are not silently dropped by the
        # symbol-budget cap above.  They rank low (score=None) so they fall
        # after all FTS hits in diverse_ranked; without this injection they
        # would be cut off when the FTS budget is already full.  Keep at most
        # one symbol per reference file to minimise token cost.
        if _ref_anchor_ids:
            _have_sel = {s.symbol_id for s in selected_symbols}
            _ref_missing_by_file: dict[str, SymbolRecord] = {}
            for _rs in raw_symbols:
                if _rs.symbol_id in _ref_anchor_ids and _rs.symbol_id not in _have_sel:
                    _ref_missing_by_file.setdefault(_rs.file_path or "", _rs)
            if _ref_missing_by_file:
                selected_symbols = selected_symbols + list(_ref_missing_by_file.values())[:8]
        # Zoekt/semantic anchor symbols suffer the same problem: they rank low
        # (score=None from direct file-path lookup) so the symbol-budget cap at
        # bounded_max_symbols cuts them before any anchor file gets a slot in the
        # file list.  Inject one symbol per anchor file that missed the budget.
        _anchor_injected: dict[str, SymbolRecord] = {}
        if anchor_ids:
            _have_sel = {s.symbol_id for s in selected_symbols}
            _anchor_missing_by_file: dict[str, SymbolRecord] = {}
            for _rs in raw_symbols:
                if _rs.symbol_id in anchor_ids and _rs.symbol_id not in _have_sel:
                    _anchor_missing_by_file.setdefault(_rs.file_path or "", _rs)
            if _anchor_missing_by_file:
                selected_symbols = selected_symbols + list(_anchor_missing_by_file.values())[:bounded_max_files]
                _anchor_injected = _anchor_missing_by_file
        family_member_ids: set[str] = set()
        # Skip sibling-family completion for an exact-symbol query: the named
        # definition is the answer, so pulling in related families across other
        # files would re-bloat a grep-style lookup (the dominant explore use).
        if effective_complete and not exact_hits:
            additions = self._complete_sibling_families(selected_symbols, query=query, seed_set=seed_set)
            if additions:
                # Single-file scope: keep family completions inside the target file.
                if _single_file_seed:
                    additions = [s for s in additions if s.file_path == _single_file_seed]
                have = {symbol.symbol_id for symbol in selected_symbols}
                fresh = [symbol for symbol in additions if symbol.symbol_id not in have]
                selected_symbols = selected_symbols + fresh
                family_member_ids = {symbol.symbol_id for symbol in fresh}

        # Test-coverage pinning: for each exact-hit function F, also include the
        # corresponding test_F symbol so the agent sees the test alongside the
        # implementation on the very first explore call. This eliminates the
        # secondary grep/read step that agents use to discover which test to run.
        # Uses an explicit lexical probe so the test is found even when the
        # BM25 top-N (max_symbols=20) didn't include it.
        if exact_ids and not query_wants_tests:
            _have = {s.symbol_id for s in selected_symbols}
            _exact_fn_names_lower = {
                r.symbol_name.lower() for r in selected_symbols if r.symbol_id in exact_ids and r.symbol_name
            }
            for _fn_name in _exact_fn_names_lower:
                _test_sym = f"test_{_fn_name}"
                if not self._exact_name_exists(_test_sym):
                    continue
                _test_hits = self.search_symbols(
                    _test_sym, limit=5, mode="lexical", snippet="none", auto_index=False, _candidate_files=set()
                )  # zoekt already ran above
                for _tr in _exact_symbol_hits(_test_hits, _test_sym):
                    if _tr.symbol_id not in _have and _TEST_PATH_RE.search(_tr.file_path or ""):
                        selected_symbols = [*selected_symbols, _tr]
                        _have.add(_tr.symbol_id)
                        break  # one test per function is enough

        # Rank so the highest-signal symbols claim the files that survive the file
        # and budget caps: the reserved in-scope symbols, then the query's completed
        # family, then definitions by score, then trivial variables/constants last.
        def _explore_priority(symbol: SymbolRecord) -> int:
            if symbol.symbol_id in exact_ids:
                return -1
            if symbol.symbol_id in _reserved_seed_ids:
                return 0
            # Direct definition hits must claim files BEFORE sibling-family
            # completions -- otherwise a loose affix (e.g. "select" pulling the
            # whole Select* widget family) hijacks the file slots above the
            # actually-relevant definitions.
            if (symbol.kind or "").lower() in _DEFINITION_KINDS:
                return 1
            if symbol.symbol_id in family_member_ids:
                return 2
            return 3

        selected_symbols = [
            symbol
            for _, symbol in sorted(enumerate(selected_symbols), key=lambda pair: (_explore_priority(pair[1]), pair[0]))
        ]
        selected_files: list[str] = []
        by_file: dict[str, list[SymbolRecord]] = {}
        for symbol in selected_symbols:
            by_file.setdefault(symbol.file_path, []).append(symbol)
            if symbol.file_path not in selected_files:
                selected_files.append(symbol.file_path)
        # Family-completion can add files past the normal cap; allow them since the
        # extra siblings render signatures-only (cheap), but stay bounded.
        # Hard-respect the caller's max_files. Sibling-family completion gets a small
        # fixed headroom (a couple of extra files for cross-file families) but is no
        # longer allowed to balloon: the old cap let an explicit max_files=1 expand
        # to 16 files, which then truncated every section to fit the token budget.
        file_cap = bounded_max_files + 2 if family_member_ids else bounded_max_files
        selected_files = selected_files[:file_cap]
        # Zoekt anchor files get injected to selected_symbols above, but the
        # file cap cuts them (they sort last because score=None).  Promote the
        # first anchor file into the file list so Zoekt's unique surface recall
        # (files FTS5 missed entirely) reaches the caller's result.
        if _anchor_injected:
            _anchor_in_cap = {f for f in selected_files if f in _anchor_injected}
            if not _anchor_in_cap:
                _first = next(iter(_anchor_injected.keys()))
                selected_files[-1:] = [_first]
        # Path-alignment reranking: stable re-sort by query-word overlap with file path
        # parts + symbol names. Surfaces files whose paths/symbols match query terms
        # (e.g. 'timezone.py' for a timezone query) with zero latency and no API calls.
        # Only fires when LEMONCROW_RERANK=1 and there are enough candidates to reorder.
        if (
            str(os.environ.get("LEMONCROW_RERANK") or "").strip().lower() in {"1", "true", "yes", "on"}
            and len(selected_files) > 3
        ):
            _stop = {"the", "a", "an", "in", "of", "to", "for", "is", "it", "on", "at", "fix", "bug", "issue"}
            _qwords = frozenset(re.split(r"[\s\W]+", query.lower())) - _stop - {""}
            # Build symbol-name lookup for each file from already-ranked symbols
            _file_syms: dict[str, set[str]] = {}
            for _sym in ranked_symbols:
                if _sym.file_path in set(selected_files) and _sym.symbol_name:
                    _file_syms.setdefault(_sym.file_path, set()).update(re.split(r"[_A-Z]", _sym.symbol_name.lower()))

            def _align_score(fp: str) -> float:
                path_parts = frozenset(re.split(r"[/._-]+", fp.lower()))
                sym_parts = _file_syms.get(fp, set())
                return len(_qwords & (path_parts | sym_parts))

            _max_align = max((_align_score(f) for f in selected_files), default=0)
            if _max_align > 0:
                selected_files = [
                    f
                    for f, _ in sorted(
                        ((f, _align_score(f)) for f in selected_files),
                        key=lambda x: -x[1],
                    )
                ]
        trimmed_symbols = [symbol for symbol in selected_symbols if symbol.file_path in set(selected_files)]
        trimmed_by_file: dict[str, list[SymbolRecord]] = {}
        for symbol in trimmed_symbols:
            trimmed_by_file.setdefault(symbol.file_path, []).append(symbol)

        skeleton_ids: set[str] = set()
        skeleton_families: dict[str, str] = {}
        if effective_skeletonize:
            skeleton_ids, skeleton_families = self._select_skeleton_symbols(trimmed_symbols, seed_set=seed_set)
            # An exact name match is the whole point of the query -- never reduce
            # it to a signature-only skeleton; always show its full body.
            skeleton_ids -= exact_ids
        # Completed family members are supplementary "here's the rest of the family"
        # context, not direct hits -- always render them signatures-only so surfacing
        # a whole family stays cheap and never forces the budget to drop relevant
        # files. The actual search hits still render per the skeletonize flag.
        if family_member_ids:
            for symbol in trimmed_symbols:
                if symbol.symbol_id in family_member_ids and symbol.file_path not in seed_set:
                    skeleton_ids.add(symbol.symbol_id)
                    skeleton_families.setdefault(symbol.symbol_id, "completion")

        entry_points = [
            {
                "symbol_id": symbol.symbol_id,
                "symbol_name": symbol.symbol_name,
                "qualified_name": symbol.qualified_name,
                "file_path": symbol.file_path,
                "kind": symbol.kind,
                "start_line": symbol.start_line,
                "end_line": symbol.end_line,
                "score": symbol.score,
                "provenance": symbol.provenance,
            }
            for symbol in trimmed_symbols
        ]

        files_payload: list[dict[str, Any]] = []
        for file_path in selected_files:
            symbols = trimmed_by_file.get(file_path, [])
            file_entry: dict[str, Any] = {
                "file_path": file_path,
                "language": symbols[0].language if symbols else "unknown",
                # Per-file semantic cosine for the learned reranker's feature vector
                # (0.0 when the file was not a semantic anchor). Cheap to carry; only
                # the reranker reads it.
                "semantic_score": float(_sem_scores.get(file_path, 0.0)),
                "symbols": [
                    {
                        "symbol_id": symbol.symbol_id,
                        "symbol_name": symbol.symbol_name,
                        "qualified_name": symbol.qualified_name,
                        "kind": symbol.kind,
                        "start_line": symbol.start_line,
                        "end_line": symbol.end_line,
                        "provenance": symbol.provenance,
                    }
                    for symbol in symbols
                ],
            }
            if include_source:
                # Single-file scope: generate sections in score-DESC order so the
                # budget trim (which pops from the tail) keeps the most query-relevant
                # source rather than the earliest-in-file symbol.
                source_order = (
                    sorted(symbols, key=lambda s: -(s.score or 0.0)) if file_path == _single_file_seed else symbols
                )
                sections = [
                    {
                        **self._source_section_for_symbol(
                            symbol,
                            line_numbers=line_numbers,
                            skeleton=symbol.symbol_id in skeleton_ids,
                            max_tokens=(
                                _EXPLORE_SOURCE_SECTION_EXACT_MAX_TOKENS
                                if symbol.symbol_id == _top_exact_id
                                else _EXPLORE_SOURCE_SECTION_MAX_TOKENS
                            ),
                        ),
                        "_score": symbol.score or 0.0,
                        # matched=True when symbol had an FTS/exact score (not just
                        # seed-injected). Kept after _score is stripped so the
                        # renderer can tag query-relevant sections.
                        "matched": (symbol.score or 0.0) > 0.001,
                    }
                    for symbol in source_order
                ]
                merged_sections = self._merge_nearby_source_sections(sections)
                file_entry["source_sections"] = merged_sections
            files_payload.append(file_entry)

        relationships: dict[str, list[dict[str, Any]]] = {
            "callers": [],
            "callees": [],
            "usages": [],
        }
        if include_relationships:
            for symbol in trimmed_symbols[:3]:
                callers = self._neighborhood(
                    "callers",
                    symbol_id=symbol.symbol_id,
                    depth=bounded_depth,
                    limit=20,
                    budget_tokens=max(600, budget_tokens // 6),
                    auto_index=False,
                )
                if "error" not in callers:
                    relationships["callers"].append(
                        {
                            "symbol_id": symbol.symbol_id,
                            "symbol_name": symbol.symbol_name,
                            "related": callers.get("related", []),
                            "edges": callers.get("edges", []),
                        }
                    )
                callees = self._neighborhood(
                    "callees",
                    symbol_id=symbol.symbol_id,
                    depth=bounded_depth,
                    limit=20,
                    budget_tokens=max(600, budget_tokens // 6),
                    auto_index=False,
                )
                if "error" not in callees:
                    relationships["callees"].append(
                        {
                            "symbol_id": symbol.symbol_id,
                            "symbol_name": symbol.symbol_name,
                            "related": callees.get("related", []),
                            "edges": callees.get("edges", []),
                        }
                    )
                references = self._neighborhood(
                    "refs",
                    symbol_id=symbol.symbol_id,
                    group_by="none",
                    snippet_lines=0,
                    limit=20,
                    auto_index=False,
                    budget_tokens=max(600, budget_tokens // 6),
                )
                if "error" not in references:
                    refs_payload = references.get("references", [])
                    if isinstance(refs_payload, list):
                        relationships["usages"].append(
                            {
                                "symbol_id": symbol.symbol_id,
                                "symbol_name": symbol.symbol_name,
                                "references": refs_payload,
                            }
                        )

        # Dedup: ranked_symbols holds many symbols per file, so the naive
        # comprehension repeated a file once per matching symbol. Emit each related
        # file once, capped to what the renderer surfaces (_CONTEXT_RELATED_CAP=10).
        _extra_seen: set[str] = set(selected_files)
        additional_relevant_files: list[str] = []
        for _sym in ranked_symbols:
            if _sym.file_path in _extra_seen:
                continue
            _extra_seen.add(_sym.file_path)
            additional_relevant_files.append(_sym.file_path)
            if len(additional_relevant_files) >= 10:
                break
        # Deep-recall tail: the primary pipeline ranks only the top ~max_symbols
        # symbols, so a concept / natural-language query whose gold file matches
        # the query only weakly (its symbol bodies share a few discriminative
        # terms, ranking it past the top-K) never surfaces it at all. A second,
        # deeper lexical pass over the SAME FTS index recovers those files as an
        # ordered tail. Collected here but attached to the payload AFTER budget
        # packing (see below) so it is a pure path-only recall aid that never
        # competes with source sections for the token budget, and is appended
        # STRICTLY AFTER the primary candidates downstream (monotonic: it never
        # reorders the top). Skipped for exact-symbol queries -- the named
        # definition already answers those, so deeper recall is pure noise.
        _deep_recall_files: list[str] = []
        if self._lexical_search_enabled() and not exact_hits:
            with contextlib.suppress(Exception):
                _deep = self.search_symbols(
                    query,
                    limit=_EXPLORE_DEEP_RECALL_LIMIT,
                    snippet="none",
                    auto_index=False,
                    _candidate_files=set(),  # zoekt already ran above; skip duplicate
                )
                _deep_seen = set(_extra_seen)
                for _sym in _deep:
                    fp = _sym.file_path
                    if not fp or fp in _deep_seen:
                        continue
                    _deep_seen.add(fp)
                    _deep_recall_files.append(fp)
                    if len(_deep_recall_files) >= _EXPLORE_DEEP_RECALL_FILES:
                        break
        full_payload: dict[str, Any] = {
            "query": query,
            "repo_id": self.repo_id,
            "entry_points": entry_points,
            "files": files_payload,
            "additional_relevant_files": additional_relevant_files,
            "exact_match": bool(exact_hits),
            "truncated": len(selected_symbols) > len(trimmed_symbols),
            "cache_hit": False,
            "provenance": _LOCAL_PROVENANCE,
        }
        # Only ship relationships when populated -- the default explore call has
        # include_relationships=False, and an empty {callers,callees,usages} dict
        # was being serialised on every response for no signal.
        if any(relationships.values()):
            full_payload["relationships"] = relationships
        # Budget-aware file trim: drop the lowest-priority files until the payload
        # fits, so explore degrades to fewer (most-relevant) files instead of
        # collapsing to "no results" when a completed family + relationships overflow.
        if include_source:
            # Budget-aware file trim: measure total once, then subtract per-file
            # token costs instead of re-serialising the full payload on every pop.
            # Old loop was O(N x payload_size); this is O(N + payload_size).
            _trim_total = self._compute_total_tokens(full_payload)
            if len(files_payload) > 1 and _trim_total > budget_tokens:
                _per_file_tokens = [estimate_tokens(_canonical_json(fe)) for fe in files_payload]
                while len(files_payload) > 1 and _trim_total > budget_tokens:
                    _trim_total -= _per_file_tokens.pop()
                    files_payload.pop()
                    full_payload["files"] = files_payload
                    full_payload["truncated"] = True
            # Single-file / last-file fallback: trim sections by score (stored in
            # _score) so the most query-relevant source survives, not the
            # earliest-in-file. Restore file order for display after trimming.
            if _single_file_seed and files_payload:
                _trim_total = self._compute_total_tokens(full_payload)
                if _trim_total > budget_tokens:
                    sections = files_payload[0].get("source_sections") or []
                    sections.sort(key=lambda s: -(s.get("_score") or 0.0))
                    _per_sec_tokens = [estimate_tokens(_canonical_json(s)) for s in sections]
                    while len(sections) > 1 and _trim_total > budget_tokens:
                        _trim_total -= _per_sec_tokens.pop()
                        sections.pop()
                        files_payload[0]["source_sections"] = sections
                        full_payload["files"] = files_payload
                        full_payload["truncated"] = True
                    sections.sort(key=lambda s: int(s.get("start_line") or 0))
                    files_payload[0]["source_sections"] = sections
                    full_payload["files"] = files_payload
            # Strip internal annotations before packing (matched stays for the renderer).
            for fe in files_payload:
                for sec in fe.get("source_sections", []):
                    sec.pop("_score", None)
                    sec.pop("_max_tokens", None)
        skeletonized_meta: list[dict[str, Any]] = []
        tokens_saved_total = 0
        for file_entry in files_payload:
            for section in file_entry.get("source_sections", []):
                if not section.get("skeleton"):
                    continue
                section_id = str(section.get("symbol_id") or "")
                skeletonized_meta.append(
                    {
                        "symbol_id": section_id,
                        "qualified_name": section.get("qualified_name"),
                        "file_path": section.get("file_path"),
                        "family": skeleton_families.get(section_id, ""),
                    }
                )
                tokens_saved_total += int(section.get("tokens_saved") or 0)
        if skeletonized_meta:
            full_payload["skeletonized"] = skeletonized_meta
            full_payload["skeleton_tokens_saved"] = tokens_saved_total
        packed = self._pack_single_payload(
            full_payload,
            budget_tokens=budget_tokens,
            essential_keys=_EXPLORE_ESSENTIAL_KEYS,
            optional_keys_in_drop_order=_EXPLORE_OPTIONAL_KEYS,
        )
        # Attach the deep-recall tail AFTER packing so it bypasses the token
        # budget entirely: these are path-only navigation candidates, not source,
        # so they must not be dropped when the source sections fill the budget
        # (which is exactly when a weak-signal gold file most needs surfacing).
        if _deep_recall_files:
            packed["deep_recall"] = _deep_recall_files
        self._cache_set("code.explore", cache_args, packed)
        return packed, _zk_list

    def tool_routes(
        self,
        *,
        file_glob: str | None = None,
        language: str | None = None,
        limit: int = 200,
        budget_tokens: int = 4000,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        if auto_index:
            self._ensure_indexed()
        self._sync_symbol_intel()
        normalized_language = language.lower().strip() if isinstance(language, str) and language.strip() else None
        cache_args = {
            "file_glob": file_glob,
            "language": normalized_language,
            "limit": limit,
            "budget_tokens": budget_tokens,
        }
        hit, cached = self._cache_get("code.routes", cache_args)
        if hit and cached is not None:
            return self._mark_cache_hit(cached)

        routes, source_truncated = self._indexed_route_records(
            file_glob=file_glob,
            language=normalized_language,
            limit=max(1, limit),
        )
        full_payload = self._build_routes_payload(
            routes,
            file_glob=file_glob,
            language=normalized_language,
            truncated=source_truncated,
        )
        full_total_tokens = self._compute_total_tokens({**full_payload, "tokens_saved": 0})

        def build_payload(packed_items: list[dict[str, Any]]) -> dict[str, Any]:
            return self._finalize_packed_payload(
                self._build_routes_payload(
                    packed_items,
                    file_glob=file_glob,
                    language=normalized_language,
                    truncated=source_truncated or len(packed_items) < len(routes),
                ),
                full_total_tokens=full_total_tokens,
            )

        packed = self._fit_items_to_budget(
            routes,
            budget_tokens=budget_tokens,
            essential_keys=_ROUTES_ESSENTIAL_KEYS,
            optional_keys_in_drop_order=_ROUTES_OPTIONAL_KEYS,
            build_payload=build_payload,
        )
        payload = self._maybe_attach_overflow_metadata(
            packed_payload=packed,
            full_payload=full_payload,
            full_total_tokens=full_total_tokens,
            budget_tokens=budget_tokens,
        )
        self._cache_set("code.routes", cache_args, payload)
        return payload

    def tool_context(
        self,
        *,
        task: str,
        seed_files: list[str] | None = None,
        budget_tokens: int = 4000,
        max_symbols: int = 4,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        effective_budget_tokens = self._effective_budget_tokens("context", budget_tokens)
        if auto_index:
            self._ensure_indexed()
        self._sync_symbol_intel()
        normalized_seeds = [self._normalize_file_arg(seed) for seed in seed_files or []]
        cache_args = {
            "task": task,
            "seed_files": normalized_seeds,
            "budget_tokens": effective_budget_tokens,
            "max_symbols": max_symbols,
        }
        hit, cached = self._cache_get("code.context", cache_args)
        if hit and cached is not None:
            return self._mark_cache_hit(cached)

        raw = self.context_pack(
            task=task,
            seed_files=normalized_seeds,
            budget_tokens=effective_budget_tokens,
            max_symbols=max_symbols,
            auto_index=False,
        )
        payload = self._pack_single_payload(
            raw.model_dump(mode="json"),
            budget_tokens=effective_budget_tokens,
            essential_keys=_CONTEXT_ESSENTIAL_KEYS,
            optional_keys_in_drop_order=[
                "provenance",
                "budget_tokens",
                "token_count",
                "tokens_saved_vs_full_files",
                "content",
                "telemetry",
                "code_blocks",
                "repo_map",
                "import_neighbors",
                "related_symbols",
                "entry_points",
            ],
            base_tokens_saved=raw.tokens_saved_vs_full_files,
        )
        self._cache_set("code.context", cache_args, payload)
        return payload

    def tool_usages(
        self,
        query: str | None = None,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        symbol_name: str | None = None,
        file_path: str | None = None,
        kind: str | None = None,
        language: str | None = None,
        file_glob: str | None = None,
        group_by: Literal["file", "caller", "none"] = "file",
        snippet_lines: int = 3,
        limit: int = 20,
        budget_tokens: int = 4000,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        effective_budget_tokens = self._effective_budget_tokens("usages", budget_tokens)
        if auto_index:
            self._ensure_indexed()
        self._sync_symbol_intel()
        normalized_file_path = self._normalize_file_arg(file_path) if file_path else None
        cache_args = {
            "query": query,
            "symbol_id": symbol_id,
            "qualified_name": qualified_name,
            "symbol_name": symbol_name,
            "file_path": normalized_file_path,
            "kind": kind,
            "language": language,
            "file_glob": file_glob,
            "group_by": group_by,
            "snippet_lines": snippet_lines,
            "limit": limit,
            "budget_tokens": effective_budget_tokens,
        }
        hit, cached = self._cache_get("code.usages", cache_args)
        if hit and cached is not None:
            return self._mark_cache_hit(cached)
        payload = self._neighborhood(
            "refs",
            query=query,
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            symbol_name=symbol_name,
            file_path=normalized_file_path,
            kind=kind,
            language=language,
            file_glob=file_glob,
            group_by=group_by,
            snippet_lines=snippet_lines,
            limit=limit,
            auto_index=False,
            budget_tokens=effective_budget_tokens,
        )
        if "error" not in payload:
            self._cache_set("code.usages", cache_args, payload)
        return payload

    def tool_callers(
        self,
        query: str | None = None,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        symbol_name: str | None = None,
        file_path: str | None = None,
        kind: str | None = None,
        language: str | None = None,
        depth: int = 1,
        limit: int = 20,
        snapshot: bool = False,
        budget_tokens: int = 4000,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        return self._neighborhood(
            "callers",
            query=query,
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            symbol_name=symbol_name,
            file_path=file_path,
            kind=kind,
            language=language,
            depth=depth,
            limit=limit,
            snapshot=snapshot,
            budget_tokens=budget_tokens,
            auto_index=auto_index,
        )

    def tool_callees(
        self,
        query: str | None = None,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        symbol_name: str | None = None,
        file_path: str | None = None,
        kind: str | None = None,
        language: str | None = None,
        depth: int = 1,
        limit: int = 20,
        snapshot: bool = False,
        budget_tokens: int = 4000,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        return self._neighborhood(
            "callees",
            query=query,
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            symbol_name=symbol_name,
            file_path=file_path,
            kind=kind,
            language=language,
            depth=depth,
            limit=limit,
            snapshot=snapshot,
            budget_tokens=budget_tokens,
            auto_index=auto_index,
        )

    def tool_pattern(
        self,
        *,
        pattern: str,
        rewrite: str | None = None,
        language: str | None = None,
        file_glob: str | None = None,
        dry_run: bool = True,
        limit: int = 20,
        budget_tokens: int = 4000,
    ) -> dict[str, Any]:
        self._ensure_indexed()
        effective_budget_tokens = self._effective_budget_tokens("pattern", budget_tokens)
        adapter = AstGrepAdapter(self.repo_root)
        if rewrite is None:
            cache_args = {
                "pattern": pattern,
                "language": language,
                "file_glob": file_glob,
                "limit": limit,
                "budget_tokens": effective_budget_tokens,
            }
            native_cache_args = {**cache_args, "native": True}
            hit, cached = self._cache_get("code.pattern", native_cache_args)
            if hit and cached is not None:
                return self._mark_cache_hit(cached)
            native = self._native_python_pattern_search(
                pattern=pattern,
                language=language,
                file_glob=file_glob,
                limit=limit,
            )
            if native is not None:
                payload = self._pack_pattern_matches(native, budget_tokens=effective_budget_tokens)
                self._cache_set(
                    "code.pattern",
                    native_cache_args,
                    payload,
                )
                return payload
            hit, cached = self._cache_get("code.pattern", cache_args)
            if hit and cached is not None:
                return self._mark_cache_hit(cached)
            try:
                result = adapter.search(pattern=pattern, language=language, file_glob=file_glob, limit=limit)
            except AstGrepToolUnavailable as exc:
                native_unavailable = self._native_python_pattern_search(
                    pattern=pattern,
                    language=language,
                    file_glob=file_glob,
                    limit=limit,
                )
                if native_unavailable is not None:
                    payload = self._pack_pattern_matches(native_unavailable, budget_tokens=effective_budget_tokens)
                    return payload
                return exc.payload
            if len(result.matches) > limit:
                result = PatternSearchResult(
                    matches=result.matches[:limit],
                    truncated=True,
                    total_matches=result.total_matches if result.total_matches is not None else len(result.matches),
                )
            payload = self._pack_pattern_matches(
                result,
                budget_tokens=effective_budget_tokens,
            )
            self._cache_set("code.pattern", cache_args, payload)
            return payload

        try:
            rewrite_result = adapter.rewrite(
                pattern=pattern,
                rewrite=rewrite,
                language=language,
                file_glob=file_glob,
                dry_run=dry_run,
            )
        except AstGrepToolUnavailable as exc:
            return exc.payload
        if not dry_run and rewrite_result.files_changed:
            self._reindex_files(rewrite_result.files_changed)
        return self._pack_pattern_rewrite(rewrite_result, budget_tokens=effective_budget_tokens)

    def _native_python_pattern_search(
        self,
        *,
        pattern: str,
        language: str | None,
        file_glob: str | None,
        limit: int,
    ) -> PatternSearchResult | None:
        """Native Python structural search for common benchmark-critical patterns.

        ast-grep remains the advanced backend, but decorators/calls should not
        fail just because an external binary is unavailable.
        """
        normalized = pattern.strip()
        if language not in {None, "python", "py"}:
            return None
        mode: Literal["decorator", "call", "call_any", "def", "class"] | None = None
        target_name: str | None = None
        if normalized.startswith("@"):
            mode = "decorator"
            target_name = normalized[1:].split("(", 1)[0].strip()
        elif normalized in {"$F($$$ARGS)", "$F($$$)", "$F()"}:
            mode = "call_any"
        elif match := re.fullmatch(r"([A-Za-z_][A-Za-z0-9_\.]*)\(\s*(?:\$\$\$|\.{3}|)\s*\)", normalized):
            mode = "call"
            target_name = match.group(1)
        elif match := re.fullmatch(
            r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(?:\$\$\$|\.{3}|[^)]*)\)\s*:?",
            normalized,
        ):
            mode = "def"
            target_name = match.group(1)
        elif match := re.fullmatch(
            r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(\s*(?:\$\$\$|\.{3}|[^)]*)\s*\))?\s*:?",
            normalized,
        ):
            mode = "class"
            target_name = match.group(1)
        if mode is None or (mode != "call_any" and not target_name):
            return None

        matches: list[PatternMatch] = []
        candidates = sorted(path for path in self._indexed_files() if path.endswith(".py"))
        if file_glob:
            candidates = [path for path in candidates if _matches_file_glob(path, file_glob)]

        def names_match(observed: str | None) -> bool:
            if observed is None or target_name is None:
                return False
            return observed == target_name or ("." not in target_name and observed.endswith(f".{target_name}"))

        def build_match(
            rel: str,
            lines: list[str],
            node: ast.AST,
            *,
            captures: dict[str, str],
        ) -> PatternMatch:
            line = int(getattr(node, "lineno", 1) or 1)
            column = int(getattr(node, "col_offset", 0) or 0) + 1
            end_line = int(getattr(node, "end_lineno", line) or line)
            end_column = int(getattr(node, "end_col_offset", 0) or 0) + 1
            snippet = lines[line - 1].strip() if 1 <= line <= len(lines) else ""
            return PatternMatch(
                file_path=rel,
                line=line,
                column=column,
                end_line=max(line, end_line),
                end_column=max(column, end_column),
                snippet=snippet,
                captures=captures,
            )

        max_matches = max(0, limit)
        truncated = False

        def append_match(match: PatternMatch) -> None:
            nonlocal truncated
            if len(matches) < max_matches:
                matches.append(match)
                return
            truncated = True

        if mode == "decorator" and target_name:
            raw_matches = self.search_text(f"@{target_name}", path=".", limit=max_matches + 1)
            decorator_re = re.compile(r"^\s*@\s*([A-Za-z_][A-Za-z0-9_\.]*)")
            for raw_match in raw_matches:
                if not raw_match.file_path.endswith(".py"):
                    continue
                if file_glob and not _matches_file_glob(raw_match.file_path, file_glob):
                    continue
                match = decorator_re.match(raw_match.text)
                if match is None or not names_match(match.group(1)):
                    continue
                append_match(
                    PatternMatch(
                        file_path=raw_match.file_path,
                        line=raw_match.line,
                        column=raw_match.column,
                        end_line=raw_match.line,
                        end_column=raw_match.column + len(raw_match.text),
                        snippet=raw_match.text.strip(),
                        captures={"decorator": match.group(1)},
                    )
                )
                if truncated:
                    break
            matches.sort(key=lambda item: (item.file_path, item.line, item.column, item.snippet))
            return PatternSearchResult(
                matches=matches, truncated=truncated, total_matches=None if truncated else len(matches)
            )

        if mode in {"def", "class"} and target_name:
            wanted_kinds = ("class",) if mode == "class" else ("function", "method")
            placeholders = ",".join("?" for _ in wanted_kinds)
            with self._connect() as conn:
                self._init_schema(conn)
                rows = conn.execute(
                    f"""
                    SELECT *, NULL AS score FROM symbols
                    WHERE repo_id = ? AND symbol_name = ? AND kind IN ({placeholders})
                    ORDER BY file_path, start_line, end_line, qualified_name, symbol_id
                    LIMIT ?
                    """,
                    (self.repo_id, target_name, *wanted_kinds, max_matches + 1),
                ).fetchall()
            for row in rows:
                symbol = _row_to_symbol(row)
                if file_glob and not _matches_file_glob(symbol.file_path, file_glob):
                    continue
                append_match(
                    PatternMatch(
                        file_path=symbol.file_path,
                        line=symbol.start_line,
                        column=1,
                        end_line=symbol.end_line,
                        end_column=1,
                        snippet=symbol.signature,
                        captures={"name": symbol.symbol_name},
                    )
                )
                if truncated:
                    break
            return PatternSearchResult(
                matches=matches, truncated=truncated, total_matches=None if truncated else len(matches)
            )

        for rel in candidates:
            if truncated:
                break
            source = self._read_file(rel)
            lines = source.splitlines()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            if mode == "decorator":
                for node in ast.walk(tree):
                    if truncated:
                        break
                    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                        continue
                    for decorator in node.decorator_list:
                        name = self._python_call_name(decorator)
                        if name is None and isinstance(decorator, ast.Name):
                            name = decorator.id
                        if not names_match(name):
                            continue
                        append_match(
                            build_match(
                                rel,
                                lines,
                                decorator,
                                captures={"decorator": name or target_name or ""},
                            )
                        )
                        if truncated:
                            break
            elif mode in {"call", "call_any"}:
                for node in ast.walk(tree):
                    if truncated:
                        break
                    if not isinstance(node, ast.Call):
                        continue
                    name = self._python_call_name(node.func)
                    if not name or (mode == "call" and not names_match(name)):
                        continue
                    append_match(build_match(rel, lines, node, captures={"F": name}))
            elif mode == "def":
                for node in ast.walk(tree):
                    if truncated:
                        break
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == target_name:
                        append_match(build_match(rel, lines, node, captures={"name": node.name}))
            elif mode == "class":
                for node in ast.walk(tree):
                    if truncated:
                        break
                    if isinstance(node, ast.ClassDef) and node.name == target_name:
                        append_match(build_match(rel, lines, node, captures={"name": node.name}))
        matches.sort(key=lambda item: (item.file_path, item.line, item.column, item.snippet))
        total_matches = None if truncated else len(matches)
        return PatternSearchResult(matches=matches, truncated=truncated, total_matches=total_matches)

    def tool_status(
        self,
        *,
        budget_tokens: int = 2000,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        if auto_index:
            self._ensure_indexed()
        self._sync_symbol_intel()
        cache_args = {
            "budget_tokens": budget_tokens,
            "index_version": self._current_index_version(),
            "autosync_enabled": self._autosync_enabled,
            "autosync_debounce_ms": self._autosync_debounce_ms,
            "head_sha": self._safe_current_head_sha(),
        }
        hit, cached = self._cache_get("code.status", cache_args)
        if hit and cached is not None:
            return self._mark_cache_hit(cached)
        index_stats = self._index_snapshot()
        cache_stats = self._cache.stats(
            repo_id=self.repo_id,
            index_version=self._current_index_version(),
            tool_name=None,
        )
        stale_after_seconds = 86_400
        files_indexed = int(index_stats.get("files_indexed", 0) or 0)
        index_age_seconds = index_stats.get("index_age_seconds")
        if files_indexed <= 0:
            freshness_status = "empty"
            freshness_reason = "no indexed files"
        elif isinstance(index_age_seconds, int) and index_age_seconds > stale_after_seconds:
            freshness_status = "stale"
            freshness_reason = "index older than stale threshold"
        else:
            freshness_status = "fresh"
            freshness_reason = "index within freshness threshold"

        provider_thresholds = {
            "required_health_status": "ok",
        }
        warnings: list[dict[str, Any]] = []
        provider_counts = {"ok": 0, "degraded": 0, "unhealthy": 0}
        providers: list[dict[str, Any]] = []
        for provider in self.intel_store.providers:
            provider_name = str(getattr(provider, "name", provider.__class__.__name__.lower()))
            entry: dict[str, Any] = {"name": provider_name}
            try:
                health = provider.health()
            except Exception as exc:
                logging.exception("Recovered from broad exception handler")
                health = ProviderHealth(status="unhealthy", reason=str(exc))
            if isinstance(health, ProviderHealth):
                entry["status"] = health.status
                entry["ok"] = health.ok
                if health.reason:
                    entry["reason"] = str(health.reason)
            else:
                ok = bool(health)
                entry["status"] = "ok" if ok else "unhealthy"
                entry["ok"] = ok
            provider_status = str(entry.get("status") or "unhealthy")
            if provider_status in provider_counts:
                provider_counts[provider_status] += 1
            if provider_status != "ok":
                warnings.append(
                    {
                        "code": "provider_health_not_ok",
                        "level": "warning",
                        "provider": provider_name,
                        "message": f"provider '{provider_name}' health is {provider_status}",
                    }
                )
            index_sha_fn = getattr(provider, "index_sha", None)
            if callable(index_sha_fn):
                with contextlib.suppress(Exception):
                    index_sha = index_sha_fn()
                    if index_sha:
                        entry["index_sha"] = str(index_sha)
            entry["freshness"] = "unknown"
            providers.append(entry)

        payload = {
            "repo_id": self.repo_id,
            "repo_root": str(self.repo_root),
            "db_path": str(self.db_path),
            "index_version": self._current_index_version(),
            "index": index_stats,
            "cache": cache_stats,
            "providers": providers,
            "provider_freshness": {
                "thresholds": provider_thresholds,
                "summary": {
                    **provider_counts,
                    "total": sum(provider_counts.values()),
                },
            },
            "warnings": warnings,
            "freshness": {
                "status": freshness_status,
                "reason": freshness_reason,
                "indexed": files_indexed > 0,
                "last_indexed_at": index_stats.get("last_indexed_at"),
                "index_age_seconds": index_age_seconds,
                "stale_after_seconds": stale_after_seconds,
            },
            "autosync": self._autosync_status(),
            "provenance": _LOCAL_PROVENANCE,
        }
        payload = cast(dict[str, Any], self._json_safe(payload))
        packed = self._pack_single_payload(
            payload,
            budget_tokens=budget_tokens,
            essential_keys=_STATUS_ESSENTIAL_KEYS,
            optional_keys_in_drop_order=[],
        )
        self._cache_set("code.status", cache_args, packed)
        return packed

    def tool_cache_status(
        self,
        *,
        cache_tool: str | None = None,
        budget_tokens: int = 4000,
    ) -> dict[str, Any]:
        effective_budget_tokens = self._effective_budget_tokens("cache_status", budget_tokens)
        tool_name = self._normalize_cache_tool(cache_tool)
        cache_stats = self._cache.stats(
            repo_id=self.repo_id,
            index_version=self._current_index_version(),
            tool_name=tool_name,
        )
        payload = {
            "repo_id": self.repo_id,
            "index_version": self._current_index_version(),
            "entry_count": int(cache_stats.get("entry_count", 0)),
            "entries_by_tool": cache_stats.get("entries_by_tool", {}),
            "total_bytes": int(cache_stats.get("total_bytes", 0)),
            "max_bytes": int(cache_stats.get("max_bytes", 0)),
            "scope": {
                "cache_tool": cache_tool or "all",
                "tool_name": tool_name,
            },
            "last_hit_at": cache_stats.get("last_hit_at", ""),
            "provenance": _LOCAL_PROVENANCE,
        }
        return self._pack_single_payload(
            payload,
            budget_tokens=effective_budget_tokens,
            essential_keys=_CACHE_STATUS_ESSENTIAL_KEYS,
            optional_keys_in_drop_order=["index_version", "last_hit_at", "scope"],
        )

    def tool_cache_invalidate(
        self,
        *,
        cache_tool: str | None = None,
        budget_tokens: int = 4000,
    ) -> dict[str, Any]:
        tool_name = self._normalize_cache_tool(cache_tool)
        index_version = self._current_index_version()
        invalidated = self._cache.invalidate(repo_id=self.repo_id, index_version=index_version, tool_name=tool_name)
        return self._pack_single_payload(
            {
                "repo_id": self.repo_id,
                "index_version": index_version,
                **invalidated,
                "scope": {"cache_tool": cache_tool or "all", "tool_name": tool_name},
                "provenance": _LOCAL_PROVENANCE,
            },
            budget_tokens=budget_tokens,
            essential_keys=_CACHE_INVALIDATE_ESSENTIAL_KEYS,
            optional_keys_in_drop_order=[],
        )

    @overload
    def search_symbols(
        self,
        query: str,
        *,
        limit: int = 20,
        mode: SearchMode = "auto",
        kind: str | None = None,
        language: str | None = None,
        snippet: Literal["none", "head", "full"] = "none",
        snippet_lines: int = 8,
        file_glob: str | None = None,
        scope: Literal["repo", "external"] = "repo",
        since: str | None = None,
        touched_by: str | None = None,
        auto_index: bool = True,
        provenance_filter: str | None = None,
        _candidate_files: set[str] | None = None,
    ) -> list[SymbolRecord]: ...

    @overload
    def search_symbols(
        self,
        query: str,
        *,
        scope: Literal["deleted"],
        limit: int = 20,
        mode: SearchMode = "auto",
        kind: str | None = None,
        language: str | None = None,
        snippet: Literal["none", "head", "full"] = "none",
        snippet_lines: int = 8,
        file_glob: str | None = None,
        since: str | None = None,
        touched_by: str | None = None,
        auto_index: bool = True,
        provenance_filter: str | None = None,
    ) -> list[DeletedHistoryItem]: ...

    def search_symbols(
        self,
        query: str,
        *,
        limit: int = 20,
        mode: SearchMode = "auto",
        kind: str | None = None,
        language: str | None = None,
        snippet: Literal["none", "head", "full"] = "none",
        snippet_lines: int = 8,
        file_glob: str | None = None,
        scope: Literal["repo", "external", "deleted"] = "repo",
        since: str | None = None,
        touched_by: str | None = None,
        auto_index: bool = True,
        provenance_filter: str | None = None,
        _candidate_files: set[str] | None = None,
    ) -> list[SymbolRecord] | list[DeletedHistoryItem]:
        """Deterministic multi-channel symbol search with routed-provider fallback."""
        if auto_index and scope != "deleted":
            self._ensure_indexed()
        if scope == "deleted":
            return self._deleted_history_adapter().search(
                query,
                limit=limit,
                since_ts=_parse_since_filter(since),
                touched_by=_normalize_touched_by(touched_by),
                language=language,
            )
        resolved_mode = resolve_search_mode(query, mode)
        candidate_files: set[str] | None = None
        rerank_limit = self._search_reranker.pre_rerank_limit(limit, mode=resolved_mode, scope=scope)
        if scope == "repo" and provenance_filter == "commit":
            hits = self._search_commit_chunks(query, limit=rerank_limit)
            if file_glob:
                hits = [hit for hit in hits if _matches_file_glob(hit.file_path, file_glob)]
            hits = [hit for hit in hits if not should_skip_relative_path(hit.file_path)]
            if _is_precise_symbol_query(query):
                exact_hits = _exact_symbol_hits(hits, query)
                if exact_hits:
                    hits = exact_hits
            hits = self._search_reranker.rerank(
                query,
                hits,
                mode=resolved_mode,
                scope=scope,
                source_loader=self._load_symbol_source_for_rerank,
            )
            return [
                self._attach_snippet(symbol, snippet=snippet, snippet_lines=snippet_lines) for symbol in hits[:limit]
            ]
        _cf_t: threading.Thread | None = None
        _cf_box: list[list[str]] = [[]]
        _t0 = 0.0
        if scope == "repo" and resolved_mode != "semantic":
            if _candidate_files is not None:
                # Caller pre-computed (or deliberately skipped) zoekt; reuse it.
                candidate_files = _candidate_files
            else:
                # Start zoekt in a background thread but defer the join until
                # AFTER lexical search runs — both proceed in parallel so
                # wall-clock = max(zoekt, lexical) instead of zoekt + lexical.
                # The candidate_files filter is applied post-hoc once both finish.
                _t0 = time.monotonic()

                def _fetch_cf() -> None:
                    with contextlib.suppress(Exception):
                        _cf_box[0] = self._zoekt_candidate_files(query, max_files=max(limit * 4, 40))

                _cf_t = threading.Thread(target=_fetch_cf, daemon=True)
                _cf_t.start()
                # candidate_files resolved after lexical returns below
        if resolved_mode == "lexical":
            hits = self.intel_store.search_symbols(query, limit=limit, kind=kind, language=language, scope=scope)
            if scope == "repo" and not hits:
                # Run local FTS while zoekt is still in flight — parallel.
                hits = self._search_symbols_local(
                    query,
                    limit=limit,
                    kind=kind,
                    language=language,
                    candidate_files=None,  # zoekt filter applied post-hoc
                )
            # Collect zoekt results — has been running in parallel since T=0.
            if scope == "repo" and _cf_t is not None:
                _cf_t.join(max(0.0, 0.8 - (time.monotonic() - _t0)))
                candidate_files = set(_cf_box[0] or [])
            if scope == "repo" and candidate_files:
                _filtered = [hit for hit in hits if hit.file_path in candidate_files]
                if _filtered:  # guard: don't discard all results if zoekt is cold
                    hits = _filtered
        else:
            candidate_limit = semantic_candidate_limit(rerank_limit)
            if scope == "external":
                hits = self.intel_store.search_symbols(
                    query,
                    limit=candidate_limit,
                    kind=kind,
                    language=language,
                    scope="external",
                )
            else:
                lexical_hits = self.intel_store.search_symbols(
                    query,
                    limit=candidate_limit,
                    kind=kind,
                    language=language,
                    scope="repo",
                )
                if not lexical_hits:
                    # Run local FTS while zoekt is still in flight — parallel.
                    lexical_hits = self._search_symbols_local(
                        query,
                        limit=candidate_limit,
                        kind=kind,
                        language=language,
                        candidate_files=None,  # zoekt filter applied post-hoc
                    )
                # Collect zoekt results — has been running in parallel since T=0.
                if _cf_t is not None:
                    _cf_t.join(max(0.0, 0.8 - (time.monotonic() - _t0)))
                    candidate_files = set(_cf_box[0] or [])
                if candidate_files:
                    _filtered_hits = [hit for hit in lexical_hits if hit.file_path in candidate_files]
                    if _filtered_hits:  # guard: don't discard all results if zoekt is cold
                        lexical_hits = _filtered_hits
                semantic_hits: list[SymbolRecord] = []
                if _query_is_natural_language(query) or resolved_mode == "semantic":
                    _sem_fut = _SEMANTIC_SYMBOL_EXECUTOR.submit(
                        self._search_symbols_semantic_local,
                        query,
                        limit=candidate_limit,
                        kind=kind,
                        language=language,
                    )
                    try:
                        semantic_hits = _sem_fut.result(timeout=_SEMANTIC_SYMBOL_DEADLINE_S)
                        _tally_semantic_symbol_outcome("ok")
                    except (TimeoutError, concurrent.futures.CancelledError):
                        _tally_semantic_symbol_outcome("timeout")
                        if _SEMANTIC_SYMBOL_HARD_FAIL:
                            raise
                        # Don't block on a cold ANN-matrix load (see
                        # _SEMANTIC_SYMBOL_DEADLINE_S) -- fall back to lexical_hits
                        # alone; the abandoned future keeps warming the cache.
                        pass
                    except Exception as exc:  # must degrade gracefully (unless hard-fail), see below
                        # ANY embedder/ANN failure (missing dependency, OOM, backend
                        # error, bad response shape, ...) must degrade to
                        # lexical_hits alone -- exactly like _semantic_candidate_files'
                        # contextlib.suppress(Exception) a few hundred lines down.
                        # Without this, one bad embedder call propagates out of
                        # search_symbols and kills the WHOLE multi-channel query (and
                        # the tool_explore call built on it), not just the semantic
                        # signal. Confirmed via a real run: EVERY natural-language
                        # query enters this branch (_query_is_natural_language), so an
                        # embedder fault here silently zeroed MRR for the entire
                        # semantic gold kind (1688/1688 queries) while lexical-only
                        # golds (short, non-NL queries that skip this branch) were
                        # unaffected -- previously only KeyError/TypeError/ValueError
                        # were caught here.
                        _tally_semantic_symbol_outcome("exception")
                        logger.warning("semantic symbol search failed, falling back to lexical-only: %s", exc)
                        if _SEMANTIC_SYMBOL_HARD_FAIL:
                            raise
                # Merge commit chunks as a third candidate source (LINEAGE-03)
                commit_hits: list[SymbolRecord] = []
                with contextlib.suppress(Exception):
                    commit_hits = self._search_commit_chunks(query, limit=candidate_limit)
                if resolved_mode == "semantic":
                    hits = (semantic_hits + commit_hits)[:rerank_limit]
                else:
                    hits = self._semantic_ranker.reciprocal_rank_fuse(
                        lexical_hits,
                        semantic_hits + commit_hits,
                        limit=rerank_limit,
                        semantic_additive_k=_SEMANTIC_ADDITIVE_TOP_K,
                    )
        if file_glob:
            hits = [hit for hit in hits if _matches_file_glob(hit.file_path, file_glob)]
        hits = [hit for hit in hits if not should_skip_relative_path(hit.file_path)]
        if provenance_filter is not None:
            hits = [h for h in hits if h.provenance == provenance_filter]
        if _is_precise_symbol_query(query):
            exact_hits = _exact_symbol_hits(hits, query)
            if exact_hits:
                hits = exact_hits
        hits = self._search_reranker.rerank(
            query,
            hits,
            mode=resolved_mode,
            scope=scope,
            source_loader=self._load_symbol_source_for_rerank,
        )
        return [self._attach_snippet(symbol, snippet=snippet, snippet_lines=snippet_lines) for symbol in hits[:limit]]

    def _fts_document_count(self, conn: sqlite3.Connection) -> int:
        """Total indexed-symbol count, cached per index_version (a reindex bumps the
        version and invalidates the entry).  Denominator for IDF term pruning;
        count(*) over the FTS is ~18ms so it must never run per query."""
        version = self._current_index_version()
        cached = self._fts_doc_count_cache.get(version)
        if cached is not None:
            return cached
        row = conn.execute("SELECT count(*) FROM symbol_fts").fetchone()
        total = int(row[0]) if row else 0
        self._fts_doc_count_cache[version] = total
        return total

    def _discriminative_fts_terms(
        self, conn: sqlite3.Connection, terms: list[str]
    ) -> tuple[list[str], list[str], frozenset[str]]:
        """IDF pruning of FTS query terms -> (or_terms, prefix_terms, common_terms).

        Tokens whose document frequency exceeds _FTS_COMMON_TERM_DF_FRACTION of the
        corpus (``get``, ``name``, ``field`` -- present in a large fraction of all
        symbols) bloat the bm25 posting-list scan without adding precision: the
        rarer tokens decide the match, and the exact/substring channels already
        cover the common token.

        - ``or_terms`` drives the FTS OR channel and is never empty -- when every
          token is common it keeps the single rarest so a lone common token (e.g.
          ``field``) still matches via FTS.
        - ``prefix_terms`` drives the prefix-completion channel and keeps ONLY
          discriminative tokens (may be empty -> the channel is skipped).  A common
          token's prefix expansion (``field*`` -> field/fields/fieldname/...) is the
          single most expensive variant and is already covered by the substring
          channel, so it is never prefix-expanded.

        Frequencies come from the fts5vocab index (one ~0.03ms lookup per term)."""
        unique: list[str] = []
        seen: set[str] = set()
        for term in terms[:12]:
            if term and term not in seen:
                seen.add(term)
                unique.append(term)
        if not unique:
            return [], [], frozenset()
        total = self._fts_document_count(conn)
        if total <= 0:
            return unique, unique, frozenset()
        cap = max(_FTS_COMMON_TERM_DF_FLOOR, int(total * _FTS_COMMON_TERM_DF_FRACTION))
        try:
            # One batched vocab lookup for all terms instead of a round-trip per
            # term: a 7-8 term query (multi-term/regex) drops from ~8 executes to
            # 1.  Terms absent from the vocab table simply return no row and
            # default to df=0 below -- identical to the per-term lookup.
            placeholders = ",".join("?" for _ in unique)
            doc_rows = conn.execute(
                f"SELECT term, doc FROM symbol_fts_vocab WHERE term IN ({placeholders})",
                tuple(unique),
            ).fetchall()
        except sqlite3.OperationalError:
            # Vocab table unavailable (e.g. mid-migration DB) -- skip pruning.
            return unique, unique, frozenset()
        doc_by_term = {str(r["term"]): int(r["doc"]) if r["doc"] is not None else 0 for r in doc_rows}
        freqs: list[tuple[str, int]] = [(term, doc_by_term.get(term, 0)) for term in unique]
        df_by_term = dict(freqs)
        # Tokens common enough that their trigram-substring scan is too costly for
        # its marginal recall (df > _SUBSTRING_ANCHOR_DF_CAP): the substring channel
        # skips these; FTS bm25 already indexes the whole token. See
        # _search_symbols_local. (Distinct from the FTS OR/prefix `cap` above, which
        # is ~10% of the corpus -- far too loose to bound a trigram scan.)
        common_terms = frozenset(t for t, d in freqs if d > _SUBSTRING_ANCHOR_DF_CAP)
        discriminative = sorted((t for t, d in freqs if d <= cap), key=lambda t: df_by_term[t])
        if discriminative:
            # Add rarest-first until the cumulative posting-list size hits the budget.
            kept: list[str] = []
            used = 0
            for term in discriminative:
                if kept and used + df_by_term[term] > _FTS_DF_BUDGET:
                    break
                kept.append(term)
                used += df_by_term[term]
            # Skip prefix-expansion of 1-char terms only (``"d"*`` matches every
            # d-token); 2+ char prefixes stay (short identifiers like "fit" need them).
            prefix_terms = [t for t in kept if len(t) >= 2]
            return kept, prefix_terms, common_terms
        # Every token is common -- keep the single rarest so recall doesn't collapse.
        return [min(freqs, key=lambda item: item[1])[0]], [], common_terms

    def _exact_name_exists(self, name: str) -> bool:
        """Indexed existence probe: any symbol with this exact name (ASCII-CI)?

        Gates tool_explore's exact-name fallback / token / test_-pin probes:
        their output is consumed ONLY through _exact_symbol_hits(), so when no
        indexed symbol_name/qualified_name equals ``name`` case-insensitively,
        the full lexical search they'd run cannot yield an exact hit -- skip it
        (~0.2ms index seek vs ~10-15ms multi-channel search). COLLATE NOCASE ==
        Python .lower() equality for the ASCII identifiers _SYMBOL_QUERY_RE
        admits. Fails open on DB errors: the guard may cost time, never results.
        """
        normalized = name.strip()
        if not normalized:
            return False
        try:
            with self._connect(readonly=True) as conn:
                for column in ("symbol_name", "qualified_name"):
                    row = conn.execute(
                        f"SELECT 1 FROM symbols WHERE repo_id = ? AND {column} COLLATE NOCASE = ? LIMIT 1",
                        (self.repo_id, normalized),
                    ).fetchone()
                    if row is not None:
                        return True
                return False
        except sqlite3.Error:
            return True

    def _search_symbols_local(
        self,
        query: str,
        *,
        limit: int = 20,
        kind: str | None = None,
        language: str | None = None,
        candidate_files: set[str] | None = None,
    ) -> list[SymbolRecord]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        normalized_query_lower = normalized_query.lower()
        terms = _query_terms(normalized_query)
        first_term = terms[0] if terms else normalized_query_lower[:4]
        strong_fetch_limit = max(limit * 8, 80)
        query_mentions_tests = _query_implies_test_scope(normalized_query)
        kind_boosts = {
            "class": 18.0,
            "interface": 18.0,
            "type": 15.0,
            "struct": 15.0,
            "enum": 14.0,
            "method": 11.0,
            "function": 10.0,
        }
        if normalized_query and normalized_query[0].isupper():
            kind_boosts["class"] = kind_boosts.get("class", 0.0) + 8.0
            kind_boosts["type"] = kind_boosts.get("type", 0.0) + 8.0

        filters = ["repo_id = ?"]
        params: list[Any] = [self.repo_id]
        if kind:
            filters.append("kind = ?")
            params.append(kind)
        if language:
            filters.append("language = ?")
            params.append(language)
        if candidate_files:
            normalized_candidates = sorted({self._normalize_file_arg(path) for path in candidate_files if path})
            if normalized_candidates:
                filters.append(f"file_path IN ({','.join('?' for _ in normalized_candidates)})")
                params.extend(normalized_candidates)
        where_sql = " AND ".join(filters)
        # Same predicates, aliased for joins against the trigram FTS table (alias s).
        where_sql_s = " AND ".join(f"s.{f}" for f in filters)

        term_set = {term for term in terms if term}
        centrality_map = self._symbol_centrality_map()
        scored: dict[str, tuple[float, int]] = {}
        # Idea D: importing_files will be populated after the first DB query;
        # the adjustment function will check this set via closure.
        importing_files_for_boost: set[str] = set()
        # IDF weights for the query terms.  A name/signature/path match on a rare,
        # discriminative token (the identifier the query is really about) should
        # count for far more than a match on a token that occurs in a large
        # fraction of the corpus ("value", "data", "get", "self").  Without this,
        # a distractor file that happens to echo several common query words can
        # out-score the one file that matches the single rare identifier -- the
        # dominant precision failure on long, prose-heavy (SWE-bench / issue)
        # queries.  Populated once from the FTS vocab inside the connection block
        # below and read by `adjustment`; every term defaults to weight 1.0, so
        # behaviour is unchanged when the vocab lookup is unavailable.
        term_idf: dict[str, float] = {}

        # Per-query scoring context for the compiled-only symbol scorer
        # (ranking.score_symbol_row). term_idf and importing_files_for_boost are
        # passed by reference: both are still being populated below, and the
        # scorer only runs (via consider_rows) after that completes, so it sees
        # the finished dict/set.
        scoring_ctx = SymbolScoringContext(
            kind_boosts=kind_boosts,
            term_idf=term_idf,
            terms=terms,
            term_set=term_set,
            normalized_query_lower=normalized_query_lower,
            centrality_map=centrality_map,
            importing_files_for_boost=importing_files_for_boost,
            query_mentions_tests=query_mentions_tests,
            tokenize=_identifier_terms,
            is_test_path=_is_test_file_path,
        )

        # Rows for one symbol_id recur across channels (exact/CI-exact overlap, FTS
        # AND/OR/prefix hit the same symbols) but always carry the same symbols-table
        # columns; only the computed `score` column differs. The first-seen row is
        # kept per symbol and `score_symbol_row` (a pure function of the row once
        # importing_files_for_boost is populated above) is computed once per symbol
        # per query. SymbolRecord construction is deferred to the final top-`limit`
        # slice below: candidate rows (1-2k/query) never pay pydantic validation.
        row_cache: dict[str, Mapping[str, Any]] = {}
        adjustment_cache: dict[str, float] = {}

        def consider_rows(
            rows: Sequence[Mapping[str, Any]], *, channel_rank: int, base: float, use_row_score: bool = False
        ) -> None:
            for row in rows:
                symbol_id = str(row["symbol_id"])
                adj = adjustment_cache.get(symbol_id)
                if adj is None:
                    row_cache[symbol_id] = row
                    adj = adjustment_cache[symbol_id] = score_symbol_row(row, scoring_ctx)
                channel_score = float(row["score"]) * 100.0 if use_row_score and row["score"] is not None else 0.0
                score = base + channel_score + adj
                existing = scored.get(symbol_id)
                if existing is None or score > existing[0] or (score == existing[0] and channel_rank < existing[1]):
                    scored[symbol_id] = (score, channel_rank)

        # Phase 1 (main thread, sequential): IDF pruning + two exact-name seeks.
        # These must stay serial: IDF needs the fts_vocab table; exact seeks are
        # ~1ms total (index-backed) and the results are needed before ranking.
        # Read-only connection: all phase-1 work is pure SELECT.  WAL mode lets
        # readers run concurrently with no write-lock wait (eliminates the 30 s
        # timeout cliff when autosync is writing to a live repo DB).
        # Exact-name candidates.  A normal query seeks its own name; a pure
        # identifier alternation (identifiers joined by `|`, e.g.
        # "apply_fuzzy_replace|resolve_symbol_edit|ResolvedSymbolEdit") must seek
        # EACH alternative.  Seeking the whole pipe-joined string matches no
        # symbol, so the two highest-priority exact channels (base 1300/1180)
        # stay dead and the alternation is ranked only by noisy subtoken BM25 --
        # the definition files lose to cousins/tests that repeat those subtokens.
        # Gate strictly on EVERY alternative being a bare/dotted identifier: a
        # mixed regex alternation ("auto.?mode|MODE_AUTO|mode==\"auto\"") is a
        # grep pattern, not a name list, and exact-pinning one stray identifier
        # in it only perturbs the FTS ranking.  Falls back to the whole query
        # otherwise, preserving behaviour for every non-alternation query.
        exact_name_candidates = [normalized_query]
        if "|" in normalized_query:
            alt_names = _split_pipe_query(normalized_query)
            if alt_names and all(_SYMBOL_QUERY_RE.match(alt) for alt in alt_names):
                exact_name_candidates = alt_names
        exact_name_lower = [name.lower() for name in exact_name_candidates]
        exact_name_set = set(exact_name_candidates)
        _exact_ph = ",".join("?" for _ in exact_name_lower)
        with self._connect(readonly=True) as conn:
            self._init_schema(conn)
            # IDF-pruned FTS queries: high document-frequency tokens are dropped so
            # the OR/prefix bm25 scan stays small (see _discriminative_fts_terms).
            or_fts_terms, prefix_fts_terms, common_fts_terms = self._discriminative_fts_terms(conn, terms)
            # Precompute per-term IDF weights (see term_idf declaration above).
            # doc = document frequency of the term in the FTS vocab; the weight
            # is the corpus-normalized inverse document frequency, scaled and
            # bounded so a corpus-common term still counts a little (floor 0.3)
            # and an ultra-rare one cannot run away (cap _COVERAGE_IDF_CAP).
            _idf_total = self._fts_document_count(conn)
            _idf_terms = list(dict.fromkeys(t for t in terms[:12] if t))
            if _idf_total > 0 and _idf_terms:
                try:
                    _idf_ph = ",".join("?" for _ in _idf_terms)
                    _idf_rows = conn.execute(
                        f"SELECT term, doc FROM symbol_fts_vocab WHERE term IN ({_idf_ph})",
                        tuple(_idf_terms),
                    ).fetchall()
                    _df_by = {str(r["term"]): int(r["doc"] or 0) for r in _idf_rows}
                    _max_idf = math.log(_idf_total + 1.0)
                    for _t in _idf_terms:
                        _nidf = math.log((_idf_total + 1.0) / (_df_by.get(_t, 0) + 1.0)) / _max_idf
                        term_idf[_t] = idf_weight(_nidf)
                except sqlite3.OperationalError:
                    pass
            fts_query = _fts_or_query_from_terms(or_fts_terms)
            # Morphological recall: add each discriminative term's stem as an extra
            # prefix so an inflected query word reaches the code's canonical
            # identifier form (e.g. "parsing"->"pars*" matches "parse", the symbol
            # the query describes).  Confined to the lower-priority prefix channel
            # so exact / AND / OR matches keep decisive precedence; bm25 IDF and the
            # name-token adjustment govern the final order.
            _stemmed_prefix_terms = list(prefix_fts_terms)
            for _pt in prefix_fts_terms:
                _stem = _query_stem(_pt)
                if _stem and _stem not in _stemmed_prefix_terms:
                    _stemmed_prefix_terms.append(_stem)
            fts_prefix_query = _fts_prefix_query_from_terms(_stemmed_prefix_terms)
            # Exact + case-insensitive name lookup.  Split into one index-backed seek
            # per column: an `OR` across symbol_name/qualified_name lets SQLite use
            # NEITHER NOCASE index (it falls back to a full repo scan), and the old
            # `ORDER BY file_path, start_line` forced a temp-b-tree sort on top.  The
            # final ranking re-sorts everything by score, so per-channel order only
            # ever decided an arbitrary LIMIT cut -- drop it.  `COLLATE NOCASE IN`
            # keeps using idx_symbols_repo_name_nocase / idx_symbols_repo_qual_nocase
            # (one index seek per alternative; `IN (?)` degenerates to `= ?`).
            ci_exact_rows = conn.execute(
                f"SELECT *, NULL AS score FROM symbols WHERE {where_sql}"
                f" AND symbol_name COLLATE NOCASE IN ({_exact_ph}) LIMIT ?",
                tuple([*params, *exact_name_lower, strong_fetch_limit]),
            ).fetchall()
            ci_exact_rows += conn.execute(
                f"SELECT *, NULL AS score FROM symbols WHERE {where_sql}"
                f" AND qualified_name COLLATE NOCASE IN ({_exact_ph}) LIMIT ?",
                tuple([*params, *exact_name_lower, strong_fetch_limit]),
            ).fetchall()
        # Case-sensitive matches rank highest (channel 0); the rest are CI-exact.
        exact_rows = [
            row
            for row in ci_exact_rows
            if row["symbol_name"] in exact_name_set or row["qualified_name"] in exact_name_set
        ]
        # Idea D: Collect seed files from exact hits to find their importers.
        # Files that import a seed file are likely closely related to the query.
        seed_files = {row["file_path"] for row in ci_exact_rows if row["file_path"]}
        if seed_files:
            # Query imports table: find files (source_file) that import any seed file (target_file)
            placeholders = ",".join("?" for _ in seed_files)
            importer_rows = conn.execute(
                f"SELECT DISTINCT source_file FROM imports WHERE repo_id = ? AND target_file IN ({placeholders})",
                tuple([self.repo_id, *seed_files]),
            ).fetchall()
            importing_files_for_boost.update(row["source_file"] for row in importer_rows if row["source_file"])
        consider_rows(exact_rows, channel_rank=0, base=1300.0)
        consider_rows(ci_exact_rows, channel_rank=1, base=1180.0)

        # Phase 2 (parallel workers): channels 2-6 each run on their own read
        # connection submitted to _SEARCH_CHANNEL_EXECUTOR.  WAL mode lets readers
        # run concurrently with zero blocking; each channel runs in a dedicated OS
        # process, so channels achieve genuine CPU-level parallelism without GIL
        # contention.  Wall-clock time collapses from sum(channel_times) to
        # max(channel_times) (~15ms vs ~45ms).
        # Build the AND query from the IDF-pruned terms rather than the raw
        # query string.  Zero-hit tokens (e.g. a parameter name the agent wants
        # to ADD, like 'keep_attrs') have df=0 in the vocab and are not in
        # or_fts_terms, so they can't kill AND semantics.  We reconstruct a
        # query string from or_fts_terms (already rarest-first, already pruned)
        # so _fts_and_query's natural-language guard and length checks still apply.
        _and_input = " ".join(or_fts_terms) if or_fts_terms else normalized_query
        fts_and_q = _fts_and_query(_and_input)
        like_pattern = f"%{normalized_query_lower}%"
        path_anchor = or_fts_terms[0] if or_fts_terms else first_term
        first_term_like = f"%{path_anchor}%"
        path_patterns = [first_term_like]
        if _query_is_pathy_literal(normalized_query) and like_pattern != first_term_like:
            path_patterns.insert(0, like_pattern)

        # Substring (trigram + direct-scan) LIKE pattern: use the full raw query
        # only for pathy literals (no |, *, or other regex metacharacters).  For
        # regex/pipe multi-term queries, the raw query with embedded wildcards
        # would scan the entire trigram table; fall back to the rarest IDF term.
        substring_pattern = like_pattern if _query_is_pathy_literal(normalized_query) else first_term_like

        # Frozen parameter tuples for each channel (built once, passed to workers).
        _fts_extra = tuple([self.repo_id, *([kind] if kind else []), *([language] if language else [])])
        _base_params = tuple(params)  # repo_id [+ kind] [+ language] [+ candidate_files]

        # Each entry: (sql, params_tuple, channel_rank, base_score, use_row_score)
        _ch: list[tuple[str, tuple[Any, ...], int, float, bool]] = []

        # Multi-term AND channel (highest FTS precision, base 1100).
        if fts_and_q:
            _ch.append(
                (
                    f"SELECT s.*, abs(bm25(symbol_fts)) / (10.0 + abs(bm25(symbol_fts))) AS score"
                    f" FROM symbol_fts JOIN symbols s ON s.symbol_id = symbol_fts.symbol_id"
                    f" WHERE symbol_fts MATCH ? AND s.repo_id = ?{' AND s.kind = ?' if kind else ''}{' AND s.language = ?' if language else ''}"
                    f" ORDER BY rank LIMIT ?",
                    (fts_and_q, *_fts_extra, strong_fetch_limit),
                    2,
                    1100.0,
                    True,
                )
            )

        # OR channel (high recall, base 980).
        if fts_query:
            _ch.append(
                (
                    # FTS5 bm25() is NEGATIVE (more-negative = better match), so
                    # abs(bm25) grows with relevance.  Score with the monotone,
                    # saturating abs/(K+abs) transform -- the same shape the AND
                    # channel uses -- so the strongest BM25 hits earn the highest
                    # intra-channel score.  The previous 1/(1+abs) form was INVERTED:
                    # it handed a weak one-common-term match (abs~1 -> ~0.5) a far
                    # bigger bonus than a dense rare-term match (abs~20 -> ~0.05),
                    # actively burying the most relevant symbols on the multi-term /
                    # natural-language queries where the OR channel is the main signal.
                    f"SELECT s.*, abs(bm25(symbol_fts)) / (10.0 + abs(bm25(symbol_fts))) AS score"
                    f" FROM symbol_fts JOIN symbols s ON s.symbol_id = symbol_fts.symbol_id"
                    f" WHERE symbol_fts MATCH ? AND s.repo_id = ?{' AND s.kind = ?' if kind else ''}{' AND s.language = ?' if language else ''}"
                    f" ORDER BY rank LIMIT ?",
                    (fts_query, *_fts_extra, strong_fetch_limit),
                    2,
                    980.0,
                    True,
                )
            )

        # Prefix channel (partial-word match, base 940).
        if fts_prefix_query and fts_prefix_query != fts_query:
            _ch.append(
                (
                    # Correctly-signed BM25 transform (see the OR channel above for
                    # why 1/(1+abs) was inverted).
                    f"SELECT s.*, abs(bm25(symbol_fts)) / (10.0 + abs(bm25(symbol_fts))) AS score"
                    f" FROM symbol_fts JOIN symbols s ON s.symbol_id = symbol_fts.symbol_id"
                    f" WHERE symbol_fts MATCH ? AND s.repo_id = ?{' AND s.kind = ?' if kind else ''}{' AND s.language = ?' if language else ''}"
                    f" ORDER BY rank LIMIT ?",
                    (fts_prefix_query, *_fts_extra, strong_fetch_limit),
                    3,
                    940.0,
                    True,
                )
            )

        # Substring channel via trigram index (base 860).
        # Skipped when the anchor is a COMMON whole word (df > cap): its trigram
        # postings (common 3-grams like 'inc','ncl') are enormous, so the LIKE scans
        # 300-500ms on linux to surface a handful of non-token substring matches --
        # while FTS bm25 already indexes the whole token and returns it in ~15ms. The
        # channel only earns its keep for RARE anchors, where it catches substrings
        # the FTS tokenizer splits apart (e.g. 'mbedde' inside 'Embedder').
        # NO intra-channel ORDER BY either: `ORDER BY file_path,start_line LIMIT n`
        # would force a full materialize+sort before the LIMIT (another 2.7s on a
        # common term); the outer ranker re-sorts, so per-channel order is discarded.
        if len(normalized_query_lower) < 3:
            # <3-char queries can't use the trigram index; rare, fall back to direct scan.
            _ch.append(
                (
                    f"SELECT *, NULL AS score FROM symbols"
                    f" WHERE {where_sql} AND"
                    f" (lower(symbol_name) LIKE ? OR lower(qualified_name) LIKE ? OR lower(signature) LIKE ?)"
                    f" LIMIT ?",
                    (*_base_params, substring_pattern, substring_pattern, substring_pattern, strong_fetch_limit),
                    4,
                    860.0,
                    False,
                )
            )
        elif path_anchor not in common_fts_terms:
            _ch.append(
                (
                    f"SELECT s.*, NULL AS score FROM symbol_trigram t"
                    f" JOIN symbols s ON s.symbol_id = t.symbol_id"
                    f" WHERE {where_sql_s} AND (t.name LIKE ? OR t.qualified_name LIKE ?)"
                    f" LIMIT ?",
                    (*_base_params, substring_pattern, substring_pattern, strong_fetch_limit),
                    4,
                    860.0,
                    False,
                )
            )

        # Path channel (base 820): matches against distinct file paths instead of
        # one row per symbol. `file_path_trigram`/`files` carry one row per FILE
        # (thousands, even on django/linux) vs. `symbol_trigram`/`symbols`' one
        # row per SYMBOL (often 100K+) -- matching the small table first, then
        # joining back to symbols via idx_symbols_repo_file, returns the exact
        # same rows for a fraction of the scan cost.
        if len(normalized_query_lower) >= 3 and len(path_anchor) >= 3:
            _path_like_sql = " OR ".join("fpt.file_path LIKE ?" for _ in path_patterns)
            _ch.append(
                (
                    f"SELECT s.*, NULL AS score FROM file_path_trigram fpt"
                    f" JOIN symbols s ON s.repo_id = fpt.repo_id AND s.file_path = fpt.file_path"
                    f" WHERE {where_sql_s} AND ({_path_like_sql})"
                    f" LIMIT ?",
                    (*_base_params, *path_patterns, strong_fetch_limit),
                    5,
                    820.0,
                    False,
                )
            )
        else:
            _path_like_sql = " OR ".join("lower(f.file_path) LIKE ?" for _ in path_patterns)
            _ch.append(
                (
                    f"SELECT s.*, NULL AS score FROM files f"
                    f" JOIN symbols s ON s.repo_id = f.repo_id AND s.file_path = f.file_path"
                    f" WHERE {where_sql_s} AND ({_path_like_sql})"
                    f" LIMIT ?",
                    (*_base_params, *path_patterns, strong_fetch_limit),
                    5,
                    820.0,
                    False,
                )
            )

        # Submit all channels in parallel; collect in submission order so that
        # higher-priority channels (AND > OR > prefix) are merged first.
        _db = self.db_path
        # Supplement channels (rank >= 4: substring + path trigram) get a tight
        # deadline so a common-gram scan self-aborts instead of dominating latency;
        # the precise FTS/exact channels (rank < 4) keep the full budget.
        _futures = [
            _SEARCH_CHANNEL_EXECUTOR.submit(
                _run_search_channel,
                _db,
                sql,
                ch_params,
                _SUPPLEMENT_CHANNEL_DEADLINE_S if _rank >= 4 else _SEARCH_CHANNEL_DEADLINE_S,
            )
            for sql, ch_params, _rank, _, _ in _ch
        ]
        for _fut, (_, _, _rank, _base_score, _use_score) in zip(_futures, _ch, strict=False):
            try:
                # 8 s timeout: a timed-out channel contributes no rows (graceful
                # degradation) instead of blocking the caller indefinitely.  Normal
                # queries finish in <200 ms; 8 s is 40x headroom.
                consider_rows(
                    _fut.result(timeout=8.0),
                    channel_rank=_rank,
                    base=_base_score,
                    use_row_score=_use_score,
                )
            except (TimeoutError, ValueError):
                pass

        ranked = sorted(
            scored.items(),
            key=lambda kv: (
                -kv[1][0],
                kv[1][1],
                row_cache[kv[0]]["file_path"],
                row_cache[kv[0]]["start_line"],
                row_cache[kv[0]]["end_line"],
                row_cache[kv[0]]["qualified_name"],
                kv[0],
            ),
        )
        emit_product_local(
            "code_context_retrieved",
            repo_id=self.repo_id,
            operation="search",
            result_count=len(ranked),
        )
        return [
            _row_to_symbol(row_cache[symbol_id]).model_copy(update={"score": score})
            for symbol_id, (score, _channel_rank) in ranked[:limit]
        ]

    def _build_symbol_embeddings(self, conn: sqlite3.Connection, index_version: int) -> None:
        """Embed every symbol into the persistent vector store -- part of the code
        index build, run once at index time. No-op unless an embedder is configured
        (default Null), so non-semantic indexing is unaffected. Keeping embedding
        here makes the query hot path read-only: it embeds only the query, never
        documents.
        """
        if not self._semantic_ranker.available:
            return
        embedder = self._semantic_ranker.embedder
        dim = int(getattr(embedder, "dim", 0))
        if dim <= 0:
            return
        rows = conn.execute("SELECT * FROM symbols WHERE repo_id = ?", (self.repo_id,)).fetchall()
        if not rows:
            return
        # Skip symbols whose vector is already current. symbol_id encodes the file
        # content hash, so an unchanged symbol keeps its id across reindexes and is
        # skipped here; an edited symbol gets a new id and its stale vector is pruned
        # by _delete_file_index. index_version stays provenance-only -- gating on it
        # would make every reindex re-embed the whole repo instead of just the delta.
        fresh = self._ann_symbol_index.existing_stamped_ids(conn, embedder_name=embedder.name, embedding_dim=dim)
        pending = [sym for sym in (_row_to_symbol(row) for row in rows) if sym.symbol_id not in fresh]
        if not pending:
            return
        # Sort by file so each outer chunk touches a small, contiguous set of
        # files instead of symbols interleaved across the whole repo (the DB
        # SELECT above has no ORDER BY). _read_file_slice's cache is capped at
        # 64MB (_FILE_BYTES_CACHE_MAX_BYTES) and wipes itself wholesale once
        # full -- on a repo whose source exceeds that by 10-20x (linux), a
        # file-interleaved chunk thrashes the cache on nearly every symbol,
        # which shows up as disk I/O wait (D state), not GPU or CPU time.
        pending.sort(key=lambda sym: sym.file_path)
        logger.debug("[embed] %s: %d symbols to embed (%s)", self.repo_root.name, len(pending), embedder.name)
        # Batched encoding: ceil(N/batch) model calls (embed_symbols), not one per
        # symbol -- the difference between minutes and seconds on a large repo.
        # Truncate source: a few symbols span enormous spans whose full token
        # sequence blows GPU memory (attention is O(seq^2)). The head carries the
        # semantic signal; cap chars so every input is bounded. Override via env.
        max_chars = int(os.environ.get("LEMONCROW_EMBED_MAX_CHARS", "4000"))
        # Process + commit in outer chunks instead of embedding every pending
        # symbol before a single final write: holding all of them at once means
        # two full float copies in memory simultaneously (embed_symbols' result
        # dict, then this method's new_vectors dict) -- tens of GB at linux scale
        # (1.24M symbols x 1536 dims), which OOM-killed a real backfill run before
        # it ever reached the write. Chunking bounds peak memory to one chunk and,
        # as a side effect, makes an interrupted backfill resumable: whatever
        # chunks already committed are skipped by the `fresh` check above on the
        # next run instead of restarting from zero.
        chunk_size = int(os.environ.get("LEMONCROW_EMBED_COMMIT_CHUNK", "20000"))
        total_embedded = 0
        for chunk_start in range(0, len(pending), chunk_size):
            chunk = pending[chunk_start : chunk_start + chunk_size]
            source_texts = {
                sym.symbol_id: self._read_file_slice(sym.file_path, sym.start_byte, sym.end_byte)[:max_chars]
                for sym in chunk
            }
            by_id = {sym.symbol_id: sym for sym in chunk}
            vectors = self._semantic_ranker.embed_symbols(chunk, source_texts=source_texts)
            new_vectors = {
                sid: (by_id[sid].content_hash, vec) for sid, vec in vectors.items() if vec and len(vec) == dim
            }
            if new_vectors:
                self._ann_symbol_index.upsert_vectors(
                    conn,
                    embedder_name=embedder.name,
                    embedding_dim=dim,
                    index_version=index_version,
                    vectors=new_vectors,
                )
                conn.commit()
                total_embedded += len(new_vectors)
        logger.debug("[embed] %s: got %d vectors", self.repo_root.name, total_embedded)

    def _ann_flat_cache_paths(self, embedder_name: str, embedding_dim: int) -> tuple[Path, Path, Path]:
        """Sibling flat-file cache paths for the resident ANN matrix.

        Three files, not one, so a partial/interrupted write can never be
        mistaken for a complete one: matrix.npy and ids.npy are mmap-able
        directly (np.load(..., mmap_mode="r")), and meta.json -- always
        (re)written LAST, after both arrays are safely renamed into place --
        is what a reader trusts to decide the pair is current. A crash
        between the two array renames leaves meta.json still pointing at
        the OLD (embedder, dim, index_version) key, so a reader for that key
        finds the length/shape check below satisfied by the old, still
        self-consistent pair; a reader for a NEW key (the common case, since
        this only reruns after a reindex bump) correctly treats it as a miss
        and rebuilds from SQL.
        """
        # Built via plain string formatting, NOT Path.with_suffix() chaining: the
        # db stem/embedder segment already contains a "." (db_path.stem itself,
        # e.g. "code_context"), and with_suffix() replaces EVERYTHING from the
        # first "." onward in the final path component -- three separate
        # with_suffix() calls on the same dotted base collapsed every
        # (embedder, dim) combination onto the identical "code_context.matrix.npy"
        # / ".ids.npy" / ".meta.json" filenames, silently sharing one cache slot
        # (and one stale/wrong entry) across different embedders and dims.
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", embedder_name)
        base_name = f"{self.db_path.stem}.ann_{safe}_{embedding_dim}"
        parent = self.db_path.parent
        return (
            parent / f"{base_name}.matrix.npy",
            parent / f"{base_name}.ids.npy",
            parent / f"{base_name}.meta.json",
        )

    def _load_ann_flat_cache(
        self, embedder_name: str, embedding_dim: int, index_version: int
    ) -> tuple[list[str], Any] | None:
        """mmap-load the resident matrix from the flat-file cache, or None on any miss.

        A cold load through SQLite (fetchall + join over hundreds of thousands
        of blob rows) is CPU-bound Python/DB-API marshaling overhead, not disk
        or compute -- confirmed via a real run: 3.7s in fetchall() + 1.4s in
        the byte-join for linux's 727k vectors, vs 87ms for the matmul that
        actually does the search. A flat file the OS can page in directly
        skips that marshaling entirely on every process restart after the
        first successful build.
        """
        matrix_path, ids_path, meta_path = self._ann_flat_cache_paths(embedder_name, embedding_dim)
        try:
            import numpy as np

            meta = json.loads(meta_path.read_text())
            if (
                meta.get("embedder_name") != embedder_name
                or int(meta.get("embedding_dim", -1)) != embedding_dim
                or int(meta.get("index_version", -1)) != index_version
            ):
                return None
            ids_arr = np.load(ids_path, mmap_mode="r")
            matrix = np.load(matrix_path, mmap_mode="r")
            # Sanity check, not just a trust exercise: guards the narrow window
            # (see _ann_flat_cache_paths) where the two arrays could have been
            # replaced out of step with each other under the same cache key.
            if matrix.ndim != 2 or matrix.shape[1] != embedding_dim or matrix.shape[0] != ids_arr.shape[0]:
                return None
            return [str(x) for x in ids_arr], matrix
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def _save_ann_flat_cache(
        self, embedder_name: str, embedding_dim: int, index_version: int, ids: list[str], matrix: Any
    ) -> None:
        """Best-effort persist of a freshly SQL-loaded matrix for the next cold start.

        Never raises: a failed write just means the next process pays the SQL
        load again, exactly today's behaviour. Atomic per-file replace (temp +
        os.replace) so a reader never observes a half-written array; meta.json
        last so it only ever certifies a fully-written pair (see
        _ann_flat_cache_paths for the residual edge case this doesn't cover).
        """
        if not ids:
            # Never persist an empty result: a transient 0-row read (wrong
            # repo_id, a reindex mid-flight, an embedder not yet backfilled)
            # would otherwise "validly" match this exact cache key for every
            # future request and silently serve zero semantic hits until the
            # key itself changes (a new index_version) -- confirmed via a real
            # run in this session's own testing.
            return
        try:
            import numpy as np

            matrix_path, ids_path, meta_path = self._ann_flat_cache_paths(embedder_name, embedding_dim)
            ids_arr = np.asarray(ids, dtype=f"<U{max(1, max((len(i) for i in ids), default=1))}")
            # np.save() auto-appends ".npy" to any filename that doesn't already
            # end in it, so a plain "<name>.npy.tmp" would silently become
            # "<name>.npy.tmp.npy" and the later os.replace() below would raise
            # FileNotFoundError on the name we actually asked for. Keep the temp
            # name ending in ".npy" so np.save writes exactly the path we track.
            tmp_matrix = matrix_path.with_name(matrix_path.stem + ".tmp.npy")
            tmp_ids = ids_path.with_name(ids_path.stem + ".tmp.npy")
            tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
            np.save(tmp_matrix, np.ascontiguousarray(matrix, dtype=np.float32))
            np.save(tmp_ids, ids_arr)
            os.replace(tmp_matrix, matrix_path)
            os.replace(tmp_ids, ids_path)
            tmp_meta.write_text(
                json.dumps(
                    {
                        "embedder_name": embedder_name,
                        "embedding_dim": embedding_dim,
                        "index_version": index_version,
                        "count": len(ids),
                    }
                )
            )
            os.replace(tmp_meta, meta_path)
        except Exception:
            logger.debug("ANN flat-file cache write failed; will retry next cold load", exc_info=True)

    def _ensure_ann_matrix_loaded(
        self, embedder_name: str, embedding_dim: int, index_version: int
    ) -> tuple[list[str], Any] | None:
        """Return (ids, matrix) for this cache key, loading + caching it if needed.

        Single point of entry for every path that can (re)build the resident
        ANN matrix -- the query-time semantic search below, ``prewarm_semantic_matrix``,
        and the background ``_maybe_warm_ann_cache`` thread all call this instead
        of duplicating the load. ``_ann_matrix_loading_lock`` makes the actual
        (SQL or flat-file) load single-flight across ALL of them: whichever
        caller gets there first does the real work; every other concurrent or
        subsequent caller for the SAME key either reuses the just-populated
        ``self._ann_vectors_cache`` (checked both before AND after the lock wait,
        so a caller that loses the initial race still benefits once the winner
        finishes -- a non-blocking "busy -> give up" design here would instead
        make the loser permanently treat this query as semantic-empty even
        after the cache warmed, confirmed via a real run) or gives up after the
        30s safety-net wait (returns None; a real timeout expiring never blocks
        past what the caller itself already bounds).
        """
        cache_key = (embedder_name, embedding_dim, index_version)
        cached = self._ann_vectors_cache
        if cached is not None and cached[0] == cache_key:
            return cached[1], cached[2]
        got_lock = self._ann_matrix_loading_lock.acquire(timeout=30.0)
        try:
            cached2 = self._ann_vectors_cache
            if cached2 is not None and cached2[0] == cache_key:
                return cached2[1], cached2[2]
            if not got_lock:
                return None
            flat_cached = self._load_ann_flat_cache(embedder_name, embedding_dim, index_version)
            if flat_cached is not None:
                ids, matrix = flat_cached
            else:
                with self._connect() as conn:
                    self._init_schema(conn)
                    ids, matrix = self._ann_symbol_index.load_current_matrix(
                        conn, embedder_name=embedder_name, embedding_dim=embedding_dim
                    )
                self._save_ann_flat_cache(embedder_name, embedding_dim, index_version, ids, matrix)
            self._ann_vectors_cache = (cache_key, ids, matrix)
            return ids, matrix
        finally:
            if got_lock:
                self._ann_matrix_loading_lock.release()

    def _search_symbols_semantic_local(
        self,
        query: str,
        *,
        limit: int = 20,
        kind: str | None = None,
        language: str | None = None,
    ) -> list[SymbolRecord]:
        # Semantic search reads the index-time vector store (_build_symbol_embeddings)
        # via ANN; the hot path embeds ONLY the query. A configured embedder is the
        # single enable -- there is no separate ANN flag (ANN vs exact is internal).
        if not self._semantic_ranker.available:
            return []
        return self._search_symbols_semantic_ann(query, limit=limit, kind=kind, language=language)

    def _search_symbols_semantic_ann(
        self,
        query: str,
        *,
        limit: int,
        kind: str | None = None,
        language: str | None = None,
    ) -> list[SymbolRecord]:
        """Opt-in semantic search over the persistent per-symbol vector store.

        Exact brute-force cosine over packed float32 blobs -- no JSON parsing and
        no approximate index. The blob store reconstructs with np.frombuffer
        (~100x faster than json.loads: 2s vs 218s for linux's 1.24M vectors) and
        the matmul itself is memory-bandwidth-bound (~141ms at linux scale).

        Small repos (≤ _ANN_CACHE_LIMIT vectors): load once, cache as a numpy
        matrix, rank with a single ``matrix @ query_vec`` product (<10ms warm).

        Large repos (> _ANN_CACHE_LIMIT): stream in _ANN_CHUNK_SIZE rows at a
        time, frombuffer each chunk, keep a rolling top-K heap -- peak RAM ≈ one
        chunk (~300 MB at dim=1536, chunk=50k) instead of the full matrix (7.5 GB
        for linux). No matrix cache in this path.

        N5 (model-id/dim drift) and N16 (index_version staleness) are enforced
        in both paths via the embedder_name + embedding_dim filters.
        """
        # Rows below this threshold are loaded into a cached matrix (fast repeat
        # queries, ~2s one-time load then a single in-memory matmul per query);
        # above it, chunked streaming re-reads the whole table from disk on
        # EVERY query instead -- the dominant cost at linux scale (~20-30s/query,
        # ~25 chunked disk reads of a 1.24M-row table per query, even though the
        # matmul itself is only ~1.9 GFLOPs / tens of ms once resident). 200k was
        # conservative for memory-constrained hosts (a full matrix is
        # rows*dim*4 bytes, ~7.5GB for linux's 1.24M*1536); raised to 2M so a
        # single ~7.5GB resident matrix (trivial on a machine with double-digit+
        # GB RAM) beats 25 disk round-trips per query. Overridable down for truly
        # memory-constrained hosts.
        _ANN_CACHE_LIMIT = int(os.environ.get("LEMONCROW_ANN_CACHE_LIMIT", "2000000"))
        _ANN_CHUNK_SIZE = 50_000  # rows/chunk ≈ 300 MB peak at dim=1536

        embedder = self._semantic_ranker.embedder
        embedding_dim = embedder.dim
        if embedding_dim <= 0:
            return []
        query_vector = self._semantic_ranker.embed_query(query)
        if not query_vector:
            return []
        index_version = self._current_index_version()
        try:
            import numpy as np
        except ModuleNotFoundError:
            return []

        qvec = np.asarray(query_vector, dtype=np.float32)
        # Hydration window: over-fetch to survive kind/language filters.
        window = limit if (kind is None and language is None) else max(limit * 20, 200)

        # In-DB ANN via the sqlite-vector extension: avoids re-streaming the whole
        # symbol_vectors table from disk on every query, which is what the numpy
        # path below does above _ANN_CACHE_LIMIT rows -- the dominant cost at
        # linux scale (~20-30s/query, ~25 chunked disk reads of a 1.24M-row table
        # per single query). use_quantization=False (TurboQuant off): exact,
        # uncompressed cosine distances while this path is newly wired in.
        # Returns None (falls through to the numpy path below) when the
        # extension, index, or scan is unavailable for any reason.
        if os.environ.get("LEMONCROW_DISABLE_SQLITE_VECTOR", "").strip().lower() not in {"1", "true", "yes", "on"}:
            sqlite_vec_result = self._search_symbols_sqlite_vector(
                query_vector,
                embedder_name=embedder.name,
                embedding_dim=embedding_dim,
                index_version=index_version,
                limit=limit,
                window=window,
                kind=kind,
                language=language,
                use_quantization=False,
            )
            if sqlite_vec_result is not None:
                return sqlite_vec_result

        cache_key = (embedder.name, embedding_dim, index_version)
        cached = self._ann_vectors_cache

        # ── small-repo fast path: cached matrix ──────────────────────────────
        if cached is not None and cached[0] == cache_key:
            ids, matrix = cached[1], cached[2]
        else:
            # Count vectors to choose load strategy without loading data yet. The
            # store lives in the connection's main schema (an unqualified name
            # resolves there even with the empty vectors.sqlite attached).
            with self._connect() as count_conn:
                self._init_schema(count_conn)
                try:
                    vec_count: int = count_conn.execute(
                        "SELECT COUNT(*) FROM symbol_vectors WHERE repo_id=? AND embedder_name=? AND embedding_dim=?",
                        (self._ann_symbol_index.repo_id, embedder.name, embedding_dim),
                    ).fetchone()[0]
                except (KeyError, TypeError, ValueError):
                    vec_count = 0

            if vec_count > _ANN_CACHE_LIMIT:
                # ── large-repo chunked path (bounded RAM, exact) ──────────────
                # Stream _ANN_CHUNK_SIZE rows at a time; each chunk's packed blobs
                # reconstruct with one np.frombuffer (no json.loads, no per-row
                # Python lists), so peak RAM ≈ one chunk (~300 MB at dim=1536)
                # instead of the full matrix (7.5 GB for linux). A rolling
                # min-heap keeps the top-window without a full-corpus sort.
                import heapq

                _bytes_per_vec = embedding_dim * 4
                heap: list[tuple[float, str]] = []  # min-heap by score
                offset = 0
                with self._connect() as stream_conn:
                    self._init_schema(stream_conn)
                    while True:
                        rows = stream_conn.execute(
                            "SELECT symbol_id, vector_blob FROM symbol_vectors"
                            " WHERE repo_id=? AND embedder_name=? AND embedding_dim=?"
                            " LIMIT ? OFFSET ?",
                            (self._ann_symbol_index.repo_id, embedder.name, embedding_dim, _ANN_CHUNK_SIZE, offset),
                        ).fetchall()
                        if not rows:
                            break
                        offset += _ANN_CHUNK_SIZE
                        chunk_ids: list[str] = []
                        buf = bytearray()
                        for r in rows:
                            blob = r[1]
                            if not isinstance(blob, (bytes, bytearray, memoryview)) or len(blob) != _bytes_per_vec:
                                continue
                            chunk_ids.append(str(r[0]))
                            buf += bytes(blob)
                        if not chunk_ids:
                            continue
                        chunk_vecs = np.frombuffer(bytes(buf), dtype=np.float32).reshape(len(chunk_ids), embedding_dim)
                        chunk_scores = chunk_vecs @ qvec
                        for i, score in enumerate(chunk_scores):
                            if score <= 0:
                                continue
                            entry = (float(score), chunk_ids[i])
                            if len(heap) < window:
                                heapq.heappush(heap, entry)
                            elif score > heap[0][0]:
                                heapq.heapreplace(heap, entry)
                # Sort heap descending, hydrate.
                top = sorted(heap, key=lambda x: -x[0])
                top_ids = [t[1] for t in top]
                top_scores = {t[1]: t[0] for t in top}
                hydrated = self._hydrate_symbols_by_id(top_ids, kind=kind, language=language)
                results: list[SymbolRecord] = []
                for sid in top_ids:
                    rec = hydrated.get(sid)
                    if rec is None:
                        continue
                    results.append(rec.model_copy(update={"score": top_scores[sid]}))
                    if len(results) >= limit:
                        break
                return results
            else:
                # small repo: load all + cache matrix -- single-flight + flat-file
                # backed (see _ensure_ann_matrix_loaded). None means another
                # loader is still working and the 30s safety-net wait expired
                # (the caller's own deadline, e.g. _SEMANTIC_SYMBOL_DEADLINE_S,
                # would fire well before that for any reasonable configuration).
                loaded = self._ensure_ann_matrix_loaded(embedder.name, embedding_dim, index_version)
                if loaded is None:
                    return []
                ids, matrix = loaded

        if len(ids) == 0:
            return []
        # Vectorised cosine: unit-normalised vectors → single matrix product.
        scores = matrix @ qvec
        order = np.argsort(-scores)
        results = []
        pos = 0
        total = int(order.shape[0])
        while len(results) < limit and pos < total:
            batch = order[pos : pos + window]
            pos += window
            batch_ids = [ids[int(i)] for i in batch]
            hydrated = self._hydrate_symbols_by_id(batch_ids, kind=kind, language=language)
            for i in batch:
                rec = hydrated.get(ids[int(i)])
                if rec is None:
                    continue
                results.append(rec.model_copy(update={"score": float(scores[int(i)])}))
                if len(results) >= limit:
                    break
        return results

    def warm_query_path(self) -> dict[str, float]:
        """Warm every one-time cost the first live query would otherwise pay.

        - OS page cache: on a large cold DB the lexical structures (symbols /
          symbol_trigram / symbol_fts / file_line_fts) are spread thin across
          the file, and the first query pages them in from disk (measured 33s
          cold vs <0.4s warm on an 11.9GB DB). No-match probes touch exactly
          the structures the real channels scan; the page cache is
          kernel-global, so every process sharing the DB benefits.
        - centrality map: load the persisted map (or compute+persist once).
        - ANN vector matrix: resident-cache load via ``prewarm_semantic_matrix``
          (no-op without an embedder or above the cache cap).

        Fail-open, read-only apart from centrality persistence, safe from a
        background thread at server startup. Returns per-step seconds.
        """
        timings: dict[str, float] = {}
        _t0 = time.perf_counter()
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0)
            try:
                for probe_sql in (
                    "SELECT COUNT(*) FROM symbols WHERE symbol_name LIKE '%zqx9zzz%'",
                    "SELECT COUNT(*) FROM symbol_trigram WHERE name LIKE '%zqx9zzz%'",
                    "SELECT COUNT(*) FROM symbol_fts WHERE symbol_fts MATCH 'zqx9zzz'",
                    "SELECT COUNT(*) FROM file_line_fts WHERE file_line_fts MATCH 'zqx9zzz'",
                ):
                    with contextlib.suppress(sqlite3.Error):
                        conn.execute(probe_sql).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            logger.debug("lexical page-cache warm skipped", exc_info=True)
        timings["page_cache_s"] = round(time.perf_counter() - _t0, 3)
        _t0 = time.perf_counter()
        with contextlib.suppress(Exception):
            self._symbol_centrality_map()
        timings["centrality_s"] = round(time.perf_counter() - _t0, 3)
        _t0 = time.perf_counter()
        with contextlib.suppress(Exception):
            self.prewarm_semantic_matrix()
        timings["ann_matrix_s"] = round(time.perf_counter() - _t0, 3)
        return timings

    def prewarm_semantic_matrix(self) -> bool:
        """Load the ANN vector matrix into the in-memory cache ahead of the first
        query so it never pays the cold load (matrix read + unpack: ~200 ms for a
        24k-vector repo). Returns True when the matrix is resident afterwards.

        No-op (returns False) when no embedder is configured, nothing is stored,
        or the store exceeds the matrix-cache cap -- those repos use the chunked
        streaming path, which holds no cached matrix by design. Safe to call from
        a background thread at index-ready time; idempotent within an index
        version (the cache key carries index_version, so a reindex re-warms).
        """
        ranker = self._semantic_ranker
        if not getattr(ranker, "available", False):
            return False
        embedder = ranker.embedder
        dim = int(getattr(embedder, "dim", 0))
        if dim <= 0:
            return False
        index_version = self._current_index_version()
        cache_key = (embedder.name, dim, index_version)
        cached = self._ann_vectors_cache
        if cached is not None and cached[0] == cache_key:
            return True
        # A reindex bumps index_version (and a repo grown past the cap takes the
        # count>cap early-return below); either way any previously loaded matrix
        # is now stale. Drop it up front so we never keep a superseded matrix
        # resident when this returns without re-warming.
        if cached is not None:
            self._ann_vectors_cache = None
        cap = int(os.environ.get("LEMONCROW_ANN_CACHE_LIMIT", "2000000"))
        with self._connect() as conn:
            self._init_schema(conn)
            try:
                count = conn.execute(
                    "SELECT COUNT(*) FROM symbol_vectors WHERE repo_id=? AND embedder_name=? AND embedding_dim=?",
                    (self._ann_symbol_index.repo_id, embedder.name, dim),
                ).fetchone()[0]
            except sqlite3.Error:
                return False
            if count == 0 or count > cap:
                return False
        # Single-flight + flat-file backed (see _ensure_ann_matrix_loaded) so
        # this shares the same warm cache as query-time semantic search and
        # _maybe_warm_ann_cache instead of racing them with its own redundant load.
        loaded = self._ensure_ann_matrix_loaded(embedder.name, dim, index_version)
        return loaded is not None and len(loaded[0]) > 0

    def _sqlite_vector_conn(self) -> sqlite3.Connection | None:
        """Cached per-thread direct connection to vectors.sqlite with the
        sqlite_vector extension loaded, or None when it is unavailable.

        The extension only operates on a connection's *main* schema, so the vectors
        DB is opened directly (bare ``symbol_vectors``) instead of via the attached
        ``vectors`` alias used elsewhere. Kept open for the engine's lifetime so the
        TurboQuant data stays resident. A missing package or a failed load sets a
        permanent flag so the numpy fallback is used without re-probing; a merely
        absent DB file is transient (retried on the next query).
        """
        if self._sqlite_vec_disabled:
            return None
        conn = getattr(self._sqlite_vec_tls, "conn", None)
        if conn is not None:
            return conn
        ext_path = _sqlite_vector_extension_path()
        if ext_path is None:
            self._sqlite_vec_disabled = True
            return None
        vpath = self.vectors_db_path
        if not vpath.exists():
            return None
        conn = None
        try:
            # check_same_thread=False so disposal can close connections created on
            # other worker threads; each is still only *used* by its creating thread
            # (stored on the thread-local), so there is no concurrent use beyond the
            # best-effort close at teardown.
            conn = sqlite3.connect(vpath, timeout=30.0, check_same_thread=False)
            conn.enable_load_extension(True)
            conn.load_extension(ext_path)
            conn.enable_load_extension(False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA mmap_size = 268435456")
            # Adds vector_blob + backfills from JSON on stores that predate it, so
            # the scan works without waiting for a full reindex.
            ensure_symbol_vector_schema(conn)
        except (sqlite3.Error, AttributeError):
            # AttributeError: a Python build compiled without enable_load_extension.
            logger.debug("sqlite-vector unavailable; using numpy path", exc_info=True)
            if conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    conn.close()
            self._sqlite_vec_disabled = True
            return None
        self._sqlite_vec_tls.conn = conn
        with self._sqlite_vec_lock:
            self._sqlite_vec_conns.append(conn)
        return conn

    def _ensure_sqlite_vector_index(
        self,
        conn: sqlite3.Connection,
        *,
        embedder_name: str,
        embedding_dim: int,
        index_version: int,
        quantize: bool = True,
    ) -> bool:
        """Make ``conn`` ready to scan symbol_vectors; False → fall back.

        ``vector_init`` is required on every connection regardless of mode.
        ``quantize=True`` (default) prepares the one-time TurboQuant column so
        callers can use ``vector_quantize_scan`` -- persisted in the DB, keyed by
        ``index_version`` via a small marker table, so a reindex re-quantizes
        exactly once and separate CLI processes reuse a prior build.
        ``quantize=False`` skips all of that and only does ``vector_init``, for
        callers using the unquantized ``vector_full_scan`` instead -- no
        persisted state, no preload, cheaper to keep ready but scans raw floats
        rather than the compressed TurboQuant representation.
        ``preload`` is a per-connection RAM speedup, capped so a huge corpus stays
        mmap-backed. Readiness is memoised on the thread-local, so the steady state
        is a single key comparison per query.
        """
        tls = self._sqlite_vec_tls
        key = (embedder_name, embedding_dim, index_version, quantize)
        if getattr(tls, "ready_key", None) == key:
            return True
        spec = f"type=FLOAT32,dimension={embedding_dim},distance=COSINE"
        try:
            conn.execute("SELECT vector_init('symbol_vectors', 'vector_blob', ?)", (spec,))
        except sqlite3.Error:
            logger.debug("sqlite-vector vector_init failed; using numpy path", exc_info=True)
            return False
        if not quantize:
            tls.ready_key = key
            return True
        with self._sqlite_vec_lock:
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS _sqliteai_quant_state ("
                    " embedder_name TEXT NOT NULL, embedding_dim INTEGER NOT NULL,"
                    " index_version INTEGER NOT NULL,"
                    " PRIMARY KEY (embedder_name, embedding_dim))"
                )
                row = conn.execute(
                    "SELECT index_version FROM _sqliteai_quant_state WHERE embedder_name = ? AND embedding_dim = ?",
                    (embedder_name, embedding_dim),
                ).fetchone()
                if row is None or int(row[0]) != index_version:
                    # Fully (re)quantize the column for this index version — the
                    # extension re-quantizes all rows, folding in any added since
                    # the previous build. Raises (caught below) on a heterogeneous
                    # column during a model-drift window → numpy fallback.
                    conn.execute("SELECT vector_quantize('symbol_vectors', 'vector_blob', 'qtype=TURBO4')")
                    conn.execute(
                        "INSERT INTO _sqliteai_quant_state (embedder_name, embedding_dim, index_version)"
                        " VALUES (?, ?, ?)"
                        " ON CONFLICT(embedder_name, embedding_dim)"
                        " DO UPDATE SET index_version = excluded.index_version",
                        (embedder_name, embedding_dim, index_version),
                    )
                    conn.commit()
            except sqlite3.Error:
                logger.debug("sqlite-vector quantize failed; using numpy path", exc_info=True)
                return False
        # Preload the quantized data into RAM once per connection, under the cap
        # (the scan works without preload, reading the quantized rows via mmap).
        try:
            mem_row = conn.execute("SELECT vector_quantize_memory('symbol_vectors', 'vector_blob')").fetchone()
            mem_bytes = int(mem_row[0]) if mem_row and mem_row[0] is not None else 0
            if 0 < mem_bytes <= _SQLITE_VEC_PRELOAD_MAX_BYTES:
                conn.execute("SELECT vector_quantize_preload('symbol_vectors', 'vector_blob')")
        except sqlite3.Error:
            logger.debug("sqlite-vector preload skipped", exc_info=True)
        tls.ready_key = key
        return True

    def _search_symbols_sqlite_vector(
        self,
        query_vector: list[float],
        *,
        embedder_name: str,
        embedding_dim: int,
        index_version: int,
        limit: int,
        window: int,
        kind: str | None,
        language: str | None,
        use_quantization: bool = True,
    ) -> list[SymbolRecord] | None:
        """In-DB ANN over symbol_vectors via the sqlite-vector extension, replacing
        the numpy matrix scan.

        ``use_quantization=True`` (default) scans the one-time-built TurboQuant
        column via ``vector_quantize_scan``. ``use_quantization=False`` skips
        quantization entirely and scans the raw float32 blobs directly via
        ``vector_full_scan`` -- no persisted quantization state, simpler to reason
        about for an A/B latency comparison, exact (not compressed) distances.

        Returns ranked records on success, or None to signal the caller to fall
        back to the numpy path (extension unavailable, not scannable, or a query
        error). ``distance`` is cosine distance in [0, 2]; similarity is
        ``1 - distance``. The scan over-fetches (``window``) so the N5 drift filter
        and the kind/language hydration filter have candidates to spare.
        """
        import struct

        conn = self._sqlite_vector_conn()
        if conn is None:
            return None
        if not self._ensure_sqlite_vector_index(
            conn,
            embedder_name=embedder_name,
            embedding_dim=embedding_dim,
            index_version=index_version,
            quantize=use_quantization,
        ):
            return None
        try:
            qblob = struct.pack(f"{embedding_dim}f", *query_vector)
        except (struct.error, TypeError, ValueError):
            return None
        scan_k = max(window, 200)
        scan_fn = "vector_quantize_scan" if use_quantization else "vector_full_scan"
        try:
            rows = conn.execute(
                "SELECT sv.symbol_id AS sid, v.distance AS dist "
                f"FROM {scan_fn}('symbol_vectors', 'vector_blob', ?, ?) AS v "
                "JOIN symbol_vectors sv ON sv.rowid = v.rowid "
                "WHERE sv.repo_id = ? AND sv.embedder_name = ? AND sv.embedding_dim = ? "
                "ORDER BY v.distance",
                (qblob, scan_k, self.repo_id, embedder_name, embedding_dim),
            ).fetchall()
        except sqlite3.Error:
            logger.debug("sqlite-vector scan failed; using numpy path", exc_info=True)
            return None
        if not rows:
            # A genuine "nothing embedded yet" repo and a *dedicated connection
            # pointed at the wrong/stale vectors store* both surface identically
            # here (0 rows), and are indistinguishable from inside this query.
            # Falling through to the numpy path (which reads via the engine's
            # normal, always-correctly-routed connection) costs nothing extra
            # when the data truly doesn't exist -- that path also finds 0 rows
            # and returns [] itself -- but recovers the real ranked results when
            # it does exist. Returning [] here instead would let a scan against
            # a wrong/empty vectors store silently masquerade as "confidently
            # zero real matches", which is exactly what happened for this
            # session's benchmark DBs: symbol_vectors lives in each repo's main
            # db file (single-file legacy schema), while _sqlite_vector_conn()
            # opens a dedicated connection straight to the sibling vectors.sqlite
            # -- empty for these repos -- so every semantic query returned 0
            # results in ~4ms instead of falling back, silently gutting the
            # semantic channel corpus-wide.
            return None
        top_ids = [str(r["sid"]) for r in rows]
        scores = {str(r["sid"]): 1.0 - float(r["dist"]) for r in rows}
        hydrated = self._hydrate_symbols_by_id(top_ids, kind=kind, language=language)
        results: list[SymbolRecord] = []
        for sid in top_ids:
            rec = hydrated.get(sid)
            if rec is None:
                continue
            results.append(rec.model_copy(update={"score": scores[sid]}))
            if len(results) >= limit:
                break
        if not results:
            # Rows existed but the kind/language hydration filter dropped every
            # one. Return None (not []) so the caller falls back to the numpy
            # path instead of treating an over-filtered scan as a confident
            # empty result.
            return None
        return results

    def _hydrate_symbols_by_id(
        self,
        symbol_ids: list[str],
        *,
        kind: str | None = None,
        language: str | None = None,
    ) -> dict[str, SymbolRecord]:
        """Load full records for the given ids (the semantic-ranking winners),
        applying the optional kind/language filter in SQL. Only the ranked top
        slice is hydrated, so the hot path never builds records for non-winners."""
        if not symbol_ids:
            return {}
        filters = ["repo_id = ?"]
        params: list[Any] = [self.repo_id]
        if kind:
            filters.append("kind = ?")
            params.append(kind)
        if language:
            filters.append("language = ?")
            params.append(language)
        placeholders = ",".join("?" * len(symbol_ids))
        filters.append(f"symbol_id IN ({placeholders})")
        params.extend(symbol_ids)
        where_sql = " AND ".join(filters)
        # Read-only + no _init_schema: we only reach here after vectors were loaded,
        # so the symbols table exists. Skipping the ~15 CREATE TABLE IF NOT EXISTS
        # statements per query is most of the semantic hot-path latency.
        with self._connect(readonly=True) as conn:
            rows = conn.execute(
                f"SELECT *, NULL AS score FROM symbols WHERE {where_sql}",
                tuple(params),
            ).fetchall()
        return {str(row["symbol_id"]): _row_to_symbol(row) for row in rows}

    def get_symbol(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        """Retrieve exact symbol metadata and source by byte offsets."""
        if auto_index:
            self._ensure_indexed()
        return self.intel_store.get_symbol(
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            file_path=file_path,
            symbol_name=symbol_name,
        )

    def _get_symbol_local(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
    ) -> dict[str, Any]:
        clauses = ["repo_id = ?"]
        params: list[Any] = [self.repo_id]
        if symbol_id:
            clauses.append("symbol_id = ?")
            params.append(symbol_id)
        if qualified_name:
            clauses.append("qualified_name = ?")
            params.append(qualified_name)
        if symbol_name:
            clauses.append("symbol_name = ?")
            params.append(symbol_name)
        if file_path:
            clauses.append("file_path = ?")
            params.append(self._normalize_file_arg(file_path))
        if len(clauses) == 1:
            raise ValueError("symbol_id, qualified_name, symbol_name, or file_path is required")
        with self._connect() as conn:
            self._init_schema(conn)
            row = conn.execute(
                f"SELECT *, NULL AS score FROM symbols WHERE {' AND '.join(clauses)} ORDER BY file_path, start_line LIMIT 1",
                tuple(params),
            ).fetchone()
        if row is None:
            raise LookupError("symbol not found")
        symbol = _row_to_symbol(row)
        path = self.repo_root / symbol.file_path
        try:
            source = path.read_bytes()[symbol.start_byte : symbol.end_byte].decode("utf-8", errors="replace")
        except OSError:
            # The index can reference a file absent from disk (deleted, moved, or
            # snapshot-excluded since indexing). Return the symbol metadata with an
            # empty body so callers (explore relationship resolution, node, ...)
            # degrade instead of crashing on one stale entry.
            source = ""
        emit_product_local("code_symbol_retrieved", repo_id=self.repo_id, kind=symbol.kind)
        return {**symbol.model_dump(mode="json"), "source": source}

    def file_outline(
        self,
        *,
        file_path: str | None = None,
        limit: int = 200,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        """Return compact file or repository symbol outlines."""
        if auto_index:
            self._ensure_indexed()
        params: list[Any] = [self.repo_id]
        where = "repo_id = ?"
        if file_path:
            where += " AND file_path = ?"
            params.append(self._normalize_file_arg(file_path))
        params.append(limit)
        with self._connect() as conn:
            self._init_schema(conn)
            rows = conn.execute(
                f"""
                SELECT *, NULL AS score FROM symbols
                WHERE {where}
                ORDER BY file_path, start_line
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            record = _row_to_symbol(row)
            entry: dict[str, Any] = {
                "name": record.symbol_name,
                "kind": record.kind,
                "signature": record.signature,
                "line_start": record.start_line,
                "line_end": record.end_line,
            }
            # Drop qualified_name when it duplicates name (the common case for
            # module-level symbols) — redundant bytes the agent never needs.
            if record.qualified_name and record.qualified_name != record.symbol_name:
                entry["qualified_name"] = record.qualified_name
            grouped.setdefault(record.file_path, []).append(entry)
        return {"repo_id": self.repo_id, "files": grouped, "symbol_count": len(rows)}

    def repo_map(self, *, seed_files: list[str] | None = None, budget_tokens: int = 2000) -> dict[str, Any]:
        """Build an Aider-style PageRank repo map with a token budget."""
        normalized = [self._normalize_file_arg(seed) for seed in seed_files or []]
        result = build_repo_map(self.repo_root, seed_files=normalized, budget_tokens=budget_tokens)
        return result.model_dump(mode="json") | {"mode": "map"}

    def _render_context_code_block(self, symbol: SymbolRecord, source_block: str) -> str:
        block_header = f"### {symbol.qualified_name} ({symbol.file_path}:{symbol.start_line}-{symbol.end_line})"
        return f"{block_header}\n```{symbol.language}\n{source_block}\n```"

    def _context_content_with_candidate(self, lines: list[str], *, block: str | None = None) -> str:
        candidate_lines = list(lines)
        if block is not None:
            candidate_lines.extend([block, ""])
        return "\n".join(candidate_lines).strip()

    def _fit_context_code_block_source(
        self,
        *,
        lines: list[str],
        symbol: SymbolRecord,
        source: str,
        budget_tokens: int,
        max_source_chars: int,
        allow_over_budget: bool,
    ) -> str | None:
        capped_source = hard_cap_chars(source, max_source_chars)
        full_block = self._render_context_code_block(symbol, capped_source)
        if count_tokens(self._context_content_with_candidate(lines, block=full_block)) <= budget_tokens:
            return capped_source

        search_high = min(max_source_chars, max(1, len(source)))
        best_source: str | None = None
        low = 1
        high = max(1, search_high)
        while low <= high:
            mid = (low + high) // 2
            candidate_source = hard_cap_chars(source, mid)
            candidate_block = self._render_context_code_block(symbol, candidate_source)
            if count_tokens(self._context_content_with_candidate(lines, block=candidate_block)) <= budget_tokens:
                best_source = candidate_source
                low = mid + 1
            else:
                high = mid - 1

        if best_source is not None:
            return best_source
        return capped_source if allow_over_budget else None

    def context_pack(
        self,
        *,
        task: str,
        seed_files: list[str] | None = None,
        budget_tokens: int = 4000,
        max_symbols: int = 8,
        auto_index: bool = True,
    ) -> ContextPack:
        """Build a compact, deterministic context bundle with capped entry points and code blocks."""
        if auto_index:
            self._ensure_indexed()
        context_policy = resolve_output_policy("context")
        normalized_seeds = [self._normalize_file_arg(seed) for seed in seed_files or []]
        search_query = task
        lexical_anchor_files = self._zoekt_candidate_files(search_query, max_files=max(max_symbols * 4, 24))
        context_seed_files = list(dict.fromkeys([*normalized_seeds, *lexical_anchor_files]))
        repo_map_payload = self.repo_map(seed_files=context_seed_files, budget_tokens=max(200, budget_tokens // 4))
        bounded_max_symbols = max(1, min(max_symbols, context_policy.max_related_symbols))
        symbol_hits = self.search_symbols(
            search_query,
            limit=self._context_pack_search_limit(
                max_symbols=bounded_max_symbols,
                max_symbols_per_file=context_policy.max_symbols_per_file,
            ),
            auto_index=False,
        )
        seed_symbols = self._symbols_for_files(
            context_seed_files,
            limit=max(
                bounded_max_symbols * max(1, context_policy.max_symbols_per_file),
                bounded_max_symbols,
            ),
        )
        selected = self._dedupe_symbols([*seed_symbols, *symbol_hits])
        selected = [symbol for symbol in selected if self._is_context_pack_symbol(symbol)]
        selected = self._prioritize_context_symbols(search_query, selected)
        selected = self._prune_overlapping_context_symbols(selected)
        selected = self._cap_symbols_per_file(selected, max_per_file=max(1, context_policy.max_symbols_per_file))
        selected = selected[:bounded_max_symbols]

        neighbors = self._import_neighbors(context_seed_files)
        # N9: generated/scaffolding files are dropped from "Related Symbols"
        # entirely -- they are noise once the hand-written entry points are
        # surfaced. The cap on related count is applied afterwards.
        neighbor_files = [path for path in self._context_neighbor_files(neighbors) if not is_generated_path(path)][
            : context_policy.max_related_symbols
        ]
        graph_related = self._context_graph_related_symbols(
            selected,
            query=search_query,
            limit=context_policy.max_related_symbols,
            max_symbols_per_file=max(1, context_policy.max_symbols_per_file),
        )
        selected_ids = {item.symbol_id for item in selected}
        related_symbols = [item for item in graph_related if not is_generated_path(item.file_path)]
        related_ids = {item.symbol_id for item in related_symbols} | selected_ids
        if len(related_symbols) < context_policy.max_related_symbols and neighbor_files:
            neighbor_symbol_limit = max(
                1,
                context_policy.max_related_symbols * max(1, context_policy.max_symbols_per_file),
            )
            neighbor_symbols = self._search_symbols_local(
                search_query,
                limit=neighbor_symbol_limit,
                candidate_files=set(neighbor_files),
            )
            if not neighbor_symbols:
                neighbor_symbols = self._symbols_for_files(neighbor_files, limit=neighbor_symbol_limit)
            related_seed = [
                symbol
                for symbol in neighbor_symbols
                if self._is_context_pack_symbol(symbol)
                and symbol.symbol_id not in related_ids
                and not is_generated_path(symbol.file_path)
            ]
            neighbor_related = self._prioritize_context_symbols(search_query, related_seed)
            related_symbols.extend(neighbor_related)
            related_symbols = self._prune_overlapping_context_symbols(related_symbols)
            related_symbols = self._cap_symbols_per_file(
                related_symbols, max_per_file=max(1, context_policy.max_symbols_per_file)
            )
            related_symbols = related_symbols[: context_policy.max_related_symbols]
        entry_points = [self._context_symbol_summary(symbol) for symbol in selected]
        related_summaries = [self._context_symbol_summary(symbol) for symbol in related_symbols]

        lines = ["# LemonCrow code context", f"task: {task}", ""]
        if repo_map_payload.get("outline"):
            lines.extend(["## repo_map", str(repo_map_payload["outline"]), ""])
        lines.append("## entry_points")
        if entry_points:
            lines.extend(
                [
                    f"- {item['file_path']}:{item['start_line']} — {item['qualified_name']} [{item['kind']}]"
                    for item in entry_points
                ]
            )
        else:
            lines.append("- none")
        lines.append("")
        lines.append("## related_symbols")
        if related_summaries:
            lines.extend(
                [
                    f"- {item['file_path']}:{item['start_line']} — {item['qualified_name']} [{item['kind']}]"
                    for item in related_summaries
                ]
            )
        elif context_policy.include_edges and neighbor_files:
            lines.extend([f"- {item}" for item in neighbor_files])
        else:
            lines.append("- none")
        lines.append("")
        lines.append("## code_blocks")

        packed_symbols: list[SymbolRecord] = []
        code_blocks: list[dict[str, Any]] = []
        naive_tokens = 0
        max_code_blocks = max(1, context_policy.max_code_blocks)
        code_block_candidates = self._dedupe_symbols([*selected, *graph_related])
        naive_file_tokens: dict[str, int] = {}
        for symbol in code_block_candidates:
            if len(packed_symbols) >= max_code_blocks:
                break
            file_tokens = naive_file_tokens.get(symbol.file_path)
            if file_tokens is None:
                # A concurrent autosync reindex may delete the file out from under
                # us; skip its naive-baseline contribution instead of aborting the
                # whole pack. Cache per file so multiple symbols sharing a file do
                # not re-read it.
                with contextlib.suppress(OSError):
                    file_tokens = count_tokens(self._read_file(symbol.file_path))
                naive_file_tokens[symbol.file_path] = file_tokens or 0
            naive_tokens += naive_file_tokens[symbol.file_path]
            try:
                symbol_payload = self.get_symbol(symbol_id=symbol.symbol_id, auto_index=False)
            except LookupError:
                # A concurrent reindex deleted this symbol row out from under us;
                # skip the one entry instead of aborting the whole pack.
                continue
            source_block = self._fit_context_code_block_source(
                lines=lines,
                symbol=symbol,
                source=str(symbol_payload.get("source") or ""),
                budget_tokens=budget_tokens,
                max_source_chars=context_policy.max_code_block_chars,
                allow_over_budget=not packed_symbols,
            )
            if source_block is None:
                continue
            block = self._render_context_code_block(symbol, source_block)
            lines.append(block)
            lines.append("")
            packed_symbols.append(symbol)
            code_blocks.append(
                {
                    "symbol_id": symbol.symbol_id,
                    "qualified_name": symbol.qualified_name,
                    "file_path": symbol.file_path,
                    "start_line": symbol.start_line,
                    "end_line": symbol.end_line,
                    "language": symbol.language,
                    "source": source_block,
                }
            )
        if not packed_symbols:
            lines.append("- none")
            lines.append("")

        content = "\n".join(lines).strip()
        token_count = count_tokens(content)
        tokens_saved = max(0, naive_tokens - token_count)
        emit_product_local(
            "code_context_retrieved",
            repo_id=self.repo_id,
            operation="context_pack",
            result_count=len(packed_symbols),
        )
        return ContextPack(
            task=task,
            budget_tokens=budget_tokens,
            token_count=token_count,
            tokens_saved_vs_full_files=tokens_saved,
            symbols=packed_symbols,
            entry_points=entry_points,
            related_symbols=related_summaries,
            code_blocks=code_blocks,
            repo_map=str(repo_map_payload.get("outline", "")),
            import_neighbors=neighbor_files,
            content=content,
            telemetry={
                "repo_id": self.repo_id,
                "selected_symbols": len(packed_symbols),
                "entry_points": len(entry_points),
                "related_symbols": len(related_summaries),
                "call_graph_related_symbols": len(graph_related),
                "token_budget_fit": token_count <= budget_tokens,
            },
        )

    def search_text(
        self,
        query: str,
        *,
        path: str = ".",
        limit: int = 50,
        ignore_case: bool = False,
    ) -> list[TextMatch]:
        """Literal text search over the warmed line index, with rg as legacy fallback."""
        search_path = self._resolve_inside_repo(path)
        indexed = self._search_text_index(query, search_path=search_path, limit=limit, ignore_case=ignore_case)
        if indexed:
            return indexed
        if shutil.which("rg") is not None:
            args = [
                "rg",
                "--fixed-strings",
                "--line-number",
                "--column",
                "--no-heading",
                "--color",
                "never",
                "--max-count",
                str(limit),
            ]
            if ignore_case:
                args.append("--ignore-case")
            args.extend([query, str(search_path)])
            proc = subprocess.run(args, check=False, capture_output=True, text=True)
            if proc.returncode not in {0, 1}:
                raise RuntimeError(proc.stderr.strip() or "ripgrep failed")
            return self._parse_rg_output(proc.stdout, limit=limit)
        return self._python_text_search(query, search_path, limit=limit, ignore_case=ignore_case)

    def _search_text_index(
        self,
        query: str,
        *,
        search_path: Path,
        limit: int,
        ignore_case: bool,
    ) -> list[TextMatch]:
        normalized = query.strip()
        if not normalized:
            return []
        fts_query = _safe_fts_query(normalized)
        rel = _safe_relpath(self.repo_root, search_path)
        path_clause = ""
        path_params: list[Any] = []
        if search_path != self.repo_root:
            if search_path.is_file():
                path_clause = " AND file_path = ?"
                path_params.append(rel)
            else:
                path_clause = " AND (file_path = ? OR file_path LIKE ?)"
                path_params.extend([rel, f"{rel.rstrip('/')}/%"])
        query_lower = normalized.lower()
        rows: list[sqlite3.Row] = []
        with self._connect() as conn:
            self._init_schema(conn)
            if fts_query:
                rows = conn.execute(
                    f"""
                    SELECT file_path, line, text
                    FROM file_line_fts
                    WHERE file_line_fts MATCH ? AND repo_id = ?{path_clause}
                    ORDER BY file_path, line
                    LIMIT ?
                    """,
                    tuple([fts_query, self.repo_id, *path_params, max(limit * 8, 80)]),
                ).fetchall()
            if not rows:
                like = f"%{query_lower if ignore_case else normalized}%"
                text_expr = "lower(text)" if ignore_case else "text"
                rows = conn.execute(
                    f"""
                    SELECT file_path, line, text
                    FROM file_line_fts
                    WHERE repo_id = ?{path_clause} AND {text_expr} LIKE ?
                    ORDER BY file_path, line
                    LIMIT ?
                    """,
                    tuple([self.repo_id, *path_params, like, max(limit * 8, 80)]),
                ).fetchall()
        matches: list[TextMatch] = []
        for row in rows:
            text = str(row["text"])
            haystack = text.lower() if ignore_case else text
            needle = query_lower if ignore_case else normalized
            index = haystack.find(needle)
            if index < 0:
                continue
            matches.append(
                TextMatch(
                    file_path=str(row["file_path"]),
                    line=int(row["line"]),
                    column=index + 1,
                    text=text,
                )
            )
            if len(matches) >= limit:
                break
        return matches

    def _should_use_text_substring_search(
        self,
        query: str,
        *,
        mode: SearchMode,
        scope: Literal["repo", "external", "deleted"],
        kind: str | None,
        language: str | None,
        file_glob: str | None,
        provenance_filter: str | None,
    ) -> str | None:
        """Resolve the literal string to substring-search on for *query*, or
        None to skip substring search in favor of semantic/symbol ranking.

        Bare-token fast path only: a lowercase token with no whitespace,
        underscore or dot (4-40 chars) that is not itself an indexed symbol.
        There's exactly one token to decide on here, so a cheap pre-search
        prediction is safe and unambiguous -- unlike a multi-word query, there
        is nothing else it could mean.

        Multi-word queries are no longer predicted here. This used to have a
        second branch that recognized identifier-shaped tokens (snake_case /
        camelCase) embedded in a longer query -- e.g. "clone entity strip
        technical fields _finalEncrypted" -- but that was a shape allowlist,
        and the next shape that came along missed it: a kebab-case HTTP header
        literal ("x-flipt-accept-server-version") embedded as a Go *string
        value*, not a declared symbol at all. Rather than add a third shape
        special case, multi-word queries now fall through to symbol/semantic
        search first; ``tool_search`` reacts to an actual empty result via
        ``_existence_gated_fallback_token``, which probes raw query tokens for
        verbatim existence (symbol OR text content) regardless of shape.
        """
        normalized = query.strip()
        if scope != "repo" or mode != "lexical" or kind is not None or provenance_filter is not None:
            return None
        if any(char.isspace() for char in normalized):
            return None
        if not (4 <= len(normalized) <= 40):
            return None
        if "_" in normalized or "." in normalized:
            return None
        if normalized != normalized.lower():
            return None
        if self._has_exact_repo_symbol(normalized, kind=kind, language=language, file_glob=file_glob):
            return None
        return normalized

    def _has_exact_repo_symbol(
        self,
        query: str,
        *,
        kind: str | None,
        language: str | None,
        file_glob: str | None,
    ) -> bool:
        clauses = [
            "repo_id = ?",
            "(symbol_name = ? OR qualified_name = ? OR lower(symbol_name) = ? OR lower(qualified_name) = ?)",
        ]
        params: list[Any] = [self.repo_id, query, query, query.lower(), query.lower()]
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if language:
            clauses.append("language = ?")
            params.append(language)
        with self._connect() as conn:
            self._init_schema(conn)
            rows = conn.execute(
                f"""
                SELECT file_path
                FROM symbols
                WHERE {" AND ".join(clauses)}
                LIMIT 20
                """,
                tuple(params),
            ).fetchall()
        if file_glob:
            return any(_matches_file_glob(str(row["file_path"]), file_glob) for row in rows)
        return bool(rows)

    def _existence_gated_fallback_token(
        self,
        query: str,
        *,
        kind: str | None,
        language: str | None,
        file_glob: str | None,
    ) -> str | None:
        """After an ``intent="auto"`` symbol/semantic search comes back empty,
        find a token embedded in *query* that exists verbatim somewhere in the
        repo -- as a declared symbol OR as raw line content -- and return it so
        the caller can substring-search on it instead.

        This is the general replacement for the old shape-based multi-word
        probe: it never inspects a token's shape (snake_case, camelCase,
        kebab-case, quoted, ALL_CAPS, a dotted path, ...), only whether it
        exists. Existence, not shape, is what makes this durable against token
        shapes not yet special-cased -- a query embedding a literal this
        doesn't recognize by pattern still resolves, because the gate is "is
        this a real substring", not "does this look like an identifier". A
        token that doesn't exist anywhere changes nothing, so ordinary
        natural-language queries with no embedded literal behave exactly as
        before -- this only ever runs after symbol/semantic search already
        found nothing, so it never costs an otherwise-successful query.
        """
        search_path = "src/lemoncrow" if (self.repo_root / "src" / "lemoncrow").is_dir() else "."
        probe_limit = 20 if file_glob else 1
        for token in _fallback_probe_tokens(query):
            if self._has_exact_repo_symbol(token, kind=kind, language=language, file_glob=file_glob):
                return token
            matches = self.search_text(token, path=search_path, limit=probe_limit, ignore_case=True)
            if file_glob:
                matches = [m for m in matches if _matches_file_glob(m.file_path, file_glob)]
            if matches:
                return token
        return None

    def _substring_symbol_hits(
        self,
        query_lower: str,
        *,
        limit: int,
        file_glob: str | None,
    ) -> list[SymbolRecord]:
        like_pattern = f"%{query_lower}%"
        with self._connect() as conn:
            self._init_schema(conn)
            rows = conn.execute(
                """
                SELECT *, NULL AS score
                FROM symbols
                WHERE repo_id = ? AND (
                    lower(symbol_name) LIKE ?
                    OR lower(qualified_name) LIKE ?
                    OR lower(signature) LIKE ?
                )
                ORDER BY
                    CASE WHEN kind IN ('class', 'method', 'function') THEN 0 ELSE 1 END,
                    CASE WHEN lower(symbol_name) LIKE ? OR lower(qualified_name) LIKE ? THEN 0 ELSE 1 END,
                    length(symbol_name),
                    file_path,
                    start_line
                LIMIT ?
                """,
                (
                    self.repo_id,
                    like_pattern,
                    like_pattern,
                    like_pattern,
                    f"{query_lower}%",
                    f"{query_lower}%",
                    max(limit * 12, 120),
                ),
            ).fetchall()
        hits = [_row_to_symbol(row) for row in rows]
        if file_glob:
            hits = [hit for hit in hits if _matches_file_glob(hit.file_path, file_glob)]
        return hits[:limit]

    def _tool_text_substring_search(
        self,
        query: str,
        *,
        limit: int,
        file_glob: str | None,
        budget_tokens: int,
        since_ts: int | None = None,
        touched_by: str | None = None,
    ) -> dict[str, Any]:
        search_path = "src/lemoncrow" if (self.repo_root / "src" / "lemoncrow").is_dir() else "."
        query_lower = query.lower()
        symbol_hits = self._substring_symbol_hits(query_lower, limit=max(limit * 40, 200), file_glob=file_glob)
        ranked_symbol_hits = sorted(
            (
                item
                for item in symbol_hits
                if query_lower in item.symbol_name.lower() or query_lower in item.qualified_name.lower()
            ),
            key=lambda item: self._text_substring_symbol_score(query_lower, item),
            reverse=True,
        )
        symbol_items = [item.model_dump(mode="json", exclude_none=True) for item in ranked_symbol_hits[:limit]]
        raw_limit = max(limit * 50, 500)
        matches = self.search_text(query, path=search_path, limit=raw_limit, ignore_case=True)
        if file_glob:
            matches = [match for match in matches if _matches_file_glob(match.file_path, file_glob)]
        ranked = sorted(
            matches,
            key=lambda match: self._text_substring_score(query, match),
            reverse=True,
        )
        symbol_paths = {str(item.get("file_path") or "") for item in symbol_items}
        text_items = [
            item
            for item in (self._text_match_search_item(query, match) for match in ranked[:limit])
            if str(item.get("file_path") or "") not in symbol_paths
        ]
        items = self._dedupe_search_items(symbol_items + text_items)
        if since_ts is not None or touched_by is not None:
            changed_files = self._deleted_history_adapter().changed_files(
                since_ts=since_ts,
                touched_by=touched_by,
            )
            items = [item for item in items if str(item.get("file_path") or "") in changed_files]
        payload = self._pack_items_payload(
            items,
            budget_tokens=budget_tokens,
            essential_keys=_SEARCH_ESSENTIAL_KEYS,
            optional_keys_in_drop_order=["snippet", "score", "repo_id"],
            extra_payload={
                "mode": "lexical",
                "snippet": "none",
                "provenance": _LOCAL_PROVENANCE,
                "text_search": True,
            },
        )
        return payload

    def _text_substring_score(self, query: str, match: TextMatch) -> tuple[int, int, int, int]:
        lowered_text = match.text.lower()
        lowered_path = match.file_path.lower()
        query_lower = query.lower()
        definition = int(bool(re.search(rf"\b(def|class)\s+[A-Za-z0-9_]*{re.escape(query_lower)}", lowered_text)))
        symbolish = int(bool(re.search(rf"[A-Za-z0-9_]*{re.escape(query_lower)}", lowered_text)))
        path_hit = int(query_lower in lowered_path)
        return (definition, symbolish, path_hit, -len(match.file_path))

    def _text_substring_symbol_score(self, query_lower: str, symbol: SymbolRecord) -> tuple[int, int, int, int, int]:
        symbol_name_lower = symbol.symbol_name.lower()
        qualified_name_lower = symbol.qualified_name.lower()
        preferred_kind = int(symbol.kind in {"class", "method", "function"})
        startswith = int(symbol_name_lower.startswith(query_lower) or qualified_name_lower.startswith(query_lower))
        bare_startswith = int(symbol_name_lower.lstrip("_").startswith(query_lower))
        path_hit = int(query_lower in symbol.file_path.lower())
        return (preferred_kind, startswith, bare_startswith, path_hit, -len(symbol.symbol_name))

    def _text_match_search_item(self, query: str, match: TextMatch) -> dict[str, Any]:
        name = self._text_match_name(query, match.text)
        return {
            "symbol_id": f"text:{match.file_path}:{match.line}:{match.column}",
            "symbol_name": name,
            "qualified_name": name,
            "file_path": match.file_path,
            "kind": "text_match",
            "start_line": match.line,
            "signature": match.text.strip()[:240],
            "provenance": _LOCAL_PROVENANCE,
            "score": 1.0,
        }

    def _text_match_name(self, query: str, text: str) -> str:
        match = re.search(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", text)
        if match:
            return match.group(1)
        token = re.search(rf"([A-Za-z_][A-Za-z0-9_]*{re.escape(query)}[A-Za-z0-9_]*)", text)
        if token:
            return token.group(1)
        return query

    _zoekt_ready_waited: bool = False

    def _zoekt_wait_ready_once(self, supervisor: Any) -> None:
        """Bounded first-use wait for the zoekt webserver (once per engine).

        The hot query path never blocks on startup, so queries issued while
        index shards are still loading silently lose the entire Zoekt channel
        -- a recall hit, not just latency. A bounded wait exactly once per
        engine converts that silent quality loss into a small first-query
        delay; after the first attempt (ready or not) searches never block
        again and keep the existing degrade-to-empty behaviour.
        Budget via LEMONCROW_ZOEKT_READY_TIMEOUT_S (default 5s, 0 disables).
        """
        if self._zoekt_ready_waited:
            return
        self._zoekt_ready_waited = True
        try:
            timeout = float(os.environ.get("LEMONCROW_ZOEKT_READY_TIMEOUT_S", "5.0"))
        except ValueError:
            timeout = 5.0
        if timeout <= 0:
            return
        with contextlib.suppress(Exception):
            supervisor.server.wait_until_searchable(timeout)

    def _zoekt_candidate_files(
        self,
        query: str,
        *,
        path: str = ".",
        max_files: int = 40,
    ) -> list[str]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        try:
            from lemoncrow.infra.code_intel.zoekt.adapter import get_zoekt_supervisor
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return []
        with contextlib.suppress(Exception):
            search_path = self._resolve_inside_repo(path)
            supervisor = get_zoekt_supervisor(self.repo_root)
            if not supervisor.should_route(search_path):
                return []
            self._zoekt_wait_ready_once(supervisor)
            result = supervisor.search(
                query=normalized_query,
                search_path=search_path,
                max_files=max(1, min(max_files, 200)),
                max_chars_per_file=800,
                include_outline=False,
                _include_index_age=False,
            )
            # ordered dedup: zoekt returns files in descending score order;
            # preserve that order so callers can prefer high-signal files.
            seen: dict[str, None] = {}
            for match in result.matches:
                raw_path = Path(match.path)
                resolved = raw_path if raw_path.is_absolute() else (self.repo_root / raw_path)
                with contextlib.suppress(ValueError):
                    rel = _safe_relpath(self.repo_root, resolved.resolve())
                    seen[rel] = None
            return list(seen)
        return []

    def _semantic_candidate_files(self, query: str, *, max_files: int = 40) -> dict[str, float]:
        """Files whose symbols are the nearest semantic (embedding) neighbours of the
        query -- an additive recall channel for the explore fusion, mirroring
        _zoekt_candidate_files.  Fuses ONLY when an embedder is configured and
        available: the embedder is the gated provider, so with none configured this
        is a silent no-op (never an error or a "pro-gated" notice).  Graceful --
        returns an empty dict when embeddings are not indexed or the search fails.
        Opt out with LEMONCROW_EXPLORE_SEMANTIC=0.

        Returns ``{file_path: best_cosine}`` (the max cosine over the file's
        symbols). The caller uses the keys as recall anchors AND the scores as a
        ranking signal, so a file the embedder ranked highest is promoted rather
        than scored ~0 on lexical/centrality.
        """
        normalized_query = query.strip()
        if os.environ.get("LEMONCROW_EXPLORE_SEMANTIC", "1") == "0":
            return {}
        if not normalized_query or not getattr(self._semantic_ranker, "available", False):
            return {}
        # Cosine floor: drop clearly-dissimilar neighbours that would be noise.
        # Default 0.35, not 0.55: BGE-code is an asymmetric (query-prefix vs
        # document) model whose correct matches land at ~0.4-0.55 cosine, so a
        # 0.55 floor discarded nearly every true semantic anchor before fusion.
        min_score = float(os.environ.get("LEMONCROW_SEMANTIC_MIN_SCORE", "0.35"))
        files: dict[str, float] = {}
        with contextlib.suppress(Exception):
            for symbol in self._search_symbols_semantic_local(normalized_query, limit=max(8, max_files)):
                if (symbol.score or 0.0) < min_score:
                    break  # results are score-sorted; no point continuing
                fp = symbol.file_path
                if fp not in files:  # first occurrence is the file's best (score-desc order)
                    files[fp] = float(symbol.score or 0.0)
                if len(files) >= max_files:
                    break
        return files

    def _zoekt_text_matches(
        self,
        query: str,
        *,
        limit: int,
        file_glob: str | None = None,
        path: str = ".",
    ) -> list[TextMatch]:
        candidate_files = self._zoekt_candidate_files(query, path=path, max_files=max(1, min(limit, 200)))
        if not candidate_files:
            return []
        matches: list[TextMatch] = []
        lower_query = query.lower()
        for rel in sorted(candidate_files):
            if file_glob and not _matches_file_glob(rel, file_glob):
                continue
            with contextlib.suppress(OSError):
                lines = (self.repo_root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
                for line_no, text in enumerate(lines, start=1):
                    column = text.lower().find(lower_query)
                    if column < 0:
                        continue
                    matches.append(TextMatch(file_path=rel, line=line_no, column=column + 1, text=text))
                    if len(matches) >= limit:
                        return matches
        return matches

    def find_references(
        self,
        query: str | None = None,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        symbol_name: str | None = None,
        file_path: str | None = None,
        kind: str | None = None,
        language: str | None = None,
        file_glob: str | None = None,
        group_by: Literal["file", "caller", "none"] = "file",
        snippet_lines: int = 3,
        limit: int = 20,
        auto_index: bool = True,
        budget_tokens: int = 4000,
    ) -> dict[str, Any]:
        effective_budget_tokens = self._effective_budget_tokens("relation", budget_tokens)
        if auto_index:
            self._ensure_indexed()
        resolved = self._resolve_symbol_targets(
            operation_name="usages",
            query=query,
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            symbol_name=symbol_name,
            file_path=file_path,
            kind=kind,
            language=language,
            file_glob=file_glob,
        )
        if resolved.get("error"):
            return self._pack_single_payload(
                resolved,
                budget_tokens=effective_budget_tokens,
                essential_keys=["error", "message", "matches", "cache_hit", "provenance"],
                optional_keys_in_drop_order=["provenance_breakdown"],
            )
        targets = cast(list[dict[str, Any]], resolved["targets"])
        primary_target = targets[0]
        relation_policy = resolve_output_policy("relation")
        # Bound the intermediate reference set at the source so a very common
        # identifier cannot inflate the pre-sort collection without limit. The
        # final policy/limit caps below still apply as a backstop; this ceiling
        # keeps headroom for dedup + sort while truncating each provider's
        # contribution as it is collected.
        collection_ceiling = max(
            limit,
            relation_policy.max_related_symbols if relation_policy.max_related_symbols > 0 else 0,
        )
        collection_ceiling = max(collection_ceiling * 4, 100)
        references: list[UsageReference] = []
        ceiling_truncated = False
        for target in targets:
            local_refs = self.intel_store.find_references(
                symbol_id=str(target["symbol_id"]),
                qualified_name=str(target["qualified_name"]),
                file_path=str(target["file_path"]),
                symbol_name=str(target["symbol_name"]),
            )
            cross_lang_refs = self._cross_lang_usage_references(target)
            references.extend(local_refs[: max(collection_ceiling - len(references), 0)])
            references.extend(cross_lang_refs[: max(collection_ceiling - len(references), 0)])
            if len(references) >= collection_ceiling:
                # Source-level ceiling fired: the pre-sort set was capped before
                # the downstream policy cap could weigh in, so the result is
                # genuinely incomplete regardless of relation_policy.
                ceiling_truncated = True
                break
        ordered_references = sorted(
            references,
            key=lambda item: (
                item.file_path,
                item.line,
                item.column,
                item.end_line,
                item.end_column,
                item.provenance,
            ),
        )
        items = self._dedupe_usage_items(
            [self._usage_item(reference, snippet_lines=snippet_lines) for reference in ordered_references]
        )
        if not items:
            fallback_query = symbol_name or query or qualified_name
            if fallback_query:
                fallback_provenance = "zoekt_text"
                text_hits = self._zoekt_text_matches(
                    fallback_query,
                    limit=collection_ceiling,
                    file_glob=file_glob,
                )
                if not text_hits:
                    fallback_provenance = "text"
                    text_hits = self.search_text(fallback_query, path=".", limit=collection_ceiling, ignore_case=False)
                items = [
                    {
                        "file_path": match.file_path,
                        "line": match.line,
                        "column": match.column,
                        "end_line": match.line,
                        "end_column": match.column + len(fallback_query),
                        "snippet": match.text,
                        "caller": None,
                        "edge_kind": "text_match",
                        "confidence": 0.25,
                        "provenance": fallback_provenance,
                    }
                    for match in text_hits
                ]
                items = self._dedupe_usage_items(items)
        if file_glob:
            items = [item for item in items if _matches_file_glob(str(item["file_path"]), file_glob)]
        if not relation_policy.include_snippet:
            for item in items:
                item.pop("snippet", None)
        truncated_by_policy = False
        if relation_policy.max_related_symbols > 0 and len(items) > relation_policy.max_related_symbols:
            items = items[: relation_policy.max_related_symbols]
            truncated_by_policy = True
        full_payload = self._build_usages_payload(
            target=primary_target,
            items=items,
            group_by=group_by,
            truncated=truncated_by_policy or ceiling_truncated,
            ambiguity=cast(dict[str, Any] | None, resolved.get("ambiguity")),
        )
        full_total_tokens = self._compute_total_tokens({**full_payload, "tokens_saved": 0})

        def build_payload(packed_items: list[dict[str, Any]]) -> dict[str, Any]:
            return self._finalize_packed_payload(
                self._build_usages_payload(
                    target=primary_target,
                    items=packed_items,
                    group_by=group_by,
                    truncated=truncated_by_policy or ceiling_truncated or len(packed_items) < len(items),
                    ambiguity=cast(dict[str, Any] | None, resolved.get("ambiguity")),
                ),
                full_total_tokens=full_total_tokens,
            )

        return self._fit_items_to_budget(
            items[:limit],
            budget_tokens=effective_budget_tokens,
            essential_keys=_USAGES_ESSENTIAL_KEYS,
            optional_keys_in_drop_order=_USAGES_OPTIONAL_KEYS,
            build_payload=build_payload,
            enforce_protected_top_rank=False,
        )

    def _cross_lang_usage_references(self, target: dict[str, Any]) -> list[UsageReference]:
        symbol_id = str(target.get("symbol_id") or "")
        symbol_name = str(target.get("symbol_name") or "")
        if not symbol_id:
            return []
        refs: list[UsageReference] = []
        for edge in self._cross_lang_store().query_by_target_symbol(
            tgt_symbol_id=symbol_id, tgt_symbol_name=symbol_name
        ):
            refs.append(
                UsageReference(
                    file_path=edge.src_file_path,
                    line=edge.src_line,
                    column=1,
                    end_line=edge.src_line,
                    end_column=1,
                    caller=edge.src_qualified_name,
                    provenance="cross_lang",
                    edge_kind=edge.edge_kind,
                    confidence=edge.confidence,
                )
            )
        return refs

    def _tool_call_graph(
        self,
        direction: CallGraphDirection,
        *,
        query: str | None = None,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        symbol_name: str | None = None,
        file_path: str | None = None,
        kind: str | None = None,
        language: str | None = None,
        depth: int = 1,
        limit: int = 20,
        snapshot: bool = False,
        budget_tokens: int = 4000,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        effective_budget_tokens = self._effective_budget_tokens(direction, budget_tokens)
        if auto_index:
            self._ensure_indexed()
        self._sync_symbol_intel()
        normalized_file_path = self._normalize_file_arg(file_path) if file_path else None
        bounded_depth = max(1, depth)
        cache_args = {
            "direction": direction,
            "query": query,
            "symbol_id": symbol_id,
            "qualified_name": qualified_name,
            "symbol_name": symbol_name,
            "file_path": normalized_file_path,
            "kind": kind,
            "language": language,
            "depth": bounded_depth,
            "limit": limit,
            "snapshot": snapshot,
            "budget_tokens": effective_budget_tokens,
        }
        hit, cached = self._cache_get(f"code.{direction}", cache_args)
        if hit and cached is not None:
            return self._mark_cache_hit(cached)
        resolved = self._resolve_symbol_targets(
            operation_name=direction,
            query=query,
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            symbol_name=symbol_name,
            file_path=normalized_file_path,
            kind=kind,
            language=language,
            file_glob=None,
        )
        if resolved.get("error"):
            return self._pack_single_payload(
                resolved,
                budget_tokens=effective_budget_tokens,
                essential_keys=["error", "message", "matches", "cache_hit", "provenance"],
                optional_keys_in_drop_order=["provenance_breakdown"],
            )
        targets = cast(list[dict[str, Any]], resolved["targets"])
        primary_target = targets[0]
        lookup = self.intel_store.find_callers if direction == "callers" else self.intel_store.find_callees
        traversals: list[CallGraphTraversalResult] = []
        for target in targets:
            traversal = traverse_call_graph(
                target,
                direction=direction,
                depth=bounded_depth,
                limit=limit,
                snapshot=snapshot,
                lookup_neighbors=lambda current_symbol_id: lookup(symbol_id=current_symbol_id),
            )
            if traversal.data_status == "unavailable" and direction == "callers":
                fallback = self._fallback_callers_from_references(
                    target=target,
                    limit=limit,
                )
                if fallback.data_status == "available":
                    traversal = fallback
            traversals.append(traversal)
        if len(traversals) == 1:
            traversal = traversals[0]
        else:
            nodes_by_identity: dict[tuple[str, str, int, int, str], CallGraphNode] = {}
            edges_by_key: dict[tuple[str, str, int], CallGraphEdge] = {}
            merged_truncated = False
            status_rank = {"unavailable": 0, "empty": 1, "available": 2}
            merged_status = "unavailable"
            for current in traversals:
                merged_truncated = merged_truncated or current.truncated
                if status_rank[current.data_status] > status_rank[merged_status]:
                    merged_status = current.data_status
                for node in current.nodes:
                    node_key = (
                        node.symbol_id,
                        node.file_path,
                        node.start_line,
                        node.end_line,
                        node.qualified_name,
                    )
                    nodes_by_identity[node_key] = node
                for edge in current.edges:
                    key = (edge.caller_symbol_id, edge.callee_symbol_id, edge.depth)
                    edges_by_key[key] = edge
            merged_nodes = sorted(
                nodes_by_identity.values(),
                key=lambda item: (item.file_path, item.start_line, item.symbol_id),
            )
            merged_edges = sorted(
                edges_by_key.values(),
                key=lambda item: (item.depth, item.caller_symbol_id, item.callee_symbol_id),
            )
            if merged_edges:
                merged_status = "available"
            merged_message = (
                "routed call edge data is unavailable"
                if merged_status == "unavailable"
                else "no related call edges were found" if merged_status == "empty" else None
            )
            merged_snapshot = None
            if snapshot:
                merged_snapshot = {
                    "direction": direction,
                    "depth": bounded_depth,
                    "target_symbol_id": str(primary_target["symbol_id"]),
                    "target_count": len(targets),
                    "node_count": len(merged_nodes),
                    "edge_count": len(merged_edges),
                }
            traversal = CallGraphTraversalResult(
                nodes=merged_nodes,
                edges=merged_edges,
                truncated=merged_truncated,
                data_status=cast(Any, merged_status),
                message=merged_message,
                snapshot=merged_snapshot,
            )
        payload = build_call_graph_payload(
            primary_target,
            direction=direction,
            depth=bounded_depth,
            result=traversal,
        )
        ambiguity = cast(dict[str, Any] | None, resolved.get("ambiguity"))
        if ambiguity is not None:
            payload["ambiguity"] = ambiguity
        relation_policy = resolve_output_policy("relation")
        if relation_policy.max_related_symbols > 0:
            # Respect the caller's explicit limit when it exceeds the compact-policy
            # default (12). Without this, passing limit=50 still silently truncates
            # to 12 because the compact policy runs after the traversal.
            max_related = max(limit, relation_policy.max_related_symbols)
            related_before = len(cast(list[dict[str, Any]], payload.get("related", [])))
            edges_before = len(cast(list[dict[str, Any]], payload.get("edges", [])))
            payload["related"] = cast(list[dict[str, Any]], payload.get("related", []))[:max_related]
            payload["edges"] = cast(list[dict[str, Any]], payload.get("edges", []))[:max_related]
            payload["related_count"] = len(cast(list[dict[str, Any]], payload.get("related", [])))
            payload["edge_count"] = len(cast(list[dict[str, Any]], payload.get("edges", [])))
            payload["truncated"] = (
                bool(payload.get("truncated", False)) or related_before > max_related or edges_before > max_related
            )
        if not relation_policy.include_edges:
            payload["edges"] = []
            payload["edge_count"] = 0
        payload["provenance"] = str(primary_target.get("provenance") or _LOCAL_PROVENANCE)
        packed = self._pack_single_payload(
            payload,
            budget_tokens=effective_budget_tokens,
            essential_keys=_CALL_GRAPH_ESSENTIAL_KEYS,
            optional_keys_in_drop_order=_CALL_GRAPH_OPTIONAL_KEYS,
        )
        if "related" not in packed and "related" in payload:
            packed["related"] = payload["related"]
            packed["related_count"] = payload.get("related_count", len(cast(list[Any], payload["related"])))
        if "edges" not in packed and "edges" in payload:
            packed["edges"] = payload["edges"]
            packed["edge_count"] = payload.get("edge_count", len(cast(list[Any], payload["edges"])))
        # Re-apply shortening to restored fields (they bypassed _finalize_packed_payload shortening)
        packed = apply_field_name_shortening(packed)
        if "data_status" not in packed and "data_status" in payload:
            packed["data_status"] = payload["data_status"]
        if "error" not in packed:
            self._cache_set(f"code.{direction}", cache_args, packed)
        return packed

    def _neighborhood(
        self,
        relation: Literal["self", "callers", "callees", "refs"],
        *,
        query: str | None = None,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        symbol_name: str | None = None,
        file_path: str | None = None,
        line: int | None = None,
        kind: str | None = None,
        language: str | None = None,
        file_glob: str | None = None,
        depth: int = 1,
        limit: int = 20,
        group_by: Literal["file", "caller", "none"] = "file",
        snippet_lines: int = 3,
        snapshot: bool = False,
        budget_tokens: int = 4000,
        auto_index: bool = True,
    ) -> dict[str, Any]:
        """Unified symbol-graph access: one resolve+project entry over the shared
        code index. ``relation`` selects the projection:

        * ``self``    -- the symbol's own definition (``depth``/``group_by``/
          ``snippet_lines`` ignored).
        * ``callers`` -- inbound call edges (transitive via ``depth``).
        * ``callees`` -- outbound call edges (transitive via ``depth``).
        * ``refs``    -- all references/usages (flat; ``depth`` ignored).

        ``node``/``callers``/``callees``/``usages`` and ``explore``'s relationship
        pass all funnel through here, so symbol-graph access has a single code
        path. Each branch still delegates to its existing engine method, so
        payloads are unchanged -- this is the seam the projections collapse into.
        """
        if relation == "self":
            return self.tool_symbol(
                symbol_id=symbol_id,
                qualified_name=qualified_name,
                symbol_name=symbol_name,
                file_path=file_path,
                line=line,
                budget_tokens=budget_tokens,
                auto_index=auto_index,
            )
        if relation in ("callers", "callees"):
            return self._tool_call_graph(
                relation,
                query=query,
                symbol_id=symbol_id,
                qualified_name=qualified_name,
                symbol_name=symbol_name,
                file_path=file_path,
                kind=kind,
                language=language,
                depth=depth,
                limit=limit,
                snapshot=snapshot,
                budget_tokens=budget_tokens,
                auto_index=auto_index,
            )
        if relation == "refs":
            return self.find_references(
                query=query,
                symbol_id=symbol_id,
                qualified_name=qualified_name,
                symbol_name=symbol_name,
                file_path=file_path,
                kind=kind,
                language=language,
                file_glob=file_glob,
                group_by=group_by,
                snippet_lines=snippet_lines,
                limit=limit,
                auto_index=auto_index,
                budget_tokens=budget_tokens,
            )
        raise ValueError(f"unknown neighborhood relation: {relation!r}")

    @property
    def intel_db_path(self) -> Path:
        return self.db_path.parent / "intel.sqlite"

    @property
    def vectors_db_path(self) -> Path:
        return self.db_path.parent / "vectors.sqlite"

    @property
    def fts_db_path(self) -> Path:
        return self.db_path.parent / "fts.sqlite"

    def _secondary_db_identity(self) -> tuple[tuple[int, int], ...]:
        """(dev, ino) fingerprint of the secondary DB files ((-1,-1) = missing)."""
        out: list[tuple[int, int]] = []
        for p in (self.intel_db_path, self.vectors_db_path, self.fts_db_path):
            try:
                st = os.stat(p)
                out.append((st.st_dev, st.st_ino))
            except OSError:
                out.append((-1, -1))
        return tuple(out)

    def _init_secondary_schemas(self) -> None:
        """Create schema in secondary DBs using dedicated connections.

        Each secondary DB is initialised separately (no ATTACH needed here)
        so table names need no schema prefix.

        Runs once per file identity: _connect() calls this on EVERY main
        connection open (per tool call), and re-running three connects plus
        full CREATE-IF-NOT-EXISTS executescripts each time was pure overhead
        (~ms per call). The (dev, ino) key re-triggers DDL when any secondary
        file is replaced or first created.

        Serialized: concurrent `PRAGMA journal_mode = WAL` on a fresh
        DELETE-mode DB fails immediately with SQLITE_BUSY (the WAL switch does
        not invoke the busy handler), and several threads can arrive here on
        first connect (main tool call, autosync worker, file watcher).
        """
        key = self._secondary_db_identity()
        if key == self._secondary_schemas_key and (-1, -1) not in key:
            return
        with self._secondary_schema_lock:
            key = self._secondary_db_identity()
            if key == self._secondary_schemas_key and (-1, -1) not in key:
                return
            self._init_secondary_schemas_locked()

    def _init_secondary_schemas_locked(self) -> None:
        # --- intel.sqlite ---
        self.intel_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.intel_db_path, timeout=30.0) as ic:
            ic.execute("PRAGMA journal_mode = WAL")
            # Migration: call_edges.snippet was written on every index but
            # never read anywhere (verified: no SELECT in this codebase
            # touches it) -- pure dead weight (~5MB/repo here). Drop it.
            # (An earlier revision of this migration also normalized
            # references.snippet into a shared dedup table -- measured net
            # LOSS: snippets average ~48 bytes, so the UNIQUE-on-text index a
            # dedup table needs plus the FK index on references.snippet_id
            # cost more than the duplicate text saved. references.snippet
            # stays inline TEXT.) CREATE TABLE IF NOT EXISTS cannot reshape an
            # existing table, so detect the old column shape and rebuild empty
            # -- same trick used elsewhere in this file for schema migrations:
            # clear the main DB's `files` table so autosync/code_warm
            # repopulates via a normal reindex, never a blocking rebuild on
            # the request path.
            _ref_cols = {row[1] for row in ic.execute('PRAGMA table_info("references")').fetchall()}
            _edge_cols = {row[1] for row in ic.execute("PRAGMA table_info(call_edges)").fetchall()}
            if _ref_cols and ("snippet" not in _ref_cols or "snippet" in _edge_cols):
                ic.executescript('DROP TABLE IF EXISTS "references"; DROP TABLE IF EXISTS call_edges;')
                with contextlib.suppress(sqlite3.OperationalError, sqlite3.DatabaseError):
                    with sqlite3.connect(self.db_path, timeout=30.0) as mc:
                        mc.execute("DELETE FROM files")
            ic.executescript("""
                CREATE TABLE IF NOT EXISTS \"references\" (
                    repo_id TEXT NOT NULL,
                    symbol_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line INTEGER NOT NULL,
                    column INTEGER NOT NULL,
                    end_column INTEGER NOT NULL,
                    enclosing_symbol_name TEXT,
                    enclosing_qualified_name TEXT,
                    snippet TEXT NOT NULL,
                    UNIQUE(repo_id, symbol_name, file_path, line, column, enclosing_qualified_name)
                );
                CREATE TABLE IF NOT EXISTS call_edges (
                    repo_id TEXT NOT NULL,
                    caller_symbol_name TEXT NOT NULL,
                    caller_qualified_name TEXT NOT NULL,
                    caller_file_path TEXT NOT NULL,
                    caller_start_line INTEGER NOT NULL,
                    caller_end_line INTEGER NOT NULL,
                    callee_name TEXT NOT NULL,
                    callee_short_name TEXT NOT NULL DEFAULT '',
                    call_line INTEGER NOT NULL,
                    call_column INTEGER NOT NULL,
                    UNIQUE(repo_id, caller_qualified_name, caller_file_path, call_line, call_column, callee_name)
                );
                CREATE TABLE IF NOT EXISTS centrality_map (
                    repo_id        TEXT NOT NULL,
                    name_key       TEXT NOT NULL,
                    score          REAL NOT NULL,
                    index_version  INTEGER NOT NULL,
                    PRIMARY KEY (repo_id, name_key)
                );
                CREATE INDEX IF NOT EXISTS idx_references_name ON \"references\"(repo_id, symbol_name);
                CREATE INDEX IF NOT EXISTS idx_references_file ON \"references\"(repo_id, file_path);
                CREATE INDEX IF NOT EXISTS idx_call_edges_callee ON call_edges(repo_id, callee_name);
                CREATE INDEX IF NOT EXISTS idx_call_edges_callee_short ON call_edges(repo_id, callee_short_name);
                CREATE INDEX IF NOT EXISTS idx_call_edges_caller ON call_edges(repo_id, caller_file_path, caller_start_line);
            """)

        # --- vectors.sqlite ---
        self.vectors_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.vectors_db_path, timeout=30.0) as vc:
            vc.execute("PRAGMA journal_mode = WAL")
            vc.executescript("""
                CREATE TABLE IF NOT EXISTS symbol_vectors (
                    repo_id        TEXT NOT NULL,
                    symbol_id      TEXT NOT NULL,
                    content_hash   TEXT NOT NULL,
                    embedder_name  TEXT NOT NULL,
                    embedding_dim  INTEGER NOT NULL,
                    index_version  INTEGER NOT NULL,
                    vector_blob    BLOB NOT NULL,
                    PRIMARY KEY (repo_id, symbol_id)
                );
                CREATE INDEX IF NOT EXISTS idx_symbol_vectors_provenance
                    ON symbol_vectors(repo_id, embedder_name, embedding_dim, index_version);
            """)

        # --- fts.sqlite ---
        self.fts_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.fts_db_path, timeout=30.0) as fc:
            fc.execute("PRAGMA journal_mode = WAL")
            fc.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS file_line_fts USING fts5(
                    repo_id UNINDEXED,
                    file_path UNINDEXED,
                    line UNINDEXED,
                    text
                );
            """)

        # Recompute AFTER the DDL: the files may have just been created, so the
        # pre-DDL identity (with (-1, -1) placeholders) must not be cached.
        self._secondary_schemas_key = self._secondary_db_identity()

    def _attach_secondary_dbs(self, conn: sqlite3.Connection, *, readonly: bool = False) -> None:
        """Attach intel, vectors, and fts secondary databases to *conn*.

        Read-only connections are opened with uri=True, so URI-mode ATTACH works.
        Write-mode connections use plain file paths because they are not opened
        in URI mode (plain sqlite3.connect(path)).  Suppresses errors for missing
        files (e.g. read-only open before the first write-mode open has created them).
        """
        for alias, path in (
            ("intel", self.intel_db_path),
            ("vectors", self.vectors_db_path),
            ("fts", self.fts_db_path),
        ):
            if readonly:
                # Read-only main connection uses uri=True so URI ATTACH works.
                attach_target = f"file:{path}?mode=ro"
            else:
                # Write-mode main connection is not opened with uri=True;
                # use a plain path so SQLite doesn't reject the URI.
                attach_target = str(path)
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(f"ATTACH DATABASE ? AS {alias}", (attach_target,))
            # Match main-DB memory settings on every attached DB -- sized the same
            # way as the main connection (see _apply_pragmas/_sized_mmap_bytes),
            # not the old fixed 256MB/4MB, since file_line_fts (in the attached
            # 'fts' db) can be as large as the main symbol table.
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(f"PRAGMA {alias}.mmap_size = {_sized_mmap_bytes(path)}")
                conn.execute(f"PRAGMA {alias}.cache_size = -{_sized_cache_kb(4_096)}")
            if not readonly:
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(f"PRAGMA {alias}.journal_mode = WAL")
                    conn.execute(f"PRAGMA {alias}.synchronous = NORMAL")

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        scoped = getattr(self._scoped_conn_tls, "conn", None)
        if scoped is not None:
            return _ReusedConnection(scoped)  # type: ignore[return-value]
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if readonly:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=30.0)
        else:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            # Ensure secondary DBs are initialised before attaching.
            self._init_secondary_schemas()
        self._apply_pragmas(conn, readonly=readonly)
        conn.row_factory = sqlite3.Row
        self._attach_secondary_dbs(conn, readonly=readonly)
        return conn

    @contextlib.contextmanager
    def _reuse_connection(self) -> Iterator[None]:
        """Within this scope all _connect() calls on this thread share one
        connection, removing per-query connect + PRAGMA overhead. Reentrant (a
        nested scope is a no-op). Commits then closes the shared connection on
        exit; per-call ``with``/``closing`` blocks inside get a proxy whose
        close()/__exit__ are no-ops, so they cannot tear it down early."""
        if os.environ.get("NO_REUSE"):  # diagnostic bypass
            yield
            return
        if getattr(self._scoped_conn_tls, "conn", None) is not None:
            yield
            return
        conn = self._connect()
        self._scoped_conn_tls.conn = conn
        self._file_cache_tls.cache = {}  # activate per-call file cache (dict[str, bytes])
        self._file_cache_tls.lines = {}  # per-call splitlines cache (dict[str, list[str]])
        try:
            yield
        finally:
            self._scoped_conn_tls.conn = None
            self._file_cache_tls.cache = None  # clear file cache
            self._file_cache_tls.lines = None
            # Keep the in-memory vector matrix across tool calls when it is small
            # enough: re-reading + unpacking the blob store is NOT cheap (317 ms
            # for a 42k-vector repo, seconds at linux scale), so dropping it made
            # every interactive semantic query pay a full reload. Retain up to
            # LEMONCROW_ANN_CACHE_MAX_MB (default 512) of matrix so the common
            # single-repo session stays warm; drop anything larger so a giant
            # corpus never pins RAM (those use the chunked path anyway). The
            # cache key carries index_version, so a reindex still invalidates it.
            _cache = self._ann_vectors_cache
            if _cache is not None:
                _mtx = _cache[2]
                _cap_mb = int(os.environ.get("LEMONCROW_ANN_CACHE_MAX_MB", "512"))
                if getattr(_mtx, "nbytes", 0) > _cap_mb * 1024 * 1024:
                    self._ann_vectors_cache = None
            with contextlib.suppress(Exception):
                conn.commit()
            with contextlib.suppress(Exception):
                conn.close()
            # NOTE: pooling this write connection across tool calls was tried
            # and reverted: engine code leaves partially-consumed cursors
            # behind, and an un-reset statement pins the WAL read snapshot, so
            # a pooled connection serves STALE reads after an external write
            # (reproduced via test_autosync_incremental_reindex). close() is
            # the only reset sqlite3 guarantees. The former per-call costs
            # were fixed at their sources instead (_init_secondary_schemas
            # identity skip, _ram_budget_bytes TTL cache).
            # Periodic heap trim: return freed Python arenas and glibc pages to
            # the OS so RSS stays bounded across long sessions.  Time-gated (30s
            # min interval): gc.collect() on a large loaded heap costs ~76ms, so
            # the old every-5-calls cadence burned >20% of wall time at
            # benchmark rates (25+ calls/s) for zero extra RSS benefit.
            self._tool_call_count += 1
            if self._tool_call_count % 5 == 0 and time.monotonic() - self._last_heap_trim_ts >= 30.0:
                self._last_heap_trim_ts = time.monotonic()
                self._trim_heap()

    def _trim_heap(self) -> None:
        """Return freed memory to the OS after heavy tool calls.

        Runs Python's cyclic GC to reclaim reference cycles, then calls
        ``malloc_trim(0)`` via libc so the top of the glibc heap is returned
        to the kernel.  Keeps long-lived MCP sessions from accumulating RSS
        as Python's allocator holds on to freed arenas between tool calls.
        No-op (silently) on non-Linux where ``malloc_trim`` is unavailable.
        """
        import gc

        gc.collect()
        try:
            import ctypes as _ct

            _ct.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def _apply_pragmas(self, conn: sqlite3.Connection, *, readonly: bool = False) -> None:
        conn.execute("PRAGMA busy_timeout = 30000")
        # Use the OS page cache for reads instead of a private anonymous buffer pool.
        # With a 840 MB code_context.sqlite, the default SQLite page cache (8 MB of
        # private anonymous mmap) is tiny and provides no meaningful hit rate; the OS
        # page cache already holds the hot pages.  mmap_size lets SQLite read directly
        # from those shared pages, converting private-anon RSS to reclaimable
        # file-backed pages -- sized to the actual file (see _sized_mmap_bytes) so a
        # DB bigger than the old fixed 256MB cap (this docstring's own 840MB example
        # exceeded it) gets fully mapped instead of falling back past that point.
        # cache_size used to be a fixed 4 MB; it now scales to available RAM (see
        # _sized_cache_kb, floored at the old 4MB) since a private buffer is still
        # useful for the WAL write path and any pages mmap doesn't cover.
        conn.execute(f"PRAGMA mmap_size = {_sized_mmap_bytes(self.db_path)}")
        conn.execute(f"PRAGMA cache_size = -{_sized_cache_kb(4_096)}")
        if readonly:
            return
        if self._wal_primed:
            conn.execute("PRAGMA synchronous = NORMAL")
            return
        # journal_mode=WAL is a persistent DB-level setting; probe/set it once per
        # engine instead of on every read-write connection.
        row = conn.execute("PRAGMA journal_mode").fetchone()
        current_mode = str(row[0]).lower() if row else ""
        if current_mode == "wal":
            self._wal_primed = True
        else:
            # WAL gives concurrent readers + a single writer across processes, so
            # reads never get "database is locked". The switch only fails while
            # another connection holds a lock; busy_timeout (set above) lets it
            # wait for a quiet moment, and once flipped WAL persists on the file.
            # _wal_primed latches ONLY on a confirmed flip: a suppressed failure
            # here (e.g. an autosync/watcher read overlapping the very first
            # write connect) must retry on the next connect, otherwise the DB
            # stays in DELETE journal mode and any reader overlapping a
            # multi-db write transaction later raises an immediate
            # 'database is locked' (mid-transaction BUSY skips the handler).
            with contextlib.suppress(sqlite3.OperationalError):
                result = conn.execute("PRAGMA journal_mode=WAL").fetchone()
                if result is not None and str(result[0]).lower() == "wal":
                    self._wal_primed = True
                else:
                    logger.debug(
                        "code index WAL switch deferred (journal_mode=%s)",
                        result[0] if result else "unknown",
                    )
        conn.execute("PRAGMA synchronous = NORMAL")

    def connection(self) -> sqlite3.Connection:
        conn = self._connect()
        self._init_schema(conn)
        return conn

    def _schema_current(self, conn: sqlite3.Connection) -> bool:
        """Read-only probe: True when the schema is fully created AND migrated.

        Lets ``_init_schema`` skip its DDL/write path entirely, so pure-read
        engines (benchmark harnesses, read-only MCP tools on provisioned DBs)
        never take a write lock on a live DB another process may be writing.
        Mirrors every migration below -- any miss falls through to the full
        write path.
        """
        try:
            objects = {
                str(r[0])
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','index')").fetchall()
            }
            required = {
                "engine_state",
                "files",
                "symbols",
                "symbol_fts",
                "symbol_fts_vocab",
                "symbol_trigram",
                "file_path_trigram",
                "imports",
                "commit_chunks",
                # newest indexes stand in for the full index set
                "idx_symbols_repo_lower_name",
                "idx_symbols_repo_kind",
            }
            if not required <= objects:
                return False
            # symbol_fts must carry the prefix indexes (see the _init_schema migration)
            fts_sql_row = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'symbol_fts'").fetchone()
            if fts_sql_row is None or "prefix=" not in str(fts_sql_row[0] or ""):
                return False
            # old monolithic tables must have been migrated out of the main DB
            if objects & {"references", "call_edges", "centrality_map", "file_line_fts"}:
                return False
            if "mtime_ns" not in {str(row[1]) for row in conn.execute("PRAGMA table_info(files)")}:
                return False
            if "signature" in {str(row[1]) for row in conn.execute("PRAGMA table_info(symbol_trigram)")}:
                return False
            if (
                conn.execute("SELECT 1 FROM symbol_trigram LIMIT 1").fetchone() is None
                and conn.execute("SELECT 1 FROM symbols LIMIT 1").fetchone() is not None
            ):
                return False  # trigram backfill pending
            if (
                conn.execute("SELECT 1 FROM file_path_trigram LIMIT 1").fetchone() is None
                and conn.execute("SELECT 1 FROM files LIMIT 1").fetchone() is not None
            ):
                return False  # file path trigram backfill pending
            if conn.execute("SELECT 1 FROM engine_state WHERE key = 'index_version'").fetchone() is None:
                return False
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'sqlite_stat1'").fetchone() is None:
                with contextlib.suppress(sqlite3.OperationalError):
                    if conn.execute("SELECT 1 FROM call_edges LIMIT 1").fetchone() is not None:
                        return False  # one-time ANALYZE pending
            return True
        except sqlite3.Error:
            return False

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        if self._schema_current(conn):
            self._schema_ready = True
            return
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS engine_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files (
                repo_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                language TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL DEFAULT 0,
                indexed_at TEXT NOT NULL,
                PRIMARY KEY (repo_id, file_path)
            );
            CREATE TABLE IF NOT EXISTS symbols (
                symbol_id TEXT PRIMARY KEY,
                repo_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                language TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                kind TEXT NOT NULL,
                signature TEXT NOT NULL,
                start_byte INTEGER NOT NULL,
                end_byte INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                parent_symbol TEXT,
                doc_summary TEXT,
                content_hash TEXT NOT NULL
            );
            -- prefix indexes turn the "term"* prefix-channel MATCH from a
            -- term-range scan + doclist merge into direct doclist seeks for
            -- prefixes of 2-6 chars (the bulk of what _fts_prefix_query_from_terms
            -- emits after IDF pruning). Results are identical; cost is index size.
            CREATE VIRTUAL TABLE IF NOT EXISTS symbol_fts USING fts5(
                symbol_id UNINDEXED,
                name,
                qualified_name,
                signature,
                file_path UNINDEXED,
                source,
                prefix='2 3 4 5 6'
            );
            -- Read-only term->document-frequency view over the FTS index (zero write
            -- cost, auto-maintained). Powers IDF pruning of common query tokens.
            CREATE VIRTUAL TABLE IF NOT EXISTS symbol_fts_vocab USING fts5vocab(symbol_fts, 'row');
            CREATE VIRTUAL TABLE IF NOT EXISTS symbol_trigram USING fts5(
                symbol_id UNINDEXED,
                name,
                qualified_name,
                file_path,
                tokenize='trigram'
            );
            -- One row per FILE (not per symbol like symbol_trigram): the path
            -- channel in _search_symbols_local matches path patterns against this
            -- much smaller table (file-cardinality: thousands even on a huge repo)
            -- and joins back to symbols via idx_symbols_repo_file, instead of
            -- scanning symbol_trigram's one-row-per-symbol duplication of every
            -- file's path.
            CREATE VIRTUAL TABLE IF NOT EXISTS file_path_trigram USING fts5(
                repo_id UNINDEXED,
                file_path,
                tokenize='trigram'
            );
            CREATE INDEX IF NOT EXISTS idx_symbols_repo_name_nocase
                ON symbols(repo_id, symbol_name COLLATE NOCASE);
            CREATE INDEX IF NOT EXISTS idx_symbols_repo_qual_nocase
                ON symbols(repo_id, qualified_name COLLATE NOCASE);
            CREATE TABLE IF NOT EXISTS imports (
                repo_id TEXT NOT NULL,
                source_file TEXT NOT NULL,
                raw_import TEXT NOT NULL,
                target_file TEXT,
                UNIQUE(repo_id, source_file, raw_import, target_file)
            );
            CREATE INDEX IF NOT EXISTS idx_symbols_repo_file ON symbols(repo_id, file_path);
            CREATE INDEX IF NOT EXISTS idx_symbols_repo_name ON symbols(repo_id, symbol_name);
            -- Covers _hef_exact_symbol_candidates: WHERE repo_id=? AND lower(symbol_name) IN (?)
            -- Turns the full per-repo scan into O(k) index lookups (k = #query identifiers).
            -- 742x speedup on large repos (django: 35ms → 0ms).
            CREATE INDEX IF NOT EXISTS idx_symbols_repo_lower_name ON symbols(repo_id, lower(symbol_name));
            -- Covers _complete_sibling_families: WHERE repo_id=? AND lower(kind)=? ...
            -- instr(lower(symbol_name),?) scans only the matching-kind rows rather than
            -- the whole repo, cutting ~200K-row scans to ~20K (10x for 10 kinds).
            CREATE INDEX IF NOT EXISTS idx_symbols_repo_kind ON symbols(repo_id, lower(kind));
            CREATE INDEX IF NOT EXISTS idx_imports_target ON imports(repo_id, target_file);
            CREATE TABLE IF NOT EXISTS commit_chunks (
                commit_sha     TEXT PRIMARY KEY,
                author_date    INTEGER NOT NULL,
                files_touched  TEXT NOT NULL,
                symbols_touched TEXT,
                summary        TEXT NOT NULL,
                summary_model  TEXT NOT NULL,
                embedding      BLOB,
                index_version  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_commit_author_date ON commit_chunks(author_date);
            CREATE INDEX IF NOT EXISTS idx_commit_files ON commit_chunks(files_touched);
            """)
        # Migration: older DBs predate the files.mtime_ns column used to fast-skip
        # unchanged files during incremental reindex. CREATE TABLE IF NOT EXISTS
        # never adds a column to an existing table, so add it here when absent.
        file_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(files)")}
        if "mtime_ns" not in file_columns:
            conn.execute("ALTER TABLE files ADD COLUMN mtime_ns INTEGER NOT NULL DEFAULT 0")
        # Migration: old trigram schema had 5 content columns (name, qualified_name,
        # signature, file_path). Drop and rebuild with the slim 3-column schema
        # (name, qualified_name, file_path) — signature added ~5x size bloat and
        # is covered by symbol_fts anyway.
        with contextlib.suppress(Exception):
            _trig_cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(symbol_trigram)")}
            if "signature" in _trig_cols:
                conn.execute("DROP TABLE IF EXISTS symbol_trigram")
                conn.execute(
                    "CREATE VIRTUAL TABLE symbol_trigram USING fts5("
                    " symbol_id UNINDEXED, name, qualified_name, file_path,"
                    " tokenize='trigram')"
                )
        # Backfill the substring trigram index for DBs built before it existed, so the
        # substring/path channels use the index instead of full-scanning symbols.
        if conn.execute("SELECT 1 FROM symbol_trigram LIMIT 1").fetchone() is None:
            if conn.execute("SELECT 1 FROM symbols LIMIT 1").fetchone() is not None:
                conn.execute(
                    "INSERT INTO symbol_trigram(symbol_id, name, qualified_name, file_path) "
                    "SELECT symbol_id, symbol_name, qualified_name, file_path FROM symbols"
                )
        # Backfill the file-level path trigram index for DBs built before it existed
        # (see _search_symbols_local's path channel).
        if conn.execute("SELECT 1 FROM file_path_trigram LIMIT 1").fetchone() is None:
            if conn.execute("SELECT 1 FROM files LIMIT 1").fetchone() is not None:
                conn.execute(
                    "INSERT INTO file_path_trigram(repo_id, file_path) SELECT DISTINCT repo_id, file_path FROM files"
                )
        # Migration: symbol_fts built before the prefix indexes existed. FTS5 cannot
        # add prefix indexes to an existing table, so drop and recreate it empty
        # (with its dependent fts5vocab table), then clear `files` so the change
        # detector sees every file as new: the background autosync / code_warm
        # reindex repopulates symbol_fts and bumps index_version exactly like any
        # other reindex (no blocking rebuild on the request path). Query results
        # are unchanged -- prefix indexes only accelerate "term"* MATCH queries.
        _fts_sql_row = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'symbol_fts'").fetchone()
        if _fts_sql_row is not None and "prefix=" not in str(_fts_sql_row[0] or ""):
            conn.execute("DROP TABLE IF EXISTS symbol_fts_vocab")
            conn.execute("DROP TABLE IF EXISTS symbol_fts")
            conn.execute(
                "CREATE VIRTUAL TABLE symbol_fts USING fts5("
                " symbol_id UNINDEXED, name, qualified_name, signature,"
                " file_path UNINDEXED, source, prefix='2 3 4 5 6')"
            )
            conn.execute("CREATE VIRTUAL TABLE symbol_fts_vocab USING fts5vocab(symbol_fts, 'row')")
            conn.execute("DELETE FROM files")
        conn.execute("INSERT OR IGNORE INTO engine_state(key, value) VALUES ('index_version', '0')")
        # Self-heal DBs built before planner statistics were collected: with data
        # but no sqlite_stat1, SQLite full-scans call_edges (it mis-picks the
        # repo_id-only index for caller-keyed queries). A one-time guarded ANALYZE
        # fixes existing DBs without requiring a reindex; new indexes get stats
        # via PRAGMA optimize at the end of _index_repo_unsafe.
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'sqlite_stat1'").fetchone() is None:
            if conn.execute("SELECT 1 FROM call_edges LIMIT 1").fetchone() is not None:
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute("ANALYZE")
        # Migration: if these tables still exist in the main DB (old monolithic schema),
        # move their data to the appropriate secondary DB and drop from main.
        _main_tables = {
            str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','index')").fetchall()
        }
        for _tbl in ('"references"', "call_edges", "centrality_map"):
            _bare = _tbl.strip('"')
            if _bare in _main_tables:
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute(f"INSERT OR IGNORE INTO intel.{_tbl} SELECT * FROM {_tbl}")
                    conn.execute(f"DROP TABLE IF EXISTS {_tbl}")
        if "file_line_fts" in _main_tables:
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute("INSERT INTO fts.file_line_fts SELECT repo_id, file_path, line, text FROM file_line_fts")
                conn.execute("DROP TABLE IF EXISTS file_line_fts")
        self._schema_ready = True

    def index_ready(self) -> bool:
        """True once the symbol index has at least one indexed file for this repo."""
        if self._index_ready_cached:
            return True
        try:
            with self._connect() as conn:
                self._init_schema(conn)
                row = conn.execute("SELECT 1 FROM files WHERE repo_id = ? LIMIT 1", (self.repo_id,)).fetchone()
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return False
        if row is not None:
            self._index_ready_cached = True
            return True
        return False

    def _ensure_autosync_worker_alive(self) -> None:
        if not self._autosync_enabled or self._autosync_stop.is_set():
            return
        t = self._autosync_thread
        if t is not None and t.is_alive():
            return
        self._autosync_thread = None
        self._start_autosync_worker()

    def _maybe_warm_ann_cache(self) -> None:
        """Kick off a background pre-warm of the semantic ANN matrix cache.

        ``_search_symbols_semantic_ann``'s small-repo fast path (vector count at
        or below ``_ANN_CACHE_LIMIT``, which covers even linux's 1.24M) loads the
        whole vectors table into an in-memory matrix and caches it in
        ``self._ann_vectors_cache`` on first use. That load is real cost -- tens
        of seconds at linux scale, confirmed dominated by materializing ~1.24M
        rows through the sqlite3 DB-API (the numpy reconstruction itself is only
        ~2s) -- and today it lands on whichever query happens to be first
        against a freshly started process, i.e. exactly the query a real user is
        waiting on. Doing the same load here instead, once, on a background
        thread, as soon as the repo is known-ready, means real queries almost
        always find a warm cache. Idempotent per engine instance
        (``_ann_warm_started``) and safe to race against a live query doing its
        own identical load -- both write the same ``(cache_key, ids, matrix)``
        tuple.
        """
        if self._ann_warm_started:
            return
        if not self._semantic_ranker.available:
            return
        embedder = self._semantic_ranker.embedder
        embedding_dim = embedder.dim
        if embedding_dim <= 0:
            return
        self._ann_warm_started = True

        def _warm() -> None:
            try:
                index_version = self._current_index_version()
                cache_key = (embedder.name, embedding_dim, index_version)
                cached = self._ann_vectors_cache
                if cached is not None and cached[0] == cache_key:
                    return
                ann_cache_limit = int(os.environ.get("LEMONCROW_ANN_CACHE_LIMIT", "2000000"))
                with self._connect() as conn:
                    self._init_schema(conn)
                    vec_count: int = conn.execute(
                        "SELECT COUNT(*) FROM symbol_vectors WHERE repo_id=? AND embedder_name=? AND embedding_dim=?",
                        (self._ann_symbol_index.repo_id, embedder.name, embedding_dim),
                    ).fetchone()[0]
                    if vec_count == 0 or vec_count > ann_cache_limit:
                        # Nothing embedded yet, or the large-repo chunked-streaming
                        # path (which never caches by design) -- nothing to warm.
                        return
                # Single-flight + flat-file backed (see _ensure_ann_matrix_loaded):
                # shares the same warm cache as query-time semantic search and
                # prewarm_semantic_matrix instead of racing them with its own load.
                self._ensure_ann_matrix_loaded(embedder.name, embedding_dim, index_version)
            except Exception:
                logger.debug("ANN cache warm-up failed; queries will load on demand", exc_info=True)

        threading.Thread(
            target=_warm,
            name=f"lemoncrow-ann-warm-{self.repo_id[:8]}",
            daemon=True,
        ).start()

    def _ensure_indexed(self) -> None:
        if self.index_ready():
            # Change detection + reindex is the background autosync worker's job
            # (it polls every _autosync_poll_ms). Running it inline here would
            # stat every source file in the repo on every read tool call -- the
            # per-call tax that made grep/read/explore slow on large repos. Keep
            # the worker alive and let it own resync; files just edited are
            # already current via the targeted _reindex_files after each edit.
            #
            # /tmp is the one exception: never start the worker there unless
            # explicitly allowed. A /tmp benchmark snapshot that's already
            # indexed but has an embedder newly pointed at it (e.g. a
            # multi-channel eval run) hits _maybe_autosync_reindex_locked's
            # "embedder just became available" branch, which lazily backfills
            # symbol_vectors on the polling thread -- too slowly to finish
            # inside that repo's query window before the caller moves on.
            # Confirmed via a real sweep: astropy, django, matplotlib, seaborn,
            # flask, requests, xarray, pylint, pytest and scikit-learn's /tmp
            # snapshots all sat at 0% or partial symbol_vectors coverage right
            # when each repo's queries ran, timestamps matching per-repo
            # processing order exactly -- silently producing nondeterministic,
            # near-empty semantic results instead of either full coverage or a
            # loud failure. Static /tmp snapshots should never implicitly
            # resync; embed them explicitly via a dedicated backfill first.
            if self._autosync_enabled and not _is_implicit_tmp_index_blocked(self.repo_root):
                self._ensure_autosync_worker_alive()
            self._ensure_lineage_ready()
            self._maybe_warm_ann_cache()
            return
        if self._autosync_enabled:
            if _is_implicit_tmp_index_blocked(self.repo_root):
                logger.warning(
                    "[autoindex] refusing to auto-build a first-time index for %s (under /tmp) -- "
                    "implicit indexing of /tmp paths is off by default so a stray query against an "
                    "unindexed scratch/benchmark directory can't silently kick off a large background "
                    "index job. Set LEMONCROW_ALLOW_TMP_AUTOINDEX=1 to opt in, or index explicitly via "
                    "`lc code index --repo-root %s --reindex`.",
                    self.repo_root,
                    self.repo_root,
                )
                return
            self._ensure_autosync_worker_alive()
            return
        # autosync always on in practice; index will be built by the worker

    def _excluded(self, path: Path, patterns: list[str]) -> bool:
        rel = _safe_relpath(self.repo_root, path)
        return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)

    def _normalize_file_arg(self, value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            return _safe_relpath(self.repo_root, path)
        return str(path)

    def _resolve_inside_repo(self, value: str) -> Path:
        path = Path(value)
        resolved = path.resolve() if path.is_absolute() else (self.repo_root / path).resolve()
        try:
            resolved.relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError(f"path escape denied: {value}") from exc
        return resolved

    def _extract_python_symbols(self, source: str) -> list[_ExtractedSymbol]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        offsets = _line_offsets(source)
        lines = source.splitlines()
        symbols: list[_ExtractedSymbol] = []

        def line_text(line_no: int) -> str:
            if 1 <= line_no <= len(lines):
                return lines[line_no - 1].strip()
            return ""

        def add_node(node: ast.AST, name: str, kind: str, parent: str | None) -> None:
            start_line = int(getattr(node, "lineno", 1))
            end_line = int(getattr(node, "end_lineno", start_line))
            col = int(getattr(node, "col_offset", 0))
            end_col = int(getattr(node, "end_col_offset", 0))
            start_byte = offsets[max(0, start_line - 1)] + col
            end_byte = offsets[max(0, end_line - 1)] + end_col if end_col else offsets[min(end_line, len(offsets) - 1)]
            qualified = f"{parent}.{name}" if parent else name
            doc = (
                ast.get_docstring(node)
                if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
                else None
            )
            symbols.append(
                _ExtractedSymbol(
                    name=name,
                    qualified_name=qualified,
                    kind=kind,
                    signature=line_text(start_line),
                    start_byte=start_byte,
                    end_byte=max(start_byte, end_byte),
                    start_line=start_line,
                    end_line=end_line,
                    parent_symbol=parent,
                    doc_summary=(stripped.splitlines()[0][:200] if doc and (stripped := doc.strip()) else None),
                )
            )

        def walk_body(body: list[ast.stmt], parent: str | None = None) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    add_node(node, node.name, "class", parent)
                    walk_body(node.body, node.name if parent is None else f"{parent}.{node.name}")
                elif isinstance(node, ast.AsyncFunctionDef):
                    add_node(node, node.name, "method" if parent else "async_function", parent)
                elif isinstance(node, ast.FunctionDef):
                    add_node(node, node.name, "method" if parent else "function", parent)
                elif parent is None and isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            add_node(node, target.id, "variable", None)
                elif parent is None and isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    add_node(node, node.target.id, "variable", None)

        walk_body(tree.body)
        return sorted(symbols, key=lambda item: (item.start_line, item.qualified_name))

    @staticmethod
    def _python_call_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = CodeContextEngine._python_call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        if isinstance(node, ast.Call):
            return CodeContextEngine._python_call_name(node.func)
        return None

    def _extract_tag_symbols(self, path: Path, source: str, language: str) -> list[_ExtractedSymbol]:
        del language
        try:
            tags = [tag for tag in extract_tags(path) if tag.kind == "definition"]
        except (OSError, SyntaxError):
            return []
        offsets = _line_offsets(source)
        lines = source.splitlines()
        sorted_tags = sorted(tags, key=lambda tag: (tag.line, tag.name))
        symbols: list[_ExtractedSymbol] = []
        for index, tag in enumerate(sorted_tags):
            start_line = max(1, tag.line)
            next_line = sorted_tags[index + 1].line - 1 if index + 1 < len(sorted_tags) else start_line
            end_line = max(start_line, min(next_line, len(lines)))
            start_byte = offsets[start_line - 1] if start_line - 1 < len(offsets) else tag.byte_range[0]
            end_byte = offsets[end_line] if end_line < len(offsets) else tag.byte_range[1]
            signature = lines[start_line - 1].strip() if start_line <= len(lines) else tag.name
            symbols.append(
                _ExtractedSymbol(
                    name=tag.name,
                    qualified_name=tag.name,
                    kind=self._kind_from_signature(signature),
                    signature=signature,
                    start_byte=start_byte,
                    end_byte=max(start_byte, end_byte),
                    start_line=start_line,
                    end_line=end_line,
                )
            )
        return symbols

    def _python_imports(self, path: Path, source: str) -> list[tuple[str, str | None]]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        imports: list[tuple[str, str | None]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, self._resolve_python_module(path.parent, alias.name)))
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append((node.module, self._resolve_python_module(path.parent, node.module)))
        return imports

    def _javascript_imports(self, path: Path, source: str) -> list[tuple[str, str | None]]:
        imports: list[tuple[str, str | None]] = []
        for match in _JS_IMPORT_RE.finditer(source):
            raw = next(group for group in match.groups() if group)
            target = None
            if raw.startswith("."):
                target = self._resolve_relative_module(path.parent, raw, [".ts", ".tsx", ".js", ".jsx"])
            imports.append((raw, target))
        return imports

    def _resolve_python_module(self, base: Path, module: str) -> str | None:
        parts = module.split(".")
        search_bases: list[Path] = []
        for candidate in [base, *base.parents, self.repo_root, self.repo_root / "src"]:
            resolved = candidate.resolve()
            if resolved not in search_bases:
                search_bases.append(resolved)
        for search_base in search_bases:
            candidate = search_base / Path(*parts).with_suffix(".py")
            if candidate.is_file():
                return _safe_relpath(self.repo_root, candidate)
            package = search_base / Path(*parts) / "__init__.py"
            if package.is_file():
                return _safe_relpath(self.repo_root, package)
            # src-layout imports often omit the top-level src directory while
            # file-local parent probing starts below it. Also handle package-root
            # candidates such as lemoncrow.core.foo -> src/lemoncrow/core/foo.py.
            src_candidate = self.repo_root / "src" / Path(*parts).with_suffix(".py")
            if src_candidate.is_file():
                return _safe_relpath(self.repo_root, src_candidate)
            src_package = self.repo_root / "src" / Path(*parts) / "__init__.py"
            if src_package.is_file():
                return _safe_relpath(self.repo_root, src_package)
        return None

    def _resolve_relative_module(self, base: Path, raw: str, suffixes: list[str]) -> str | None:
        candidate_base = (base / raw).resolve()
        candidates: list[Path] = []
        if candidate_base.suffix:
            candidates.append(candidate_base)
        else:
            candidates.extend(candidate_base.with_suffix(suffix) for suffix in suffixes)
            candidates.extend(candidate_base / f"index{suffix}" for suffix in suffixes)
            candidates.extend(candidate_base / f"mod{suffix}" for suffix in suffixes)
        for candidate in candidates:
            if candidate.is_file():
                return _safe_relpath(self.repo_root, candidate)
        return None

    def _kind_from_signature(self, signature: str) -> str:
        stripped = signature.lstrip()
        if stripped.startswith("class "):
            return "class"
        if stripped.startswith(("interface ", "type ")):
            return "type"
        if stripped.startswith(("function ", "func ", "fn ")):
            return "function"
        if stripped.startswith(("struct ", "enum ", "trait ")):
            return "class"
        return "variable"

    def _symbols_for_files(self, file_paths: list[str], *, limit: int) -> list[SymbolRecord]:
        if not file_paths:
            return []
        placeholders = ",".join("?" for _ in file_paths)
        params: list[Any] = [self.repo_id, *file_paths, limit]
        with self._connect() as conn:
            self._init_schema(conn)
            rows = conn.execute(
                f"""
                SELECT *, NULL AS score FROM symbols
                WHERE repo_id = ? AND file_path IN ({placeholders})
                ORDER BY file_path, start_line
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [_row_to_symbol(row) for row in rows]

    def _dedupe_symbols(self, symbols: list[SymbolRecord]) -> list[SymbolRecord]:
        seen: set[str] = set()
        output: list[SymbolRecord] = []
        for symbol in symbols:
            if symbol.symbol_id in seen:
                continue
            seen.add(symbol.symbol_id)
            output.append(symbol)
        return output

    def _cap_symbols_per_file(self, symbols: list[SymbolRecord], *, max_per_file: int) -> list[SymbolRecord]:
        if max_per_file <= 0:
            return symbols
        counts: dict[str, int] = {}
        output: list[SymbolRecord] = []
        for symbol in symbols:
            seen = counts.get(symbol.file_path, 0)
            if seen >= max_per_file:
                continue
            counts[symbol.file_path] = seen + 1
            output.append(symbol)
        return output

    def _context_pack_search_limit(self, *, max_symbols: int, max_symbols_per_file: int) -> int:
        return max(
            max_symbols,
            max_symbols * max(2, max_symbols_per_file),
        )

    def _is_noise_symbol_kind(self, kind: str) -> bool:
        return kind.strip().lower() in {"import", "export"}

    def _is_context_pack_symbol(self, symbol: SymbolRecord) -> bool:
        if self._is_noise_symbol_kind(symbol.kind):
            return False
        if symbol.provenance == "commit" or symbol.kind == "commit":
            return False
        rel = str(symbol.file_path or "").strip()
        if not rel:
            return False
        with contextlib.suppress(ValueError):
            if self._resolve_inside_repo(rel).is_file():
                return True
        return False

    def _symbol_matches_compound_query(self, query_terms: list[str], symbol: SymbolRecord) -> bool:
        if len(query_terms) < 2:
            return False
        lexical = f"{symbol.symbol_name} {symbol.qualified_name} {symbol.signature}".lower()
        matched = sum(1 for term in query_terms if term and term in lexical)
        return matched >= min(len(query_terms), 3)

    def badge_counts_batch(self, symbol_names: list[str]) -> dict[str, dict[str, int]]:
        """Return caller/callee/usage counts for multiple symbols in 3 queries.

        Used by the grep badge provider to replace 3xN serial queries with 3
        bulk ``IN (?)`` queries. Returns a mapping of symbol_name →
        {callers: N, callees: N, usages: N}; missing symbols map to all zeros.
        Fail-open: any error returns the zero-filled map so grep never breaks.
        """
        if not symbol_names:
            return {}
        names = list(dict.fromkeys(symbol_names))  # deduplicate, preserve order
        ph = ",".join("?" for _ in names)
        result: dict[str, dict[str, int]] = {n: {"callers": 0, "callees": 0, "usages": 0} for n in names}
        try:
            with self._connect() as conn:
                # callers: other symbols that call each of the badge symbols
                for row in conn.execute(
                    f"SELECT callee_name, COUNT(*) AS n FROM call_edges "
                    f"WHERE repo_id = ? AND callee_name IN ({ph}) GROUP BY callee_name",
                    (self.repo_id, *names),
                ).fetchall():
                    name = str(row["callee_name"])
                    if name in result:
                        result[name]["callers"] = int(row["n"])
                # callees: symbols each badge symbol calls
                for row in conn.execute(
                    f"SELECT caller_symbol_name, COUNT(*) AS n FROM call_edges "
                    f"WHERE repo_id = ? AND caller_symbol_name IN ({ph}) GROUP BY caller_symbol_name",
                    (self.repo_id, *names),
                ).fetchall():
                    name = str(row["caller_symbol_name"])
                    if name in result:
                        result[name]["callees"] = int(row["n"])
                # usages: reference sites of each badge symbol
                for row in conn.execute(
                    f'SELECT symbol_name, COUNT(*) AS n FROM "references" '
                    f"WHERE repo_id = ? AND symbol_name IN ({ph}) GROUP BY symbol_name",
                    (self.repo_id, *names),
                ).fetchall():
                    name = str(row["symbol_name"])
                    if name in result:
                        result[name]["usages"] = int(row["n"])
        except Exception:
            logging.exception("Recovered in badge_counts_batch")
        return result

    def _symbol_popularity_scores(self, symbols: list[SymbolRecord]) -> dict[str, float]:
        """Batch-compute a usage-frequency popularity score per candidate symbol.

        Popularity blends indexed reference counts (the ``references`` table,
        keyed by ``symbol_name``) with caller counts (``call_edges``, keyed by
        ``callee_name``). Both lookups hit existing indexes, so this is cheap and
        always available -- it never requires git. The raw counts are squashed
        into [0, 1) so a wildly-popular symbol cannot dominate; popularity is
        only ever consumed as a low-priority ranking tiebreaker.
        """
        names = sorted({symbol.symbol_name for symbol in symbols if symbol.symbol_name})
        if not names:
            return {}
        placeholders = ",".join("?" for _ in names)
        ref_counts: dict[str, int] = {}
        caller_counts: dict[str, int] = {}
        try:
            with self._connect() as conn:
                self._init_schema(conn)
                for row in conn.execute(
                    f'SELECT symbol_name, COUNT(*) AS n FROM "references" '
                    f"WHERE repo_id = ? AND symbol_name IN ({placeholders}) GROUP BY symbol_name",
                    (self.repo_id, *names),
                ).fetchall():
                    ref_counts[str(row["symbol_name"])] = int(row["n"])
                for row in conn.execute(
                    f"SELECT callee_name, COUNT(*) AS n FROM call_edges "
                    f"WHERE repo_id = ? AND callee_name IN ({placeholders}) GROUP BY callee_name",
                    (self.repo_id, *names),
                ).fetchall():
                    caller_counts[str(row["callee_name"])] = int(row["n"])
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return {}
        scores: dict[str, float] = {}
        for symbol in symbols:
            raw = ref_counts.get(symbol.symbol_name, 0) + caller_counts.get(symbol.symbol_name, 0)
            # Diminishing-returns squash into [0, 1): popular-but-correct symbols
            # rise as a tiebreaker without ever outweighing match quality.
            scores[symbol.symbol_id] = raw / (raw + 5.0) if raw > 0 else 0.0
        return scores

    def _symbol_churn_scores(self, symbols: list[SymbolRecord]) -> dict[str, float]:
        """Per-symbol churn score in [0, 1] from the optional churn provider.

        Returns an empty mapping (no churn signal) unless a provider is injected,
        keeping ranking free of git/blame cost by default. Churn, like
        popularity, is consumed only as a low-priority ranking tiebreaker.
        """
        provider = self._churn_score_provider
        if provider is None or not symbols:
            return {}
        try:
            raw = provider(symbols)
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return {}
        return {symbol_id: max(0.0, min(1.0, float(value))) for symbol_id, value in raw.items()}

    def _context_symbol_signals(self, symbols: list[SymbolRecord]) -> dict[str, float]:
        """Combine usage-frequency and churn into a single tiebreaker per symbol."""
        if not symbols:
            return {}
        popularity = self._symbol_popularity_scores(symbols)
        churn = self._symbol_churn_scores(symbols)
        if not popularity and not churn:
            return {}
        combined: dict[str, float] = {}
        for symbol in symbols:
            combined[symbol.symbol_id] = popularity.get(symbol.symbol_id, 0.0) + churn.get(symbol.symbol_id, 0.0)
        return combined

    def _context_symbol_rank(
        self,
        query: str,
        symbol: SymbolRecord,
        *,
        popularity: float = 0.0,
    ) -> tuple[int, int, int, int, int, int, float, float, str, int, str]:
        normalized_query = query.strip().lower()
        symbol_name = symbol.symbol_name.lower()
        qualified_name = symbol.qualified_name.lower()
        query_terms = _identifier_terms(normalized_query)
        exact = int(normalized_query in {symbol_name, qualified_name})
        prefix = int(
            bool(normalized_query)
            and (symbol_name.startswith(normalized_query) or qualified_name.startswith(normalized_query))
        )
        compound = int(self._symbol_matches_compound_query(query_terms[:8], symbol))
        term_prefix_hits = sum(
            1 for term in query_terms[:8] if term and (symbol_name.startswith(term) or qualified_name.startswith(term))
        )
        tool_query = any(term in {"mcp", "tool"} for term in query_terms)
        tool_boost = 0
        if tool_query:
            if "mcp_server" in symbol.file_path.lower():
                tool_boost += 3
            if symbol_name.startswith("tool_"):
                tool_boost += 2
            if "mcp" in qualified_name:
                tool_boost += 1
        # N9: generated/scaffolding files rank last. This demotion sits AFTER the
        # authoritative exact-hit signal, so an exact symbol that legitimately
        # lives in a generated file is still surfaced; it only sinks generated
        # candidates beneath equally- or weaker-matched hand-written code.
        not_generated = 0 if is_generated_path(symbol.file_path) else 1
        # G7: popularity/churn is positioned AFTER every match-quality signal
        # (exact/prefix/compound/term/tool) and after the lexical/semantic score,
        # so it can only ever break ties among otherwise-equal candidates. An
        # exact-symbol hit (exact=1) is always ranked above any non-exact symbol
        # regardless of how popular the non-exact one is.
        return (
            exact,
            not_generated,
            prefix,
            compound,
            term_prefix_hits,
            tool_boost,
            float(symbol.score or 0.0),
            float(popularity),
            symbol.file_path,
            symbol.start_line,
            symbol.qualified_name,
        )

    def _prioritize_context_symbols(self, query: str, symbols: list[SymbolRecord]) -> list[SymbolRecord]:
        signals = self._context_symbol_signals(symbols)
        ranks = {
            symbol.symbol_id: self._context_symbol_rank(query, symbol, popularity=signals.get(symbol.symbol_id, 0.0))
            for symbol in symbols
        }
        return sorted(
            symbols,
            key=lambda symbol: (
                -ranks[symbol.symbol_id][0],
                -ranks[symbol.symbol_id][1],
                -ranks[symbol.symbol_id][2],
                -ranks[symbol.symbol_id][3],
                -ranks[symbol.symbol_id][4],
                -ranks[symbol.symbol_id][5],
                -ranks[symbol.symbol_id][6],
                -ranks[symbol.symbol_id][7],
                ranks[symbol.symbol_id][8],
                ranks[symbol.symbol_id][9],
                ranks[symbol.symbol_id][10],
                symbol.symbol_id,
            ),
        )

    def _prune_overlapping_context_symbols(self, symbols: list[SymbolRecord]) -> list[SymbolRecord]:
        kept: list[SymbolRecord] = []
        for symbol in symbols:
            if any(self._context_symbols_are_redundant(existing, symbol) for existing in kept):
                continue
            kept.append(symbol)
        return kept

    def _context_symbols_are_redundant(self, kept: SymbolRecord, candidate: SymbolRecord) -> bool:
        if self._normalize_file_arg(kept.file_path) != self._normalize_file_arg(candidate.file_path):
            return False
        kept_contains_candidate = self._context_symbol_contains(kept, candidate)
        candidate_contains_kept = self._context_symbol_contains(candidate, kept)
        return kept_contains_candidate or candidate_contains_kept

    def _context_symbol_contains(self, outer: SymbolRecord, inner: SymbolRecord) -> bool:
        outer_start = int(outer.start_line)
        outer_end = max(outer_start, int(outer.end_line))
        inner_start = int(inner.start_line)
        inner_end = max(inner_start, int(inner.end_line))
        return outer_start <= inner_start and outer_end >= inner_end

    def _context_neighbor_files(self, neighbors: list[str]) -> list[str]:
        files: list[str] = []
        for neighbor in neighbors:
            candidate = str(neighbor or "").strip()
            if not candidate:
                continue
            path = self.repo_root / candidate
            if path.is_file():
                files.append(candidate)
        return sorted(set(files))

    def _context_symbol_from_call_graph_node(self, node: CallGraphNode) -> SymbolRecord | None:
        node_file = str(node.file_path or "").strip()
        if not node_file:
            return None
        normalized_file = self._normalize_file_arg(node_file)
        with self._connect() as conn:
            self._init_schema(conn)
            node_symbol_id = str(node.symbol_id or "").strip()
            if node_symbol_id and not node_symbol_id.startswith(("local-call::", "local-callee::", "ref::")):
                row = conn.execute(
                    """
                    SELECT *, NULL AS score FROM symbols
                    WHERE repo_id = ? AND symbol_id = ?
                    LIMIT 1
                    """,
                    (self.repo_id, node_symbol_id),
                ).fetchone()
                if row is not None:
                    return _row_to_symbol(row)
            row = conn.execute(
                """
                SELECT *, NULL AS score FROM symbols
                WHERE repo_id = ? AND file_path = ? AND start_line = ?
                  AND (qualified_name = ? OR symbol_name = ?)
                ORDER BY
                  CASE
                    WHEN qualified_name = ? THEN 0
                    WHEN symbol_name = ? THEN 1
                    ELSE 2
                  END,
                  (end_line - start_line) ASC,
                  end_line ASC,
                  symbol_id ASC
                LIMIT 1
                """,
                (
                    self.repo_id,
                    normalized_file,
                    int(node.start_line),
                    str(node.qualified_name),
                    str(node.symbol_name),
                    str(node.qualified_name),
                    str(node.symbol_name),
                ),
            ).fetchone()
        if row is None:
            return None
        return _row_to_symbol(row)

    def _context_graph_related_symbols(
        self,
        selected: list[SymbolRecord],
        *,
        query: str,
        limit: int,
        max_symbols_per_file: int,
    ) -> list[SymbolRecord]:
        if limit <= 0 or not selected:
            return []
        relation_priority: dict[str, int] = {}
        candidates_by_id: dict[str, SymbolRecord] = {}
        selected_ids = {symbol.symbol_id for symbol in selected}
        for symbol in selected:
            for priority, lookup in enumerate((self.intel_store.find_callees, self.intel_store.find_callers)):
                nodes = lookup(
                    symbol_id=symbol.symbol_id,
                    qualified_name=symbol.qualified_name,
                    file_path=symbol.file_path,
                    symbol_name=symbol.symbol_name,
                )
                if not nodes:
                    continue
                for node in nodes:
                    candidate = self._context_symbol_from_call_graph_node(node)
                    if candidate is None or candidate.symbol_id in selected_ids:
                        continue
                    if not self._is_context_pack_symbol(candidate):
                        continue
                    candidates_by_id[candidate.symbol_id] = candidate
                    existing = relation_priority.get(candidate.symbol_id)
                    if existing is None or priority < existing:
                        relation_priority[candidate.symbol_id] = priority
        if not candidates_by_id:
            return []
        signals = self._context_symbol_signals(list(candidates_by_id.values()))
        ranks = {
            symbol_id: self._context_symbol_rank(query, symbol, popularity=signals.get(symbol_id, 0.0))
            for symbol_id, symbol in candidates_by_id.items()
        }
        ordered = sorted(
            candidates_by_id.values(),
            key=lambda symbol: (
                relation_priority.get(symbol.symbol_id, 99),
                -ranks[symbol.symbol_id][0],
                -ranks[symbol.symbol_id][1],
                -ranks[symbol.symbol_id][2],
                -ranks[symbol.symbol_id][3],
                -ranks[symbol.symbol_id][4],
                -ranks[symbol.symbol_id][5],
                -ranks[symbol.symbol_id][6],
                -ranks[symbol.symbol_id][7],
                ranks[symbol.symbol_id][8],
                ranks[symbol.symbol_id][9],
                ranks[symbol.symbol_id][10],
                symbol.symbol_id,
            ),
        )
        ordered = self._prune_overlapping_context_symbols(ordered)
        ordered = self._cap_symbols_per_file(ordered, max_per_file=max(1, max_symbols_per_file))
        return ordered[:limit]

    def _context_symbol_summary(self, symbol: SymbolRecord) -> dict[str, Any]:
        return {
            "symbol_id": symbol.symbol_id,
            "symbol_name": symbol.symbol_name,
            "qualified_name": symbol.qualified_name,
            "kind": symbol.kind,
            "file_path": symbol.file_path,
            "start_line": symbol.start_line,
            "end_line": symbol.end_line,
            "score": symbol.score,
            "provenance": symbol.provenance,
        }

    def _import_neighbors(self, seed_files: list[str]) -> list[str]:
        if not seed_files:
            return []
        placeholders = ",".join("?" for _ in seed_files)
        with self._connect() as conn:
            self._init_schema(conn)
            rows = conn.execute(
                f"""
                SELECT DISTINCT COALESCE(target_file, raw_import) AS neighbor
                FROM imports
                WHERE repo_id = ? AND source_file IN ({placeholders})
                UNION
                SELECT DISTINCT source_file AS neighbor
                FROM imports
                WHERE repo_id = ? AND target_file IN ({placeholders})
                ORDER BY neighbor
                """,
                tuple([self.repo_id, *seed_files, self.repo_id, *seed_files]),
            ).fetchall()
        return [str(row["neighbor"]) for row in rows if row["neighbor"]]

    def _indexed_files(self) -> list[str]:
        with self._connect() as conn:
            self._init_schema(conn)
            rows = conn.execute(
                "SELECT file_path FROM files WHERE repo_id = ? ORDER BY file_path",
                (self.repo_id,),
            ).fetchall()
        return [str(row["file_path"]) for row in rows]

    def _read_file(self, rel: str) -> str:
        # The index can reference files absent from disk (deleted, moved, or excluded
        # from a snapshot since indexing). Degrade to empty content so every caller
        # (explore source + relationships, repo map, rerank, ...) survives instead of
        # crashing the whole tool call on one stale entry.
        return self._read_file_bytes(rel).decode("utf-8", errors="replace")

    def _read_file_lines(self, rel: str) -> list[str]:
        """Whole-file splitlines, cached per tool call (splitting a large file
        costs ~1ms and explore renders many sections from the same file)."""
        cache: dict[str, list[str]] | None = getattr(self._file_cache_tls, "lines", None)
        if cache is not None:
            lines = cache.get(rel)
            if lines is None:
                lines = cache[rel] = self._read_file(rel).splitlines()
            return lines
        return self._read_file(rel).splitlines()

    def _read_file_bytes(self, rel: str) -> bytes:
        """Raw file bytes via the persistent stat-validated cache.

        A hit requires the cached (mtime_ns, size) to match a fresh stat(), so
        the returned bytes are always identical to what a direct read would
        return; missing/unreadable files degrade to b"" like the callers'
        previous OSError handling. The cache is wiped wholesale when it would
        exceed the cap -- the hot set (a repo's frequently-hit sources) refills
        in one call and the bookkeeping stays trivial."""
        path = self.repo_root / rel
        try:
            st = path.stat()
        except OSError:
            return b""
        entry = self._file_bytes_cache.get(rel)
        if entry is not None and entry[0] == st.st_mtime_ns and entry[1] == st.st_size:
            return entry[2]
        try:
            data = path.read_bytes()
        except (OSError, ValueError):
            return b""
        if entry is not None:
            self._file_bytes_cache_total -= len(entry[2])
            self._file_bytes_cache.pop(rel, None)
        if len(data) <= _FILE_BYTES_CACHE_MAX_BYTES:
            if self._file_bytes_cache_total + len(data) > _FILE_BYTES_CACHE_MAX_BYTES:
                self._file_bytes_cache.clear()
                self._file_bytes_cache_total = 0
            self._file_bytes_cache[rel] = (st.st_mtime_ns, st.st_size, data)
            self._file_bytes_cache_total += len(data)
        return data

    def _read_file_slice(self, rel: str, start_byte: int, end_byte: int) -> str:
        cache: dict[str, bytes] | None = getattr(self._file_cache_tls, "cache", None)
        if cache is not None:
            # Inside a _reuse_connection() scope: resolve each file at most once
            # per tool call regardless of how many symbols are drawn from it.
            data = cache.get(rel)
            if data is None:
                data = cache[rel] = self._read_file_bytes(rel)
            return data[start_byte:end_byte].decode("utf-8", errors="replace")
        return self._read_file_bytes(rel)[start_byte:end_byte].decode("utf-8", errors="replace")

    def _load_symbol_source_for_rerank(self, symbol: SymbolRecord) -> str:
        if symbol.provenance == "commit" or symbol.kind == "commit":
            return ""
        if not symbol.file_path or symbol.end_byte <= symbol.start_byte:
            return ""
        with contextlib.suppress(OSError, ValueError):
            return self._read_file_slice(symbol.file_path, symbol.start_byte, symbol.end_byte)
        return ""

    def _source_section_for_symbol(
        self,
        symbol: SymbolRecord | dict[str, Any],
        *,
        line_numbers: bool = True,
        skeleton: bool = False,
        max_tokens: int = _EXPLORE_SOURCE_SECTION_MAX_TOKENS,
    ) -> dict[str, Any]:
        payload = symbol.model_dump(mode="json") if isinstance(symbol, SymbolRecord) else symbol
        file_path = str(payload["file_path"])
        start_line = int(payload["start_line"])
        end_line = int(payload["end_line"])
        source = self._read_file_slice(file_path, int(payload["start_byte"]), int(payload["end_byte"]))
        # Walk back through the file line by line to capture decorator /
        # annotation lines that sit above this symbol's start_line but belong
        # to it structurally (e.g. @functools.lru_cache, @property, @app.route).
        # Line-based (not byte-based) so we never clip mid-token or overshoot.
        try:
            all_file_lines = self._read_file_lines(file_path)
            # The byte slice starts at the symbol's col_offset, so an indented
            # symbol (col_offset > 0) lost its first line's leading whitespace
            # while the body kept it. Re-prepend that indentation so the block
            # renders consistently with _render_source_section (line-based).
            first_raw = all_file_lines[start_line - 1]
            indent = first_raw[: len(first_raw) - len(first_raw.lstrip())]
            if indent and not source.startswith((" ", "\t")):
                source = indent + source
            decorator_prefix: list[str] = []
            scan = start_line - 2  # 0-indexed line immediately above symbol
            limit = max(0, scan - 10)  # never look back more than 10 lines
            while scan >= limit:
                raw = all_file_lines[scan]
                stripped = raw.strip()
                if stripped.startswith("@"):
                    decorator_prefix.insert(0, raw)
                    scan -= 1
                elif stripped == "" or stripped.startswith("#"):
                    # blank / comment between stacked decorators — skip
                    scan -= 1
                else:
                    break
        except (OSError, IndexError):
            decorator_prefix = []
        if decorator_prefix:
            start_line = start_line - len(decorator_prefix)
            source = "\n".join(decorator_prefix) + "\n" + source
        lines = source.splitlines()
        if line_numbers:
            full_content = "\n".join(f"{start_line + idx}\t{line}" for idx, line in enumerate(lines))
        else:
            full_content = source
        section: dict[str, Any] = {
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "symbol_id": payload["symbol_id"],
            "symbol_name": payload["symbol_name"],
            "qualified_name": payload["qualified_name"],
            "line_numbers": line_numbers,
            # Internal: preserved through _merge_nearby_source_sections so a merged
            # exact section keeps its full-source budget; stripped before packing.
            "_max_tokens": max_tokens,
        }
        if skeleton:
            skel = self._skeletonize_source(
                source,
                file_path=file_path,
                start_line=start_line,
                language=payload.get("language"),
                line_numbers=line_numbers,
            )
            if skel is not None:
                # Gate + metadata only (use the skeleton iff it is actually
                # shorter); a char-based estimate gives the same decision at a
                # fraction of the cost of BPE-encoding every symbol body twice.
                saved = estimate_tokens(full_content) - estimate_tokens(skel)
                if saved > 0:
                    section["content"] = cap_source_by_tokens(
                        skel,
                        max_tokens,
                        estimate_tokens,
                        start_line=start_line,
                        end_line=end_line,
                    )
                    section["skeleton"] = True
                    section["tokens_saved"] = saved
                    return section
        section["content"] = cap_source_by_tokens(
            full_content,
            max_tokens,
            estimate_tokens,
            start_line=start_line,
            end_line=end_line,
        )
        return section

    def _merge_nearby_source_sections(
        self,
        sections: list[dict[str, Any]],
        *,
        gap_lines: int = 4,
    ) -> list[dict[str, Any]]:
        if not sections:
            return []
        ordered = sorted(
            sections,
            key=lambda item: (
                str(item["file_path"]),
                int(item["start_line"]),
                int(item["end_line"]),
            ),
        )
        merged: list[dict[str, Any]] = [dict(ordered[0])]
        for section in ordered[1:]:
            current = merged[-1]
            same_file = str(current["file_path"]) == str(section["file_path"])
            near_or_overlap = int(section["start_line"]) <= int(current["end_line"]) + max(0, gap_lines)
            if same_file and near_or_overlap and not current.get("skeleton") and not section.get("skeleton"):
                line_numbers = bool(current.get("line_numbers", True))
                current["start_line"] = min(int(current["start_line"]), int(section["start_line"]))
                current["end_line"] = max(int(current["end_line"]), int(section["end_line"]))
                current["_max_tokens"] = max(
                    int(current.get("_max_tokens", _EXPLORE_SOURCE_SECTION_MAX_TOKENS)),
                    int(section.get("_max_tokens", _EXPLORE_SOURCE_SECTION_MAX_TOKENS)),
                )
                current["content"] = self._render_source_section(
                    str(current["file_path"]),
                    start_line=int(current["start_line"]),
                    end_line=int(current["end_line"]),
                    line_numbers=line_numbers,
                    max_tokens=int(current["_max_tokens"]),
                )
                continue
            merged.append(dict(section))
        for section in merged:
            section.pop("line_numbers", None)
        return merged

    def _render_source_section(
        self,
        file_path: str,
        *,
        start_line: int,
        end_line: int,
        line_numbers: bool,
        max_tokens: int = _EXPLORE_SOURCE_SECTION_MAX_TOKENS,
    ) -> str:
        lines = self._read_file_lines(file_path)
        if not lines:
            return ""
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), max(start_idx, end_line))
        segment = lines[start_idx:end_idx]
        if line_numbers:
            return cap_source_by_tokens(
                "\n".join(f"{start_line + idx}\t{line}" for idx, line in enumerate(segment)),
                max_tokens,
                estimate_tokens,
                start_line=start_line,
                end_line=end_line,
            )
        return cap_source_by_tokens(
            "\n".join(segment),
            max_tokens,
            estimate_tokens,
            start_line=start_line,
            end_line=end_line,
        )

    def _complete_sibling_families(
        self, symbols: list[SymbolRecord], *, query: str, seed_set: set[str]
    ) -> list[SymbolRecord]:
        """Surface sibling families that name-ranked search misses.

        FTS tokenization splits camelCase, so a bare affix query ('embedder')
        returns the base symbol but not 'OpenAIEmbedder'. For each strong suffix
        affix -- both the query's own tokens and those of the top selected
        symbols -- look up same-kind symbols whose name CONTAINS that affix
        (substring match); when >=3 exist, return the members not already selected
        so explore presents the whole family. Query-driven probes surface the
        family even when search ranked unrelated symbols (e.g. trivial variables)
        above it. Index lookups only, bounded by caps.
        """
        if not symbols:
            return []
        have_ids = {symbol.symbol_id for symbol in symbols}
        probes: list[tuple[str, str]] = []
        seen_probe: set[tuple[str, str]] = set()
        # Query-driven probes first -- the family the caller actually named, across
        # the definition kinds, regardless of how search ranked the raw hits.
        for affix in self._skeleton_affixes(query):
            for kind in _QUERY_PROBE_KINDS:
                key = (kind, affix)
                if key not in seen_probe:
                    seen_probe.add(key)
                    probes.append(key)
        for symbol in symbols[:_EXPLORE_FAMILY_PROBE_SYMBOLS]:
            kind = (symbol.kind or "").lower()
            if kind not in _SKELETON_KINDS:
                continue
            affixes = self._skeleton_affixes(symbol.symbol_name or symbol.qualified_name)
            if not affixes:
                continue
            key = (kind, affixes[0])  # suffix token -- the dominant family signal
            if key not in seen_probe:
                seen_probe.add(key)
                probes.append(key)
        if not probes:
            return []
        additions: list[SymbolRecord] = []
        seen_ids: set[str] = set()
        # Batch probes by affix: one trigram FTS lookup per unique affix instead of
        # N_kinds separate full-index scans.  The trigram index makes each lookup
        # O(k_matches) rather than O(N_repo_kind) for instr(), and sharing the scan
        # across kinds halves the number of SQL round-trips for query-driven probes.
        per_family_sql_limit = _EXPLORE_FAMILY_PER_FAMILY_CAP * 3
        affix_to_kinds: dict[str, list[str]] = {}
        for kind, affix in probes:
            affix_to_kinds.setdefault(affix, []).append(kind)
        try:
            with self._connect() as conn:
                self._init_schema(conn)
                for affix, batch_kinds in affix_to_kinds.items():
                    if len(additions) >= _EXPLORE_FAMILY_TOTAL_CAP:
                        break
                    like_pat = f"%{affix}%"
                    placeholders = ",".join("?" * len(batch_kinds))
                    rows = conn.execute(
                        f"""
                        SELECT s.*, NULL AS score
                        FROM symbol_trigram t JOIN symbols s ON s.symbol_id = t.symbol_id
                        WHERE s.repo_id = ? AND lower(s.kind) IN ({placeholders}) AND t.name LIKE ?
                        LIMIT ?
                        """,
                        (self.repo_id, *batch_kinds, like_pat, len(batch_kinds) * per_family_sql_limit),
                    ).fetchall()
                    # Partition fetched rows by kind, capping each at per_family_sql_limit.
                    kind_members: dict[str, list[SymbolRecord]] = {k: [] for k in batch_kinds}
                    for row in rows:
                        k = row["kind"].lower()
                        bucket = kind_members.get(k)
                        if bucket is not None and len(bucket) < per_family_sql_limit:
                            bucket.append(_row_to_symbol(row))
                    for kind in batch_kinds:
                        if len(additions) >= _EXPLORE_FAMILY_TOTAL_CAP:
                            break
                        members = kind_members[kind]
                        if len({m.symbol_id for m in members}) < _SKELETON_MIN_FAMILY:
                            continue
                        added = 0
                        for member in members:
                            if added >= _EXPLORE_FAMILY_PER_FAMILY_CAP or len(additions) >= _EXPLORE_FAMILY_TOTAL_CAP:
                                break
                            if member.symbol_id in have_ids or member.symbol_id in seen_ids:
                                continue
                            if member.file_path in seed_set:
                                continue
                            if int(member.end_line) - int(member.start_line) < _SKELETON_MIN_BODY_LINES:
                                continue
                            seen_ids.add(member.symbol_id)
                            additions.append(member)
                            added += 1
        except (sqlite3.Error, OSError, ValueError):
            logging.exception("Recovered from broad exception handler")
            return []
        return additions

    def _select_skeleton_symbols(
        self,
        symbols: list[SymbolRecord],
        *,
        seed_set: set[str],
    ) -> tuple[set[str], dict[str, str]]:
        """Pick redundant sibling symbols to render signatures-only.

        Index-free: groups already-selected, non-seed symbols of the same kind
        by a shared name affix (>=4 chars, non-generic). A family needs >=3
        members; the highest-scored member stays full (the exemplar), the rest
        are skeletoned. Returns (skeleton_symbol_ids, symbol_id -> "affix:kind").
        """
        from collections import defaultdict

        candidates: list[SymbolRecord] = []
        for symbol in symbols:
            if symbol.file_path in seed_set:
                continue
            if (symbol.kind or "").lower() not in _SKELETON_KINDS:
                continue
            if int(symbol.end_line) - int(symbol.start_line) < _SKELETON_MIN_BODY_LINES:
                continue
            candidates.append(symbol)

        groups: dict[tuple[str, str], list[SymbolRecord]] = defaultdict(list)
        for symbol in candidates:
            kind = (symbol.kind or "").lower()
            for affix in self._skeleton_affixes(symbol.symbol_name or symbol.qualified_name):
                groups[(kind, affix)].append(symbol)

        assigned: set[str] = set()
        skeleton_ids: set[str] = set()
        families: dict[str, str] = {}
        for (kind, affix), members in sorted(groups.items()):
            fresh = {member.symbol_id: member for member in members if member.symbol_id not in assigned}
            if len(fresh) < _SKELETON_MIN_FAMILY:
                continue
            ordered = sorted(
                fresh.values(),
                key=lambda member: (-(member.score or 0.0), member.qualified_name or member.symbol_name or ""),
            )
            assigned.add(ordered[0].symbol_id)
            for member in ordered[1:]:
                assigned.add(member.symbol_id)
                skeleton_ids.add(member.symbol_id)
                families[member.symbol_id] = f"{affix}:{kind}"
        return skeleton_ids, families

    def _skeleton_affixes(self, name: str | None) -> list[str]:
        base = (name or "").split(".")[-1]
        raw: list[str] = []
        for snake in base.split("_"):
            if snake:
                raw.extend(_CAMEL_BOUNDARY_RE.split(snake))
        tokens = [token.lower() for token in raw if token]
        tokens = [token for token in tokens if len(token) >= 4 and token not in _SKELETON_STOPWORDS]
        if not tokens:
            return []
        affixes: list[str] = []
        if tokens[-1] not in affixes:
            affixes.append(tokens[-1])
        if tokens[0] not in affixes:
            affixes.append(tokens[0])
        return affixes

    @staticmethod
    def _signature_header_end(lines: list[str]) -> int:
        """Index of the line that ends a callable's signature header (``def ...:`` / ``{``)."""
        for index, line in enumerate(lines[:8]):
            stripped = line.rstrip()
            if stripped.endswith((":", "{", "=>")):
                return index
        return 0

    def _skeletonize_source(
        self,
        source: str,
        *,
        file_path: str,
        start_line: int,
        language: str | None,
        line_numbers: bool,
    ) -> str | None:
        """Render a symbol body as signature lines only (definitions kept, bodies dropped).

        Reuses tree-sitter definition tags so nested member signatures survive.
        Returns None when there is nothing meaningful to collapse.
        """
        lines = source.splitlines()
        if len(lines) < 2:
            return None
        from lemoncrow.infra.tree_sitter.tags import extract_tags_from_text

        try:
            tags = extract_tags_from_text(source, file_path, language or None)
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return None
        keep = {0}
        for tag in tags:
            if tag.kind == "definition" and 1 <= tag.line <= len(lines):
                keep.add(tag.line - 1)
        kept = sorted(index for index in keep if 0 <= index < len(lines))
        if len(kept) <= 1:
            # Flat callable (function/method with no nested defs): keep only the
            # signature header and elide the body. Containers already keep their
            # member definition lines above, so this only fires for callables.
            header_end = self._signature_header_end(lines)
            if header_end + 1 >= len(lines):
                return None
            kept = list(range(header_end + 1))
        rendered: list[str] = []
        previous: int | None = None
        for index in kept:
            if previous is not None and index > previous + 1:
                rendered.append("\t…")
            if line_numbers:
                rendered.append(f"{start_line + index}\t{lines[index]}")
            else:
                rendered.append(lines[index])
            previous = index
        return "\n".join(rendered)

    def _usage_item(self, reference: UsageReference, *, snippet_lines: int) -> dict[str, Any]:
        payload = reference.model_dump(mode="json", exclude_none=True)
        if snippet_lines > 0 and "snippet" not in payload:
            payload["snippet"] = self._reference_snippet(reference.file_path, reference.line, snippet_lines)
        return payload

    def _reference_snippet(self, file_path: str, line: int, snippet_lines: int) -> str:
        lines = self._read_file(file_path).splitlines()
        if not lines:
            return ""
        start = max(0, line - 1)
        end = min(len(lines), start + max(1, snippet_lines))
        return "\n".join(lines[start:end])

    def _build_usages_payload(
        self,
        *,
        target: dict[str, Any],
        items: list[dict[str, Any]],
        group_by: Literal["file", "caller", "none"],
        truncated: bool,
        ambiguity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        provenance_breakdown = self._provenance_breakdown(items)
        provenance = self._items_provenance(items) if items else str(target.get("provenance") or _LOCAL_PROVENANCE)
        payload: dict[str, Any] = {
            "target": self._usage_target_summary(target),
            "references": self._group_usages(items, group_by=group_by),
            "reference_count": len(items),
            "group_by": group_by,
            "truncated": truncated,
            "cache_hit": False,
            "provenance": provenance,
            "provenance_breakdown": provenance_breakdown,
        }
        if ambiguity is not None:
            payload["ambiguity"] = ambiguity
        return payload

    def _group_usages(
        self,
        items: list[dict[str, Any]],
        *,
        group_by: Literal["file", "caller", "none"],
    ) -> list[dict[str, Any]] | dict[str, list[dict[str, Any]]]:
        if group_by == "none":
            return items
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            if group_by == "caller":
                key = str(item.get("caller") or item["file_path"])
            else:
                key = str(item["file_path"])
            grouped.setdefault(key, []).append(item)
        return grouped

    def _usage_target_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "symbol_id",
            "symbol_name",
            "qualified_name",
            "file_path",
        ]
        return {key: payload[key] for key in keys if key in payload}

    def _usage_item_identity(self, item: dict[str, Any]) -> tuple[Any, ...]:
        confidence = item.get("confidence")
        normalized_confidence = round(float(confidence), 6) if isinstance(confidence, int | float) else None
        return (
            str(item.get("file_path") or ""),
            int(item.get("line") or 0),
            int(item.get("column") or 0),
            int(item.get("end_line") or 0),
            int(item.get("end_column") or 0),
            str(item.get("caller") or ""),
            str(item.get("provenance") or ""),
            str(item.get("edge_kind") or ""),
            normalized_confidence,
        )

    def _dedupe_usage_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
        for item in items:
            deduped[self._usage_item_identity(item)] = item
        return [deduped[key] for key in sorted(deduped.keys())]

    def _symbol_lookup_matches(
        self,
        *,
        query: str | None,
        symbol_name: str | None,
        qualified_name: str | None,
        file_path: str | None,
        kind: str | None,
        language: str | None,
        file_glob: str | None,
    ) -> list[SymbolRecord]:
        target_query = query or qualified_name or symbol_name
        if not target_query:
            return []
        candidates = self.search_symbols(
            target_query,
            limit=20,
            kind=kind,
            language=language,
            snippet="none",
            file_glob=file_glob,
            auto_index=False,
        )
        exact = [
            candidate
            for candidate in candidates
            if (candidate.qualified_name == target_query or candidate.symbol_name == target_query)
            and (file_path is None or candidate.file_path == file_path)
        ]
        matches = exact or candidates
        deduped = {candidate.symbol_id: candidate for candidate in matches}
        return sorted(
            deduped.values(),
            key=lambda candidate: (
                candidate.file_path,
                candidate.start_line,
                candidate.end_line,
                candidate.qualified_name,
                candidate.symbol_id,
            ),
        )

    def _ambiguity_metadata(self, *, operation_name: str, targets: list[dict[str, Any]]) -> dict[str, Any] | None:
        if len(targets) <= 1:
            return None
        return {
            "note": f"merged {len(targets)} matching symbols for {operation_name}",
            "merged_target_count": len(targets),
            "matches": [
                {
                    "symbol_id": str(target["symbol_id"]),
                    "qualified_name": str(target.get("qualified_name") or ""),
                    "symbol_name": str(target.get("symbol_name") or ""),
                    "file_path": str(target.get("file_path") or ""),
                    "start_line": int(target.get("start_line") or 0),
                    "provenance": str(target.get("provenance") or _LOCAL_PROVENANCE),
                }
                for target in targets[:10]
            ],
        }

    def _resolve_symbol_targets(
        self,
        *,
        operation_name: str,
        query: str | None,
        symbol_id: str | None,
        qualified_name: str | None,
        symbol_name: str | None,
        file_path: str | None,
        kind: str | None,
        language: str | None,
        file_glob: str | None,
    ) -> dict[str, Any]:
        if symbol_id or qualified_name or (symbol_name and file_path):
            try:
                target = self.get_symbol(
                    symbol_id=symbol_id,
                    qualified_name=qualified_name,
                    symbol_name=symbol_name,
                    file_path=file_path,
                    auto_index=False,
                )
            except LookupError:
                return {
                    "error": "symbol_not_found",
                    "message": "no matching symbol was found",
                    "cache_hit": False,
                    "provenance": _LOCAL_PROVENANCE,
                }
            return {"targets": [target], "ambiguity": None}
        target_query = query or qualified_name or symbol_name
        if not target_query:
            raise ValueError(f"query, symbol_id, qualified_name, or symbol_name is required for code {operation_name}")
        matches = self._symbol_lookup_matches(
            query=query,
            symbol_name=symbol_name,
            qualified_name=qualified_name,
            file_path=file_path,
            kind=kind,
            language=language,
            file_glob=file_glob,
        )
        if not matches:
            return {
                "error": "symbol_not_found",
                "message": "no matching symbol was found",
                "cache_hit": False,
                "provenance": _LOCAL_PROVENANCE,
            }
        targets: list[dict[str, Any]] = []
        for candidate in matches:
            with contextlib.suppress(LookupError):
                targets.append(self.get_symbol(symbol_id=candidate.symbol_id, auto_index=False))
        if not targets:
            return {
                "error": "symbol_not_found",
                "message": "no matching symbol was found",
                "cache_hit": False,
                "provenance": _LOCAL_PROVENANCE,
            }
        return {
            "targets": targets,
            "ambiguity": self._ambiguity_metadata(operation_name=operation_name, targets=targets),
        }

    def _resolve_symbol_target(
        self,
        *,
        operation_name: str,
        query: str | None,
        symbol_id: str | None,
        qualified_name: str | None,
        symbol_name: str | None,
        file_path: str | None,
        kind: str | None,
        language: str | None,
        file_glob: str | None,
    ) -> dict[str, Any]:
        if symbol_id or qualified_name or (symbol_name and file_path):
            try:
                return self.get_symbol(
                    symbol_id=symbol_id,
                    qualified_name=qualified_name,
                    symbol_name=symbol_name,
                    file_path=file_path,
                    auto_index=False,
                )
            except LookupError:
                return {
                    "error": "symbol_not_found",
                    "message": "no matching symbol was found",
                    "cache_hit": False,
                    "provenance": _LOCAL_PROVENANCE,
                }
        target_query = query or qualified_name or symbol_name
        if not target_query:
            raise ValueError(f"query, symbol_id, qualified_name, or symbol_name is required for code {operation_name}")
        deduped = self._symbol_lookup_matches(
            query=query,
            symbol_name=symbol_name,
            qualified_name=qualified_name,
            file_path=file_path,
            kind=kind,
            language=language,
            file_glob=file_glob,
        )
        if not deduped:
            return {
                "error": "symbol_not_found",
                "message": "no matching symbol was found",
                "cache_hit": False,
                "provenance": _LOCAL_PROVENANCE,
            }
        if len(deduped) > 1:
            return {
                "error": "disambiguation_required",
                "message": f"multiple symbols match the {operation_name} query",
                "matches": [
                    {
                        "symbol_id": candidate.symbol_id,
                        "qualified_name": candidate.qualified_name,
                        "symbol_name": candidate.symbol_name,
                        "file_path": candidate.file_path,
                        "start_line": candidate.start_line,
                        "provenance": candidate.provenance,
                    }
                    for candidate in deduped[:10]
                ],
                "cache_hit": False,
                "provenance": _LOCAL_PROVENANCE,
            }
        return self.get_symbol(symbol_id=deduped[0].symbol_id, auto_index=False)

    def _find_references_local(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
    ) -> list[UsageReference]:
        try:
            target = self._get_symbol_local(
                symbol_id=symbol_id,
                qualified_name=qualified_name,
                file_path=file_path,
                symbol_name=symbol_name,
            )
        except LookupError:
            target = self._get_symbol_local(
                qualified_name=qualified_name,
                file_path=file_path,
                symbol_name=symbol_name,
            )
        target_name = str(target["symbol_name"])
        target_file = str(target["file_path"])
        target_start = int(target["start_line"])
        target_end = int(target["end_line"])
        indexed = self._indexed_references_for_symbol(
            target_name=target_name,
            target_file=target_file,
            target_start=target_start,
            target_end=target_end,
        )
        if indexed:
            return indexed
        # References are indexed for every tree-sitter language at index time, so a miss
        # here means the symbol has no recorded references. Do NOT re-parse the whole repo
        # with tree-sitter at query time -- that is O(repo) and can segfault on huge or
        # generated files. Return empty; find_references() falls back to a cheap text scan.
        return []

    def _indexed_references_for_symbol(
        self,
        *,
        target_name: str,
        target_file: str,
        target_start: int,
        target_end: int,
    ) -> list[UsageReference]:
        with self._connect() as conn:
            self._init_schema(conn)
            rows = conn.execute(
                """
                SELECT file_path, line, column, end_column, enclosing_qualified_name, snippet
                FROM "references"
                WHERE repo_id = ? AND symbol_name = ?
                ORDER BY file_path, line, column
                LIMIT 1000
                """,
                (self.repo_id, target_name),
            ).fetchall()
        results: list[UsageReference] = []
        seen: set[tuple[str, int, int]] = set()
        for row in rows:
            file_path = str(row["file_path"])
            line = int(row["line"])
            column = int(row["column"])
            if file_path == target_file and target_start <= line <= target_end:
                continue
            key = (file_path, line, column)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                UsageReference(
                    file_path=file_path,
                    line=line,
                    column=column,
                    end_line=line,
                    end_column=int(row["end_column"]),
                    caller=cast(str | None, row["enclosing_qualified_name"]),
                    snippet=str(row["snippet"]),
                    provenance="local_index",
                )
            )
        return results

    def _find_callers_local(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
    ) -> list[CallGraphNode] | None:
        try:
            target = self._get_symbol_local(
                symbol_id=symbol_id,
                qualified_name=qualified_name,
                file_path=file_path,
                symbol_name=symbol_name,
            )
        except LookupError:
            return None
        target_name = str(target["symbol_name"])
        with self._connect() as conn:
            self._init_schema(conn)
            rows = conn.execute(
                """
                SELECT DISTINCT caller_symbol_name, caller_qualified_name, caller_file_path,
                       caller_start_line, caller_end_line
                FROM call_edges
                WHERE repo_id = ? AND callee_short_name = ?
                ORDER BY caller_file_path, caller_start_line
                LIMIT 1000
                """,
                (self.repo_id, target_name),
            ).fetchall()
        if not rows:
            return []
        # call_edges is denormalized (each row carries the caller's name/file/lines),
        # so we only need to recover symbol_id + kind. Do it with ONE batched join,
        # never a per-row lookup (that N+1 is catastrophic for high-fanout callees).
        keys = [(str(r["caller_file_path"]), int(r["caller_start_line"]), str(r["caller_symbol_name"])) for r in rows]
        hydrated: dict[tuple[str, int, str], sqlite3.Row] = {}
        placeholders = ",".join("(?,?,?)" for _ in keys)
        flat: list[Any] = [self.repo_id]
        for cf, cs, cn in keys:
            flat.extend((cf, cs, cn))
        with self._connect() as conn:
            self._init_schema(conn)
            for srow in conn.execute(
                f"SELECT symbol_id, file_path, start_line, end_line, symbol_name, qualified_name, kind "
                f"FROM symbols WHERE repo_id = ? AND (file_path, start_line, symbol_name) IN (VALUES {placeholders})",
                tuple(flat),
            ).fetchall():
                hydrated[(str(srow["file_path"]), int(srow["start_line"]), str(srow["symbol_name"]))] = srow
        nodes: list[CallGraphNode] = []
        for row in rows:
            cf = str(row["caller_file_path"])
            cs = int(row["caller_start_line"])
            cn = str(row["caller_symbol_name"])
            cq = str(row["caller_qualified_name"])
            srow = hydrated.get((cf, cs, cn))
            if srow is not None:
                nodes.append(
                    CallGraphNode(
                        symbol_id=str(srow["symbol_id"]),
                        symbol_name=str(srow["symbol_name"]),
                        qualified_name=str(srow["qualified_name"]),
                        file_path=str(srow["file_path"]),
                        kind=str(srow["kind"]),
                        start_line=int(srow["start_line"]),
                        end_line=int(srow["end_line"]),
                        provenance="local_index",
                    )
                )
            else:
                synthetic_id = "local-call::" + hashlib.sha1(f"{cf}:{cs}:{cq}".encode()).hexdigest()[:16]
                nodes.append(
                    CallGraphNode(
                        symbol_id=synthetic_id,
                        symbol_name=cn,
                        qualified_name=cq,
                        file_path=cf,
                        kind="function",
                        start_line=cs,
                        end_line=int(row["caller_end_line"]),
                        provenance="local_index",
                    )
                )
        return nodes

    # ------------------------------------------------------------------
    # G6 -- symbol-level call-graph centrality (with N16 cache guard)
    # ------------------------------------------------------------------

    def _symbol_centrality_map(self) -> dict[str, float]:
        """Symbol name -> normalized eigenvector centrality (0..1), cached by index
        version. Feeds the call-graph importance signal into search ranking so
        central core symbols outrank peripheral textual matches.

        Persisted to the ``centrality_map`` table (keyed by index_version) so a
        cold engine -- a server restart, a benchmark rerun, a parallel worker --
        loads the map instead of recomputing the O(edges) power iteration. A
        reindex bumps index_version, which invalidates the persisted rows."""
        version = self._current_index_version()
        cached = getattr(self, "_centrality_name_map", None)
        if cached is not None and cached[0] == version:
            return cached[1]
        loaded = self._load_centrality_map(version)
        if loaded is not None:
            self._centrality_name_map = (version, loaded)
            return loaded
        mapping: dict[str, float] = {}
        try:
            ranking = self.call_graph_centrality(limit=1_000_000).get("ranking", [])
            max_ev = max((float(item.get("eigenvector") or 0.0) for item in ranking), default=0.0) or 1.0
            for item in ranking:
                name = str(item.get("symbol") or "")
                if not name:
                    continue
                norm_ev = float(item.get("eigenvector") or 0.0) / max_ev
                for key in (name.lower(), name.split(".")[-1].split("::")[-1].lower()):
                    if norm_ev > mapping.get(key, 0.0):
                        mapping[key] = norm_ev
        except Exception:
            logging.exception("Recovered from broad exception handler")
        self._persist_centrality_map(version, mapping)
        self._centrality_name_map = (version, mapping)
        return mapping

    def _load_centrality_map(self, version: int) -> dict[str, float] | None:
        """Load the persisted centrality map for the current index_version, or None
        if absent (first run after an index, or a pre-persistence DB)."""
        try:
            with self._connect(readonly=True) as conn:
                rows = conn.execute(
                    "SELECT name_key, score FROM centrality_map WHERE repo_id = ? AND index_version = ?",
                    (self.repo_id, version),
                ).fetchall()
        except sqlite3.OperationalError:
            return None
        if not rows:
            return None
        return {str(row["name_key"]): float(row["score"]) for row in rows}

    def _persist_centrality_map(self, version: int, mapping: dict[str, float]) -> None:
        """Persist the centrality map (best-effort) so future cold engines skip the
        power iteration. Replaces any prior rows for this repo."""
        if not mapping:
            return
        try:
            with self._connect() as conn:
                self._init_schema(conn)
                conn.execute("DELETE FROM centrality_map WHERE repo_id = ?", (self.repo_id,))
                conn.executemany(
                    "INSERT OR REPLACE INTO centrality_map(repo_id, name_key, score, index_version) "
                    "VALUES (?, ?, ?, ?)",
                    [(self.repo_id, key, value, version) for key, value in mapping.items()],
                )
        except sqlite3.OperationalError:
            logging.exception("Recovered from broad exception handler")

    def call_graph_centrality(self, *, limit: int = 50, use_cache: bool = True) -> dict[str, Any]:
        """Rank the most important symbols by call-graph centrality.

        Reads the persisted ``call_edges`` graph for this repo and returns degree
        and (power-iteration) eigenvector centrality per symbol, most central
        first. ``index_version`` is included so callers can tell which graph
        snapshot produced a ranking.

        N16: results are cached keyed to ``(index_version, limit)``. Every
        reindex bumps ``index_version`` (see ``_bump_index_version``), which
        changes the key, so a graph mutation can never serve a stale ranking.
        """
        version = self._current_index_version()
        cache_key = (version, limit)
        if use_cache:
            with self._centrality_cache_lock:
                cached = self._centrality_cache.get(cache_key)
            if cached is not None:
                return dict(cached)
        with self._connect() as conn:
            self._init_schema(conn)
            rows = conn.execute(
                """
                SELECT caller_qualified_name, callee_name
                FROM call_edges
                WHERE repo_id = ?
                """,
                (self.repo_id,),
            ).fetchall()
            # Preprocessor macros (C/C++ #define) parse as ordinary call_expression
            # nodes -- BUG_ON/WARN_ON/ASSERT/likely/unlikely are indistinguishable
            # from real function calls at the tree-sitter level, so they flood in
            # as callees from every corner of the codebase. That fan-in is exactly
            # what centrality is supposed to reward, so an unfiltered graph ranks
            # macros as the most "important" symbols in the repo -- pure noise: a
            # macro invocation is not a call to importable, navigable code. Exclude
            # any callee whose name is defined via #define anywhere in this repo
            # (signature heuristic -- avoids a schema change; macros are typically
            # ALL_CAPS or short lowercase wrappers, name alone can't tell them apart
            # from real functions, but the #define signature can). Broad '#' prefix
            # fetch + Python regex, not a strict SQL LIKE '#define%': kernel-style
            # header-guard-nested macros are written '#  define foo(x)' (variable
            # whitespace after '#', e.g. every likely()/unlikely() in
            # linux/compiler.h) and a plain LIKE prefix silently misses every one.
            macro_rows = conn.execute(
                "SELECT DISTINCT symbol_name, signature FROM symbols WHERE repo_id = ? AND signature LIKE '#%'",
                (self.repo_id,),
            ).fetchall()
        macro_names = {str(row["symbol_name"]) for row in macro_rows if _MACRO_DEFINE_RE.match(str(row["signature"]))}
        edges = [
            (str(row["caller_qualified_name"]), str(row["callee_name"]))
            for row in rows
            if str(row["callee_name"]) not in macro_names
        ]
        result = compute_call_graph_centrality(edges, limit=limit)
        result["excluded_macro_callees"] = len(macro_names)
        result["index_version"] = version
        if use_cache:
            with self._centrality_cache_lock:
                # index_version only ever increases (bumped on reindex) and it's
                # part of cache_key, so an entry from an older version can never
                # be served again -- drop them here instead of retaining one
                # whole-repo centrality ranking per historical version for the
                # life of the process.
                stale = [key for key in self._centrality_cache if key[0] != version]
                for key in stale:
                    del self._centrality_cache[key]
                self._centrality_cache[cache_key] = dict(result)
        return result

    def _fallback_callers_from_references(
        self,
        *,
        target: dict[str, Any],
        limit: int,
    ) -> CallGraphTraversalResult:
        target_symbol_id = str(target["symbol_id"])
        target_file = str(target["file_path"])
        target_start = int(target["start_line"])
        target_end = int(target["end_line"])
        references = self.intel_store.find_references(
            symbol_id=target_symbol_id,
            qualified_name=str(target["qualified_name"]),
            file_path=target_file,
            symbol_name=str(target["symbol_name"]),
        )
        references = sorted(
            [*references, *self._cross_lang_usage_references(target)],
            key=lambda item: (item.file_path, item.line, item.column, item.provenance),
        )
        nodes_by_id: dict[str, CallGraphNode] = {}
        edges: list[CallGraphEdge] = []
        seen_edges: set[tuple[str, str, int]] = set()
        truncated = False
        for reference in references:
            if reference.file_path == target_file and target_start <= reference.line <= target_end:
                continue
            node = self._caller_node_from_reference(reference, target_symbol_id=target_symbol_id)
            if node is None:
                continue
            if node.symbol_id not in nodes_by_id:
                if len(nodes_by_id) >= limit:
                    truncated = True
                    continue
                nodes_by_id[node.symbol_id] = node
            edge_key = (node.symbol_id, target_symbol_id, 1)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(
                    CallGraphEdge(
                        caller_symbol_id=node.symbol_id,
                        callee_symbol_id=target_symbol_id,
                        depth=1,
                    )
                )
        ordered_nodes = sorted(nodes_by_id.values(), key=lambda item: (item.file_path, item.start_line, item.symbol_id))
        ordered_edges = sorted(edges, key=lambda item: (item.depth, item.caller_symbol_id, item.callee_symbol_id))
        if not ordered_edges:
            return CallGraphTraversalResult(
                nodes=[],
                edges=[],
                truncated=False,
                data_status="unavailable",
                message="routed call edge data is unavailable",
                snapshot=None,
            )
        return CallGraphTraversalResult(
            nodes=ordered_nodes,
            edges=ordered_edges,
            truncated=truncated,
            data_status="available",
            message="fallback caller graph derived from symbol references",
            snapshot=None,
        )

    def _caller_node_from_reference(
        self,
        reference: UsageReference,
        *,
        target_symbol_id: str,
    ) -> CallGraphNode | None:
        normalized_file = self._normalize_file_arg(reference.file_path)
        with self._connect() as conn:
            self._init_schema(conn)
            row = conn.execute(
                """
                SELECT *, NULL AS score FROM symbols
                WHERE repo_id = ? AND file_path = ? AND start_line <= ? AND end_line >= ?
                ORDER BY (end_line - start_line) ASC, start_line DESC
                LIMIT 1
                """,
                (self.repo_id, normalized_file, reference.line, reference.line),
            ).fetchone()
        if row is not None:
            symbol = _row_to_symbol(row)
            if symbol.symbol_id == target_symbol_id:
                return None
            return CallGraphNode(
                symbol_id=symbol.symbol_id,
                symbol_name=symbol.symbol_name,
                qualified_name=symbol.qualified_name,
                file_path=symbol.file_path,
                kind=symbol.kind,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                provenance=reference.provenance or symbol.provenance,
            )
        synthetic_seed = f"{normalized_file}:{reference.line}:{reference.column}:{reference.caller or ''}"
        synthetic_id = f"ref::{hashlib.sha1(synthetic_seed.encode('utf-8')).hexdigest()[:16]}"
        fallback_name = reference.caller or f"{Path(normalized_file).name}:{reference.line}"
        return CallGraphNode(
            symbol_id=synthetic_id,
            symbol_name=fallback_name,
            qualified_name=fallback_name,
            file_path=normalized_file,
            kind="reference",
            start_line=reference.line,
            end_line=reference.end_line,
            provenance=reference.provenance or "treesitter",
        )

    def _find_callees_local(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
    ) -> list[CallGraphNode] | None:
        try:
            target = self._get_symbol_local(
                symbol_id=symbol_id,
                qualified_name=qualified_name,
                file_path=file_path,
                symbol_name=symbol_name,
            )
        except LookupError:
            return None
        target_file = str(target["file_path"])
        target_start = int(target["start_line"])
        target_end = int(target["end_line"])
        with self._connect() as conn:
            self._init_schema(conn)
            rows = conn.execute(
                """
                SELECT DISTINCT callee_name
                FROM call_edges
                WHERE repo_id = ? AND caller_file_path = ?
                  AND caller_start_line = ? AND caller_end_line = ?
                ORDER BY callee_name
                """,
                (self.repo_id, target_file, target_start, target_end),
            ).fetchall()
        target_ext = Path(target_file).suffix
        target_is_python = target_ext == ".py"
        nodes_by_identity: dict[str, CallGraphNode] = {}
        for row in rows:
            callee_name = str(row["callee_name"])
            short_name = callee_name.rsplit(".", 1)[-1]
            matched = self._indexed_symbol_payloads_for_call_name(callee_name)
            # Keep only definitions in the caller's own language. A Python
            # function cannot call a TS/JS symbol that merely shares the short
            # name (e.g. `range`, `min`), so cross-language name collisions are
            # dropped rather than surfaced as bogus callees.
            same_lang = [p for p in matched if Path(str(p.get("file_path") or "")).suffix == target_ext]
            if same_lang:
                for payload in same_lang:
                    node = CallGraphNode(
                        symbol_id=str(payload["symbol_id"]),
                        symbol_name=str(payload["symbol_name"]),
                        qualified_name=str(payload["qualified_name"]),
                        file_path=str(payload["file_path"]),
                        kind=str(payload["kind"]),
                        start_line=int(payload["start_line"]),
                        end_line=int(payload["end_line"]),
                        provenance=str(payload.get("provenance") or "local_index"),
                    )
                    nodes_by_identity[node.symbol_id] = node
                continue
            # No same-language definition: for Python callers, builtins and
            # ubiquitous container methods have no navigable target and are pure
            # noise — skip them instead of emitting a synthetic reference node.
            if target_is_python and short_name in _PY_CALLEE_NOISE:
                continue
            synthetic_id = f"local-callee::{hashlib.sha1(callee_name.encode('utf-8')).hexdigest()[:16]}"
            if synthetic_id not in nodes_by_identity:
                nodes_by_identity[synthetic_id] = CallGraphNode(
                    symbol_id=synthetic_id,
                    symbol_name=short_name,
                    qualified_name=callee_name,
                    file_path=target_file,
                    kind="reference",
                    start_line=target_start,
                    end_line=target_end,
                    provenance="local_index",
                )
        return sorted(
            nodes_by_identity.values(),
            key=lambda item: (item.file_path, item.start_line, item.symbol_id),
        )

    def _indexed_symbol_payloads_for_call_name(self, call_name: str) -> list[dict[str, Any]]:
        short_name = call_name.rsplit(".", 1)[-1]
        with self._connect() as conn:
            self._init_schema(conn)
            rows = conn.execute(
                """
                SELECT *, NULL AS score FROM symbols
                WHERE repo_id = ? AND symbol_name = ?
                ORDER BY file_path, start_line, end_line, qualified_name, symbol_id
                """,
                (self.repo_id, short_name),
            ).fetchall()
        if not rows:
            return []
        symbols = [_row_to_symbol(row) for row in rows]
        short_suffix = f".{short_name}"
        ranked = sorted(
            symbols,
            key=lambda symbol: (
                0 if symbol.qualified_name == call_name else 1 if symbol.qualified_name.endswith(short_suffix) else 2,
                symbol.file_path,
                symbol.start_line,
                symbol.end_line,
                symbol.qualified_name,
                symbol.symbol_id,
            ),
        )
        deduped: dict[str, dict[str, Any]] = {}
        for symbol in ranked:
            deduped[symbol.symbol_id] = symbol.model_dump(mode="json", exclude_none=True)
        return list(deduped.values())

    def _call_graph_node_from_indexed_row(
        self,
        *,
        file_path: str,
        start_line: int,
        end_line: int,
        symbol_name: str,
        qualified_name: str,
    ) -> CallGraphNode:
        with self._connect() as conn:
            self._init_schema(conn)
            row = conn.execute(
                """
                    SELECT *, NULL AS score FROM symbols
                    WHERE repo_id = ? AND file_path = ? AND start_line = ? AND symbol_name = ?
                    LIMIT 1
                    """,
                (self.repo_id, file_path, start_line, symbol_name),
            ).fetchone()
        if row is not None:
            symbol = _row_to_symbol(row)
            return CallGraphNode(
                symbol_id=symbol.symbol_id,
                symbol_name=symbol.symbol_name,
                qualified_name=symbol.qualified_name,
                file_path=symbol.file_path,
                kind=symbol.kind,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                provenance="local_index",
            )
        synthetic_id = (
            f"local-call::{hashlib.sha1(f'{file_path}:{start_line}:{qualified_name}'.encode()).hexdigest()[:16]}"
        )
        return CallGraphNode(
            symbol_id=synthetic_id,
            symbol_name=symbol_name,
            qualified_name=qualified_name,
            file_path=file_path,
            kind="function",
            start_line=start_line,
            end_line=end_line,
            provenance="local_index",
        )

    def _parse_rg_output(self, output: str, *, limit: int) -> list[TextMatch]:
        matches: list[TextMatch] = []
        for line in output.splitlines():
            if len(matches) >= limit:
                break
            path_text, sep, rest = line.partition(":")
            if not sep:
                continue
            line_text, sep, rest = rest.partition(":")
            if not sep:
                continue
            column_text, sep, text = rest.partition(":")
            if not sep:
                continue
            with contextlib.suppress(ValueError):
                matches.append(
                    TextMatch(
                        file_path=self._normalize_file_arg(path_text),
                        line=int(line_text),
                        column=int(column_text),
                        text=text,
                    )
                )
        return matches

    def _retrieval_key_prefix(self) -> str:
        """Namespace every cache key by the runtime retrieval config that produced it.

        ``index_version``/code-fingerprint (in RetrievalCache.make_key) catch
        index and code changes, but NOT a runtime toggle of which channels are
        even consulted -- LEMONCROW_ZOEKT_MODE / LEMONCROW_EXPLORE_SEMANTIC /
        LEMONCROW_EXPLORE_LEXICAL, or which embedder is configured. Without this,
        a query answered once under one config (e.g. zoekt off) would silently
        serve that stale, narrower-channel payload to a later call made under a
        different config (zoekt on) against the same unchanged index -- exactly
        the scenario a multi-channel comparison run exercises on purpose, and a
        real session can hit too if these env vars are ever changed without a
        server restart. Returned as a literal prefix (not hashed into ``args``)
        so entries stay filterable/bulk-invalidatable by namespace and inspectable
        directly in the table. Cheap: four short strings, no per-call I/O.
        """
        embedder = getattr(self._semantic_ranker, "embedder", None)
        embedder_name = getattr(embedder, "name", "none") if self._semantic_ranker.available else "none"
        zoekt_mode = os.environ.get("LEMONCROW_ZOEKT_MODE", "") or "default"
        semantic = os.environ.get("LEMONCROW_EXPLORE_SEMANTIC", "1")
        lexical = os.environ.get("LEMONCROW_EXPLORE_LEXICAL", "1")
        return f"zoekt={zoekt_mode},semantic={semantic},lexical={lexical},embedder={embedder_name}"

    def _cache_get(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        return self._cache.get(
            tool_name=tool_name,
            args=args,
            index_version=self._current_index_version(),
            repo_id=self.repo_id,
            key_prefix=self._retrieval_key_prefix(),
        )

    def _cache_set(self, tool_name: str, args: dict[str, Any], payload: dict[str, Any]) -> None:
        self._cache.set(
            tool_name=tool_name,
            args=args,
            index_version=self._current_index_version(),
            repo_id=self.repo_id,
            payload=payload,
            key_prefix=self._retrieval_key_prefix(),
        )

    def _reindex_files(self, file_paths: list[str]) -> None:
        """Incrementally reindex only *file_paths* -- never a whole-repo rebuild.

        Called after an edit (or codemod) touches specific files. Deleting and
        re-extracting just those files keeps post-edit latency O(edited files).
        The previous implementation called ``self.index_repo()`` (force=True),
        which wiped every table and re-parsed the entire repo on every symbol
        edit -- minutes on large repos (sympy/django) for a one-file change.
        """
        rels: list[str] = []
        existing_paths: list[Path] = []
        seen: set[str] = set()
        for raw in file_paths:
            try:
                resolved = self._resolve_inside_repo(raw)
            except ValueError:
                continue
            rel = _safe_relpath(self.repo_root, resolved)
            if rel in seen:
                continue
            seen.add(rel)
            rels.append(rel)
            if resolved.is_file():
                existing_paths.append(resolved)
        if not rels:
            return

        def _reindex_locked() -> None:
            with self._index_write_lock(block=True) as acquired:
                if not acquired:
                    # Another process is rebuilding the index; it will pick up
                    # these files. Don't pile on a concurrent write.
                    return
                with self._connect() as conn:
                    self._init_schema(conn)
                    for rel in rels:
                        self._delete_file_index(conn, rel)
                    results = (
                        self._parallel_extract(existing_paths, total=len(existing_paths)) if existing_paths else []
                    )
                    if results:
                        self._apply_file_data_batch(conn, results)
                    self._bump_index_version(conn)

        if self._autosync_enabled:
            with self._db_lock:
                # _maybe_autosync_reindex holds _autosync_lock across a
                # subprocess.run(..., timeout=600); blocking on it here would
                # keep _db_lock hostage for up to 600s. Skip this query-time
                # freshness reindex when the lock isn't immediately available --
                # the in-flight autosync run will pick these files up.
                if not self._autosync_lock.acquire(blocking=False):
                    return
                try:
                    _reindex_locked()
                finally:
                    self._autosync_lock.release()
        else:
            with self._db_lock:
                _reindex_locked()

    def _current_index_version(self) -> int:
        if self._index_version_cached is not None:
            return self._index_version_cached
        with self._connect() as conn:
            self._init_schema(conn)
            row = conn.execute("SELECT value FROM engine_state WHERE key = 'index_version'").fetchone()
        version = int(row["value"]) if row is not None else 0
        self._index_version_cached = version
        return version

    def _index_snapshot(self) -> dict[str, Any]:
        with self._connect() as conn:
            self._init_schema(conn)
            file_count_row = conn.execute(
                "SELECT COUNT(*) AS count FROM files WHERE repo_id = ?", (self.repo_id,)
            ).fetchone()
            symbol_count_row = conn.execute(
                "SELECT COUNT(*) AS count FROM symbols WHERE repo_id = ?",
                (self.repo_id,),
            ).fetchone()
            import_count_row = conn.execute(
                "SELECT COUNT(*) AS count FROM imports WHERE repo_id = ?",
                (self.repo_id,),
            ).fetchone()
            indexed_at_row = conn.execute(
                "SELECT MAX(indexed_at) AS indexed_at FROM files WHERE repo_id = ?",
                (self.repo_id,),
            ).fetchone()
        files_indexed = int(file_count_row["count"]) if file_count_row is not None else 0
        symbols_indexed = int(symbol_count_row["count"]) if symbol_count_row is not None else 0
        imports_indexed = int(import_count_row["count"]) if import_count_row is not None else 0
        last_indexed_at = str(indexed_at_row["indexed_at"]) if indexed_at_row and indexed_at_row["indexed_at"] else None
        index_age_seconds: int | None = None
        if last_indexed_at:
            with contextlib.suppress(ValueError):
                parsed = datetime.fromisoformat(last_indexed_at.replace("Z", "+00:00"))
                index_age_seconds = max(0, int((datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds()))
        return {
            "files_indexed": files_indexed,
            "symbols_indexed": symbols_indexed,
            "imports_indexed": imports_indexed,
            "last_indexed_at": last_indexed_at,
            "index_age_seconds": index_age_seconds,
        }

    def _bump_index_version(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT value FROM engine_state WHERE key = 'index_version'").fetchone()
        current = int(row["value"]) if row is not None else 0
        next_version = current + 1
        conn.execute(
            """
            INSERT INTO engine_state(key, value)
            VALUES ('index_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(next_version),),
        )
        # N16: a reindex (this version bump) must not serve stale neighbours.
        # The cached HNSW graph is keyed to index_version, but drop it eagerly so
        # the next query rebuilds against the fresh vectors immediately.
        self._ann_symbol_index.invalidate()
        self._index_version_cached = next_version
        return next_version

    def _payload_tokens(self, payload: Any) -> int:
        return estimate_tokens(_canonical_json(payload))

    def _compute_total_tokens(self, payload: dict[str, Any]) -> int:
        total_tokens = 0
        while True:
            candidate = dict(payload)
            candidate["total_tokens"] = total_tokens
            measured = self._payload_tokens(candidate)
            if measured == total_tokens:
                return measured
            total_tokens = measured

    def _dedupe_search_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str, int, int, str]] = set()
        for item in items:
            key = (
                str(item.get("symbol_id") or ""),
                str(item.get("file_path") or ""),
                int(item.get("start_line") or 0),
                int(item.get("end_line") or 0),
                str(item.get("qualified_name") or item.get("symbol_name") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _prioritize_grounded_search_items(
        self,
        items: list[dict[str, Any]],
        *,
        seed_files: list[str],
    ) -> list[dict[str, Any]]:
        if not seed_files:
            return items
        seed_set = set(seed_files)
        indexed_items = list(enumerate(items))
        indexed_items.sort(
            key=lambda entry: (
                0 if str(entry[1].get("file_path") or "") in seed_set else 1,
                entry[0],
            )
        )
        return [item for _, item in indexed_items]

    def _compact_search_items(
        self,
        items: list[dict[str, Any]],
        *,
        scope: Literal["repo", "external", "deleted"],
    ) -> list[dict[str, Any]]:
        allowed_keys = _DELETED_SEARCH_COMPACT_DEFAULT_KEYS if scope == "deleted" else _SEARCH_COMPACT_DEFAULT_KEYS
        # For external scope "origin" is the load-bearing field that distinguishes
        # external symbols from repo symbols, so it must survive compaction.
        if scope == "external":
            allowed_keys = allowed_keys | {"origin", "repo_name"}
        compacted = [{key: value for key, value in item.items() if key in allowed_keys} for item in items]
        if scope == "repo":
            result: list[dict[str, Any]] = []
            for item in compacted:
                # For commit chunks, provenance and commit_sha must survive.
                if item.get("provenance") == "commit":
                    cleaned = {
                        k: v for k, v in item.items() if k not in _SEARCH_REPO_STRIP_ITEM_KEYS or k == "provenance"
                    }
                else:
                    cleaned = {k: v for k, v in item.items() if k not in _SEARCH_REPO_STRIP_ITEM_KEYS}
                # qualified_name adds no information when it is identical to symbol_name
                if cleaned.get("qualified_name") == cleaned.get("symbol_name"):
                    cleaned.pop("qualified_name", None)
                result.append(cleaned)
            return result
        return compacted

    def _should_force_search_compaction(
        self,
        *,
        scope: Literal["repo", "external", "deleted"],
        snippet: Literal["none", "head", "full"],
        limit: int,
    ) -> bool:
        return scope == "repo" and snippet == "head" and limit >= _SEARCH_SNIPPET_FORCE_COMPACT_LIMIT

    def _effective_budget_tokens(self, operation: str, requested_budget_tokens: int) -> int:
        requested = max(1, int(requested_budget_tokens))
        safety_max = _OPERATION_TOKEN_CAPS.get(operation, resolve_output_policy(operation).max_total_tokens)
        return min(requested, safety_max)

    def _items_provenance(self, items: list[dict[str, Any]]) -> str:
        provenances = [str(item.get("provenance")) for item in items if item.get("provenance")]
        if not provenances:
            return _LOCAL_PROVENANCE
        first = provenances[0]
        if all(provenance == first for provenance in provenances):
            return first
        return _LOCAL_PROVENANCE

    def _provenance_breakdown(self, items: list[dict[str, Any]]) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for item in items:
            provenance = str(item.get("provenance") or _LOCAL_PROVENANCE)
            breakdown[provenance] = breakdown.get(provenance, 0) + 1
        return breakdown

    def _finalize_packed_payload(
        self,
        payload: dict[str, Any],
        *,
        full_total_tokens: int,
        base_tokens_saved: int = 0,
    ) -> dict[str, Any]:
        finalized = dict(payload)
        # Compute total_tokens from a probe that omits tokens_saved entirely.
        # Including tokens_saved in the measurement creates a circular dependency:
        # tokens_saved changes the JSON size → changes total_tokens → changes
        # tokens_saved.  At digit-count boundaries (e.g. 99 → 100) the ceiling
        # estimator jumps by 1 token, so the fixed point X + f(X) = full_total
        # may not exist as an integer and the naive iteration oscillates forever.
        # Omitting tokens_saved from the probe breaks the cycle with at most a
        # 1-2 token error on an informational field -- acceptable.
        probe = {k: v for k, v in finalized.items() if k != "tokens_saved"}
        total_tokens = self._compute_total_tokens(probe)
        finalized["tokens_saved"] = max(base_tokens_saved, full_total_tokens - total_tokens)
        finalized["total_tokens"] = self._compute_total_tokens(finalized)
        return apply_field_name_shortening(finalized)

    def _fit_items_to_budget(
        self,
        items: list[dict[str, Any]],
        *,
        budget_tokens: int,
        essential_keys: list[str],
        optional_keys_in_drop_order: list[str],
        build_payload: Callable[[list[dict[str, Any]]], dict[str, Any]],
        enforce_protected_top_rank: bool = True,
    ) -> dict[str, Any]:
        minimal_items, _, _ = self._budget.pack(
            items,
            0,
            essential_keys=essential_keys,
            optional_keys_in_drop_order=optional_keys_in_drop_order,
        )
        protected_items = minimal_items[: min(PROTECTED_TOP_RANK, len(minimal_items))]
        protected_payload = build_payload(protected_items)
        if enforce_protected_top_rank and protected_items and protected_payload["total_tokens"] > budget_tokens:
            # Degrade gracefully: return the top-ranked item(s) even if over budget.
            # A slightly over-budget result is strictly better than a hard error with 0 items.
            return protected_payload

        best_payload = build_payload(minimal_items)
        if best_payload["total_tokens"] > budget_tokens:
            for end in range(len(minimal_items) - 1, -1, -1):
                candidate = build_payload(minimal_items[:end])
                if candidate["total_tokens"] <= budget_tokens:
                    return candidate
            return build_payload([])

        # Fast path: pack at the full budget_tokens in one shot.  When all optional
        # keys are retained (the common case: explore payload ~6K < 9K budget), this
        # is identical to the binary-search answer but costs 1 BudgetPacker call
        # instead of log2(budget_tokens)≈14.  Each saved iteration avoids a
        # build_payload→_finalize_packed_payload→_compute_total_tokens chain (~0.6ms),
        # saving ~8ms per explore call.
        fast_packed, _, _ = self._budget.pack(
            items,
            budget_tokens,
            essential_keys=essential_keys,
            optional_keys_in_drop_order=optional_keys_in_drop_order,
        )
        fast_candidate = build_payload(fast_packed)
        if fast_candidate["total_tokens"] <= budget_tokens:
            return fast_candidate

        low = 0
        high = max(0, budget_tokens)
        while low <= high:
            mid = (low + high) // 2
            packed_items, _, _ = self._budget.pack(
                items,
                mid,
                essential_keys=essential_keys,
                optional_keys_in_drop_order=optional_keys_in_drop_order,
            )
            candidate = build_payload(packed_items)
            if candidate["total_tokens"] <= budget_tokens:
                best_payload = candidate
                low = mid + 1
            else:
                high = mid - 1

        return best_payload

    def _pack_items_payload(
        self,
        items: list[dict[str, Any]],
        *,
        budget_tokens: int,
        essential_keys: list[str],
        optional_keys_in_drop_order: list[str],
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        extra = dict(extra_payload or {})
        provenance = self._items_provenance(items)
        provenance_breakdown = self._provenance_breakdown(items)
        include_provenance_breakdown = len(provenance_breakdown) > 1
        full_payload = {
            "items": items,
            "cache_hit": False,
            "provenance": provenance,
            **extra,
        }
        if include_provenance_breakdown:
            full_payload["provenance_breakdown"] = provenance_breakdown
        full_total_tokens = self._compute_total_tokens({**full_payload, "tokens_saved": 0})

        def build_payload(packed_items: list[dict[str, Any]]) -> dict[str, Any]:
            packed_provenance_breakdown = self._provenance_breakdown(packed_items)
            return self._finalize_packed_payload(
                {
                    "items": packed_items,
                    "cache_hit": False,
                    "provenance": provenance,
                    **extra,
                    **(
                        {"provenance_breakdown": packed_provenance_breakdown}
                        if len(packed_provenance_breakdown) > 1
                        else {}
                    ),
                },
                full_total_tokens=full_total_tokens,
            )

        packed = self._fit_items_to_budget(
            items,
            budget_tokens=budget_tokens,
            essential_keys=essential_keys,
            optional_keys_in_drop_order=optional_keys_in_drop_order,
            build_payload=build_payload,
        )
        return self._maybe_attach_overflow_metadata(
            packed_payload=packed,
            full_payload=full_payload,
            full_total_tokens=full_total_tokens,
            budget_tokens=budget_tokens,
        )

    def _pack_pattern_matches(
        self,
        result: PatternSearchResult,
        *,
        budget_tokens: int,
    ) -> dict[str, Any]:
        items = [match.to_dict() for match in result.matches]
        full_payload = {
            "matches": items,
            "truncated": bool(result.truncated),
            "total_matches": result.total_matches if result.total_matches is not None else len(items),
            "cache_hit": False,
            "provenance": "ast-grep",
        }
        full_total_tokens = self._compute_total_tokens({**full_payload, "tokens_saved": 0})

        def build_payload(packed_items: list[dict[str, Any]]) -> dict[str, Any]:
            truncated = bool(result.truncated) or len(packed_items) < len(items)
            return self._finalize_packed_payload(
                {
                    "matches": packed_items,
                    "truncated": truncated,
                    "total_matches": result.total_matches if result.total_matches is not None else len(items),
                    "cache_hit": False,
                    "provenance": "ast-grep",
                },
                full_total_tokens=full_total_tokens,
            )

        packed = self._fit_items_to_budget(
            items,
            budget_tokens=budget_tokens,
            essential_keys=_PATTERN_ESSENTIAL_KEYS,
            optional_keys_in_drop_order=_PATTERN_OPTIONAL_KEYS,
            build_payload=build_payload,
            enforce_protected_top_rank=False,
        )
        return self._maybe_attach_overflow_metadata(
            packed_payload=packed,
            full_payload=full_payload,
            full_total_tokens=full_total_tokens,
            budget_tokens=budget_tokens,
        )

    def _pack_pattern_rewrite(
        self,
        result: PatternRewriteResult,
        *,
        budget_tokens: int,
    ) -> dict[str, Any]:
        # `diff` is the one essential field, so the budget packer never drops it.
        # A repo-wide rewrite would otherwise emit an unbounded verbatim diff into
        # the model context; head+tail truncate so large rewrites stay bounded
        # while small previews pass through untouched. files_changed always lists
        # every affected file regardless of truncation.
        diff_lines = (result.diff or "").splitlines(keepends=True)
        diff_head, diff_tail = 170, 30
        if len(diff_lines) > diff_head + diff_tail:
            elided = len(diff_lines) - diff_head - diff_tail
            diff = (
                "".join(diff_lines[:diff_head])
                + f"... ({elided} more diff lines elided; see files_changed)\n"
                + "".join(diff_lines[-diff_tail:])
            )
        else:
            diff = result.diff
        return self._pack_single_payload(
            {
                "diff": diff,
                "files_changed": result.files_changed,
                "provenance": "ast-grep",
            },
            budget_tokens=budget_tokens,
            essential_keys=["diff", "files_changed", "provenance"],
            optional_keys_in_drop_order=[],
        )

    def _pack_single_payload(
        self,
        payload: dict[str, Any],
        *,
        budget_tokens: int,
        essential_keys: list[str],
        optional_keys_in_drop_order: list[str],
        base_tokens_saved: int = 0,
    ) -> dict[str, Any]:
        full_payload = dict(payload)
        full_payload.setdefault("cache_hit", False)
        full_payload.setdefault("provenance", _LOCAL_PROVENANCE)
        full_total_tokens = self._compute_total_tokens({**full_payload, "tokens_saved": max(0, base_tokens_saved)})
        minimal_payload = {key: full_payload[key] for key in essential_keys if key in full_payload}

        def build_payload(packed_items: list[dict[str, Any]]) -> dict[str, Any]:
            packed_payload = dict(packed_items[0]) if packed_items else dict(minimal_payload)
            packed_payload["cache_hit"] = False
            packed_payload["provenance"] = str(full_payload.get("provenance") or _LOCAL_PROVENANCE)
            return self._finalize_packed_payload(
                packed_payload,
                full_total_tokens=full_total_tokens,
                base_tokens_saved=base_tokens_saved,
            )

        packed = self._fit_items_to_budget(
            [full_payload],
            budget_tokens=budget_tokens,
            essential_keys=[*essential_keys, "cache_hit", "tokens_saved", "provenance"],
            optional_keys_in_drop_order=optional_keys_in_drop_order,
            build_payload=build_payload,
            enforce_protected_top_rank=False,
        )
        return self._maybe_attach_overflow_metadata(
            packed_payload=packed,
            full_payload=full_payload,
            full_total_tokens=full_total_tokens,
            budget_tokens=budget_tokens,
            base_tokens_saved=base_tokens_saved,
        )

    def _maybe_attach_overflow_metadata(
        self,
        *,
        packed_payload: dict[str, Any],
        full_payload: dict[str, Any],
        full_total_tokens: int,
        budget_tokens: int,
        base_tokens_saved: int = 0,
    ) -> dict[str, Any]:
        if full_total_tokens <= budget_tokens:
            return packed_payload
        if "error" in packed_payload:
            return packed_payload
        overflow_tokens = full_total_tokens - budget_tokens
        if overflow_tokens < _OVERFLOW_SPILL_MIN_EXCESS_TOKENS:
            return packed_payload
        packed_total_tokens = int(packed_payload.get("total_tokens", self._compute_total_tokens(packed_payload)))
        reduction_tokens = max(0, full_total_tokens - packed_total_tokens)
        if reduction_tokens < _OVERFLOW_SPILL_MIN_REDUCTION_TOKENS:
            return packed_payload
        overflow_meta = self._write_overflow_artifact(full_payload)
        with_meta = dict(packed_payload)
        with_meta["overflow"] = overflow_meta
        finalized = self._finalize_packed_payload(
            with_meta,
            full_total_tokens=full_total_tokens,
            base_tokens_saved=max(base_tokens_saved, int(packed_payload.get("tokens_saved", 0) or 0)),
        )
        if finalized.get("total_tokens", 0) > budget_tokens:
            return packed_payload
        return finalized

    def _write_overflow_artifact(self, payload: dict[str, Any]) -> dict[str, Any]:
        artifact_root = default_store_root() / "overflow" / "code"
        artifact_root.mkdir(parents=True, exist_ok=True)
        artifact_payload = self._prune_overflow_artifact_payload(payload)
        canonical = _canonical_json(artifact_payload)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        filename = f"{self.repo_id}-{int(time.time() * 1000)}-{digest}.json"
        artifact_path = artifact_root / filename
        artifact_path.write_text(
            json.dumps(artifact_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "spilled": True,
            "artifact_path": str(artifact_path),
            "artifact_format": "json",
        }

    def _prune_overflow_artifact_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        artifact_payload = cast(dict[str, Any], self._json_safe(payload))
        for key in (
            "tokens_saved",
            "total_tokens",
            "cache_hit",
            "overflow",
            "rendered",
            "rendered_format",
        ):
            artifact_payload.pop(key, None)
        return artifact_payload

    def _mark_cache_hit(self, payload: dict[str, Any]) -> dict[str, Any]:
        cached = cast(dict[str, Any], json.loads(_canonical_json(payload)))
        cached["cache_hit"] = True
        cached["provenance"] = "cached"
        cached["total_tokens"] = self._compute_total_tokens(cached)
        return cached

    def _normalize_cache_tool(self, cache_tool: str | None) -> str | None:
        normalized = (cache_tool or "all").strip().lower()
        if normalized not in _CACHE_TOOL_ALIASES:
            choices = ", ".join(sorted(_CACHE_TOOL_ALIASES))
            raise ValueError(f"cache_tool must be one of: {choices}")
        return _CACHE_TOOL_ALIASES[normalized]

    def _attach_snippet(
        self,
        symbol: SymbolRecord,
        *,
        snippet: Literal["none", "head", "full"],
        snippet_lines: int,
    ) -> SymbolRecord:
        if snippet == "none":
            return symbol.model_copy(update={"snippet": None})
        symbol = self._refresh_stale_symbol(symbol)
        safe_line_count = max(1, snippet_lines)
        snippet_text = self._read_symbol_snippet(
            symbol.file_path,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            mode=snippet,
            snippet_lines=safe_line_count,
        )
        return symbol.model_copy(update={"snippet": snippet_text})

    def _refresh_stale_symbol(self, symbol: SymbolRecord) -> SymbolRecord:
        """Query-time freshness guard for the one file about to be rendered.

        ``_read_symbol_snippet`` always reads *current* bytes off disk, but it
        slices them using ``symbol.start_line``/``end_line`` from the index. If
        the file changed since it was last indexed -- edited outside
        ``lc.edit``, or the async post-edit reindex (``_reindex_edited_files``)
        hasn't landed yet -- those line numbers can point at the wrong region
        of the now-current file. Detect that with a single stat + indexed
        lookup, and if stale, incrementally reindex just this file (already
        the O(edited files) path used after every edit) before serving it.

        Bounded to the single file behind *symbol* -- never a repo-wide stat
        sweep (see ``_ensure_indexed``'s autosync-owns-that-tax comment).
        Fail-open: any I/O/DB error just returns *symbol* unchanged so a
        freshness check can never break a query.
        """
        try:
            resolved = self._resolve_inside_repo(symbol.file_path)
            if not resolved.is_file():
                return symbol
            disk_mtime_ns = resolved.stat().st_mtime_ns
        except (OSError, ValueError):
            return symbol
        try:
            with self._connect() as conn:
                self._init_schema(conn)
                row = conn.execute(
                    "SELECT mtime_ns FROM files WHERE repo_id = ? AND file_path = ?",
                    (self.repo_id, symbol.file_path),
                ).fetchone()
        except sqlite3.Error:
            return symbol
        indexed_mtime_ns = int(row["mtime_ns"]) if row is not None else None
        if indexed_mtime_ns == disk_mtime_ns:
            return symbol
        try:
            self._reindex_files([symbol.file_path])
        except Exception:
            logger.debug("query-time freshness reindex failed for %s", symbol.file_path, exc_info=True)
            return symbol
        try:
            with self._connect() as conn:
                self._init_schema(conn)
                fresh = conn.execute(
                    """
                    SELECT start_line, end_line FROM symbols
                    WHERE repo_id = ? AND file_path = ? AND qualified_name = ?
                    LIMIT 1
                    """,
                    (self.repo_id, symbol.file_path, symbol.qualified_name),
                ).fetchone()
        except sqlite3.Error:
            return symbol
        if fresh is None:
            return symbol
        return symbol.model_copy(update={"start_line": int(fresh["start_line"]), "end_line": int(fresh["end_line"])})

    def _read_symbol_snippet(
        self,
        file_path: str,
        *,
        start_line: int,
        end_line: int,
        mode: Literal["head", "full"],
        snippet_lines: int,
    ) -> str:
        try:
            resolved = self._resolve_inside_repo(file_path)
            lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        except (OSError, ValueError):
            # File deleted/unreadable (or resolved outside the repo) since it was
            # indexed; degrade to "" like the other readers rather than crashing
            # result rendering.
            return ""
        if not lines:
            return ""
        start_index = max(0, start_line - 1)
        max_end_index = min(len(lines), start_index + snippet_lines)
        if mode == "full":
            max_end_index = min(max_end_index, max(start_index + 1, end_line))
        snippet_slice = lines[start_index:max_end_index]
        return "\n".join(snippet_slice)

    def _normalize_files_path(self, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = self._normalize_file_arg(value).strip().strip("/")
        if normalized in {"", "."}:
            return None
        return normalized

    def _matches_files_filters(
        self,
        file_path: str,
        *,
        path: str | None,
        pattern: str | None,
        max_depth: int | None,
    ) -> bool:
        if path and file_path != path and not file_path.startswith(f"{path}/"):
            return False
        if pattern and not _matches_file_glob(file_path, pattern):
            return False
        if max_depth is None:
            return True
        if path and file_path == path:
            relative = ""
        elif path and file_path.startswith(f"{path}/"):
            relative = file_path[len(path) + 1 :]
        else:
            relative = file_path
        depth = relative.count("/") if relative else 0
        return depth <= max_depth

    def _indexed_file_records(
        self,
        *,
        path: str | None,
        pattern: str | None,
        max_depth: int | None,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            self._init_schema(conn)
            file_rows = conn.execute(
                """
                SELECT file_path, language
                FROM files
                WHERE repo_id = ?
                ORDER BY file_path
                """,
                (self.repo_id,),
            ).fetchall()
            symbol_count_rows = conn.execute(
                """
                SELECT file_path, COUNT(*) AS symbol_count
                FROM symbols
                WHERE repo_id = ?
                GROUP BY file_path
                """,
                (self.repo_id,),
            ).fetchall()
            top_symbol_rows = conn.execute(
                """
                SELECT file_path, symbol_name
                FROM (
                    SELECT
                        file_path,
                        symbol_name,
                        ROW_NUMBER() OVER (
                            PARTITION BY file_path
                            ORDER BY start_line, symbol_name
                        ) AS row_no
                    FROM symbols
                    WHERE repo_id = ?
                )
                WHERE row_no <= 3
                ORDER BY file_path, row_no
                """,
                (self.repo_id,),
            ).fetchall()
        symbol_counts = {str(row["file_path"]): int(row["symbol_count"]) for row in symbol_count_rows}
        top_symbols: dict[str, list[str]] = {}
        for row in top_symbol_rows:
            file_path = str(row["file_path"])
            top_symbols.setdefault(file_path, []).append(str(row["symbol_name"]))

        records: list[dict[str, Any]] = []
        for row in file_rows:
            file_path = str(row["file_path"])
            if not self._matches_files_filters(file_path, path=path, pattern=pattern, max_depth=max_depth):
                continue
            record = IndexedFileRecord(
                file_path=file_path,
                language=str(row["language"] or "unknown"),
                symbol_count=symbol_counts.get(file_path, 0),
                top_symbols=top_symbols.get(file_path, []),
            )
            records.append(record.model_dump(mode="json", exclude_none=True))
        return records

    def _indexed_route_records(
        self,
        *,
        file_glob: str | None,
        language: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], bool]:
        candidates = self._indexed_file_records(path=None, pattern=file_glob, max_depth=None)
        allowed_languages = {"python", "javascript", "typescript"}
        if language is not None and language not in allowed_languages:
            return [], False
        routes: list[dict[str, Any]] = []
        truncated = False
        for candidate in candidates:
            file_path = str(candidate.get("file_path") or "")
            file_language = str(candidate.get("language") or "unknown").lower()
            if file_language not in allowed_languages:
                continue
            if language is not None and file_language != language:
                continue
            for route in self._extract_routes_from_file(file_path=file_path, language=file_language):
                routes.append(route)
                if len(routes) >= limit:
                    truncated = True
                    break
            if truncated:
                break
        routes.sort(
            key=lambda item: (
                str(item.get("file_path")),
                int(item.get("line", 0)),
                str(item.get("route")),
            )
        )
        return routes[:limit], truncated

    def _extract_routes_from_file(
        self,
        *,
        file_path: str,
        language: str,
    ) -> list[dict[str, Any]]:
        with contextlib.suppress(OSError, UnicodeDecodeError):
            source = self._resolve_inside_repo(file_path).read_text(encoding="utf-8", errors="replace")
            lines = source.splitlines()
            records: list[RouteRecord] = []
            if language == "python":
                records.extend(self._extract_python_routes(file_path=file_path, lines=lines))
            elif language in {"javascript", "typescript"}:
                records.extend(self._extract_js_routes(file_path=file_path, lines=lines, language=language))
            return [record.model_dump(mode="json", exclude_none=True) for record in records]
        return []

    def _extract_python_routes(self, *, file_path: str, lines: list[str]) -> list[RouteRecord]:
        records: list[RouteRecord] = []
        for index, line in enumerate(lines, start=1):
            fastapi_match = _FASTAPI_DECORATOR_RE.search(line)
            if fastapi_match:
                method = fastapi_match.group("verb").upper()
                if method == "WEBSOCKET":
                    method = "WS"
                records.append(
                    RouteRecord(
                        framework="fastapi",
                        method=method,
                        route=fastapi_match.group("route"),
                        file_path=file_path,
                        line=index,
                        language="python",
                        handler=self._next_python_def_name(lines, index),
                        router=fastapi_match.group("router"),
                        provenance=_LOCAL_PROVENANCE,
                    )
                )
                continue
            fastapi_api_route_match = _FASTAPI_API_ROUTE_RE.search(line)
            if fastapi_api_route_match:
                route = fastapi_api_route_match.group("route")
                methods = self._parse_methods(fastapi_api_route_match.group("rest"))
                for method in methods:
                    records.append(
                        RouteRecord(
                            framework="fastapi",
                            method=method,
                            route=route,
                            file_path=file_path,
                            line=index,
                            language="python",
                            handler=self._next_python_def_name(lines, index),
                            router=fastapi_api_route_match.group("router"),
                            provenance=_LOCAL_PROVENANCE,
                        )
                    )
                continue
            flask_match = _FLASK_ROUTE_RE.search(line)
            if flask_match:
                route = flask_match.group("route")
                methods = self._parse_methods(flask_match.group("rest"))
                for method in methods:
                    records.append(
                        RouteRecord(
                            framework="flask",
                            method=method,
                            route=route,
                            file_path=file_path,
                            line=index,
                            language="python",
                            handler=self._next_python_def_name(lines, index),
                            router=flask_match.group("router"),
                            provenance=_LOCAL_PROVENANCE,
                        )
                    )
                continue
            flask_rule_match = _FLASK_ADD_URL_RULE_RE.search(line)
            if flask_rule_match:
                route = flask_rule_match.group("route")
                methods = self._parse_methods(flask_rule_match.group("rest"))
                handler = self._parse_python_handler(flask_rule_match.group("rest"))
                for method in methods:
                    records.append(
                        RouteRecord(
                            framework="flask",
                            method=method,
                            route=route,
                            file_path=file_path,
                            line=index,
                            language="python",
                            handler=handler,
                            router=flask_rule_match.group("router"),
                            provenance=_LOCAL_PROVENANCE,
                        )
                    )
                continue
            django_match = _DJANGO_PATH_RE.search(line)
            if django_match:
                records.append(
                    RouteRecord(
                        framework="django",
                        method="ANY",
                        route=django_match.group("route"),
                        file_path=file_path,
                        line=index,
                        language="python",
                        handler=django_match.group("handler"),
                        provenance=_LOCAL_PROVENANCE,
                    )
                )
                continue
            django_url_match = _DJANGO_URL_RE.search(line)
            if django_url_match:
                records.append(
                    RouteRecord(
                        framework="django",
                        method="ANY",
                        route=django_url_match.group("route"),
                        file_path=file_path,
                        line=index,
                        language="python",
                        handler=django_url_match.group("handler"),
                        provenance=_LOCAL_PROVENANCE,
                    )
                )
        return records

    def _extract_js_routes(self, *, file_path: str, lines: list[str], language: str) -> list[RouteRecord]:
        records: list[RouteRecord] = []
        for index, line in enumerate(lines, start=1):
            match = _EXPRESS_ROUTE_RE.search(line)
            if match:
                verb = match.group("verb").upper()
                method = "ANY" if verb in {"ALL", "USE"} else verb
                records.append(
                    RouteRecord(
                        framework="express",
                        method=method,
                        route=match.group("route"),
                        file_path=file_path,
                        line=index,
                        language=language,
                        handler=match.group("handler"),
                        router=match.group("router"),
                        provenance=_LOCAL_PROVENANCE,
                    )
                )
            chain_match = _EXPRESS_ROUTE_CHAIN_RE.search(line)
            if not chain_match:
                continue
            base_route = chain_match.group("route")
            chain = chain_match.group("chain") or ""
            for method_match in _EXPRESS_CHAIN_METHOD_RE.finditer(chain):
                verb = method_match.group("verb").upper()
                method = "ANY" if verb in {"ALL", "USE"} else verb
                records.append(
                    RouteRecord(
                        framework="express",
                        method=method,
                        route=base_route,
                        file_path=file_path,
                        line=index,
                        language=language,
                        handler=method_match.group("handler"),
                        router=chain_match.group("router"),
                        provenance=_LOCAL_PROVENANCE,
                    )
                )
        return records

    def _next_python_def_name(self, lines: list[str], decorator_line: int) -> str | None:
        for line in lines[decorator_line:]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("@"):
                continue
            match = re.match(r"^(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
            if match:
                return match.group(1)
            break
        return None

    def _parse_methods(self, value: str | None) -> list[str]:
        if not value:
            return ["GET"]
        methods = [match.group("method").upper() for match in _METHOD_LITERAL_RE.finditer(value)]
        return methods or ["GET"]

    def _parse_python_handler(self, value: str | None) -> str | None:
        if not value:
            return None
        named = re.search(r"view_func\s*=\s*([A-Za-z_][A-Za-z0-9_\.]*)", value)
        if named:
            return named.group(1)
        positional = re.search(r",\s*([A-Za-z_][A-Za-z0-9_\.]*)", value)
        if positional:
            return positional.group(1)
        return None

    def _files_flat(
        self,
        items: list[dict[str, Any]],
        *,
        include_metadata: bool,
    ) -> list[dict[str, Any]]:
        flat: list[dict[str, Any]] = []
        for item in items:
            entry: dict[str, Any] = {"file_path": str(item["file_path"])}
            if include_metadata:
                entry["language"] = str(item.get("language") or "unknown")
                entry["symbol_count"] = int(item.get("symbol_count") or 0)
                entry["top_symbols"] = list(item.get("top_symbols") or [])
            flat.append(entry)
        return flat

    def _files_grouped(
        self,
        items: list[dict[str, Any]],
        *,
        include_metadata: bool,
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            language = str(item.get("language") or "unknown")
            entry: dict[str, Any] = {"file_path": str(item["file_path"])}
            if include_metadata:
                entry["language"] = language
                entry["symbol_count"] = int(item.get("symbol_count") or 0)
                entry["top_symbols"] = list(item.get("top_symbols") or [])
            grouped.setdefault(language, []).append(entry)
        return grouped

    def _files_tree(
        self,
        items: list[dict[str, Any]],
        *,
        include_metadata: bool,
    ) -> dict[str, Any]:
        tree: dict[str, Any] = {}
        for item in items:
            parts = str(item["file_path"]).split("/")
            cursor: dict[str, Any] = tree
            for segment in parts[:-1]:
                child = cursor.get(segment)
                if not isinstance(child, dict):
                    child = {}
                    cursor[segment] = child
                cursor = child
            file_name = parts[-1]
            if include_metadata:
                cursor[file_name] = {
                    "language": str(item.get("language") or "unknown"),
                    "symbol_count": int(item.get("symbol_count") or 0),
                }
            else:
                cursor[file_name] = {}
        return tree

    def _format_files_payload(
        self,
        items: list[dict[str, Any]],
        *,
        format: Literal["tree", "flat", "grouped"],
        include_metadata: bool,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        if format == "flat":
            return self._files_flat(items, include_metadata=include_metadata)
        if format == "grouped":
            return self._files_grouped(items, include_metadata=include_metadata)
        return self._files_tree(items, include_metadata=include_metadata)

    def _build_files_payload(
        self,
        items: list[dict[str, Any]],
        *,
        path: str | None,
        pattern: str | None,
        format: Literal["tree", "flat", "grouped"],
        include_metadata: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "repo_root": str(self.repo_root),
            "path": path,
            "pattern": pattern,
            "format": format,
            "file_count": len(items),
            "files": self._format_files_payload(items, format=format, include_metadata=include_metadata),
            "truncated": truncated,
            "cache_hit": False,
            "provenance": _LOCAL_PROVENANCE,
        }

    def _build_routes_payload(
        self,
        items: list[dict[str, Any]],
        *,
        file_glob: str | None,
        language: str | None,
        truncated: bool,
    ) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "repo_root": str(self.repo_root),
            "file_glob": file_glob,
            "language": language,
            "route_count": len(items),
            "routes": items,
            "truncated": truncated,
            "cache_hit": False,
            "provenance": _LOCAL_PROVENANCE,
        }

    def _python_text_search(self, query: str, search_path: Path, *, limit: int, ignore_case: bool) -> list[TextMatch]:
        query_cmp = query.lower() if ignore_case else query
        paths = [search_path] if search_path.is_file() else iter_source_files(search_path)
        matches: list[TextMatch] = []
        for path in paths:
            rel = _safe_relpath(self.repo_root, path)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                # File deleted mid-scan (rg unavailable path); skip it instead of
                # aborting the whole text search, mirroring the fallback loop.
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                hay = line.lower() if ignore_case else line
                column = hay.find(query_cmp)
                if column >= 0:
                    matches.append(TextMatch(file_path=rel, line=line_no, column=column + 1, text=line))
                    if len(matches) >= limit:
                        return matches
        # Source-file search exhausted (or rg unavailable).  Walk remaining
        # non-source files so that prose/config/doc literals are discoverable
        # — mirrors what rg would return over the same directory.
        if not matches and not search_path.is_file():
            seen = {p.resolve() for p in paths}
            for path in _walk_non_source_files(search_path):
                if len(matches) >= limit:
                    break
                if path.resolve() in seen:
                    continue
                try:
                    rel = _safe_relpath(self.repo_root, path)
                    for line_no, line in enumerate(
                        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
                    ):
                        hay = line.lower() if ignore_case else line
                        column = hay.find(query_cmp)
                        if column >= 0:
                            matches.append(TextMatch(file_path=rel, line=line_no, column=column + 1, text=line))
                            if len(matches) >= limit:
                                return matches
                except (OSError, UnicodeDecodeError):
                    continue
        return matches

    def _sync_symbol_intel(self) -> None:
        self.intel_store.refresh()

    def _sync_external_artifact_state(self, state_key: str, signature: str) -> bool:
        with self._connect() as conn:
            self._init_schema(conn)
            row = conn.execute("SELECT value FROM engine_state WHERE key = ?", (state_key,)).fetchone()
            previous = str(row["value"]) if row is not None else None
            conn.execute(
                """
                INSERT INTO engine_state(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (state_key, signature),
            )
            if previous is not None and previous != signature:
                self._bump_index_version(conn)
                return True
        return False

    def _parse_autosync_debounce(self, raw_value: str | None) -> int:
        if raw_value is None:
            return 500
        with contextlib.suppress(ValueError):
            return max(50, int(raw_value))
        return 500

    def _autosync_status(self) -> dict[str, Any]:
        watcher_alive = self._file_watcher is not None and self._file_watcher.is_alive()
        return {
            "enabled": self._autosync_enabled,
            "state": self._autosync_state,
            "mode": "incremental" if self._autosync_enabled else "scaffold_only",
            "debounce_ms": self._autosync_debounce_ms,
            "poll_ms": self._autosync_poll_ms,
            "pending_events": self._autosync_pending_events,
            "last_event_at": self._autosync_last_event_at,
            "reindex_count": self._autosync_reindex_count,
            "history": list(self._autosync_history),
            "file_watcher": {
                "enabled": self._watcher_enabled,
                "alive": watcher_alive,
                "debounce_ms": self._watcher_debounce_ms,
                "gitignore_loaded": self._watcher_gitignore_spec is not None,
            },
        }

    def _source_tree_signature(self) -> str:
        parts: list[str] = []
        repo_root_str = str(self.repo_root)
        for path in iter_source_files(self.repo_root):
            with contextlib.suppress(OSError):
                stat = path.stat()
                # iter_source_files yields paths rooted at self.repo_root, so compute the
                # relative key with a pure-string op. _safe_relpath() would call realpath()
                # per file -- an O(files x path-depth) syscall storm that made this change
                # detector take minutes on large repos (e.g. VS Code).
                rel = os.path.relpath(path, repo_root_str)
                parts.append(f"{rel}|{stat.st_mtime_ns}|{stat.st_size}")
        digest_input = "\n".join(sorted(parts)).encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()

    def _run_index_subprocess(self, *, force: bool = False) -> bool:
        """Delegate index building to a fresh child process.

        Keeps the ProcessPoolExecutor and its gigabytes of CoW-forked heap out
        of the MCP / servicectl parent process. The subprocess runs
        ``lc code index``, acquires the SQLite write-lock independently,
        and exits — releasing all indexing memory on completion.

        Returns True on success, False on error (caller retries on next poll).
        """
        cmd = [
            sys.executable,
            "-m",
            "lemoncrow.gateway.cli",
            "code",
            "index",
            "--repo-root",
            str(self.repo_root),
            "--no-stats",
        ]
        # Pass a custom db-path when the engine was constructed with one so
        # the subprocess writes to the same SQLite file we read from.
        if self.db_path != _default_db_path(self.repo_root):
            cmd.extend(["--db-path", str(self.db_path)])
        if force:
            cmd.append("--reindex")
        try:
            env = os.environ.copy()
            if not env.get("LEMONCROW_INDEX_MAX_WORKERS", "").strip():
                env["LEMONCROW_INDEX_MAX_WORKERS"] = str(_resolve_autosync_index_max_workers())
            result = subprocess.run(cmd, capture_output=True, timeout=600, env=env)
            if result.returncode != 0:
                stderr_tail = result.stderr[-500:].decode("utf-8", errors="replace").strip()
                logging.warning(
                    "code index subprocess failed (rc=%d): %s",
                    result.returncode,
                    stderr_tail,
                )
                return False
            # Subprocess wrote a new index_version to SQLite; drop the two
            # in-process caches keyed on that version so the next read and any
            # ANN neighbour lookup both reflect the updated state.
            self._index_version_cached = None
            self._ann_vectors_cache = None
            return True
        except Exception:
            logging.exception("code index subprocess error for %s", self.repo_root)
            return False

    def _maybe_autosync_reindex(self, *, _from_watcher: bool = False) -> None:
        if not self._autosync_lock.acquire(blocking=False):
            return
        try:
            self._maybe_autosync_reindex_locked(_from_watcher=_from_watcher)
        finally:
            self._autosync_lock.release()

    def _maybe_autosync_reindex_locked(self, *, _from_watcher: bool = False) -> None:
        # When called from the file watcher we already know a change happened,
        # so skip the expensive _source_tree_signature() stat walk entirely.
        if _from_watcher:
            self._autosync_state = "syncing"
            if not self._run_index_subprocess():
                # Reindex failed; leave the signature/pending state stale so the
                # next poll retries instead of recording a failed sync as done.
                self._autosync_state = "idle"
                self._record_autosync_event(event="reindex", reason="watcher_triggered", reindexed=False)
                return
            self._autosync_signature = self._source_tree_signature()
            self._autosync_last_sync_ms = int(time.time() * 1000)
            self._autosync_pending_events = 0
            self._autosync_state = "idle"
            self._autosync_reindex_count += 1
            self._record_autosync_event(event="reindex", reason="watcher_triggered", reindexed=True)
            return

        current_signature = self._source_tree_signature()
        if self._autosync_signature is None:
            self._autosync_signature = current_signature
            self._autosync_last_sync_ms = int(time.time() * 1000)
            self._autosync_state = "idle"
            self._record_autosync_event(event="bootstrap", reason="seed_signature", reindexed=False)
            return
        if current_signature == self._autosync_signature:
            self._autosync_state = "idle"
            self._autosync_pending_events = 0
            return
        now_ms = int(time.time() * 1000)
        self._autosync_last_event_at = datetime.now(UTC).isoformat()
        self._autosync_pending_events = max(1, self._autosync_pending_events + 1)
        if now_ms - self._autosync_last_sync_ms < self._autosync_debounce_ms:
            self._autosync_state = "debouncing"
            self._record_autosync_event(event="change_detected", reason="within_debounce_window", reindexed=False)
            return
        self._autosync_state = "syncing"
        if not self._run_index_subprocess():
            # Reindex failed; leave the signature/pending state stale so the next
            # poll retries instead of recording a failed sync as complete.
            self._autosync_state = "idle"
            self._record_autosync_event(event="reindex", reason="source_signature_changed", reindexed=False)
            return
        self._autosync_signature = self._source_tree_signature()
        self._autosync_last_sync_ms = int(time.time() * 1000)
        self._autosync_pending_events = 0
        self._autosync_state = "idle"
        self._autosync_reindex_count += 1
        self._record_autosync_event(event="reindex", reason="source_signature_changed", reindexed=True)

    def _maybe_refresh_zoekt_index(self) -> None:
        """Keep the git-repo Zoekt shard fresh at commit granularity.

        Background-only. zoekt-git-index indexes committed git objects, so a
        working-tree edit can't change its content -- only a HEAD move
        (commit/checkout/merge) can. The refresh is inherently incremental and
        no-ops when HEAD is unchanged, so calling it each poll is cheap.
        """
        try:
            from lemoncrow.infra.code_intel.zoekt.adapter import get_zoekt_supervisor

            get_zoekt_supervisor(self.repo_root).refresh_index_if_head_changed()
        except (ImportError, OSError, ValueError):
            logging.debug("zoekt autosync refresh skipped", exc_info=True)

    def _parse_autosync_poll_ms(self, raw_value: str | None) -> int:
        if raw_value is None:
            return 10000
        with contextlib.suppress(ValueError):
            return max(1000, int(raw_value))
        return 10000

    def _start_autosync_worker(self) -> None:
        if self._autosync_thread is not None:
            return
        self._autosync_thread = threading.Thread(
            target=self._autosync_worker_loop,
            name=f"lemoncrow-code-autosync-{self.repo_id[:8]}",
            daemon=True,
        )
        self._autosync_thread.start()
        weakref.finalize(self._gc_sentinel, self._stop_autosync_worker)

    def _stop_autosync_worker(self) -> None:
        self._autosync_stop.set()
        self._stop_file_watcher()

    # --- File watcher (event-driven via watchdog) ---

    def _start_file_watcher(self) -> None:
        """Start a watchdog Observer that monitors the repo root for source-file changes."""
        if self._file_watcher is not None:
            return
        if Observer is None:
            self._watcher_enabled = False
            return
        try:
            # Load .gitignore patterns at startup so the handler can filter in-process.
            self._watcher_gitignore_spec = _watcher_load_gitignore_patterns(self.repo_root)
            self._watcher_gitignore_mtime = time.time()
            self._watcher_token = WeakRefToken(self)
            self._watcher_event_handler = _SourceFileEventHandler(self._watcher_token)
            self._file_watcher = Observer(timeout=0.5)
            self._file_watcher.schedule(
                self._watcher_event_handler,
                str(self.repo_root),
                recursive=True,
                event_filter=_WATCHER_EVENT_FILTER,
            )
            self._file_watcher.start()
            logger.debug(
                "File watcher started on %s (debounce=%dms, gitignore=%s)",
                self.repo_root,
                self._watcher_debounce_ms,
                "loaded" if self._watcher_gitignore_spec is not None else "none",
            )
        except OSError as exc:
            # Common: inotify instance limit (EMFILE/ENOSPC) in containers or
            # CI. Fall back to polling gracefully.
            logger.warning(
                "File watcher unavailable on %s (OSError: %s), falling back to polling",
                self.repo_root,
                exc,
            )
            self._file_watcher = None
            self._watcher_enabled = False
        except Exception:
            logger.exception("Failed to start file watcher on %s, falling back to polling", self.repo_root)
            self._file_watcher = None
            self._watcher_enabled = False

    def _stop_file_watcher(self) -> None:
        """Stop the watchdog Observer if running."""
        watcher = self._file_watcher
        if watcher is not None and watcher.is_alive():
            try:
                watcher.stop()
                watcher.join(timeout=3)
            except Exception:
                logger.exception("Error stopping file watcher")
        self._file_watcher = None
        self._watcher_event_handler = None
        self._watcher_gitignore_spec = None
        self._watcher_token = None  # break the engine<->token cycle promptly

    def _watcher_path_is_ignored(self, path: str) -> bool:
        """Check if *path* should be ignored by the file watcher.

        Fast path: check hard-skip dirs (.git, etc.) without any I/O.
        Slow path: consult the loaded .gitignore pathspec (reloaded periodically).
        """
        if _watcher_check_ignored_fast(path, _WATCHER_HARD_SKIP_DIRS):
            return True
        spec = self._watcher_gitignore_spec
        if spec is not None:
            try:
                rel = os.path.relpath(path, str(self.repo_root))
                if rel.startswith(".."):
                    return True  # outside repo root
                return spec.match_file(rel.replace("\\", "/"))
            except ValueError:
                return True
        # No gitignore spec: conservatively keep the file (reindex on any source change).
        return False

    def _watcher_reload_gitignore_if_stale(self) -> None:
        """Reload .gitignore patterns if any .gitignore file has changed on disk."""
        now = time.time()
        if now - self._watcher_gitignore_mtime < 30:
            return  # check at most every 30s
        self._watcher_gitignore_mtime = now
        # Quick check: has any .gitignore mtime changed?
        try:
            repo_root = self.repo_root
            for gi in repo_root.rglob(".gitignore"):
                current_mtime = gi.stat().st_mtime_ns
                if self._watcher_gi_mtimes.get(str(gi)) != current_mtime:
                    # At least one gitignore changed — full reload.
                    new_spec = _watcher_load_gitignore_patterns(repo_root)
                    if new_spec is not None:
                        self._watcher_gitignore_spec = new_spec
                        # Update cached mtimes
                        for gi2 in repo_root.rglob(".gitignore"):
                            try:
                                self._watcher_gi_mtimes[str(gi2)] = gi2.stat().st_mtime_ns
                            except OSError:
                                pass
                    return
        except (OSError, ValueError):
            logger.debug("gitignore reload check failed", exc_info=True)

    def _notify_watcher_event(self) -> None:
        """Called by the watchdog event handler when a source file changes.

        Schedules a debounced reindex via the existing autosync machinery.
        """
        if not self._autosync_enabled:
            return
        # Check if .gitignore files have changed (cheap mtime probe every 30s).
        self._watcher_reload_gitignore_if_stale()
        # Respect the watcher-specific debounce window.
        now_ms = int(time.time() * 1000)
        last = self._watcher_last_event_ms
        if now_ms - last < self._watcher_debounce_ms:
            return  # still within debounce window
        self._watcher_last_event_ms = now_ms
        self._autosync_last_event_at = datetime.now(UTC).isoformat()
        self._autosync_pending_events = max(1, self._autosync_pending_events + 1)
        self._maybe_autosync_reindex(_from_watcher=True)

    def _parse_watcher_enabled(self, raw_value: str | None) -> bool:
        if raw_value is None:
            return Observer is not None  # default: enabled when watchdog is available
        return raw_value.strip().lower() not in {"0", "false", "no", "off"}

    def _parse_watcher_debounce(self, raw_value: str | None) -> int:
        if raw_value is None:
            return 2000  # default 2s, matching CodeGraph
        with contextlib.suppress(ValueError):
            return max(100, int(raw_value))
        return 2000

    def _autosync_worker_loop(self) -> None:
        try:
            self._deleted_history_adapter().start_background_warmup()
        except Exception:
            logging.exception("Failed to start background warmup")
        # Background-owned initial build: if nothing has populated the index yet
        # (no external prewarm / `lc code index`), build it here so the
        # first tool call hits a warm index instead of paying a cold build on the
        # request path.
        if not self.index_ready():
            try:
                self._run_index_subprocess()
            except Exception:
                logging.exception("autosync: initial index build failed")
        # When the file watcher is active, polling is a safety net only -- the
        # watcher handles real-time change detection. When it's absent (no
        # `watchdog` installed, EMFILE/inotify limit, etc.) polling is the
        # *only* detection path, but each poll still does a full source-tree
        # stat walk + two `git ls-files` subprocesses, so it must never run
        # hotter than the same 60s floor used for the watcher's safety net.
        poll_ms = max(self._autosync_poll_ms, 60000)
        while not self._autosync_stop.wait(poll_ms / 1000.0):
            try:
                if not self.index_ready():
                    # Still empty (e.g. the initial build lost an index-lock race
                    # with a concurrent prewarm). Keep retrying until it exists.
                    self._run_index_subprocess()
                else:
                    # Polling-based check is the safety net; skip when watcher is
                    # active (the watcher already triggers reindex on change).
                    if self._file_watcher is None or not self._file_watcher.is_alive():
                        self._maybe_autosync_reindex()
                self._maybe_refresh_zoekt_index()
            except Exception as exc:
                logging.exception("Recovered from broad exception handler")
                self._record_autosync_event(event="worker_error", reason=str(exc), reindexed=False)

    def _detected_repo_languages(self) -> frozenset[str]:
        """Lightweight language detection from file extensions in the symbol index."""
        ext_map = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
        }
        langs: set[str] = set()
        try:
            with self._connect(readonly=True) as conn:
                for row in conn.execute(
                    "SELECT file_path FROM files WHERE repo_id = ? LIMIT 2000",
                    (self.repo_id,),
                ):
                    ext = Path(row[0]).suffix.lower()
                    if ext in ext_map:
                        langs.add(ext_map[ext])
        except sqlite3.Error:
            pass
        return frozenset(langs)

    def _record_autosync_event(self, *, event: str, reason: str, reindexed: bool) -> None:
        entry = {
            "at": datetime.now(UTC).isoformat(),
            "event": event,
            "reason": reason,
            "reindexed": reindexed,
        }
        self._autosync_history.append(entry)
        if len(self._autosync_history) > 20:
            self._autosync_history = self._autosync_history[-20:]

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        return str(value)

    def _current_head_sha(self) -> str:
        from lemoncrow.pro.code_intel.git_history import require_pygit2

        pygit2 = require_pygit2()
        repo = pygit2.Repository(str(self.repo_root))
        return str(repo.revparse_single("HEAD").id)

    def _safe_current_head_sha(self) -> str | None:
        with contextlib.suppress(Exception):
            value = self._current_head_sha()
            if value is None:
                return None
            return str(value)
        return None

    def _lineage_embedder_metadata(self) -> tuple[str, int]:
        from lemoncrow.pro.code_intel.git_history.embedder import embedder_name, embedding_dim

        return embedder_name(), embedding_dim()

    def _persist_lineage_embedder_metadata(self, conn: sqlite3.Connection, *, name: str, dim: int) -> None:
        conn.executemany(
            "INSERT INTO engine_state(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [
                ("commit_lineage_embedder_name", name),
                ("commit_lineage_embedder_dim", str(dim)),
            ],
        )

    def _ensure_lineage_ready(self) -> None:
        """Start background lineage bootstrap if commit_chunks is empty or stale.

        Non-blocking: launches a daemon thread. Safe to call multiple times.

        Disabled by default: the commit-summary lineage walker makes one LLM
        summarize call per commit, which is prohibitively slow on deep-history
        repos (~26-40 min on VS Code). Opt in with LEMONCROW_LINEAGE_ENABLED=1.
        The graveyard walker (deleted/renamed symbols) is independent and stays on.
        """
        if os.getenv("LEMONCROW_LINEAGE_ENABLED") != "1":
            return
        with self._lineage_lock:
            if self._lineage_thread is not None:
                return
        current_head = self._safe_current_head_sha()
        if current_head is None:
            return
        current_embedder_name, current_embedder_dim = self._lineage_embedder_metadata()
        needs_update = False
        full_rebuild = False
        with contextlib.suppress(Exception), contextlib.closing(self._connect()) as conn:
            self._init_schema(conn)
            head_row = conn.execute("SELECT value FROM engine_state WHERE key = 'commit_lineage_head'").fetchone()
            previous_head = str(head_row["value"]) if head_row is not None else None
            embedder_name_row = conn.execute(
                "SELECT value FROM engine_state WHERE key = 'commit_lineage_embedder_name'"
            ).fetchone()
            stored_embedder_name = str(embedder_name_row["value"]) if embedder_name_row is not None else None
            embedder_dim_row = conn.execute(
                "SELECT value FROM engine_state WHERE key = 'commit_lineage_embedder_dim'"
            ).fetchone()
            stored_embedder_dim = int(embedder_dim_row["value"]) if embedder_dim_row is not None else None
            count_row = conn.execute("SELECT COUNT(*) AS n FROM commit_chunks").fetchone()
            chunk_count = int(count_row["n"]) if count_row is not None else 0
            stale_row = conn.execute(
                "SELECT COUNT(*) AS n FROM commit_chunks WHERE index_version < ?",
                (_LINEAGE_INDEX_VERSION,),
            ).fetchone()
            has_stale = stale_row is not None and int(stale_row["n"]) > 0
            metadata_changed = (
                stored_embedder_name != current_embedder_name or stored_embedder_dim != current_embedder_dim
            )
            has_lineage_state = (
                previous_head is not None or stored_embedder_name is not None or stored_embedder_dim is not None
            )
            full_rebuild = chunk_count > 0 and has_lineage_state and (has_stale or metadata_changed)
            if full_rebuild or previous_head != current_head or chunk_count == 0:
                needs_update = True
        if not needs_update:
            return
        with self._lineage_lock:
            # Re-check under the lock so two concurrent read tools cannot both pass
            # the initial guard and each spawn a bootstrap thread.
            if self._lineage_thread is not None:
                return
            self._lineage_rebuild_full = full_rebuild
            self._lineage_thread = threading.Thread(
                target=self._lineage_bootstrap_worker,
                name=f"lemoncrow-lineage-{self.repo_id[:8]}",
                daemon=True,
            )
        self._lineage_thread.start()

    def _lineage_bootstrap_worker(self) -> None:
        """Background thread: walk, summarise, embed, persist commit chunks."""
        try:
            with contextlib.closing(self._connect()) as conn:
                self._init_schema(conn)
                watermark_row = conn.execute(
                    "SELECT value FROM engine_state WHERE key = 'commit_lineage_watermark'"
                ).fetchone()
                since_sha = (
                    None
                    if self._lineage_rebuild_full
                    else (str(watermark_row["value"]) if watermark_row is not None else None)
                )
            self._walk_and_summarise(since_sha=since_sha, full_rebuild=self._lineage_rebuild_full)
        except Exception:
            logging.exception("Recovered from broad exception handler")
            logger.debug(
                "lineage bootstrap failed", exc_info=True
            )  # fail-open — lineage is additive, never blocks search
        finally:
            self._lineage_rebuild_full = False

    def _walk_and_summarise(self, *, since_sha: str | None, full_rebuild: bool = False) -> None:
        """Walk commits, summarise, embed, upsert to commit_chunks in batches of 50.

        Two-pass design: summarise all commits first (LLM), then embed all summaries
        (vector model). This avoids contention when both operations share the same
        backend (e.g. a local Ollama server that serialises requests).
        """
        from lemoncrow.pro.code_intel.git_history import require_pygit2
        from lemoncrow.pro.code_intel.git_history.embedder import embed_summary
        from lemoncrow.pro.code_intel.git_history.models import CommitSummary
        from lemoncrow.pro.code_intel.git_history.summarizer import (
            SummarizerError,
            summarize_commit,
        )
        from lemoncrow.pro.code_intel.git_history.walker import iter_commit_records

        def _get_diff_text(repo: Any, commit: Any) -> str:
            try:
                if not commit.parents:
                    return ""
                parent = commit.parents[0]
                diff = parent.tree.diff_to_tree(commit.tree)
                return diff.patch or ""
            except Exception:
                logging.exception("Recovered from broad exception handler")
                return ""

        pygit2 = require_pygit2()
        repo = pygit2.Repository(str(self.repo_root))

        # Pass 1: summarise all commits (LLM calls — no embedding yet)
        summaries: list[CommitSummary] = []
        for record in iter_commit_records(self.repo_root, since_sha=since_sha):
            try:
                commit_obj = repo.revparse_single(record.sha)
                diff_text = _get_diff_text(repo, commit_obj)
            except Exception:
                logging.exception("Recovered from broad exception handler")
                diff_text = ""

            try:
                summary = summarize_commit(record, diff_text=diff_text)
            except SummarizerError:
                continue
            except Exception:
                logging.exception("Recovered from broad exception handler")
                continue
            summaries.append(summary)

        # Pass 2: embed + persist (vector calls — LLM is now idle)
        batch: list[tuple[Any, ...]] = []
        rebuild_rows: list[tuple[Any, ...]] = []

        for summary in summaries:
            try:
                embedding_blob = embed_summary(summary)
            except Exception:
                logging.exception("Recovered from broad exception handler")
                embedding_blob = None

            row = (
                summary.sha,
                summary.author_date,
                json.dumps(summary.files_touched),
                None,  # symbols_touched — deferred to follow-up phase
                summary.summary,
                summary.summary_model,
                embedding_blob,
                _LINEAGE_INDEX_VERSION,
            )
            if full_rebuild:
                rebuild_rows.append(row)
                continue

            batch.append(row)

            if len(batch) >= 50:
                self._flush_commit_batch(batch, watermark_sha=batch[-1][0])
                batch.clear()

        if full_rebuild:
            watermark_sha = rebuild_rows[-1][0] if rebuild_rows else None
            self._replace_commit_chunks(rebuild_rows, watermark_sha=watermark_sha)
        elif batch:
            self._flush_commit_batch(batch, watermark_sha=batch[-1][0])

        current_head = self._safe_current_head_sha()
        if current_head:
            current_embedder_name, current_embedder_dim = self._lineage_embedder_metadata()
            with contextlib.closing(self._connect()) as conn:
                self._persist_lineage_embedder_metadata(conn, name=current_embedder_name, dim=current_embedder_dim)
                conn.execute(
                    "INSERT INTO engine_state(key, value) VALUES (?, ?)"
                    " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    ("commit_lineage_head", current_head),
                )
                conn.commit()

    def _replace_commit_chunks(self, rows: list[tuple[Any, ...]], *, watermark_sha: str | None) -> None:
        """Atomically replace commit lineage rows after a full rebuild completes."""
        with contextlib.closing(self._connect()) as conn:
            self._init_schema(conn)
            conn.execute("DELETE FROM commit_chunks")
            if rows:
                conn.executemany(
                    """INSERT OR REPLACE INTO commit_chunks
                       (commit_sha, author_date, files_touched, symbols_touched,
                        summary, summary_model, embedding, index_version)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
            if watermark_sha is None:
                conn.execute("DELETE FROM engine_state WHERE key = 'commit_lineage_watermark'")
            else:
                conn.execute(
                    "INSERT INTO engine_state(key, value) VALUES (?, ?)"
                    " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    ("commit_lineage_watermark", watermark_sha),
                )
            conn.commit()

    def _flush_commit_batch(self, batch: list[tuple[Any, ...]], *, watermark_sha: str) -> None:
        """Upsert a batch of commit chunks and advance the resume watermark."""
        with contextlib.closing(self._connect()) as conn:
            self._init_schema(conn)
            conn.executemany(
                """INSERT OR REPLACE INTO commit_chunks
                   (commit_sha, author_date, files_touched, symbols_touched,
                    summary, summary_model, embedding, index_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                batch,
            )
            conn.execute(
                "INSERT INTO engine_state(key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                ("commit_lineage_watermark", watermark_sha),
            )
            conn.commit()

    def _search_commit_chunks(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[SymbolRecord]:
        """Embed query and return top-limit commit chunks as SymbolRecord objects.

        Each result has provenance="commit" and commit_sha set.
        Applies LEMONCROW_LINEAGE_COMMIT_SCORE_PENALTY (default 0.1) to the score.
        Returns [] if commit_chunks is empty or embeddings unavailable.
        """
        from lemoncrow.infra.storage.vector import cosine_similarity
        from lemoncrow.pro.code_intel.git_history.embedder import decode_embedding

        query_vec: list[float] | None = None
        with contextlib.suppress(Exception):
            query_vec = self._semantic_ranker._embed_query(query)

        if not query_vec:
            return []

        rows: list[sqlite3.Row] = []
        with contextlib.suppress(Exception), contextlib.closing(self._connect()) as conn:
            self._init_schema(conn)
            rows = conn.execute(
                "SELECT commit_sha, author_date, files_touched, summary, summary_model, embedding "
                "FROM commit_chunks WHERE embedding IS NOT NULL "
                "ORDER BY author_date DESC LIMIT 2000"
            ).fetchall()

        if not rows:
            return []

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            try:
                stored_vec = decode_embedding(bytes(row["embedding"]))
                sim = cosine_similarity(query_vec, stored_vec)
                adjusted = sim - self._lineage_score_penalty
                scored.append((adjusted, row))
            except Exception:
                logging.exception("Recovered from broad exception handler")
                continue

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:limit]

        results: list[SymbolRecord] = []
        for score_val, row in top:
            try:
                files = json.loads(row["files_touched"]) if row["files_touched"] else []
                primary_file = files[0] if files else ""
                sha = str(row["commit_sha"])
                results.append(
                    SymbolRecord(
                        symbol_id=sha,
                        repo_id=self.repo_id,
                        file_path=primary_file,
                        language="",
                        symbol_name=sha[:8],
                        qualified_name=str(row["summary"])[:80],
                        kind="commit",
                        signature=str(row["summary"]),
                        start_byte=0,
                        end_byte=0,
                        start_line=0,
                        end_line=0,
                        content_hash=sha,
                        score=round(score_val, 4),
                        provenance="commit",
                        commit_sha=sha,
                    )
                )
            except Exception:
                logging.exception("Recovered from broad exception handler")
                continue
        return results

    def _deleted_history_adapter(self) -> DeletedHistorySearchAdapter:
        if self._deleted_history_search_adapter is None:
            from lemoncrow.pro.code_intel.git_history.adapter import DeletedHistorySearchAdapter

            self._deleted_history_search_adapter = DeletedHistorySearchAdapter(
                repo_root=self.repo_root,
                repo_id=self.repo_id,
                connection_factory=self.connection,
            )
        return self._deleted_history_search_adapter


__all__ = ["CodeContextEngine"]

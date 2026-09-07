"""Heuristics for classifying generated / scaffolding source files (N9).

Generated and scaffolding files (protobuf stubs, mocks, vendored or minified
bundles, ``*.generated.*`` outputs) are noise for retrieval: their symbols are
rarely the thing a developer is reasoning about, and they crowd out hand-written
code in "Related Symbols". This module centralises the path-based classification
so ranking and related-symbol selection can deprioritise them consistently.

Classification is purely path-based (no file reads) so it stays cheap enough to
run inside the ranking hot path.
"""

from __future__ import annotations

import functools
import re
from pathlib import PurePosixPath

# Suffix patterns that mark a file as machine-generated. Matched against the
# lower-cased basename so they are OS- and case-insensitive.
_GENERATED_SUFFIXES: tuple[str, ...] = (
    ".pb.go",  # protoc-gen-go
    "_pb2.py",  # python protobuf
    "_pb2_grpc.py",  # python grpc stubs
    ".pb.cc",
    ".pb.h",
    "_pb.js",
    "_pb.d.ts",
    ".min.js",  # minified bundles
    ".min.css",
    ".min.mjs",
    ".g.dart",  # dart codegen
    ".freezed.dart",
)

# Substrings that appear in generated/scaffolding basenames.
_GENERATED_BASENAME_SUBSTRINGS: tuple[str, ...] = (
    ".generated.",
    "_generated.",
    ".designer.",  # winforms/visual studio designer
)

# Path segments (directories) whose contents are conventionally generated,
# vendored, or mock scaffolding.
_GENERATED_DIR_SEGMENTS: frozenset[str] = frozenset(
    {
        "__generated__",
        "__mocks__",
        "generated",
        "node_modules",
        "vendor",
    }
)

_MOCK_BASENAME_RE = re.compile(r"(?:^|[._-])mocks?(?:[._-]|$)")


@functools.lru_cache(maxsize=65536)
def is_generated_path(path: str | None) -> bool:
    """Return True when ``path`` looks like a generated/scaffolding file.

    Conservative and path-only: it never reads file contents, so it is safe to
    call during ranking. Hand-written source must never be misclassified, so the
    heuristics target unambiguous machine-generated markers. Memoized: ranking
    calls this once per candidate ROW (thousands per query) while the distinct
    paths per process are few, and the result is a pure function of the path.
    """
    raw = (path or "").strip().replace("\\", "/")
    if not raw:
        return False
    posix = PurePosixPath(raw)
    name = posix.name.lower()
    if not name:
        return False
    if any(name.endswith(suffix) for suffix in _GENERATED_SUFFIXES):
        return True
    if any(token in name for token in _GENERATED_BASENAME_SUBSTRINGS):
        return True
    if _MOCK_BASENAME_RE.search(name):
        return True
    parts_lower = {segment.lower() for segment in posix.parts[:-1]}
    return bool(parts_lower & _GENERATED_DIR_SEGMENTS)

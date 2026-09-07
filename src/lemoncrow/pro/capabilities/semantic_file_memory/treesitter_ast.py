"""Tree-sitter based outline extractor for non-Python/TS languages.

Uses ``tree-sitter-language-pack`` (a maintained bundle of tree-sitter grammars
with pre-built wheels) to parse Go / Rust / Java / Ruby / C / C++. For each
language we declare which AST node kinds count as top-level declarations and
which are "containers" whose children's signatures should also be extracted
(e.g. Java class → method signatures).

Returns a plain text outline (signature lines, no bodies) suitable for the
smart_read "outline" payload. Languages not in the config table fall through
to the regex-based generic outline in capability.py.

Adding a new language is editing ``_LANG_CONFIG``:

    "kotlin": LangCfg(
        keep={"package_header", "import_list", "class_declaration", "function_declaration", ...},
        container={"class_declaration", "object_declaration"},
        member={"function_declaration", "property_declaration"},
        body_kinds={"class_body", "function_body"},
    ),
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

_logger = logging.getLogger(__name__)
_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LangCfg:
    """Per-language tree-sitter node-kind allowlists."""

    # Top-level node kinds we keep verbatim (entire byte range).
    # Use for short declarations: imports, package, use, type aliases, etc.
    keep_full: frozenset[str] = field(default_factory=frozenset)
    # Top-level node kinds we keep as signature-only (header lines up to body).
    keep_signature: frozenset[str] = field(default_factory=frozenset)
    # Container kinds whose member signatures we also recurse into.
    container: frozenset[str] = field(default_factory=frozenset)
    # Member kinds inside a container that we keep as signature-only.
    member: frozenset[str] = field(default_factory=frozenset)
    # Child node kinds that delimit the start of a body (used to trim signatures).
    body_kinds: frozenset[str] = field(default_factory=frozenset)
    # tree-sitter-language-pack key (defaults to the dict key).
    parser_name: str = ""
    # Transparent wrapper kinds we descend into recursively (no output of their
    # own). Used for grammars where declarations are buried in wrapper nodes.
    unwrap: frozenset[str] = field(default_factory=frozenset)
    # Data key/value kinds we keep as the first source line only (rstripped).
    keep_first_line: frozenset[str] = field(default_factory=frozenset)


_LANG_CONFIG: dict[str, LangCfg] = {
    "go": LangCfg(
        keep_full=frozenset(
            {
                "package_clause",
                "import_declaration",
                "const_declaration",
                "var_declaration",
                "type_declaration",
            }
        ),
        keep_signature=frozenset({"function_declaration", "method_declaration"}),
        body_kinds=frozenset({"block"}),
    ),
    "rust": LangCfg(
        keep_full=frozenset(
            {
                "use_declaration",
                "const_item",
                "static_item",
                "type_item",
                "macro_definition",
            }
        ),
        keep_signature=frozenset(
            {
                "function_item",
                "function_signature_item",
                "struct_item",
                "enum_item",
                "foreign_mod_item",
            }
        ),
        container=frozenset({"impl_item", "trait_item", "mod_item"}),
        member=frozenset({"function_item", "function_signature_item", "const_item"}),
        body_kinds=frozenset({"block", "declaration_list", "field_declaration_list", "enum_variant_list"}),
    ),
    "java": LangCfg(
        keep_full=frozenset({"package_declaration", "import_declaration"}),
        container=frozenset(
            {
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
                "annotation_type_declaration",
            }
        ),
        member=frozenset(
            {
                "method_declaration",
                "constructor_declaration",
                "field_declaration",
            }
        ),
        body_kinds=frozenset(
            {
                "class_body",
                "interface_body",
                "enum_body",
                "record_body",
                "annotation_type_body",
                "block",
            }
        ),
    ),
    "ruby": LangCfg(
        keep_full=frozenset({"call", "assignment"}),  # require/load + top-level constants
        container=frozenset({"class", "module"}),
        member=frozenset({"method", "singleton_method"}),
        body_kinds=frozenset({"body_statement"}),
    ),
    "c": LangCfg(
        keep_full=frozenset(
            {
                "preproc_include",
                "preproc_def",
                "preproc_function_def",
                "declaration",
                "type_definition",
                "enum_specifier",
            }
        ),
        keep_signature=frozenset({"function_definition"}),
        container=frozenset({"struct_specifier", "union_specifier"}),
        member=frozenset({"field_declaration"}),
        body_kinds=frozenset({"compound_statement", "field_declaration_list"}),
    ),
    "cpp": LangCfg(
        keep_full=frozenset(
            {
                "preproc_include",
                "preproc_def",
                "declaration",
                "type_definition",
                "using_declaration",
                "namespace_alias_definition",
                "alias_declaration",
                "enum_specifier",
            }
        ),
        keep_signature=frozenset({"function_definition", "template_declaration"}),
        container=frozenset(
            {
                "struct_specifier",
                "class_specifier",
                "union_specifier",
                "namespace_definition",
            }
        ),
        member=frozenset({"field_declaration", "function_definition"}),
        body_kinds=frozenset({"compound_statement", "field_declaration_list", "declaration_list"}),
    ),
    "csharp": LangCfg(
        keep_full=frozenset(
            {
                "using_directive",
                "file_scoped_namespace_declaration",
                "global_attribute",
                "global_statement",
            }
        ),
        container=frozenset(
            {
                "namespace_declaration",
                "class_declaration",
                "interface_declaration",
                "struct_declaration",
                "enum_declaration",
                "record_declaration",
                "delegate_declaration",
            }
        ),
        member=frozenset(
            {
                "method_declaration",
                "constructor_declaration",
                "property_declaration",
                "field_declaration",
                "event_declaration",
                "indexer_declaration",
            }
        ),
        body_kinds=frozenset({"declaration_list", "block", "enum_member_declaration_list"}),
    ),
    "kotlin": LangCfg(
        keep_full=frozenset({"package_header", "import_list", "type_alias"}),
        keep_signature=frozenset({"function_declaration"}),
        container=frozenset(
            {
                "class_declaration",
                "object_declaration",
                "companion_object",
            }
        ),
        member=frozenset(
            {
                "function_declaration",
                "property_declaration",
                "secondary_constructor",
                "primary_constructor",
            }
        ),
        body_kinds=frozenset({"class_body", "function_body"}),
    ),
    "php": LangCfg(
        keep_full=frozenset(
            {
                "namespace_definition",
                "namespace_use_declaration",
                "const_declaration",
            }
        ),
        keep_signature=frozenset({"function_definition"}),
        container=frozenset(
            {
                "class_declaration",
                "interface_declaration",
                "trait_declaration",
                "enum_declaration",
            }
        ),
        member=frozenset(
            {
                "method_declaration",
                "property_declaration",
                "const_declaration",
            }
        ),
        body_kinds=frozenset({"declaration_list", "compound_statement"}),
    ),
    "swift": LangCfg(
        keep_full=frozenset(
            {
                "import_declaration",
                "typealias_declaration",
            }
        ),
        keep_signature=frozenset({"function_declaration"}),
        container=frozenset(
            {
                "class_declaration",
                "protocol_declaration",
                "struct_declaration",
                "enum_declaration",
                "extension_declaration",
            }
        ),
        member=frozenset(
            {
                "function_declaration",
                "init_declaration",
                "property_declaration",
                "protocol_property_declaration",
            }
        ),
        body_kinds=frozenset({"class_body", "protocol_body", "function_body", "enum_class_body"}),
    ),
    "scala": LangCfg(
        keep_full=frozenset(
            {
                "package_clause",
                "import_declaration",
                "val_definition",
                "var_definition",
                "type_definition",
            }
        ),
        keep_signature=frozenset({"function_definition"}),
        container=frozenset(
            {
                "class_definition",
                "object_definition",
                "trait_definition",
            }
        ),
        member=frozenset(
            {
                "function_definition",
                "val_definition",
                "var_definition",
            }
        ),
        body_kinds=frozenset({"template_body", "block"}),
    ),
    "bash": LangCfg(
        keep_full=frozenset({"variable_assignment", "declaration_command"}),
        keep_signature=frozenset({"function_definition"}),
        body_kinds=frozenset({"compound_statement"}),
    ),
    "make": LangCfg(
        keep_first_line=frozenset({"variable_assignment", "rule"}),
    ),
    "toml": LangCfg(
        keep_full=frozenset({"pair"}),
        keep_first_line=frozenset({"table", "table_array_element"}),
    ),
    # sql — unwrap the `statement` wrapper; signature-trim table/function bodies.
    "sql": LangCfg(
        unwrap=frozenset({"statement"}),
        keep_signature=frozenset({"create_table", "create_view", "create_index", "create_function", "alter_table"}),
        body_kinds=frozenset({"column_definitions", "function_body", "create_query", "index_fields"}),
    ),
    # yaml — descend 3 wrapper levels, keep top-level mapping keys' first line only.
    "yaml": LangCfg(
        unwrap=frozenset({"stream", "document", "block_node", "block_mapping"}),
        keep_first_line=frozenset({"block_mapping_pair"}),
    ),
    # json — descend document→object, keep top-level pair first line (low value, guard-gated).
    "json": LangCfg(
        unwrap=frozenset({"document", "object"}),
        keep_first_line=frozenset({"pair"}),
    ),
    # JavaScript — ES modules and CommonJS class/function declarations.
    # IIFE-wrapped files (e.g. UMD bundles) produce no outline; guard falls through to full.
    "javascript": LangCfg(
        keep_full=frozenset({"import_statement"}),
        keep_first_line=frozenset({"lexical_declaration", "variable_declaration"}),
        keep_signature=frozenset({"function_declaration", "generator_function_declaration"}),
        container=frozenset({"class_declaration"}),
        member=frozenset({"method_definition", "field_definition"}),
        body_kinds=frozenset({"statement_block", "class_body"}),
        unwrap=frozenset({"export_statement"}),
    ),
    # TypeScript — export_statement wraps most top-level declarations; unwrap it
    # so the inner node is processed by keep_signature / container rules.
    # import_statement uses keep_first_line (not keep_full) because TS compiler
    # files have massive multi-import blocks that would bloat the outline.
    # lexical_declaration covers const/let; variable_declaration covers var.
    "typescript": LangCfg(
        keep_first_line=frozenset({"import_statement", "lexical_declaration", "variable_declaration"}),
        keep_signature=frozenset(
            {
                "function_declaration",
                "function_signature",
                "interface_declaration",
                "type_alias_declaration",
                "enum_declaration",
            }
        ),
        container=frozenset({"class_declaration"}),
        member=frozenset(
            {
                "method_definition",
                "method_signature",
                "property_signature",
                "public_field_definition",
            }
        ),
        body_kinds=frozenset({"statement_block", "class_body", "interface_body", "enum_body"}),
        unwrap=frozenset({"export_statement"}),
    ),
    # ── Lua ────────────────────────────────────────────────────────────────────
    # Outline shows all top-level function declarations (local, module-dot, and
    # colon-method forms) as signatures plus local variable first lines.
    "lua": LangCfg(
        keep_first_line=frozenset({"variable_declaration"}),
        keep_signature=frozenset({"function_declaration"}),
        body_kinds=frozenset({"block"}),
    ),
    # ── HTML ───────────────────────────────────────────────────────────────────
    # Flatten the element tree via unwrap so every opening/self-closing tag is
    # surfaced as a first-line snippet. doctype is shown verbatim.
    "html": LangCfg(
        keep_full=frozenset({"doctype"}),
        keep_first_line=frozenset({"start_tag", "self_closing_tag"}),
        unwrap=frozenset({"element", "script_element", "style_element"}),
    ),
    # ── CSS ────────────────────────────────────────────────────────────────────
    # Rule-sets and at-rules get signature-trimmed (selector/query kept, block
    # body replaced with { ... }). Import/charset/namespace kept verbatim.
    "css": LangCfg(
        keep_full=frozenset({"import_statement", "charset_statement", "namespace_statement"}),
        keep_signature=frozenset(
            {
                "rule_set",
                "media_statement",
                "keyframes_statement",
                "supports_statement",
            }
        ),
        body_kinds=frozenset({"block", "keyframe_block_list"}),
    ),
    # ── Markdown ───────────────────────────────────────────────────────────────
    # Headings (ATX and setext) and fenced code-block openers are surfaced;
    # paragraph prose is dropped. section wrappers are unwrapped so nested
    # headings at any depth are captured.
    "markdown": LangCfg(
        keep_first_line=frozenset({"atx_heading", "fenced_code_block"}),
        keep_full=frozenset({"setext_heading"}),
        unwrap=frozenset({"section"}),
    ),
}

# Languages that have a configured outliner. Imported by capability.py to gate
# the tree-sitter branch.
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(_LANG_CONFIG.keys())

_NON_DEFINITION_KEEP_FULL_KINDS: frozenset[str] = frozenset(
    {
        "import_declaration",
        "import_list",
        "namespace_use_declaration",
        "package_clause",
        "package_declaration",
        "package_header",
        "preproc_include",
        "using_declaration",
        "using_directive",
        "use_declaration",
    }
)


def _get_parser(lang: str) -> Any:
    # Hermetic/offline runs (codebench containers) cannot reach the grammar CDN;
    # a miss holds a process-wide download lock for the full global timeout and
    # stalls every indexer worker. Opt out instead of paying that hang.
    if os.environ.get("LEMONCROW_TREE_SITTER_OFF", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    try:
        from tree_sitter_language_pack import get_parser

        cfg = _LANG_CONFIG.get(lang)
        name = (cfg.parser_name if cfg else "") or lang
        # Parser instances are Rust-backed and unsendable; create them per call
        # instead of caching them across thread lifetimes.
        return get_parser(name)
    except Exception as exc:
        logging.exception("Recovered from broad exception handler")
        _logger.warning("tree-sitter parser unavailable for %s: %s", lang, exc)
        return None


def tree_sitter_parser(language: str) -> Any:
    """Return the configured parser for a language, or None when unavailable."""
    return _get_parser(language)


def supported_tree_sitter_languages() -> frozenset[str]:
    """Return languages with configured tree-sitter structural support."""
    return SUPPORTED_LANGUAGES


def definition_node_kinds(language: str) -> frozenset[str]:
    """Return node kinds that represent repo-map definitions for a language."""
    cfg = _LANG_CONFIG.get(language)
    if cfg is None:
        return frozenset()
    keep_full_definitions = cfg.keep_full - _NON_DEFINITION_KEEP_FULL_KINDS
    return frozenset(keep_full_definitions | cfg.keep_signature | cfg.container | cfg.member | cfg.keep_first_line)


def transparent_node_kinds(language: str) -> frozenset[str]:
    """Return wrapper node kinds that should be traversed without tagging."""
    cfg = _LANG_CONFIG.get(language)
    return cfg.unwrap if cfg is not None else frozenset()


def _node_attr(node: Any, name: str) -> Any:
    """Read a tree-sitter node attribute that might be exposed as method or value."""
    val = getattr(node, name, None)
    if val is None:
        return None
    return val() if callable(val) else val


def _child_count(node: Any) -> int:
    return int(_node_attr(node, "child_count"))


def _children(node: Any) -> list[Any]:
    n = _child_count(node)
    return [node.child(i) for i in range(n)]


def _byte_range(node: Any) -> tuple[int, int]:
    # The tree-sitter-language-pack binding exposes start_byte/end_byte as
    # methods and offers a ByteRange object that isn't subscriptable, so we
    # ignore byte_range and read the bytes directly via _node_attr.
    return int(_node_attr(node, "start_byte")), int(_node_attr(node, "end_byte"))


def _kind(node: Any) -> str:
    return str(_node_attr(node, "kind") or _node_attr(node, "type") or "")


def _signature_slice(source: bytes, node: Any, body_kinds: frozenset[str]) -> bytes:
    """Extract bytes from node start up to its body, or its end if no body found."""
    start, end = _byte_range(node)
    for child in _children(node):
        if _kind(child) in body_kinds:
            child_start, _ = _byte_range(child)
            return source[start:child_start].rstrip() + b" { ... }"
    return source[start:end].rstrip()


def _extract_member_signatures(container: Any, source: bytes, cfg: LangCfg, indent: str = "    ") -> list[bytes]:
    """Walk into a container and collect signature lines for its members."""
    out: list[bytes] = []
    # Find the body child (class_body, declaration_list, etc.) and iterate its children.
    body_node = None
    for child in _children(container):
        if _kind(child) in cfg.body_kinds:
            body_node = child
            break
    if body_node is None:
        return out
    for child in _children(body_node):
        kind = _kind(child)
        if kind not in cfg.member:
            continue
        sig = _signature_slice(source, child, cfg.body_kinds)
        # Indent each line for readability.
        sig_text = sig.decode("utf-8", errors="replace")
        first_line = sig_text.splitlines()[0] if sig_text else ""
        out.append((indent + first_line).encode("utf-8"))
    return out


def outline_text(language: str, source: str | bytes) -> str | None:
    """Build a structural skeleton for a supported language.

    Returns ``None`` when:
      * the language has no config, or
      * the parser is unavailable, or
      * parsing fails for any reason.

    Caller is responsible for falling back to the generic regex outline.
    """
    if isinstance(source, bytes):
        # Defensive: callers occasionally hand us raw bytes despite the str
        # contract. Normalize once here so `.encode("utf-8")` below operates
        # on str, not bytes (bytes has no `.encode`, so that would otherwise
        # raise).
        source = source.decode("utf-8", errors="replace")
    cfg = _LANG_CONFIG.get(language)
    if cfg is None:
        return None
    parser = _get_parser(language)
    if parser is None:
        return None
    try:
        source_bytes = source.encode("utf-8")
        try:
            tree = parser.parse(source_bytes)
        except TypeError:
            # tree-sitter binding versions disagree on the source type: some
            # want bytes, some want str. Retry the other.
            tree = parser.parse(source)
    except Exception as exc:
        logging.exception("Recovered from broad exception handler")
        _logger.warning("tree-sitter parse failed for %s: %s", language, exc)
        return None
    root = _node_attr(tree, "root_node")
    pieces: list[bytes] = []

    def visit(node: Any) -> None:
        for child in _children(node):
            kind = _kind(child)
            if kind in cfg.unwrap:
                # Transparent wrapper: descend without emitting anything.
                visit(child)
            elif kind in cfg.keep_full:
                start, end = _byte_range(child)
                pieces.append(source_bytes[start:end].rstrip())
            elif kind in cfg.keep_signature:
                pieces.append(_signature_slice(source_bytes, child, cfg.body_kinds))
            elif kind in cfg.keep_first_line:
                start, end = _byte_range(child)
                text = source_bytes[start:end].decode("utf-8", errors="replace")
                lines = text.splitlines()
                pieces.append(lines[0].rstrip().encode("utf-8") if lines else b"")
            elif kind in cfg.container:
                # Container declaration line + member signatures inside.
                header = _signature_slice(source_bytes, child, cfg.body_kinds)
                pieces.append(header)
                pieces.extend(_extract_member_signatures(child, source_bytes, cfg))
            # else: skip — and crucially do NOT recurse (preserves top-level-only output).

    visit(root)
    if not pieces:
        return None
    return b"\n".join(pieces).decode("utf-8", errors="replace")

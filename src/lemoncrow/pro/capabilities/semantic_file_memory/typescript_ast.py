"""TypeScript / JavaScript semantic analysis using regex-based heuristics.

A full TS compiler would be ideal, but a careful regex approach correctly
handles the vast majority of patterns found in real codebases.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, cast

from .models import FileOutline, ImportInfo, SymbolInfo, SymbolOutline

try:
    from tree_sitter_languages import get_parser
except ImportError:  # pragma: no cover - optional dependency fallback
    logging.getLogger(__name__).debug("tree_sitter_languages unavailable; using regex fallback", exc_info=True)
    get_parser = None

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Exported top-level symbols
_RE_EXPORT_FUNCTION = re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(<[^>]*>)?\s*\(", re.M)
_RE_EXPORT_ARROW = re.compile(
    r"^export\s+(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:<[^>]*>\s*)?\(", re.M
)
_RE_EXPORT_CLASS = re.compile(r"^export\s+(?:abstract\s+)?class\s+(\w+)", re.M)
_RE_EXPORT_INTERFACE = re.compile(r"^export\s+(?:default\s+)?interface\s+(\w+)", re.M)
_RE_EXPORT_TYPE = re.compile(r"^export\s+type\s+(\w+)\s*(?:<[^>]*>)?\s*=", re.M)
_RE_EXPORT_ENUM = re.compile(r"^export\s+(?:const\s+)?enum\s+(\w+)", re.M)
_RE_EXPORT_CONST = re.compile(r"^export\s+(?:const|let|var)\s+(\w+)", re.M)
_RE_EXPORT_DEFAULT = re.compile(r"^export\s+default\s+(\w+)", re.M)
_RE_NAMED_EXPORT = re.compile(r"^export\s*\{([^}]+)\}", re.M)

# Class members
_RE_CLASS_METHOD = re.compile(
    r"^\s{2,}(?:(?:public|private|protected|static|async|override|abstract)\s+)*"
    r"(?:async\s+)?(\w+)\s*(?:<[^>]*>)?\s*\(",
    re.M,
)
_RE_CLASS_PROP = re.compile(
    r"^\s{2,}(?:(?:public|private|protected|static|readonly|override|declare)\s+)+"
    r"(\w+)\s*(?:!)?\s*(?::\s*[^=;\n]+)?\s*(?:=|;|$)",
    re.M,
)

# Imports
_RE_IMPORT_FROM = re.compile(
    r"^import\s+(?:type\s+)?(?:\{([^}]+)\}|\*\s+as\s+(\w+)|(\w+))(?:,\s*\{([^}]+)\})?\s+" r"from\s+['\"](.*?)['\"](;?)",
    re.M,
)
_RE_REQUIRE = re.compile(r"(?:const|let|var)\s+(\w+)\s*=\s*require\(['\"](.*?)['\"]\)", re.M)
_RE_JSDOC = re.compile(r"/\*\*([\s\S]*?)\*/", re.M)
_RE_TOP_LEVEL_CONST_ARROW = re.compile(
    r"^(?:export\s+)?const\s+(\w+)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>",
    re.M,
)
_RE_CLASS_DECL = re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)\b", re.M)
_RE_METHOD_SIG = re.compile(
    r"^\s*(?:(?:public|private|protected|static|async|override|abstract|readonly)\s+)*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*(?::\s*[^{}]+)?\s*\{",
    re.M,
)


def _find_lineno(source: str, match_start: int) -> int:
    return source[:match_start].count("\n") + 1


def _strip_jsdoc(comment: str) -> str:
    """Extract first sentence from JSDoc."""
    lines = [ln.strip().lstrip("* ").strip() for ln in comment.splitlines()]
    text = " ".join(ln for ln in lines if ln and not ln.startswith("@"))
    return text[:200].split(".")[0].strip()


def analyze_typescript(source: str | bytes) -> tuple[list[SymbolInfo], list[ImportInfo], str]:
    """
    Analyze TypeScript/JavaScript source.

    Returns:
        symbols   - exported and class-member symbols
        imports   - parsed import statements
        summary   - compact stat string
    """
    if isinstance(source, bytes):
        # Defensive: some callers hand us raw bytes despite the str contract.
        # Normalize once here so both the tree-sitter and regex paths below
        # (neither of which accepts bytes) get consistent str input.
        source = source.decode("utf-8", errors="replace")
    symbols, imports, summary = _analyze_with_tree_sitter(source)
    if symbols or imports:
        return symbols, imports, summary
    return _analyze_with_regex(source)


def _analyze_with_tree_sitter(source: str) -> tuple[list[SymbolInfo], list[ImportInfo], str]:
    if get_parser is None:
        return [], [], "typescript_ast:fallback=regex"

    parser = _get_ts_parser()
    if parser is None:
        return [], [], "typescript_ast:fallback=regex"

    try:
        source_bytes = source.encode("utf-8", errors="replace")
        try:
            tree = parser.parse(source_bytes)
        except TypeError:
            # Some tree-sitter-language-pack grammars expect str, not bytes.
            tree = parser.parse(source)
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return [], [], "typescript_ast:fallback=regex"

    symbols: list[SymbolInfo] = []
    imports: list[ImportInfo] = []
    seen_symbols: set[str] = set()

    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.children))
        text = _node_text(source_bytes, node)
        lineno = int(node.start_point[0]) + 1 if hasattr(node, "start_point") else 1

        if node.type == "import_statement":
            module_match = re.search(r"from\s+['\"](.*?)['\"]", text)
            module = module_match.group(1) if module_match else ""
            names = _extract_import_names(text)
            if module:
                imports.append(
                    ImportInfo(
                        module=module,
                        names=names or ["*"],
                        lineno=lineno,
                        is_from=True,
                    )
                )
            continue

        if node.type in {
            "function_declaration",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
        }:
            name, kind = _node_name_and_kind(node.type, text)
            if name and name not in seen_symbols:
                seen_symbols.add(name)
                symbols.append(
                    SymbolInfo(
                        name=name,
                        kind=kind,
                        lineno=lineno,
                        signature=text.splitlines()[0][:200],
                        is_export=_is_export_node(node, source_bytes),
                    )
                )

    fn_count = sum(1 for s in symbols if s.kind in ("function",))
    cls_count = sum(1 for s in symbols if s.kind == "class")
    iface_count = sum(1 for s in symbols if s.kind in ("interface", "type", "enum"))
    imp_count = len(imports)
    summary = (
        f"typescript_ast:functions={fn_count};classes={cls_count};"
        f"interfaces={iface_count};imports={imp_count};engine=tree_sitter"
    )
    return symbols, imports, summary


def _analyze_with_regex(source: str) -> tuple[list[SymbolInfo], list[ImportInfo], str]:
    symbols: list[SymbolInfo] = []
    imports: list[ImportInfo] = []
    exports: set[str] = set()

    # --- Exports ---
    for pattern, kind in [
        (_RE_EXPORT_FUNCTION, "function"),
        (_RE_EXPORT_CLASS, "class"),
        (_RE_EXPORT_INTERFACE, "interface"),
        (_RE_EXPORT_TYPE, "type"),
        (_RE_EXPORT_ENUM, "enum"),
    ]:
        for m in pattern.finditer(source):
            name = m.group(1)
            if not name or name in exports:
                continue
            exports.add(name)
            lineno = _find_lineno(source, m.start())
            symbols.append(
                SymbolInfo(
                    name=name,
                    kind=kind,
                    lineno=lineno,
                    signature=m.group(0).rstrip(),
                    is_export=True,
                )
            )

    for m in _RE_EXPORT_ARROW.finditer(source):
        name = m.group(1)
        if name and name not in exports:
            exports.add(name)
            symbols.append(
                SymbolInfo(
                    name=name,
                    kind="function",
                    lineno=_find_lineno(source, m.start()),
                    signature=m.group(0).rstrip(),
                    is_export=True,
                )
            )

    for m in _RE_EXPORT_CONST.finditer(source):
        name = m.group(1)
        if name and name not in exports:
            exports.add(name)
            symbols.append(
                SymbolInfo(
                    name=name,
                    kind="variable",
                    lineno=_find_lineno(source, m.start()),
                    signature=m.group(0).rstrip(),
                    is_export=True,
                )
            )

    # Named re-exports: export { foo, bar as baz }
    for m in _RE_NAMED_EXPORT.finditer(source):
        for raw in m.group(1).split(","):
            name = raw.strip().split(" as ")[0].strip()
            if name:
                exports.add(name)

    # Default export
    for m in _RE_EXPORT_DEFAULT.finditer(source):
        name = m.group(1)
        if name and name not in exports:
            exports.add(name)

    # --- Imports ---
    for m in _RE_IMPORT_FROM.finditer(source):
        named_group, ns_group, default_group, extra_group, module = (
            m.group(1),
            m.group(2),
            m.group(3),
            m.group(4),
            m.group(5),
        )
        names: list[str] = []
        if named_group:
            names += [n.strip().split(" as ")[0].strip() for n in named_group.split(",") if n.strip()]
        if extra_group:
            names += [n.strip().split(" as ")[0].strip() for n in extra_group.split(",") if n.strip()]
        if ns_group:
            names.append(f"* as {ns_group}")
        if default_group:
            names.append(default_group)
        imports.append(
            ImportInfo(
                module=module,
                names=names or ["*"],
                lineno=_find_lineno(source, m.start()),
                is_from=True,
            )
        )

    for m in _RE_REQUIRE.finditer(source):
        imports.append(
            ImportInfo(
                module=m.group(2),
                names=[m.group(1)],
                lineno=_find_lineno(source, m.start()),
                is_from=False,
            )
        )

    fn_count = sum(1 for s in symbols if s.kind in ("function",))
    cls_count = sum(1 for s in symbols if s.kind == "class")
    iface_count = sum(1 for s in symbols if s.kind in ("interface", "type", "enum"))
    imp_count = len(imports)
    summary = (
        f"typescript_ast:functions={fn_count};classes={cls_count};"
        f"interfaces={iface_count};imports={imp_count};engine=regex"
    )
    return symbols, imports, summary


_TS_PARSER: Any | None = None


def _get_ts_parser() -> Any | None:
    global _TS_PARSER
    if _TS_PARSER is not None:
        return _TS_PARSER
    if get_parser is None:
        return None
    for lang in ("typescript", "tsx", "javascript"):
        try:
            _TS_PARSER = get_parser(lang)
            if _TS_PARSER is not None:
                return _TS_PARSER
        except Exception:
            logging.exception("Recovered from broad exception handler")
            continue
    return None


def _node_text(source_bytes: bytes, node: Any) -> str:
    try:
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return ""


def _extract_import_names(text: str) -> list[str]:
    names: list[str] = []
    named = re.search(r"\{([^}]+)\}", text)
    if named:
        names.extend(n.strip().split(" as ")[0].strip() for n in named.group(1).split(",") if n.strip())
    default_match = re.match(r"import\s+(\w+)\s*(,|from)", text)
    if default_match:
        names.append(default_match.group(1))
    ns_match = re.search(r"\*\s+as\s+(\w+)", text)
    if ns_match:
        names.append(f"* as {ns_match.group(1)}")
    return names


def _node_name_and_kind(node_type: str, text: str) -> tuple[str, str]:
    mapping = {
        "function_declaration": "function",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
    }
    kind = mapping.get(node_type, "symbol")
    m = re.search(r"\b(?:function|class|interface|type|enum)\s+(\w+)", text)
    return (m.group(1) if m else "", kind)


def _is_export_node(node: Any, source_bytes: bytes) -> bool:
    current = node
    while current is not None:
        if getattr(current, "type", "") == "export_statement":
            return True
        current = getattr(current, "parent", None)
    text = _node_text(source_bytes, node)
    return text.lstrip().startswith("export ")


def outline(path: str, source: str | bytes, *, lang: str = "typescript") -> FileOutline:
    """Return a compact TS/JS outline (classes/functions/methods + imports)."""
    if isinstance(source, bytes):
        # Defensive: some callers hand us raw bytes despite the str contract.
        source = source.decode("utf-8", errors="replace")

    parser = _get_ts_parser()
    if parser is None:
        return _outline_with_regex(path, source, lang=lang)

    try:
        source_bytes = source.encode("utf-8", errors="replace")
        try:
            tree = parser.parse(source_bytes)
        except TypeError:
            # Some tree-sitter-language-pack grammars expect str, not bytes.
            tree = parser.parse(source)
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return _outline_with_regex(path, source, lang=lang)

    imports: list[str] = []
    symbols: list[SymbolOutline] = []
    root = tree.root_node

    for node in root.children:
        node_type = getattr(node, "type", "")
        node_text = _node_text(source_bytes, node)

        if node_type == "import_statement":
            module_match = re.search(r"from\s+['\"](.*?)['\"]", node_text)
            if module_match:
                imports.append(module_match.group(1))
            continue

        if node_type == "function_declaration":
            fn_name = _child_identifier_text(source_bytes, node, "name")
            if fn_name:
                symbols.append(
                    SymbolOutline(
                        name=fn_name,
                        kind="function",
                        start_line=int(node.start_point[0]) + 1,
                        end_line=int(node.end_point[0]) + 1,
                    )
                )
            continue

        if node_type == "class_declaration":
            cls_name = _child_identifier_text(source_bytes, node, "name")
            if cls_name:
                symbols.append(
                    SymbolOutline(
                        name=cls_name,
                        kind="class",
                        start_line=int(node.start_point[0]) + 1,
                        end_line=int(node.end_point[0]) + 1,
                    )
                )
                body = node.child_by_field_name("body")
                if body is not None:
                    for child in body.children:
                        if getattr(child, "type", "") != "method_definition":
                            continue
                        method_name = _method_name(source_bytes, child)
                        if method_name:
                            symbols.append(
                                SymbolOutline(
                                    name=f"{cls_name}.{method_name}",
                                    kind="method",
                                    start_line=int(child.start_point[0]) + 1,
                                    end_line=int(child.end_point[0]) + 1,
                                )
                            )
            continue

        if node_type in {"lexical_declaration", "variable_declaration"}:
            const_text = node_text.lstrip()
            if not const_text.startswith("const "):
                continue
            for decl in node.children:
                if getattr(decl, "type", "") != "variable_declarator":
                    continue
                value = decl.child_by_field_name("value")
                if value is None or getattr(value, "type", "") != "arrow_function":
                    continue
                name = _child_identifier_text(source_bytes, decl, "name")
                if name:
                    symbols.append(
                        SymbolOutline(
                            name=name,
                            kind="function",
                            start_line=int(decl.start_point[0]) + 1,
                            end_line=int(decl.end_point[0]) + 1,
                        )
                    )

    symbols.sort(key=lambda sym: (sym.start_line, sym.end_line, sym.name))
    unique_imports = sorted(dict.fromkeys(imports))
    norm_lang = cast(
        Literal["typescript", "tsx", "javascript"],
        lang if lang in {"typescript", "tsx", "javascript"} else "typescript",
    )
    return FileOutline(
        path=path,
        lang=norm_lang,
        loc=len(source.splitlines()),
        symbols=symbols,
        imports=unique_imports,
    )


def _child_identifier_text(source_bytes: bytes, node: Any, field: str) -> str:
    child = node.child_by_field_name(field)
    if child is None:
        return ""
    return _node_text(source_bytes, child).strip()


def _method_name(source_bytes: bytes, node: Any) -> str:
    name = _child_identifier_text(source_bytes, node, "name")
    if name:
        return name
    for child in getattr(node, "children", []):
        if getattr(child, "type", "") in {
            "property_identifier",
            "identifier",
            "private_property_identifier",
        }:
            return _node_text(source_bytes, child).strip()
    return ""


def _outline_with_regex(path: str, source: str, *, lang: str) -> FileOutline:
    imports: list[str] = []
    symbols: list[SymbolOutline] = []

    for m in _RE_IMPORT_FROM.finditer(source):
        imports.append(m.group(5))

    for m in _RE_EXPORT_CLASS.finditer(source):
        name = m.group(1)
        lineno = _find_lineno(source, m.start())
        symbols.append(SymbolOutline(name=name, kind="class", start_line=lineno, end_line=lineno))

    for m in _RE_EXPORT_FUNCTION.finditer(source):
        name = m.group(1)
        lineno = _find_lineno(source, m.start())
        symbols.append(SymbolOutline(name=name, kind="function", start_line=lineno, end_line=lineno))

    for m in _RE_TOP_LEVEL_CONST_ARROW.finditer(source):
        name = m.group(1)
        lineno = _find_lineno(source, m.start())
        symbols.append(SymbolOutline(name=name, kind="function", start_line=lineno, end_line=lineno))

    # Extract class methods by scanning each class body for balanced braces.
    for class_match in _RE_CLASS_DECL.finditer(source):
        class_name = class_match.group(1)
        open_idx = source.find("{", class_match.end())
        if open_idx < 0:
            continue

        depth = 1
        i = open_idx + 1
        while i < len(source) and depth > 0:
            ch = source[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth != 0:
            continue

        body = source[open_idx + 1 : i - 1]
        base_lineno = _find_lineno(source, open_idx)
        for m in _RE_METHOD_SIG.finditer(body):
            method_name = m.group(1)
            if method_name == "constructor":
                continue
            method_lineno = base_lineno + body[: m.start()].count("\n")
            symbols.append(
                SymbolOutline(
                    name=f"{class_name}.{method_name}",
                    kind="method",
                    start_line=method_lineno,
                    end_line=method_lineno,
                )
            )

    symbols.sort(key=lambda sym: (sym.start_line, sym.end_line, sym.name))
    unique_imports = sorted(dict.fromkeys(imports))
    norm_lang = cast(
        Literal["typescript", "tsx", "javascript"],
        lang if lang in {"typescript", "tsx", "javascript"} else "typescript",
    )
    return FileOutline(
        path=path,
        lang=norm_lang,
        loc=len(source.splitlines()),
        symbols=symbols,
        imports=unique_imports,
    )

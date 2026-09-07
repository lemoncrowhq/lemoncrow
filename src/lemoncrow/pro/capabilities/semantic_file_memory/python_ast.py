"""Advanced Python AST analysis for semantic file memory."""

from __future__ import annotations

import ast
import contextlib
import logging

from .models import FileOutline, ImportInfo, SymbolInfo, SymbolOutline

logger = logging.getLogger(__name__)


def _estimate_complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Cyclomatic complexity proxy: count branches and loops inside the function."""
    count = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (
                ast.If,
                ast.While,
                ast.For,
                ast.ExceptHandler,
                ast.With,
                ast.Assert,
                ast.comprehension,
            ),
        ):
            count += 1
        elif isinstance(child, ast.BoolOp):
            count += len(child.values) - 1
    return count


def _extract_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    out: list[str] = []
    for dec in node.decorator_list:
        try:
            out.append(ast.unparse(dec))
        except Exception:
            logging.exception("Recovered from broad exception handler")
            out.append("?")
    return out


def _extract_docstring(node: ast.AST) -> str:
    body = getattr(node, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        val = body[0].value.value
        if isinstance(val, str):
            # Return first line only to keep it compact
            return val.strip().split("\n")[0][:200]
    return ""


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{prefix} {node.name}({args}){ret}:"
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return f"def {node.name}(...):"


def _return_type_hint(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    if node.returns:
        try:
            return ast.unparse(node.returns)
        except Exception:
            logging.exception("Recovered from broad exception handler")
            logger.warning(
                "Suppressed exception at python_ast.py:67",
                exc_info=True,
            )
    return ""


def analyze_python(source: str) -> tuple[list[SymbolInfo], list[ImportInfo], str, str, int]:
    """
    Comprehensive Python AST analysis.

    Returns:
        symbols        - all symbols (functions, methods, classes, variables)
        imports        - all import statements
        ast_summary    - compact stat string e.g. ``python_ast:functions=5;classes=2``
        module_doc     - first line of module-level docstring (if any)
        total_complexity - sum of all function complexities
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], [], "python_ast:parse_error", "", 0

    symbols: list[SymbolInfo] = []
    imports: list[ImportInfo] = []
    top_level_names: set[str] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_level_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    top_level_names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top_level_names.add(node.target.id)

    module_doc = _extract_docstring(tree)

    def _add_function(
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        parent_class: str = "",
        indent: str = "",
    ) -> None:
        kind = "async_function" if isinstance(fn, ast.AsyncFunctionDef) else "function"
        if parent_class:
            kind = "method"
        sig = indent + _build_signature(fn)
        qual_name = f"{parent_class}.{fn.name}" if parent_class else fn.name
        complexity = _estimate_complexity(fn)
        symbols.append(
            SymbolInfo(
                name=qual_name,
                kind=kind,
                lineno=fn.lineno,
                signature=sig,
                is_export=qual_name in top_level_names or parent_class in top_level_names,
                end_lineno=getattr(fn, "end_lineno", fn.lineno),
                docstring=_extract_docstring(fn),
                decorators=_extract_decorators(fn),
                is_private=fn.name.startswith("_"),
                type_hint=_return_type_hint(fn),
                complexity=complexity,
            )
        )

    def _add_class(cls: ast.ClassDef) -> None:
        base_names: list[str] = []
        for base in cls.bases:
            with contextlib.suppress(Exception):
                base_names.append(ast.unparse(base))
        base_str = f"({', '.join(base_names)})" if base_names else ""
        symbols.append(
            SymbolInfo(
                name=cls.name,
                kind="class",
                lineno=cls.lineno,
                signature=f"class {cls.name}{base_str}:",
                is_export=cls.name in top_level_names,
                end_lineno=getattr(cls, "end_lineno", cls.lineno),
                docstring=_extract_docstring(cls),
                decorators=_extract_decorators(cls),
                is_private=cls.name.startswith("_"),
            )
        )
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _add_function(item, parent_class=cls.name, indent="    ")
            elif isinstance(item, ast.ClassDef):
                _add_class(item)  # nested class

    def _add_variable(node: ast.Assign | ast.AnnAssign, lineno: int) -> None:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    symbols.append(
                        SymbolInfo(
                            name=t.id,
                            kind="variable",
                            lineno=lineno,
                            signature=t.id,
                            is_export=t.id in top_level_names,
                            is_private=t.id.startswith("_") and not t.id.startswith("__"),
                        )
                    )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                type_str = ast.unparse(node.annotation)
            except Exception:
                logging.exception("Recovered from broad exception handler")
                type_str = ""
            symbols.append(
                SymbolInfo(
                    name=node.target.id,
                    kind="variable",
                    lineno=lineno,
                    signature=f"{node.target.id}: {type_str}",
                    is_export=node.target.id in top_level_names,
                    is_private=node.target.id.startswith("_"),
                    type_hint=type_str,
                )
            )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _add_function(node)
        elif isinstance(node, ast.ClassDef):
            _add_class(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            _add_variable(node, node.lineno)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ImportInfo(
                        module=alias.name,
                        names=[alias.asname or alias.name],
                        lineno=node.lineno,
                        is_from=False,
                        alias=alias.asname or "",
                    )
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [alias.asname or alias.name for alias in node.names]
            imports.append(
                ImportInfo(
                    module=node.module,
                    names=names,
                    lineno=node.lineno,
                    is_from=True,
                )
            )

    fn_count = sum(1 for s in symbols if s.kind in ("function", "async_function", "method"))
    cls_count = sum(1 for s in symbols if s.kind == "class")
    imp_count = len(imports)
    total_complexity = sum(s.complexity for s in symbols)
    ast_summary = (
        f"python_ast:functions={fn_count};classes={cls_count};imports={imp_count};complexity={total_complexity}"
    )
    return symbols, imports, ast_summary, module_doc, total_complexity


def stub_function_bodies(source: str, *, max_body_lines: int = 3) -> str:
    """Return source with long function/method bodies replaced by ``...``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines()
    # Map of first-body-lineno → (last-body-lineno, indent)
    stubs: dict[int, tuple[int, str]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if not body:
            continue
        first_stmt = body[0]
        last_stmt = body[-1]

        # Preserve docstring: if first stmt is a string literal, start stub after it
        stub_start_stmt = first_stmt
        has_docstring = isinstance(first_stmt, ast.Expr) and isinstance(
            getattr(first_stmt, "value", None), ast.Constant
        )
        if has_docstring and len(body) > 1:
            stub_start_stmt = body[1]

        body_len = getattr(last_stmt, "end_lineno", last_stmt.lineno) - stub_start_stmt.lineno
        if body_len <= max_body_lines:
            continue

        indent = " " * (node.col_offset + 4)
        start = stub_start_stmt.lineno
        end = getattr(last_stmt, "end_lineno", last_stmt.lineno)
        stubs[start] = (end, indent)

    if not stubs:
        return source

    out: list[str] = []
    skip_until = -1
    for idx, line in enumerate(lines):
        lineno = idx + 1
        if lineno <= skip_until:
            continue
        if lineno in stubs:
            end_ln, indent = stubs[lineno]
            out.append(f"{indent}...")
            skip_until = end_ln
        else:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Backward-compatible aliases (used by __init__.py and tests)
# ---------------------------------------------------------------------------


def _python_full_ast(source: str) -> tuple[list[SymbolInfo], list[ImportInfo], str]:
    """Legacy 3-tuple API: (symbols, imports, ast_summary_str)."""
    symbols, imports, ast_summary, _doc, _cx = analyze_python(source)
    return symbols, imports, ast_summary


def _ast_truncated_source(source: str, *, max_body_lines: int = 3) -> str:
    """Legacy alias for stub_function_bodies."""
    return stub_function_bodies(source, max_body_lines=max_body_lines)


def outline(path: str, source: str) -> FileOutline:
    """Return a compact Python outline (classes/functions/methods + imports)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return FileOutline(path=path, lang="python", loc=len(source.splitlines()), symbols=[], imports=[])

    imports: list[str] = []
    symbols: list[SymbolOutline] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
            continue

        if isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                SymbolOutline(
                    name=node.name,
                    kind="function",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                )
            )
            continue

        if isinstance(node, ast.ClassDef):
            symbols.append(
                SymbolOutline(
                    name=node.name,
                    kind="class",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                )
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(
                        SymbolOutline(
                            name=f"{node.name}.{item.name}",
                            kind="method",
                            start_line=item.lineno,
                            end_line=getattr(item, "end_lineno", item.lineno),
                        )
                    )

    unique_imports = sorted(dict.fromkeys(imports))
    symbols.sort(key=lambda sym: (sym.start_line, sym.end_line, sym.name))
    return FileOutline(
        path=path,
        lang="python",
        loc=len(source.splitlines()),
        symbols=symbols,
        imports=unique_imports,
    )

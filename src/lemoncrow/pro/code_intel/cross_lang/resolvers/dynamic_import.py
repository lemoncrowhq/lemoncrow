"""Literal-only importlib.import_module edge extraction."""

from __future__ import annotations

import ast
from pathlib import Path

from lemoncrow.pro.code_intel.cross_lang.edges import CrossLangCandidate


def resolve_dynamic_imports(repo_root: Path) -> list[CrossLangCandidate]:
    candidates: list[CrossLangCandidate] = []
    for path in sorted(repo_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        rel = path.relative_to(repo_root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "import_module"
                and isinstance(func.value, ast.Name)
                and func.value.id == "importlib"
            ):
                continue
            module_name = _literal_str(node.args[0]) if node.args else None
            if module_name is None:
                continue
            module_path = _resolve_module_path(repo_root, module_name)
            candidates.append(
                CrossLangCandidate(
                    src_language="python",
                    src_file_path=rel,
                    src_line=node.lineno,
                    tgt_symbol_name=module_name,
                    tgt_language="python",
                    tgt_file_path=module_path,
                    edge_kind="dynamic_import",
                    confidence=0.8 if module_path else 0.55,
                )
            )
    return candidates


def _resolve_module_path(repo_root: Path, module_name: str) -> str | None:
    module_base = repo_root / Path(*module_name.split("."))
    candidates = [module_base.with_suffix(".py"), module_base / "__init__.py"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.relative_to(repo_root).as_posix()
    return None


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None

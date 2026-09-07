"""Literal-only subprocess edge extraction for Python entrypoints."""

from __future__ import annotations

import ast
from pathlib import Path

from lemoncrow.pro.code_intel.cross_lang.edges import CrossLangCandidate


def resolve_subprocess(repo_root: Path) -> list[CrossLangCandidate]:
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
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr in {"run", "Popen", "call"}
            ):
                continue
            script_path = _literal_script_path(node.args[0]) if node.args else None
            if script_path is None:
                continue
            resolved = (repo_root / script_path).resolve()
            rel_target = resolved.relative_to(repo_root).as_posix() if resolved.is_file() else script_path
            candidates.append(
                CrossLangCandidate(
                    src_language="python",
                    src_file_path=rel,
                    src_line=node.lineno,
                    tgt_symbol_name="main",
                    tgt_language="python",
                    tgt_file_path=rel_target,
                    edge_kind="subprocess",
                    confidence=0.7 if resolved.is_file() else 0.45,
                )
            )
    return candidates


def _literal_script_path(node: ast.AST) -> str | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    for element in node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str) and element.value.endswith(".py"):
            return element.value
    return None

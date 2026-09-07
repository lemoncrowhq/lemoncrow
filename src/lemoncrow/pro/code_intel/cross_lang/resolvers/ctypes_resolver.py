"""Literal-only ctypes and cffi edge extraction."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from lemoncrow.pro.code_intel.cross_lang.edges import CrossLangCandidate

_CDEF_FUNCTION_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def resolve_ctypes(repo_root: Path) -> list[CrossLangCandidate]:
    candidates: list[CrossLangCandidate] = []
    for path in sorted(repo_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        rel = path.relative_to(repo_root).as_posix()
        ctypes_handles: dict[str, int] = {}
        cffi_handles: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                value = node.value
                if isinstance(value, ast.Call):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if _is_ctypes_loader(value):
                                ctypes_handles[target.id] = node.lineno
                            elif _is_cffi_factory(value):
                                cffi_handles.add(target.id)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
            ):
                handle_name = node.func.value.id
                if handle_name in ctypes_handles and node.func.attr not in {"CDLL", "LoadLibrary", "WinDLL"}:
                    candidates.append(
                        CrossLangCandidate(
                            src_language="python",
                            src_file_path=rel,
                            src_line=node.lineno,
                            tgt_symbol_name=node.func.attr,
                            tgt_language="c",
                            tgt_file_path=None,
                            edge_kind="ffi_ctypes",
                            confidence=0.85 if ctypes_handles[handle_name] == node.lineno else 0.65,
                        )
                    )
                if handle_name in cffi_handles and node.func.attr == "cdef":
                    literal = _literal_str(node.args[0]) if node.args else None
                    if literal is None:
                        continue
                    for function_name in _CDEF_FUNCTION_RE.findall(literal):
                        candidates.append(
                            CrossLangCandidate(
                                src_language="python",
                                src_file_path=rel,
                                src_line=node.lineno,
                                tgt_symbol_name=function_name,
                                tgt_language="c",
                                tgt_file_path=None,
                                edge_kind="ffi_cffi",
                                confidence=0.45,
                            )
                        )
    return candidates


def _is_ctypes_loader(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in {"CDLL", "WinDLL", "LoadLibrary"}:
        if isinstance(func.value, ast.Name) and func.value.id == "ctypes":
            return True
        if isinstance(func.value, ast.Attribute) and isinstance(func.value.value, ast.Name):
            return func.value.value.id == "ctypes" and func.value.attr == "cdll"
    return False


def _is_cffi_factory(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "cffi"
        and func.attr == "FFI"
    ) or (isinstance(func, ast.Name) and func.id == "FFI")


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None

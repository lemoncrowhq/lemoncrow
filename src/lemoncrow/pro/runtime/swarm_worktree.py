"""Git worktree lifecycle helpers for the LemonCrow swarm harness."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DirtyPath:
    status: str
    path: str
    source_path: str | None = None

    @property
    def is_delete(self) -> bool:
        return self.source_path is None and "D" in self.status

    @property
    def is_untracked(self) -> bool:
        return self.status == "??"


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed


def git_repo_root(path: Path) -> Path:
    completed = _git(path, "rev-parse", "--show-toplevel")
    return Path(completed.stdout.strip()).resolve()


def read_head_ref(repo_root: Path) -> str:
    return _git(repo_root, "rev-parse", "HEAD").stdout.strip()


_C_QUOTE_ESCAPES = {
    "a": 0x07,
    "b": 0x08,
    "f": 0x0C,
    "n": 0x0A,
    "r": 0x0D,
    "t": 0x09,
    "v": 0x0B,
    '"': 0x22,
    "\\": 0x5C,
}


def _unquote_git_path(raw: str) -> str:
    """Decode a C-quoted path as emitted by ``git status --porcelain``.

    Git wraps paths containing spaces, quotes, backslashes, control chars or
    (with ``core.quotePath=true``, the default) non-ASCII bytes in double
    quotes and backslash-escapes them (non-ASCII bytes as 3-digit octal).
    """
    if len(raw) < 2 or not (raw.startswith('"') and raw.endswith('"')):
        return raw
    inner = raw[1:-1]
    out = bytearray()
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch != "\\" or i + 1 >= len(inner):
            out.extend(ch.encode("utf-8"))
            i += 1
            continue
        esc = inner[i + 1]
        if esc in _C_QUOTE_ESCAPES:
            out.append(_C_QUOTE_ESCAPES[esc])
            i += 2
        elif esc.isdigit():
            out.append(int(inner[i + 1 : i + 4], 8))
            i += 4
        else:
            out.extend(esc.encode("utf-8"))
            i += 2
    return out.decode("utf-8", errors="surrogateescape")


def _split_rename(raw_path: str) -> tuple[str, str] | None:
    """Split ``old -> new`` porcelain rename entries, respecting C-quoting."""
    if raw_path.startswith('"'):
        i = 1
        while i < len(raw_path) and raw_path[i] != '"':
            i += 2 if raw_path[i] == "\\" else 1
        rest = raw_path[i + 1 :]
        if rest.startswith(" -> "):
            return raw_path[: i + 1], rest[4:]
        return None
    if " -> " in raw_path:
        source_path, target_path = raw_path.split(" -> ", 1)
        return source_path, target_path
    return None


def _parse_dirty_line(line: str) -> DirtyPath:
    status = line[:2]
    if "U" in status or status in {"AA", "DD"}:
        raise RuntimeError(f"cannot fan out swarm from conflicted worktree entry: {line}")
    raw_path = line[3:]
    rename = _split_rename(raw_path)
    if rename is not None:
        source_path, target_path = rename
        return DirtyPath(
            status=status,
            path=_unquote_git_path(target_path),
            source_path=_unquote_git_path(source_path),
        )
    return DirtyPath(status=status, path=_unquote_git_path(raw_path))


def collect_dirty_paths(repo_root: Path) -> list[str]:
    completed = _git(repo_root, "status", "--porcelain")
    return [line.rstrip() for line in completed.stdout.splitlines() if line.strip()]


def collect_dirty_snapshot(repo_root: Path) -> list[DirtyPath]:
    return [_parse_dirty_line(line) for line in collect_dirty_paths(repo_root)]


def _safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


_SKIP_UNTRACKED_DIR_NAMES = frozenset(
    {
        ".lemoncrow-benchmarks",
        ".cache",
        ".codegraph",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "build",
        "coverage",
        "dist",
        "node_modules",
    }
)
_ALLOWLISTED_HIDDEN_DIRS = frozenset({".planning"})
_SKIP_UNTRACKED_FILE_SUFFIXES = frozenset(
    {
        ".db",
        ".log",
        ".pyc",
        ".pyd",
        ".pyo",
        ".sqlite",
        ".sqlite3",
    }
)
_SKIP_UNTRACKED_FILE_NAMES = frozenset({"semantic_file_index.json"})

# Gitignored files commonly required to run experiments — secrets, local
# config, TLS material, cloud credentials, etc.  Matched by exact name,
# name prefix, or suffix against every file found in the base worktree.
_SECRET_FILE_NAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".envrc",
        ".netrc",
        ".secrets",
        ".secret",
        ".pgpass",
        ".my.cnf",
        "credentials.json",
        "secrets.json",
        "service-account.json",
        "serviceaccount.json",
        "gcloud-credentials.json",
    }
)
_SECRET_FILE_PREFIXES: tuple[str, ...] = (".env.",)
_SECRET_FILE_SUFFIXES: frozenset[str] = frozenset(
    {".pem", ".key", ".crt", ".cert", ".p12", ".pfx", ".secret", ".secrets"}
)


def _should_skip_untracked_path(path: str) -> bool:
    parts = Path(path).parts
    if not parts:
        return False
    head = parts[0]
    if head.startswith(".") and head not in _ALLOWLISTED_HIDDEN_DIRS:
        return True
    if any(part in _SKIP_UNTRACKED_DIR_NAMES for part in parts):
        return True
    candidate = Path(path)
    return candidate.name in _SKIP_UNTRACKED_FILE_NAMES or candidate.suffix in _SKIP_UNTRACKED_FILE_SUFFIXES


def _is_secret_file(name: str, suffix: str) -> bool:
    return (
        name in _SECRET_FILE_NAMES
        or any(name.startswith(p) for p in _SECRET_FILE_PREFIXES)
        or suffix in _SECRET_FILE_SUFFIXES
    )


def _sync_secret_files(*, base_worktree: Path, child_worktree: Path) -> None:
    """Copy commonly gitignored secret/config files into the child worktree.

    ``git worktree add`` and ``git status`` both skip ignored files, so we
    glob directly in the base worktree.  Only files that already exist in
    the base are copied; we never create paths that aren't there.
    """
    for src in base_worktree.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(base_worktree)
        # Skip anything nested inside dirs we already exclude (e.g. .venv)
        if any(part in _SKIP_UNTRACKED_DIR_NAMES for part in rel.parts[:-1]):
            continue
        if _is_secret_file(src.name, src.suffix):
            dst = child_worktree / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


class SwarmWorktreeManager:
    def __init__(self, *, repo_root: Path, pool_root: Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.pool_root = Path(pool_root).resolve()

    def create_worktree(self, *, run_id: str, child_id: str, ref: str = "HEAD") -> Path:
        path = self.pool_root / child_id
        path.parent.mkdir(parents=True, exist_ok=True)
        _git(self.repo_root, "worktree", "add", "--detach", str(path), ref)
        return path

    def sync_dirty_state(self, *, base_worktree: Path, child_worktree: Path) -> None:
        for item in collect_dirty_snapshot(base_worktree):
            if item.is_untracked and _should_skip_untracked_path(item.path):
                continue
            if item.source_path:
                _safe_remove(child_worktree / item.source_path)
            if item.is_delete:
                _safe_remove(child_worktree / item.path)
                continue
            source = base_worktree / item.path
            destination = child_worktree / item.path
            if not source.exists():
                continue
            if source.is_dir():
                _safe_remove(destination)
                shutil.copytree(source, destination, dirs_exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        _sync_secret_files(base_worktree=base_worktree, child_worktree=child_worktree)

    def remove_worktree(self, worktree_path: Path) -> None:
        path = Path(worktree_path)
        if not path.exists():
            return
        _git(self.repo_root, "worktree", "remove", "--force", str(path))

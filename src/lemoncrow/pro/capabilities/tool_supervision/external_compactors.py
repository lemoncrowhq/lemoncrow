"""Soft-detection of external command-output compactor binaries.

LemonCrow already rewrites `cat`/`grep`/`find` in-process (see `classify_command`'s
`read`/`grep`/`find_glob` rewrite targets in `bash_exec.py`) because those map
cleanly onto LemonCrow's own indexed tools. Third-party CLIs like rtk
(https://github.com/rtk-ai/rtk) go further and compact a much wider surface --
git, gh, test runners, linters, cloud CLIs -- but reimplementing all of that
natively would be a large, ever-growing maintenance burden for output formats
LemonCrow doesn't control. Instead, when a compatible binary happens to already
be installed on the host, LemonCrow can opt into shelling out to it for a
narrow, explicitly safe allowlist of read-only/idempotent commands.

This integration is soft by design -- same detect-if-present, else-degrade
contract as `astgrep.binaries.discover_astgrep_binary` (which is also
unconditional, not settings-gated):
  - on by default: if a registered binary is found on PATH it's used
    automatically, no opt-in required. Set `LEMONCROW_BASH_EXTERNAL_COMPACTORS=0`
    (or `tool_supervision.external_compactors=false`) to opt back out; never a
    hard dependency -- silently falls back to the plain shell path when the
    binary is absent;
  - detection is lazy (only probed the first time an eligible command is
    seen) and cached for the life of the MCP server process;
  - the allowlist only covers commands that are safe to compact -- i.e. the
    compactor's presence/absence can never change whether a side-effecting
    command (git commit, docker run, ...) runs once or twice.

Adding a second compactor is one entry in `_COMPACTORS` below -- no other
file needs to change.
"""

from __future__ import annotations

import functools
import os
import re
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

_ENV_ENABLED = "LEMONCROW_BASH_EXTERNAL_COMPACTORS"


@dataclass(frozen=True)
class ExternalCompactor:
    """Describes one third-party command-output compactor binary."""

    name: str
    binary: str
    version_args: tuple[str, ...]
    # Exact "head [subcommand ...]" token prefixes this compactor is trusted
    # to wrap, e.g. ("git", "status") or ("pytest",). Kept deliberately narrow
    # to read-only/idempotent commands -- a compactor invocation must never be
    # the thing that decides whether a side-effecting command runs twice.
    safe_prefixes: frozenset[tuple[str, ...]]
    # Optional source-of-truth rewrite command. RTK knows command-specific
    # invocation details that cannot be reproduced by merely prepending the
    # binary (`next build` -> `rtk next`, `./gradlew test` -> `rtk gradlew test`).
    # LemonCrow still performs the local safety allowlist match first.
    rewrite_args: tuple[str, ...] = ()
    # How a user installs the binary -- surfaced by `lc doctor` and the
    # install script when the binary is absent.
    install_hint: str = ""


@dataclass(frozen=True)
class CompactorResolution:
    """Structured detection result for one compactor, cached per process."""

    available: bool
    path: Path | None = None
    version: str | None = None
    reason: str | None = None


RTK = ExternalCompactor(
    name="rtk",
    binary="rtk",
    version_args=("--version",),
    safe_prefixes=frozenset(
        {
            # git/gh: read-only inspection only -- never commit/push/reset/
            # clean/branch (branch creation/deletion is a git-branch mutation).
            ("git", "status"),
            ("git", "log"),
            ("git", "diff"),
            ("git", "show"),
            ("gh", "pr", "list"),
            ("gh", "pr", "view"),
            ("gh", "issue", "list"),
            ("gh", "issue", "view"),
            ("gh", "run", "list"),
            ("gh", "run", "view"),
            ("glab", "mr", "list"),
            ("glab", "mr", "view"),
            ("glab", "issue", "list"),
            ("glab", "issue", "view"),
            ("glab", "pipeline", "list"),
            ("glab", "pipeline", "view"),
            # Test runners -- rerunning is safe by construction.
            ("pytest",),
            ("jest",),
            ("vitest",),
            ("cargo", "test"),
            ("go", "test"),
            ("dotnet", "test"),
            ("gradlew", "test"),
            ("mvn", "test"),
            ("mvn", "integration-test"),
            ("rspec",),
            ("playwright", "test"),
            ("rake", "test"),
            # Linters/type-checkers/read-only build/format checks.
            ("eslint",),
            ("tsc",),
            ("mypy",),
            ("ruff", "check"),
            ("ruff", "format", "--check"),
            ("rubocop",),
            ("golangci-lint", "run"),
            ("cargo", "check"),
            ("cargo", "clippy"),
            ("cargo", "build"),
            ("go", "vet"),
            ("go", "build"),
            ("dotnet", "build"),
            ("dotnet", "format"),
            ("gradlew", "check"),
            ("gradlew", "lint"),
            ("gradlew", "assemble"),
            ("gradlew", "build"),
            ("mvn", "compile"),
            ("mvn", "package"),
            ("mvn", "verify"),
            ("next", "build"),
            ("prettier", "--check"),
            ("biome", "check"),
            ("biome", "lint"),
            ("biome", "format", "--check"),
            ("prisma", "validate"),
            # Containers/cluster inspection -- read-only listings and logs.
            ("docker", "ps"),
            ("docker", "images"),
            ("docker", "logs"),
            ("docker", "inspect"),
            ("docker", "stats"),
            ("docker", "top"),
            ("docker", "compose", "ps"),
            ("docker", "compose", "logs"),
            ("docker", "compose", "config"),
            ("kubectl", "get"),
            ("kubectl", "logs"),
            ("kubectl", "describe"),
            ("kubectl", "events"),
            ("kubectl", "top"),
            ("oc", "get"),
            ("oc", "logs"),
            ("oc", "describe"),
            ("oc", "events"),
            ("oc", "top"),
            # Package listings -- read-only.
            ("pip", "list"),
            ("pip", "outdated"),
            ("pip3", "list"),
            ("pip3", "outdated"),
            ("pnpm", "list"),
            # AWS read-only inspection (rtk ships dedicated filters for these).
            ("aws", "sts", "get-caller-identity"),
            ("aws", "ec2", "describe-instances"),
            ("aws", "lambda", "list-functions"),
            ("aws", "logs", "get-log-events"),
            ("aws", "cloudformation", "describe-stack-events"),
            ("aws", "iam", "list-roles"),
            ("aws", "s3", "ls"),
        }
    ),
    rewrite_args=("rewrite",),
    # Keep the tag in sync with LEMONCROW_RTK_TAG in scripts/lib/common.sh.
    install_hint="cargo install --git https://github.com/rtk-ai/rtk --tag v0.43.0",
)

# Registry of known compactors. Append here to support another binary; no
# other call site needs to change.
_COMPACTORS: tuple[ExternalCompactor, ...] = (RTK,)

_lock = threading.Lock()
_resolved: dict[str, CompactorResolution] = {}


def registered_compactors() -> tuple[ExternalCompactor, ...]:
    """The compactors LemonCrow knows how to detect (diagnostics/UI surface)."""
    return _COMPACTORS


def external_compactors_enabled() -> bool:
    # On by default -- an absent env var means "use it if detected". The env
    # var is only there as an explicit opt-out ("0"/"false"/"no"/"off").
    return os.environ.get(_ENV_ENABLED, "1").strip().lower() not in {"0", "false", "no", "off"}


def _probe(compactor: ExternalCompactor) -> CompactorResolution:
    found = shutil.which(compactor.binary)
    if not found:
        return CompactorResolution(available=False, reason=f"{compactor.binary} not found on PATH")
    path = Path(found)
    try:
        result = subprocess.run(
            [str(path), *compactor.version_args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CompactorResolution(available=False, reason=f"{compactor.binary} --version failed: {exc}")
    if result.returncode != 0:
        return CompactorResolution(available=False, reason=f"{compactor.binary} --version exited {result.returncode}")
    output = (result.stdout or result.stderr).strip()
    version = output.splitlines()[0] if output else None
    return CompactorResolution(available=True, path=path, version=version)


def resolve_compactor(name: str) -> CompactorResolution:
    """Detect *name* on PATH and cache the result for this process's lifetime.

    Resolved lazily (only called once an eligible command is actually seen)
    and never re-probed within the same process.
    """
    with _lock:
        cached = _resolved.get(name)
        if cached is not None:
            return cached
        compactor = next((c for c in _COMPACTORS if c.name == name), None)
        resolution = (
            CompactorResolution(available=False, reason=f"no compactor registered named {name!r}")
            if compactor is None
            else _probe(compactor)
        )
        _resolved[name] = resolution
        return resolution


# Wrapper prefixes that transparently exec their trailing argv unchanged --
# `uv run pytest -q` really does run `pytest -q` (same argv, just inside the
# project venv), so matching `("pytest",)` after peeling `("uv", "run")` off
# is exactly as safe as matching it directly. Deliberately NOT `npm run
# <script>` / `yarn <script>` / `pnpm run <script>`: <script> is a
# package.json-defined alias, not a real command name -- it could invoke
# anything, so treating `npm run test` as if it were literally `jest` would
# be a guess, not a fact.
_TRANSPARENT_RUNNER_PREFIXES: frozenset[tuple[str, ...]] = frozenset(
    {
        ("uv", "run"),
        ("poetry", "run"),
        ("pipenv", "run"),
        ("npx",),
    }
)


def _strip_transparent_runners(lowered: tuple[str, ...]) -> tuple[str, ...]:
    """Peel off transparent runner prefixes so matching sees the real tool."""
    while True:
        for prefix in _TRANSPARENT_RUNNER_PREFIXES:
            if lowered[: len(prefix)] == prefix and len(lowered) > len(prefix):
                lowered = lowered[len(prefix) :]
                break
        else:
            if (
                len(lowered) > 2
                and lowered[1] == "-m"
                and (Path(lowered[0]).name == "python" or re.fullmatch(r"python[0-9.]*", Path(lowered[0]).name))
            ):
                lowered = lowered[2:]
                continue
            return lowered


def _normalized_command_tokens(tokens: list[str]) -> tuple[str, ...]:
    lowered = _strip_transparent_runners(tuple(tok.lower() for tok in tokens))
    if not lowered:
        return lowered
    return (Path(lowered[0]).name, *lowered[1:])


def compactor_for_command(tokens: list[str]) -> ExternalCompactor | None:
    """Return the compactor whose local safety allowlist covers *tokens*."""
    if not tokens:
        return None
    lowered = _normalized_command_tokens(tokens)
    best: ExternalCompactor | None = None
    best_len = -1
    for compactor in _COMPACTORS:
        for prefix in compactor.safe_prefixes:
            if len(prefix) <= len(lowered) and lowered[: len(prefix)] == prefix and len(prefix) > best_len:
                best = compactor
                best_len = len(prefix)
    return best


@functools.lru_cache(maxsize=512)
def _rewrite_cached(binary_path: str, rewrite_args: tuple[str, ...], command: str) -> str | None:
    """Ask a compactor for its canonical invocation, then validate it locally."""
    if not rewrite_args:
        return None
    try:
        result = subprocess.run(
            [binary_path, *rewrite_args, command],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    # RTK 0.43.0 returns 3 for several successful rewrites (mypy, next,
    # gradlew, mvn, go) while still printing the canonical command. Treat that
    # observed code as success only when stdout passes every validation below.
    if result.returncode not in {0, 3}:
        return None
    rewritten = result.stdout.strip()
    if not rewritten or "\n" in rewritten or any(ch in rewritten for ch in ";|&<>`$"):
        return None
    try:
        tokens = shlex.split(rewritten)
    except ValueError:
        return None
    if not tokens or Path(tokens[0]).name != Path(binary_path).name:
        return None
    return shlex.join([binary_path, *tokens[1:]])


def rewrite_compactor_command(
    compactor: ExternalCompactor,
    resolution: CompactorResolution,
    command: str,
) -> str | None:
    if not resolution.available or resolution.path is None:
        return None
    return _rewrite_cached(str(resolution.path), compactor.rewrite_args, command)


def reset() -> None:
    """Clear process-local detection and rewrite caches (tests)."""
    with _lock:
        _resolved.clear()
    _rewrite_cached.cache_clear()


__all__ = [
    "RTK",
    "CompactorResolution",
    "ExternalCompactor",
    "compactor_for_command",
    "external_compactors_enabled",
    "registered_compactors",
    "reset",
    "resolve_compactor",
    "rewrite_compactor_command",
]

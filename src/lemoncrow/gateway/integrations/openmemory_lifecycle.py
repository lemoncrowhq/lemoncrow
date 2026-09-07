"""OpenMemory (and Letta) sidecar lifecycle orchestration.

This module owns the *non-CLI* business logic for the self-hosted OpenMemory and
Letta sidecars: project-root resolution, on-disk path layout, git checkout of the
OpenMemory upstream, generation of the API/UI env files, and the
``docker compose`` / ``make`` invocations that build and run the stacks.

It is a NEW sibling of the MCP-over-HTTP client in ``openmemory.py`` -- that
client talks to a running server, while this module manages the lifecycle of the
server process itself. Keeping them separate avoids bloating the transport
client (RESEARCH assumption A2).

These functions are infrastructure: they raise ``RuntimeError`` (or let
``subprocess`` errors propagate) rather than ``click.ClickException``. The thin
CLI wrappers in ``cli/commands/`` convert those into user-facing CLI errors.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Letta compose helpers                                                       #
# --------------------------------------------------------------------------- #


def letta_compose_file() -> Path:
    return Path.cwd() / "deploy" / "letta" / "docker-compose.yml"


def run_compose(args: list[str]) -> None:
    subprocess.run(["docker", "compose", "-f", str(letta_compose_file()), *args], check=True)


# --------------------------------------------------------------------------- #
# Project-root + OpenMemory path layout                                       #
# --------------------------------------------------------------------------- #


def project_root() -> Path:
    env = os.environ.get("LEMONCROW_INSTALL_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    install_record = Path.home() / ".lemoncrow" / "install_dir"
    with contextlib.suppress(OSError):
        recorded = install_record.read_text(encoding="utf-8").strip()
        if recorded:
            recorded_path = Path(recorded).expanduser().resolve()
            if recorded_path.exists():
                return recorded_path
    return Path(__file__).resolve().parents[4]


def openmemory_dir(root: Path) -> Path:
    return Path(root) / "openmemory"


def openmemory_checkout_dir(root: Path) -> Path:
    return openmemory_dir(root) / "mem0"


def openmemory_workdir(root: Path) -> Path:
    return openmemory_checkout_dir(root) / "openmemory"


def openmemory_service_env_path(root: Path) -> Path:
    return openmemory_dir(root) / "service.env"


def openmemory_api_env_path(root: Path) -> Path:
    return openmemory_workdir(root) / "api" / ".env"


def openmemory_ui_env_path(root: Path) -> Path:
    return openmemory_workdir(root) / "ui" / ".env"


def openmemory_log_path(root: Path) -> Path:
    return openmemory_dir(root) / "openmemory.log"


def mcp_dir(root: Path) -> Path:
    return Path(root) / "mcp"


def mcp_log_path(root: Path) -> Path:
    return mcp_dir(root) / "mcp.log"


# --------------------------------------------------------------------------- #
# OpenMemory lifecycle operations                                             #
# --------------------------------------------------------------------------- #


def ensure_service_env(root: Path) -> Path:
    env_path = openmemory_service_env_path(root)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        # Do not persist API keys to disk in plaintext.
        # Keep sensitive secrets in process environment at runtime instead.
        "USER": os.environ.get("LEMONCROW_OPENMEMORY_USER_ID", os.environ.get("USER", "")),
        "LEMONCROW_OPENMEMORY_URL": os.environ.get("LEMONCROW_OPENMEMORY_URL", "http://127.0.0.1:8765"),
    }
    lines = []
    for key, value in values.items():
        if not value:
            continue
        escaped = value.replace('"', '\\"')
        lines.append(f'{key}="{escaped}"')
    if lines:
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif not env_path.exists():
        env_path.write_text("", encoding="utf-8")
    return env_path


def ensure_checkout(root: Path) -> Path:
    repo_dir = openmemory_checkout_dir(root)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    repo_url = os.environ.get("LEMONCROW_OPENMEMORY_REPO_URL", "https://github.com/mem0ai/mem0.git")
    repo_ref = os.environ.get("LEMONCROW_OPENMEMORY_REF", "main")
    if (repo_dir / ".git").exists():
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "--depth=1", "origin", repo_ref], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "checkout", repo_ref], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", repo_ref], check=True)
    else:
        subprocess.run(["git", "clone", "--depth=1", "--branch", repo_ref, repo_url, str(repo_dir)], check=True)
    workdir = openmemory_workdir(root)
    if not workdir.exists():
        raise RuntimeError(f"OpenMemory checkout is missing {workdir}")
    return workdir


def write_env_files(root: Path) -> None:
    api_env = openmemory_api_env_path(root)
    ui_env = openmemory_ui_env_path(root)
    user_id = (
        os.environ.get("LEMONCROW_OPENMEMORY_USER_ID", "").strip() or os.environ.get("USER", "").strip() or "lemoncrow"
    )
    api_url = os.environ.get("LEMONCROW_OPENMEMORY_URL", "http://127.0.0.1:8765").strip() or "http://127.0.0.1:8765"
    openai_api_key = os.environ.get("LEMONCROW_OPENMEMORY_OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")).strip()
    api_env.parent.mkdir(parents=True, exist_ok=True)
    api_lines = [
        f"OPENAI_API_KEY={openai_api_key}",
        f"USER={user_id}",
    ]
    # 0600 at creation, not write_text() + chmod: this file holds an OpenAI API
    # key, and write_text() creates it 0644 under the default umask -- a chmod
    # afterwards still leaves a world-readable window. O_CREAT's mode only
    # applies to a *new* file, so an existing one is re-chmod'ed explicitly.
    fd = os.open(api_env, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("\n".join(api_lines) + "\n")
    os.chmod(api_env, 0o600)
    ui_env.parent.mkdir(parents=True, exist_ok=True)
    ui_lines = [
        f"NEXT_PUBLIC_API_URL={api_url}",
        f"NEXT_PUBLIC_USER_ID={user_id}",
    ]
    ui_env.write_text("\n".join(ui_lines) + "\n", encoding="utf-8")


def run_make(root: Path, *args: str) -> None:
    workdir = openmemory_workdir(root)
    env = {**os.environ}
    user_id = (
        os.environ.get("LEMONCROW_OPENMEMORY_USER_ID", "").strip() or os.environ.get("USER", "").strip() or "lemoncrow"
    )
    env["USER"] = user_id
    env["NEXT_PUBLIC_API_URL"] = os.environ.get("LEMONCROW_OPENMEMORY_URL", "http://127.0.0.1:8765")
    subprocess.run(["make", *args], cwd=workdir, env=env, check=True)

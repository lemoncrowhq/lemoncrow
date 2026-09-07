"""Detached entrypoint: run a review out-of-band, auto-apply, append the verdict.

Spawned by the ``live_review`` PostToolUse hook via ``python -m``. Must never
raise — a detached reviewer that crashes would be invisible and useless.

The live pass auto-applies high-confidence (``type:patch``) fixes;
the deep pass stays read-only. A single-flight lock ensures one review per
(session, repo) at a time, so a burst of edits cannot fan out N concurrent
reviewers (and N concurrent auto-applies to the same files).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def _lock_path(root: str, session_id: str, repo_root: str) -> Path:
    key = hashlib.sha256(f"{session_id}\x00{Path(repo_root).resolve()}".encode()).hexdigest()[:16]
    return Path(root) / "reviews" / f"{key}.lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _claim_review_lock(root: str, session_id: str, repo_root: str) -> Path | None:
    """Single-flight per (session, repo).

    A burst of edits spawns many reviewer children; only the one that claims the
    lock runs the review, the rest exit immediately. Returns the lock path to
    release on completion, or ``None`` when a live reviewer already holds it.
    """
    path = _lock_path(root, session_id, repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):  # one retry to reclaim a stale lock left by a dead reviewer
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = int(path.read_text("utf-8") or "-1")
            except (OSError, ValueError):
                holder = -1
            if holder > 0 and _pid_alive(holder):
                return None
            with contextlib.suppress(OSError):
                path.unlink()
            continue
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        return path
    return None


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["LEMONCROW_IN_REVIEW"] = "1"
    parser = argparse.ArgumentParser(prog="lemoncrow-live-reviewer")
    parser.add_argument("--session", required=True)
    parser.add_argument("--mode", default="live", choices=["live", "deep"])
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--root", default="")
    args = parser.parse_args(argv)

    try:
        from lemoncrow.pro.capabilities.live_reviewer.runner import run_review
        from lemoncrow.pro.capabilities.live_reviewer.settings import load_reviewer_settings
        from lemoncrow.pro.capabilities.live_reviewer.sink import append_verdict

        root = (
            args.root
            or os.environ.get("LEMONCROW_ROOT")
            or os.environ.get("LEMONCROW_STORE_ROOT")
            or os.path.expanduser("~/.lemoncrow")
        )
        repo_root = os.environ.get("CLAUDE_WORKSPACE_ROOT") or os.getcwd()
        # Single-flight: don't run a second review for this session+repo concurrently.
        lock = _claim_review_lock(root, args.session, repo_root)
        if lock is None:
            return 0
        try:
            settings = load_reviewer_settings(root)
            verdict = run_review(args.session, args.mode, args.path, settings, root)
            # Live pass auto-applies high-confidence (type:patch) fixes;
            # deep pass stays read-only. LEMONCROW_IN_REVIEW is set and the
            # in-process edit never re-enters the PostToolUse hook, so no review loop.
            if settings.auto_apply and args.mode == "live":
                from lemoncrow.pro.capabilities.live_reviewer.apply import apply_review_patches

                applied = apply_review_patches(root, repo_root, args.session, record=verdict)
                if applied.get("count"):
                    verdict["auto_applied"] = applied
            append_verdict(root, args.session, verdict)
        finally:
            with contextlib.suppress(OSError):
                lock.unlink()
    except Exception:  # noqa: BLE001 - detached child must never surface a crash
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

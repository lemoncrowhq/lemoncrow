"""Optional PR bot for approved lesson candidates (WP-16)."""

from __future__ import annotations

import difflib
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation.lesson_models import LessonCandidate, LessonPromotion
from lemoncrow.core.foundation.models import Playbook
from lemoncrow.core.foundation.renderer import render_playbook_markdown
from lemoncrow.infra.storage.bundle import StoreBundle

_BOT_ENV_FLAG = "LEMONCROW_LESSON_PR_BOT_ENABLED"
_BOT_TOKEN = "GITHUB_TOKEN"
_BOT_EMAIL = "bot@lemoncrow"


def _is_enabled(env: Mapping[str, str]) -> bool:
    return env.get(_BOT_ENV_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def _run_subprocess(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=True)


class LessonPrBot:
    """Create a focused GitHub PR for one approved lesson candidate."""

    def __init__(
        self,
        *,
        store: StoreBundle,
        root: Path,
        env: Mapping[str, str] | None = None,
        run_cmd: Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]] = _run_subprocess,
    ) -> None:
        self.store = store
        self.root = root
        self.env = env or os.environ
        self._run_cmd = run_cmd

    def sync_pr(self, *, lesson_id: str, dry_run: bool = False) -> dict[str, Any]:
        if not _is_enabled(self.env) or not self.env.get(_BOT_TOKEN, "").strip():
            return {"skipped": True, "reason": "disabled"}

        candidate = self.store.lessons.get_lesson_candidate(lesson_id)
        if candidate is None:
            raise ValueError(f"lesson not found: {lesson_id}")
        if candidate.status != "approved":
            raise ValueError("lesson must be approved before sync-pr")

        block = self._resolve_block(candidate)
        if block is None:
            return {"skipped": True, "reason": "no_playbook_patch"}

        block_path = self.store.knowledge.blocks_dir / f"{block.id}.md"
        new_content = render_playbook_markdown(block)
        old_content = block_path.read_text(encoding="utf-8") if block_path.exists() else ""

        branch = f"lemoncrow/lesson/{lesson_id}"
        pr_title = f"LemonCrow lesson: {lesson_id}"
        pr_body = self._build_pr_body(candidate)

        if dry_run:
            return {
                "skipped": False,
                "dry_run": True,
                "lesson_id": lesson_id,
                "branch": branch,
                "block_path": str(block_path),
                "pr_title": pr_title,
                "pr_body": pr_body,
                "diff": self._render_diff(old_content, new_content, block_path.name),
            }

        block_path.parent.mkdir(parents=True, exist_ok=True)
        block_path.write_text(new_content, encoding="utf-8")

        self._checkout_branch(branch)
        self._run_cmd(["git", "add", str(block_path)], self.root)
        self._run_cmd(
            [
                "git",
                "commit",
                "-m",
                f"lesson: promote {lesson_id}",
                "-m",
                f"co-author: LemonCrow Lesson Bot <{_BOT_EMAIL}>",
            ],
            self.root,
        )
        created = self._run_cmd(
            [
                "gh",
                "pr",
                "create",
                "--title",
                pr_title,
                "--body",
                pr_body,
                "--head",
                branch,
            ],
            self.root,
        )
        pr_url = created.stdout.strip()
        self._upsert_promotion_url(lesson_id=lesson_id, pr_url=pr_url)
        return {
            "skipped": False,
            "dry_run": False,
            "lesson_id": lesson_id,
            "branch": branch,
            "block_path": str(block_path),
            "pr_url": pr_url,
        }

    def _resolve_block(self, candidate: LessonCandidate) -> Playbook | None:
        if candidate.proposed_block is not None:
            return candidate.proposed_block
        promotions = self.store.lessons.list_lesson_promotions(limit=200)
        for promotion in promotions:
            if promotion.lesson_id != candidate.id:
                continue
            block_id = promotion.published_block_id or promotion.edited_block_id
            if block_id:
                return self.store.knowledge.get_block(block_id)
        return None

    def _checkout_branch(self, branch: str) -> None:
        listed = self._run_cmd(["git", "branch", "--list", branch], self.root)
        if listed.stdout.strip():
            self._run_cmd(["git", "checkout", branch], self.root)
            return
        self._run_cmd(["git", "checkout", "-b", branch], self.root)

    def _build_pr_body(self, candidate: LessonCandidate) -> str:
        evidence = "\n".join(f"- {trace_id}" for trace_id in candidate.evidence_trace_ids)
        return (
            "Promoted lesson candidate from LemonCrow runtime.\n\n"
            f"- lesson_id: {candidate.id}\n"
            f"- domain: {candidate.domain}\n"
            "- evidence_trace_ids:\n"
            f"{evidence}\n"
        )

    def _render_diff(self, old_text: str, new_text: str, filename: str) -> str:
        return "\n".join(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                lineterm="",
            )
        )

    def _upsert_promotion_url(self, *, lesson_id: str, pr_url: str) -> None:
        for promotion in self.store.lessons.list_lesson_promotions(limit=200):
            if promotion.lesson_id != lesson_id:
                continue
            updated = LessonPromotion(
                id=promotion.id,
                lesson_id=promotion.lesson_id,
                published_block_id=promotion.published_block_id,
                edited_block_id=promotion.edited_block_id,
                pr_url=pr_url,
                created_at=promotion.created_at,
            )
            self.store.lessons.upsert_lesson_promotion(updated)
            return

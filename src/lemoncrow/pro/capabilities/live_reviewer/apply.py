"""Apply high-confidence patch findings emitted by a review.

Reviewers may emit typed findings; patch entries are mechanical, high-confidence
fixes expressed as old_string/new_string. This applies a selected subset (or all)
through LemonCrow's rich-edit machinery — the same path the edit tool uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lemoncrow.pro.capabilities.live_reviewer.sink import latest_verdict


def patch_findings(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Well-formed patch findings from a review record."""
    if not isinstance(record, dict):
        return []
    findings = record.get("findings")
    if not isinstance(findings, list):
        return []
    out: list[dict[str, Any]] = []
    for finding in findings:
        if (
            isinstance(finding, dict)
            and finding.get("type") == "patch"
            and isinstance(finding.get("file"), str)
            and isinstance(finding.get("old_string"), str)
            and isinstance(finding.get("new_string"), str)
        ):
            out.append(finding)
    return out


def apply_review_patches(
    root: str | Path,
    repo_root: str | Path,
    session_id: str,
    *,
    indices: Sequence[int] | None = None,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a review's patch findings (all, or the given indices).

    Uses ``record`` when provided (the just-produced verdict, so the live pass can
    auto-apply without a sink round-trip); otherwise reads the latest verdict.
    """
    patches = patch_findings(record if record is not None else latest_verdict(root, session_id))
    if indices is not None:
        wanted = set(indices)
        patches = [p for i, p in enumerate(patches) if i in wanted]
    if not patches:
        return {"applied": [], "failed": [], "count": 0}

    from lemoncrow.pro.capabilities.tool_supervision.rich_edit import apply_rich_edits

    edits = [{"file_path": p["file"], "old_string": p["old_string"], "new_string": p["new_string"]} for p in patches]
    result = apply_rich_edits(edits, atomic=False, repo_root=Path(repo_root))
    return {
        "applied": result.get("applied", []),
        "failed": result.get("failed", []),
        "count": len(edits),
    }

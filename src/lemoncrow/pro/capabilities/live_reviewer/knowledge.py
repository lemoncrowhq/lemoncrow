"""Layered review knowledge — distributed to the team via the repo.

Three sources, mirroring baseline's repo + personal layering:

- **repo lessons** — the first heading of each ``<repo>/.lemoncrow/lessons/blocks/*.md``.
- **repo overlay (team)** — ``<repo>/.lemoncrow/review.json`` (notes/boost/suppress).
- **personal overlay (you)** — ``<lemoncrow_root>/review_overlay.json`` (per-user).

Committing ``.lemoncrow/lessons/`` and ``.lemoncrow/review.json`` distributes the learned
rules to the whole team: every clone gets them and the reviewer applies them for
everyone. All reads are fail-open.
"""

from __future__ import annotations

import json
from pathlib import Path

_MAX_LESSONS = 8
_MAX_ITEMS = 12
_KEYS = ("notes", "suppress", "boost")


def overlay_path(root: str | Path) -> Path:
    """Per-user (personal) overlay under the LemonCrow root."""
    return Path(root) / "review_overlay.json"


def repo_overlay_path(repo_root: str | Path) -> Path:
    """Team overlay committed in the repo (shared with everyone who clones it)."""
    return Path(repo_root) / ".lemoncrow" / "review.json"


def _empty() -> dict[str, list[str]]:
    return {"notes": [], "suppress": [], "boost": []}


def _load_overlay_at(path: Path) -> dict[str, list[str]]:
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty()
    if not isinstance(data, dict):
        return _empty()

    def _strs(key: str) -> list[str]:
        value = data.get(key)
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()][:_MAX_ITEMS]

    return {"notes": _strs("notes"), "suppress": _strs("suppress"), "boost": _strs("boost")}


def load_overlay(root: str | Path) -> dict[str, list[str]]:
    """Personal overlay (per-user)."""
    return _load_overlay_at(overlay_path(root))


def load_repo_overlay(repo_root: str | Path) -> dict[str, list[str]]:
    """Team overlay (committed in the repo)."""
    return _load_overlay_at(repo_overlay_path(repo_root))


def write_overlay(path: Path, overlay: dict[str, list[str]]) -> bool:
    """Persist an overlay (only the canonical keys). Fail-open -> False."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({key: overlay.get(key, []) for key in _KEYS}, indent=2), encoding="utf-8")
        return True
    except OSError:
        return False


# Managed allow-list so the team overlay is committable even though projects
# typically gitignore the whole .lemoncrow/ directory. We never touch the repo's
# top-level .gitignore — only this nested one inside .lemoncrow/.
_REPO_SHARE_GITIGNORE = (
    "# Managed by LemonCrow. Keep per-user/cache files out of git but SHARE the\n"
    "# team review knowledge. Add more '!name' lines to share other files.\n"
    "*\n"
    "!.gitignore\n"
    "!review.json\n"
)


def ensure_repo_share_gitignore(repo_root: str | Path) -> None:
    """Write <repo>/.lemoncrow/.gitignore allow-listing the team overlay.

    Idempotent and non-destructive: only created when absent, so a user's own
    customised allow-list is never clobbered. (Note: for the re-include to take
    effect the project's top-level ignore must target ``.lemoncrow/*`` contents,
    not the bare ``.lemoncrow/`` directory — git cannot re-include under a fully
    ignored directory.)
    """
    path = Path(repo_root) / ".lemoncrow" / ".gitignore"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(_REPO_SHARE_GITIGNORE, encoding="utf-8")
    except OSError:
        pass


def _repo_lessons(repo_root: str | Path) -> list[str]:
    blocks = Path(repo_root) / ".lemoncrow" / "lessons" / "blocks"
    if not blocks.is_dir():
        return []
    try:
        files = sorted(blocks.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    out: list[str] = []
    for path in files[:_MAX_LESSONS]:
        try:
            text = path.read_text("utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                out.append(stripped)
                break
    return out


def _merge_dedup(*lists: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for lst in lists:
        for item in lst:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def collect_review_context(root: str | Path, repo_root: str | Path) -> str:
    """Build the reviewer's knowledge block from team + personal layers."""
    personal = load_overlay(root)
    team = load_repo_overlay(repo_root)
    lessons = _repo_lessons(repo_root)
    boost = _merge_dedup(team["boost"], personal["boost"])
    suppress = _merge_dedup(team["suppress"], personal["suppress"])
    team_keys = {note.lower() for note in team["notes"]}
    personal_notes = [note for note in personal["notes"] if note.lower() not in team_keys]
    if not (team["notes"] or personal_notes or lessons or boost or suppress):
        return ""
    lines = ["## Repository review knowledge — apply these"]
    lines.extend(f"- team rule: {note}" for note in team["notes"])
    lines.extend(f"- repo lesson: {lesson}" for lesson in lessons)
    lines.extend(f"- your rule: {note}" for note in personal_notes)
    if boost:
        lines.append("Weight these areas more heavily: " + ", ".join(boost))
    if suppress:
        lines.append("Do NOT raise findings about (suppressed): " + ", ".join(suppress))
    return "\n".join(lines) + "\n"

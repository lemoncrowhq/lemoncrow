"""G16 -- PR-risk profile + commit-provenance classification.

Two read-only, fail-open analyses over the existing substrate:

* :func:`pr_risk` -- fuse blast-radius (``change_impact``), per-file complexity,
  git churn, and a test-gap signal into a single 0..1 risk score with a tier and
  the contributing factors. Higher blast radius, higher churn, missing affected
  tests, and higher complexity all push the score up.
* :func:`commit_provenance` -- classify the commits touching a file/symbol into
  ``bugfix`` / ``refactor`` / ``feature`` / ``perf`` / ``rename`` / ``revert`` /
  ``docs`` / ``test`` / ``chore`` from commit-message + touched-file heuristics.
  Honest: heuristic classification with a tagged 0..1 confidence per commit.

Nothing here mutates state. Git access is via the existing
:mod:`lemoncrow.pro.code_intel.git_history` walker / pygit2 bootstrap and is
guarded so a non-git or pygit2-less environment degrades to a churn of 0 rather
than raising.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from lemoncrow.pro.capabilities.semantic_file_memory.graph_analytics import _is_test_path

logger = logging.getLogger(__name__)

# Risk-fusion weights (sum to 1.0). Blast radius dominates, then churn, then the
# binary test-gap penalty, then raw complexity. Chosen so any single strong
# signal lifts the score into at least the "medium" tier.
_W_BLAST = 0.40
_W_CHURN = 0.25
_W_TESTGAP = 0.20
_W_COMPLEXITY = 0.15

_TIER_BOUNDS = ((0.25, "low"), (0.50, "medium"), (0.75, "high"))


def _tier_for(score: float) -> str:
    for bound, label in _TIER_BOUNDS:
        if score < bound:
            return label
    return "critical"


def _blast_factor(impact_total: int) -> float:
    """Map blast-radius file count to 0..1 with diminishing returns.

    0 -> 0.0, 1 -> 0.2, 3 -> ~0.5, >=10 -> ~1.0. A log-ish saturating curve so a
    single importer registers risk without one huge fan-out pinning everything.
    """
    if impact_total <= 0:
        return 0.0
    return min(1.0, impact_total / 10.0)


def _complexity_factor(total_complexity: int) -> float:
    """Saturating map of summed cyclomatic complexity to 0..1."""
    if total_complexity <= 0:
        return 0.0
    return min(1.0, total_complexity / 50.0)


def _file_churn_score(repo_root: Path, file_path: str, *, window_days: int = 180) -> dict[str, Any]:
    """Commit count touching *file_path* within the window -> 0..1 churn score.

    Mirrors :meth:`BlameAnnotator._compute_churn` semantics at file granularity:
    ``score = min(1.0, commit_count / 20)``. Fail-open: any git/pygit2 problem
    returns a zero-churn, ``available=False`` record so PR-risk still produces a
    number.
    """
    rel = file_path
    try:
        from lemoncrow.pro.code_intel.git_history import require_pygit2

        pygit2 = require_pygit2()
        repo = pygit2.Repository(str(repo_root))
        candidate = Path(file_path)
        if candidate.is_absolute():
            try:
                rel = str(candidate.resolve().relative_to(Path(repo_root).resolve()))
            except ValueError:
                rel = file_path
        cutoff = int(time.time()) - window_days * 86400
        head = repo.revparse_single("HEAD")
        commit_count = 0
        for commit in repo.walk(head.id, pygit2.enums.SortMode.TIME):
            if commit.commit_time < cutoff:
                break
            if not commit.parents:
                continue
            parent = commit.parents[0]
            diff = parent.tree.diff_to_tree(commit.tree)
            touched = any(patch.delta.new_file.path == rel or patch.delta.old_file.path == rel for patch in diff)
            if touched:
                commit_count += 1
        return {
            "commit_count": commit_count,
            "score": min(1.0, commit_count / 20.0),
            "window_days": window_days,
            "available": True,
        }
    except Exception:
        logger.exception("Recovered from broad exception handler")
        return {"commit_count": 0, "score": 0.0, "window_days": window_days, "available": False}


def _file_complexity(cap: Any, file_path: str, repo_root: Path) -> int:
    """Indexed complexity_score for a file (summarising it first if needed)."""
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = repo_root / file_path
    try:
        if candidate.is_file():
            summary = cap.get_cached(candidate) or cap.summarize_file(candidate)
            return int(getattr(summary, "complexity_score", 0))
    except Exception:
        logger.exception("Recovered from broad exception handler")
    return 0


def pr_risk(
    *,
    repo_root: Path,
    lemoncrow_root: Path,
    paths: list[str],
    window_days: int = 180,
) -> dict[str, Any]:
    """Fuse blast-radius + complexity + churn + test-gap into a 0..1 PR risk.

    *paths* are the changed files (relative to ``repo_root`` or absolute). Each
    file is scored independently and the overall PR score is the max of the
    per-file scores (the riskiest file dominates a review).
    """
    from lemoncrow.pro.capabilities.semantic_file_memory import SemanticFileMemoryCapability

    try:
        cap = SemanticFileMemoryCapability(lemoncrow_root)
        per_file: list[dict[str, Any]] = []
        for raw in paths:
            abs_path = Path(raw) if Path(raw).is_absolute() else repo_root / raw
            # Fold the file (and thereby its importers, if present) into the index.
            if abs_path.is_file():
                cap.summarize_file(abs_path)
            impact = cap.change_impact(str(abs_path))
            impact_total = len(impact.get("direct_importers", [])) + len(impact.get("transitive_importers", []))
            # Re-derive affected tests with the robust component-based test
            # detector rather than change_impact's loose substring filter, so a
            # working directory that merely contains "test" (e.g. a tmp dir named
            # ``test_run0/``) cannot mask a genuine test gap.
            importers = list(impact.get("direct_importers", [])) + list(impact.get("transitive_importers", []))
            affected_tests = [f for f in importers if _is_test_path(f)]
            test_gap = 1.0 if not affected_tests else 0.0
            complexity = _file_complexity(cap, raw, repo_root)
            churn = _file_churn_score(repo_root, str(abs_path), window_days=window_days)

            blast_f = _blast_factor(impact_total)
            complexity_f = _complexity_factor(complexity)
            churn_f = float(churn["score"])
            score = _W_BLAST * blast_f + _W_CHURN * churn_f + _W_TESTGAP * test_gap + _W_COMPLEXITY * complexity_f
            score = round(min(1.0, score), 4)
            per_file.append(
                {
                    "path": raw,
                    "score": score,
                    "tier": _tier_for(score),
                    "factors": {
                        "blast_radius": {
                            "impacted_files": impact_total,
                            "affected_tests": affected_tests,
                            "factor": round(blast_f, 4),
                        },
                        "churn": {
                            "commit_count": churn["commit_count"],
                            "factor": round(churn_f, 4),
                            "available": churn["available"],
                        },
                        "test_gap": {"missing_tests": not affected_tests, "factor": test_gap},
                        "complexity": {"score": complexity, "factor": round(complexity_f, 4)},
                    },
                    "risk_level": impact.get("risk_level", "low"),
                }
            )
        overall = round(max((f["score"] for f in per_file), default=0.0), 4)
        return {
            "kind": "pr_risk",
            "overall_score": overall,
            "overall_tier": _tier_for(overall),
            "file_count": len(per_file),
            "files": sorted(per_file, key=lambda f: -float(f["score"])),
            "weights": {
                "blast_radius": _W_BLAST,
                "churn": _W_CHURN,
                "test_gap": _W_TESTGAP,
                "complexity": _W_COMPLEXITY,
            },
            "heuristic": True,
        }
    except Exception:
        logger.exception("Recovered from broad exception handler")
        return {
            "kind": "pr_risk",
            "overall_score": 0.0,
            "overall_tier": "low",
            "file_count": 0,
            "files": [],
            "heuristic": True,
            "error": "recovered",
        }


# ----------------------------------------------------------------------------
# commit_provenance
# ----------------------------------------------------------------------------

# Ordered most-specific-first; the first category whose pattern matches wins so
# that e.g. "revert" beats a generic "fix" mention. Each entry is a regex over
# the (lowercased) first line / subject of the commit message.
_MESSAGE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("revert", re.compile(r"\brevert(s|ed|ing)?\b|^revert\b|this reverts commit")),
    ("rename", re.compile(r"\bren(ame|aming)\b|\bmov(e|ed|ing)\b")),
    ("perf", re.compile(r"\bperf\b|\bperformance\b|\boptimi[sz]e?(s|d|ing)?\b|\bspeed ?up\b|\bfaster\b")),
    ("bugfix", re.compile(r"\bfix(es|ed|ing)?\b|\bbug\b|\bhotfix\b|\bpatch\b|\bregression\b|\bcrash\b")),
    ("refactor", re.compile(r"\brefactor(s|ed|ing)?\b|\bcleanup\b|\btidy\b|\bsimplif(y|ies|ied)\b")),
    ("docs", re.compile(r"\bdocs?\b|\bdocumentation\b|\breadme\b|\bchangelog\b")),
    ("test", re.compile(r"\btests?\b|\bunit ?test\b|\bcoverage\b|\bpytest\b|\bspec\b")),
    (
        "feature",
        re.compile(r"\bfeat\b|\bfeature\b|\badd(s|ed|ing)?\b|\bimplement(s|ed|ing)?\b|\bintroduce(s|d)?\b|\bsupport\b"),
    ),
    ("chore", re.compile(r"\bchore\b|\bbump\b|\bdeps?\b|\bdependenc(y|ies)\b|\bci\b|\bbuild\b|\brelease\b")),
)

# Conventional-commit prefix (``feat:``/``fix(scope):``) -> category. Higher
# confidence than a free-text match because the author declared intent.
_CONVENTIONAL = re.compile(r"^(?P<type>[a-z]+)(?:\([^)]*\))?!?:")
_CONVENTIONAL_MAP = {
    "feat": "feature",
    "fix": "bugfix",
    "perf": "perf",
    "refactor": "refactor",
    "docs": "docs",
    "test": "test",
    "chore": "chore",
    "build": "chore",
    "ci": "chore",
    "revert": "revert",
    "style": "refactor",
}


def classify_commit_message(message: str, files_touched: list[str] | None = None) -> dict[str, Any]:
    """Classify a single commit by message + touched-file heuristics.

    Returns ``{category, confidence, signal}``. A conventional-commit prefix is
    the strongest signal (confidence 0.9); a free-text keyword match is 0.6; a
    file-shape-only inference (all touched files are docs/tests) is 0.5; and a
    pure fallback is ``chore`` at 0.2. Confidence is honest, not certainty.
    """
    subject = (message or "").strip().splitlines()[0].lower() if message and message.strip() else ""
    files = files_touched or []

    conv = _CONVENTIONAL.match(subject)
    if conv is not None:
        mapped = _CONVENTIONAL_MAP.get(conv.group("type"))
        if mapped is not None:
            return {"category": mapped, "confidence": 0.9, "signal": "conventional_prefix"}

    # "this reverts commit" can appear in the body, so scan the whole message for
    # the revert signature before falling back to subject-only rules.
    full = (message or "").lower()
    if "this reverts commit" in full:
        return {"category": "revert", "confidence": 0.85, "signal": "revert_body"}

    for category, pattern in _MESSAGE_RULES:
        if pattern.search(subject):
            return {"category": category, "confidence": 0.6, "signal": "message_keyword"}

    # File-shape inference when the message is uninformative.
    if files:
        lowered = [f.lower() for f in files]
        if all(_is_doc_file(f) for f in lowered):
            return {"category": "docs", "confidence": 0.5, "signal": "file_shape"}
        if all(_is_test_file(f) for f in lowered):
            return {"category": "test", "confidence": 0.5, "signal": "file_shape"}

    return {"category": "chore", "confidence": 0.2, "signal": "fallback"}


def _is_doc_file(path: str) -> bool:
    return path.endswith((".md", ".markdown", ".rst", ".txt")) or "/docs/" in path


def _is_test_file(path: str) -> bool:
    base = path.rsplit("/", 1)[-1]
    return (
        base.startswith("test_")
        or base.endswith("_test.py")
        or ".test." in base
        or ".spec." in base
        or "/tests/" in path
        or "/test/" in path
    )


def commit_provenance(
    *,
    repo_root: Path,
    path: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Classify the recent commits touching *path* (or the whole repo).

    Reuses :func:`iter_commit_records` for the bounded, bot-filtered walk; each
    record is classified by :func:`classify_commit_message`. When *path* is set
    only commits that touched it are kept. Fail-open: a non-git environment
    returns an empty, structurally-valid result.
    """
    try:
        from lemoncrow.pro.code_intel.git_history.walker import iter_commit_records

        target_rel: str | None = None
        if path:
            candidate = Path(path)
            if candidate.is_absolute():
                try:
                    target_rel = str(candidate.resolve().relative_to(Path(repo_root).resolve()))
                except ValueError:
                    target_rel = path
            else:
                target_rel = path

        commits: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        # Walk more than `limit` raw records when filtering by path, so we still
        # surface up to `limit` *matching* commits.
        walk_limit = limit if target_rel is None else max(limit * 10, 200)
        for record in iter_commit_records(repo_root, limit=walk_limit):
            if target_rel is not None and target_rel not in record.files_touched:
                continue
            verdict = classify_commit_message(record.message, record.files_touched)
            category = str(verdict["category"])
            counts[category] = counts.get(category, 0) + 1
            commits.append(
                {
                    "sha": record.sha[:12],
                    "category": category,
                    "confidence": verdict["confidence"],
                    "signal": verdict["signal"],
                    "subject": record.message.splitlines()[0][:120] if record.message else "",
                    "files_touched": len(record.files_touched),
                }
            )
            if len(commits) >= limit:
                break
        return {
            "kind": "commit_provenance",
            "path": path,
            "commit_count": len(commits),
            "by_category": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "commits": commits,
            "heuristic": True,
            "note": "Heuristic classification from message + touched-file shape; confidence is tagged.",
        }
    except Exception:
        logger.exception("Recovered from broad exception handler")
        return {
            "kind": "commit_provenance",
            "path": path,
            "commit_count": 0,
            "by_category": {},
            "commits": [],
            "heuristic": True,
            "error": "recovered",
        }

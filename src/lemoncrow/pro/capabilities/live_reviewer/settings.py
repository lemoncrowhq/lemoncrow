"""Typed settings for the live / automated code reviewer.

The base plugin settings store (``plugin_runtime.PLUGIN_DEFAULT_SETTINGS``) is
bool-only, so the reviewer's int/str keys are read directly from the raw
``plugin_settings.json`` here. Defaults keep the reviewer fully OFF (opt-in).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemoncrow.core.capabilities.default_definitions import (
    DEFAULT_OWNED_MODEL,
    READONLY_OWNED_MODEL,
)
from lemoncrow.core.capabilities.plugin_runtime import plugin_settings_path

MIN_DEEP_INTERVAL = 5
MAX_DEEP_INTERVAL = 1000
DEFAULT_DEEP_INTERVAL = 50


@dataclass(frozen=True)
class ReviewerSettings:
    """Resolved reviewer configuration. All gates default OFF."""

    live_reviewer: bool = False
    live_reviewer_model: str = ""
    deep_edit_count_reviewer: bool = False
    deep_edit_count_interval: int = DEFAULT_DEEP_INTERVAL
    review_model: str = ""
    agentic: bool = True
    # Not a gate: when the live pass is on, auto-apply high-confidence patch fixes.
    # Set liveReviewerAutoApply=false for review-only.
    auto_apply: bool = True

    @property
    def enabled(self) -> bool:
        return self.live_reviewer or self.deep_edit_count_reviewer

    def model_for(self, mode: str) -> str:
        """Pinned model for a pass; falls back to a sensible owned default."""
        if mode == "deep":
            return self.review_model or DEFAULT_OWNED_MODEL
        return self.live_reviewer_model or READONLY_OWNED_MODEL


def _raw_settings(root: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(plugin_settings_path(root).read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    nested = data.get("lemoncrow")
    if isinstance(nested, dict):
        return nested
    return data


def _clamp_interval(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_DEEP_INTERVAL
    return max(MIN_DEEP_INTERVAL, min(MAX_DEEP_INTERVAL, n))


def load_reviewer_settings(root: str | Path) -> ReviewerSettings:
    """Read reviewer settings from ``plugin_settings.json`` (raw, typed)."""
    raw = _raw_settings(root)
    return ReviewerSettings(
        live_reviewer=bool(raw.get("liveReviewer", False)),
        live_reviewer_model=str(raw.get("liveReviewerModel", "") or ""),
        deep_edit_count_reviewer=bool(raw.get("deepEditCountReviewer", False)),
        deep_edit_count_interval=_clamp_interval(raw.get("deepEditCountInterval", DEFAULT_DEEP_INTERVAL)),
        review_model=str(raw.get("reviewModel", "") or ""),
        agentic=bool(raw.get("agenticReviewer", True)),
        auto_apply=bool(raw.get("liveReviewerAutoApply", True)),
    )


def split_provider_model(model: str) -> tuple[str, str]:
    """Split ``provider/model`` into ``(provider, model)``.

    Returns ``("", model)`` when there is no ``provider/`` prefix.
    """
    text = (model or "").strip()
    if "/" in text:
        provider, _, rest = text.partition("/")
        return provider.strip().lower(), rest.strip()
    return "", text

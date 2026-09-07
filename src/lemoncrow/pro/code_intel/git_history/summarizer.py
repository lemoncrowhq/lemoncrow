"""LLM-based commit summariser for Context Lineage (M1).

Uses the internal_llm.chat() abstraction so Ollama and OpenAI backends
both work without change. Model is configurable via LEMONCROW_LINEAGE_MODEL
env var; otherwise it falls back to the active backend's default model.
"""

from __future__ import annotations

import os

from lemoncrow.core.capabilities.code_intel_contract import SummarizerError
from lemoncrow.infra.internal_llm import chat
from lemoncrow.pro.code_intel.git_history.models import CommitRecord, CommitSummary

_CURRENT_PROMPT_VERSION = "v1"

_PROMPT_V1 = (
    "Summarise this commit in 80-120 words. Cover:\n"
    "1. Primary objective (what problem was solved)\n"
    "2. Key files and functions changed\n"
    "3. Technical terminology a future reader would search for\n\n"
    "Do not include the commit hash or author. Do not include any code.\n"
    "Do not editorialise. Plain prose only.\n\n"
    "<COMMIT_MESSAGE>\n{message}\n</COMMIT_MESSAGE>\n\n"
    "<DIFF_TRUNCATED_TO_2K_TOKENS>\n{diff}\n</DIFF_TRUNCATED_TO_2K_TOKENS>"
)

_DEFAULT_MODEL = "claude-haiku-4-5"
_ENV_MODEL_KEY = "LEMONCROW_LINEAGE_MODEL"


def _backend_default_model() -> str:
    backend = os.environ.get("LEMONCROW_LLM_BACKEND", "none").lower().strip()
    if backend == "ollama":
        return os.environ.get("LEMONCROW_OLLAMA_MODEL", "").strip() or "qwen3.6:27b"
    if backend in {"openai", "openai_compatible"}:
        return os.environ.get("LEMONCROW_OPENAI_MODEL", "").strip() or "gpt-4o-mini"
    if backend == "litellm":
        return os.environ.get("LEMONCROW_LITELLM_MODEL", "").strip() or "gpt-4o-mini"
    return _DEFAULT_MODEL


def _resolve_model() -> str:
    configured = os.environ.get(_ENV_MODEL_KEY, "").strip()
    if configured:
        return configured
    return _backend_default_model()


def summarize_commit(
    record: CommitRecord,
    *,
    diff_text: str = "",
    model: str | None = None,
) -> CommitSummary:
    """Summarise `record` using _PROMPT_V1.

    Args:
        record: CommitRecord from iter_commit_records().
        diff_text: Raw unified diff text. Truncated to ~8000 chars before
            sending to the model. Pass "" when diff is unavailable.
        model: Override model name. Defaults to LEMONCROW_LINEAGE_MODEL env var
            or the active backend's default model.

    Returns:
        CommitSummary with prompt_version="v1".

    Raises:
        SummarizerError: If the LLM call fails or returns an empty string.
    """
    effective_model = model or _resolve_model()
    truncated_diff = diff_text[:8000] if diff_text else "(no diff available)"
    prompt = _PROMPT_V1.format(message=record.message, diff=truncated_diff)
    messages = [{"role": "user", "content": prompt}]
    try:
        raw = chat(messages, model=effective_model)
    except Exception as exc:
        raise SummarizerError(f"LLM call failed for {record.sha[:8]}: {exc}") from exc
    raw_str = raw if isinstance(raw, str) else str(raw)
    if not raw_str.strip():
        raise SummarizerError(f"LLM returned empty summary for {record.sha[:8]}")
    return CommitSummary(
        sha=record.sha,
        author_date=record.author_date,
        files_touched=record.files_touched,
        summary=raw_str.strip(),
        summary_model=effective_model,
        prompt_version=_CURRENT_PROMPT_VERSION,
    )


__all__ = ["_CURRENT_PROMPT_VERSION", "_PROMPT_V1", "SummarizerError", "summarize_commit"]

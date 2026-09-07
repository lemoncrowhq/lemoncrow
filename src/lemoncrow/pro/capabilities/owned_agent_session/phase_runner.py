"""Phase-linear runner for owned agent sessions.

Runs Survey→Plan→Implement as one growing conversation, placing provider-aware
cache breakpoints so each phase can cache-hit the previous phase's context:

- Anthropic: explicit ``cache_control: {type: ephemeral}`` blocks on the system
  message and after Survey's assistant response.
- OpenAI: automatic prefix caching (no markers needed — stable prefix + seed).
- Gemini: server-side context cache (``cachedContent``) passed in ``extra_body``.
- Others via litellm: stable prefix only, no explicit caching markers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from lemoncrow.pro.capabilities.owned_agent_session.keepalive import KeepaliveThread
from lemoncrow.pro.capabilities.owned_agent_session.receipt import PhaseTokens, SessionReceipt
from lemoncrow.pro.capabilities.owned_agent_session.session import OwnedAgentSession
from lemoncrow.pro.capabilities.owned_agent_session.stem_prompt import STEM_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _phase_user_message(phase: str, task: str, mode: str = "code") -> str:
    """Build the user message for a phase, injecting mode context as a prefix.

    Mode context goes in the user turn, NOT in the system prompt.
    This keeps the system prefix byte-stable for cache reuse.
    """
    mode_prefix = {
        "code": "[MODE: code — read and edit files]",
        "explore": "[MODE: explore — read-only, no edits]",
        "research": "[MODE: research — read and search, no edits]",
        "plan": "[MODE: plan — read-only, planning focus]",
    }.get(mode, "[MODE: code]")

    phase_instructions = {
        "survey": f"Phase: Survey\n\nRead relevant files and understand the current codebase state for this task:\n{task}",
        "plan": "Phase: Plan\n\nBased on your survey, outline a precise implementation plan.",
        "implement": "Phase: Implement\n\nExecute the plan. Make the file edits.",
        "single": task,
    }

    instr = phase_instructions.get(phase, task)
    return f"{mode_prefix}\n\n{instr}"


def _provider_cache_style(provider: str, model: str = "") -> str:
    """Return cache strategy: 'anthropic', 'openai', 'gemini', or 'none'.

    When provider is 'openrouter', the upstream model string determines which
    cache strategy applies (OpenRouter routes to the underlying provider).
    """
    p = provider.lower()
    m = model.lower()

    # Direct Anthropic or Claude model
    if "anthropic" in p or "claude" in m:
        return "anthropic"
    # Bedrock Claude — same cache_control API
    if "bedrock" in p:
        return "anthropic" if "claude" in m or "anthropic" in m else "none"
    # OpenAI or Azure
    if "openai" in p or "azure" in p or "gpt" in m or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    # Gemini / Vertex
    if "gemini" in p or "google" in p or "vertex" in p or "gemini" in m:
        return "gemini"
    # OpenRouter — inspect the sub-model
    if "openrouter" in p:
        if "claude" in m or "anthropic" in m:
            return "anthropic"
        if "gpt" in m or "openai" in m:
            return "openai"
        if "gemini" in m or "google" in m:
            return "gemini"
        return "none"
    return "none"


def _system_message(provider: str, model: str = "") -> dict[str, Any]:
    """Build the stable system message.

    Anthropic gets ``cache_control`` embedded in the content list so
    ``_apply_cache_control`` in litellm_client is a no-op (content is already a
    list, not a string — the existing guard skips double-patching).
    All other providers receive a plain string system message.
    """
    cache_style = _provider_cache_style(provider, model)
    if cache_style == "anthropic":
        return {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": STEM_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    return {"role": "system", "content": STEM_SYSTEM_PROMPT}


def _assistant_with_breakpoint(content: str, *, provider: str, model: str = "") -> dict[str, Any]:
    """Return an assistant message; Anthropic gets an ephemeral breakpoint."""
    if _provider_cache_style(provider, model) == "anthropic":
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}],
        }
    return {"role": "assistant", "content": content}


@dataclass
class PhaseResult:
    phase: str
    content: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


def _call_llm(
    messages: list[dict[str, Any]],
    *,
    model: str,
    provider: str,
    gemini_cached_content: str | None = None,
) -> tuple[str, int, int, int, int]:
    """Call the LLM with provider-aware caching and return a 5-tuple.

    Returns ``(content, input_tokens, output_tokens, cache_read, cache_write)``.

    - Anthropic: ``chat_with_result`` (``cache_control`` blocks already in messages).
    - OpenAI: litellm directly with ``seed=42`` for deterministic prefix reuse
      (automatic prefix caching handles the rest — no explicit markers needed).
    - Gemini: litellm with ``extra_body={"cachedContent": name}`` when a context
      cache has been created for this session.
    - Others: litellm directly, stable prefix only.
    """
    cache_style = _provider_cache_style(provider, model)

    if cache_style == "anthropic":
        from lemoncrow.infra.internal_llm.litellm_client import chat_with_result

        result = chat_with_result(
            messages,
            model=model,
            api_key=os.environ.get("AWS_BEARER_TOKEN_BEDROCK") if provider.lower() == "bedrock" else None,
        )
        return (
            result.content,
            result.input_tokens,
            result.output_tokens,
            result.cache_read_input_tokens,
            result.cache_write_input_tokens,
        )

    # Non-Anthropic: route through the infra litellm wrapper with
    # provider-specific parameters so no provider SDK is imported on the user's
    # path outside src/lemoncrow/infra/internal_llm/.
    from lemoncrow.infra.internal_llm.litellm_client import LiteLLMUnavailable, chat_with_result

    extra_kwargs: dict[str, Any] = {}
    if cache_style == "openai":
        # Seed stabilises prefix for OpenAI automatic prefix caching
        extra_kwargs["seed"] = 42
    if cache_style == "gemini" and gemini_cached_content:
        extra_kwargs["extra_body"] = {"cachedContent": gemini_cached_content}

    try:
        result = chat_with_result(messages, model=model, extra_kwargs=extra_kwargs or None)
    except LiteLLMUnavailable as exc:
        raise RuntimeError("litellm is required for non-Anthropic providers") from exc
    except Exception as exc:
        raise RuntimeError(f"LLM call failed ({provider}/{model}): {exc}") from exc

    return (
        result.content,
        result.input_tokens,
        result.output_tokens,
        result.cache_read_input_tokens,
        result.cache_write_input_tokens,
    )


def run_phase_linear(
    session: OwnedAgentSession,
    task: str,
    *,
    dry_run: bool = False,
    gemini_cached_content: str | None = None,
) -> SessionReceipt:
    """Run Survey→Plan→Implement as one phase-linear conversation.

    Cache breakpoints (provider-specific):
    - System message: Anthropic gets ``cache_control: ephemeral``; others get
      a plain system string (Gemini uses ``cachedContent`` instead).
    - Post-Survey assistant response: Anthropic gets a second breakpoint so Plan
      cache-hits Survey's output.  OpenAI/Gemini rely on stable-prefix semantics.

    Args:
        session: The ``OwnedAgentSession`` (provider, model, phase_linear already set).
        task: The task description from the user.
        dry_run: If True, skip the Implement phase.
        gemini_cached_content: ``cachedContent`` name for Gemini context cache.
    """
    receipt = SessionReceipt(
        session_id=session.session_id,
        provider=session.provider,
        model=session.model,
    )

    working: list[dict[str, Any]] = [_system_message(session.provider, session.model)]
    phases = ["survey", "plan"] + ([] if dry_run else ["implement"])

    keepalive = KeepaliveThread(model=session.model, provider=session.provider)
    keepalive.start()
    try:
        for phase in phases:
            prompt = _phase_user_message(phase, task, mode=session.current_mode)
            working.append({"role": "user", "content": prompt})
            session.add_user_turn(prompt)
            session.current_phase = phase

            logger.debug("phase=%s provider=%s model=%s msgs=%d", phase, session.provider, session.model, len(working))

            content, inp, out, cr, cw = _call_llm(
                working,
                model=session.model,
                provider=session.provider,
                gemini_cached_content=gemini_cached_content,
            )

            # After Survey: mark assistant response with breakpoint (Anthropic) or plain
            mark = phase == "survey" and session.phase_linear
            if mark:
                turn = _assistant_with_breakpoint(content, provider=session.provider, model=session.model)
                working.append(turn)
                session.add_assistant_turn(
                    content, mark_breakpoint=_provider_cache_style(session.provider, session.model) == "anthropic"
                )
            else:
                working.append({"role": "assistant", "content": content})
                session.add_assistant_turn(content, mark_breakpoint=False)

            receipt.phases.append(
                PhaseTokens(
                    phase=phase,
                    input_tokens=inp,
                    output_tokens=out,
                    cache_read_tokens=cr,
                    cache_write_tokens=cw,
                )
            )
    finally:
        keepalive.stop()

    return receipt


def run_single_shot(
    session: OwnedAgentSession,
    task: str,
    *,
    dry_run: bool = False,
    gemini_cached_content: str | None = None,
) -> SessionReceipt:
    """Run a single-turn owned session (no phase-linear split)."""
    receipt = SessionReceipt(
        session_id=session.session_id,
        provider=session.provider,
        model=session.model,
    )
    messages: list[dict[str, Any]] = [
        _system_message(session.provider, session.model),
        {"role": "user", "content": _phase_user_message("single", task, mode=session.current_mode)},
    ]
    session.add_user_turn(task)

    if dry_run:
        receipt.phases.append(PhaseTokens(phase="dry_run"))
        return receipt

    keepalive = KeepaliveThread(model=session.model, provider=session.provider)
    keepalive.start()
    try:
        content, inp, out, cr, cw = _call_llm(
            messages,
            model=session.model,
            provider=session.provider,
            gemini_cached_content=gemini_cached_content,
        )
    finally:
        keepalive.stop()
    session.add_assistant_turn(content)
    receipt.phases.append(
        PhaseTokens(
            phase="single",
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cr,
            cache_write_tokens=cw,
        )
    )
    return receipt


__all__ = ["PhaseResult", "_provider_cache_style", "run_phase_linear", "run_single_shot"]

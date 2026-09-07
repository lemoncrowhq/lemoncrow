"""FailureAnalysisCapability - cluster failures and suggest fixes.

Implements a lightweight failure-analysis loop:
- cluster similar failures from historical traces
- derive root-cause hypotheses from shared error and command patterns
- suggest fixes using matched Playbooks
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from lemoncrow.core.foundation.models import CommandRecord, Trace
from lemoncrow.infra.storage.bundle import StoreBundle


@dataclass
class FailureIncident:
    fingerprint: str
    count: int
    trace_ids: list[str]
    sample_errors: list[str]
    common_commands: list[str]
    root_cause_hypothesis: str
    confidence: float
    suggested_playbooks: list[str]
    suggested_fixes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "count": self.count,
            "trace_ids": self.trace_ids,
            "sample_errors": self.sample_errors,
            "common_commands": self.common_commands,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "confidence": self.confidence,
            "suggested_playbooks": self.suggested_playbooks,
            "suggested_fixes": self.suggested_fixes,
        }


_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d+\b")
_QUOTED_RE = re.compile(r"(['\"]).*?\1")
_PATH_RE = re.compile(r"(/[^\s:]+)+")


def _normalise_error(text: str) -> str:
    value = " ".join((text or "").strip().lower().split())
    value = _UUID_RE.sub("<id>", value)
    value = _HEX_RE.sub("<hex>", value)
    value = _NUM_RE.sub("<n>", value)
    value = _QUOTED_RE.sub("<quoted>", value)
    value = _PATH_RE.sub("<path>", value)
    return value[:260]


def _tokenise(text: str) -> set[str]:
    tokens = re.findall(r"[a-z_]{3,}", text.lower())
    stop = {
        "the",
        "and",
        "with",
        "from",
        "this",
        "that",
        "error",
        "failed",
        "failure",
        "trace",
    }
    return {tok for tok in tokens if tok not in stop}


def _trace_command_text(item: str | CommandRecord) -> str:
    return item if isinstance(item, str) else item.command


class FailureAnalysisCapability:
    """Analyzes failures across traces and proposes actionable remediations."""

    def __init__(self, store: StoreBundle, context_reuse: Any) -> None:
        self._store = store
        self._context_reuse = context_reuse

    def analyze(
        self,
        *,
        domain: str | None = None,
        lookback: int = 200,
        min_cluster_size: int = 2,
    ) -> dict[str, Any]:
        traces = self._failed_traces(domain=domain, lookback=lookback)
        clusters = self._cluster_by_fingerprint(traces)

        incidents: list[FailureIncident] = []
        for fingerprint, group in clusters.items():
            if len(group) < min_cluster_size:
                continue
            incidents.append(self._build_incident(fingerprint, group))

        incidents.sort(key=lambda x: x.count, reverse=True)
        return {
            "total_failed_traces": len(traces),
            "cluster_count": len(incidents),
            "incidents": [i.to_dict() for i in incidents],
        }

    def analyze_for_error(
        self,
        *,
        task: str,
        error: str,
        domain: str | None = None,
        lookback: int = 200,
    ) -> dict[str, Any]:
        report = self.analyze(domain=domain, lookback=lookback, min_cluster_size=1)
        incidents = report.get("incidents", [])
        if not incidents:
            return {
                "matched": False,
                "reason": "no prior failed traces available",
                "suggested_fixes": self._fallback_fixes(task=task, error=error, domain=domain),
            }

        current_fp = _normalise_error(error)
        current_tokens = _tokenise(current_fp)

        best: dict[str, Any] | None = None
        best_score = -1.0
        for incident in incidents:
            fp = str(incident.get("fingerprint", ""))
            tokens = _tokenise(fp)
            overlap = len(current_tokens & tokens)
            union = max(1, len(current_tokens | tokens))
            score = overlap / union
            score += min(0.4, float(incident.get("count", 0)) / 20.0)
            if score > best_score:
                best_score = score
                best = incident

        suggested = self._fallback_fixes(task=task, error=error, domain=domain)
        if best:
            existing = list(best.get("suggested_fixes", []))
            for item in suggested:
                if item not in existing:
                    existing.append(item)
            best = dict(best)
            best["suggested_fixes"] = existing[:8]

        return {
            "matched": best is not None,
            "match_score": round(max(0.0, best_score), 3),
            "current_fingerprint": current_fp,
            "incident": best,
        }

    def _failed_traces(self, *, domain: str | None, lookback: int) -> list[Trace]:
        failed = self._store.history.list_traces(domain=domain, status="failed", limit=lookback)
        partial = self._store.history.list_traces(domain=domain, status="partial", limit=lookback)
        out: list[Trace] = list(failed)
        for trace in partial:
            if trace.errors_seen and trace.id not in {t.id for t in out}:
                out.append(trace)
        return out

    def _cluster_by_fingerprint(self, traces: list[Trace]) -> dict[str, list[Trace]]:
        groups: dict[str, list[Trace]] = defaultdict(list)
        for trace in traces:
            errors = [_normalise_error(e) for e in trace.errors_seen if e]
            if not errors:
                if trace.output_summary:
                    errors = [_normalise_error(trace.output_summary)]
                elif trace.diff_summary:
                    errors = [_normalise_error(trace.diff_summary)]
                else:
                    errors = ["unknown_failure"]
            # Use the most specific-looking fingerprint: longest normalized message.
            fingerprint = sorted(errors, key=len, reverse=True)[0]
            groups[fingerprint].append(trace)
        return groups

    def _build_incident(self, fingerprint: str, traces: list[Trace]) -> FailureIncident:
        command_counts: Counter[str] = Counter()
        error_samples: list[str] = []
        for trace in traces:
            command_counts.update(_trace_command_text(cmd) for cmd in trace.commands_run if cmd)
            for err in trace.errors_seen:
                if err and err not in error_samples:
                    error_samples.append(err)

        common_commands = [cmd for cmd, _ in command_counts.most_common(3)]
        root_cause = self._root_cause_hypothesis(fingerprint, common_commands, traces)

        # Use representative task/error pair to fetch reusable procedures.
        rep = traces[0]
        suggested_blocks = self._context_reuse.retrieve(
            task=rep.task,
            domain=rep.domain,
            errors=rep.errors_seen[:3],
            limit=3,
        )
        playbook_ids = [entry.block.id for entry in suggested_blocks]
        suggested_fixes = self._incident_fixes(suggested_blocks, common_commands)

        confidence = min(0.95, 0.45 + (len(traces) * 0.08))
        return FailureIncident(
            fingerprint=fingerprint,
            count=len(traces),
            trace_ids=[t.id for t in traces[:12]],
            sample_errors=error_samples[:5],
            common_commands=common_commands,
            root_cause_hypothesis=root_cause,
            confidence=round(confidence, 3),
            suggested_playbooks=playbook_ids,
            suggested_fixes=suggested_fixes,
        )

    def _root_cause_hypothesis(
        self,
        fingerprint: str,
        commands: list[str],
        traces: list[Trace],
    ) -> str:
        domain = traces[0].domain if traces else "unknown"
        if commands:
            return (
                f"Recurring {domain} failure matches fingerprint '{fingerprint}'. "
                f"Most common failing path includes command '{commands[0]}'. "
                "Likely unresolved precondition or invalid intermediate state."
            )
        return (
            f"Recurring {domain} failure matches fingerprint '{fingerprint}'. "
            "Likely unresolved precondition or missing guardrail before execution."
        )

    def _incident_fixes(self, suggested_blocks: list[Any], commands: list[str]) -> list[str]:
        fixes: list[str] = []
        for entry in suggested_blocks:
            block = entry.block
            if block.procedure:
                fixes.append(f"Apply Playbook '{block.title}': {block.procedure[0]}")
            if block.verification:
                fixes.append(f"Add verification gate: {block.verification[0]}")
        if commands:
            fixes.append(f"Add command guardrail around '{commands[0]}' with explicit pass/fail checks")
        fixes.append("Capture this failure signature in rubric checks to block silent retries")
        # Deduplicate while preserving order.
        out: list[str] = []
        for fix in fixes:
            if fix not in out:
                out.append(fix)
        return out[:8]

    def _fallback_fixes(self, *, task: str, error: str, domain: str | None) -> list[str]:
        scored = self._context_reuse.retrieve(
            task=task,
            domain=domain,
            errors=[error],
            limit=3,
        )
        fixes: list[str] = []
        for entry in scored:
            block = entry.block
            if block.procedure:
                fixes.append(f"Apply Playbook '{block.title}': {block.procedure[0]}")
        fixes.append("Stop retries after 2 repeats and run explicit rescue path")
        fixes.append("Persist normalized error fingerprint and include it in next prompt context")
        return fixes[:6]

"""Z-score anomaly detection for per-tool call frequencies."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from .models import AnomalyAlert

_WINDOW_SIZE = 50  # rolling window of per-batch call counts
_BATCH_SIZE = 5  # invocations aggregated into one count before windowing
_Z_WARNING = 2.5  # z-score threshold for WARNING
_Z_CRITICAL = 3.5  # z-score threshold for CRITICAL
_BURST_WINDOW = 10  # events to look back for burst detection
_BURST_THRESHOLD = 5  # same tool > N times in BURST_WINDOW is a burst


class ToolAnomalyDetector:
    """
    Detects abnormal tool usage via rolling Z-score analysis and burst detection.

    For each tool type, maintains a rolling window of call counts per
    observation batch.  When the current count is a statistical outlier
    (Z-score > threshold), an :class:`AnomalyAlert` is emitted.
    """

    def __init__(self) -> None:
        # tool_name → deque of per-batch call counts
        self._windows: dict[str, deque[int]] = {}
        # tool_name → total invocations seen (for reporting)
        self._totals: dict[str, int] = {}
        # per-tool counts accumulating in the in-progress observation batch
        self._batch_counts: dict[str, int] = {}
        self._batch_position = 0
        # flat event queue for burst detection
        self._recent_tools: deque[str] = deque(maxlen=_BURST_WINDOW)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def record(self, tool: str) -> None:
        """Record a single tool invocation.

        Invocations are grouped into fixed-size observation batches. When a
        batch closes, each tool's count within it is pushed onto that tool's
        rolling window (zero for tools idle this batch), giving a varying
        time series so the Z-score compares the latest batch against the
        tool's own history.
        """
        self._totals[tool] = self._totals.get(tool, 0) + 1
        self._batch_counts[tool] = self._batch_counts.get(tool, 0) + 1
        self._recent_tools.append(tool)
        self._batch_position += 1
        if self._batch_position >= _BATCH_SIZE:
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Push the current batch's per-tool counts onto each rolling window."""
        for tool in self._totals:
            window = self._windows.setdefault(tool, deque(maxlen=_WINDOW_SIZE))
            window.append(self._batch_counts.get(tool, 0))
        self._batch_counts.clear()
        self._batch_position = 0

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def z_score(self, tool: str) -> float | None:
        """Return Z-score of the most recent call batch, or None if insufficient data."""
        window = self._windows.get(tool)
        if not window or len(window) < 5:
            return None
        values = list(window)
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance)
        if std < 0.01:
            return None
        current = values[-1]
        return (current - mean) / std

    def detect_anomalies(self) -> list[AnomalyAlert]:
        """Scan all tracked tools and return a list of active anomaly alerts."""
        alerts: list[AnomalyAlert] = []

        for tool, window in self._windows.items():
            if len(window) < 5:
                continue
            z = self.z_score(tool)
            if z is None:
                continue
            if z >= _Z_CRITICAL:
                alerts.append(
                    AnomalyAlert(
                        tool=tool,
                        severity="critical",
                        message=f"tool '{tool}' called abnormally often (z={z:.2f})",
                        z_score=z,
                    )
                )
            elif z >= _Z_WARNING:
                alerts.append(
                    AnomalyAlert(
                        tool=tool,
                        severity="warning",
                        message=f"tool '{tool}' usage elevated (z={z:.2f})",
                        z_score=z,
                    )
                )

        # Burst detection (same tool > threshold times in recent window)
        burst_counts: dict[str, int] = {}
        for t in self._recent_tools:
            burst_counts[t] = burst_counts.get(t, 0) + 1
        for tool, count in burst_counts.items():
            if count >= _BURST_THRESHOLD:
                alerts.append(
                    AnomalyAlert(
                        tool=tool,
                        severity="warning",
                        message=f"burst: tool '{tool}' called {count}x in last {_BURST_WINDOW} events",
                        z_score=0.0,
                    )
                )

        return alerts

    def summary(self) -> dict[str, Any]:
        return {
            tool: {
                "total_calls": self._totals[tool],
                "z_score": self.z_score(tool),
            }
            for tool in self._totals
        }

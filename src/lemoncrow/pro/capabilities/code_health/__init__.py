"""Code health & history analytics (WS10: G15, G16, N17).

All capabilities here are additive, read-only analytics that fail open: any
broad recovery logs ``"Recovered from broad exception handler"`` and returns a
structurally-valid empty result rather than raising into the agent surface.

* G15 :mod:`doc_drift` -- cross-reference design docs against the live symbol
  index (``design_gaps`` = stale/missing doc symbols; ``verify_design`` =
  signature drift).
* G16 :mod:`pr_risk` -- fuse blast-radius + complexity + churn + test-gap into a
  0..1 PR-risk score (``pr_risk``); heuristic commit classification
  (``commit_provenance``).
* N17 :mod:`doc_index` -- opt-in heading-tree indexing of Markdown design docs
  into a SEPARATE retrievable store (does not change code-retrieval defaults).
"""

from __future__ import annotations

from lemoncrow.pro.capabilities.code_health.doc_drift import (
    DocDriftAnalyzer,
    design_gaps,
    verify_design,
)
from lemoncrow.pro.capabilities.code_health.doc_index import (
    DesignDocStore,
    doc_indexing_enabled,
    index_design_docs,
    recall_design_docs,
)
from lemoncrow.pro.capabilities.code_health.pr_risk import (
    classify_commit_message,
    commit_provenance,
    pr_risk,
)

__all__ = [
    "DesignDocStore",
    "DocDriftAnalyzer",
    "classify_commit_message",
    "commit_provenance",
    "design_gaps",
    "doc_indexing_enabled",
    "index_design_docs",
    "pr_risk",
    "recall_design_docs",
    "verify_design",
]

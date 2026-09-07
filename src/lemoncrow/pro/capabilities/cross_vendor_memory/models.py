"""Re-export of cross-vendor memory audit models.

The pydantic models are defined open in ``cross_vendor_memory_contract`` (data
contract, not IP; pydantic cannot be mypyc-compiled). This module keeps the
names importable at their original path so the compiled pro logic can
``from .models import AuditEvent``.
"""

from __future__ import annotations

from lemoncrow.core.capabilities.cross_vendor_memory_contract import AuditEvent as AuditEvent

__all__ = ["AuditEvent"]

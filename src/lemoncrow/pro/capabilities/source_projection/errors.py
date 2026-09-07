"""Projection error types — re-exported from the open contract.

The definitions live in
:mod:`lemoncrow.core.capabilities.source_projection_contract` (open, uncompiled)
so the projection algorithm modules in this package compile to ``.so``.
"""

from __future__ import annotations

from lemoncrow.core.capabilities.source_projection_contract import (
    MinifiedEditError,
    ProjectionEditError,
)

__all__ = ["MinifiedEditError", "ProjectionEditError"]

"""Re-export of the routing execution contract.

The ``RouteExecutionContract`` model and host-capability metadata are defined
open in ``quality_router_execution_contract`` (data contract + declarative host
metadata, not routing IP; pydantic cannot be mypyc-compiled). Kept importable at
the original path for the compiled pro logic.
"""

from __future__ import annotations

from lemoncrow.core.capabilities.quality_router_execution_contract import (
    ExecutionMode as ExecutionMode,
)
from lemoncrow.core.capabilities.quality_router_execution_contract import (
    RouteExecutionContract as RouteExecutionContract,
)
from lemoncrow.core.capabilities.quality_router_execution_contract import (
    route_execution_contract as route_execution_contract,
)

__all__ = ["ExecutionMode", "RouteExecutionContract", "route_execution_contract"]

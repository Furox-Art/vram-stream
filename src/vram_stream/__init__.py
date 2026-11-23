"""Memory-budgeted matrix multiplication for arrays backed by system memory."""

from .engine import DeviceUnavailableError, MatmulResult, matmul
from .planner import TilePlan, plan_matmul

__all__ = [
    "DeviceUnavailableError",
    "MatmulResult",
    "TilePlan",
    "matmul",
    "plan_matmul",
]

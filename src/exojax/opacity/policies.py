from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MemoryPolicy:
    """Optional policy to centralize memory/runtime knobs.

    When supplied to OpaPremodit, these values override ctor params.
    """

    allow_32bit: Optional[bool] = None
    nstitch: Optional[int] = None
    cutwing: Optional[float] = None


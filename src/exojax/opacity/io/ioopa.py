from __future__ import annotations

from typing import Any, Dict, Literal
import numpy as np

from exojax import __version__
from exojax.opacity.base import OpaCalc
from exojax.opacity.ckd.api import OpaCKD
from exojax.opacity.lpf.api import OpaDirect
from exojax.opacity.modit.api import OpaModit
from exojax.opacity.premodit.api import OpaPremodit
from exojax.opacity.premodit.ioopa import saveopa_premodit


def saveopa(
    opa: OpaCalc,
    path: str,
    *,
    format: Literal["zarr", "npz"] = "zarr",
    extra_meta: Dict[str, Any] | None = None,
    aux: Dict[str, Any] | None = None,
) -> None:
    """Generic entry point for persisting ``Opa*`` calculators to disk.

    Currently only :class:`OpaPremodit` is implemented; other calculators raise
    ``NotImplementedError`` placeholders (``saveopa_ckd``, ``saveopa_modit``,
    ``saveopa_direct``) to document the expected extension points.
    """
    if isinstance(opa, OpaPremodit):
        saveopa_premodit(
            opa,
            path,
            format=format,
            extra_meta=extra_meta,
            aux=aux,
        )
        return
    if isinstance(opa, OpaCKD):
        raise NotImplementedError(
            "saveopa_ckd is not implemented yet for OpaCKD instances."
        )
    if isinstance(opa, OpaModit):
        raise NotImplementedError(
            "saveopa_modit is not implemented yet for OpaModit instances."
        )
    if isinstance(opa, OpaDirect):
        raise NotImplementedError(
            "saveopa_direct is not implemented yet for OpaDirect instances."
        )
    raise TypeError(
        "saveopa does not support persisting instances of " f"{opa.__class__.__name__}."
    )



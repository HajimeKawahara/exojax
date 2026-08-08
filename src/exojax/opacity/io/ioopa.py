from __future__ import annotations

from typing import Any, Dict, Literal

from exojax.opacity.base import OpaCalc
from exojax.opacity.ckd.api import OpaCKD
from exojax.opacity.diffgrid.api import OpaDiffgrid
from exojax.opacity.diffgrid.io import saveopa_diffgrid
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

    Diffgrid and PreMODIT calculators can be saved as NPZ or Zarr archives.
    Other calculators raise ``NotImplementedError`` placeholders to document
    the expected extension points.
    """
    if isinstance(opa, OpaDiffgrid):
        saveopa_diffgrid(
            opa,
            path,
            format=format,
            extra_meta=extra_meta,
            aux=aux,
        )
        return
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


"""Prepare named molecular opacities on a common wavenumber grid."""

from collections.abc import Mapping
from numbers import Integral

import jax.numpy as jnp
import numpy as np

from exojax.database.contracts import MDBSnapshot


def _validate_names(mapping, name):
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError(f"{name} must be a nonempty mapping keyed by species names.")
    if any(not isinstance(key, str) or not key for key in mapping):
        raise ValueError(f"{name} keys must be nonempty strings.")


def _validate_grid(grid, name):
    grid = np.asarray(grid)
    if (
        grid.ndim != 1
        or grid.size == 0
        or not np.all(np.isfinite(grid))
        or np.any(grid <= 0)
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError(f"{name} must be a finite, positive, increasing 1D grid.")
    return grid


def validate_opacity_grids(opas):
    """Validate prepared line opacities before evaluating a mixture.

    Args:
        opas: Nonempty mapping of species names to prepared calculators with
            ``nu_grid``, ``molmass``, ``ready``, and ``xsmatrix(T, P)``. All
            wavenumber grids must agree exactly, including their values.
            ``xsmatrix`` must accept dynamic temperature and pressure profiles,
            as in OpaPremodit; fixed-pressure OpaDiffgrid is not supported.

    This host-side check reads only fixed calculator metadata. It can also run
    while tracing a function that captures ``opas`` in a closure. Keep the
    calculators and their grids unchanged after compiling that function.
    CKD tables require their own validation and mixing rule.
    """
    _validate_names(opas, "opas")
    reference = None
    for name, opa in opas.items():
        if getattr(opa, "method", None) == "ckd":
            raise ValueError(f"{name}: CKD tables require CKD mixing, not line addition.")
        if getattr(opa, "method", None) == "diffgrid":
            raise ValueError(
                f"{name}: OpaDiffgrid uses a fixed pressure grid; "
                "the mixture interface requires xsmatrix(T, P) with dynamic pressure."
            )
        if not getattr(opa, "ready", False) or not callable(
            getattr(opa, "xsmatrix", None)
        ):
            raise ValueError(f"{name}: a prepared line opacity calculator is required.")
        grid = _validate_grid(getattr(opa, "nu_grid", None), f"{name}.nu_grid")
        mass = np.asarray(getattr(opa, "molmass", np.nan))
        if mass.ndim != 0 or not np.isfinite(mass) or mass <= 0:
            raise ValueError(f"{name}.molmass must be a finite positive scalar.")
        if reference is not None and not np.array_equal(grid, reference):
            raise ValueError(f"{name}: all opacities must have identical nu_grid values.")
        reference = grid


class _ZeroOpacity:
    """Keep an explicitly empty line selection in a named mixture."""

    ready = True
    method = "zero"

    def __init__(self, nu_grid, molmass):
        self.nu_grid = nu_grid
        self.molmass = molmass

    def xsmatrix(self, temperature, pressure):
        return jnp.zeros(
            (jnp.shape(temperature)[0], len(self.nu_grid)),
            dtype=jnp.result_type(temperature, pressure, float),
        )


def _build_single_opa(mdb, nu_grid, **kwargs):
    """Adapt snapshots, current MDBs, and legacy custom MDBs to PreMODIT."""
    from exojax.opacity.premodit.api import OpaPremodit

    if isinstance(mdb, MDBSnapshot):
        return OpaPremodit.from_snapshot(mdb, nu_grid, **kwargs)
    if callable(getattr(mdb, "to_snapshot", None)):
        return OpaPremodit.from_mdb(mdb, nu_grid, **kwargs)
    return OpaPremodit(mdb=mdb, nu_grid=nu_grid, **kwargs)


def build_premodit(databases, nu_grid, *, on_empty="raise", **kwargs):
    """Build ready, named PreMODIT calculators for one spectral segment.

    Args:
        databases: Nonempty mapping of species names to MDBs or MDBSnapshots.
            The keys bind opacities to abundances; molecular masses come from
            the database payloads, not from parsing those keys.
        nu_grid: Common increasing wavenumber grid in cm-1.
        on_empty: ``"raise"`` (default) rejects an empty line payload.
            ``"zero"`` preserves its name and molecular mass with zero line
            opacity. It does not suppress database loading failures or other
            construction errors. Database loading has already happened here.
        **kwargs: Shared OpaPremodit constructor options. Supply
            ``auto_trange`` or ``manual_params`` to complete preparation.

    Returns:
        A new dictionary of prepared opacity calculators, keyed like
        ``databases``. Call this function outside JIT and capture the result
        in the spectral model. Abundances are supplied at evaluation time.

    Notes:
        A zero line contribution does not remove a species from atmospheric
        composition, mean molecular weight, or continuum calculations.
        To customize individual calculators or reuse saved opacities, build
        a dictionary directly and call :func:`validate_opacity_grids`.
    """
    _validate_names(databases, "databases")
    nu_grid = _validate_grid(nu_grid, "nu_grid").copy()
    if on_empty not in ("raise", "zero"):
        raise ValueError("on_empty must be 'raise' or 'zero'.")
    if kwargs.get("auto_trange") is None and kwargs.get("manual_params") is None:
        raise ValueError("Supply auto_trange or manual_params to prepare the opacities.")

    opas = {}
    for name, mdb in databases.items():
        if callable(getattr(mdb, "to_snapshot", None)):
            mdb = mdb.to_snapshot()
        if isinstance(mdb, MDBSnapshot):
            nu_lines = mdb.lines.nu_lines
            molmass = mdb.meta.molmass
        else:
            nu_lines = mdb.nu_lines
            molmass = mdb.molmass
        if np.size(nu_lines) == 0:
            if on_empty == "raise":
                raise ValueError(f"{name}: no selected lines; use on_empty='zero' explicitly.")
            opas[name] = _ZeroOpacity(nu_grid, molmass)
        else:
            opas[name] = _build_single_opa(mdb, nu_grid, **kwargs)
    validate_opacity_grids(opas)
    return opas


def multiopa_premodit(
    multimdb,
    nu_grid_list,
    auto_trange,
    nstitch_list=None,
    diffmode=0,
    dit_grid_resolution=0.2,
    allow_32bit=False,
):
    """Build legacy nested PreMODIT lists without a database-layer dependency.

    Each row of ``multimdb`` belongs to one grid in ``nu_grid_list``. A single
    NumPy grid is also accepted. ``nstitch_list`` contains one positive integer
    per grid. New code should use :func:`build_premodit` for named species.
    """
    grids = list(nu_grid_list) if isinstance(nu_grid_list, (list, tuple)) else [nu_grid_list]
    if len(multimdb) != len(grids):
        raise ValueError("multimdb and nu_grid_list must have the same number of segments.")
    stitching = [1] * len(grids) if nstitch_list is None else list(nstitch_list)
    if len(stitching) != len(grids) or any(
        not isinstance(n, Integral) or isinstance(n, bool) or n < 1 for n in stitching
    ):
        raise ValueError("nstitch_list must contain one positive integer per segment.")
    return [
        [
            _build_single_opa(
                mdb,
                grid,
                auto_trange=auto_trange,
                nstitch=nstitch,
                diffmode=diffmode,
                dit_grid_resolution=dit_grid_resolution,
                allow_32bit=allow_32bit,
            )
            for mdb in row
        ]
        for row, grid, nstitch in zip(multimdb, grids, stitching)
    ]

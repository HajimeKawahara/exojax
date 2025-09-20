from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass(frozen=True)
class PreMODITInfo:
    """Immutable holder for PreMODIT precomputed grids (value object).

    Mirrors the current opainfo tuple structure:
      (multi_index_uniqgrid, elower_grid, ngamma_ref_grid, n_Texp_grid, R, pmarray)
    """

    multi_index_uniqgrid: np.ndarray
    elower_grid: np.ndarray
    ngamma_ref_grid: np.ndarray
    n_Texp_grid: np.ndarray
    R: Union[float, np.ndarray]
    pmarray: np.ndarray

    def as_tuple(self):
        """Return a tuple identical in layout to legacy `opainfo`."""
        return (
            self.multi_index_uniqgrid,
            self.elower_grid,
            self.ngamma_ref_grid,
            self.n_Texp_grid,
            self.R,
            self.pmarray,
        )


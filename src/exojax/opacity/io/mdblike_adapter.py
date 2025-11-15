from typing import Optional
from dataclasses import dataclass
import numpy as np
from exojax.database.contracts import MDBSnapshot

@dataclass
class _MDBLikeFromSnapshot:
    """Minimal mdb-like adapter built from MDBSnapshot.

    Provides only the attribute surface used by OpaPremodit.__init__ so the
    old constructor path stays unchanged. All arrays are NumPy arrays.
    """

    dbtype: str
    molmass: float
    T_gQT: np.ndarray
    gQT: np.ndarray
    nu_lines: np.ndarray
    elower: np.ndarray
    line_strength_ref_original: np.ndarray
    # HITRAN-only (optional)
    isotope: Optional[int] = None
    uniqiso: Optional[np.ndarray] = None
    n_air: Optional[np.ndarray] = None
    gamma_air: Optional[np.ndarray] = None
    # ExoMol-only (optional)
    n_Texp: Optional[np.ndarray] = None
    alpha_ref: Optional[np.ndarray] = None
    # HITRAN-only bookkeeping of per-line isotope ids
    isoid: Optional[np.ndarray] = None

    @classmethod
    def from_snapshot(cls, snap: MDBSnapshot) -> "_MDBLikeFromSnapshot":
        meta = snap.meta
        lines = snap.lines
        isotope_ids = snap.isotope
        uniqiso = snap.uniqiso
        isotope_sel: Optional[int] = None

        if meta.dbtype == "hitran":
            # Preserve the scalar isotope selection (e.g., 1) expected by HitranPartitionProvider.
            if uniqiso is not None and len(uniqiso) > 0:
                unique_iso = np.unique(uniqiso)
                if unique_iso.size == 1:
                    isotope_sel = int(unique_iso[0])
                else:
                    isotope_sel = 0
            elif isotope_ids is not None and len(isotope_ids) > 0:
                unique_iso = np.unique(isotope_ids)
                if unique_iso.size == 1:
                    isotope_sel = int(unique_iso[0])
                else:
                    isotope_sel = 0

        return cls(
            dbtype=meta.dbtype,
            molmass=meta.molmass,
            T_gQT=meta.T_gQT,
            gQT=meta.gQT,
            nu_lines=lines.nu_lines,
            elower=lines.elower,
            line_strength_ref_original=lines.line_strength_ref_original,
            isotope=isotope_sel,
            uniqiso=uniqiso,
            n_air=snap.n_air,
            gamma_air=snap.gamma_air,
            n_Texp=snap.n_Texp,
            alpha_ref=snap.alpha_ref,
            isoid=isotope_ids,
        )

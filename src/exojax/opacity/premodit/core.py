import warnings
import numpy as np
from exojax.utils.constants import Patm, Tref_original


def _select_broadening_mode(broadening_parameter_resolution, dit_grid_resolution):
    if dit_grid_resolution is not None:
        warnings.warn(
            "dit_grid_resolution is not None. Ignoring broadening_parameter_resolution.",
            UserWarning,
        )
        broadening_parameter_resolution = {
            "mode": "manual",
            "value": dit_grid_resolution,
        }

    mode = broadening_parameter_resolution["mode"]
    val = broadening_parameter_resolution["value"]
    if mode == "manual":
        dit_grid_resolution = val
        single_broadening = False
        single_broadening_parameters = None
    elif mode == "single":
        dit_grid_resolution = None
        single_broadening = True
        if val is None:
            val = [None, None]
        single_broadening_parameters = val
    elif mode == "minmax":
        dit_grid_resolution = np.inf
        single_broadening = False
        single_broadening_parameters = None
    else:
        raise ValueError(
            "Unknown mode in broadening_parameter_resolution e.g. manual/single/minmax."
        )

    return (
        broadening_parameter_resolution,
        dit_grid_resolution,
        single_broadening,
        single_broadening_parameters,
    )


def _compute_common_broadening_parameters(mdb, Tref_broadening):
    """compute gamma_common for the database

    Args:
        mdb (mdb class): mdbExomol, mdbHitemp, mdbHitran
        Tref_broadening (float): reference temperature for broadening in Kelvin

    Notes:
        gamma (T) = (gamma at Tref_original) * (Tref_original/Tref_broadening)**n
        * (T/Tref_broadening)**-n * (P/1bar)

    Returns:
        n_Texp: temperature exponent for the broadening
        gamma_ref: reference gamma value
    """
    if mdb.dbtype == "hitran":
        print("OpaPremodit: gamma_air and n_air are used. gamma_ref = gamma_air/Patm")
        return (
            mdb.n_air,
            mdb.gamma_air * (Tref_original / Tref_broadening) ** (mdb.n_air) / Patm,
        )
    elif mdb.dbtype == "exomol":
        return mdb.n_Texp, mdb.alpha_ref * (Tref_original / Tref_broadening) ** (
            mdb.n_Texp
        )
    else:
        raise ValueError("Unknown database type.")

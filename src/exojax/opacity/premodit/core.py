"""Core functionality for PreMODIT opacity calculations."""

import warnings
from typing import Tuple, Optional, Dict, Any, Union, List

import numpy as np
from exojax.utils.constants import Patm, Tref_original


def _select_broadening_mode(
    broadening_parameter_resolution: Dict[str, Any],
    dit_grid_resolution: Optional[float],
) -> Tuple[Dict[str, Any], Optional[float], bool, Optional[List[Optional[float]]]]:
    """Select broadening mode configuration.

    Args:
        broadening_parameter_resolution: Dict with 'mode' and 'value' keys
        dit_grid_resolution: Optional override for dit grid resolution

    Returns:
        Tuple of (config_dict, dit_grid_res, single_broadening, single_params)

    Raises:
        ValueError: If mode is not recognized
    """
    # Handle deprecated dit_grid_resolution parameter
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

    # Process each mode
    if mode == "manual":
        return (
            broadening_parameter_resolution,
            val,  # dit_grid_resolution
            False,  # single_broadening
            None,  # single_broadening_parameters
        )
    elif mode == "single":
        single_params = val if val is not None else [None, None]
        return (
            broadening_parameter_resolution,
            None,  # dit_grid_resolution
            True,  # single_broadening
            single_params,  # single_broadening_parameters
        )
    elif mode == "minmax":
        return (
            broadening_parameter_resolution,
            np.inf,  # dit_grid_resolution
            False,  # single_broadening
            None,  # single_broadening_parameters
        )
    else:
        raise ValueError(
            f"Unknown mode '{mode}' in broadening_parameter_resolution. "
            "Supported modes: manual, single, minmax"
        )


def _compute_common_broadening_parameters(
    mdb, Tref_broadening: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute gamma_common for the database.

    Args:
        mdb: Molecular database (mdbExomol, mdbHitemp, mdbHitran)
        Tref_broadening: Reference temperature for broadening in Kelvin

    Notes:
        gamma(T) = (gamma at Tref_original) * (Tref_original/Tref_broadening)^n
                  * (T/Tref_broadening)^(-n) * (P/1bar)

    Returns:
        Tuple of (n_Texp, gamma_ref) for temperature exponent and reference gamma

    Raises:
        ValueError: If database type is not supported
    """
    dbtype = mdb.dbtype

    if dbtype == "hitran":
        print("OpaPremodit: gamma_air and n_air are used. gamma_ref = gamma_air/Patm")
        n_Texp = mdb.n_air
        gamma_ref = (
            mdb.gamma_air * (Tref_original / Tref_broadening) ** mdb.n_air / Patm
        )
        return n_Texp, gamma_ref

    elif dbtype == "exomol":
        n_Texp = mdb.n_Texp
        gamma_ref = mdb.alpha_ref * (Tref_original / Tref_broadening) ** mdb.n_Texp
        return n_Texp, gamma_ref

    else:
        raise ValueError(
            f"Unknown database type: '{dbtype}'. " "Supported types: hitran, exomol"
        )

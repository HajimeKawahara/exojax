"""Core functionality for PreMODIT opacity calculations."""

import warnings
from typing import Tuple, Optional, Dict, Any, List

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


def _compute_broadening_parameters_hitran(
    n_air: np.ndarray,
    gamma_air: np.ndarray,
    Tref_broadening: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute HITRAN broadening parameters at a reference temperature.

    Args:
        n_air: air temperature exponent
        gamma_air: gamma factor of air pressure broadening
        Tref_broadening: Reference temperature for broadening in Kelvin.

    Notes:
        gamma(T) = (gamma at Tref_original) * (Tref_original/Tref_broadening)^n
                  * (T/Tref_broadening)^(-n) * (P/1bar)

    Returns:
        Tuple of (n_Texp, gamma_ref) for temperature exponent and reference gamma
    """
    print("OpaPremodit: gamma_air and n_air are used. gamma_ref = gamma_air/Patm")

    n_Texp = n_air
    gamma_ref = gamma_air * (Tref_original / Tref_broadening) ** n_air / Patm
    return n_Texp, gamma_ref


def _compute_broadening_parameters_exomol(
    n_Texp: np.ndarray,
    alpha_ref: np.ndarray,
    Tref_broadening: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ExoMol broadening parameters at a reference temperature.

    Args:
        n_Texp : temperature exponent
        alpha_ref : broadening parameter
        Tref_broadening: Reference temperature for broadening in Kelvin.

    Notes:
        gamma(T) = (gamma at Tref_original) * (Tref_original/Tref_broadening)^n
                  * (T/Tref_broadening)^(-n) * (P/1bar)

    Returns:
        Tuple of (n_Texp, gamma_ref) for temperature exponent and reference gamma
    """
    gamma_ref = alpha_ref * (Tref_original / Tref_broadening) ** n_Texp
    return n_Texp, gamma_ref

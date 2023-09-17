"""Atmospheric profile function."""

from exojax.utils.constants import kB, m_u
import jax.numpy as jnp
import numpy as np
from jax.lax import scan
from jax import jit


def pressure_layer_logspace(log_pressure_top=-8.,
                            log_pressure_btm=2.,
                            nlayer=20,
                            mode='ascending',
                            reference_point=0.5,
                            numpy=False):
    """ Pressure layer evenly spaced in logspace, i.e. logP interval is constant 

    Args:
       log_pressure_top: log10(P[bar]) at the top layer
       log_pressure_btm: log10(P[bar]) at the bottom layer
       nlayer: the number of the layers
       mode: ascending or descending
       reference_point (float): reference point in a layer (0-1). Center:0.5, lower boundary:1.0, upper boundary:0
       numpy: if True use numpy array instead of jnp array

    Returns:
         pressure: pressure layer
         dParr: delta pressure layer
         pressure_decrease_rate: pressure decrease rate of the layer (k-factor; k < 1) pressure[i-1] = pressure_decrease_rate*pressure[i]

    Note:
        d logP is constant using this function. 
    """
    dlogP = (log_pressure_btm - log_pressure_top) / (nlayer - 1)
    if numpy:
        pressure = np.logspace(log_pressure_top, log_pressure_btm, nlayer)
    else:
        pressure = jnp.logspace(log_pressure_top, log_pressure_btm, nlayer)

    k = 10**-dlogP
    dParr = (k**(reference_point - 1.0) - k**reference_point) * pressure
    
    if mode == 'descending':
        pressure = pressure[::-1]
        dParr = dParr[::-1]

    return pressure, dParr, k


@jit
def normalized_layer_height(temperature, pressure_decrease_rate,
                            mean_molecular_weight, radius_btm, gravity_btm):
    """compute normalized height/radius at the upper boundary of the atmospheric layer, neglecting atmospheric mass. 

    Args:
        temperature (1D array): temperature profile (K) of the layer, (Nlayer, from atmospheric top to bottom)
        pressure_decrease_rate:  pressure decrease rate of the layer (k-factor; k < 1) pressure[i-1] = pressure_decrease_rate*pressure[i]
        mean_molecular_weight (1D array): mean molecular weight profile, (Nlayer, from atmospheric top to bottom) 
        radius_btm (float): radius (cm) at the lower boundary of the bottom layer, R0 or r_N
        gravity_btm (float): gravity (cm/s2) at the lower boundary of the bottom layer, g_N

    Returns:
        1D array (Nlayer) : layer height normalized by radius_btm starting from top atmosphere
        1D array (Nlayer) : radius at lower bondary normalized by radius_btm starting from top atmosphere
    """
    inverse_Tarr = temperature[::-1]
    inverse_mmr_arr = mean_molecular_weight[::-1]
    Mat = jnp.vstack([inverse_Tarr, inverse_mmr_arr]).T

    def compute_radius(normalized_radius_lower, arr):
        T_layer = arr[0:1][0]
        mmw_layer = arr[1:2][0]
        gravity_lower = gravity_btm / normalized_radius_lower
        Hn_lower = pressure_scale_height(gravity_lower, T_layer,
                                         mmw_layer) / radius_btm
        fac = pressure_decrease_rate**(-Hn_lower /
                                       normalized_radius_lower) - 1.0
        normalized_height_layer = fac * normalized_radius_lower
        carry = normalized_radius_lower + normalized_height_layer
        return carry, [normalized_height_layer, normalized_radius_lower]

    _, results = scan(compute_radius, 1.0, Mat)
    normalized_height = results[0][::-1]
    normalized_radius_lower = results[1][::-1]
    return normalized_height, normalized_radius_lower


def gh_product(temperature, mean_molecular_weight):
    """prodict of gravity and pressure scale height

    Args:
        temperature: isothermal temperature (K)
        mean_molecular_weight: mean molecular weight

    Returns:
        gravity x pressure scale height cm2/s2
    """
    return kB * temperature / (m_u * mean_molecular_weight)


def pressure_scale_height(gravity, T, mean_molecular_weight):
    """pressure scale height assuming an isothermal atmosphere.

    Args:
        gravity: gravity acceleration (cm/s2)
        T: isothermal temperature (K)
        mean_molecular_weight: mean molecular weight

    Returns:
        pressure scale height (cm)
    """

    return gh_product(T, mean_molecular_weight) / gravity


def atmprof_powerlow(pressure, T0, alpha):
    """powerlaw temperature profile

    Args:
        pressure: pressure array (bar)
        T0 (float): T at P=1 bar in K
        alpha (float): powerlaw index

    Returns:
        array: temperature profile
    """
    return T0 * pressure**alpha


def atmprof_gray(pressure, gravity, kappa, Tint):
    """

    Args:
        pressure (1D array): pressure array (bar)
        gravity (float): gravity (cm/s2)
        kappa: infrared opacity 
        Tint: temperature equivalence of the intrinsic energy flow

    Returns:
        array: temperature profile

    """

    tau = pressure * 1.e6 * kappa / gravity
    Tarr = (0.75 * Tint**4 * (2.0 / 3.0 + tau))**0.25
    return Tarr


def atmprof_Guillot(pressure, gravity, kappa, gamma, Tint, Tirr, f=0.25):
    """

    Notes:
        Guillot (2010) Equation (29)

    Args:
        pressure: pressure array (bar)
        gravity: gravity (cm/s2)
        kappa: thermal/IR opacity (kappa_th in Guillot 2010)
        gamma: ratio of optical and IR opacity (kappa_v/kappa_th), gamma > 1 means thermal inversion
        Tint: temperature equivalence of the intrinsic energy flow
        Tirr: temperature equivalence of the irradiation
        f = 1 at the substellar point, f = 1/2 for a day-side average 
            and f = 1/4 for an averaging over the whole planetary surface

    Returns:
        array: temperature profile

    """
    tau = pressure * 1.e6 * kappa / gravity  # Equation (51)
    invsq3 = 1.0 / jnp.sqrt(3.0)
    fac = 2.0 / 3.0 + invsq3 * (
        1.0 / gamma + (gamma - 1.0 / gamma) * jnp.exp(-gamma * tau / invsq3))
    Tarr = (0.75 * Tint**4 * (2.0 / 3.0 + tau) +
            0.75 * Tirr**4 * f * fac)**0.25

    return Tarr


def Teq2Tirr(Teq):
    """Tirr from equilibrium temperature and intrinsic temperature.

    Args:
       Teq: equilibrium temperature
       
    Return:
       Tirr: iradiation temperature

    Note:
       Here we assume A=0 (albedo) and beta=1 (fully-energy distributed)
    """
    return (2.0**0.5) * Teq


def Teff2Tirr(Teff, Tint):
    """Tirr from effective temperature and intrinsic temperature.

    Args:
       Teff: effective temperature
       Tint: intrinsic temperature

    Return:
       Tirr: iradiation temperature

    Note:
       Here we assume A=0 (albedo) and beta=1 (fully-energy distributed)
    """
    return (4.0 * Teff**4 - Tint**4)**0.25

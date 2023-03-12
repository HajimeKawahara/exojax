"""Atmospheric profile function."""

from exojax.utils.constants import kB, m_u
import jax.numpy as jnp
import numpy as np
from jax.lax import scan
from jax import jit


def pressure_layer_logspace(log_pressure_top=-8.,
                            log_pressure_btm=2.,
                            NP=20,
                            mode='ascending',
                            numpy=False):
    """generating the pressure layer.

    Args:
       log_pressure_top: log10(P[bar]) at the top layer
       log_pressure_btm: log10(P[bar]) at the bottom layer
       NP: the number of the layers
       mode: ascending or descending
       numpy: if True use numpy array instead of jnp array

    Returns:
         pressure: pressure layer
         dParr: delta pressure layer
         k: k-factor, P[i-1] = k*P[i]

    Note:
        dParr[i] = Parr[i] - Parr[i-1], dParr[0] = (1-k) Parr[0] for ascending mode
    """
    dlogP = (log_pressure_btm - log_pressure_top) / (NP - 1)
    k = 10**-dlogP
    if numpy:
        pressure = np.logspace(log_pressure_top, log_pressure_btm, NP)
    else:
        pressure = jnp.logspace(log_pressure_top, log_pressure_btm, NP)
    dParr = (1.0 - k) * pressure
    if mode == 'descending':
        pressure = pressure[::-1]
        dParr = dParr[::-1]

    return pressure, dParr, k

@jit
def normalized_layer_height(temperature, pressure, dParr,
                            mean_molecular_weight, radius_btm, gravity_btm):
    """compute normalized height/radius at the upper boundary of the atmospheric layer, neglecting atmospheric mass. 

    Args:
        temperature (1D array): temperature profile (K) of the layer, (Nlayer, from atmospheric top to bottom)
        pressure (1D array): pressure profile (bar) of the layer, (Nlayer, from atmospheric top to bottom)
        dParr (1D array): pressure difference profile (bar) of the layer, (Nlayer, from atmospheric top to bottom)
        mean_molecular_weight (1D array): mean molecular weight profile, (Nlayer, from atmospheric top to bottom) 
        radius_btm (float): radius (cm) at the lower boundary of the bottom layer, R0 or r_N
        gravity_btm (float): gravity (cm/s2) at the lower boundary of the bottom layer, g_N

    Returns:
        1D array (Nlayer) : layer height normalized by radius_btm starting from top atmosphere
        1D array (Nlayer) : radius normalized by radius_btm starting from top atmosphere
    """

    inverse_Tarr = temperature[::-1]
    inverse_dlogParr = (dParr / pressure)[::-1]
    inverse_mmr_arr = mean_molecular_weight[::-1]
    Mat = jnp.vstack([inverse_Tarr, inverse_dlogParr, inverse_mmr_arr]).T

    def compute_radius(normalized_radius, arr):
        T_layer = arr[0:1][0]
        dlogP_layer = arr[1:2][0]
        mmw_layer = arr[2:3][0]
        gravity_layer = gravity_btm / normalized_radius
        normalized_height_layer = pressure_scale_height(
            gravity_layer, T_layer, mmw_layer) * dlogP_layer / radius_btm
        return normalized_radius + normalized_height_layer, [normalized_height_layer, normalized_radius]

    _, results = scan(compute_radius, 1.0, Mat)
    normalized_height = results[0][::-1]
    normalized_radius = results[1][::-1]
    return normalized_height, normalized_radius 


def pressure_scale_height(g, T, mu):
    """pressure scale height assuming an isothermal atmosphere.

    Args:
        g: gravity acceleration (cm/s2)
        T: isothermal temperature (K)
        mu: mean molecular weight

    Returns:
        pressure scale height (cm)
    """

    return kB * T / (m_u * mu * g)


def atmprof_powerlow(Parr, T0, alpha):
    """powerlaw temperature profile

    Args:
        Parr: pressure array (bar)
        T0 (float): T at P=1 bar in K
        alpha (float): powerlaw index

    Returns:
        array: temperature profile
    """
    return T0 * (Parr)**alpha


def atmprof_gray(Parr, g, kappa, Tint):
    """

    Args:
        Parr: pressure array (bar)
        g: gravity (cm/s2)
        kappa: infrared opacity 
        Tint: temperature equivalence of the intrinsic energy flow

    Returns:
        array: temperature profile

    """

    tau = Parr * 1.e6 * kappa / g
    Tarr = (0.75 * Tint**4 * (2.0 / 3.0 + tau))**0.25
    return Tarr


def atmprof_Guillot(Parr, g, kappa, gamma, Tint, Tirr, f=0.25):
    """

    Notes:
        Guillot (2010) Equation (29)

    Args:
        Parr: pressure array (bar)
        g: gravity (cm/s2)
        kappa: thermal/IR opacity (kappa_th in Guillot 2010)
        gamma: ratio of optical and IR opacity (kappa_v/kappa_th), gamma > 1 means thermal inversion
        Tint: temperature equivalence of the intrinsic energy flow
        Tirr: temperature equivalence of the irradiation
        f = 1 at the substellar point, f = 1/2 for a day-side average 
            and f = 1/4 for an averaging over the whole planetary surface

    Returns:
        array: temperature profile

    """
    tau = Parr * 1.e6 * kappa / g  # Equation (51)
    invsq3 = 1.0 / jnp.sqrt(3.0)
    fac = 2.0 / 3.0 + invsq3 * (
        1.0 / gamma + (gamma - 1.0 / gamma) * jnp.exp(-gamma * tau / invsq3))
    Tarr = (0.75 * Tint**4 * (2.0 / 3.0 + tau) +
            0.75 * Tirr**4 * f * fac)**0.25

    return Tarr


def Teq2Tirr(Teq, Tint):
    """Tirr from equilibrium temperature and intrinsic temperature.

    Args:
       Teq: equilibrium temperature
       Tint: intrinsic temperature

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

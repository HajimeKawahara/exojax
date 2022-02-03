"""Viscosity of droplets."""

import numpy as np


def eta_Rosner_H2(T):
    """dynamic viscocity of the H2 atmosphere by Rosner (2000)

    Args:
       T: temperature (K) (applicable from 179 to 11940K)

    Returns:
       dynamic viscosity (g/s/cm)
    """
    return eta_Rosner(T, 2.02072025e-6)


def eta_Rosner(T, vfactor):
    """dynamic viscocity by Rosner (2000)

    Args:
       T: temperature (K)
       vfactor: vfactor

    Returns:
       dynamic viscosity (g/s/cm)
    """
    eta = vfactor*(T**0.66)
    return eta


def get_LJPparam():
    """Lennard-Jones Potential Parameters.

    Returns:
       LJPparam_d: Dict for Lennard-Jones Potential Parameters (d (cm))
       LJPparam_epsilon_per_kB: Dict for Lennard-Jones Potential Parameters (epsilon/kB)

    Note:
       Lennard-Jones Potential Parameters (LJPparam) were taken from p107, Table 3.2.1 of Transport process in chemically reacting flow systems by Daniel E. Rosner, originally from Svehla (1962).
    """
    LJPparam_d = {}
    LJPparam_epsilon_per_kB = {}

    LJPparam_d['H2'] = 2.827e-8
    LJPparam_epsilon_per_kB['H2'] = 59.7

    LJPparam_d['He'] = 2.551e-8
    LJPparam_epsilon_per_kB['He'] = 10.22

    LJPparam_d['N2'] = 3.798e-8
    LJPparam_epsilon_per_kB['N2'] = 71.4

    LJPparam_d['CO2'] = 3.941e-8
    LJPparam_epsilon_per_kB['CO2'] = 195.2

    LJPparam_d['H2O'] = 2.641e-8
    LJPparam_epsilon_per_kB['H2O'] = 809.1

    LJPparam_d['CH4'] = 3.758e-8
    LJPparam_epsilon_per_kB['CH4'] = 148.6

    LJPparam_d['CO'] = 3.69e-8
    LJPparam_epsilon_per_kB['CO'] = 91.7

    LJPparam_d['O2'] = 3.467e-8
    LJPparam_epsilon_per_kB['O2'] = 106.7

    LJPparam_d['Air'] = 3.711e-8
    LJPparam_epsilon_per_kB['Air'] = 78.6

    return LJPparam_d, LJPparam_epsilon_per_kB


def calc_vfactor(atm='H2', LJPparam=None):
    """

    Args:
       atm: molecule consisting of atmosphere, "H2", "O2", and "N2" 
       LJPparam: Custom Lennard-Jones Potential Parameters (d (cm) and epsilon/kB)

    Returns:
       vfactor: dynamic viscosity factor for Rosner eta = viscosity*T**0.66
       applicable tempature range (K,K) 

    Note:
       The dynamic viscosity is from the Rosner book (3-2-12) and caption in p106 Hirschfelder et al. (1954) within Trange.


    """
    from exojax.spec.molinfo import molmass
    from exojax.utils.constants import kB, m_u

    mu = molmass(atm)

    if LJPparam is None:
        LJPparam_d, LJPparam_epsilon_per_kB = get_LJPparam()
        epsilon_per_kB = LJPparam_epsilon_per_kB[atm]
        d = LJPparam_d[atm]
    else:
        epsilon_per_kB = LJPparam[0]
        d = LJPparam[1]

    vfactor = 5.0/16.0*np.sqrt(np.pi*kB*mu*m_u) / \
        (np.pi*d*d)/1.22*(1.0/epsilon_per_kB)**0.16
    Trange = [3.0*epsilon_per_kB, 200.0*epsilon_per_kB]

    return vfactor, Trange


if __name__ == '__main__':

    vfactor, Tr = calc_vfactor('H2')
    dvisc = eta_Rosner(1500.0, vfactor)
    print(vfactor, dvisc)
    dvisc = eta_Rosner_H2(1500.0)
    print(dvisc)

"""Terminal velocity of cloud particles.

Note:
   The code in this module is based on Hans R Pruppacher and James D Klett. Microstructure of atmospheric clouds and precipitation and Akerman and Marley 2001.
"""

import jax.numpy as jnp
from jax import jit


def vf_stokes(r, g, eta, drho, Nkn=0.):
    """terminal velocity of Stokes flow (Reynolds number < 2, Davies number <
    42)

    Args:
        r: particle size (cm)
        g: gravity (cm/s2)
        eta: dynamic viscosity (g/s/cm)
        drho: density difference between condensates and atmosphere (g/cm3)
        Nkn: Knudsen number


    Return:
        terminal velocity (cm/s)

    Note:
        (1.0+1.255*Nkn) is the Cunningham factor

    Note:
       See also (10-138) p415 in Hans R Pruppacher and James D Klett. Microstructure of atmospheric clouds and precipitation. In Microphysics of clouds and precipitation, pages 10–73. Springer, 2010. Equation (B1) in Appendix B of Ackerman and Marley 2001.
    """
    return 2.0*g*r*r*drho*(1.0+1.255*Nkn)/(9.0*eta)


def Ndavies(r, g, eta, drho, rho):
    """Davies (Best) number.

    Args:
        r: particle size (cm)
        g: gravity (cm/s2)
        eta: dynamic viscosity (g/s/cm)
        drho: density difference between condensates and atmosphere (g/cm3)
        rho: atmosphere density (g/cm3)

    Returns:
       Davies number (Best Number)
    """
    return 32.0*g*r**3*drho*rho/(3.0*eta**2)


def vf_midNre(r, g, eta, drho, rho):
    """terminal velocity (2 < Reynolds number < 500, 42 < Davies number <
    10**5)

    Args:
        r: particle size (cm)
        g: gravity (cm/s2)
        eta: dynamic viscosity (g/s/cm)
        drho: density difference between condensates and atmosphere (g/cm3)
        rho: atmosphere density (g/cm3)

    Return:
        terminal velocity (cm/s)
    """
    ND = Ndavies(r, g, eta, drho, rho)
    x = jnp.log(ND)
    logNre = -0.0088*x**2 + 0.85*x - 2.49
    return eta/(2.0*rho*r)*jnp.exp(logNre)


def vf_largeNre(r, g, eta, drho, rho):
    """terminal velocity  ( Reynolds number > 500, Davies number >10**5 )

    Args:
        r: particle size (cm)
        g: gravity (cm/s2)
        eta: dynamic viscosity (g/s/cm)
        drho: density difference between condensates and atmosphere (g/cm3)
        rho: atmosphere density (g/cm3)

    Return:
        terminal velocity (cm/s)
    """
    ND = Ndavies(r, g, eta, drho, rho)
    Cd = 0.45
    return eta/(2.0*rho*r)*jnp.sqrt(ND/Cd)


@jit
def vf(r, g, eta, rhoc, rho, Nkn=0.0):
    """terminal velocity.

    Args:
        r: particle size (cm)
        g: gravity (cm/s2)
        eta: dynamic viscosity (g/s/cm)
        rhoc: condensate density (g/cm3)
        rho: atmosphere density (g/cm3)
        Nkn: Knudsen number

    Return:
        terminal velocity (cm/s)    

    Example:

        >>> #terminal velocity at T=300K, for Earth atmosphere/gravity.
        >>> g=980.
        >>> drho=1.0
        >>> rho=1.29*1.e-3 #g/cm3
        >>> vfactor,Tr=vc.calc_vfactor(atm="Air")
        >>> eta=vc.eta_Rosner(300.0,vfactor)
        >>> r=jnp.logspace(-5,0,70)
        >>> vf(r,g,eta,drho,rho) #terminal velocity (cm/s)
    """
    drho = rhoc-rho
    ND = Ndavies(r, g, eta, drho, rho)
    cond = [ND < 42.877543, (ND >= 42.877543) *
            (ND < 119643.38), ND >= 119643.38]
    choice = [vf_stokes(r, g, eta, drho, Nkn), vf_midNre(
        r, g, eta, drho, rho), vf_largeNre(r, g, eta, drho, rho)]
    vft = jnp.select(cond, choice)
    return vft


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import exojax.atm.viscosity as vc

    g = 980.
    drho = 1.0
    rho = 1.29*1.e-3  # g/cm3
    vfactor, Tr = vc.calc_vfactor(atm='Air')
    eta = vc.eta_Rosner(300.0, vfactor)
    r = jnp.logspace(-5, 0, 70)
    plt.figure(figsize=(5, 3))
    plt.plot(r*1.e4, vf(r, g, eta, drho, rho), color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('r (micron)')
    plt.ylabel('terminal velocity (cm/s)')
    plt.savefig('vterm.pdf', bbox_inches='tight', pad_inches=0.0)
    plt.show()

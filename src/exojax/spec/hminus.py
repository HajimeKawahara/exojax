"""H minus opacity by John (1988)"""

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan
from exojax.utils.constants import kB, ccgs, hcgs


def log_hminus_continuum(nus, temperature, number_density_e, number_density_h):
    """John (1988) H- continuum opacity.

    Args:
       nus: wavenumber grid (cm-1) [Nnu]
       temperature: gas temperature array [K] [Nlayer]
       number_density_e: electron number density array [Nlayer]
       number_density_h: H atom number density array [Nlayer]

    Returns:
       log10(absorption coefficient) [Nlayer,Nnu]
    """
    # wavelength in units of microns
    wavelength_um = 1e4/nus
    # first, compute the cross sections (in cm4/dyne)
    vkappa_bf = vmap(bound_free_absorption, (None, 0), 0)
    vkappa_ff = vmap(free_free_absorption, (None, 0), 0)
    mkappa_bf = vmap(vkappa_bf, (0, None), 0)
    mkappa_ff = vmap(vkappa_ff, (0, None), 0)
    kappa_bf = mkappa_bf(wavelength_um, temperature)
    kappa_ff = mkappa_ff(wavelength_um, temperature)
#    kappa_bf = bound_free_absorption(wavelength_um, temperature)
#    kappa_ff = free_free_absorption(wavelength_um, temperature)

    electron_pressure = number_density_e * kB * \
        temperature  # //electron pressure in dyne/cm2
    hydrogen_density = number_density_h

    # and now finally the absorption_coeff (in cm-1)
    absorption_coeff = (kappa_bf + kappa_ff) * \
        electron_pressure * hydrogen_density

    return jnp.log10(absorption_coeff.T)


def bound_free_absorption(wavelength_um, temperature):
    """bound free absorption of H-

    Note:
       alpha has a value of 1.439e4 micron-1 K-1, the value stated in John (1988) is wrong

    Args:
        wavelength_um: wavelength in the unit of micron
        temperature: temperature in the unit of Kelvin

    Returns:
        absorption coefficient [cm4/dyne]
    """
    # here, we express alpha using physical constants
    alpha = ccgs*hcgs/kB*10000.0
    lambda_0 = 1.6419  # photo-detachment threshold

    #   //tabulated constant from John (1988)
    def f(wavelength_um):
        C_n = jnp.vstack(
            [jnp.arange(7), [0.0, 152.519, 49.534, -
                             118.858, 92.536, -34.194, 4.982]]
        ).T

        def body_fun(val, x):
            i, C_n_i = x
            return val, val + C_n_i * jnp.power(jnp.clip(1.0/wavelength_um - 1.0/lambda_0, a_min=0, a_max=None), (i-1)/2.0)

        return scan(body_fun, jnp.zeros_like(wavelength_um), C_n)[-1].sum(0)

    # first, we calculate the photo-detachment cross-section (in cm2)
    kappa_bf = (1e-18 * wavelength_um ** 3 *
                jnp.power(jnp.clip(1.0/wavelength_um - 1.0/lambda_0,
                          a_min=0, a_max=None), 1.5) * f(wavelength_um)
                )

    kappa_bf = jnp.where(
        (wavelength_um <= lambda_0) & (wavelength_um > 0.125),
        (0.750 * jnp.power(temperature, -2.5) * jnp.exp(alpha / lambda_0 / temperature) *
         (1.0 - jnp.exp(-alpha / wavelength_um / temperature)) * kappa_bf),
        0
    )
    return kappa_bf


def free_free_absorption(wavelength_um, temperature):
    """free free absorption of H- (coefficients from John (1988))

    Note:
       to follow his notation (which starts at an index of 1), the 0-index components are 0 for wavelengths larger than 0.3645 micron

    Args:
        wavelength_um: wavelength in the unit of micron
        temperature: temperature in the unit of Kelvin

    Returns:
        absorption coefficient [cm4/dyne]
    """
    A_n1 = [0.0, 0.0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830]
    B_n1 = [0.0, 0.0, 285.8270, -1158.3820, 2427.7190, -1841.4000, 444.5170]
    C_n1 = [0.0, 0.0, -2054.2910, 8746.5230, -
            13651.1050, 8624.9700, -1863.8650]
    D_n1 = [0.0, 0.0, 2827.7760, -11485.6320,
            16755.5240, -10051.5300, 2095.2880]
    E_n1 = [0.0, 0.0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880]
    F_n1 = [0.0, 0.0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850]

    # for wavelengths between 0.1823 micron and 0.3645 micron
    A_n2 = [0.0, 518.1021, 473.2636, -482.2089, 115.5291, 0.0, 0.0]
    B_n2 = [0.0, -734.8666, 1443.4137, -737.1616, 169.6374, 0.0, 0.0]
    C_n2 = [0.0, 1021.1775, -1977.3395, 1096.8827, -245.6490, 0.0, 0.0]
    D_n2 = [0.0, -479.0721, 922.3575, -521.1341, 114.2430, 0.0, 0.0]
    E_n2 = [0.0, 93.1373, -178.9275, 101.7963, -21.9972, 0.0, 0.0]
    F_n2 = [0.0, -6.4285, 12.3600, -7.0571, 1.5097, 0.0, 0.0]

    def ff(wavelength, A_n, B_n, C_n, D_n, E_n, F_n):
        x = 0

        for i in range(1, 7):
            x += (jnp.power(5040.0/temperature, (i+1)/2.0) *
                  (wavelength**2 * A_n[i] + B_n[i] + C_n[i]/wavelength + D_n[i]/wavelength**2 +
                   E_n[i]/wavelength**3 + F_n[i]/wavelength**4))

        return x*1e-29

    kappa_ff = jnp.where(
        wavelength_um > 0.3645,
        ff(wavelength_um, A_n1, B_n1, C_n1, D_n1, E_n1, F_n1),
        0
    ) + jnp.where(
        (wavelength_um >= 0.1823) & (wavelength_um <= 0.3645),
        ff(wavelength_um, A_n2, B_n2, C_n2, D_n2, E_n2, F_n2),
        0
    )

    return kappa_ff


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    Tin = 3000.0
    wav = np.linspace(0.8, 2.4, 100)
    plt.plot(wav, free_free_absorption(wav, Tin))
    plt.plot(wav, bound_free_absorption(wav, Tin))
    plt.yscale('log')
    plt.show()

    Nlayer = 100
    Parr = np.logspace(-8, 2, Nlayer)
    Tarr = np.linspace(2000, 3000, Nlayer)
    Nnu = 40000
    nus = np.linspace(1900.0, 2300.0, Nnu, dtype=np.float64)
    kB = 1.380649e-16
    narr = (Parr*1.e6)/(kB*Tarr)
    vmrh = 0.001
    vmre = vmrh
    number_density_e = vmre*narr
    number_density_h = vmrh*narr
    print(jnp.shape(log_hminus_continuum(
        nus, Tarr, number_density_e, number_density_h)))

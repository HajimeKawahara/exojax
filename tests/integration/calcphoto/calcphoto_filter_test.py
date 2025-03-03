import jax.numpy as jnp
import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.utils.photometry import download_filter_from_svo
from exojax.utils.photometry import average_resolution
from exojax.utils.photometry import download_zero_magnitude_flux_from_svo
from exojax.utils.instfunc import nx_even_from_resolution_eslog

# from astropy import units as u

# http://svo2.cab.inta-csic.es/theory/fps/
up_resolution_factor = 2**5
filter_name = "2MASS/2MASS.Ks"
nu_ref, transmission_ref = download_filter_from_svo(filter_name)
nu_center, f0 = download_zero_magnitude_flux_from_svo(filter_name, unit="cm-1")
resolution_photo = average_resolution(nu_ref) * up_resolution_factor
print("resolution_photo=", resolution_photo)

Nx = nx_even_from_resolution_eslog(np.min(nu_ref), np.max(nu_ref), resolution_photo)
nu_ref_min = 5460.0
nu_ref_max = 6950.0


import matplotlib.pyplot as plt
plt.plot(nu_ref, transmission_ref)
plt.axvline(nu_ref_min, color="red")
plt.axvline(nu_ref_max, color="red")
plt.yscale("log")
plt.savefig("transmission.png")
plt.show()

nus, wav, res = wavenumber_grid(
    nu_ref[0] + 1.0, nu_ref[-1] + 1.0, Nx, unit="cm-1", xsmode="premodit"
)
transmission = np.interp(nus, nu_ref, transmission_ref)

def calc_apparent_magnitute(flux, f_ref=1.0):
    #flux (erg/s/cm^2/cm-1)
    pass


    



def calc_photo(mu, f_ref=1.0):
    mu = jnp.concatenate(mu)
    mu = mu * f_ref  # [erg/s/cm^2/cm^{-1}]
    # [erg/s/cm^2/cm^{-1}] => [erg/s/cm^2/cm]
    mu = mu / (jnp.concatenate(wavd_p) * 1.0e-8) ** 2.0e0
    # [erg/s/cm^2/cm] => [W/m^2/um]
    mu = mu * 1.0e-7 * 1.0e4 * 1.0e-4

    fdl = jnp.trapz(mu * jnp.concatenate(tr), jnp.concatenate(wavd_p))
    dl = jnp.trapz(jnp.concatenate(tr), jnp.concatenate(wavd_p))
    f = fdl / dl

    H_mag = -2.5 * jnp.log10(f / f0)

    return H_mag

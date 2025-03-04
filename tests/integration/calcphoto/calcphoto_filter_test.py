import jax.numpy as jnp
import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.utils.photometry import download_filter_from_svo
from exojax.utils.photometry import average_resolution
from exojax.utils.photometry import download_zero_magnitude_flux_from_svo
from exojax.utils.photometry import apparent_magnitude
from exojax.utils.instfunc import nx_even_from_resolution_eslog
from exojax.utils.constants import RJ
from exojax.utils.constants import pc
    
import matplotlib.pyplot as plt

# from astropy import units as u

# http://svo2.cab.inta-csic.es/theory/fps/
up_resolution_factor = 2**5
filter_name = "2MASS/2MASS.Ks"
nu_ref, transmission_ref = download_filter_from_svo(filter_name)
nu_center, f0 = download_zero_magnitude_flux_from_svo(filter_name, unit="cm-1")
resolution_photo = average_resolution(nu_ref) * up_resolution_factor
print("resolution_photo=", resolution_photo)

Nx = nx_even_from_resolution_eslog(np.min(nu_ref), np.max(nu_ref), resolution_photo)

plt.plot(nu_ref, transmission_ref)
plt.savefig("transmission.png")
#plt.show()

nus_filter, wav_filter, res = wavenumber_grid(nu_ref[0], nu_ref[-1], Nx, unit="cm-1", xsmode="premodit")
transmission_filter = np.interp(nus_filter, nu_ref, transmission_ref)
#apparent_magnitude(1.0, 1.0, 1.0, nus_filter, transmission_filter, f0)


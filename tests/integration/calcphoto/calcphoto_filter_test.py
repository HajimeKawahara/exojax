import jax.numpy as jnp
import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.spec.specop import SopPhoto
from exojax.utils.photometry import apparent_magnitude
from exojax.utils.instfunc import nx_even_from_resolution_eslog
from exojax.utils.constants import RJ
from exojax.utils.constants import pc
    
import matplotlib.pyplot as plt

# from astropy import units as u

# http://svo2.cab.inta-csic.es/theory/fps/
filter_name = "2MASS/2MASS.Ks"
sop_photo = SopPhoto(filter_name, download=True)

plt.plot(sop_photo.nu_ref, sop_photo.transmission_ref)
plt.savefig("transmission.png")
#plt.show()

#nus_filter, wav_filter, res = wavenumber_grid(nu_ref[0], nu_ref[-1], Nx, unit="cm-1", xsmode="premodit")
#transmission_filter = np.interp(nus_filter, nu_ref, transmission_ref)
#apparent_magnitude(1.0, 1.0, 1.0, nus_filter, transmission_filter, f0)


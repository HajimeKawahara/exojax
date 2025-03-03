import jax.numpy as jnp
import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.spec.unitconvert import wav2nu
#from astropy import units as u

#http://svo2.cab.inta-csic.es/theory/fps/
up_resolution_factor = 2**5
filter_name = '2MASS/2MASS.H'


from astroquery.svo_fps import SvoFps
data = SvoFps.get_transmission_data(filter_name)
unit = str(data['Wavelength'].unit)
wl_ref = data['Wavelength']
nu_ref, transmission_ref = wav2nu(np.array(wl_ref), unit=unit, values=np.array(data['Transmission']))


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot(data['Wavelength'], data['Transmission'])
ax = fig.add_subplot(212)
plt.plot(nu_ref, transmission_ref)
plt.show()


print(nu_ref)
exit()
#####
wl_min = np.min(wl_ref)
wl_max = np.max(wl_ref)
dlmd = (wl_max - wl_min) / len(wl_ref)
Rinst_p = 0.5 * (wl_min + wl_max) / dlmd
R = Rinst_p * up_resolution_factor

nu_min = 1.0e8/(wl_max + 5.0)
nu_max = 1.0e8/(wl_min - 5.0)
Nx = np.ceil(R * np.log(nu_max/nu_min)) + 1 # ueki                                                                
Nx = np.ceil(Nx/2.) * 2 # make even  
nus_k,wav_k,res_k = wavenumber_grid(wl_min-5.,wl_max+5.,Nx,unit="AA",xsmode="premodit")

#from scipy import interpolate
#f = interpolate.interp1d(data['Wavelength'], data['Transmission'])
#tr = f(wavd_p)


tr = np.interp(wav_k, wl_ref, transmission_ref)




def calc_photo(mu, f_ref=1.0):
    mu = jnp.concatenate(mu)
    mu = mu * f_ref # [erg/s/cm^2/cm^{-1}]                                                                              
    # [erg/s/cm^2/cm^{-1}] => [erg/s/cm^2/cm]                                                                           
    mu = mu / (jnp.concatenate(wavd_p)*1.0e-8)**2.0e0
    # [erg/s/cm^2/cm] => [W/m^2/um]                                                                                     
    mu = mu * 1.0e-7 * 1.0e4 * 1.0e-4

    fdl = jnp.trapz(mu*jnp.concatenate(tr), jnp.concatenate(wavd_p))
    dl = jnp.trapz(jnp.concatenate(tr), jnp.concatenate(wavd_p))
    f = fdl / dl

    H_mag = -2.5 * jnp.log10(f / f0)

    return H_mag





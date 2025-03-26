from exojax.test.data import TESTDATA_LUH16A
from exojax.test.data import TESTDATA_JUPITER
from exojax.test.data import get_testdata_filename
from exojax.spec.unitconvert import wav2nu
from exojax.spec.unitconvert import nu2wav
import numpy as np
import pandas as pd


def sample_emission_spectrum(noise=0.03, seed=0):
    """load a sample emission spectrum of Luhman 16A, used in PAPER I (Kawahara et al. 2022)

    Args:
        noise (float, optional): additional noise level. Defaults to 0.03. None: original data
        seed (int, optional): seed of random number generator. Defaults to 0.

    Returns:
        array: wavenumber grid (cm-1)
        array: relative flux (normalized)
        array: error of relative flux
        float: K-band magnitude
        float: error of K-band magnitude
        str: filter_id
    """

    filename = get_testdata_filename(TESTDATA_LUH16A)
    df = pd.read_csv(filename)
    wav_micron = df["wavelength_micron"].values
    relative_flux = df["normalized_flux"].values
    err_flux = df["err_normalized_flux"].values
    nu_grid, relative_flux = wav2nu(wav_micron, unit="um", values=relative_flux)
    nu_grid, err_flux = wav2nu(wav_micron, unit="um", values=err_flux)

    relative_flux, err_flux = _add_artificial_noise(noise, seed, relative_flux, err_flux)

    # masking outliers (probably telluric residulas)
    mask = (nu_grid < 4366.95) + (nu_grid > 4367.05)
    nu_grid = nu_grid[mask]
    relative_flux = relative_flux[mask]
    err_flux = err_flux[mask]

    # MKO filter, Burgasser et al. 2013 ApJ 772,129
    Kmag = 9.44
    Kmag_err = 0.07
    filter_id = "MKO/NSFCam.Ks"

    return nu_grid, relative_flux, err_flux, Kmag, Kmag_err, filter_id


def sample_reflection_spectrum(noise=0.2, seed=0):
    # made by Jupiter_Hires_Modeling_NIR.ipynb in exojaxample_jupiter
    filename = get_testdata_filename(TESTDATA_JUPITER)
    dat = np.loadtxt(filename)
    unmask_nus_obs = dat[:, 0]
    unmask_spectra = dat[:, 1]
    #unmask_wav_obs = nu2wav(unmask_nus_obs, unit="AA")
    mask = unmask_nus_obs < 6174
    nus_obs = unmask_nus_obs[mask]
    relative_flux = unmask_spectra[mask]
    relative_flux = relative_flux / np.median(relative_flux)

    err_flux = np.zeros_like(relative_flux)
    relative_flux, err_flux = _add_artificial_noise(noise, seed, relative_flux, err_flux)

    return nus_obs, relative_flux, err_flux
        
def _add_artificial_noise(noise, seed, relative_flux, err_flux):
    if noise is not None:
        np.random.seed(seed)
        relative_flux = relative_flux + np.random.normal(0, noise, len(relative_flux))
        err_flux = np.sqrt(err_flux**2 + noise**2)
    return relative_flux,err_flux
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nus, relflux, err_flux = sample_reflection_spectrum()
    plt.plot(nus, relflux)
    
    #nus, relflux, err_flux, Kmag, Kmag_err, filter_id = sample_emission_spectrum()
    #plt.plot(nus, relflux)
    plt.show()

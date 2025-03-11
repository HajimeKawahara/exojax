from exojax.test.data import TESTDATA_LUH16A
from exojax.test.data import get_testdata_filename
from exojax.spec.unitconvert import wav2nu
import pandas as pd

def sample_emission_spectrum():
    """load a sample emission spectrum of Luhman 16A, used in PAPER I (Kawahara et al. 2022)
    
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

    #MKO filter, Burgasser et al. 2013 ApJ 772,129
    Kmag = 9.44 
    Kmag_err = 0.07 
    filter_id = "MKO/NSFCam.Ks"
    return nu_grid, relative_flux, err_flux,  Kmag, Kmag_err, filter_id


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nus, relflux, err_flux,  Kmag, Kmag_err, filter_id = sample_emission_spectrum()
    plt.plot(nus,relflux)
    plt.show()
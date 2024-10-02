from exojax.spec.unitconvert import nu2wav
import numpy as np


def load_jupiter_reflection():
    dat = np.loadtxt(
        "jupiter_corrected.dat"
    )  # made by Jupiter_Hires_Modeling_NIR.ipynb
    unmask_nus_obs = dat[:, 0]
    unmask_spectra = dat[:, 1]
    unmask_wav_obs = nu2wav(unmask_nus_obs, unit="AA")
    #mask = (unmask_nus_obs < 6163.5) + (
    #        (unmask_nus_obs > 6166) & (unmask_nus_obs < 6174)
    #    ) 
    mask = (unmask_nus_obs < 6174)
    nus_obs = unmask_nus_obs[mask]
    wav_obs = nu2wav(nus_obs, unit="AA")
    spectra = unmask_spectra[mask]
    return wav_obs, nus_obs, spectra, unmask_wav_obs, unmask_nus_obs, unmask_spectra, mask

# Spectral Samples 

Observatory: JWST

Target: WASP39-b

Type: transmission spectra

This spectral data aggregates spectra originally published under Creative Commons Attribution 4.0 International (CC BY 4.0). Each spectrum retains its original attribution.

## NIRISS SOSS

Author(s): Carter, A. L., & May, E. M. (2024)

Title: Products and Models for "A benchmark JWST near-infrared spectrum for the exoplanet WASP-39 b"

Source: https://zenodo.org/records/10161743

License: Creative Commons Attribution 4.0 International (CC BY 4.0)

file: niriss_order1.csv (original filename = Fit_LimbDarkening/NIRISS_SOSS_Order1/bins_scale2.csv )

file: niriss_order2.csv (original filename = Fit_LimbDarkening/NIRISS_SOSS_Order2/bins_scale2.csv )

## NIRSPEC G395H

processed by Shotaro Tada, data belonging in ExoJAX

file: wasp39b_nirspec_g395h_rp_std.npy

file: jwst_nirspec_g395h_disp.fits  

file: wasp39b_nirspec_g395h_rp_mean.npy


## MIRI

Authors: Powell, D., Feinstein, A., Lee, E., Zhang, M., Tsai, S.-M., Taylor, J., Kirk, J., Bell, T., Barstow, J., & Gao, P. (2023).

Title: Products and Models for "Sulphur Dioxide in the Mid-Infrared Transmission Spectrum of WASP-39b" [Data set]. 

Source: https://doi.org/10.5281/zenodo.10055845a

License: Creative Commons Attribution 4.0 International (CC BY 4.0)

file: miri.h5 (original filename = Eureka-wasp-39b-spectrum.h5)

Example: 
    with h5py.File(infile, "r") as f:
        dppm = np.array(f["dppm"])                     # 
        dppm_err = np.array(f["dppm_error"])           # 
        wl = np.array(f["wavelength"])

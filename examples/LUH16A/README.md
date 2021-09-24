# LUH16A

HMC-NUTS fitting of exojax to a real spectrum of a brown dwarf Luhman 16 A.

## Fiducial model
LUH16A/FidEMb/fit.py -- ExoMol Mp prior from astrometry, fA=1, .broad applied

## Luhman 16 A spectrum
- data/luhman16a_spectra_detector1.csv

The high-dispersion spectrum of Luhman 16 A as observed by VLT/CRIRES (detector1), originally for [Crossfield et al. (2014)](https://www.nature.com/articles/nature12955?proof=t).

Contact for this spectrum: [Ian J. M. Crossfield](https://crossfield.ku.edu/)


|  directroy       | opacity calculator |  Limb Darkening  | Gaussian Process |        |
| ---------------- |  ----------------- | ---------------- | ---------------- | ------ |
| FidEMb           |     LPF (direct)   |                  |                  |  9.2 h |
| FidEMbu          |     LPF (direct)   |:heavy_check_mark:|                  | 11.2 h |
| FidEMbug         |     LPF (direct)   |:heavy_check_mark:|:heavy_check_mark:|        |
| FidEMb_modit     |     MODIT          |                  |                  |        |
| FidEMbu_modit    |     MODIT          |:heavy_check_mark:|                  |        |

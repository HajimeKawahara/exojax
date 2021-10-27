# LUH16A

HMC-NUTS fitting of exojax to a real spectrum of a brown dwarf [Luhman 16 A](https://en.wikipedia.org/wiki/Luhman_16).

## Luhman 16 A spectrum
- data/luhman16a_spectra_detector1.csv

The high-dispersion spectrum of Luhman 16 A as observed by VLT/CRIRES (detector1), originally for [Crossfield et al. (2014)](https://www.nature.com/articles/nature12955?proof=t).

Contact for this spectrum: [Ian J. M. Crossfield](https://crossfield.ku.edu/)


## Models

### fiducial models used in Section 5

ExoMol Mp prior from astrometry, .broad applied. A=A100,V=V100

|  directroy       | opacity calculator |  Limb Darkening  | Gaussian Process |           |
| ---------------- |  ----------------- | ---------------- | ---------------- | --------- |
| FidEMb           |     LPF (direct)   |                  |                  |           |
| FidEMbu          |     LPF (direct)   |:heavy_check_mark:|                  | 25.0 h (A)|
| FidEMbug         |     LPF (direct)   |:heavy_check_mark:|:heavy_check_mark:| 29.5 h (A)|
| FidEMb_modit     |     MODIT          |                  |                  |           |
| FidEMbu_modit    |     MODIT          |:heavy_check_mark:|                  |           |
| FidEMbug_modit   |     MODIT          |:heavy_check_mark:|:heavy_check_mark:| 24.6 h (V)|

Note: N=4500

### other models

|  directroy       | description       |opacity calculator |  Limb Darkening  | Gaussian Process |            |
| ---------------- | ----------------- | ----------------- | ---------------- | ---------------- | ---------- |
| FreeT            | layer-by-layer T  |    LPF (direct)   |:heavy_check_mark:|:heavy_check_mark:| 28.3 h (A) |
| MassEMbug        | mass est (ExoMol) |    LPF (direct)   |:heavy_check_mark:|:heavy_check_mark:| 34.2 h (A) |
| MassHitbug       | mass est (Hitemp) |    LPF (direct)   |:heavy_check_mark:|:heavy_check_mark:| 50.6 h (V) |
| HDC              | additional noise  |    LPF (direct)   |:heavy_check_mark:|                  |            |

Note: N=4500
Note: 1/3 of the wavelength range was used for FreeT.

### Comparisons

Do the lines below the CIA photosphere affect the results? -> No. See the posteriors.

|  directroy       | opacity calculator |  Limb Darkening  | # of water lines |        |
| ---------------- |  ----------------- | ---------------- | ---------------- | ------ |
| FidEMbu          |     LPF (direct)   |:heavy_check_mark:|   334            | 11.2 h |
| FidEMbux         |     LPF (direct)   |:heavy_check_mark:|   802            | 18.3 h |

Note: N=1500

<img src="https://user-images.githubusercontent.com/15956904/135429319-ee298bac-6448-47a7-a0f0-3d2dbfbd18e2.png" Titie="comparison" Width=850px>
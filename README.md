# ExoJAX
 [![License](https://img.shields.io/github/license/HajimeKawahara/exojax)](https://github.com/HajimeKawahara/exojax/blob/develop/LICENSE)
 [![Docs](https://img.shields.io/badge/docs-exojax-brightgreen)](http://secondearths.sakura.ne.jp/exojax/)
 [![arxiv](https://img.shields.io/badge/arxiv-2105.14782-blue)](http://arxiv.org/abs/2105.14782)
 [![arxiv](https://img.shields.io/badge/arxiv-2410.06900-red)](http://arxiv.org/abs/2410.06900)
 [![paper](https://img.shields.io/badge/paper-ApJS_258_31_(2022)-orange)](https://iopscience.iop.org/article/10.3847/1538-4365/ac3b4d) 
 <a href="https://codeclimate.com/github/HajimeKawahara/exojax/maintainability"><img src="https://api.codeclimate.com/v1/badges/97c5e8835f3ef9c4ad7c/maintainability" /></a>

Differentiable spectral modelling of exoplanets/brown dwarfs/M dwarfs using JAX!
Read [the docs](http://secondearths.sakura.ne.jp/exojax/) 🐕. 

In short, ExoJAX allows you to do gradient based optimizations, HMC-NUTS, and SVI using the latest database.

<img src="https://github.com/user-attachments/assets/186d738a-8ce2-4adf-9512-4aa1e43bcf90" Titie="exojax" Width=850px>

<details><summary>ExoJAX Classes</summary>

- Databases: *db (mdb: molecular, adb: atomic, cdb: continuum, pdb: particulates)
- Opacity Calculators: opa  (Voigt profile, CIA, Mie, Rayleigh scattering etc)
- Atmospheric Radiative Transfer: art (emission w, w/o scattering, reflection, transmission)
- Spectral Operator: sop (planet rotation, instrumental broadening, photometry)
- Atmospheric Microphysics: amp (clouds etc)

</details>

## License

🐈 Copyright 2020-2025 ExoJAX contributors. ExoJAX is publicly available under the MIT license.

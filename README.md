# ExoJAX
 [![License](https://img.shields.io/github/license/HajimeKawahara/exojax)](https://github.com/HajimeKawahara/exojax/blob/develop/LICENSE)
 [![Docs](https://img.shields.io/badge/docs-exojax-brightgreen)](http://secondearths.sakura.ne.jp/exojax/)
 [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/HajimeKawahara/exojax)
 [![paper](https://img.shields.io/badge/paper-ApJS_258_31_(2022)-orange)](https://iopscience.iop.org/article/10.3847/1538-4365/ac3b4d) 
 [![paper](https://img.shields.io/badge/paper-ApJ_985_263_(2025)-red)](https://iopscience.iop.org/article/10.3847/1538-4357/adcba2)

Differentiable spectral modelling of exoplanets/brown dwarfs/M dwarfs using JAX!
Read [the docs](http://secondearths.sakura.ne.jp/exojax/) 🐕 or [deepwiki for ExoJAX](https://deepwiki.com/HajimeKawahara/exojax). 

In short, ExoJAX allows you to do gradient based optimizations, HMC-NUTS, and SVI using the latest database.

<img src="https://github.com/user-attachments/assets/186d738a-8ce2-4adf-9512-4aa1e43bcf90" Titie="exojax" Width=850px>

<details><summary>ExoJAX Classes</summary>

- Databases (`exojax.database`) : *db (mdb: molecular, adb: atomic, cdb: continuum, pdb: particulates)
- Opacity Calculators (`exojax.opacity`) : opa  (Voigt profile, CIA, Mie, Rayleigh scattering etc)
- Atmospheric Radiative Transfer (`exojax.rt`) : art (emission w, w/o scattering, reflection, transmission)
- Spectral Operator (`exojax.postproc`) : sop (planet rotation, instrumental broadening, photometry)
- Atmospheric Microphysics (`exojax.atm`) : amp (clouds etc)

</details>

## License

🐈 Copyright 2020-2026 ExoJAX contributors. ExoJAX is publicly available under the MIT license.

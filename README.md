# ExoJAX
 [![License](https://img.shields.io/github/license/HajimeKawahara/exojax)](https://github.com/HajimeKawahara/exojax/blob/develop/LICENSE)
 [![Docs](https://img.shields.io/badge/docs-exojax-brightgreen)](http://secondearths.sakura.ne.jp/exojax/)
 [![arxiv](https://img.shields.io/badge/arxiv-2105.14782-blue)](http://arxiv.org/abs/2105.14782)
 [![paper](https://img.shields.io/badge/paper-ApJS_258_31_(2022)-orange)](https://iopscience.iop.org/article/10.3847/1538-4365/ac3b4d) 
 <a href="https://codeclimate.com/github/HajimeKawahara/exojax/maintainability"><img src="https://api.codeclimate.com/v1/badges/97c5e8835f3ef9c4ad7c/maintainability" /></a>

Differentiable spectral modelling of exoplanets/brown dwarfs/M dwarfs using JAX!
Read [the docs](http://secondearths.sakura.ne.jp/exojax/develop) 🐕. 
In short, ExoJAX allows you to do gradient based optimizations and HMC-NUTS samplings using the latest database.

ExoJAX is at least compatible with

- PPLs: [NumPyro](https://github.com/pyro-ppl/numpyro), [blackjax](https://github.com/blackjax-devs/blackjax), [bayeux](https://github.com/jax-ml/bayeux) 
- Optimizers: [JAXopt](https://github.com/google/jaxopt), [optax](https://github.com/google-deepmind/optax)

<img src="https://github.com/user-attachments/assets/70d4291b-f818-4204-ab96-b3bd6c99c48f" Titie="exojax" Width=850px>

<details><summary>ExoJAX Classes</summary>

- Databases: *db (mdb: molecular, adb: atomic, cdb:continuum, pdb: particulates)
- Opacity Calculators: opa  (Voigt profile, CIA, Mie, Rayleigh scattering etc)
- Atmospheric Radiative Transfer: art (emission w, w/o scattering, refelction, transmission)
- Spectral Operator: sop (planet rotation, instrumental boradening)
- Atompsheric Microphysics: amp (clouds etc)

</details>

## Get Started 

See [this page](http://secondearths.sakura.ne.jp/exojax/develop/tutorials/get_started.html) for the first step!

## Real Examples

to be updated

## References
[![paper](https://img.shields.io/badge/paper_I-ApJS_258_31_(2022)-orange)](https://iopscience.iop.org/article/10.3847/1538-4365/ac3b4d) 

- Paper I: Kawahara, Kawashima, Masuda, Crossfield, Pannier, van den Bekerom, [ApJS 258, 31 (2022)](https://iopscience.iop.org/article/10.3847/1538-4365/ac3b4d)
- Paper II: in prep

## License

🐈 Copyright 2020-2024 ExoJAX contributors. ExoJAX is publicly available under the MIT license.

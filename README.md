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

<img src="https://github.com/HajimeKawahara/exojax/assets/15956904/8aa9673b-b64b-4b65-a76c-2966ef1edbc7" Titie="exojax" Width=850px>

<details><summary>ExoJAX Classes</summary>

- Databases: *db (mdb: molecular, adb: atomic, cdb:continuum, pdb: particulates)
- Opacity Calculators: opa  (Voigt profile, CIA, Mie, Rayleigh scattering etc)
- Atmospheric Radiative Transfer: art (emission w, w/o scattering, refelction, transmission)
- Atompsheric Microphysics: amp (clouds etc)

</details>

## Get Started 

See [this page](http://secondearths.sakura.ne.jp/exojax/develop/tutorials/get_started.html) for the first step!

## Functions

<details open><summary>Voigt Profile :heavy_check_mark: </summary>

```python3
from exojax.spec import voigt
nu=numpy.linspace(-10,10,100)
voigt(nu,1.0,2.0) #sigma_D=1.0, gamma_L=2.0
```

</details>

<details><summary>Cross Section using HITRAN/HITEMP/ExoMol :heavy_check_mark: </summary>
 
```python
from exojax.utils.grids import wavenumber_grid
from exojax.spec.api import MdbExomol
from exojax.spec.opacalc import OpaPremodit
from jax import config
config.update("jax_enable_x64", True)

nu_grid,wav,res=wavenumber_grid(1900.0,2300.0,200000,xsmode="premodit",unit="cm-1",)
mdb = MdbExomol(".database/CO/12C-16O/Li2015",nu_grid)
opa = OpaPremodit(mdb,nu_grid,auto_trange=[900.0,1100.0])
xsv = opa.xsvector(1000.0, 1.0) # cross section for 1000K, 1 bar
```

 <img src="https://user-images.githubusercontent.com/15956904/111430765-2eedf180-873e-11eb-9740-9e1a313d590c.png" Titie="exojax auto cross section" Width=850px> 

</details>



<details><summary>Do you just want to plot the line strength at T=1000K? </summary>

```python
mdb.change_reference_temperature(1000.) # at 1000K
plt.plot(mdb.nu_lines,mdb.line_strength_ref,".")
```

</details>

<details><summary>Emission Spectrum :heavy_check_mark: </summary>

```python
art = ArtEmisPure(nu_grid=nu_grid, pressure_btm=1.e2, pressure_top=1.e-8, nlayer=100)
F = art.run(dtau, Tarr)
```

<img src="https://user-images.githubusercontent.com/15956904/116488770-286ea000-a8ce-11eb-982d-7884b423592c.png" Titie="exojax auto \emission spectrum" Width=850px> 

</details>

<details><summary>Transmission Spectrum :heavy_check_mark: </summary></details>
<details><summary>Reflection Spectrum :heavy_check_mark: </summary></details>

## Installation

```
pip install exojax
```

or

```
python setup.py install
```

<details><summary>Note on installation w/ GPU support</summary>

:books: You need to install CUDA, JAX w/ NVIDIA GPU support.

Visit [here](https://github.com/google/jax) for the installation of GPU supported JAX.

</details>


## References
[![paper](https://img.shields.io/badge/paper_I-ApJS_258_31_(2022)-orange)](https://iopscience.iop.org/article/10.3847/1538-4365/ac3b4d) 

- Paper I: Kawahara, Kawashima, Masuda, Crossfield, Pannier, van den Bekerom, [ApJS 258, 31 (2022)](https://iopscience.iop.org/article/10.3847/1538-4365/ac3b4d)

## License

🐈 Copyright 2020-2024 ExoJAX contributors. ExoJAX is publicly available under the MIT license.

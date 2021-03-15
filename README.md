# exojax

<details open><summary>Voigt Profile :heavy_check_mark: </summary>

```python
 from exojax.spec import voigt
 import jax.numpy as jnp
 nu=jnp.linspace(-10,10,100)
 voigt(nu,1.0,2.0) #sigma_D=1.0, gamma_L=2.0
```

</details>

<details><summary>Opacity using HITRAN/HITEMP/ExoMol :heavy_check_mark: </summary><img src="https://github.com/HajimeKawahara/exojax/blob/develop/documents/figures/plottau.png" Titie="exojax" Width=850px> </details>

<details><summary>Auto-differentiable Radiative Transfer :heavy_check_mark: </summary> <img src="https://github.com/HajimeKawahara/exojax/blob/develop/documents/exojax.png" Titie="exojax" Width=850px> </details>

<details><summary>HMC-NUTS of Emission Spectra :heavy_check_mark: </summary></details>

<details><summary>HMC-NUTS of Transmission Spectra :x: </summary>Not supported yet. </details>

<details><summary>Cloud modeling :x: </summary> Not supported yet. </details>



## install

```
python setup.py install
```

under development since Dec. 2020.

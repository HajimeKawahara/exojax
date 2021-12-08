Installation and References
----------------------------------

*Dec 7th (2021) Hajime Kawahara*

At a minimum, you can start to use exojax by

.. code:: sh
	  
	  pip install exojax

via `pypi <https://pypi.org/project/exojax/>`_.

Alternatively, clone the code from `github page <https://github.com/HajimeKawahara/exojax>`_ and run

.. code:: sh
	  
	  python setup.py install

Installation w/ GPU support
==============================

However, to leverage the power of JAX, numpyro, you need to prepare a GPU environment. For this purpose, jaxlib and numpyro must be linked.

You should check cuda version of your environment as

.. code:: sh

	  nvcc -V

Also, check required jaxlib versions by numpyro at
`NumPyro <https://github.com/pyro-ppl/numpyro>`_.
Here is an example of installation for jaxlib for cuda 11.1 to 11.4.

.. code:: sh

           pip install --upgrade "jax[cuda111]"  -f https://storage.googleapis.com/jax-releases/jax_releases.html
	   cd ../exojax/
	   python setup.py install


In addition, you may need cuDNN for `response.ipgauss2 <../exojax/exojax.spec.html#exojax.spec.response.ipgauss2>`_ and `response.rigidrot2 <../exojax/exojax.spec.html#exojax.spec.response.rigidrot2>`_. For ubuntu, download .deb from NVIDIA and install it:
	   
.. code:: sh
	  sudo dpkg -i libcudnn8_8.2.0.53-1+cuda11.3_amd64.deb

References
=================

|:green_circle:| **ExoJAX Paper I**:  Kawahara, Kawashima, Masuda, Crossfield, Pannier, van den Bekerom (2021) accepted by ApJS: `arXiv:2105.14782 <http://arxiv.org/abs/2105.14782>`_


Many methods/databases are used in ExoJAX.

- JAX: Bradbury, J., Frostig, R., Hawkins, P., et al. 2018, JAX: composable transformations of Python+NumPy programs, `JAX <http://github.com/google/jax>`_
- numpyro: Phan, D., Pradhan, N., & Jankowiak, M. 2019, `arXiv:1912.11554 <http://arxiv.org/abs/1912.11554>`_
- Algorithm 916: Zaghloul and Ali (2012) `arXiv:1106.0151 <https://arxiv.org/abs/1106.0151>`_
- DIT: `van den Bekerom and Pannier (2021) <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ 
- ExoMol: `Tennyson et al. (2016) <https://www.sciencedirect.com/science/article/abs/pii/S0022285216300807?via%3Dihub>`_
- HITRAN/HiTEMP
- VALD3
- Other many packages/algorithms. See `arXiv:2105.14782 <http://arxiv.org/abs/2105.14782>`_ for the details.


Related Projects
=====================
  
- `RADIS <https://github.com/radis/radis>`_

 | ExoJAX gets a lot of inspiration from a fast line-by-line code for high resolution infrared molecular spectra `RADIS <https://github.com/radis/radis>`_, including DIT, the use of Vaex, and so on.

- `REACH <http://secondearths.sakura.ne.jp/reach/>`_
  
 | Exojax was originally developed to interpret the data obtained a new capability of high-dispersion coronagraphy at Subaru telescope, the `REACH <http://secondearths.sakura.ne.jp/reach/>`_ project (SCExAO+IRD). REACH is supported by `RESCEU <http://www.resceu.s.u-tokyo.ac.jp/top.php>`_, ABC, and `JSPS KAKENHI JP20H00170 <https://kaken.nii.ac.jp/en/grant/KAKENHI-PROJECT-20H00170/>`_ (Kawahara). See also `Lozi et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018SPIE10703E..59L/abstract>`_ for SCExAO, `Kotani et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018SPIE10702E..11K/abstract>`_  for IRD, `Jovanovic et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017arXiv171207762J/abstract>`_ for Post-Coronagraphic Injection and `Kawahara et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJS..212...27K/abstract>`_ for High Dispersion Coronagraphy.

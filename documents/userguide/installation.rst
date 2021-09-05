Installation and References
----------------------------------

*Sep 5th (2021) Hajime Kawahara*

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

Primary paper:  Kawahara, Kawashima, Masuda, Crossfield, van den Bekerom, Parker (2021) under review: `arXiv:2105.14782 <http://arxiv.org/abs/2105.14782>`_


Many methods are used in Exojax.

- JAX:
- numpyro: 
- DIT: `van den Bekerom and Pannier (2021) <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ 


Related Projects
=====================
  
- RADIS

 |

- REACH
  
 | Exojax was originally developed to interpret the data obtained a new capability of high-dispersion coronagraphy at Subaru telescope, the REACH project (SCExAO+IRD). REACH is supported by `RESCEU <http://www.resceu.s.u-tokyo.ac.jp/top.php>`_, ABC, and `JSPS KAKENHI JP20H00170 <https://kaken.nii.ac.jp/en/grant/KAKENHI-PROJECT-20H00170/>`_ (Kawahara). See also `Lozi et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018SPIE10703E..59L/abstract>`_ for SCExAO, `Kotani et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018SPIE10702E..11K/abstract>`_  for IRD, `Jovanovic et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017arXiv171207762J/abstract>`_ for Post-Coronagraphic Injection and `Kawahara et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJS..212...27K/abstract>`_ for High Dispersion Coronagraphy.

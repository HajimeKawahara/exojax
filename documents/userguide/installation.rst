Installation and References
----------------------------------

*last update January 19th (2025) Hajime Kawahara*

.. admonition:: Python 3.9 or later is required

    ExoJAX requires python 3.9 or later.

.. Warning:: 

	For using `vaex <https://github.com/vaexio/vaex>`_ in the common API for the molecular database I/O, we currently recommend using Python 3.9 or 3.10. 



Linux, Windows WSL, Mac
============================

At the very least, you can start using exojax through `pypi <https://pypi.org/project/exojax/>`_.

.. code:: sh
	
	pip install exojax


Alternatively, clone the code from `github page <https://github.com/HajimeKawahara/exojax>`_ and run

.. code:: sh
	
	python setup.py install

If the older version of ExoJAX has already been installed, you need to remove all of the old modules (Otherwise, modules that have been deleted and no more exist in the current version remain). To do so, the following procedure is recommended:

.. code:: sh
	
	python setup.py clean --all 
	python setup.py install

However, to take advantage of the power of JAX, you need to prepare a GPU environment (if you have). For this, jaxlib need to be linked.

You should check the cuda version of your environment as

.. code:: sh

	nvcc -V

Here is an example of installation for jaxlib in linux system. See `JAX installation page <https://jax.readthedocs.io/en/latest/installation.html>`_ for the details.

.. code:: sh
	
	pip install -U "jax[cuda12]"
	

Windows Anaconda
===================

Not supported yet.
		

		
References
=================

|:green_circle:| **ExoJAX Paper I**:  Kawahara, Kawashima, Masuda, Crossfield, Pannier, van den Bekerom (2021) accepted by ApJS: `arXiv:2105.14782 <http://arxiv.org/abs/2105.14782>`_

|:green_circle:| **ExoJAX Paper II**:  Kawahara, Kawashima, Tada et al: `arXiv:2410.06900 <http://arxiv.org/abs/2410.06900>`_


Many techniques/databases are used in ExoJAX.

- JAX: Bradbury, J., Frostig, R., Hawkins, P., et al. 2018, JAX: composable transformations of Python+NumPy programs, `JAX <http://github.com/google/jax>`_
- NumPyro: Phan, D., Pradhan, N., & Jankowiak, M. 2019, `arXiv:1912.11554 <http://arxiv.org/abs/1912.11554>`_
- JAXopt: Blondel, M., Berthet, Q., Cuturi, M. et al. 2021 `arXiv:2105.15183 <http://arxiv.org/abs/2105.15183>`_
- Vaex: Breddels and Veljanoski (2018) `arXiv:https://arxiv.org/abs/1801.02638 <https://arxiv.org/abs/1801.02638>`_
- Algorithm 916: Zaghloul and Ali (2012) `arXiv:1106.0151 <https://arxiv.org/abs/1106.0151>`_
- DIT: `van den Bekerom and Pannier (2021) <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ 
- ExoMol: `Tennyson et al. (2016) <https://www.sciencedirect.com/science/article/abs/pii/S0022285216300807?via%3Dihub>`_
- HITRAN/HITEMP
- VALD3
- VIRGA for refractive indices of condensates
- PyMieScatt for Mie scattering
- Flux-adding treatment by `Robinson and Crisp (2018) <https://www.sciencedirect.com/science/article/pii/S0022407317305101?via%3Dihub>`_
- RADIS, see below.
- Other many packages/algorithms. See `arXiv:2105.14782 <http://arxiv.org/abs/2105.14782>`_ and a forthcoming paper (Kawahara, Kawashima et al.) for the details.


Related Projects
=====================

- `RADIS <https://github.com/radis/radis>`_

| ExoJAX draws a lot of inspiration from a fast line-by-line code for high-resolution infrared molecular spectra `RADIS <https://github.com/radis/radis>`_, including DIT, the use of Vaex, and so on. 
| Since version 1.2 we have been using a common molecular database I/O API in Radis.

- `REACH <http://secondearths.sakura.ne.jp/reach/>`_

| ExoJAX was originally developed to interpret data from a new high-dispersion coronagraphic capability at the Subaru telescope, the `REACH <http://secondearths.sakura.ne.jp/reach/>`_ project (SCExAO+IRD). REACH is supported by `RESCEU <http://www.resceu.s.u-tokyo.ac.jp/top.php>`_, ABC and `JSPS KAKENHI JP20H00170 <https://kaken.nii.ac.jp/en/grant/KAKENHI-PROJECT-20H00170/>`_ (Kawahara). See also `Lozi et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018SPIE10703E..59L/abstract>`_ for SCExAO, `Kotani et al. (2018) <https://ui.adsabs.harvard.edu/abs/2018SPIE10702E..11K/abstract>`_ for IRD, `Jovanovic et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017arXiv171207762J/abstract>`_ for post-coronagraphic injection, and `Kawahara et al. (2014) <https://ui.adsabs.harvard.edu/abs/2014ApJS..212...27K/abstract>`_ for high dispersion coronagraphy.

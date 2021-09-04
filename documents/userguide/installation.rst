Installation and References
----------------------------------

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


- Kawahara, Kawashima, Masuda, Crossfield, van den Bekerom, Parker (2021) under review: [arXiv:2105.14782](http://arxiv.org/abs/2105.14782)

Installation
------------------

At a minimum, you can start to use exojax by

.. code:: sh
	  
	  python setup.py install
	  
However, to leverage the power of JAX, numpyro, you need to prepare a GPU environment. For this purpose, jaxlib and numpyro must be linked. Here is an example of installation for python 3.7 environment.

.. code:: sh
	  
	   pip install --upgrade jax jaxlib==0.1.62+cuda112 -f https://storage.googleapis.com/jax-releases/cuda112/jaxlib-0.1.62+cuda112-cp37-none-manylinux2010_x86_64.whl
	   git clone https://github.com/pyro-ppl/numpyro.git
	   cd numpyro
	   python setup.py install
	   cd ../exojax/
	   python setup.py install


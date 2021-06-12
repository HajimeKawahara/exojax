Experimental module
------------------------


The experimental module is invisible to normal users. However, you can use like that

.. code:: ipython3

	  from exolax.expermental import lpfs.clpf

lpfs
=======

One of the major issues in exojax is the JVP mode is used for forward differentiation in NumPyro/HMC-NUTS. However, the optimization needs the VJP. So we put rlpf for that purpose. In future we want to unify these functions to a single function. 


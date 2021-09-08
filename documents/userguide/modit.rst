Modified Discrete Integral Transform (MODIT)
==============================================

*Sep 5 (2021) Hajime Kawahara*

MODIT is a modified version of `Discrete Integral Transform <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ for rapid spectral synthesis, originally proposed by D.C.M van den Bekerom and E.Pannier. The modifications are as follows:

- MODIT uses the wavenumber grid an evenly-spaced in logarithm (ESLOG).
- MODIT uses a 2D lineshape density (LSD) matrix (gammaL x wavenumber) while the original DIT uses a 3D LSD (gammaL x Doppler wodth x wavenumber).

These formulations are based on the fact that the normalized Doppler width is common for a common temperature and isotope mass. In addition, ESLOG is straightforward to include radial velocity shift. Exojax computes the opacity in layer by layer therefore MODIT is suitable to the formulation of Exojax.

The procedure in `modit.xsvector <../exojax/exojax.spec.html#exojax.spec.modit.xsvector>`_ (MODIT cross section vector) is as follows:

- Compute the linedensity shape (LSD) matrix and apply an FFT to the LSD.
- Compute the Voigt kernel in Fourier space
- multiply them and apply an inverse FFT to it.
  
The kernel for the Voigt profile is given in `ditkernel module <../exojax/exojax.spec.html#module-exojax.spec.ditkernel>`_. The computation of the LSD is a bit tricky in MODIT/Exojax. Let me explain it in the next section.

How is the LSD computed in Exojax/MODIT?
---------------------------

MODIT/DIT needs to compute the lineshape density (LSD) matrix. `inc2D_givenx <../exojax/exojax.spec.html#exojax.spec.modit.inc2D_givenx>`_ is a core function to compute the 2D LSD. Let's compute the LSD with the wavenumber grid [0,1,2,3,4] and the grid of gammaL [0,1,2,3] as follows

.. code:: ipython
       
       >>> import jax.numpy as jnp
       >>> lsd=jnp.zeros((5,4)) #LSD initialization
       >>> wavgrid=jnp.arange(5)
       >>> ngammaLgrid=jnp.arange(4)

Then compute the LSD for a single line at wavenumber of 2.3 and gammaL of 1.2 with the line strength of 1.0.
      
.. code:: ipython
              
       >>> #assume a line at wav=2.3, ngammaL=1.2 with the strength of 1.0
       >>> wav=jnp.array([2.3])
       >>> ngammaL=jnp.array([1.2])
       >>> w=jnp.array([1.0])

We need to compute contributions to the LSD for 'wav' and 'gammaL'. MODIT/DIT uses the linear interpolation to the grid values. This can be done by `getix <../exojax/exojax.spec.html#exojax.spec.dit.getix>`_. The following code is an example how to use 'getix'. Here we have a grid of 'yv=[0,1,2,3,4,5]' and want to compute the contribution and indices of 'y=1.1' and 'y=1.4'. The outputs are the contribution at i=index+1 and index, i.e. the contribution of 'y=1.1' to the LSD is 0.1 at index=2 (yv=2) and 1-0.1 = 0.9 at index=1 (yv=1) and 'y=4.3' to the LSD is 0.3 at index=5 (yv=5) and 1-0.3 = 0.7 at index=4 (yv=4). 

.. code:: ipython

       >>> #An example how to use getix
       >>> from exojax.spec.dit import getix
       >>> import jax.numpy as jnp
       >>> y=jnp.array([1.1,4.3])
       >>> yv=jnp.arange(6)
       >>> getix(y,yv)
       (DeviceArray([0.10000002, 0.3000002 ], dtype=float32), DeviceArray([1, 4], dtype=int32))    

       
For wavenumber, the F64 precision is required. So, `npgetix <../exojax/exojax.spec.html#exojax.spec.dit.npgetix>`_ is used for precomputation, which is numpy version of getix. Then, back to the original problem, we need to pre-compute the contribution and index for wavgrid as follows:

       
.. code:: ipython
              
       >>> from exojax.spec.dit import npgetix
       >>> cx, ix=npgetix(wav,wavgrid)
       >>> cx, ix
       (array([0.29999995]), array([2]))

`inc2D_givenx <../exojax/exojax.spec.html#exojax.spec.modit.inc2D_givenx>`_ computes the LSD with 'y' and 'yv' and a given contribution for 'x'. 'w' is the weight, i.e. the line strength. Then, we get the LSD for the line as follows: 
       
.. code:: ipython
              
       >>> from exojax.spec.modit import inc2D_givenx
       >>> inc2D_givenx(lsd,w,cx,ix,ngammaL,ngammaLgrid)
       DeviceArray([[0.        , 0.        , 0.        , 0.        ],
                    [0.        , 0.        , 0.        , 0.        ],
                    [0.        , 0.56      , 0.14000005, 0.        ],
                    [0.        , 0.23999995, 0.06000001, 0.        ],
                    [0.        , 0.        , 0.        , 0.        ]],            dtype=float32)


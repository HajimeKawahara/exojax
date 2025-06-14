MODIT
==============================================

*Sep 5 (2021) Hajime Kawahara*

MODIT = Modified Discrete Integral Transform
--------------------------------------------------

With an increase in the number of lines of :math:`N_l`, the direct LPF tends to be intractable even when using GPUs, in particular for :math:`N_l \gtrsim 10^3`. MODIT is a modified version of `Discrete Integral Transform <https://www.sciencedirect.com/science/article/abs/pii/S0022407320310049>`_ for rapid spectral synthesis, originally proposed by D.C.M van den Bekerom and E. Pannier. The modifications are as follows:

- MODIT uses the wavenumber grid an evenly-spaced in logarithm (ESLOG).
- MODIT uses a 2D lineshape density (`LSD <https://en.wikipedia.org/wiki/Lucy_in_the_Sky_with_Diamonds#LSD_rumours>`_) matrix (gammaL x wavenumber) while the original DIT uses a 3D LSD (gammaL x Doppler wodth x wavenumber).

These formulations are based on the fact that the normalized Doppler width is common for a common temperature and isotope mass. In addition, ESLOG is straightforward to include radial velocity shift. ExoJAX computes the opacity in layer by layer therefore MODIT is suitable to the formulation of ExoJAX. In MODIT, we define a new variable :math:`q= R_0 \log{\nu}`, where  :math:`R_0` is the spectral resolution of the wavenumber grid (For the dimensional consistency, `\nu` should be divided by a reference wavenumber,  :math:`\nu_0=1 \mathrm{cm}^{-1}`.). The discretization of  :math:`q` provides ESLOG. Then, the Gaussian profile is expressed as:

:math:`f_G(\nu; \nu_l; \beta_T)d \nu = \frac{1}{\sqrt{2 \pi} \beta_T} e^{-(\nu - \nu_l)^2/2 \beta_T^2} d \nu = \frac{e^{\frac{q - q_l}{R}}}{\sqrt{2 \pi} a_D} \exp\left[{- \frac{R_0^2}{2 a_D^2} \left(e^{\frac{q - q_l}{R_0}} -1\right)^2 }\right] d q`
:math:`\approx  f_G(q; q_l; a_D) d q`
      
where :math:`\nu_l` is the line center, :math:`q_l \equiv R_0 \log{\nu_l}`, and :math:`a_D \equiv \frac{R_0 \beta_T}{\nu_l} = \sqrt{\frac{k_B T}{M m_u}} \frac{R_0}{c}` is the standard deviation of the Gaussian profile in the :math:`q` space, which does not depend on the line center :math:`\nu_l`. Therefore, the Doppler broadening parameter in ESLOG is common for a given temperature :math:`T` and molecular mass :math:`M`. We also define :math:`\gamma_L` normalized by :math:`\nu_l/R_0` as
:math:`\tilde{\gamma}_L \equiv \frac{R_0 \gamma_L}{\nu_l}`.
This quantity depends on the line properties, but is demensionless. MODIT defines the LSD matrix :math:`\mathfrak{S}` in two dimensions: :math:`q` and :math:`\tilde{\gamma}`.

Then, the procedure in `modit.xsvector <../exojax/exojax.spec.html#exojax.spec.modit.xsvector>`_ (MODIT cross section vector) is summarized as:

- Compute the 2D LSD matrix :math:`\mathfrak{S}` and apply an FFT to the LSD.
- Compute the Voigt kernel in Fourier space
- multiply them and apply an inverse FFT to it.
  
The computation of the LSD is a bit tricky in MODIT/ExoJAX. Let me explain it in the next section. But, the other two processes can be understood as follows:

For a given temperature and molecule, the synthesis of the cross--section can be expressed using the LSD matrix :math:`\mathfrak{S}_{jk} = \mathfrak{S} ({\nu_l}_j,\tilde{\gamma}_k)`
as follows:

:math:`\sigma (q_i) =  \sum_{jk} \mathfrak{S}_{jk} \acute{V}(q_i - q_j;\tilde{\beta}, \tilde{\gamma}_k) = \sum_{k} \mathrm{FT}^{-1} [ \mathrm{FT} (\mathfrak{S}_{jk})  \mathrm{FT} (\acute{V} ({\nu_l}_j, \tilde{\beta}, \tilde{\gamma}_{L,k}))`,
      
where :math:`\acute{V}(q_i;\tilde{\beta},\tilde{\gamma}_j)` is the profile in the :math:`q` space that satisfies :math:`V (\nu_l, \beta, \gamma_{L}) d \nu = \acute{V}(q;\tilde{\beta};\gamma_L) dq`.  We can approximate the Lorentz profile of the :math:`q` space as :math:`f_L(\nu;\nu_l;\gamma_L) d \nu \approx f_L(q;q_l;\tilde{\gamma}_L) d q` for :math:`|\log{\nu} - \log{\nu_l}| \ll 1`. In this case, we can regard :math:`\acute{V}` as a Voigt profile itself, that is, :math:`\acute{V}(q;\tilde{\beta};\tilde{\gamma}_L) \approx V(q;\tilde{\beta};\tilde{\gamma}_L)`. Then we find

:math:`\sigma (q_i) = \sum_{k} \mathrm{FT}^{-1} (\tilde{\mathfrak{S}}_{jk} K_{ij})`,

where :math:`\tilde{\mathfrak{S}}_{jk}` is the discrete Fourier conjugate of :math:`{\mathfrak{S}}_{jk}` for the :math:`q` direction (index of :math:`j`). The kernel for the Voigt profile

:math:`K_{jk} \equiv \exp{(-2  \pi^2 \tilde{\beta}^2 \tilde{q}_j^2 - 2 \pi \tilde{\gamma}_{L,k} |\tilde{q}_j|  )}`

is given in `ditkernel module <../exojax/exojax.spec.html#module-exojax.spec.ditkernel>`_, where :math:`\tilde{q}_j` is the conjugate of :math:`q_i`. In addition, we include the correction of the aliasing effect of the Lorentz profile (van den Bekerom et al. in preparation) in ExoJAX. Because the molecular mass and temperature should be common in the above formulation, we use this equation to compute the synthesized cross section for each isotope and atmospheric layer. 



How is the LSD computed in ExoJAX/MODIT?
------------------------------------------

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

We need to compute contributions to the LSD for 'wav' and 'gammaL'. MODIT/DIT uses the linear interpolation to the grid values. This can be done by `getix <../exojax/exojax.spec.html#exojax.spec.lsd.getix>`_. The following code is an example how to use 'getix'. Here we have a grid of 'yv=[0,1,2,3,4,5]' and want to compute the contribution and indices of 'y=1.1' and 'y=1.4'. The outputs are the contribution at i=index+1 and index, i.e. the contribution of 'y=1.1' to the LSD is 0.1 at index=2 (yv=2) and 1-0.1 = 0.9 at index=1 (yv=1) and 'y=4.3' to the LSD is 0.3 at index=5 (yv=5) and 1-0.3 = 0.7 at index=4 (yv=4). 

.. code:: ipython

       >>> #An example how to use getix
       >>> from exojax.utils.indexing import getix
       >>> import jax.numpy as jnp
       >>> y=jnp.array([1.1,4.3])
       >>> yv=jnp.arange(6)
       >>> getix(y,yv)
       (DeviceArray([0.10000002, 0.3000002 ], dtype=float32), DeviceArray([1, 4], dtype=int32))    

       
For wavenumber, the F64 precision is required. So, `npgetix <../exojax/exojax.spec.html#exojax.spec.lsd.npgetix>`_ is used for precomputation, which is numpy version of getix. Then, back to the original problem, we need to pre-compute the contribution and index for wavgrid as follows:

       
.. code:: ipython
              
       >>> from exojax.utils.indexing import npgetix
       >>> cx, ix=npgetix(wav,wavgrid)
       >>> cx, ix
       (array([0.29999995]), array([2]))

`inc2D_givenx <../exojax/exojax.spec.html#exojax.spec.modit.inc2D_givenx>`_ computes the LSD with 'y' and 'yv' and a given contribution for 'x'. 'w' is the weight, i.e. the line strength. Then, we get the LSD for the line as follows: 
       
.. code:: ipython
              
       >>> from exojax.opacity.modit.modit import inc2D_givenx
       >>> inc2D_givenx(lsd,w,cx,ix,ngammaL,ngammaLgrid)
       DeviceArray([[0.        , 0.        , 0.        , 0.        ],
                    [0.        , 0.        , 0.        , 0.        ],
                    [0.        , 0.56      , 0.14000005, 0.        ],
                    [0.        , 0.23999995, 0.06000001, 0.        ],
                    [0.        , 0.        , 0.        , 0.        ]],            dtype=float32)


Spectral Operators (``sop``)
==================================

Last Update: April 11th (2024) Hajime Kawahara

In the post-radiative transfer, the observed spectrum differs from the raw spectrum due to several modifications.
For instance, it might experience rotational broadening due to the planet's rotation, wavelength shifts 
due to differences in line-of-sight velocities, or the influence of the instrument's profile (IP). 
In ExoJAX, these responses to the spectrum are termed the "Spectral Operator" (``sop``). 
Within the ``postproc.specop`` module, classes like ``SopRotation`` and ``SopInstProfile`` allow for the easy handling of these responses.


SopRotation
-----------------------

``SopRotation`` provides an operator for the Doppler broadening caused by the rotation of spherical bodies, such as planets and stars.
Currently, only rigid rotation has been implemented. 
See 
:doc:`../tutorials/get_started`
for example.

SopInstProfile
-----------------------

On the other hand, ``SopInstProfile`` convolves instrument-derived profiles or converts them into the instrument's sampling. 
Currently, only the Gaussian profile (``ipgauss``) has been implemented for the former. For the latter, a ``sampling`` instance is used.

See 
:doc:`../tutorials/get_started`
again for example.

Velocity range for a Gaussian instrumental profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The instrumental resolving power is supplied through the Gaussian standard
deviation passed to ``ipgauss``; it is not inferred from the bin spacing of
``nu_grid``. For an instrumental resolving power :math:`R_\mathrm{inst}`,

.. math::

    \beta_\mathrm{inst}
    = \frac{c}{2\sqrt{2\ln 2}\,R_\mathrm{inst}},

where :math:`\beta_\mathrm{inst}` is in km/s. The velocity range should cover
at least five Gaussian standard deviations:

.. math::

    \mathtt{vrmax} \geq 5\beta_\mathrm{inst}.

``SopInstProfile.check_vrmax`` performs this check explicitly and emits a
``UserWarning`` if the configured range is too small. ``ipgauss`` also calls
this check automatically.

For repeated JIT-compiled calculations such as NumPyro NUTS, determine
``vrmax`` and create ``SopInstProfile`` before starting the inference. This
keeps the velocity-grid shape fixed throughout the calculation.

.. code:: python

    from exojax.postproc.specop import SopInstProfile
    from exojax.utils.instfunc import resolution_to_gaussian_std

    Rinst = 3000.0
    beta_inst = resolution_to_gaussian_std(Rinst)
    vrmax_inst = 5.0 * beta_inst

    sop_inst = SopInstProfile(nu_grid, vrmax=vrmax_inst)
    sop_inst.check_vrmax(beta_inst)

    # This call repeats the same check before applying the convolution.
    spectrum_inst = sop_inst.ipgauss(spectrum, beta_inst)


Convolution methods available in sop
---------------------------------------

Both ``SopRotation`` and ``SopInstProfile`` use FFT for convolution. 

- ``convolution_method = "exojax.signal.convolve"`` : FFT-based convolution

When the number of grid points in the input spectrum is large, 
this can cause memory overflow and slow down the calculation speed. For such situations, the OLA (Overlap and Add) method, 
which divides the input into a suitable number of parts and performs FFT on each, can be used. 
Try the following option during the initialization of ``sop``:

- ``convolution_method = "exojax.signal.ola"`` : Overlap-and-Add convolution, One can change the number of the division by ``sop.ola_ndiv`` (default=4).







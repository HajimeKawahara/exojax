Spectral Operators (``sop``)
==================================

Last Update: April 11th (2024) Hajime Kawahara

In the post-radiative transfer, the observed spectrum differs from the raw spectrum due to several modifications.
For instance, it might experience rotational broadening due to the planet's rotation, wavelength shifts 
due to differences in line-of-sight velocities, or the influence of the instrument's profile (IP). 
In ExoJAX, these responses to the spectrum are termed the "Spectral Operator" (``sop``). 
Within the ``spec.specop`` module, classes like ``SopRotation`` and ``SopInstProfile`` allow for the easy handling of these responses.


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


Convolution methods available in sop
---------------------------------------

Both ``SopRotation`` and ``SopInstProfile`` use FFT for convolution. 

- ``convolution_method = "exojax.signal.convolve"`` : FFT-based convolution

When the number of grid points in the input spectrum is large, 
this can cause memory overflow and slow down the calculation speed. For such situations, the OLA (Overlap and Add) method, 
which divides the input into a suitable number of parts and performs FFT on each, can be used. 
Try the following option during the initialization of ``sop``:

- ``convolution_method = "exojax.signal.ola"`` : Overlap-and-Add convolution, One can change the number of the division by ``sop.ola_ndiv`` (default=4).








Spectral Operators (sop)
==========================

In the post-radiative transfer, the observed spectrum differs from the raw spectrum due to several modifications.
For instance, it might experience rotational broadening due to the planet's rotation, wavelength shifts 
due to differences in line-of-sight velocities, or the influence of the instrument's profile (IP). 
In ExoJAX, these responses to the spectrum are termed the "Spectral Operator." 
Within the spec.specop module, classes like SopRotation and SopInstProfile allow for the easy handling of these responses.


SopRotation
-----------------------

SopInstProfile
-----------------------

Convolution methods available in sop
---------------------------------------

- convolution_method="exojax.signal.convolve" : FFT-based convolution

- convolution_method="exojax.signal.ola" : Overlap-and-Add convolution, One can change the nuber of the division by self.ola_ndiv (default=4).







Radiative Transfer
======================

Exojax uses a layer-based atmospheric model for `radiative transfer <https://en.wikipedia.org/wiki/Radiative_transfer>`_ (RT). 
In ExoJAX, one can utilize spectral models for emission, reflection, and transmission. This necessitates solving for radiative transfer. 
There are various methods to solve radiative transfer, and the following describes those available in ExoJAX.

Regarding emission in ExoJAX, there are two types: without scattering and with scattering. 
The non-scattering type assumes **pure absorption**, for which there are two methods: 
one that transfers flux (**fbased**) and another that transfers intensity (**ibased**).

For emission with scattering in ExoJAX, there are implementations for treating the scattering component as an effective reflectivity 
using **the flux adding treatment** (Robinson and Salvador), and as an effective transmission using **LART** method.
These are the fbased computation.

Regarding reflected light in ExoJAX, the flux-adding treatment can be utilized.

These are currently all based on the two-stream approximation. In the future, a four-stream implementation is planned, but as of December 2023, 
it has not yet been implemented.

For transmission spectroscopy in ExoJAX, the options are primarily limited to differences in the integration methods. 
Both the Trapezoid integration method and the method using Simpson's rule are available.

.. toctree::
    :maxdepth: 1

    rtransfer_fbased_pure.rst
    rtransfer_ibased_pure.rst
    rtransfer_fbased.rst
	rtransfer_fbased_reflection.rst
    rtransfer_transmission.rst

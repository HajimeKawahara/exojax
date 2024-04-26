Radiative Transfer
======================

Radiative Transfer Schemes in ExoJAX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../rt.png

ExoJAX supports the radiative transfer functionalities of the emission (pure absorption, incl. scattering), reflection, and transmission spectra.

Exojax uses a layer-based atmospheric model for `radiative transfer <https://en.wikipedia.org/wiki/Radiative_transfer>`_ (RT). 
In ExoJAX, one can utilize spectral models for emission, reflection, and transmission. This necessitates solving for radiative transfer. 
There are various methods to solve radiative transfer, and the following describes those available in ExoJAX.

Regarding emission in ExoJAX, there are two types: without scattering and with scattering. 
The non-scattering type assumes **pure absorption**, for which there are two methods: 
one that transfers flux (**fbased**) and another that transfers intensity (**ibased**).

For emission with scattering in ExoJAX, there are implementations for treating the scattering component as an effective reflectivity 
using **the flux-adding treatment** (`Robinson and Crisp 2018 <https://www.sciencedirect.com/science/article/pii/S0022407317305101?via%3Dihub>`_), 
and as an effective transmission using the **LART** method.
These are the fbased computation.

Regarding reflected light in ExoJAX, the flux-adding treatment can be utilized.

All of the fbased schemes are currently based on the two-stream approximation, althogh the ibased schemes can specify the number of the streams. 
In the future, a four-stream implementation for the fbased schemes is planned, but as of December 2023, 
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


Atmospheric Radiative Transfer (art) class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ExoJAX's code is primarily written in a function-based manner, allowing for the execution of each process of radiative transfer individually. 
However, for those who are not interested in the details, the ``art`` class can be utilized as an interface for radiative transfer.

+-----------------------+------------------+----------------+
|**art** in atmrt.py    |spectrum type     |including...    |
+-----------------------+------------------+----------------+
|ArtEmisPure            |Emission          | no scattering  |
+-----------------------+------------------+----------------+
|ArtEmisScat            |Emission          | w/ scattering  |
+-----------------------+------------------+----------------+
|ArtReflectPure         |Reflection        | no emission    |
+-----------------------+------------------+----------------+
|ArtReflectEmis         |Reflection        | w/ emission    |
+-----------------------+------------------+----------------+
|ArtTransPure           |Transmission      |                |
+-----------------------+------------------+----------------+

See the following APIs for the details of these art classes:

- `exojax.spec.atmrt.ArtEmisPure <../exojax/exojax.spec.html#exojax.spec.atmrt.ArtEmisPure>`_
- `exojax.spec.atmrt.ArtEmisScat <../exojax/exojax.spec.html#exojax.spec.atmrt.ArtEmisScat>`_
- `exojax.spec.atmrt.ArtReflectPure <../exojax/exojax.spec.html#exojax.spec.atmrt.ArtReflectPure>`_
- `exojax.spec.atmrt.ArtReflectEmis <../exojax/exojax.spec.html#exojax.spec.atmrt.ArtReflectEmis>`_
- `exojax.spec.atmrt.ArtTransPure <../exojax/exojax.spec.html#exojax.spec.atmrt.ArtTransPure>`_


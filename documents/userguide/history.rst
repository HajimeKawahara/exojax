History
===============

Version 2.0
-----------------------

In this update, we include further refactoring, additional documentation, and new features for device-memory saving algorithms, 
`opart <https://github.com/HajimeKawahara/exojax/issues/542>`_
and The 
`nu-stitching <https://github.com/HajimeKawahara/exojax/issues/542>`_
, which were not described in the initial submission 
(
`version 1 on the ArXiv manuscript <https://arxiv.org/abs/2410.06900v1>`_ 
). 

Opart: layer-by-layer opacity and radiative transfer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The 
`opart <https://github.com/HajimeKawahara/exojax/issues/542>`_
method has recently demonstrated at least comparable computation times to conventional methods for cases like pure absorption with emission (``EmisPure``). By calculating without separating ``opa`` and ``art``, it eliminates dependency on ``nlayer`` for memory requirements. This means that calculations can likely be performed even on GPUs with limited device memory, such as those without the large 80GB memory capacity of an A100 GPU. We consider this a valuable addition for version 2.
See 
`here <https://secondearths.sakura.ne.jp/exojax/tutorials/get_started_opart.html>`_
as a tutorial.

Available Opart classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``OpartEmisPure``
- ``OpartReflectPure``
- ``OpartEmisScat``
- ``OpartReflectEmis``

#542 #546 #547 #558 #565 #559 #556 #557 #568

Wavenumber Stitching ($\nu$-stitching)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We implemented the 
`nu-stitching <https://github.com/HajimeKawahara/exojax/issues/566>`_
, the method to divide the wavenumber grid, using 
`OLA <https://secondearths.sakura.ne.jp/exojax/tutorials/Open_Close_Cross_Section.html>`_
, in `PreMODIT` as an option (with ``nstitch`` and ``cutwing`` arguments). 

.. code-block:: python

    from exojax.opacity import OpaPremodit
    opa = OpaPremodit(mdb, nus, nstitch=10, auto_trange=[500,1300], cutwing = 0.015)

See 
`here <https://secondearths.sakura.ne.jp/exojax/tutorials/Cross_Section_using_OpaStitch.html>`_ 
for the details.

Note that $\nu$-stitching can be used even in ``opart``.

#566  #581

Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- (**OpaPremodit, OpaModit**) Reduction in device memory usage by performing zero-padding within the scan #577 #578
- (**MdbExomol**) Exomol `broadener_species` option, the capability of `air` and/or other species #580

Documentation Update
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- visualization of LBD #548 #553
- GP modeling in getting started #554
- How to use SVI with ExoJAX #572 
- How to use Nested Sampling with ExoJAX #107 #570
- others #560 #561

Removed 
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- spec.modit.calc_xsection_from_lsd #576
- spec.modit.xsvector #576
- spec.modit.xsmatrix #576

Version 1.6 
-----------------------


The target for version 1.6 is to introduce features that facilitate the application to more practical data.
Also, v 1.6 will be the fixed version of ExoJAX2 prior to the paper II submission. It is planned to be released as a pre-release of ExoJAX (ver 2).

- practical use of a cloud model. See ``documents/tutorials/jupiters``
- comprehensive review
- eq methods (and neq ) in ``opa`` and ``mdb``  to enable to use == (and !=) operators. #509
- compatible to radis 0.15.2, in particular, "vaex as an option" strategy #500 #506
- as a result, ExoJAX supports python 3.9 3.10, 3.11, 3.12,  
- new ``art``, ``ArtAbsPure``, computes a pure absorption spectrum w/ and w/o a single reflection at Psurf.   

Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Comparison with a tabulated cross section #491
- with pRT #427

Bug Fixes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- side effect on a multiple call of ``opa`` #510 
- Stark broadening #489
- solar abundance definition (``utils.zsol.nsol``)  #535

Changes 
^^^^^^^^^^^^^^^^^^^^^^^^^^^

removed
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``recexomol`` #501
- ``lpf.auto_crosssection`` #498
- ``utils.recexomol`` #501
- ``Sij0`` attribute from api.py (``MdbExomol``, ``MdbHitemp``, ``MdbHitran``) #515
- ``atm.mixratio.vmr2mmr``, ``atm.mixratio.mmr2vmr`` (use ``atm.atmconvert.vmr_to_mmr``, ``atm.atmconvert.mmr_to_vmr`` instead) 
- ``spec.dit.dtauM_vald_old``

all methods in ``dynamics`` package
- ``getE.getE``, ``rvfunc.rvcoref``, ``rvfunc.rvf``,``rvfunc.rvf2``, ``rvfunc.rv_semi_amplitude``, ``rvfunc.rvf1``,

several methods in ``spec.rtransfer`` (deprecated since v1.5)
- ``dtauM``,``dtauCIA``, ``dtauHminus``, ``dtauVALD``, ``pressure_layer``

renamed
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``line_strength_ref_original`` was introduced for line strength at T=Tref_original, instead of ``line_strength_ref`` #515
- ``self.line_strength_ref`` (which depends on self.Tref) → ``self.line_strength(Tref)`` #515
- ``instances_from_dataframe`` → ``attributes_from_dataframe``

changes arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``qr_interp(T)`` → ``qr_interp(T,Tref)`` in ``MdbExomol`` #515
- ``qr_interp(iso,T)`` → ``qr_interp(iso,T,Tref)`` in ``MdbHitemp/Hitran`` #515
- ``qr_interp_line(T)`` → ``qr_interp_line(T,Tref)`` in ``MdbHitemp/Hitran`` #515

new methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- computes X, Y, Z, ``utils.zsol.mass_fraction_XYZ(number_ratio_elements)`` #535



Version 1.5
-----------------------

- Intensity-based radiative transfer for pure absorption #411 #412 #419
- Toon-type two-stream radiative transfer (with scattering/reflection), flux-adding (and LART) #428 #477
- Forward modeling of the reflection spectrum, with an example of Jupiter's reflection spectrum. #477
- Transmission code improvement (Simpson) and more tests #421 #424 #463 #464
- Rayleigh scattering #115 #430 #434
- Cloud (Mie) scattering using Ackerman and Marley cloud model #477

Version 1.4
-------------------

- transmission spectra #356
- operators on spectra class #363
- multiple molecules handler #388

ExoJAX now recommends using 64-bit as default, but can use 32 bit if you are confident (and for real science use).


Version 1.3
-------------------

- more GPU memory saved method in PreMODIT (so called diffmode) #332
- opacity calculator class opa #346 see this tutorial
- atmospheric RT class art #346
- optimal Elower max (reducing device memory use) in PreMODIT #331 #332 see this tutorial
- optional quantum states #336 #338 see this tutorial
- faster IP and spin convolutions #350
- molecular mass mdb.molmass available #328

Version 1.2
-------------------

- Moved on a common I/O of molecular/atomic database with radis.api for ExoMol/HITRAN/HITEMP #272
- Removed old moldb for ExoMol/HITRAN/HITEMP
- PreMODIT algorithm, applicable to a wide wavenumber range #265 #287 #288 #307
- Memory saved version of spin rotation and instrumental response #295


Version 1.1
-------------------

- VALD3
- reverse mode available
- compatibility with JAXopt #212 

Version 1.0
-------------------

- Auto-differentiable Spectrum Model of exoplanets/brown dwarfs built on JAX using the molecular/atomic database, ExoMol, HITRAN/HITEMP and VALD3.
- Bayesian inference using HMC-NUTS/NumPyro
- Two opacity calculators available: Direct computation of the Voigt line profile (lpf) and the modified discrete integral transform (MODIT). The latter is a fast opacity calculator for the number of lines N >~ 1000.
- Transparent open-source code with documentations, including a peer-reviewed paper, API, user guide, and tutorials using real data
- HITRAN/CIA and H- as continuous opacity
- Cloud modeling based on Ackerman and Marley
- Quick computation of the opacity and emission spectra for observers (autospec)

Before Version 1
----------------------


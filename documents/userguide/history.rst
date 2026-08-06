History
===============

Version 2.6.0
---------------

ExoJAX 2.6.0 expands correlated-k and radiative-transfer workflows, adds a
minimal hydrogen Balmer-line opacity path, and improves inference-oriented
spectral post-processing.

Highlights
^^^^^^^^^^^^^^^^^^^

- Added wide-wavelength JWST correlated-k examples and support for
  patch-based, bounded-memory CKD table precomputation (#713, #718, #721).
- Added intensity outputs and direct quadratic limb-darkening estimation for
  pure-emission, CKD, and Opart workflows (#717).
- Added the SFM-2st source-function correction for emission with scattering,
  including CKD support (#719, #720).
- Added a minimal hydrogen Balmer-line database and direct natural+Doppler
  Voigt opacity. Stark broadening and the other follow-up items in #722 are
  outside this release (#723).
- Added validation and warnings for an insufficient instrumental-profile
  velocity range (#725, #726).
- Added ``sampling_band_integral`` for sampling spectra with band integrals
  (#727).
- Restored the deprecated ``exojax.spec`` import compatibility layer for the
  package split introduced in v2.2. These shims remain scheduled for removal
  in v3.
- Restored the VALD PyTables cache backend and made it the dependency-safe
  default. The legacy ``pandas`` engine name remains an alias, while ``vaex``
  is still available as an optional backend.
- Set the supported Python floor to 3.10 and aligned package metadata with the
  Python 3.10--3.13 CI matrix.
- Refactored and refreshed the documentation navigation and introductory
  material (#714, #716).

**Full Changelog**: https://github.com/HajimeKawahara/exojax/compare/v2.5.0...v2.6.0

Version 2.2.2
---------------

What's Changed
^^^^^^^^^^^^^^^^^^^

* Tutorial update by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/641
* url for HITRAN/CIA updated by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/644
* version 2.2.2 by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/645


**Full Changelog**: https://github.com/HajimeKawahara/exojax/compare/v2.2.1...v2.2.2

Version 2.2.1
-----------------

- bug fix for when loading OpaPremodit with MdbHitemp #639

- added save/load mode for `OpaCKD` #635 #636 #637 

What's Changed
^^^^^^^^^^^^^^^^^^^^^

* send Release v2 2 0 back to develop by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/633
* Add NPZ-based I/O support for OpaCKD precomputed tables by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/635
* Opackd loadonly by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/636
* added HMC example for OpaCKD by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/637
* Tutorials update by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/638
* Bugfix hitemp opapremodit by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/639
* version 2.2.1 by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/640

Full Changelog: https://github.com/HajimeKawahara/exojax/compare/v2.2.0...v2.2.1


Version 2.2.0
-----------------------
The main target for version 2.2.0 is the capability of differentiable TCE (thermal chemical equilibrium).
Note that we are developing the differentiable TCE package `ExoGibbs <https://github.com/HajimeKawahara/exogibbs>`_.

`Equilibrium Retrieval <https://secondearths.sakura.ne.jp/exojax/tutorials/equilibrium_chemistry.html>`_

Overview
^^^^^^^^^^^^^^

Release 2.2.0 (branch ``release_v2_2_0``) is a substantial update since tag v2.1,
focusing on PreMODIT performance/memory, database API refactors, packaging modernization,
and new snapshot/provider abstractions to decouple opacity from DB classes.
Summary below is based on local Git history; I can cross-check the milestone if you paste it.

pyproject
^^^^^^^^^^^^^^^^^^^^

We migrate ExoJAX from ``setup.py`` to a modern PEP 517/518 and PEP 621 compliant ``pyproject.toml`` build. (#626)

restructure
^^^^^^^^^^^^^^^^^

- ``MdbExomol``, ``MdbHitemp``, ``MdbHitran``, ``MdbHargreaves``, ``AdbVald``, ``AdbSepVald``, ``AdbKurucz``, ``XdbExomolHR``, ``PdbCloud``, ``CdbCIA`` moved to subdirectories under ``database`` (#622, #623, #624)

These classes can be called as::

   from exojax.database import MdbExomol

Connect MDB to OpaPremodit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**What’s new:** OpaPremodit now accepts databases via high-level helpers and data-only snapshots.
This decouples opacity from DB classes, improves memory use, and keeps APIs clean.

Quick Start (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

- ExoMol::

    mdb = MdbExomol(".../CO/12C-16O/Li2015", nu_grid, crit=1e-30, Ttyp=1000.0)
    opa = OpaPremodit.from_mdb(
        mdb, nu_grid,
        auto_trange=(500.0, 1500.0),
        broadening_resolution={"mode": "manual", "value": 0.2},
        memory_policy=MemoryPolicy(allow_32bit=True, nstitch=4, cutwing=1.0)
    )

- HITRAN/HITEMP::

    mdb = MdbHitran("path/to/CO.par", nu_grid, isotope=0)
    opa = OpaPremodit.from_mdb(mdb, nu_grid, ...)

Example::

   from exojax.database.exomol.api import MdbExomol
   from exojax.opacity.premodit.api import OpaPremodit
   from exojax.opacity.policies import MemoryPolicy

   mdb = MdbExomol(".database/CO/12C-16O/Li2015", nu_grid)
   opa = OpaPremodit.from_mdb(
       mdb, nu_grid,
       auto_trange=(500,1500),
       broadening_resolution={"mode":"manual","value":0.2},
       memory_policy=MemoryPolicy(allow_32bit=True, nstitch=2)
   )
   xs = opa.xsvector(T=1200.0, P=0.1)  # or opa.xsmatrix(Tarr, Parr)

Snapshots (advanced/portable)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Export once, construct anywhere::

    snap = mdb.to_snapshot()
    opa = OpaPremodit.from_snapshot(snap, nu_grid)

- Optional providers override:
    - **Partition:** ExoMol/HITRAN providers are auto-selected; pass a custom ``pf_provider`` if needed.
    - **Broadening:** ExoMol/HITRAN strategies are auto-selected; pass a custom ``broadening_strategy`` to override.

Memory and Runtime Controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``memory_policy``: central knob for ``allow_32bit``, ``nstitch`` (ν-stitching), and ``cutwing``.
  Policy values override constructor args.

- ``broadening_resolution``:
    - ``manual``: fixed resolution (highest memory)
    - ``minmax``: DB-driven min/max grids (medium memory)
    - ``single``: single broadening set (lowest memory)

Notes
~~~~~

- ``from_mdb`` calls ``.to_snapshot()`` internally and frees the DB by default
  (``delete_mdb_after_init=True``). Keep it with ``delete_mdb_after_init=False``.
- Ensure MDB lines intersect ``nu_grid``; otherwise a ``ValueError`` is raised.
- Direct constructor ``OpaPremodit(mdb, nu_grid, ...)`` remains supported;
  ``from_mdb`` is the preferred path for clarity and memory efficiency.

Highlights
^^^^^^^^^^^^^^^

- PreMODIT: adds ``MemoryPolicy``, provider-based partition/broadening,
  ``from_mdb`` / ``from_snapshot`` constructors, and precomputed ``line_strength_Tref`` for speed.
- Snapshots: new ``MDBSnapshot`` / ``MDBMeta`` / ``Lines`` DTOs;
  ``MdbExomol``, ``MdbHitran``, ``MdbHitemp`` implement ``.to_snapshot()``.
- Database refactor: split ExoMol vs HITRAN/HITEMP broadening paths;
  reorganized modules; added ``XdbExomolHR`` for high-res empirical lines.
- Atomic DB (VALD/Kurucz): ``AdbVald``, ``AdbSepVald``, ``AdbKurucz`` exposed via ``exojax.database``.
- CIA: dedicated ``exojax.database.cia.api.CdbCIA`` with ``contdb`` kept as a deprecation shim.
- Float32: optional 32-bit mode for PreMODIT (faster + lower memory).
- Chemistry: groundwork to connect equilibrium chemistry (``exogibbs``), new tutorial/docs, and H2 CIA usage in examples.
- Packaging: switch to ``pyproject.toml`` + setuptools-scm; Python ≥ 3.9; CI config updates.
- Tests/Docs: snapshot tests for MDBs, additional opacity tests, relaxed tolerances where needed; “Get Started” refresh.

Breaking/Deprecations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``exojax.database.moldb`` removed; use ``exojax.database.exomol`` / ``hitran`` / ``hitemp`` classes (``MdbExomol``, ``MdbHitran``, ``MdbHitemp``) or atomic ``Adb*``.
- Prefer ``exojax.database.cia.api.CdbCIA``; ``contdb.CdbCIA`` emits ``DeprecationWarning``.
- Minimum Python now 3.9 (declared in ``pyproject.toml``).

Included PRs (since v2.1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- #631 di_mdb_opa: DB–opacity integration refactor, memory/policies/providers, tests/docs.
- #630 premodit_mem: PreMODIT memory improvements + snapshot path and tests.
- #629 connect_exogibbs: equilibrium chemistry integration + tutorial.
- #626 pyproject: packaging migration, Python version floor, CI.
- #624 mmrefacmdb / #623 morerefactmdb / #622 refacmdb: DB refactors (molinfo/qstate moves, ExoMolHR restructure, HITRAN/HITEMP splits).
- #620 32bitxs: float32 cross-section support + tests.

bug fixes
^^^^^^^^^^^^^^^


- #619 #620 32-bit overflow

What's Changed
^^^^^^^^^^^^^^^

* bug fix of the overflow of the line strength when 32 bit mode by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/620
* refactoring database module by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/622
* More refactoring of database by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/623
* refactoring database (cia, molinfo) by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/624
* Pyproject by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/626
* HMC retrieval of the Equilibrium Chemistry, Connecting to ExoGibbs by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/629
* Detach OpaPremodit from mdb to reduce memory footprint by @sh-tada in https://github.com/HajimeKawahara/exojax/pull/630
* PreMODIT DI: value object, MemoryPolicy, ctor parity, and docs (non-breaking) by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/631
* Release v2.2.0 by @HajimeKawahara in https://github.com/HajimeKawahara/exojax/pull/632

**Full Changelog**: https://github.com/HajimeKawahara/exojax/compare/v2.1...v2.2.0

Version 2.1
-----------------------

The main target of v2.1 is major refactoring for the new development of ExoJAX3 (#606 #607 #608) and to implement the differentiable CKD (#484).
This capability will be used to computes magnitude and/or low/mid res spectrum to calibrate high-res spectra.


New Features
^^^^^^^^^^^^^^^^^

- Correlated-k distribution (CKD; beta-version), #484 #610 #614
  - `tutorial for emission <https://secondearths.sakura.ne.jp/exojax/tutorials/ckd_emispure.html>`_
  - `tutorial for transmission <https://secondearths.sakura.ne.jp/exojax/tutorials/ckd_transpure.html>`_
  - `what is CKD? <https://secondearths.sakura.ne.jp/exojax/tutorials/ckd_principle.html>`_

- extra database for ExoMolHR #595 #600 , see this `tutorial <https://secondearths.sakura.ne.jp/exojax/tutorials/exomolhr.html>`_
- FeH line list by Hargreaves et al. (2010), #603 by @YuiKasagi
- checkpoint (memory release) for transmission reverse-mode diff #604 by @sh-tada

See #597 as direction toward ExoJAX3

Refactoring
^^^^^^^^^^^^^^^^^

Separates database, opacity, rt, postproc from spec, and a script for fixing 
#606 #607 #611

The main change in this update is the redistribution of modules previously located in `exojax.spec` into four function-specific subpackages: `exojax.database` (db), `exojax.opacity` (opa), `exojax.rt` (art), and `exojax.postproc`.
To assist with this transition, we have included a script that automatically updates your import statements from `exojax.spec` to their appropriate new locations.
The script is located at `exojax/scripts/fix_v2_1.py`. You will need to install libcst to run it.

... code-block:: sh

    pip install libcst

Please make sure to back up your original files before using the script.
Usage instructions are as follows:

... code-block:: sh

    python -m fix_v2_1 your_project/


- `OpaPremodit`, `OpaDirect`, `OpaModit` can be called such as  

... code-block:: python

    from exojax.opacity import OpaPremodit

Bug Fixes
^^^^^^^^^^^^^
-  Reflect updates to astropy (unit "Angstrom") by @YuiKasagi  #615

Removed methods
^^^^^^^^^^^^^^^^^^^^^^^^

- removed `dtau_mmwl`
- merged `nu2wav` and `wav2nu` from `unitconvert` to `utils.grids`

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

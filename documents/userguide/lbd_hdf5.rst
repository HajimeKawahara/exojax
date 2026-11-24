Distributing PreMODIT LBDs as HDF5
==================================

``save_lbd`` exports the prepared log-biased line shape density (LBD), coordinate grids,
partition function and metadata needed by another PreMODIT implementation.
``load_lbd`` returns NumPy arrays and metadata without loading the original
line list. Use :doc:`saveopa` to save and restore an ExoJAX calculator instead.

The format supports one ExoMol isotopologue and one broadener per file,
``diffmode=0, 1, 2``, float64 coefficients, and ``nstitch=1``. The broadener may
have different widths for different lines; this does not require the
``single_broadening`` approximation. The HDF5 file contains no per-line
arrays, Python objects or serialized provider classes.

Creating a file
---------------

Enable float64 before generating the LBD. The existing optional PyExoCross
backend can be selected explicitly; the supported version is pinned by
``exojax[pyexocross]``. This example uses a small band and an interior interval
to illustrate the metadata. Establish the usable interval and pressure range
by comparison with a wider calculation before distributing a real dataset.

.. code-block:: python

    import numpy as np
    from jax import config

    config.update("jax_enable_x64", True)

    from exojax.database.exomol.api import MdbExomol
    from exojax.opacity import OpaPremodit, save_lbd, load_lbd

    nu_grid = np.geomspace(2000.0, 2010.0, 10000)  # cm^-1
    mdb = MdbExomol(
        ".database/CO/12C-16O/Li2015",
        nurange=nu_grid,
        backend="pyexocross",
        gpu_transfer=False,
        bkgdatm="H2",
        crit=0.0,
        Ttyp=1000.0,
    )
    opa = OpaPremodit.from_snapshot(
        mdb.to_snapshot(), nu_grid,
        auto_trange=(500.0, 1500.0), diffmode=2, nstitch=1,
    )
    metadata = {
        "mol_name": "CO",
        "key_iso_ll": "12C-16O__Li2015",
        "DOI": "10.1088/0067-0049/216/1/15",
        "Date_ID": "co-h2-lbd-v1",
        "broadener": "H2",
        "valid_wavenumber_range": [2001.0, 2009.0],
        "provenance": {
            "loader": {"name": "pyexocross", "version": "1.1.16"},
            "source_url": "https://www.exomol.com/db/CO/12C-16O/Li2015/",
            "line_selection": {"crit": 0.0, "Ttyp": 1000.0, "elower_max": None},
        },
    }
    save_lbd(opa, "12C-16O__Li2015.H2.premodit.h5", metadata=metadata)
    arrays, saved_metadata = load_lbd("12C-16O__Li2015.H2.premodit.h5")

The writer requires the four identity strings, ``broadener``,
``valid_wavenumber_range`` and a JSON-compatible ``provenance`` dictionary.
``DOI`` identifies the line-list publication; ``Date_ID`` identifies the LBD
data release. Keep the source line-list release and source file hashes in
``provenance``. Optional ``validation`` metadata can describe the tested
temperature and pressure domain, comparison method and measured errors.
Its default is an empty dictionary, which makes no precision claim. The usable
interval must lie inside the generating grid and is supplied by the producer,
not inferred by the writer.

For reproducibility, also record the actual input files, broadening recipe,
fallback widths and exponents, intensity-cutoff units and reference
temperature, energy cutoff, and any additional line selection. Selecting
PyExoCross does not by itself establish agreement with PyExoCross spectra:
the ExoJAX adapter still calculates line strengths and assigns broadening.
The snapshot and calculator do not preserve all source metadata, so collect
it while constructing the molecular database.

The writer uses standard lossless gzip and shuffle compression. It publishes
the completed file only after writing succeeds and refuses to overwrite an
existing path. Float32 coefficients are rejected rather than promoted after
generation. Neither writing nor reading downloads molecular data.

HDF5 contract
-------------

The root attributes identify ``format="exojax-premodit-lbd"`` and
``schema_version="1.0"``. Compatibility is determined by the schema and
encoding, not by an exact match to the producer's ExoJAX version. The reader
accepts version ``1.x`` with a nonnegative integer minor version and the
required 1.0 fields; unknown additive metadata is ignored. Other major
versions and unknown coefficient encodings are rejected.

The root datasets ``mol_name``, ``key_iso_ll``, ``DOI`` and ``Date_ID`` are
UTF-8 strings. ``mol_mass`` is the actual isotopologue mass used for Doppler
broadening, with ``units="u"``. ``provenance_json`` holds source information,
the broadener, usable wavenumber interval and producer/reference settings.
These names follow ExoMolOP metadata conventions, but the numerical payload
is specific to PreMODIT.

The writer reserves these keys in ``provenance_json``; user ``provenance``
must not override them:

* ``broadener``, ``valid_wavenumber_range`` and ``validation`` come from the
  corresponding input metadata fields.
* ``writer`` contains ``exojax_version`` and ``created_at``.
* ``reference_evaluation`` contains ``profile_kernel``, ``nstitch=1``,
  ``alias="close"`` and ``natural_broadening=False``.
* ``abundance_convention="isotopologue_no_abundance_weighting"`` and
  ``pressure_convention="broadener_partial_pressure"`` define the strength
  and pressure interpretation.

.. list-table:: Numerical datasets
   :header-rows: 1
   :widths: 42 23 35

   * - Path
     - Shape
     - Meaning / units
   * - ``/premodit/lbd_coeff``
     - ``(D+1, Nnu, Nb, NE)``
     - Mixed coefficient encoding described below
   * - ``/premodit/nu_grid``
     - ``(Nnu,)``
     - Ascending, equally spaced in log wavenumber; ``cm^-1``
   * - ``/premodit/elower_grid``
     - ``(NE,)``
     - Lower-state energy; ``cm^-1``
   * - ``/premodit/ngamma_ref_grid``
     - ``(Ngamma,)``
     - Normalized Lorentz HWHM at the reference conditions; ``1``
   * - ``/premodit/n_Texp_grid``
     - ``(Nn,)``
     - Broadening temperature exponent; ``1``
   * - ``/premodit/multi_index_uniqgrid``
     - ``(Nb, 2)``
     - Integer indices into width and exponent grids, respectively
   * - ``/partition_function/temperature``
     - ``(NQ,)``
     - Partition-function temperature; ``K``
   * - ``/partition_function/Q``
     - ``(NQ,)``
     - Partition function used during generation; ``1``

``D`` is ``diffmode``. The LBD attributes are
``encoding="log0_linear_derivatives"`` and
``axes="derivative_order,wavenumber,broadening_pair,elower"``.
``multi_index_uniqgrid`` has ``index_base=0``. The partition-function group
has ``interpolation="linear_Q_in_T"``.

The ``/premodit`` attributes include ``R``, ``diffmode``, ``dE`` in cm\ :sup:`-1`,
``Tref``, ``Twt``, ``Tref_broadening``, ``Tmin`` and ``Tmax`` in K,
``temperature_unit="K"``, ``pressure_unit="bar"``, and
``pressure_reference=1.0``. ``c2`` records :math:`hc/k_B` in cm K and
``doppler_constant`` records :math:`\sqrt{k_B/m_u}/c` in
:math:`\sqrt{\mathrm{u}/\mathrm{K}}` for use with the saved mass.
The three reference temperatures have different meanings and need not be
equal. ``R`` is :math:`1/\Delta\ln\nu`, not an independently chosen
instrument resolution.

``load_lbd`` returns these numerical datasets as arrays with their dataset
names, except that the two partition-function arrays are named
``pf_temperature`` and ``pf_Q``. Returned metadata includes the four identity
strings, ``mol_mass``, the ``premodit`` and ``partition_function`` attribute
dictionaries, and the decoded ``provenance`` dictionary. The file is closed
before returning. Arrays are loaded into host memory once.

The reader validates required units, dimensions, index bounds and numerical
values. It permits ``-inf`` only for zeroth-order LBD coefficients. It does
not reconstruct an ``OpaPremodit`` object. The alternating FFT ``pmarray``
is omitted because it can be regenerated as ``(-1)**np.arange(Nnu + 1)``.

Coefficient and pressure conventions
------------------------------------

Only the zeroth-order coefficient is a natural logarithm. Empty bins have
``C0=-inf``. First and second derivatives are signed linear values; do not
exponentiate them or clip negative values. For
:math:`\delta=1/T-1/T_{wt}`, the line-shape density is

.. math::

   S_{ib}(T) =
   \frac{1-e^{-c_2\nu_i/T}}{1-e^{-c_2\nu_i/T_{ref}}}
   \frac{Q(T_{ref})}{Q(T)}
   \sum_e e^{-c_2 E_e(1/T-1/T_{ref})}
   \left[e^{C_{0ibe}}+C_{1ibe}\delta+
   \frac{1}{2}C_{2ibe}\delta^2\right].

Omit unavailable derivative terms for ``D=0`` or ``D=1``. ``C0`` is the log
of the numerical zeroth-order bin strength in cm molecule\ :sup:`-1`;
``C1`` and ``C2`` have units cm molecule\ :sup:`-1` K and
cm molecule\ :sup:`-1` K\ :sup:`2`. This mixed encoding cannot be described
by one cross-section unit attribute. Interpolate ``Q`` linearly in ``T``;
the temperature domain and ``Tref`` must lie inside the supplied Q table.

With ``(j_b, k_b) = multi_index_uniqgrid[b]``, the normalized Lorentz HWHM is

.. math::

   \widetilde\gamma_b(T,P) =
   \widetilde\gamma_{ref,j_b}
   (T/T_{ref,b})^{-n_{k_b}}\,P/(1\,\mathrm{bar}).

The generating line widths were normalized as
``ngamma_ref = gamma_ref * R / nu_line``. Input pressure is the pressure
of the declared broadener in bar; it equals total pressure for a pure
background gas. The normalized Doppler standard deviation is
``doppler_constant * sqrt(T / mol_mass) * R``. After Voigt convolution of the
LSD, multiply by ``R / nu_grid`` to obtain cm\ :sup:`2` molecule\ :sup:`-1`.
Use the recorded kernel and boundary prescription for reference comparisons.

The strengths describe the named isotopologue without an isotopic-abundance
factor. Combining H2 and He opacities by a weighted sum generally does not
reproduce a profile broadened by their mixture. Variable-composition
broadening requires a separate numerical contract.

Consumer example and distribution
---------------------------------

``examples/read_premodit_lbd.py`` is a standalone consumer requiring only
NumPy and h5py. Copy it alongside the data to inspect a file or obtain the
inputs to a consumer's Voigt convolution:

.. code-block:: python

    from read_premodit_lbd import load_lbd, evaluation_inputs

    arrays, metadata = load_lbd("12C-16O__Li2015.H2.premodit.h5")
    lsd, ngamma, nsigma = evaluation_inputs(
        arrays, metadata, temperature=1000.0, pressure_bar=1.0
    )

The results are LSD ``(Nnu, Nb)``, normalized Lorentz widths ``(Nb,)`` and a
Doppler standard deviation scalar. The example does not implement Voigt
convolution, resampling, atmospheric abundance scaling or a TauREx plugin.
These files are not ExoMolOP ``xsecarr`` tables; keep them outside directories
searched by the existing TauREx HDF5 cross-section reader.

PreMODIT excludes line centers outside its generating grid. Generate a guard
region and verify the usable interior by widening both the grid and input
line selection until the interior spectrum converges. Reading a wavenumber
slice from HDF5 does not establish that it can be evaluated independently.
The temperature design range also does not guarantee line-list completeness
or cross-section accuracy at every pressure.

Distribute the HDF5 file, this schema description, the reader and small
reference spectra with their temperature, pressure and kernel settings.
A release manifest can list immutable URLs, byte sizes, SHA-256 digests,
identity, broadener, grid and validation ranges. Scientific inputs remain
inside each HDF5 file. No separate metadata file is required to read an LBD.

``examples/export_premodit_lbd.py`` prepares an HDF5 file, a small reference
NPZ file and a manifest from an existing local ExoMol line list:

.. code-block:: console

    python examples/export_premodit_lbd.py /data/CO/12C-16O/Li2015 ./co-lbd \
        --nu-range 4200 4300 --valid-range 4220 4280 \
        --temperature-range 800 1200 --doi 10.1088/0067-0049/216/1/15 \
        --release pilot-1

This example requires the raw files locally and does not download or publish
data. The NPZ contains ``nu_grid``, ``temperature_K``, ``pressure_bar``,
``cross_section_cm2_per_molecule`` and ``profile_kernel``. Its spectra are
evaluated on the full grid for the listed temperature-pressure pairs. They
check agreement between implementations, not approximation error against a
direct line-by-line calculation. The example records
``validation.status="reference_only"`` and
``guard_convergence_checked=False`` until that separate comparison is done.

ExoAtom atomic lines
===================

``AdbExoAtom`` reads one neutral atom, ion, or isotope from the
`ExoAtom database <https://www.exomol.com/exoatom/>`_ using PyExoCross.
It supplies the atomic line interface used by ``OpaDirect`` without inheriting
the molecular database class. The raw states/transitions reader is shared
with the PyExoCross ExoMol backend; mass, partition functions, line strengths,
and broadening are handled by the atomic adapter.

Install the optional dependency in a separate Python 3.9--3.12 environment:

.. code:: sh

    pip install 'exojax[pyexocross]'

For a source checkout, use ``pip install -e '.[pyexocross]'``. This selects
PyExoCross 1.1.16 and compatible dependencies, including NumPy below 2.

Loading and direct opacity
--------------------------

Dataset paths follow the official directory layout: ``Li/NIST``,
``Li/Kurucz``, ``Li_p/NIST`` for Li II, or ``H/1H/NIST`` for an isotope.
``local_databases`` supplies a root directory for relative paths. Missing
``.adef.json``, ``.states``, ``.trans``, and ``.pf`` files are downloaded;
``download=False`` requires existing local files. Bzip2-compressed states
and transitions are also accepted. No parsed-file cache is created.

This example uses the radiative lifetimes supplied by the Kurucz dataset:

.. code:: python

    import jax
    import jax.numpy as jnp
    import numpy as np
    from exojax.database import AdbExoAtom
    from exojax.opacity import OpaDirect

    jax.config.update("jax_enable_x64", True)
    nu_grid = np.linspace(14900.0, 14910.0, 2001)
    adb = AdbExoAtom("Li/Kurucz", nu_grid, local_databases=".database/exoatom")
    opa = OpaDirect(adb, nu_grid)
    xs = jax.jit(opa.xsvector)(3000.0, 0.0)
    xsm = jax.jit(opa.xsmatrix)(
        jnp.array([3000.0, 5000.0]), jnp.zeros(2)
    )

Broadening
----------

When both level lifetimes are available and positive (infinity denotes a
stable level), the default Lorentzian half width is
``gamma_natural = (1/tau_upper + 1/tau_lower) / (4*pi*c)`` in cm-1.
The default profile includes Doppler and natural broadening only and is
independent of pressure. A single transition's Einstein A is not substituted
for the complete level decay rate.

NIST-derived ExoAtom files generally lack lifetimes. Missing or invalid
lifetimes leave the corresponding natural width undefined, and
``OpaDirect`` requires an explicit ``atomic_broadening(T, P)`` callback.
The callback also allows pressure broadening to be specified for Kurucz data.
It returns the **total Lorentzian HWHM in cm-1**, with shape ``(Nline,)``,
for temperature in K and pressure in bar. Values must be finite and
nonnegative; no additional width is added automatically.

For example, an explicitly chosen Doppler-only model for NIST data is:

.. code:: python

    adb = AdbExoAtom("Li/NIST", nu_grid, local_databases=".database/exoatom")
    opa = OpaDirect(
        adb, nu_grid,
        atomic_broadening=lambda T, P: jnp.zeros_like(adb.A),
    )

See the `ExoAtom data description <https://arxiv.org/abs/2512.24612>`_
for source data conventions and the
`NIST lifetime reference <https://physics.nist.gov/Pubs/AtSpec/node18.html>`_
for level decay rates.

Selection and numerical conventions
-----------------------------------

``margin`` expands the wavenumber interval; ``crit`` selects strengths above
a cutoff at ``Tref`` (296 K by default), and ``elower_max`` limits lower-state
energies. Further selections use ``adb.masking(boolean_mask)`` or
``adb.apply_mask_mdb(boolean_mask)`` before constructing the opacity calculator.
With ``gpu_transfer=False``, call ``adb.generate_jnp_arrays()`` first.

Supplied transition wavenumbers, fractional J values, state degeneracies,
source masses, and the dataset's partition table are preserved. An explicit
isotope mass takes precedence over the species mass. ExoAtom statistical
weights receive no extra nuclear-spin or isotopic-abundance factor. Partition
functions use linear interpolation; temperatures outside the source table
return NaN instead of extrapolating. ``Tref`` must lie within that table.

Reference strengths are stored directly in logarithmic form, using the same
constants as ExoJAX's temperature scaling. Thus high-excitation lines remain
usable at high temperature even if ``Sij0`` underflows at 296 K. ``crit=0``
keeps these lines; a positive cutoff is evaluated in log space.

This adapter supports ``OpaDirect`` with the Voigt profile, one species and
one transition file per dataset. Atomic abundances and ionization fractions
remain inputs to the optical-depth calculation. Large-list memory and
performance have not been benchmarked; states are held in memory while
transitions are read in chunks.

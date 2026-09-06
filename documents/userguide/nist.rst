NIST atomic lines
=================

``AdbNist`` retrieves one atomic species through RADIS and supplies line
strengths, masses, and partition-function ratios to ``OpaDirect``. Species
names use spectroscopic notation, such as ``Fe_II`` or ``Fe II``. The
partition function must be available in the bundled Barklem & Collet (2016)
table. ``Irwin=True`` selects Irwin (1981) for Fe I only.

NIST does not supply a complete set of damping parameters. An explicit
``atomic_broadening(T, P)`` callback is therefore required. It returns the
**total Lorentzian HWHM in cm-1** for each selected line, with shape
``(Nline,)``. Temperature is in K and pressure in bar. Values must be finite
and nonnegative. Use JAX operations inside the callback to retain temperature
and pressure derivatives. No other Lorentzian width is added automatically.

The example below uses a specified constant width to demonstrate the API;
it is not a pressure-broadening prescription for O I.

.. code:: python

    import jax
    import jax.numpy as jnp
    import numpy as np
    from exojax.database import AdbNist
    from exojax.opacity import OpaDirect

    jax.config.update("jax_enable_x64", True)
    nu_grid = np.linspace(12850.0, 12870.0, 2001)
    adb = AdbNist("O_I", nu_grid, local_databases=".database/nist")

    def broadening(T, P):
        return jnp.full_like(adb.A, 0.01)

    opa = OpaDirect(adb, nu_grid, atomic_broadening=broadening)
    xs = opa.xsvector(5000.0, 0.1)
    xsm = jax.jit(opa.xsmatrix)(
        jnp.array([5000.0, 6000.0]), jnp.array([0.1, 0.01])
    )

Select lines with ``adb.masking(boolean_mask)`` before constructing the
opacity calculator and its callback. Host and existing JAX arrays stay
aligned. With ``gpu_transfer=False``, call ``adb.generate_jnp_arrays()``
before creating ``OpaDirect``. The adapter discards lines with missing or
invalid required parameters and applies the reference-strength cutoff.

RADIS handles download and local caching. ``engine`` defaults to
``"pytables"`` and ``cache`` is passed to RADIS unchanged. The registry name
defaults to ``ExoJAX-NIST-{molecule}``; supply a different ``databank_name``
when storing the same species in another directory. The initial fetch
downloads the species line list before applying the wavenumber-range filter.
The selected
wavenumbers are vacuum values calculated by RADIS from the level-energy
differences. No damping constants or ionization energies are synthesized.

The Einstein coefficient of a single transition does not generally determine
its natural width. In the usual radiative model,
``gamma_rad = (Gamma_upper + Gamma_lower) / (4*pi*c)``, where each level's
decay rate is the sum of its downward transition probabilities. Using
``A_ul / (4*pi*c)`` assumes a stable lower level and negligible competing
upper-level branches. NIST lines in a selected wavelength range need not
contain every branch; an automatic sum over those lines is insufficient.
See the `NIST atomic-lifetime reference
<https://physics.nist.gov/Pubs/AtSpec/node18.html>`_.

The same callback can override total Lorentzian widths for VALD or Kurucz.
Their existing broadening calculations remain the default when no callback
is provided. ``OpaModit`` and ``OpaPremodit`` do not accept ``AdbNist``.
Atomic abundances and ionization fractions remain inputs to the optical-depth
calculation, separate from the cross section.

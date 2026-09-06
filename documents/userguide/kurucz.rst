Kurucz atomic lines
==================

Load an existing Kurucz line list with ``AdbKurucz(path, nurange=nu_grid)``.
To let RADIS download and cache one atomic species, use the explicit
``from_radis`` constructor:

.. code-block:: python

    import numpy as np
    from exojax.database import AdbKurucz
    from exojax.opacity import OpaDirect

    nu_grid = np.linspace(19950.0, 20050.0, 2000)  # vacuum cm-1
    adb = AdbKurucz.from_radis(
        "Fe_I", nu_grid,
        local_databases=".database/radis-kurucz",
        margin=10.0,
        gpu_transfer=False,
    )
    # Select lines on the host before generating JAX line arrays if needed.
    adb.generate_jnp_arrays()
    opa = OpaDirect(adb, nu_grid)
    cross_section = opa.xsvector(3000.0, 0.1)  # K, bar; output in cm2
    print(adb.provenance)

``from_radis`` requires a finite positive wavenumber interval or grid. Its
``margin`` expands the fetch interval as well as the ExoJAX selection interval.
Supported species must have ExoJAX partition functions and atomic metadata;
examples are ``Fe_I`` and ``Fe_II``. Unsupported species are rejected before
downloading. Missing or nonfinite line parameters are reported as errors.

RADIS manages downloads, cache registration, and cache policies. ``cache`` is
passed to RADIS, and the default ``True`` reuses registered cache files when
available. The initial download covers the species line list; the requested
wavenumber interval filters the cached data loaded into ExoJAX.
``databank_name`` defaults to ``"ExoJAX-Kurucz-{molecule}"``; choose
a different name when maintaining a separate cache for the same species.
``engine="pytables"`` is the default, and ``"vaex"`` requires that optional
backend. ``adb.provenance`` records the species, RADIS version, and cache paths;
download URLs and dates remain in RADIS cache metadata.

Line positions and Einstein A coefficients come from RADIS. Its air/vacuum
conversion can produce small differences from the existing local-file reader.
ExoJAX continues to compute the reference line strengths, partition functions,
and broadening. ``Irwin=True`` selects the Irwin polynomial for Fe I consistently
at the reference and evaluation temperatures; other species retain the
Barklem and Collet partition functions. ``gpu_transfer=False`` delays line-array
transfer; partition-function calculations still use JAX.

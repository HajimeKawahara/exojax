DiffGrid
========

`Last update: August 25th (2026)`

**DiffGrid** is a pressure-layer-aligned opacity table for repeated,
differentiable spectral calculations.  It is useful when the atmospheric
pressure grid is fixed but the temperature profile changes many times, as in
optimization or HMC-NUTS retrievals.

An :class:`exojax.opacity.diffgrid.api.OpaDiffgrid` object is built from a
differentiable opacity calculator, called the *teacher*.
:class:`exojax.opacity.premodit.api.OpaPremodit` is the standard teacher.  At
each temperature node, DiffGrid stores

- the logarithm of the cross section, and
- its derivative with respect to inverse temperature, :math:`1/T`.

The cross section between nodes is evaluated with cubic Hermite
interpolation.  Pressure is not interpolated: every table row is tied to one
atmospheric layer.

Basic construction
------------------

The following example builds a DiffGrid table for CO.  ``ArtEmisPure`` supplies
the pressure layers that will also be used by the forward model.

.. code-block:: python

    from jax import config
    config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import numpy as np

    from exojax.database.exomol.api import MdbExomol
    from exojax.opacity import OpaDiffgrid, OpaPremodit
    from exojax.rt import ArtEmisPure
    from exojax.utils.grids import wavenumber_grid

    nu_grid, wav, resolution = wavenumber_grid(
        22920.0,
        23000.0,
        3500,
        unit="AA",
        xsmode="diffgrid",
    )
    art = ArtEmisPure(
        nu_grid=nu_grid,
        pressure_top=0.1,
        pressure_btm=10.0,
        nlayer=16,
        rtsolver="fbased2st",
        nstream=2,
    )

    temperature_min = 500.0
    temperature_max = 1500.0
    art.change_temperature_range(temperature_min, temperature_max)

    mdb = MdbExomol(
        ".database/CO/12C-16O/Li2015",
        nurange=nu_grid,
    )
    teacher = OpaPremodit(
        mdb=mdb,
        nu_grid=nu_grid,
        auto_trange=(temperature_min, temperature_max),
        broadening_resolution={"mode": "single", "value": None},
    )

Inverse-temperature nodes
-------------------------

DiffGrid interpolates in :math:`1/T`, so nodes should normally be evenly
spaced in inverse temperature rather than in temperature.  The endpoints must
cover every layer temperature allowed by the model or prior.

.. code-block:: python

    inverse_temperature_grid = np.linspace(
        1.0 / temperature_max,
        1.0 / temperature_min,
        17,
    )
    temperature_grid = 1.0 / inverse_temperature_grid

    opa = OpaDiffgrid(
        teacher,
        temperature_grid=temperature_grid,
        pressure_grid=np.asarray(art.pressure),
    )

The input temperature nodes may be given in either order; ``OpaDiffgrid``
stores them in increasing :math:`1/T` order.  They must be finite, positive,
and unique, and at least two nodes are required.  More nodes improve accuracy
but increase both construction time and memory use.

Fixed pressure layers
---------------------

The pressure array is part of the table definition.  Rebuild the table after
changing the number of layers, pressure bounds, or layer placement.  Validate
the pressure grid once, outside a JIT-compiled model:

.. code-block:: python

    opa.check_pressure_grid(np.asarray(art.pressure))

This check raises an error if the shape differs or values disagree beyond the
validation tolerance.  A traced pressure argument is intentionally rejected;
pressure should not be a sampled parameter of a compiled DiffGrid opacity
call.

Computing a cross-section matrix
--------------------------------

``xsmatrix`` accepts one temperature per stored pressure layer and returns an
array with shape ``(nlayer, nnu)`` in cm2.  Omit pressure inside JIT or a
NumPyro model because it is already bound to the table.

.. code-block:: python

    temperature = art.powerlaw_temperature(1000.0, 0.08)
    cross_section = opa.xsmatrix(temperature)

    mmr_profile = art.constant_mmr_profile(1.0e-3)
    gravity = 1.0e5  # cm s-2
    dtau = art.opacity_profile_xs(
        cross_section,
        mmr_profile,
        opa.molmass,
        gravity,
    )
    flux = art.run(dtau, temperature)

Unlike other opacity calculators, DiffGrid does not provide a scalar
``xsvector(T, P)`` operation.  Its unit of evaluation is the complete,
layer-aligned matrix.

Checking interpolation accuracy
-------------------------------

The required node density depends on the molecule, spectral range,
temperature range, and data precision. Compare DiffGrid with its teacher away
from the stored nodes, where interpolation error can be measured. The helper
below returns the temperature at the midpoint of every interval in
:math:`1/T`; these are harmonic, not arithmetic, temperature means.

.. code-block:: python

    from exojax.opacity.diffgrid.diagnostics import (
        compare_diffgrid_with_teacher,
        diffgrid_interval_midpoint_temperatures,
    )

    validation_temperatures = diffgrid_interval_midpoint_temperatures(opa)
    for validation_temperature in validation_temperatures:
        validation_profile = np.full(
            art.pressure.shape,
            validation_temperature,
        )
        summary = compare_diffgrid_with_teacher(
            opa,
            teacher,
            validation_profile,
            quantiles=(0.99,),
        )
        print(
            validation_temperature,
            summary.absolute_log_cross_section_error_quantiles[0],
            summary.maximum_absolute_log_cross_section_error,
        )

The reported error is the absolute natural-log cross-section ratio after the
cross-section floor stored by ``opa`` has been applied to both calculators.
For example, an error of 0.05 corresponds to a multiplicative ratio of about
1.05. Quantiles combine every pressure-layer and wavenumber entry. The
diagnostic reports numbers only; the application chooses its thresholds and
decides whether to reject a table.

``compare_diffgrid_with_teacher`` is a host-side archive-build diagnostic, not
an operation for a JIT-compiled retrieval. It processes one temperature
profile at a time, uses a reusable JAX kernel for the pointwise error, and
transfers that one error matrix to the host for exact NumPy quantiles. Looping
over profiles as above therefore avoids constructing a large
``(nprofile, nlayer, nnu)`` batch.

A stronger application-level test propagates both matrices through the full
forward model and compares the spectral difference with the observational
noise.  Increase the number of inverse-temperature nodes until that difference
is negligible for the intended analysis.

Memory use and applicability
----------------------------

The two main arrays have shape ``(nlayer, ntemperature, nnu)``.  Their payload
can be inspected directly:

.. code-block:: python

    payload_bytes = (
        np.asarray(opa.log_cross_section_grid).nbytes
        + np.asarray(opa.log_cross_section_derivative_grid).nbytes
    )
    print("DiffGrid payload (MiB):", payload_bytes / 2.0**20)

After construction, the DiffGrid object is self-contained; the teacher and
line database do not participate in ``xsmatrix`` calls and may be released if
they are no longer needed for validation.  See :doc:`save_diffgrid` to persist
the table and load it later without its teacher or molecular database.

DiffGrid is most effective when the same pressure grid is evaluated many
times.  It is not appropriate when pressure itself varies, when temperatures
can leave the tabulated interval, or when the table is larger than evaluating
the teacher directly.  Concrete out-of-range temperatures raise an error.
Inside compiled code, out-of-range layers produce ``NaN`` cross sections so
that invalid parameter proposals do not silently extrapolate.

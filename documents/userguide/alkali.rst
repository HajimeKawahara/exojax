Na and K opacity
================

Load neutral Na or K lines with ``AdbKurucz``, then compute cross sections
with ``OpaDirect(..., line_profile="alkali_subvoigt")``. Each database must
contain one neutral species; ``AdbVald`` is also supported.

Download the Kurucz lists
`gf1100.all (Na I) <http://kurucz.harvard.edu/linelists/gfall/gf1100.all>`_ and
`gf1900.all (K I) <http://kurucz.harvard.edu/linelists/gfall/gf1900.all>`_
into the working directory. This example loads each species and computes
cross sections at one temperature and pressure, then for two atmospheric layers:

.. literalinclude:: ../../examples/alkali_opacity.py
    :language: python
    :start-at: import jax

Run ``python examples/alkali_opacity.py`` from the repository root with both
line-list files there. ``xs`` has shape ``(512,)`` and ``xs_layers`` has shape
``(2, 512)``. Both contain cross sections in cm2 per atom; temperatures are in K,
total pressures in bar, and vacuum wavenumbers in cm-1.

``margin=9000.0`` includes lines whose wings reach the grid from outside it.
``vmr_fraction`` gives the H, He, H2 broadener fractions; apply the Na/K
abundances separately when converting cross sections to atmospheric opacity.
The coarse grid illustrates the API. Use a finer grid to resolve line cores
and evaluate large grids in chunks to limit direct line-by-line memory use.

For automatic download and caching, use
``AdbKurucz.from_radis("Na_I", nu_grid, margin=9000.0)`` (or ``"K_I"``)
instead of the local-file constructor. Its default broadener fractions match
this example; see :doc:`kurucz` for cache options.

Omitting ``line_profile`` selects the existing Voigt profile.
``OpaAlkali(adb, nu_grid)`` is a convenience wrapper for the sub-Voigt option
with the same ``xsvector`` and ``xsmatrix`` methods. See
:ref:`alkali-line-profile` for the wing prescription and comparison with Voigt.

Allard (2019) Na--H2 resonance wings
------------------------------------

``OpaAlkaliTable`` provides the Na I D1/D2 resonance doublet using the
`Allard et al. (2019) <https://doi.org/10.1051/0004-6361/201935593>`_
density-expansion tables. It treats collisions with **H2 only**. Download
`opacity.tar.gz <https://cdsarc.cds.unistra.fr/ftp/J/A+A/628/A120/opacity.tar.gz>`_
from CDS and retain the accompanying
`README <https://cdsarc.cds.unistra.fr/ftp/J/A+A/628/A120/README.pdf>`_.
The original archive can be read directly, including its nested archives;
an extracted ``ALLARD_NaH2`` directory is also accepted. No network access
occurs when constructing or evaluating the calculator.

.. code-block:: python

    import jax
    import numpy as np
    from exojax.opacity import OpaAlkaliTable

    jax.config.update("jax_enable_x64", True)
    nu_grid = np.linspace(12000.0, 22000.0, 10001)
    opa = OpaAlkaliTable(
        nu_grid, "opacity.tar.gz", model="allard2019_na_h2",
        vmr_perturber=0.85, core_transition=(20.0, 30.0),
    )
    xs = jax.jit(opa.xsvector)(1000.0, 1.0)
    xs_layers = jax.jit(opa.xsmatrix)(
        np.array([1000.0, 1500.0]), np.array([1.0, 10.0]),
    )

The inputs are temperature in K, total pressure in bar, and increasing
vacuum wavenumber in cm-1. The perturber density is
``vmr_perturber * P * 1e6 / (kB * T)`` in cm-3. Setting the fraction to
0.85 includes only H2 collisions at that partial density; it does not
add He broadening. Do not sum two complete single-perturber doublets to
approximate H2/He broadening.

The output is cm2 per **ground-state neutral Na atom**, including both
resonance components. Oscillator strengths are already included. Apply
abundance and, if required, lower-state population and stimulated-emission
corrections separately. An atomic line list must exclude these two lines
when other transitions are added. This differs from ``OpaAlkali``, which
applies its prescription to every selected line.

Numerical prescription and limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CDS files provide signed density-expansion coefficients, not a
temperature/pressure opacity grid. ExoJAX follows the powers starting at
``q**1`` in ``lect_sig.f`` and the published numerical examples. The
distributed Python reader starts at ``q**0``; it is not used. The Na D1
far red wing is a separate table and uses the supplied linear density
scaling. It is joined to the near-wing grid, with the near-wing value
used at the shared endpoint.

The reference programs output separate Lorentz cores and wings without
a unique joining prescription. ExoJAX uses a **numerical hybrid**:

* Inside 20 cm-1 of each unperturbed line center, a Voigt core uses the
  table's impact width and shift, thermal Doppler width, and natural width
  derived from the table's oscillator strength and D1/D2 statistical weights.
* Between 20 and 30 cm-1, a cubic smoothstep blends the core with the wing.
  ``core_transition`` changes these two detunings.
* Outside 30 cm-1, the tabulated collision wing is used. Doppler and natural
  convolution of the whole wing is not performed.

This construction does not reproduce an exact unified line core or its
full asymmetry. The transition interval is an exposed numerical choice;
check sensitivity to it when fitting near-core observations. No numerical
renormalization is applied to the assembled profile.

Negative results of the truncated density expansion are set to zero
before linear interpolation in wavenumber; the Fortran program instead
omits nonpositive points. The wing is zero outside each table's finite
spectral range, rather than extrapolated. Temperature interpolation is
linear in the assembled cross sections, evaluating each bounding table
at the requested density and Doppler temperature. It is piecewise
differentiable; derivatives may jump at interpolation knots or clipping
boundaries.

The full archive contains 500, 600, 725, 1000, 1500, 2000, 2500 and 3000 K.
The supported Na--H2 density range is 0 to 1e21 cm-3. Out-of-range T or
density, negative pressure and nonfinite inputs return NaNs, also under
``jax.jit``. No silent temperature or density extrapolation is performed.
64-bit JAX is required. With 32-bit arithmetic, reverse-mode derivatives
through the large perturber density can underflow even when the cross
sections look reasonable. Enable it before constructing the calculator.

Allard (2024) K--He resonance wings
-----------------------------------

Select ``model="allard2024_k_he"`` to use the K I D1/D2 doublet of
`Allard et al. (2024) <https://doi.org/10.1051/0004-6361/202348711>`_.
This model includes **He collisions only**; it does not provide K--H2
broadening. Download the ``D1`` and ``D2`` directories from
`CDS J/A+A/683/A188 <https://cdsarc.cds.unistra.fr/ftp/J/A+A/683/A188/>`_,
and use their parent directory as ``data_path``. A local tar archive of
the distribution is also accepted. All 14 tables are required, covering
500, 800, 1000, 1500, 2000, 2500 and 3000 K for each component.

The loader requires ``tableD2_KHe_800_1e21_2025.omg``, which implements
the `2025 correction <https://doi.org/10.1051/0004-6361/202554036e>`_.
It ignores the superseded 800 K D2 table if both versions are present.
The archive's ``README.pdf`` and ``lect_sig.f`` describe its format and
normalization; retain them alongside the downloaded tables.

.. code-block:: python

    opa_k = OpaAlkaliTable(
        np.linspace(10000.0, 16000.0, 10001),
        "A188", model="allard2024_k_he", vmr_perturber=0.15,
    )
    xs_k = jax.jit(opa_k.xsvector)(1000.0, 1.0)

Here ``xs_k`` is in cm2 per ground-state neutral K atom, and the He
density is 0.15 times the total gas number density. The numerical core/wing
join, temperature interpolation, clipping and finite spectral support
follow the prescription above. K D1 does not use a separate red-wing table.
Its spectral support is particularly asymmetric and varies with temperature.

For K--He, ExoJAX adopts 1e21 cm-3, the supplied reference density, as
the maximum accepted He density. This is an adopted software boundary,
not a published guarantee of the truncated expansion's accuracy at every
wavenumber. Values beyond this boundary return NaNs. Validate the profile
and the chosen core transition for the atmospheric conditions of interest.

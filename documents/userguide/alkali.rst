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

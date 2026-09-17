.. _atmospheric-layer-contract:

Atmospheric Layer Contract
==========================

An ExoJAX pressure grid contains ``nlayer`` atmospheric layers.  All pressure
arrays are ordered from the low-pressure top to the high-pressure bottom.  The
physical extent of the grid is defined by ``nlayer + 1`` boundaries::

    pressure_boundary[0]       top of the atmosphere
             layer 0
    pressure_boundary[1]
                ...
    pressure_boundary[nlayer]  bottom of the atmosphere

``pressure_boundary`` is the source of truth for atmospheric interfaces and
for joining the atmosphere to another model.  The related arrays obey the
following contract:

* ``pressure_boundary`` has ``nlayer + 1`` values and contains the physical
  upper and lower boundary of every layer.  ``pressure_top_boundary`` and
  ``pressure_btm_boundary`` are aliases for its first and last values.
* ``pressure`` has ``nlayer`` representative values.  Temperature, abundance,
  opacity, and other layer profiles are evaluated at these pressures.
* ``dParr`` has ``nlayer`` values and is defined by
  ``dParr = pressure_boundary[1:] - pressure_boundary[:-1]``.

Treat these grid arrays as immutable model configuration.  Create a new Art
object instead of modifying ``pressure_boundary``, ``pressure``, or ``dParr``
in place, so that all derived quantities remain consistent.  Code should not
depend on their concrete NumPy/JAX container type; use ``jax.numpy.asarray``
when a device array is required.

For a logarithmically spaced grid with reference point :math:`r`, the
representative pressure in layer :math:`i` is

.. math::

   \log P_i = (1-r)\log P_{i,\mathrm{upper}}
              + r\log P_{i,\mathrm{lower}}.

ExoJAX uses :math:`r=0.5`, so ``pressure[i]`` is the geometric center of its
two boundaries.  A surface is a boundary condition below the last atmospheric
layer; it is not an additional layer with another representative pressure.

Creating a grid from exact boundaries
-------------------------------------

Use ``from_pressure_boundaries`` when either endpoint must be an exact physical
interface.  The class method is inherited from ``ArtCommon`` and is available
on the Art and Opart classes.  For example, a transmission atmosphere whose
bottom is exactly 100 bar can be constructed as follows:

.. code-block:: python

    from exojax.rt import ArtTransPure

    art = ArtTransPure.from_pressure_boundaries(
        pressure_top_boundary=1.0e-8,
        pressure_btm_boundary=1.0e2,
        nlayer=100,
        integration="simpson",
    )

Here ``art.pressure_boundary[0]`` and ``art.pressure_boundary[-1]`` are the
specified pressures.  ExoJAX derives the representative ``art.pressure`` and
``art.dParr`` arrays from those boundaries.  For compatibility with the
existing Art API, ``art.pressure_top`` and ``art.pressure_btm`` remain the
first and last representative pressures.

The factory validates that every boundary, representative pressure,
``dParr``, and pressure-decrease rate is representable in the active JAX
precision.  Enable JAX 64-bit mode before constructing the Art object when an
extreme pressure range requires it.

The atmosphere--surface connection is always at
``pressure_boundary[-1]``:

* In transmission, ``radius_btm`` and ``gravity_btm`` are defined at this
  boundary.  The region below ``radius_btm`` is the opaque disk.
* When an emission or reflection solver exposes a bottom/surface source or
  reflectivity, that boundary condition is applied immediately below the last
  atmospheric layer at this boundary.
* When coupling a lower model, set its top pressure equal to
  ``pressure_boundary[-1]`` and pass its upward radiation as the bottom/surface
  source where the selected solver supports one.

Solvers without a bottom/surface argument retain their solver-specific lower
boundary condition; the pressure-grid contract does not add such an argument.

``ArtAbsPure`` can integrate to a ``pressure_surface`` inside an existing
layer.  This partial-layer operation is useful for absorption-only paths, but
it is separate from the canonical atmosphere--surface connection above.  For
a shared physical interface, construct the grid with that pressure as its
bottom boundary.

Legacy representative-endpoint constructors
-------------------------------------------

Existing constructors retain their representative-endpoint interpretation.  In

.. code-block:: python

    art = ArtTransPure(
        pressure_top=1.0e-8,
        pressure_btm=1.0e2,
        nlayer=100,
    )

``pressure_top`` and ``pressure_btm`` are the first and last representative
layer pressures, not the physical grid boundaries.  If their logarithmic
spacing is :math:`\Delta` and the reference point is :math:`r`, the derived
outer boundaries are

.. math::

   \begin{aligned}
   \log P_{\mathrm{boundary},0} &= \log P_{\mathrm{top}} - r\Delta, \\
   \log P_{\mathrm{boundary},N} &= \log P_{\mathrm{btm}} + (1-r)\Delta.
   \end{aligned}

They therefore extend by half a logarithmic grid cell at each end for the
default :math:`r=0.5`.  Use ``pressure_boundary[-1]``, rather than
``pressure_btm``, when connecting a legacy grid to a surface or lower layer.

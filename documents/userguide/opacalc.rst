Opacity Calculator Classes (``opa``)
======================================

Several opacity calculation methods are available in ExoJAX. 
Opacity calculations can be controlled using the opacity calculation class 
(``opa``) for each method.


+--------------------------+-------------+-----------+------------------------------------+
|**calculator**            |**opa**      |**nu_grid**| **features**                       |
+--------------------------+-------------+-----------+------------------------------------+
|:doc:`premodit` (default) |OpaPremodit  |ESLOG      | memory saving, need to set T range |
+--------------------------+-------------+-----------+------------------------------------+
|:doc:`diffgrid`           |OpaDiffgrid  |ESLOG      | fixed pressure, repeated T profiles|
+--------------------------+-------------+-----------+------------------------------------+
|CKD                       |OpaCKD       |arbitrary  |Correlated k-Distribution           |
+--------------------------+-------------+-----------+------------------------------------+
|Direct LPF                |OpaDirect    |arbitrary  | line by line, arbitrary T          |
+--------------------------+-------------+-----------+------------------------------------+
|:doc:`modit`              |OpaMODIT     |ESLOG      | arbitrary T                        |
+--------------------------+-------------+-----------+------------------------------------+
|DIT                       |N/A          |ESLIN      | linear grid                        |
+--------------------------+-------------+-----------+------------------------------------+

``OpaDirect`` uses ``line_profile="voigt"`` by default. For neutral Na or K
in a VALD/Kurucz/ExoAtom database, ``line_profile="alkali_subvoigt"`` selects the
:doc:`Na/K sub-Voigt calculation <alkali>`. ``OpaAlkali`` is a convenience
wrapper for this option. Both use the direct LPF method.

Hydrogen profiles and prescribed NLTE populations
------------------------------------------------

For ``AdbHydrogen``, the experimental ``opa.line_profiles(T, velocity=0.0)``
method returns individual natural-plus-thermal-Doppler Voigt profiles with
shape ``(Nline, Nnu)``. Each profile integrates to unity over wavenumber,
so its units are cm. A finite spectral window may omit part of the wings;
profiles are not renormalized to that window. LTE strengths and level
populations are not included. Existing ``xsvector`` and ``xsmatrix`` calls
continue to return LTE cross sections.

``velocity`` is the line-of-sight velocity in km/s, positive for recession.
It shifts the center to ``nu_line * (1 - velocity / c)`` while retaining
the rest-frame widths. This approximation assumes velocities much smaller
than the speed of light and includes no relativistic intensity transformation.
Use ``jax.vmap`` to evaluate different temperatures and velocities per layer.

``exojax.opacity.lpf.atomic.line_coefficients`` combines these profiles with
prescribed lower- and upper-level number densities in cm\ :sup:`-3`.
Profiles have shape ``(..., Nline, Nnu)`` and populations ``(..., Nline)``.
The function sums over lines and returns extinction in cm\ :sup:`-1` and
emissivity in erg s\ :sup:`-1` cm\ :sup:`-3` sr\ :sup:`-1` per cm\ :sup:`-1`.
It includes stimulated emission and uses the same profile for absorption
and emission, with photon energies evaluated at the rest line centers.
``adb.n_lower`` and ``adb.n_upper`` are quantum numbers, not populations.

This two-layer Halpha example uses illustrative populations and orders layers
from the observer toward the lower boundary:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpy as np

    from exojax.database import AdbHydrogen
    from exojax.opacity import OpaDirect
    from exojax.opacity.lpf.atomic import line_coefficients
    from exojax.rt.layeropacity import layer_optical_depth_from_extinction
    from exojax.rt.planck import piBarr
    from exojax.rt.rtransfer import rtrun_emis_pureabs_ibased_intensity_surface

    jax.config.update("jax_enable_x64", True)
    adb = AdbHydrogen(n_upper_max=3)
    center = adb.nu_lines[0]
    nu_grid = np.linspace(center - 5.0, center + 5.0, 1001)
    opa = OpaDirect(adb, nu_grid)

    temperatures = jnp.array([6000.0, 9000.0])
    velocities = jnp.array([-5.0, 0.0])
    profiles = jax.vmap(opa.line_profiles)(temperatures, velocities)
    n2 = jnp.array([[2.0e3], [3.0e4]])
    n3 = n2 * jnp.array([[0.03], [0.12]])
    extinction, emissivity = line_coefficients(
        profiles, adb.nu_lines, adb.A, adb.glower, adb.gupper, n2, n3
    )
    dtau = layer_optical_depth_from_extinction(
        extinction, jnp.array([3.0e9, 2.0e9])
    )
    safe_extinction = jnp.where(extinction > 0.0, extinction, 1.0)
    source = jnp.where(
        extinction > 0.0, jnp.pi * emissivity / safe_extinction, 0.0
    )
    lower_boundary = piBarr(jnp.array([4500.0]), nu_grid)[0]
    pi_intensity = rtrun_emis_pureabs_ibased_intensity_surface(
        dtau, source, lower_boundary, jnp.array([1.0])
    )

Here the source and lower boundary both use the ``piBarr`` convention,
so the result is pi times the physical specific intensity. Passing
``emissivity / extinction`` and a physical boundary intensity instead gives
physical intensity directly. The example has positive extinction; it does
not handle population inversion or zero extinction with nonzero emissivity.

This prototype accepts populations supplied by the caller. It does not solve
NLTE statistical equilibrium, include Stark broadening, or provide a
KELT-9 b retrieval model.

Links to API
----------------

- `OpaPremodit <../exojax/exojax.spec.html#exojax.opacity.api.OpaPremodit>`_
- :class:`exojax.opacity.diffgrid.api.OpaDiffgrid`

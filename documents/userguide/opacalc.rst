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

Atomic profiles and prescribed NLTE populations
-----------------------------------------------

For VALD, Kurucz, NIST, and ExoAtom, the experimental
``opa.line_profiles(T, P=0.0, *, velocity=0.0)`` method returns individual
Voigt profiles with shape ``(Nline, Nnu)``. It uses the same thermal Doppler
and Lorentzian widths as ``xsvector``. NIST still requires an explicit
:doc:`broadening callback <nist>`; this method adds no broadening model.
It is available only with ``line_profile="voigt"``.

Each profile integrates to unity over wavenumber, so its units are cm.
A finite spectral window may omit part of the wings; profiles are not
renormalized to that window. LTE strengths and level populations are not
included. Existing ``xsvector`` and ``xsmatrix`` calls return LTE cross sections.

``T`` (K), ``P`` (bar), and ``velocity`` (km/s) are scalars. Positive velocity
means recession and shifts the center to ``nu_line * (1 - velocity / c)``
while retaining rest-frame widths. This approximation assumes velocities
much smaller than the speed of light and includes no relativistic intensity
transformation. Use ``jax.vmap`` to evaluate different conditions per layer.

``line_coefficients`` combines profiles with prescribed lower- and upper-level
number densities in cm\ :sup:`-3`. Profiles have shape
``(..., Nline, Nnu)`` and populations ``(..., Nline)``. The function sums over
lines and returns extinction in cm\ :sup:`-1` and emissivity in
erg s\ :sup:`-1` cm\ :sup:`-3` sr\ :sup:`-1` per cm\ :sup:`-1`.
It includes stimulated emission and uses the same profile for absorption
and emission, with photon energies evaluated at rest line centers.

This Halpha example uses an illustrative constant Lorentzian width and
prescribed populations; neither is a hydrogen atmosphere model:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpy as np

    from exojax.database import AdbNist
    from exojax.opacity import OpaDirect
    from exojax.opacity.lpf.line_coefficients import line_coefficients

    jax.config.update("jax_enable_x64", True)
    nu_grid = np.linspace(15228.0, 15238.0, 1001)
    adb = AdbNist("H_I", nu_grid)

    def broadening(T, P):
        return jnp.full_like(adb.A, 0.001)

    opa = OpaDirect(adb, nu_grid, atomic_broadening=broadening)
    profiles = opa.line_profiles(8000.0, velocity=5.0)
    n_lower = jnp.full_like(adb.A, 1.0e4)
    n_upper = jnp.full_like(adb.A, 1.0e2)
    extinction, emissivity = line_coefficients(
        profiles, adb.nu_lines, adb.A, adb.glower, adb.gupper,
        n_lower, n_upper,
    )

These functions accept populations supplied by the caller; they do not solve
NLTE statistical equilibrium. Extinction includes stimulated emission and
can be zero or negative. Any subsequent transfer calculation must handle
the applicable population regime.

Links to API
----------------

- `OpaPremodit <../exojax/exojax.spec.html#exojax.opacity.api.OpaPremodit>`_
- :class:`exojax.opacity.diffgrid.api.OpaDiffgrid`

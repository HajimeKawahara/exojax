Radiative-convective equilibrium
================================

``exojax.atm.rce.solve_rce`` determines layer and lower-boundary temperatures
on a fixed pressure grid. The core provides the active-set Newton iteration;
the caller supplies the radiation callback and required ``neutral_gradient``.

``exojax.rt.flux`` provides interface-temperature reconstruction and
pure-absorption radiation routines. Thermodynamics, composition, and
boundary assumptions belong to the application, as illustrated by the
:doc:`../tutorials/rce_earth` forward example.

Run the gray example from the repository root:

.. code-block:: bash

   JAX_PLATFORMS=cpu MPLBACKEND=Agg python examples/rce_gray.py

It enables JAX x64 inside ``main`` and saves ``rce_gray.png``. No opacity
database download is needed.

State and radiation contract
----------------------------

All arrays run from top to bottom. Pressures are in bar, temperatures in K,
and bolometric fluxes in erg/s/cm2. Center pressures and temperatures have
length N; pressure boundaries and interface fluxes have length N+1. Each
center must lie inside its pressure layer. The additional unknown
``bottom_temperature`` is at the last pressure boundary, below the lowest
center.

The JAX-compatible callback ``radiative_flux(T, T_bottom)`` returns net upward
bolometric flux at every interface. It must update temperature-dependent
opacity and sources at every trial state. Downward stellar flux is subtracted
from upward minus downward thermal flux. ``direct_beam_fluxes`` takes incident
flux through a horizontal surface and an incidence cosine ``mu0``; the flux
already includes the projection factor. Its bottom value includes radiation
reaching the lower boundary.

For the provided pure-absorption transfer kernel, reconstruct boundary
temperatures with ``exojax.rt.flux.reconstruct_boundary_temperature``.
This interpolates log T in log P, takes the first center temperature at the top, and uses
``T_bottom`` at the bottom. Also pass the actual center Planck sources as
``source_center`` and
``upper_fraction=(P_center-P_upper)/(P_lower-P_upper)``. This splits each
layer at its center using the pressure column fraction, with opacity held
constant within the layer. The source is linear in optical depth on each
side of the center. Retaining the center source makes radiation respond
directly to each solved layer temperature and suppresses alternating errors
in center temperature.
Returned fluxes remain at the original N+1 pressure boundaries.

With this transfer kernel's default black boundary, the bottom gas and
surface sources share ``T_bottom``. Its value is solved from energy balance,
including downward radiation and convective transport; it is not the
internal effective temperature defined by ``internal_flux = sigma * T_int**4``.

Convective closure and convergence
----------------------------------

The solver uses logarithmic temperatures and damped Newton steps for a fixed
convective mask, then updates that mask. Connections join adjacent centers,
with the last connection joining the lowest center to the bottom boundary.
The required ``neutral_gradient`` is a positive scalar, an N-element array
on those connections, or a JAX-compatible callable ``(T, T_bottom) -> gradient``
returning either shape. A callable is reevaluated at every trial state;
its temperature derivatives enter the Newton Jacobian. The caller defines
the neutral gradient from the chosen thermodynamics and composition model.

At the top, radiative flux equals ``internal_flux`` and convection is zero.
Inactive connections require radiative energy balance and a temperature
gradient at or below the supplied neutral value. Active connections require
that gradient and nonnegative upward convective flux. Both activation and deactivation are
supported. The host iteration is not itself differentiable; the fixed-mask
``rce_residual`` is JAX-compatible.

Check ``result.converged`` before using a solution. ``result.status`` reports
iteration limits, mask cycling, singular Jacobians, domain or line-search
failures, and inconsistent physical residuals. The result includes center and
bottom temperatures, radiative and convective fluxes, the convective mask,
``flux_residual``, ``gradient_residual``, ``scaled_residual``, iteration counts,
and ``domain_valid``. Convective flux is zero on inactive connections;
``flux_residual`` is total upward flux minus internal flux.

Convergence checks the selected equations and convective stability inequalities,
including the returned physical state. Flux tolerance is
``flux_atol + flux_rtol * abs(internal_flux)``; it is not normalized by a much
larger stellar flux. ``gradient_atol`` controls neutral-gradient errors. Enable
JAX x64 at the call site for precision-sensitive calculations; a small
temperature step alone does not establish convergence.

The default mask is entirely radiative. Its intermediate radiative solution
may lie outside an opacity table even when the final RCE solution is inside.
Use ``convective_mask_initial`` for an informed initial guess or pass the
previous solution's mask and temperatures during parameter continuation.
A failed solve does not establish that no equilibrium exists. Compare
multiple initial states and refine the pressure and angular grids, checking
the temperature profile and convective boundary as well as energy balance.

Implicit sensitivities
----------------------

``exojax.atm.rce_implicit.make_implicit_rce_solver`` constructs a
``solve(params)`` function compatible with ``jax.jit``, ``jax.grad``, and
``jax.jvp``. It uses the existing host solver for the forward solution and
implicitly differentiates the converged equations. It does not differentiate
the Newton iterations, stopping decisions, or mask updates.

The factory takes
``(pressure_bar, pressure_boundaries_bar, temperature_initial,
bottom_temperature_initial, internal_flux, radiative_flux, neutral_gradient,
*, valid_state=None, **solver_options)``. Pressure arrays, initial guesses,
and numerical settings are fixed when constructing it. Every call to
``solve(params)`` solves the forward problem again with the current parameters.
All differentiable physical inputs must be explicit leaves of ``params``,
which is a PyTree of real floating-point scalars or arrays. Do not hide
differentiated values in callback closures.

``radiative_flux`` has signature ``(T, T_bottom, params)``. The neutral
gradient can remain a scalar or array, or use the same callback signature.
The optional ``valid_state`` also receives ``(T, T_bottom, params)``.
``internal_flux`` can be a fixed scalar or a callable ``internal_flux(params)``.
Other solver options have the same meaning as in ``solve_rce``.
The returned ``ImplicitRceResult`` is a NamedTuple with ``temperature``,
``bottom_temperature``, ``convective_mask``, and ``converged`` fields.

Let :math:`u=\log(T_1,\ldots,T_N,T_{\mathrm{bottom}})`, let :math:`\theta`
denote the parameters, and hold the converged mask :math:`m` fixed locally.
Writing the selected equations as :math:`R(u,\theta;m)=0`, the JVP solves

.. math::

   R_u\,\dot u=-R_\theta\,\dot\theta.

This includes parameter and temperature dependence in both radiation and the
neutral gradient. For reverse mode, an incoming cotangent :math:`\bar u`
gives the equivalent adjoint equations

.. math::

   R_u^\mathsf{T}\lambda=\bar u,
   \qquad
   \bar\theta=-R_\theta^\mathsf{T}\lambda.

JAX also includes any direct parameter dependence of a surrounding objective
and converts log-temperature sensitivities to temperature sensitivities.
These formulas require a nonsingular :math:`R_u` and a locally smooth branch
of every supplied physical callback. The mask is recomputed in each forward
solve; only its local derivative is held fixed.

Sensitivity evaluation requires strict separation from a convective switch:

.. math::

   F_{\mathrm{conv},i}>\epsilon_F\quad\text{on active connections},
   \qquad
   \nabla_i-\nabla_{\mathrm{neutral},i}<-\epsilon_\nabla
   \quad\text{on inactive connections},

where :math:`\epsilon_F` is the combined flux tolerance and
:math:`\epsilon_\nabla` is ``gradient_atol``. A converged primal solution can
fail this stricter sensitivity guard. Sensitivities are then invalidated
with NaN coefficients. This guard does not detect other branch changes
inside user callbacks. A singular tangent system can also produce nonfinite
sensitivities; check sensitivity finiteness as well as primal convergence.

Convergence tolerances alone do not guarantee accurate sensitivities,
especially for poorly conditioned equations in thin upper layers. Check
tolerance and grid refinement, and compare against finite differences of
separately converged solutions over a range of perturbation sizes. Those
perturbations must remain on the same convective and physical branches.

Nonconvergence or an expected input/domain ``ValueError`` in the host solve
returns NaN temperatures and ``converged=False``. Use ordinary ``solve_rce``
for detailed failure diagnostics.

The forward solve runs on the host, with its JAX callback calculations on
CPU. A CPU backend must be available, and callbacks must support CPU
execution without explicitly pinning operations or inputs to a GPU. The
wrapper uses `JAX pure callbacks
<https://docs.jax.dev/en/latest/_autosummary/jax.pure_callback.html>`_
and a custom JVP; wrapping the call in ``jit`` does not turn the forward
Newton loop into an accelerator program. Model callbacks must be pure,
without externally visible side effects.

The following self-contained one-layer example uses two independent,
artificial flux formulas to check the API; it is not a radiative-transfer
model. Near :math:`F=a=1`, its solution is :math:`T=300F^{1/4}` and
:math:`T_{\mathrm{bottom}}=400(F/a)^{1/4}`, where :math:`F` and :math:`a`
are the two supplied parameters.

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from exojax.atm.rce_implicit import make_implicit_rce_solver

   jax.config.update("jax_enable_x64", True)

   def radiation(temperature, bottom_temperature, params):
       return jnp.array([
           (temperature[0] / 300.0)**4,
           params["coefficient"] * (bottom_temperature / 400.0)**4,
       ])

   solve = make_implicit_rce_solver(
       jnp.array([1.0]), jnp.array([0.5, 4.0]),
       jnp.array([280.0]), 380.0,
       lambda params: params["flux"], radiation, 0.3,
       flux_atol=1.0e-9, flux_rtol=0.0,
   )
   params = {"flux": jnp.array(1.0), "coefficient": jnp.array(1.0)}
   result = jax.jit(solve)(params)
   assert bool(result.converged)

   bottom = lambda values: solve(values).bottom_temperature
   derivatives = jax.jit(jax.grad(bottom))(params)
   direction = {"flux": jnp.array(1.0), "coefficient": jnp.array(0.0)}
   _, directional_derivative = jax.jvp(bottom, (params,), (direction,))
   print(result.bottom_temperature)   # 400 K
   print(derivatives)                 # flux: 100, coefficient: -100
   print(directional_derivative)      # 100

Connecting an existing CKD table
----------------------------------------

Prepare or load an ``OpaCKD`` object before the nonlinear solve; see
:doc:`../tutorials/ckd_precompute_patches` and
:doc:`../tutorials/ckd_transpure_loadonly`. The following callback uses a
ready object ``opa`` and a single absorber with mass mixing ratio ``mmr``
and molecular mass ``molecular_mass`` in atomic mass units. ``gravity`` is in
cm/s2. Set the pressure arrays, initial temperatures and internal flux for
the column before running this snippet. This example explicitly supplies a
dry neutral gradient of 2/7 and has no stellar input.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import numpy as np

   from exojax.atm.rce import solve_rce
   from exojax.rt.flux import (
       integrate_ckd_flux,
       reconstruct_boundary_temperature,
       rtrun_emis_pureabs_ibased_linsap_fluxes,
   )
   from exojax.rt.layeropacity import layer_optical_depth_ckd
   from exojax.rt.planck import piBarr
   from exojax.rt.rtransfer import initialize_gaussian_quadrature

   jax.config.update("jax_enable_x64", True)
   pressure = jnp.asarray(pressure_bar)
   boundaries = jnp.asarray(pressure_boundaries_bar)
   dpressure = jnp.diff(boundaries)
   fraction = (pressure - boundaries[:-1]) / dpressure
   info = opa.ckd_info
   widths = info.band_edges[:, 1] - info.band_edges[:, 0]
   mus, weights_mu = initialize_gaussian_quadrature(8)
   tmin, tmax = float(info.T_grid[0]), float(info.T_grid[-1])
   pmin, pmax = float(info.P_grid[0]), float(info.P_grid[-1])
   pressure_valid = bool(jnp.all((pressure >= pmin) & (pressure <= pmax)))

   def valid_state(temperature, bottom_temperature):
       # Opacity is evaluated at centers, not at the black boundary.
       temperature = np.asarray(temperature, dtype=float)
       return pressure_valid and bool(
           np.all((temperature >= tmin) & (temperature <= tmax))
       )

   def radiative_flux(temperature, bottom_temperature):
       xs = opa.xstensor_ckd(temperature, pressure)  # (N, Ng, Nband)
       dtau = layer_optical_depth_ckd(
           dpressure, xs, mmr, molecular_mass, gravity
       )
       boundary_temperature = reconstruct_boundary_temperature(
           pressure, boundaries, temperature, bottom_temperature
       )
       up, down = rtrun_emis_pureabs_ibased_linsap_fluxes(
           dtau,
           piBarr(boundary_temperature, info.nu_bands)[:, None, :],
           mus,
           weights_mu,
           source_center=piBarr(temperature, info.nu_bands)[:, None, :],
           upper_fraction=fraction,
       )
       return integrate_ckd_flux(up - down, info.weights, widths)

   result = solve_rce(
       pressure,
       boundaries,
       temperature_initial,
       bottom_temperature_initial,
       internal_flux,
       radiative_flux,
       neutral_gradient=2.0 / 7.0,
       valid_state=valid_state,
   )
   if not result.converged:
       raise RuntimeError(result.status)

CKD interpolation clamps outside its table, so the explicit domain guard
rejects invalid trials before evaluating radiation. The bottom temperature
does not need the same table bound because its opacity is not interpolated.
The g weights sum to one; band widths and centers are in cm-1. Validate
spectral coverage and the band-center Planck approximation before interpreting
the integrated flux as bolometric. Independently sorted gas k distributions
cannot generally be added at equal g; multiple line absorbers need an overlap
treatment.

Scope
-----

The solver does not prescribe scattering, moist thermodynamics, composition,
or an elemental inventory solve. These must be provided consistently by the
callbacks; the supplied transfer kernel assumes pure absorption. The
Earth-analogue tutorial demonstrates moist composition feedback within a
fixed-pressure approximation. There is no interior cooling evolution.
Numerical convergence does not establish thermal stability or hysteresis.
Non-isothermal pure-absorption transport is motivated by methods
such as `HELIOS <https://arxiv.org/abs/1606.05474>`_; the discretization and
boundary conditions used here are specified above.

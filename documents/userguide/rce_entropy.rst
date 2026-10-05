Equilibrium entropy for gas-only RCE
===================================

The optional helpers in ``examples/_rce_chemistry.py`` and
``examples/_rce_entropy.py`` connect differentiable ExoGibbs chemistry to the
device RCE column interface. Chemistry is evaluated once at all layer centers
and the bottom boundary. Radiation and the convection condition use that same
state and the same dynamic elemental abundances.

Standard thermodynamic states
-----------------------------

This path requires ExoGibbs' ``exogibbs.thermo.standard`` provider from
`the standard thermodynamics change
<https://github.com/HajimeKawahara/exogibbs/commit/f6edda2>`_ or a revision
containing it. Its explicit
FastChem/NASA model supplies coherent standard Gibbs energies, entropies, and
heat capacities. Atomic and electron reference functions restore the absolute
standard states missing from formation equilibrium constants alone. Entropy
is evaluated from those states, without differentiating the minimized Gibbs
energy or nesting derivatives of the equilibrium solution.

The provider is an explicit alternative chemical model with 538 gas species
on its documented temperature domain. It retains the original FastChem
coefficients where audited, replaces 34 inconsistent fits with independently
identified NASA thermodynamic records, and excludes ``C4H6O4`` because there is
no validated replacement for that entry. Its metadata records the species
mapping, exclusions, and source hashes. The original FastChem preset is
unchanged. The original 539-species set is **not** certified by these checks.
The effect of the model changes and the suitability of the temperature domain
for an application must be assessed with that application's opacity and prior
choices; this helper does not establish WASP-18b prior coverage.
The provider's 80-state audit found changes up to 11.75 percent in electron
mole fraction among accepted comparisons. Use the same model for both the
TCE comparison and RCE, including the electron abundance supplied to H-minus.

Preparation is explicit. Use the provider's chemical setup, atomic masses,
isotope convention, and temperature domain together. For a source checkout,
make ``examples`` importable, for example with ``PYTHONPATH=src:examples``::

   import jax
   import jax.numpy as jnp
   from exogibbs.thermo.standard import prepare_fastchem_thermodynamics
   from _rce_chemistry import prepare_chemistry
   from _rce_entropy import prepare_equilibrium_entropy

   jax.config.update("jax_enable_x64", True)
   thermo = prepare_fastchem_thermodynamics(temperature_range=(1500.0, 4500.0))
   pressure_nodes = jnp.array([0.03, 0.4, 2.0])  # centers, then bottom; bar
   chemistry = prepare_chemistry(
       thermo.chemical_setup,
       pressure_nodes,
       element_masses_u=thermo.element_masses_u,
       isotope_convention=thermo.isotope_convention,
       required_species=("H2", "He1", "H1+", "H1-", "H2O1", "C1O1"),
       electron_species="e1-",
       atomic_hydrogen_species="H1",
       temperature_range=thermo.temperature_range,
       pressure_range=(1e-5, 100.0),
       epsilon_crit=1e-14,
       conservation_rtol=1e-7,
   )
   entropy = prepare_equilibrium_entropy(chemistry, thermo, entropy_scale=1e4)
   state = jax.jit(entropy)(jnp.array([2200.0, 2800.0, 3500.0]), 0.0, 0.6)
   assert jnp.all(state.valid)

Entropy and convection
----------------------

The helper returns entropy in J kg^-1 K^-1 using

.. math::

   s = \frac{R}{M_b}\sum_i n_i
       \left[\frac{s_i^0(T)}{R}-\ln x_i-\ln(P/P^0)\right].

Here ``M_b`` is the mass computed directly from the conserved elemental
amounts and the specified neutral-atom masses. It is shared by all nodes;
charge neutrality cancels the electron mass contribution. The changing total
gas amount during dissociation is not used as a mass denominator. The
``n*log(x)`` term uses its continuous zero-amount limit. Derivatives at an
exactly absent species use a fixed-absence extension; arbitrary derivatives
across that boundary are not claimed.

``make_entropy_evaluator`` converts this state to ``ColumnEvaluation``::

   from _rce_entropy import make_entropy_evaluator

   evaluate_column = make_entropy_evaluator(
       entropy,
       radiative_flux,
       lambda parameters: (parameters["log_metal_scale"], parameters["c_over_o"]),
   )

The supplied ``radiative_flux(T, T_bottom, parameters, chemistry_state)`` must
return the N+1 upward net boundary fluxes. The chemistry state includes the
bottom as its final node; layer opacity calculations consume its first N
entries. Both elemental parameters remain dynamic and shared across all nodes.
The callback is skipped if chemistry or thermodynamics is invalid.
Pass this evaluator to the device forward solver. Implicit solution
sensitivities additionally require ``exojax.atm.rce_device_implicit``; the
combined chemistry/RCE acceptance test skips explicitly if that optional
implementation is unavailable.

The dimensionless instability is
``diff(s) / (entropy_scale * diff(log(pressure_nodes)))``. Positive values are
unstable; active convective connections impose equal specific entropy.
``entropy_scale`` is a fixed positive reference in entropy units. Set
``stability_atol`` for this normalization: a legacy temperature-gradient
tolerance is not interchangeable with it. For constant composition and heat
capacity, the conversion factor is ``cp / entropy_scale`` and the usual
adiabat is recovered.

Every included species must have finite standard entropy and positive standard
heat capacity at every evaluated node, even if its abundance is very small.
For a coherent stable ideal-gas equilibrium, chemical relaxation adds a
nonnegative contribution to the fixed-composition heat capacity. This
conservative check avoids another equilibrium solve. The implicit RCE solver
separately checks derivative finiteness and switching margins. Unsupported
standard states, mismatched species or mass conventions, nonconverged
chemistry, and invalid domains are rejected explicitly. Use ``jax.lax.map``
for batches containing valid and invalid columns so conditional guards remain
intact.

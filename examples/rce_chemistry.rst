Gas chemistry for RCE/TCE examples
=================================

``_rce_chemistry.py`` is an optional ExoGibbs integration helper. ExoGibbs is
not an ExoJAX installation dependency. It requires the upstream change that
retains the implicit composition JVP/VJP when ``return_diagnostics=True``.
Earlier ExoGibbs implementations differentiate the numerical iterations on
that route, which cannot provide the required reverse-mode derivative.
Use ExoGibbs with `the differentiable diagnostics fix
<https://github.com/HajimeKawahara/exogibbs/commit/44ec92d>`_ or a later
revision containing that change.

Prepare the adapter once, with explicit species names, neutral atom masses,
a bulk-element isotope convention, pressure grid, and validated table domains.
The domain limits below are illustrative smoke-test limits; they do not certify
the thermochemistry or opacity coverage of WASP-18b retrieval priors::

    import jax
    import jax.numpy as jnp
    from exogibbs.presets.fastchem4 import chemsetup
    from exojax.database.molinfo import element_mass
    from _rce_chemistry import prepare_chemistry

    jax.config.update("jax_enable_x64", True)
    setup = chemsetup(path="FastChem4/logK/logK.dat", silent=True)
    chemistry = prepare_chemistry(
        setup, jnp.array([0.01, 0.1, 1.0]),
        element_masses_u=element_mass,
        isotope_convention="ExoJAX neutral atom masses; bulk elemental chemistry",
        required_species=("H2", "He1", "H1+", "H1-", "H2O1", "C1O1"),
        electron_species="e1-", atomic_hydrogen_species="H1",
        temperature_range=(1500.0, 4500.0),
        pressure_range=(1e-5, 100.0),
        epsilon_crit=1e-14, conservation_rtol=1e-7,
    )
    solve = jax.jit(chemistry)
    state = solve(jnp.array([2200.0, 2800.0, 3500.0]), 0.0, 0.6)
    # Check jnp.all(state.diagnostics.valid) before opacity or radiation.

Arguments after temperature are ``log_metal_scale`` and ``c_over_o``. The first
scales metal/H number ratios by ``10**log_metal_scale``, holding H and He fixed.
C/O redistribution preserves the scaled C+O number sum. The returned metal mass
fraction is derived and need not remain constant when C/O changes. ``n`` uses
the conserved-element amount normalization; ``x`` divides by all particles,
including electrons. Mass fractions, mean particle mass, and electron/atomic-H
number densities derive from the same composition. Ion masses include the
corresponding electron mass correction. Isotopologue partitioning for opacity
is a separate model choice; this helper does not infer isotope abundances.

``vmap_cold`` is explicit. Dynamic abundances are never put into a scan-body
cache. Composition and convergence diagnostics share one equilibrium solve,
with first-order implicit derivatives. These derivatives require a converged,
nonsingular equilibrium; higher-order derivatives are outside this helper's
contract. Entropy and the convection condition are P5 work.

A layer is valid only if the solver converged, the composition is finite, and
both element and charge conservation pass. Element errors are relative to each
input elemental amount. Charge error divides the net charge by the sum of
absolute charged amounts. ExoGibbs' convergence norm uses absolute amounts:
trace-element conservation therefore needs its own check. In the packaged ion
table smoke test, ``epsilon_crit=1e-11`` can report convergence with relative
trace-element errors near ``1e-4``. The explicit tolerances above resolve the
tested states to ``1e-7`` relative conservation. They must be revalidated over
the intended prior, rather than relaxed until all proposals pass.

Status codes are ``SUCCESS=0``, ``NOT_CONVERGED=1``, ``OUTSIDE_DOMAIN=2``,
``CONSERVATION_FAILED=3``, and ``NONFINITE=4``. An out-of-domain column bypasses
thermochemistry and returns NaN physical arrays. For a mixed batch of valid and
invalid columns, use sequential ``jax.lax.map``; unrestricted ``vmap`` can
execute both branches. Safe rejection inside the differentiated likelihood
remains P4 work.

``hminus_optical_depth`` accepts the resulting number densities directly and
broadcasts each layer's ``mmw`` along the wavenumber axis. Do not multiply by an
additional H-minus abundance: the existing continuum formula already uses the
electron and neutral atomic H populations.

Validation is CPU/x64 and offline. With the patched ExoGibbs source on
``PYTHONPATH``, run::

    pytest tests/unittests/examples/rce_chemistry_test.py

The tests skip if the optional ExoGibbs package is absent. They include a small
ion/association gas, independently reconverged finite differences at two step
sizes, changing abundances under repeated JIT, leakage checks, failure/domain
checks, layer-dependent H-minus conversion, and the packaged FastChem4 ion
table. Passing the smoke test does not validate a complete retrieval prior.

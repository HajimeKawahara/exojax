Local convective-support selection
==================================

The private WASP-18b continuation example can optionally select the next
convective mask from a feasible linearized column. This addresses a failure
of simultaneous mask updates: removing a negative-flux connection and adding
an unstable connection can alternate between two inadmissible masks even
when their union has a valid equilibrium.

For an already prepared column, enable the option explicitly::

    from _rce_wasp18b_continuation import PreparedMaskContinuation

    solver = PreparedMaskContinuation(
        column, parameters, max_stages=15, convective_support=True,
    )
    result = solver.run(parameters)
    if result.state.converged:
        temperature = result.state.temperature

This uses the existing continuation entry point. The default remains
``convective_support=False``. No package API, dependency requirement, native
solver equation, or convergence tolerance changes. Each ``run`` starts from
the original column seed, independently of earlier parameter points.

The selector is tried after a recoverable failed native stage only when
another stage remains and the returned domain and physics are valid. It
computes the full flux/stability Jacobian at those returned temperatures.
It changes only the next mask; the next native solve performs all temperature
updates, line searches, domain/physics checks, and final acceptance. A linear
proposal alone is never a converged result. Stage records include
``convective_support_selected`` and ``next_convective_mask``.

For radiative flux error ``f``, stability excess ``s``, and a log-temperature
correction ``d``, the local equations are::

    f + Jf d + E c = 0
    s + Js d <= 0
    c >= 0
    c * (s + Js d) = 0

``E`` injects convective flux below the top boundary. Eliminating ``d`` gives
a small complementarity problem for each candidate support. The search
includes a two-connection halo around unstable radiative predictions and
the smaller native-scaled current slacks. It enumerates at most 12 candidate
connections, checks stability at all connections, skips singular supports,
and orders feasible solutions by maximum log-temperature step, Euclidean
step norm, then active indices. Its feasibility tolerances account only for
floating-point roundoff. Native tolerances decide nonlinear convergence.

A singular/nonfinite flux model, too many candidates, or no feasible local
support leaves the original mask-update rule in effect. These outcomes do
not prove that a nonlinear equilibrium is absent. Existing unrecoverable
native failures, stage budgets, and failed-state cycle checks still apply.
The optional full Jacobian adds computation and memory; the host search
does not supply implicit equilibrium derivatives or an HMC-ready solver.

CPU regression tests cover a coupled two-connection mask cycle, a nonlinear
variant with an analytic equilibrium, independent flux/stability checks,
per-point seed reset, inaccessible roots, and exhausted stage budgets. The
old mask rule fails on both cycling examples; the enabled continuation
reaches native convergence in two stages. Tests require no GPU, ExoGibbs,
line database, opacity archive, or imports from ``results``::

    JAX_PLATFORMS=cpu python -m pytest \
        tests/unittests/examples/rce_wasp18b_continuation_test.py \
        tests/unittests/examples/rce_wasp18b_convective_support_test.py \
        tests/unittests/atm/test_rce_device.py

This integration has not repeated the full WASP-18b physical calculation.
It does not establish grid/opacity convergence, thermal stability, branch
coverage, or production implicit-derivative validity.

"""Enumerate local convective supports for a frozen linear RCE surrogate.

This private example helper proposes supports only. The native nonlinear
solver must validate the temperatures, physics, and final equilibrium.
"""

from itertools import combinations

import numpy as np


def solve_linearized_complementarity(
    flux_error, stability_excess, flux_jacobian, stability_jacobian, candidates
):
    """Return feasible local steps in deterministic minimum-step order.

    ``flux_error`` has ``n`` boundaries, including TOA; ``stability_excess``
    has ``n - 1`` connections. Jacobian columns are the ``n`` log temperatures.
    Connection ``i`` supplies nonnegative convective flux at boundary ``i+1``.
    All connections, including those outside ``candidates``, must be stable.
    At most 12 candidate connections are enumerated. Singular supports are
    skipped; a singular flux Jacobian raises ``numpy.linalg.LinAlgError``.

    For each support, eliminate temperature using ``d = d0 - U c``, where
    ``Jf d0 = -f`` and ``Jf U = E``. The complementary stability slack is
    ``w = -s - Js d = q + Js U c``. Only floating-point roundoff is tolerated.
    An empty list means no nonsingular feasible support was found in this
    local surrogate and candidate set; it does not prove nonlinear absence.

    Returned dictionaries contain ``delta``, ``mask``, ``flux`` (convective,
    including zero TOA), ``stability`` (linear predicted excess), and
    ``linear_errors``. Ordering uses max absolute log-temperature step, then
    its Euclidean norm, then active connection indices.
    """
    f, s, jf, js = [np.asarray(x, dtype=float) for x in
                    (flux_error, stability_excess, flux_jacobian, stability_jacobian)]
    if f.ndim != 1 or len(f) < 1:
        raise ValueError("flux_error must be a nonempty vector")
    n = len(f)
    if s.shape != (n - 1,) or jf.shape != (n, n) or js.shape != (n - 1, n):
        raise ValueError("Inconsistent flux, stability, or Jacobian shapes")
    if not all(np.all(np.isfinite(x)) for x in (f, s, jf, js)):
        raise ValueError("Inputs must be finite")
    raw = list(candidates)
    if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
           for i in raw):
        raise ValueError("Candidate indices must be integers")
    if len(raw) > 12 or len(set(raw)) != len(raw) or any(i < 0 or i >= n - 1 for i in raw):
        raise ValueError("Use at most 12 distinct valid connection indices")
    indices = np.asarray(sorted(raw), dtype=int)
    k = len(indices)
    injection = np.zeros((n, k))
    injection[indices + 1, np.arange(k)] = 1.0
    scale = np.max(np.abs(jf), axis=1)
    if np.any(scale == 0):
        raise np.linalg.LinAlgError("The flux Jacobian has a zero row")
    solved = np.linalg.solve(jf / scale[:, None],
                             np.column_stack((-f, injection)) / scale[:, None])
    if not np.all(np.isfinite(solved)):
        raise np.linalg.LinAlgError("The flux solve produced a nonfinite result")
    d0, u = solved[:, 0], solved[:, 1:]
    q, matrix = -s - js @ d0, js @ u
    roundoff = 64 * np.finfo(float).eps * max(n, 1)
    feasible = []
    for count in range(k + 1):
        for support in combinations(range(k), count):
            active = np.asarray(support, dtype=int)
            c = np.zeros(k)
            if count:
                block = matrix[np.ix_(indices[active], active)]
                try:
                    c[active] = np.linalg.solve(block, -q[indices[active]])
                except np.linalg.LinAlgError:
                    continue
            if not np.all(np.isfinite(c)):
                continue
            delta = d0 - u @ c
            if not np.all(np.isfinite(delta)):
                continue
            flux_scale = 1 + np.abs(f) + np.abs(jf) @ np.abs(delta)
            if np.any(c < -roundoff * (flux_scale[indices + 1] + np.abs(c))):
                continue
            # Remove only roundoff-sized negative fluxes, then recheck equations.
            c = np.maximum(c, 0)
            delta = d0 - u @ c
            mask = np.zeros(n - 1, dtype=bool)
            mask[indices[active]] = True
            convective = injection @ c
            stability = s + js @ delta
            total_error = f + jf @ delta + convective
            complementarity = convective[1:] * stability
            if not all(np.all(np.isfinite(x)) for x in
                       (delta, convective, stability, total_error, complementarity)):
                continue
            flux_tolerance = roundoff * (1 + np.abs(f) + np.abs(jf) @ np.abs(delta)
                                         + np.abs(convective))
            stability_tolerance = roundoff * (1 + np.abs(s) + np.abs(js) @ np.abs(delta))
            if (np.any(np.abs(total_error) > flux_tolerance)
                    or np.any(np.abs(stability[mask]) > stability_tolerance[mask])
                    or np.any(stability[~mask] > stability_tolerance[~mask])):
                continue
            feasible.append(dict(
                delta=delta, mask=mask, flux=convective, stability=stability,
                linear_errors=dict(
                    max_total_flux=float(np.max(np.abs(total_error), initial=0)),
                    max_active_stability=float(np.max(np.abs(stability[mask]), initial=0)),
                    max_inactive_stability=float(np.max(stability[~mask], initial=0)),
                    max_complementarity_product=float(np.max(
                        np.abs(complementarity), initial=0)),
                ),
            ))
    feasible.sort(key=lambda item: (float(np.max(np.abs(item["delta"]))),
                                   float(np.linalg.norm(item["delta"])),
                                   tuple(np.flatnonzero(item["mask"]))))
    return feasible


def select_convective_mask(quantities, jacobian, flux_tolerance, stability_atol):
    """Propose a feasible mask near the current or radiative instability.

    Quantities concatenate radiative flux minus internal flux and stability
    excess; Jacobian columns differentiate log temperatures. Native-scaled
    slacks and the radiative affine solution seed a two-connection halo.
    Return None if the finite local model has no usable support, is singular,
    or needs more than 12 candidates. No temperature step is applied here.
    """
    q, full = np.asarray(quantities), np.asarray(jacobian)
    n = full.shape[1]
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(full)):
        return None
    scale = np.max(abs(full[:n]), axis=1)
    if np.any(scale == 0):
        return None
    try:
        radiative = np.linalg.solve(full[:n] / scale[:, None], -q[:n] / scale)
        unstable = np.flatnonzero((q + full @ radiative)[n:] > -stability_atol)
        current = np.flatnonzero(-q[n:] / stability_atol < -q[1:n] / flux_tolerance)
        seeds = np.union1d(unstable, current)
        candidates = sorted({int(i + j) for i in seeds for j in range(-2, 3)
                             if 0 <= i + j < n - 1})
        if len(candidates) > 12:
            return None
        roots = solve_linearized_complementarity(
            q[:n], q[n:], full[:n], full[n:], candidates)
    except np.linalg.LinAlgError:
        return None
    return roots[0]["mask"] if roots else None

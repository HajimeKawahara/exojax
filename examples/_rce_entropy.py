"""Mass-specific equilibrium entropy for optional gas-only RCE examples.

Entropy is in J kg^-1 K^-1. The amount normalization cancels against conserved
mass; normalization by the changing total gas amount would be incorrect during
dissociation. Standard-state data must describe the same chemical potentials as
the equilibrium calculation, including atomic and electron reference functions.
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from exojax.atm.rce_device import ColumnEvaluation

# Exact SI molar gas constant, in J mol^-1 K^-1.
GAS_CONSTANT = 8.31446261815324

SUCCESS = 0
INVALID_CHEMISTRY = 1
INVALID_THERMODYNAMICS = 2
NONPOSITIVE_HEAT_CAPACITY = 3
NONFINITE_ENTROPY = 4


class EquilibriumEntropyState(NamedTuple):
    """Entropy and shared chemistry at every center and the bottom boundary.

    ``valid`` and ``status`` are per-node primal diagnostics. The RCE implicit
    solver separately checks derivative finiteness and switching margins.
    """

    chemistry: object
    specific_entropy: jax.Array
    valid: jax.Array
    status: jax.Array


@dataclass(frozen=True, eq=False)
class PreparedEquilibriumEntropy:
    chemistry: object
    thermodynamics: object
    entropy_scale: float
    evaluate: Callable

    def __call__(self, temperature, log_metal_scale, c_over_o):
        return self.evaluate(temperature, log_metal_scale, c_over_o)


def specific_entropy(
    amounts,
    pressure_bar,
    masses_u,
    standard_entropy_over_r,
    *,
    reference_pressure_bar=1.0,
    conserved_mass_kg=None,
):
    """Return ideal-gas mixture entropy per conserved mass (J kg^-1 K^-1).

    ``amounts`` and standard entropy have shape (..., species); pressure has
    shape (...,). Species masses use the same bulk-element convention as the
    chemistry, including electron mass corrections for ions. Standard entropy
    is dimensionless (s0/R) at ``reference_pressure_bar``.

    ``conserved_mass_kg`` may supply the mass computed from conserved elemental
    amounts, shared by every layer. Otherwise the mass is computed from the
    species amounts. The equilibrium adapter always uses elemental mass.

    The n*log(x) term has its continuous zero-amount limit. Derivatives at an
    exactly absent species use the fixed-absence extension; physical composition
    derivatives apply in the positive interior, not across a zero boundary.
    The caller is responsible for finite, nonnegative amounts and positive
    pressure and total mass, and for matching the standard-state convention.
    """
    amounts = jnp.asarray(amounts)
    masses_u = jnp.asarray(masses_u)
    standard_entropy_over_r = jnp.asarray(standard_entropy_over_r)
    pressure_bar = jnp.asarray(pressure_bar)
    if (
        amounts.ndim < 1
        or masses_u.shape != amounts.shape[-1:]
        or standard_entropy_over_r.shape != amounts.shape
        or pressure_bar.shape != amounts.shape[:-1]
    ):
        raise ValueError("Entropy inputs must share their layer and species axes.")
    fractions = amounts / jnp.sum(amounts, axis=-1, keepdims=True)
    # Guard the logarithm itself: where(n > 0, n * log(x), 0) still evaluates
    # log(0), which can contaminate reverse-mode derivatives at absent species.
    log_fraction = jnp.log(jnp.where(fractions > 0, fractions, 1.0))
    entropy_over_r = jnp.sum(
        amounts
        * (
            standard_entropy_over_r
            - log_fraction
            - jnp.log(pressure_bar / reference_pressure_bar)[..., None]
        ),
        axis=-1,
    )
    mass_kg_per_mole_basis = (
        amounts @ masses_u * 1e-3
        if conserved_mass_kg is None
        else jnp.asarray(conserved_mass_kg)
    )
    if mass_kg_per_mole_basis.shape not in ((), amounts.shape[:-1]):
        raise ValueError("Conserved mass must be scalar or match the layer axes.")
    return GAS_CONSTANT * entropy_over_r / mass_kg_per_mole_basis


def prepare_equilibrium_entropy(chemistry, thermodynamics, *, entropy_scale):
    """Bind coherent standard thermodynamics to a fixed chemistry node grid.

    ``chemistry`` is prepared with centers followed by the bottom pressure.
    ``thermodynamics`` is ExoGibbs' ``StandardThermodynamics`` provider, built
    from that same chemical setup. Its absolute standard states may differ from
    the equilibrium chemical potentials by the provider's conserved-element
    gauge. No species or missing standard state is silently removed or filled.

    ``entropy_scale`` is a fixed positive reference in J kg^-1 K^-1. It makes
    the RCE stability residual dimensionless and is not a retrieval parameter.
    Require positive standard cp for every species, even a trace constituent.
    For a coherent stable ideal-gas equilibrium, the chemical response adds a
    nonnegative contribution to the fixed-composition heat capacity. This
    conservative check avoids a second equilibrium solve or higher-order AD.
    """
    if (
        tuple(chemistry.species) != tuple(thermodynamics.species)
        or tuple(chemistry.elements) != tuple(thermodynamics.elements)
        or chemistry.hvector_func is not thermodynamics.hvector_func
        or not np.array_equal(
            np.asarray(chemistry.formula_matrix),
            np.asarray(thermodynamics.chemical_setup.formula_matrix),
        )
    ):
        raise ValueError("Chemistry and standard states must use the same setup.")
    if thermodynamics.standard_pressure_bar != 1.0:
        raise ValueError("The chemistry adapter uses a standard pressure of 1 bar.")
    if chemistry.isotope_convention != thermodynamics.isotope_convention:
        raise ValueError(
            "Chemistry and standard states need the same isotope convention."
        )
    for index, element in enumerate(chemistry.elements):
        if element == "e-":
            continue
        if element not in thermodynamics.element_masses_u or (
            np.asarray(chemistry.element_masses_u[index])
            != np.asarray(
                thermodynamics.element_masses_u[element],
                dtype=chemistry.element_masses_u.dtype,
            )
        ):
            raise ValueError(f"Inconsistent standard-state atomic mass: {element}")
    if (
        chemistry.temperature_range[0] < thermodynamics.temperature_range[0]
        or chemistry.temperature_range[1] > thermodynamics.temperature_range[1]
    ):
        raise ValueError("The chemistry domain exceeds the standard-state domain.")
    pressure = np.asarray(chemistry.pressure_bar)
    if pressure.size < 2 or np.any(np.diff(pressure) <= 0):
        raise ValueError("Use increasing pressure centers followed by the bottom node.")
    if (
        np.ndim(entropy_scale) != 0
        or not np.isfinite(entropy_scale)
        or entropy_scale <= 0
    ):
        raise ValueError("entropy_scale must be a finite positive scalar.")
    with np.errstate(over="ignore", under="ignore"):
        scale = np.asarray(entropy_scale, dtype=pressure.dtype)
    if not np.isfinite(scale) or scale < np.finfo(pressure.dtype).tiny:
        raise ValueError(
            "entropy_scale must stay positive and finite in the device dtype."
        )
    entropy_scale = float(scale)

    def evaluate(temperature, log_metal_scale, c_over_o):
        temperature = jnp.asarray(temperature, dtype=chemistry.pressure_bar.dtype)
        state = chemistry(temperature, log_metal_scale, c_over_o)
        standard_s = jax.vmap(thermodynamics.standard_entropy_r)(temperature)
        standard_cp = jax.vmap(thermodynamics.standard_cp_r)(temperature)
        if standard_s.shape != state.n.shape or standard_cp.shape != state.n.shape:
            raise ValueError(
                "Standard-state arrays must match the fixed species order."
            )
        mass = state.b @ chemistry.element_masses_u * 1e-3
        value = specific_entropy(
            state.n,
            chemistry.pressure_bar,
            chemistry.masses_u,
            standard_s,
            reference_pressure_bar=thermodynamics.standard_pressure_bar,
            conserved_mass_kg=mass,
        )
        chemistry_valid = (
            state.diagnostics.valid
            & jnp.all(jnp.isfinite(state.n) & (state.n >= 0), axis=-1)
            & jnp.isfinite(mass)
            & (mass > 0)
        )
        standard_valid = jnp.all(
            jnp.isfinite(standard_s) & jnp.isfinite(standard_cp), axis=-1
        )
        positive_cp = jnp.all(standard_cp > 0, axis=-1)
        status = jnp.where(
            ~chemistry_valid,
            INVALID_CHEMISTRY,
            jnp.where(
                ~standard_valid,
                INVALID_THERMODYNAMICS,
                jnp.where(
                    ~positive_cp,
                    NONPOSITIVE_HEAT_CAPACITY,
                    jnp.where(jnp.isfinite(value), SUCCESS, NONFINITE_ENTROPY),
                ),
            ),
        ).astype(jnp.int32)
        return EquilibriumEntropyState(state, value, status == SUCCESS, status)

    return PreparedEquilibriumEntropy(
        chemistry, thermodynamics, entropy_scale, evaluate
    )


def make_entropy_evaluator(entropy, radiative_flux, elemental_parameters):
    """Share one N+1-node chemistry evaluation between radiation and stability.

    ``elemental_parameters(parameters)`` returns the dynamic metal/H logarithmic
    scale and C/O used at all centers and the bottom. ``radiative_flux`` takes
    ``(T, T_bottom, parameters, chemistry_state)`` and returns N+1 boundary
    fluxes. Its chemistry state includes the bottom as its final node, so layer
    opacity calculations use the first N entries. Invalid entropy/chemistry
    skips radiation. Use ``lax.map`` for mixed valid/invalid columns.
    """
    pressure = np.asarray(entropy.chemistry.pressure_bar)
    delta_log_pressure = jnp.asarray(np.diff(np.log(pressure.astype(float))))

    def evaluate(temperature, bottom_temperature, parameters):
        temperature = jnp.asarray(
            temperature, dtype=entropy.chemistry.pressure_bar.dtype
        )
        bottom_temperature = jnp.asarray(bottom_temperature, dtype=temperature.dtype)
        if (
            temperature.shape != (pressure.size - 1,)
            or jnp.ndim(bottom_temperature) != 0
        ):
            raise ValueError("Expected N center temperatures and a scalar bottom.")
        nodes = jnp.concatenate((temperature, jnp.asarray(bottom_temperature)[None]))
        log_metal_scale, c_over_o = elemental_parameters(parameters)
        state = entropy(nodes, log_metal_scale, c_over_o)
        valid = jnp.all(state.valid)
        flux = jax.lax.cond(
            valid,
            lambda _: jnp.asarray(
                radiative_flux(
                    temperature, bottom_temperature, parameters, state.chemistry
                ),
                dtype=nodes.dtype,
            ),
            lambda _: jnp.full_like(nodes, jnp.nan),
            operand=None,
        )
        status = state.status[jnp.argmax(~state.valid)]
        return ColumnEvaluation(
            flux,
            jnp.diff(state.specific_entropy)
            / (entropy.entropy_scale * delta_log_pressure),
            valid,
            status,
        )

    return evaluate

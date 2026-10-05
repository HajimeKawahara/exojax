"""Optional, first-order ExoGibbs adapter for gas-only RCE/TCE examples.

Requires ExoGibbs with differentiable ``return_diagnostics=True``. Prepare once
outside JAX transformations; pass retrieval parameters explicitly at every call.
The caller supplies validated table domains and a bulk-element isotope convention.
This helper does not provide entropy, rainout, or isotope-resolved chemistry.
"""

from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from exojax.utils.constants import bar_cgs, kB, m_u, mecgs

SUCCESS = 0
NOT_CONVERGED = 1
OUTSIDE_DOMAIN = 2
CONSERVATION_FAILED = 3
NONFINITE = 4


class ChemistryDiagnostics(NamedTuple):
    """Per-layer status and conservation checks; ``in_domain`` is column-wide."""

    status: jax.Array
    valid: jax.Array
    converged: jax.Array
    in_domain: jax.Array
    element_relative_error: jax.Array
    charge_relative_error: jax.Array
    iterations: jax.Array
    final_residual: jax.Array


class ChemistryState(NamedTuple):
    """Amounts ``n`` conserve ``b``; fractions ``x`` include free electrons.

    ``mmw`` is in atomic mass units; number densities are in cm^-3. The metal
    mass fraction is derived from the elemental abundances, excluding H/He.
    """

    n: jax.Array
    x: jax.Array
    mmw: jax.Array
    mass_fractions: jax.Array
    number_density_e: jax.Array
    number_density_h: jax.Array
    b: jax.Array
    metal_mass_fraction: jax.Array
    diagnostics: ChemistryDiagnostics


@dataclass(frozen=True, eq=False)
class PreparedChemistry:
    species: tuple[str, ...]
    elements: tuple[str, ...]
    formula_matrix: jax.Array
    masses_u: jax.Array
    element_masses_u: jax.Array
    charges: jax.Array
    isotope_convention: str
    pressure_bar: jax.Array
    temperature_range: tuple[float, float]
    pressure_range: tuple[float, float]
    hvector_func: Callable
    elemental_abundances: Callable
    evaluate: Callable

    def __call__(self, temperature, log_metal_scale, c_over_o):
        return self.evaluate(temperature, log_metal_scale, c_over_o)


def prepare_chemistry(
    setup,
    pressure_bar,
    *,
    element_masses_u: Mapping[str, float],
    isotope_convention: str,
    required_species: tuple[str, ...],
    electron_species: str,
    atomic_hydrogen_species: str,
    temperature_range: tuple[float, float],
    pressure_range: tuple[float, float],
    epsilon_crit: float = 1e-11,
    max_iter: int = 1000,
    conservation_rtol: float = 1e-8,
) -> PreparedChemistry:
    """Fix species order, masses, charges, pressure grid and validity domains.

    Element masses describe neutral atoms in the stated isotope convention;
    ion masses include the electron mass correction. ``e-`` must be the charge
    constraint row, with positive coefficients denoting excess electrons.
    H/He remain fixed. ``10**log_metal_scale`` scales metal/H number ratios;
    changing C/O preserves the scaled C+O number sum, not metal mass fraction.
    Pressure and ExoGibbs' reference pressure are in bar (``Pref=1``).

    A column outside the supplied domain is rejected before thermochemistry.
    Its physical outputs are NaN and its status is OUTSIDE_DOMAIN. Evaluate
    mixed valid/invalid columns with ``lax.map``, since ``vmap`` can evaluate
    both branches of a conditional. Only converged states have physical AD.
    """
    from exogibbs.api.equilibrium import EquilibriumOptions, equilibrium_profile

    species = tuple(setup.species or ())
    elements = tuple(setup.elements or ())
    if (
        not species
        or not elements
        or len(set(species)) != len(species)
        or len(set(elements)) != len(elements)
    ):
        raise ValueError("Unique species and element names are required.")
    missing = set(required_species) | {electron_species, atomic_hydrogen_species}
    missing -= set(species)
    if missing:
        raise ValueError(f"Missing required species: {sorted(missing)}")
    if not {"H", "He", "C", "O", "e-"}.issubset(elements):
        raise ValueError("H, He, C, O and the e- charge constraint are required.")
    if not isinstance(isotope_convention, str) or not isotope_convention.strip():
        raise ValueError("An explicit bulk-element isotope convention is required.")
    matrix = np.asarray(setup.formula_matrix, dtype=float)
    reference = np.asarray(setup.element_vector_reference, dtype=float)
    if matrix.shape != (len(elements), len(species)) or not np.all(np.isfinite(matrix)):
        raise ValueError("The formula matrix must be finite and match the fixed names.")
    charge_index = elements.index("e-")
    element_mask = np.arange(len(elements)) != charge_index
    if (
        reference.shape != (len(elements),)
        or not np.all(np.isfinite(reference))
        or np.any(reference[element_mask] <= 0)
        or reference[charge_index] != 0
    ):
        raise ValueError(
            "Reference abundances must be positive, finite and charge neutral."
        )
    if np.any(matrix[element_mask] < 0) or np.linalg.matrix_rank(matrix) != len(
        elements
    ):
        raise ValueError(
            "Element counts must be nonnegative with independent constraints."
        )
    missing_masses = set(elements) - {"e-"} - set(element_masses_u)
    if missing_masses:
        raise ValueError(f"Missing element masses: {sorted(missing_masses)}")
    element_masses = np.array(
        [mecgs / m_u if name == "e-" else element_masses_u[name] for name in elements]
    )
    masses = element_masses @ matrix
    charges = -matrix[charge_index]
    if (
        not np.all(np.isfinite(element_masses))
        or np.any(element_masses <= 0)
        or np.any(masses <= 0)
        or not np.any(charges > 0)
    ):
        raise ValueError("Positive masses and at least one positive ion are required.")
    electron_index = species.index(electron_species)
    hydrogen_index = species.index(atomic_hydrogen_species)
    expected_e = np.eye(len(elements))[charge_index]
    expected_h = np.eye(len(elements))[elements.index("H")]
    if not np.array_equal(matrix[:, electron_index], expected_e) or not np.array_equal(
        matrix[:, hydrogen_index], expected_h
    ):
        raise ValueError(
            "Electron and neutral atomic H species must match their formula columns."
        )
    for bounds in (temperature_range, pressure_range):
        if (
            len(bounds) != 2
            or not np.all(np.isfinite(bounds))
            or not 0 < bounds[0] < bounds[1]
        ):
            raise ValueError("Validity bounds must be finite, positive and increasing.")
    pressure = np.asarray(pressure_bar, dtype=float)
    if (
        pressure.ndim != 1
        or pressure.size == 0
        or not np.all(np.isfinite(pressure))
        or np.any(pressure < pressure_range[0])
        or np.any(pressure > pressure_range[1])
    ):
        raise ValueError(
            "The pressure grid must be one-dimensional and inside its domain."
        )
    if (
        not np.isfinite(epsilon_crit)
        or epsilon_crit <= 0
        or not np.isfinite(conservation_rtol)
        or conservation_rtol <= 0
        or not isinstance(max_iter, int)
        or max_iter < 1
    ):
        raise ValueError(
            "Solver tolerances must be positive and max_iter a positive integer."
        )

    pressure = jnp.asarray(pressure)
    reference = jnp.asarray(reference)
    masses = jnp.asarray(masses)
    charges = jnp.asarray(charges)
    elemental_matrix = jnp.asarray(matrix[element_mask])
    element_masses = jnp.asarray(element_masses)
    metal_mask = jnp.asarray([name not in ("H", "He", "e-") for name in elements])
    carbon, oxygen = elements.index("C"), elements.index("O")
    options = EquilibriumOptions(
        epsilon_crit=epsilon_crit, max_iter=max_iter, method="vmap_cold"
    )

    def elemental_abundances(log_metal_scale, c_over_o):
        b = reference * jnp.where(metal_mask, jnp.power(10.0, log_metal_scale), 1.0)
        total_co = b[carbon] + b[oxygen]
        return (
            b.at[carbon]
            .set(total_co * c_over_o / (1 + c_over_o))
            .at[oxygen]
            .set(total_co / (1 + c_over_o))
        )

    def evaluate(temperature, log_metal_scale, c_over_o):
        temperature = jnp.asarray(temperature)
        if temperature.shape != pressure.shape:
            raise ValueError("Temperature must match the fixed pressure grid.")
        if jnp.ndim(log_metal_scale) != 0 or jnp.ndim(c_over_o) != 0:
            raise ValueError(
                "Metal scale and C/O must be scalars shared across layers."
            )
        b = elemental_abundances(log_metal_scale, c_over_o)
        in_domain = (
            jnp.all(jnp.isfinite(temperature))
            & jnp.all(temperature >= temperature_range[0])
            & jnp.all(temperature <= temperature_range[1])
            & jnp.isfinite(log_metal_scale)
            & jnp.isfinite(c_over_o)
            & (c_over_o > 0)
            & jnp.all(jnp.isfinite(b))
            & jnp.all(b[element_mask] > 0)
        )

        def solve(_):
            result, diagnostics = equilibrium_profile(
                setup,
                temperature,
                pressure,
                b,
                Pref=1.0,
                options=options,
                return_diagnostics=True,
            )
            n, x = result.n, result.x
            mmw = x @ masses
            element_error = jnp.max(
                jnp.abs(n @ elemental_matrix.T - b[element_mask]) / b[element_mask],
                axis=-1,
            )
            charge_error = jnp.abs(n @ charges) / jnp.maximum(
                n @ jnp.abs(charges), jnp.finfo(n.dtype).tiny
            )
            finite = (
                jnp.all(jnp.isfinite(n) & jnp.isfinite(x), axis=-1)
                & jnp.isfinite(mmw)
                & (mmw > 0)
            )
            conserved = (element_error <= conservation_rtol) & (
                charge_error <= conservation_rtol
            )
            status = jnp.where(
                ~finite,
                NONFINITE,
                jnp.where(
                    ~diagnostics["converged"],
                    NOT_CONVERGED,
                    jnp.where(~conserved, CONSERVATION_FAILED, SUCCESS),
                ),
            ).astype(jnp.int32)
            number_density = bar_cgs * pressure / (kB * temperature)
            z = jnp.sum(jnp.where(metal_mask, b * element_masses, 0.0)) / jnp.sum(
                b * element_masses
            )
            return ChemistryState(
                n,
                x,
                mmw,
                x * masses / mmw[:, None],
                x[:, electron_index] * number_density,
                x[:, hydrogen_index] * number_density,
                b,
                z,
                ChemistryDiagnostics(
                    status,
                    status == SUCCESS,
                    diagnostics["converged"],
                    in_domain,
                    element_error,
                    charge_error,
                    diagnostics["n_iter"],
                    diagnostics["final_residual"],
                ),
            )

        def invalid(_):
            layer_nan = jnp.full_like(pressure, jnp.nan)
            species_nan = jnp.full(
                (pressure.size, len(species)), jnp.nan, dtype=pressure.dtype
            )
            false = jnp.zeros(pressure.shape, dtype=bool)
            return ChemistryState(
                species_nan,
                species_nan,
                layer_nan,
                species_nan,
                layer_nan,
                layer_nan,
                jnp.full_like(reference, jnp.nan),
                jnp.asarray(jnp.nan, dtype=pressure.dtype),
                ChemistryDiagnostics(
                    jnp.full(pressure.shape, OUTSIDE_DOMAIN, dtype=jnp.int32),
                    false,
                    false,
                    in_domain,
                    layer_nan,
                    layer_nan,
                    jnp.zeros(
                        pressure.shape,
                        dtype=jnp.int64 if jax.config.x64_enabled else jnp.int32,
                    ),
                    layer_nan,
                ),
            )

        return jax.lax.cond(in_domain, solve, invalid, operand=None)

    return PreparedChemistry(
        species,
        elements,
        jnp.asarray(matrix),
        masses,
        element_masses,
        charges,
        isotope_convention,
        pressure,
        tuple(temperature_range),
        tuple(pressure_range),
        setup.hvector_func,
        elemental_abundances,
        evaluate,
    )


def hminus_optical_depth(
    nu_grid,
    temperature,
    pressure_bar,
    delta_pressure_bar,
    gravity,
    chemistry: ChemistryState,
):
    """H-minus optical depth from valid chemistry, with explicit layer axes.

    The continuum API consumes electron and atomic H number densities. Do not
    multiply this result by the equilibrium H-minus abundance a second time.
    Gravity is in cm s^-2 and may be scalar or one value per layer.
    """
    from exojax.database.hminus import log_hminus_continuum

    temperature = jnp.asarray(temperature)
    log_absorption = log_hminus_continuum(
        nu_grid, temperature, chemistry.number_density_e, chemistry.number_density_h
    )
    path_length = (
        kB
        * temperature
        / (m_u * chemistry.mmw * gravity)
        * (delta_pressure_bar / pressure_bar)
    )
    return 10.0**log_absorption * path_length[:, None]

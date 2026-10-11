"""WASP-18b column assembly with explicit, externally prepared opacity assets.

The gas chemistry, entropy, CKD mixing and transfer solvers are existing APIs.
This example does not certify an opacity inventory or the retrieval prior.
Flux densities are per cm^-1 in cgs; abundance fractions include electrons.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from _rce_broadband import (
    Fluxes, absorption_fluxes, band_stellar_flux, ckd_absorption_fluxes,
    integrate_broadband_fluxes, irradiation_flux, make_band_quadrature,
)
from _rce_chemistry import prepare_chemistry
from _rce_entropy import make_entropy_evaluator, prepare_equilibrium_entropy
from _rce_observation import pixel_flux_ratio
from _rce_wasp18b_scattering import (
    scattering_layer_depth, stellar_scattering_fluxes, validate_scattering_policy,
    validate_scattering_wavenumbers,
)
from exojax.atm.rce_device import make_device_rce_solver
from exojax.atm.rce_device_implicit import make_device_implicit_rce_solver
from exojax.database.hminus import bound_free_absorption, free_free_absorption
from exojax.opacity.ckd.mixing import mix_ckd_rorr, validate_ckd_mixture_tables
from exojax.rt.planck import piBarr
from exojax.rt.rtransfer import initialize_gaussian_quadrature
from exojax.utils.constants import bar_cgs, kB, m_u


SIGMA_SB = 5.670374419e-5


@dataclass(frozen=True)
class ColumnSettings:
    """Fixed choices; optional scattering affects stellar photons only."""

    gravity_cgs: float
    internal_temperature: float
    mu0: float = 0.5
    chemistry_epsilon: float = 3e-14
    conservation_rtol: float = 3e-6
    planck_order: int = 8
    nstream: int = 8
    chemistry_conservation_convergence: bool = False
    chemistry_equilibrate_constraints: bool = False
    stellar_scattering: bool = False
    stellar_scattering_outside_range: str = "error"

    def __post_init__(self):
        values = (self.gravity_cgs, self.internal_temperature, self.mu0,
                  self.chemistry_epsilon, self.conservation_rtol)
        if not np.all(np.isfinite(values)) or min(values) <= 0 or self.mu0 > 1:
            raise ValueError("Use positive finite column settings and 0 < mu0 <= 1.")
        if self.nstream < 2 or self.nstream % 2:
            raise ValueError("nstream must be a positive even number >= 2.")
        for name in ("chemistry_conservation_convergence", "chemistry_equilibrate_constraints",
                     "stellar_scattering"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        validate_scattering_policy(self.stellar_scattering_outside_range)


def continuum_optical_depth(nu, temperature, pressure, boundaries, chemistry,
                            species, gravity, cia):
    """H-minus and CIA using the shared chemistry and layer-dependent mass.

    ``cia`` maps two chemical species names to prepared CdbCIA objects. Its
    coefficients are explicitly zero outside the tabulated spectral interval;
    this truncation must be included in the run's opacity omission audit.
    Temperature clamping is forbidden by the column's domain guard.
    """
    density = pressure * bar_cgs / (kB * temperature)
    distance = (jnp.diff(boundaries) * bar_cgs
                / (gravity * m_u * chemistry.mmw[:-1] * density))
    wavelength = 1e4 / jnp.asarray(nu)
    # Evaluate coefficients directly so an exactly absent electron/H population
    # has zero opacity and finite derivatives, without log(0).
    coefficient = jax.vmap(
        lambda t: bound_free_absorption(wavelength, t)
        + free_free_absorption(wavelength, t)
    )(temperature)
    extinction = (coefficient * (chemistry.number_density_e[:-1] * kB
                                 * temperature)[:, None]
                  * chemistry.number_density_h[:-1, None])
    for pair, table in cia.items():
        # Work in coefficient space throughout: a tabulated exact zero must
        # have a finite temperature derivative, without logarithmic 0/0 or an
        # invented opacity floor. The column guard checks the temperature domain.
        coefficients = 10.0 ** table.logac
        upper = jnp.clip(jnp.searchsorted(table.tcia, temperature, side="right"),
                         1, len(table.tcia) - 1)
        lower = upper - 1
        weight = ((temperature - table.tcia[lower])
                  / (table.tcia[upper] - table.tcia[lower]))[:, None]
        at_temperature = ((1.0 - weight) * coefficients[lower]
                          + weight * coefficients[upper])
        ac = jax.vmap(lambda values: jnp.interp(
            nu, table.nucia, values, left=0.0, right=0.0,
        ))(at_temperature)
        first = density * chemistry.x[:-1, species.index(pair[0])]
        second = density * chemistry.x[:-1, species.index(pair[1])]
        extinction += ac * (first * second)[:, None]
    return extinction * distance[:, None]


def _response_resolutions(resolving_power, pixel_nu):
    """Validate static scalar or per-pixel resolving powers without reshaping."""
    resolution = np.asarray(resolving_power, dtype=float)
    shape = np.shape(pixel_nu)
    if len(shape) != 1 or not shape[0]:
        raise ValueError("Response pixels must be a nonempty one-dimensional array.")
    if resolution.ndim != 0 and resolution.shape != shape:
        raise ValueError("Response resolving_power must be scalar or match pixel_nu shape.")
    if not np.all(np.isfinite(resolution) & (resolution > 0)):
        raise ValueError("Invalid response resolving power: use finite positive values.")
    return np.broadcast_to(resolution, shape)


def response_samples(nu, flux, pixel_nu, resolving_power=None):
    """Sample a resolved spectrum, optionally with a Gaussian response trial.

    A supplied resolving power is a sensitivity assumption, not an observed
    SOSS LSF. Both planet and star use the same normalized response in Fnu.
    The Gaussian is integrated analytically against the piecewise-linear
    spectrum, including every native sample inside +/-4 sigma. Coordinates
    and resolving power are static host inputs; flux remains differentiable.
    Resolving power may be scalar or a vector aligned with ``pixel_nu``.
    """
    if resolving_power is None:
        return jnp.interp(pixel_nu, nu, flux)
    from scipy.special import ndtr

    nu, pixel_nu = np.asarray(nu), np.asarray(pixel_nu)
    resolutions = _response_resolutions(resolving_power, pixel_nu)
    stencils = []
    for center, resolution in zip(pixel_nu, resolutions):
        sigma = center / (resolution * np.sqrt(8 * np.log(2)))
        lower, upper = center - 4 * sigma, center + 4 * sigma
        if lower < nu[0] or upper > nu[-1]:
            raise ValueError("Resolved spectrum does not cover the Gaussian response.")
        first = max(0, np.searchsorted(nu, lower, side="right") - 1)
        last = min(len(nu) - 1, np.searchsorted(nu, upper, side="left"))
        indices = np.arange(first, last + 1)
        left, right = nu[indices[:-1]], nu[indices[1:]]
        za = (np.maximum(left, lower) - center) / sigma
        zb = (np.minimum(right, upper) - center) / sigma
        integral = ndtr(zb) - ndtr(za)
        moment = sigma / np.sqrt(2 * np.pi) * (np.exp(-za**2 / 2) - np.exp(-zb**2 / 2))
        weights = np.zeros(len(indices))
        weights[:-1] += ((right - center) * integral - moment) / (right - left)
        weights[1:] += ((center - left) * integral + moment) / (right - left)
        stencils.append((indices, weights / weights.sum()))
    size = max(len(indices) for indices, _ in stencils)
    indices = np.zeros((len(stencils), size), dtype=np.int32)
    weights = np.zeros((len(stencils), size))
    for row, (index, weight) in enumerate(stencils):
        indices[row, :len(index)], weights[row, :len(index)] = index, weight
    return jnp.sum(jnp.asarray(flux)[indices] * jnp.asarray(weights), axis=-1)


class Wasp18bColumn:
    """Prepared broadband column and independently resolved observation path.

    Parameters are [log10(metal/H scale), C/O, log(horizontal irradiation f)].
    The incident horizontal flux is exp(p[2]) * Fstar / (a/Rstar)^2. ``mu0``
    controls attenuation only. The lower boundary is a solved blackbody gas
    temperature; the internal temperature specifies the net internal flux.

    CKD continuum is averaged over each band and added equally to its g terms.
    This approximation, fixed broadening and RORR require resolved validation.
    """

    def __init__(self, observation, stellar, ckd, lbl, cia, boundaries, settings,
                 *, isotope_fractions, temperature_initial,
                 bottom_temperature_initial, solver_options=None):
        from exogibbs.thermo.standard import prepare_fastchem_thermodynamics

        if not jax.config.jax_enable_x64:
            raise ValueError("Enable JAX x64 before preparing the column.")
        self.observation, self.stellar, self.settings = observation, stellar, settings
        self.ckd, self.lbl, self.cia = dict(ckd), dict(lbl), dict(cia)
        if set(self.ckd) != set(self.lbl) or not self.ckd:
            raise ValueError("CKD and resolved opacities must have identical nonempty species.")
        if set(cia) != {("H2", "H2"), ("H2", "He1")}:
            raise ValueError("Provide both H2-H2 and H2-He CIA tables.")
        validate_ckd_mixture_tables(self.ckd.values())
        boundaries = np.asarray(boundaries, dtype=float)
        if (boundaries.ndim != 1 or boundaries.size < 3
                or not np.all(np.isfinite(boundaries) & (boundaries > 0))
                or np.any(np.diff(boundaries) <= 0)):
            raise ValueError("Use increasing positive pressure boundaries for >=2 layers.")
        self.boundaries = jnp.asarray(boundaries)
        self.pressure = jnp.sqrt(self.boundaries[:-1] * self.boundaries[1:])
        self.nodes = jnp.append(self.pressure, self.boundaries[-1])
        first = next(iter(self.ckd.values())).ckd_info
        self.band_edges, self.g_weights = first.band_edges, first.weights
        if not np.allclose(np.asarray(self.band_edges)[1:, 0],
                           np.asarray(self.band_edges)[:-1, 1], rtol=1e-12, atol=0):
            raise ValueError("Broadband CKD bands must be contiguous without gaps or overlaps.")
        self.band_nodes, self.band_weights = make_band_quadrature(
            self.band_edges, order=settings.planck_order
        )
        self.band_widths = jnp.diff(self.band_edges, axis=1).ravel()
        self.star_bands = jnp.asarray(band_stellar_flux(
            stellar.nu_cm1, stellar.surface_flux_per_cm1, self.band_edges
        ))
        self.nu = jnp.asarray(next(iter(self.lbl.values())).nu_grid)
        if (self.nu.ndim != 1 or self.nu.size < 2
                or not np.all(np.isfinite(self.nu) & (np.asarray(self.nu) > 0))
                or np.any(np.diff(self.nu) <= 0)):
            raise ValueError("Use an increasing positive resolved spectral grid.")
        stellar.validate_coverage(np.asarray(self.nu))
        if settings.stellar_scattering:
            validate_scattering_wavenumbers(
                self.band_edges, outside_range=settings.stellar_scattering_outside_range)
            validate_scattering_wavenumbers(
                self.nu, outside_range=settings.stellar_scattering_outside_range)
        self.star_resolved = jnp.asarray(stellar.at(np.asarray(self.nu)))
        provider = prepare_fastchem_thermodynamics()
        tmin, tmax = provider.temperature_range
        for name, opa in self.ckd.items():
            info = opa.ckd_info
            if (self.pressure[0] < np.min(info.P_grid)
                    or self.pressure[-1] > np.max(info.P_grid)):
                raise ValueError(f"CKD pressure coverage is insufficient: {name}")
            tmin, tmax = max(tmin, float(np.min(info.T_grid))), min(tmax, float(np.max(info.T_grid)))
            resolved = self.lbl[name]
            if not np.array_equal(resolved.nu_grid, self.nu):
                raise ValueError("Resolved opacity grids must agree exactly.")
            resolved.check_pressure_grid(np.asarray(self.pressure))
            tmin = max(tmin, float(np.min(resolved.diffgrid_info.temperature_grid)))
            tmax = min(tmax, float(np.max(resolved.diffgrid_info.temperature_grid)))
        for table in cia.values():
            tgrid, ngrid, coefficients = map(np.asarray, (table.tcia, table.nucia, table.logac))
            if (tgrid.ndim != 1 or ngrid.ndim != 1 or min(tgrid.size, ngrid.size) < 2
                    or not np.all(np.isfinite(tgrid) & (tgrid > 0))
                    or not np.all(np.isfinite(ngrid) & (ngrid > 0))
                    or np.any(np.diff(tgrid) <= 0) or np.any(np.diff(ngrid) <= 0)
                    or coefficients.shape != (tgrid.size, ngrid.size)
                    or not np.all(np.isfinite(coefficients) | np.isneginf(coefficients))):
                raise ValueError("CIA requires ordered finite grids and nonnegative coefficients.")
            tmin, tmax = max(tmin, float(np.min(table.tcia))), min(tmax, float(np.max(table.tcia)))
        if tmin >= tmax:
            raise ValueError("Thermochemistry and opacity temperature domains do not overlap.")
        self.temperature_range = (tmin, tmax)
        self.chemistry = prepare_chemistry(
            provider.chemical_setup, self.nodes,
            element_masses_u=provider.element_masses_u,
            isotope_convention=provider.isotope_convention,
            required_species=tuple(self.ckd) + ("H2", "He1"),
            electron_species="e1-", atomic_hydrogen_species="H1",
            temperature_range=self.temperature_range,
            pressure_range=(boundaries[0], boundaries[-1]),
            epsilon_crit=settings.chemistry_epsilon,
            conservation_rtol=settings.conservation_rtol,
            conservation_convergence=settings.chemistry_conservation_convergence,
            equilibrate_constraints=settings.chemistry_equilibrate_constraints,
        )
        if set(isotope_fractions) != set(self.ckd) or any(
            not np.isfinite(v) or not 0 < v <= 1 for v in isotope_fractions.values()
        ):
            raise ValueError("Supply an explicit isotope fraction in (0,1] for each absorber.")
        self.isotope_fractions = dict(isotope_fractions)
        self.entropy = prepare_equilibrium_entropy(self.chemistry, provider, entropy_scale=1e4)
        self.mus, self.angular_weights = initialize_gaussian_quadrature(settings.nstream)
        self.area_ratio = float(observation.recipe["radius_ratio"]) ** 2
        self.a_over_rstar = float(observation.recipe["a_over_rstar"])
        if not np.isfinite(self.area_ratio) or not 0 < self.area_ratio < 1:
            raise ValueError("Invalid nominal area ratio in the observation recipe.")
        if not np.isfinite(self.a_over_rstar) or self.a_over_rstar <= 1:
            raise ValueError("Invalid nominal a/Rstar in the observation recipe.")
        selected = np.any(observation.W, axis=0)
        self.W = jnp.asarray(observation.W[:, selected])
        self.pixel_nu = jnp.asarray(1e4 / observation.input_wavelength_um[selected])
        self.check_response_coverage(None)
        if np.any(np.asarray(self.W) @ stellar.at(np.asarray(self.pixel_nu)) <= 0):
            raise ValueError("Each observed bin requires a positive stellar surface flux sum.")
        self.internal_flux = SIGMA_SB * settings.internal_temperature**4
        self.evaluate = make_entropy_evaluator(
            self.entropy, self.net_flux, lambda p: (p[0], p[1])
        )
        self.solver_arguments = (
            self.pressure, self.boundaries, temperature_initial,
            bottom_temperature_initial, self.internal_flux, self.evaluate,
        )
        self.solver_options = dict(flux_atol=0.1, flux_rtol=0.0, stability_atol=1e-8)
        self.solver_options.update(solver_options or {})
        self.solve = make_device_rce_solver(
            *self.solver_arguments, valid_temperature=self.temperature_valid,
            **self.solver_options,
        )

    def temperature_valid(self, temperature, bottom, parameters):
        nodes = jnp.append(temperature, bottom)
        lo, hi = self.temperature_range
        return (jnp.all(jnp.isfinite(nodes) & (nodes >= lo) & (nodes <= hi))
                & jnp.all(jnp.isfinite(parameters)) & (parameters[1] > 0)
                & jnp.isfinite(jnp.exp(parameters[2])))

    def make_implicit_solver(self, local_smoothness):
        """Connect P4 after the caller audits interpolation/RORR branch margins.

        Finite Jacobians alone cannot certify a sorting tie or table knot.
        The default P7 runner consequently reports primal solutions only.
        """
        if not callable(local_smoothness):
            raise ValueError("Supply a tested local_smoothness predicate for implicit AD.")
        return make_device_implicit_rce_solver(
            *self.solver_arguments, valid_temperature=self.temperature_valid,
            local_smoothness=local_smoothness, **self.solver_options,
        )

    def resolved_opacity_tables(self):
        """Expose large Diffgrid arrays as explicit compiled-call arguments."""
        return {name: (opa.diffgrid_info.inverse_temperature_grid,
                       opa.diffgrid_info.log_cross_section_grid,
                       opa.diffgrid_info.log_cross_section_derivative_grid,
                       opa.diffgrid_info.log_cross_section_floor)
                for name, opa in self.lbl.items()}

    def optical_depth(self, temperature, chemistry, *, resolved=False, resolved_tables=None):
        column = (jnp.diff(self.boundaries) * bar_cgs
                  / (self.settings.gravity_cgs * m_u * chemistry.mmw[:-1]))
        depths = []
        for name, opa in (self.lbl if resolved else self.ckd).items():
            abundance = chemistry.x[:-1, self.chemistry.species.index(name)]
            amount = column * abundance * self.isotope_fractions[name]
            if resolved and resolved_tables is not None:
                from exojax.opacity.diffgrid.core import cross_section_matrix

                xs = cross_section_matrix(temperature, *resolved_tables[name])
            else:
                xs = (opa.xsmatrix(temperature) if resolved
                      else opa.xstensor_ckd(temperature, self.pressure))
            depths.append(xs * amount.reshape((-1,) + (1,) * (xs.ndim - 1)))
        nu = self.nu if resolved else self.band_nodes.ravel()
        continuum = continuum_optical_depth(
            nu, temperature, self.pressure, self.boundaries, chemistry,
            self.chemistry.species, self.settings.gravity_cgs, self.cia,
        )
        if resolved:
            depth = sum(depths) + continuum
        else:
            continuum = continuum.reshape((temperature.size,) + self.band_nodes.shape)
            averaged = jnp.sum(continuum * self.band_weights, axis=-1)
            depth = mix_ckd_rorr(jnp.stack(depths), self.g_weights) + averaged[:, None, :]
        return jnp.where(jnp.isfinite(depth) & (depth >= 0), depth, jnp.nan)

    def fluxes(self, temperature, bottom, parameters, chemistry, *, resolved=False,
               resolved_tables=None):
        """Return common boundaries, optionally scattering stellar photons only.

        Reject invalid states before evaluating any clamping opacity lookup.
        Thermal-photon scattering remains omitted when the option is enabled.
        """
        def calculate(inputs):
            temperature, bottom, parameters, chemistry = inputs
            dtau = self.optical_depth(temperature, chemistry, resolved=resolved,
                                     resolved_tables=resolved_tables)
            star = self.star_resolved if resolved else self.star_bands[None, :]
            incident = irradiation_flux(star, self.a_over_rstar, jnp.exp(parameters[2]))
            common = dict(mus=self.mus, angular_weights=self.angular_weights,
                          incident_flux=incident, mu0=self.settings.mu0)
            if resolved:
                thermal = absorption_fluxes(
                    dtau, temperature, bottom, self.pressure, self.boundaries,
                    lambda t: piBarr(t, self.nu), **common,
                )
            else:
                thermal = ckd_absorption_fluxes(
                    dtau, temperature, bottom, self.pressure, self.boundaries,
                    self.band_nodes, self.band_weights, **common,
                )
            if not self.settings.stellar_scattering:
                return thermal
            # Only stellar photons scatter; retain the existing thermal source
            # reconstruction and angular quadrature at the absorption depth.
            nu = self.nu if resolved else self.band_nodes
            scattering = scattering_layer_depth(
                nu, chemistry, self.boundaries, self.settings.gravity_cgs,
                self.chemistry.species,
                outside_range=self.settings.stellar_scattering_outside_range,
            )
            if not resolved:
                scattering = jnp.sum(scattering * self.band_weights, axis=-1)[:, None, :]
            stellar = stellar_scattering_fluxes(dtau, scattering, incident, self.settings.mu0)
            upward = thermal.upward + stellar.upward
            downward = thermal.downward + stellar.downward
            return Fluxes(upward, downward, stellar.incident,
                          upward - downward - stellar.incident)

        spectral_shape = ((self.nu.size,) if resolved
                          else (self.g_weights.size, self.band_widths.size))
        invalid = jnp.full((self.pressure.size + 1,) + spectral_shape, jnp.nan)
        valid = (self.temperature_valid(temperature, bottom, parameters)
                 & jnp.all(chemistry.diagnostics.valid))
        return jax.lax.cond(
            valid, calculate, lambda _: Fluxes(invalid, invalid, invalid, invalid),
            (temperature, bottom, parameters, chemistry),
        )

    def net_flux(self, temperature, bottom, parameters, chemistry):
        return integrate_broadband_fluxes(
            self.fluxes(temperature, bottom, parameters, chemistry),
            self.g_weights, self.band_widths,
        ).net

    def check_response_coverage(self, resolving_power):
        pixel_nu = np.asarray(self.pixel_nu)
        padding = 0.0 if resolving_power is None else 4 / (
            _response_resolutions(resolving_power, pixel_nu) * np.sqrt(8 * np.log(2))
        )
        if not np.all(np.isfinite(padding) & (padding >= 0) & (padding < 1)):
            raise ValueError("Invalid response resolving power.")
        nu = np.asarray(self.nu)
        needed = np.array([np.min(pixel_nu * (1 - padding)),
                           np.max(pixel_nu * (1 + padding))])
        if needed[0] < nu[0] or needed[1] > nu[-1]:
            raise ValueError("Resolved spectrum does not cover selected pixels and response.")
        self.stellar.validate_coverage(needed)

    def spectrum(self, temperature, bottom, parameters, chemistry, *, resolving_power=None,
                 resolved_tables=None):
        """Apply P1 to the resolved emergent planet and stellar surface spectra."""
        self.check_response_coverage(resolving_power)
        up = self.fluxes(temperature, bottom, parameters, chemistry, resolved=True,
                         resolved_tables=resolved_tables).upward[0]
        return pixel_flux_ratio(
            self.W, response_samples(self.nu, up, self.pixel_nu, resolving_power),
            response_samples(self.stellar.nu_cm1, self.stellar.surface_flux_per_cm1,
                             self.pixel_nu, resolving_power),
            self.area_ratio,
        )


def spectral_error(predicted, reference, uncertainty):
    """Quantify numerical changes using the P7 observation-space budget."""
    predicted, reference, uncertainty = map(np.asarray, (predicted, reference, uncertainty))
    if (predicted.shape != reference.shape or uncertainty.shape != reference.shape
            or not np.all(np.isfinite(predicted) & np.isfinite(reference))
            or not np.all(np.isfinite(uncertainty) & (uncertainty > 0))):
        raise ValueError("Compare finite matching spectra and positive uncertainties.")
    difference = (predicted - reference) / uncertainty
    maximum, norm = float(np.max(np.abs(difference))), float(np.linalg.norm(difference))
    return {"max_sigma": maximum, "weighted_norm": norm,
            "passed": maximum < 0.1 and norm < 0.3}

"""Small RCE/TCE coupling experiment with synthetic H2O/CO opacity.

The 538-species thermochemistry is physical; the smooth cross sections, star,
geometry and noise are deliberately synthetic. This is not a WASP-18b model.
Requires the optional ExoGibbs standard-state provider and JAX x64.
"""

import jax
import jax.numpy as jnp
import numpy as np

from _rce_chemistry import prepare_chemistry
from _rce_entropy import make_entropy_evaluator, prepare_equilibrium_entropy
from _rce_observation import pixel_flux_ratio
from exojax.atm.rce_device_implicit import (
    make_device_implicit_rce_solver,
    make_rce_log_prob,
)
from exojax.rt.flux import (
    direct_beam_fluxes,
    reconstruct_boundary_temperature,
    rtrun_emis_pureabs_ibased_linsap_fluxes,
)
from exojax.rt.planck import piBarr
from exojax.rt.rtransfer import initialize_gaussian_quadrature
from exojax.utils.constants import bar_cgs, m_u


class SyntheticColumn:
    """Fixed small column shared by free-temperature TCE and implicit RCE.

    RCE parameters are [log_metal_scale, C/O, log irradiation amplitude].
    TCE parameters are [log_metal_scale, C/O, log(T_top/2500), log(T_bottom/2500)];
    log temperature is linear in log pressure, independently of RCE.
    All flux densities are per cm^-1 in cgs; the observation is dimensionless.
    """

    def __init__(self, *, chemistry_epsilon=3e-14, conservation_rtol=3e-6):
        from exogibbs.thermo.standard import prepare_fastchem_thermodynamics

        if not jax.config.jax_enable_x64:
            raise ValueError("Enable JAX x64 before preparing the synthetic column.")
        # Synthetic-experiment tolerances, not a change to the P2 adapter's
        # defaults. The gas solver uses an absolute residual; see the P6 design
        # record for the trace-element precision floor and refinement checks.
        self.chemistry_epsilon = chemistry_epsilon
        self.conservation_rtol = conservation_rtol
        self.boundaries = jnp.geomspace(0.03, 3.0, 4)
        self.pressure = jnp.sqrt(self.boundaries[:-1] * self.boundaries[1:])
        self.nodes = jnp.append(self.pressure, self.boundaries[-1])
        edges = jnp.geomspace(100.0, 40000.0, 97)
        self.nu = jnp.sqrt(edges[:-1] * edges[1:])
        self.widths = jnp.diff(edges)
        self.W = jnp.asarray(np.repeat(np.eye(32), 3, axis=1))
        self.star = piBarr(jnp.array([5800.0]), self.nu)[0]
        self.area_ratio = 0.01
        self.internal_flux = 5.670374419e-5 * 1800.0**4
        self.mus, self.angular_weights = initialize_gaussian_quadrature(4)
        provider = prepare_fastchem_thermodynamics()
        self.chemistry = prepare_chemistry(
            provider.chemical_setup,
            self.nodes,
            element_masses_u=provider.element_masses_u,
            isotope_convention=provider.isotope_convention,
            required_species=("H2O1", "C1O1"),
            electron_species="e1-",
            atomic_hydrogen_species="H1",
            temperature_range=provider.temperature_range,
            pressure_range=(0.03, 3.0),
            epsilon_crit=chemistry_epsilon,
            conservation_rtol=conservation_rtol,
        )
        self.entropy = prepare_equilibrium_entropy(
            self.chemistry, provider, entropy_scale=1e4
        )
        self.absorbers = np.array(
            [self.chemistry.species.index(s) for s in ("H2O1", "C1O1")]
        )
        # Analytic templates, in cm^2/molecule; no external opacity tables.
        wave = 1e4 / self.nu
        self.cross_sections = 2e-23 * jnp.stack(
            [
                0.1
                + jnp.exp(-0.5 * (jnp.log(wave / 1.4) / 0.17) ** 2)
                + 1.5 * jnp.exp(-0.5 * (jnp.log(wave / 2.7) / 0.2) ** 2),
                0.1
                + 2 * jnp.exp(-0.5 * (jnp.log(wave / 4.6) / 0.13) ** 2)
                + jnp.exp(-0.5 * (jnp.log(wave / 2.3) / 0.1) ** 2),
            ]
        )
        self.evaluate = make_entropy_evaluator(
            self.entropy, self.net_flux, lambda p: (p[0], p[1])
        )
        self.solve = make_device_implicit_rce_solver(
            self.pressure,
            self.boundaries,
            jnp.array([2100.0, 2500.0, 3200.0]),
            3700.0,
            self.internal_flux,
            self.evaluate,
            valid_temperature=self.temperature_valid,
            local_smoothness=self.temperature_valid,
            flux_atol=0.01,
            flux_rtol=0.0,
            stability_atol=1e-9,
            invalid_derivative="zero",
        )

    @staticmethod
    def temperature_valid(t, tb, p):
        nodes = jnp.append(t, tb)
        return jnp.all(jnp.isfinite(nodes) & (nodes > 1500.0) & (nodes < 4500.0))

    def optical_depth(self, temperature, chemistry):
        cross_section = chemistry.x[:-1, self.absorbers] @ self.cross_sections
        # A weak synthetic continuum per particle keeps every channel absorbing.
        cross_section = (cross_section + 2e-27) * (temperature[:, None] / 2500.0) ** 0.5
        cross_section *= self.pressure[:, None]
        column = (
            jnp.diff(self.boundaries) * bar_cgs / (1000.0 * m_u * chemistry.mmw[:-1])
        )
        return cross_section * column[:, None]

    def thermal_flux(self, temperature, bottom_temperature, chemistry):
        boundary_t = reconstruct_boundary_temperature(
            self.pressure, self.boundaries, temperature, bottom_temperature
        )
        dtau = self.optical_depth(temperature, chemistry)
        up, down = rtrun_emis_pureabs_ibased_linsap_fluxes(
            dtau,
            piBarr(boundary_t, self.nu),
            self.mus,
            self.angular_weights,
            source_center=piBarr(temperature, self.nu),
            upper_fraction=(self.pressure - self.boundaries[:-1])
            / jnp.diff(self.boundaries),
        )
        return up, down, dtau

    def net_flux(self, temperature, bottom_temperature, parameters, chemistry):
        up, down, dtau = self.thermal_flux(temperature, bottom_temperature, chemistry)
        incident = jnp.exp(parameters[2]) * (1800.0 / 5800.0) ** 4 * self.star
        beam = direct_beam_fluxes(dtau, incident, 0.5)
        return (up - down - beam) @ self.widths

    def spectrum(self, temperature, bottom_temperature, chemistry):
        up, _, _ = self.thermal_flux(temperature, bottom_temperature, chemistry)
        return pixel_flux_ratio(self.W, up[0], self.star, self.area_ratio)

    def tce_nodes(self, parameters):
        fraction = jnp.log(self.nodes / self.nodes[0]) / jnp.log(
            self.nodes[-1] / self.nodes[0]
        )
        return 2500.0 * jnp.exp(
            parameters[2] + fraction * (parameters[3] - parameters[2])
        )

    def tce_spectrum(self, parameters):
        nodes = self.tce_nodes(parameters)
        chemistry = self.chemistry(nodes, parameters[0], parameters[1])
        return jax.lax.cond(
            jnp.all(chemistry.diagnostics.valid),
            lambda: self.spectrum(nodes[:-1], nodes[-1], chemistry),
            lambda: jnp.full(self.W.shape[0], jnp.nan),
        )

    def rce_spectrum_from_state(self, state, parameters):
        nodes = jnp.append(state.temperature, state.bottom_temperature)
        chemistry = self.chemistry(nodes, parameters[0], parameters[1])
        return jax.lax.cond(
            jnp.all(chemistry.diagnostics.valid),
            lambda: self.spectrum(nodes[:-1], nodes[-1], chemistry),
            lambda: jnp.full(self.W.shape[0], jnp.nan),
        )

    def rce_spectrum(self, parameters):
        result = self.solve(parameters)
        return jax.lax.cond(
            result.derivative_valid,
            lambda: self.rce_spectrum_from_state(result.state, parameters),
            lambda: jnp.full(self.W.shape[0], jnp.nan),
        )

    def log_prob(self, mode, observed, sigma, lower, upper):
        """Diagonal Normal likelihood with an explicit uniform box prior.

        Returns (value, diagnostics); TCE diagnostics is a LogProbStatus integer,
        while RCE retains the detailed P4 diagnostics. The box density is
        constant and its additive normalization is omitted.
        """
        if mode not in ("rce", "tce"):
            raise ValueError("mode must be 'rce' or 'tce'")
        observed, sigma = np.asarray(observed), np.asarray(sigma)
        lower, upper = np.asarray(lower), np.asarray(upper)
        if (
            observed.shape != (self.W.shape[0],)
            or sigma.shape != observed.shape
            or not np.all(np.isfinite(observed))
            or not np.all(np.isfinite(sigma) & (sigma > 0))
        ):
            raise ValueError("Use finite bin observations and matching positive sigma.")
        nparameter = 3 if mode == "rce" else 4
        if (
            lower.shape != (nparameter,)
            or upper.shape != lower.shape
            or not np.all(np.isfinite(lower) & np.isfinite(upper) & (lower < upper))
        ):
            raise ValueError("Use finite ordered bounds for every parameter.")
        observed, sigma = jnp.asarray(observed), jnp.asarray(sigma)
        lower, upper = jnp.asarray(lower), jnp.asarray(upper)
        prior_valid = lambda p: jnp.all(jnp.isfinite(p) & (p >= lower) & (p <= upper))

        def density(predicted):
            return -0.5 * jnp.sum(
                ((predicted - observed) / sigma) ** 2 + jnp.log(2 * jnp.pi * sigma**2)
            )

        if mode == "rce":
            return make_rce_log_prob(
                self.solve,
                lambda state, p: density(self.rce_spectrum_from_state(state, p)),
                prior_valid=prior_valid,
            )

        def log_prob(p):
            def within_prior():
                nodes = self.tce_nodes(p)
                chemistry = self.chemistry(nodes, p[0], p[1])
                valid = jnp.all(chemistry.diagnostics.valid)
                return jax.lax.cond(
                    valid,
                    lambda: (
                        density(self.spectrum(nodes[:-1], nodes[-1], chemistry)),
                        jnp.int32(0),
                    ),
                    lambda: (-jnp.inf, jnp.int32(2)),
                )

            return jax.lax.cond(
                prior_valid(p), within_prior, lambda: (-jnp.inf, jnp.int32(1))
            )

        return jax.jit(log_prob)

"""Shared stellar-only scattering for the P7 column and fixed-column audit.

Thermal photons retain the existing absorption-only transfer. Rayleigh and
Thomson scattering use g=0 two-stream closures without polarization or higher
phase-function moments; their coefficient errors are not certified here.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from _rce_broadband import Fluxes, integrate_broadband_fluxes
from exojax.rt.direct_sfm import _direct_layer_sources
from exojax.rt.flux import direct_beam_fluxes
from exojax.rt.toon import params_hemispheric_mean, params_quadrature
from exojax.rt.twostream import (
    set_scat_trans_absorption_coeffs, solve_fluxadding_twostream_fluxes,
)
from exojax.utils.constants import bar_cgs, m_u


SCATTERING_SOURCES = {
    "H1": "https://adsabs.harvard.edu/pdf/1964ApJ...140.1381S (equation 15)",
    "H2": "https://research-repository.st-andrews.ac.uk/bitstream/10023/10952/1/Lee_2017_A_A_DynamicMineralClouds_FinalPubVersion.pdf (equation 27)",
    "He1": "exojax.atm.polarizability: static He polarizability 0.21e-24 cm^3",
    "e1-": "https://physics.nist.gov/cgi-bin/cuu/Value?sigmae",
}
SCATTERING_RANGE_CM1 = (200.0, 50000.0)


def validate_scattering_policy(outside_range):
    if not isinstance(outside_range, str) or outside_range not in ("error", "zero"):
        raise ValueError("Scattering outside_range must be 'error' or 'zero'.")


def validate_scattering_wavenumbers(nu_cm1, *, outside_range="error"):
    """Check positive coordinates and the explicitly selected support policy."""
    validate_scattering_policy(outside_range)
    nu = np.asarray(nu_cm1, dtype=float)
    if not nu.size or not np.all(np.isfinite(nu) & (nu > 0)):
        raise ValueError("Scattering requires positive finite wavenumbers.")
    if outside_range == "error" and not np.all(
            (nu >= SCATTERING_RANGE_CM1[0]) & (nu <= SCATTERING_RANGE_CM1[1])):
        raise ValueError("Scattering requires 200 <= wavenumber <= 50000 cm^-1.")


def _cross_sections(nu_cm1, xp, *, outside_range="error"):
    """Keep host audit and traced transfer on identical coefficient formulas."""
    validate_scattering_policy(outside_range)
    nu = xp.asarray(nu_cm1, dtype=float)
    if outside_range == "zero":
        support = (nu >= SCATTERING_RANGE_CM1[0]) & (nu <= SCATTERING_RANGE_CM1[1])
        # Evaluate the polynomial only at supported coordinates, then set all
        # four areas outside support to zero. This is not UV extrapolation.
        nu = xp.where(support, nu, SCATTERING_RANGE_CM1[0])
    wavelength_angstrom = 1e8 / nu
    h = (5.799e-13 / wavelength_angstrom**4
         + 1.422e-6 / wavelength_angstrom**6
         + 2.784 / wavelength_angstrom**8)
    h2 = (8.14e-13 / wavelength_angstrom**4
          + 1.28e-6 / wavelength_angstrom**6
          + 1.61 / wavelength_angstrom**8)
    he = (128.0 * np.pi**5 / 3.0) * (0.21e-24)**2 * nu**4
    electron = xp.full_like(nu, 6.6524587321e-25)
    areas = dict(H1=h, H2=h2, He1=he, **{"e1-": electron})
    if outside_range == "zero":
        areas = {name: xp.where(support, area, 0.) for name, area in areas.items()}
    return areas


def scattering_cross_sections(nu_cm1, *, outside_range="error"):
    """Return checked host H/H2/He Rayleigh and Thomson areas in cm^2.

    Static-He dispersion and H/H2 polynomial-fit errors remain unquantified.
    These specified coefficients are not upper envelopes of physical opacity.
    Explicit zero support is a finite model, not a bound on omitted scattering.
    """
    validate_scattering_wavenumbers(nu_cm1, outside_range=outside_range)
    return _cross_sections(nu_cm1, np, outside_range=outside_range)


def scattering_layer_depth(nu, chemistry, boundaries, gravity_cgs, species,
                           *, outside_range="error"):
    """Evaluate traced depths using fractions of all particles, including e-.

    The column validates spectral coverage before tracing. Hydrostatic gas
    mass and chemical abundances stay at the physical layer pressures.
    """
    areas = _cross_sections(nu, jnp, outside_range=outside_range)
    column = jnp.diff(boundaries) * bar_cgs / (gravity_cgs * m_u * chemistry.mmw[:-1])
    shape = (-1,) + (1,) * jnp.ndim(nu)
    return sum((column * chemistry.x[:-1, species.index(name)]).reshape(shape) * area
               for name, area in areas.items())


@jax.custom_jvp
def _single_scattering_albedo(absorption, scattering):
    """Return the albedo without changing the zero-total-depth convention."""
    total = absorption + scattering
    return scattering / jnp.where(total > 0., total, 1.)


@_single_scattering_albedo.defjvp
def _single_scattering_albedo_jvp(primals, tangents):
    absorption, scattering = primals
    d_absorption, d_scattering = tangents
    total = absorption + scattering
    denominator = jnp.where(total > 0., total, 1.)
    omega = _single_scattering_albedo(absorption, scattering)
    d_total = jnp.where(total > 0., d_absorption + d_scattering, 0.)
    # This quotient rule avoids squaring tiny positive opacity floors. It also
    # retains sensitivity to adding scattering when the current albedo is zero.
    return omega, (d_scattering - omega * d_total) / denominator


@partial(jax.jit, static_argnames=("closure",))
def stellar_scattering_fluxes(absorption_depth, scattering_depth,
                             incident_horizontal, mu0, *, closure="quadrature"):
    """Return stellar-only Fluxes with the P7 horizontal-flux convention.

    Nonnegative depths broadcast to (Nlayer, ...); incident_horizontal
    broadcasts to the trailing axes. The top has no diffuse incidence, and
    the bottom is black with no stellar source. Downward diffuse and direct
    beams remain separate; net is positive upward. No thermal source is used.
    """
    absorption, scattering = jnp.broadcast_arrays(absorption_depth, scattering_depth)
    total = absorption + scattering
    shape = (total.shape[0] + 1,) + total.shape[1:]
    depth = total.reshape(total.shape[0], -1)
    omega = _single_scattering_albedo(absorption, scattering).reshape(depth.shape)
    if closure == "quadrature":
        gamma1, gamma2, gamma3, _ = params_quadrature(omega, 0., mu0)
    elif closure == "hemispheric_mean":
        gamma1, gamma2, _ = params_hemispheric_mean(omega, 0.)
        gamma3 = .5
    else:
        raise ValueError("Use quadrature or hemispheric_mean closure")
    trans, reflect, absorbed = set_scat_trans_absorption_coeffs(gamma1, gamma2, depth)
    source_up, source_down = _direct_layer_sources(depth, gamma1, gamma2, omega, mu0, gamma3)
    direct = direct_beam_fluxes(total, incident_horizontal, mu0)
    # The layer-source API takes beam-normal irradiance at the layer top.
    beam = direct.reshape(shape[0], -1) / mu0
    upward, downward = solve_fluxadding_twostream_fluxes(
        trans, reflect, jnp.zeros_like(depth), jnp.zeros(depth.shape[1]),
        jnp.zeros(depth.shape[1]), absorption_coeff=absorbed,
        source_plus=source_up * beam[:-1], source_minus=source_down * beam[:-1],
    )
    upward, downward = upward.reshape(shape), downward.reshape(shape)
    return Fluxes(upward, downward, direct, upward - downward - direct)

def integrated_stellar_report(fluxes, g_weights, band_widths):
    """Integrate all boundaries and positive stellar heating in each layer."""
    values = {name: np.asarray(value) for name, value in
              integrate_broadband_fluxes(fluxes, g_weights, band_widths)._asdict().items()}
    if not all(np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError("Nonfinite stellar boundary flux")
    incident = float(values["incident"][0])
    if incident <= 0.:
        raise ValueError("A positive incident stellar flux is required")
    heating = np.diff(values["net"])
    reflected = float(values["upward"][0])
    bottom = float(values["downward"][-1] + values["incident"][-1])
    return dict(boundary_flux_cgs=values, layer_heating_cgs=heating,
        incident_flux_cgs=incident, reflected_fraction=reflected / incident,
        bottom_reaching_fraction=bottom / incident,
        atmosphere_absorbed_fraction=float(heating.sum()) / incident,
        minimum_layer_heating_over_incident=float(np.min(heating)) / incident,
        energy_closure_residual_over_incident=(incident - reflected - bottom - heating.sum()) / incident)

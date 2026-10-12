"""Conditional opacity diagnostics for an already converged P7 column.

These diagnostics do not certify omitted absorbers, thermal scattering, or
changes in the temperature profile after re-equilibration. A chemical abundance
is reported as an inventory entry, never as an opacity-error bound.
"""

import copy

import numpy as np

from exojax.utils.constants import bar_cgs, m_u


from _rce_wasp18b_scattering import SCATTERING_SOURCES, scattering_cross_sections


def first_scattering_probability(absorption_depth, scattering_depth, mu0):
    """Probability that an incident photon scatters before absorption/exit.

    Layer is the first axis; remaining axes broadcast. Homogeneous-layer
    absorption and scattering compete along the same direct ray. The result
    includes the first scattering only, regardless of subsequent events.
    For these coefficients and a fixed atmosphere this bounds the fraction
    of incident energy whose absorption location or escape can change.
    It does not bound thermal emission or an RCE solution's spectral error.
    """
    absorption, scattering = np.broadcast_arrays(
        np.asarray(absorption_depth, dtype=float),
        np.asarray(scattering_depth, dtype=float),
    )
    if (absorption.ndim < 1 or absorption.shape[0] == 0
            or not np.isfinite(mu0) or not 0 < mu0 <= 1
            or not np.all(np.isfinite(absorption) & (absorption >= 0))
            or not np.all(np.isfinite(scattering) & (scattering >= 0))):
        raise ValueError("Use finite nonnegative layer depths and 0 < mu0 <= 1.")
    total = absorption + scattering
    above = np.concatenate((np.zeros_like(total[:1]), np.cumsum(total[:-1], axis=0)))
    albedo = np.divide(scattering, total, out=np.zeros_like(total), where=total > 0)
    per_layer = np.exp(-above / mu0) * albedo * -np.expm1(-total / mu0)
    return np.sum(per_layer, axis=0)


def scattering_optical_depth(column, chemistry):
    """Return each scatterer's band-averaged layer depth on the P7 grid.

    Chemistry includes the bottom node, as in Wasp18bColumn. Cross sections
    use the same band quadrature and hydrostatic column as the continuum.
    """
    if not np.all(np.asarray(chemistry.diagnostics.valid)):
        raise ValueError("The scattering audit requires valid chemistry.")
    amount = (np.diff(np.asarray(column.boundaries)) * bar_cgs
              / (column.settings.gravity_cgs * m_u * np.asarray(chemistry.mmw[:-1])))
    cross_sections = scattering_cross_sections(
        column.band_nodes, outside_range=getattr(column.settings, "stellar_scattering_outside_range", "error"))
    weights = np.asarray(column.band_weights)
    components = {}
    for name, cross_section in cross_sections.items():
        abundance = np.asarray(chemistry.x[:-1, column.chemistry.species.index(name)])
        averaged = np.sum(cross_section * weights, axis=-1)
        components[name] = (amount * abundance)[:, None] * averaged[None, :]
    return components


def stellar_scattering_description(settings):
    """Describe the same finite stellar-scattering model in both runners."""
    return dict(
        closure="quadrature", asymmetry_parameter=0.,
        coefficient_support_cm1=[200., 50000.],
        outside_range=settings.stellar_scattering_outside_range,
        outside_support_physical_error_bounded=False,
        support_scope="Zero outside support, when explicitly selected, defines a finite scattering model for endpoint convergence; it does not bound omitted UV or long-wavelength scattering.",
        thermal_photon_scattering_included=False,
        thermal_scattering_error_bounded=False,
        observed_flux_convention="Thermal plus reflected upward flux uses the existing uniform hemisphere area-ratio approximation; no phase-dependent disk integration.",
        bottom_incident_scope="The incident field tracks the direct beam only; diffuse stellar light reaching the black bottom remains in downward flux.",
    )


def stellar_flux_audit(column, temperature, parameters, chemistry, *, absorption_depth=None):
    """Separate direct and diffuse stellar light at every boundary.

    A precomputed CKD absorption depth avoids recompiling the same opacity
    function for each validation point. Thermal downward flux is excluded.
    """
    import jax
    import jax.numpy as jnp
    from _rce_broadband import irradiation_flux
    from _rce_wasp18b_scattering import integrated_stellar_report, stellar_scattering_fluxes

    absorption = (jax.jit(column.optical_depth)(temperature, chemistry)
                  if absorption_depth is None else absorption_depth)
    scattering = jnp.asarray(sum(scattering_optical_depth(column, chemistry).values()))[:, None, :]
    horizontal = irradiation_flux(column.star_bands[None, :], column.a_over_rstar, jnp.exp(parameters[2]))
    stellar = stellar_scattering_fluxes(absorption, scattering, horizontal, column.settings.mu0)
    report = integrated_stellar_report(stellar, column.g_weights, column.band_widths)
    report["bottom_total_stellar_over_internal_flux"] = (
        report["boundary_flux_cgs"]["downward"][-1]
        + report["boundary_flux_cgs"]["incident"][-1]) / column.internal_flux
    return report


def audit_scattering(column, temperature, chemistry, *, absorption_depth=None):
    """Evaluate the conditional incident-energy bound on the actual CKD grid.

    A supplied absorption depth must be the column's CKD depth at this state.
    This avoids repeating the expensive RORR calculation in paired audits.
    The stellar weights are the column's actual band-integrated spectrum.
    """
    components = scattering_optical_depth(column, chemistry)
    absorption = np.asarray(column.optical_depth(temperature, chemistry)
                            if absorption_depth is None else absorption_depth)
    scattering = sum(components.values())
    probability = first_scattering_probability(
        absorption, scattering[:, None, :], column.settings.mu0,
    )
    spectral_weight = np.asarray(column.star_bands) * np.asarray(column.band_widths)
    if not np.all(np.isfinite(spectral_weight) & (spectral_weight >= 0)) or not spectral_weight.sum() > 0:
        raise ValueError("The scattering audit requires positive finite stellar energy.")
    weighted = np.asarray(column.g_weights) @ probability
    bound = float(weighted @ spectral_weight / spectral_weight.sum())
    return {
        "incident_energy_redistribution_bound_fraction": bound,
        "bound_scope": "Fixed column and declared scattering coefficients; first-scattering probability before absorption or bottom exit.",
        "certifies_rce_or_observation_error": False,
        "max_band_first_scattering_probability": float(np.max(weighted)),
        "max_vertical_scattering_depth": float(np.max(np.sum(scattering, axis=0))),
        "max_vertical_depth_by_scatterer": {
            name: float(np.max(np.sum(depth, axis=0))) for name, depth in components.items()
        },
        "sources": SCATTERING_SOURCES,
        "limitations": [
            "H/H2 polynomial and static-He cross-section errors are not bounded.",
            "Other scatterers, clouds and thermal-photon scattering are excluded.",
            "CKD correlations and band averaging have the same limitations as the baseline column.",
            "Small incident-energy redistribution does not alone bound re-equilibrated temperatures or observed bins.",
        ],
    }


def chemical_opacity_inventory(column, chemistry):
    """Map every chemical species to implemented line/continuum contributions."""
    if not np.all(np.asarray(chemistry.diagnostics.valid)):
        raise ValueError("The inventory requires valid chemistry.")
    x = np.asarray(chemistry.x[:-1])
    rows = []
    for index, name in enumerate(column.chemistry.species):
        roles = []
        if name in column.ckd:
            roles.append("line absorption")
        if name in ("H1", "e1-"):
            roles.append("H-minus continuum input")
        if name in ("H2", "He1"):
            roles.append("CIA input")
        rows.append({"species": name, "absorption_roles": roles,
                     "max_number_fraction": float(np.max(x[:, index])),
                     "min_number_fraction": float(np.min(x[:, index]))})
    return {"species": rows, "omitted_absorption_error_bounded": False,
            "limitation": "An abundance inventory does not bound omitted absorption; absent cross sections are required for a radiative comparison."}


def fixed_column_component_sensitivity(column, temperature, bottom, parameters, chemistry):
    """Remove each included opacity source at fixed temperature/composition.

    These are diagnostics of included sources, not bounds for absent species.
    No alternate RCE solve is performed, and each mixture is recomputed with
    the existing RORR operation. The baseline continuum and boundaries remain
    those of Wasp18bColumn.
    """
    from _rce_broadband import integrate_broadband_fluxes

    if not np.all(np.asarray(chemistry.diagnostics.valid)):
        raise ValueError("Component sensitivity requires valid chemistry.")
    if len(column.ckd) < 2:
        raise ValueError("Component removal requires at least two line absorbers.")

    def flux(model, state):
        spectral = model.fluxes(temperature, bottom, parameters, state)
        broadband = integrate_broadband_fluxes(spectral, model.g_weights, model.band_widths)
        values = {name: np.asarray(value) for name, value in broadband._asdict().items()}
        if not all(np.all(np.isfinite(value)) for value in values.values()):
            raise ValueError("Component sensitivity produced nonfinite fluxes.")
        return values

    baseline = flux(column, chemistry)
    cases = {}
    for species in column.ckd:
        model = copy.copy(column)
        model.ckd = {key: value for key, value in column.ckd.items() if key != species}
        cases[species] = (model, chemistry)
    for pair in column.cia:
        model = copy.copy(column)
        model.cia = {key: value for key, value in column.cia.items() if key != pair}
        cases["CIA:" + ":".join(pair)] = (model, chemistry)
    cases["H-minus"] = (column, chemistry._replace(number_density_e=chemistry.number_density_e * 0))
    output = {}
    for name, (model, state) in cases.items():
        trial = flux(model, state)
        output[name] = {
            "top_thermal_flux_change_cgs": float(trial["upward"][0] - baseline["upward"][0]),
            "max_net_flux_change_cgs": float(np.max(np.abs(trial["net"] - baseline["net"]))),
            "bottom_incident_flux_change_cgs": float(trial["incident"][-1] - baseline["incident"][-1]),
        }
    return {"temperature_and_composition_fixed": True, "removed_sources": output,
            "certifies_omitted_absorption_error": False}

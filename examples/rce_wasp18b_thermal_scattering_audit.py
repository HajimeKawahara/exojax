"""Fixed-column thermal scattering diagnostic with paired hemispheric closure.

The same black bottom, vacuum top, and piecewise constant sources are used
with and without scattering. This is not an eight-stream RCE error bound.
"""

import argparse
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from _rce_broadband import Fluxes, band_planck_flux, irradiation_flux
from _rce_observation import load_emission_candidate
from _rce_wasp18b_assets import _sha256, load_manifest
from _rce_wasp18b_audit import scattering_optical_depth
from _rce_wasp18b_scattering import SCATTERING_SOURCES
from rce_wasp18b_forward import _serializable, prepare_column, provenance
from exojax.rt.flux import reconstruct_boundary_temperature, rtrun_emis_pureabs_ibased_linsap_fluxes
from exojax.rt.toon import params_hemispheric_mean
from exojax.rt.twostream import set_scat_trans_absorption_coeffs, solve_fluxadding_twostream_fluxes


@jax.jit
def thermal_fluxes(absorption_depth, scattering_depth, source, bottom_source, incoming_top=0.):
    """Return thermal-only fluxes with g=0, hemispheric mean, and black bottom."""
    absorption, scattering = jnp.broadcast_arrays(absorption_depth, scattering_depth)
    total = absorption + scattering
    source = jnp.broadcast_to(source, total.shape)
    shape = (total.shape[0] + 1,) + total.shape[1:]
    depth = total.reshape(total.shape[0], -1)
    omega = (scattering / jnp.where(total > 0., total, 1.)).reshape(depth.shape)
    gamma1, gamma2, _ = params_hemispheric_mean(omega, 0.)
    transmission, reflection, absorbed = set_scat_trans_absorption_coeffs(gamma1, gamma2, depth)
    bottom = jnp.broadcast_to(bottom_source, total.shape[1:]).reshape(-1)
    top = jnp.broadcast_to(incoming_top, total.shape[1:]).reshape(-1)
    # For hemispheric mean, reduced pi B is exactly pi B. Avoid the
    # removable 0/0 in 2*(1-omega)/(gamma1-gamma2) at omega=1.
    up, down = solve_fluxadding_twostream_fluxes(
        transmission, reflection, source.reshape(depth.shape), jnp.zeros_like(bottom),
        bottom, source_top=top, absorption_coeff=absorbed,
    )
    up, down = up.reshape(shape), down.reshape(shape)
    return Fluxes(up, down, jnp.zeros_like(up), up - down)


@partial(jax.jit, static_argnames=("subdivisions",))
def paired_fluxes(absorption, scattering, boundary_source, center_source, upper_fraction, *, subdivisions):
    """Sample the same P7 two-segment linear source at sublayer midpoints.

    Each boundary-to-center segment has `subdivisions` constant-source
    sublayers. Extinction per physical layer and all original boundaries are
    retained. Opacity and chemistry are not reevaluated within the sublayers.
    """
    if not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError("Use a positive integer subdivision count")
    absorption, scattering = jnp.broadcast_arrays(absorption, scattering)
    boundary = jnp.broadcast_to(boundary_source, (absorption.shape[0] + 1,) + absorption.shape[1:])
    center = jnp.broadcast_to(center_source, absorption.shape)
    fraction = jnp.stack((upper_fraction, 1. - upper_fraction), axis=1) / subdivisions
    factor = fraction.reshape(fraction.shape + (1,) * (absorption.ndim))
    subshape = (absorption.shape[0], 2, subdivisions) + absorption.shape[1:]
    sub_absorption = jnp.broadcast_to(absorption[:, None, None] * factor, subshape)
    sub_scattering = jnp.broadcast_to(scattering[:, None, None] * factor, subshape)
    left, right = jnp.stack((boundary[:-1], center), axis=1), jnp.stack((center, boundary[1:]), axis=1)
    midpoint = ((jnp.arange(subdivisions) + .5) / subdivisions).reshape(
        (1, 1, subdivisions) + (1,) * (absorption.ndim - 1))
    source = (1. - midpoint) * left[:, :, None] + midpoint * right[:, :, None]
    flatshape = (absorption.shape[0] * 2 * subdivisions,) + absorption.shape[1:]
    a, s, source = (value.reshape(flatshape) for value in (sub_absorption, sub_scattering, source))
    on = thermal_fluxes(a, s, source, boundary[-1])
    off = thermal_fluxes(a, jnp.zeros_like(s), source, boundary[-1])
    return tuple(Fluxes(*(value[::2 * subdivisions] for value in flux)) for flux in (on, off))


def constant_absorption_reference(depth, source, bottom):
    """Independent mu=1/2 recurrence for the zero-scattering limit."""
    depth = np.asarray(depth)
    source = np.broadcast_to(source, depth.shape)
    transmission, emission = np.exp(-2. * depth), -np.expm1(-2. * depth) * source
    up, down = np.zeros((depth.shape[0] + 1,) + depth.shape[1:]), np.zeros((depth.shape[0] + 1,) + depth.shape[1:])
    up[-1] = bottom
    for layer in range(depth.shape[0] - 1, -1, -1):
        up[layer] = transmission[layer] * up[layer + 1] + emission[layer]
    for layer in range(depth.shape[0]):
        down[layer + 1] = transmission[layer] * down[layer] + emission[layer]
    return Fluxes(up, down, np.zeros_like(up), up - down)


def summarize(boundary_flux, top_bands):
    values = dict(zip(Fluxes._fields, boundary_flux))
    return dict(boundary_flux_cgs=values, top_upward_per_cm1=top_bands,
                layer_heating_cgs=np.diff(values["net"]))


def difference(current, reference, incident):
    delta = {key: current["boundary_flux_cgs"][key] - reference["boundary_flux_cgs"][key]
             for key in Fluxes._fields}
    return dict(boundary_flux_difference_cgs=delta,
        layer_heating_difference_cgs=np.diff(delta["net"]),
        top_upward_difference_per_cm1=current["top_upward_per_cm1"] - reference["top_upward_per_cm1"],
        top_upward_difference_cgs=float(delta["upward"][0]),
        top_upward_difference_over_reference_thermal=float(delta["upward"][0] / reference["boundary_flux_cgs"]["upward"][0]),
        max_boundary_difference_over_incident={key: float(np.max(np.abs(value))) / incident for key, value in delta.items()},
        max_layer_heating_difference_over_incident=float(np.max(np.abs(np.diff(delta["net"])))) / incident)


def audit_arrays(absorption, scattering, boundary_source, center_source, fraction,
                 g_weights, band_widths, mus, angular_weights, incident, *, band_batch=128):
    """Bound transfer work by band blocks, retaining only integrated boundaries."""
    layers, _, bands = absorption.shape
    names = ["linear_hm", "linear_p7_angular"] + [f"{mode}_{n}" for n in (1, 2, 4) for mode in ("on", "off")]
    flux = {name: np.zeros((4, layers + 1)) for name in names}
    top = {name: np.zeros(bands) for name in names}
    checks = dict(zero_scattering_max_relative_error=0., conservative_max_relative_net_error=0.)

    def collect(name, value, selection):
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            raise ValueError("Nonfinite thermal boundary flux")
        averaged = np.einsum('flgb,g->flb', array, g_weights)
        flux[name] += averaged @ band_widths[selection]
        top[name][selection] = averaged[0, 0]

    for start in range(0, bands, band_batch):
        selection = slice(start, min(start + band_batch, bands))
        a, s = jnp.asarray(absorption[..., selection]), jnp.asarray(scattering[..., selection])
        boundary, center = jnp.asarray(boundary_source[..., selection]), jnp.asarray(center_source[..., selection])
        zero = thermal_fluxes(a, jnp.zeros_like(s), center, boundary[-1])
        expected = constant_absorption_reference(a, center, boundary[-1])
        scale = float(np.max(boundary))
        checks["zero_scattering_max_relative_error"] = max(checks["zero_scattering_max_relative_error"],
            max(float(np.max(np.abs(np.asarray(x) - y))) for x, y in zip(zero, expected)) / scale)
        conservative = thermal_fluxes(jnp.zeros_like(a), s, center, boundary[-1])
        expected_net = np.asarray(boundary[-1]) / (1. + np.sum(np.asarray(s), axis=0))
        checks["conservative_max_relative_net_error"] = max(checks["conservative_max_relative_net_error"],
            float(np.max(np.abs(np.asarray(conservative.net) - expected_net))) / scale)
        for name, mu, weight in (("linear_hm", jnp.array([.5]), jnp.array([1.])),
                                  ("linear_p7_angular", mus, angular_weights)):
            up, down = rtrun_emis_pureabs_ibased_linsap_fluxes(
                a, boundary, mu, weight, source_center=center, upper_fraction=fraction)
            collect(name, Fluxes(up, down, jnp.zeros_like(up), up - down), selection)
        for n in (1, 2, 4):
            on, off = paired_fluxes(a, s, boundary, center, fraction, subdivisions=n)
            collect(f"on_{n}", on, selection)
            collect(f"off_{n}", off, selection)
    if max(checks.values()) > 5e-12:
        raise ValueError(f"Thermal limiting check failed: {checks}")
    summaries = {name: summarize(flux[name], top[name]) for name in names}
    cases = {str(n): dict(scattering_on=summaries[f"on_{n}"], scattering_off=summaries[f"off_{n}"],
        on_minus_off=difference(summaries[f"on_{n}"], summaries[f"off_{n}"], incident),
        off_minus_linear_hm=difference(summaries[f"off_{n}"], summaries["linear_hm"], incident)) for n in (1, 2, 4)}
    return dict(cases=cases, limiting_checks=checks, linear_absorption={name: summaries[name] for name in names[:2]},
        closure_difference_hm_minus_p7_angular=difference(summaries["linear_hm"], summaries["linear_p7_angular"], incident),
        source_refinement={f"{n}_minus_{n // 2}": dict(
            scattering_on=difference(summaries[f"on_{n}"], summaries[f"on_{n // 2}"], incident),
            scattering_off=difference(summaries[f"off_{n}"], summaries[f"off_{n // 2}"], incident),
            scattering_effect_boundary_change_cgs={key:
                cases[str(n)]["on_minus_off"]["boundary_flux_difference_cgs"][key]
                - cases[str(n // 2)]["on_minus_off"]["boundary_flux_difference_cgs"][key] for key in Fluxes._fields})
            for n in (2, 4)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("manifest", "observation", "state", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--band-batch", type=int, default=128)
    args = parser.parse_args(argv)
    if args.band_batch < 1 or args.output.resolve() in {args.manifest.resolve(), args.observation.resolve(), args.state.resolve()}:
        parser.error("Use a positive band batch and a separate output")
    jax.config.update("jax_enable_x64", True)
    started = perf_counter()
    report = dict(status="loading", certifies_p7=False, temperature_and_composition_fixed=True,
        thermal_only=True, stellar_incident_in_transfer=False, band_batch=args.band_batch,
        scope="Paired fixed-column hemispheric-mean thermal scattering; no RCE or P1 prediction and no rigorous error bound.",
        started_utc=datetime.now(timezone.utc).isoformat(),
        input_state=dict(path=str(args.state.resolve()), sha256=_sha256(args.state)),
        input_manifest=dict(path=str(args.manifest.resolve()), sha256=_sha256(args.manifest)))
    try:
        source = json.loads(args.state.read_text())
        state = source["state"] if "state" in source else source["stages"][-1]["state"]
        if not all(state[key] for key in ("converged", "domain_valid", "physics_valid")):
            raise ValueError("Use a converged valid input state")
        manifest, observation = load_manifest(args.manifest), load_emission_candidate(args.observation)
        if not np.array_equal(source["parameters"], manifest.config["parameters"]):
            raise ValueError("State and manifest parameters differ")
        if source.get("observation_sha256", observation.sha256) != observation.sha256:
            raise ValueError("State and observation differ")
        column = prepare_column(manifest, observation, stream_resolved=True)
        temperature, bottom = jnp.asarray(state["temperature"]), jnp.asarray(state["bottom_temperature"])
        parameters = jnp.asarray(source["parameters"])
        if temperature.shape != column.pressure.shape:
            raise ValueError("State and manifest layer counts differ")
        if "pressure_bar" in source and not np.array_equal(source["pressure_bar"], column.pressure):
            raise ValueError("State and manifest pressure centers differ")
        if not bool(column.temperature_valid(temperature, bottom, parameters)):
            raise ValueError("The state is outside the common temperature domain")
        chemistry = jax.jit(column.chemistry)(jnp.append(temperature, bottom), *parameters[:2])
        if not bool(jnp.all(chemistry.diagnostics.valid)):
            raise ValueError("Invalid recomputed chemistry")
        absorption = np.asarray(jax.jit(column.optical_depth)(temperature, chemistry))
        if not np.all(np.isfinite(absorption) & (absorption >= 0.)):
            raise ValueError("Invalid absorption depth")
        scattering = sum(scattering_optical_depth(column, chemistry).values())[:, None, :]
        boundary_temperature = reconstruct_boundary_temperature(column.pressure, column.boundaries, temperature, bottom)
        boundary_source = band_planck_flux(boundary_temperature, column.band_nodes, column.band_weights)[:, None, :]
        center_source = band_planck_flux(temperature, column.band_nodes, column.band_weights)[:, None, :]
        fraction = (column.pressure - column.boundaries[:-1]) / jnp.diff(column.boundaries)
        incident = float(jnp.sum(irradiation_flux(column.star_bands, column.a_over_rstar,
                                               jnp.exp(parameters[2])) * column.band_widths))
        report.update(audit_arrays(absorption, scattering, boundary_source, center_source, fraction,
            np.asarray(column.g_weights), np.asarray(column.band_widths), column.mus,
            column.angular_weights, incident, band_batch=args.band_batch))
        import exojax.rt.flux as flux_module
        import exojax.rt.toon as toon_module
        import exojax.rt.twostream as twostream_module
        paths = [Path(__file__), *[Path(module.__file__) for module in (flux_module, toon_module, twostream_module)],
            *[Path(__file__).with_name(name) for name in ("_rce_wasp18b.py", "_rce_broadband.py", "_rce_wasp18b_audit.py", "_rce_wasp18b_scattering.py")]]
        report.update(status="complete_diagnostic_only", asset_sha256=manifest.sha256, provenance=provenance(),
            source_sha256={str(path.resolve()): _sha256(path) for path in paths},
            state_assets_match_manifest=source.get("asset_sha256") == manifest.sha256,
            observation_sha256=observation.sha256, configuration=manifest.config,
            parameters=parameters, temperature_K=temperature, bottom_temperature_K=bottom,
            pressure_boundaries_bar=column.boundaries, band_edges_cm1=column.band_edges,
            p7_angular_cosines=column.mus, p7_angular_weights=column.angular_weights,
            included_line_species=sorted(column.ckd), incident_flux_for_normalization_cgs=incident,
            chemistry_diagnostics=chemistry.diagnostics._asdict(), device_memory_stats=jax.devices()[0].memory_stats(),
            references=dict(closure="https://arxiv.org/abs/1405.0026", cross_sections=SCATTERING_SOURCES),
            limitations=["Hemispheric mean with g=0; higher Rayleigh/Thomson phase moments and polarization are absent.",
                "Source sublayers approximate the existing two-segment pi B profile; opacity/chemistry remain fixed within each original layer.",
                "Compare on/off at the same subdivision. Closure and source-discretization differences are reported separately.",
                "Band-averaged CKD thermal fluxes are not resolved observed-bin predictions.",
                "No stellar transfer, RCE re-equilibration, scattering-fit error envelope, or eight-stream scattering error bound is inferred."])
    except Exception as error:
        report.update(status="failed", failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        report.update(seconds=perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_serializable(report), indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

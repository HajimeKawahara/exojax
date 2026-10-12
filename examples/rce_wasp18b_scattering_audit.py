"""Audit stellar scattering at a fixed P7 temperature and composition.

This computes direct and two-stream diffuse stellar fluxes at every boundary.
It excludes scattering of thermal photons and does not solve a scattering RCE
column. Rayleigh/Thomson angular redistribution uses g=0 two-stream closures;
higher phase-function moments and polarization are not represented.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from _rce_broadband import Fluxes, integrate_broadband_fluxes, irradiation_flux
from _rce_observation import load_emission_candidate
from _rce_wasp18b_assets import _sha256, load_manifest
from _rce_wasp18b_audit import audit_scattering, scattering_optical_depth
from rce_wasp18b_forward import _serializable, prepare_column, provenance
from _rce_wasp18b_scattering import integrated_stellar_report, stellar_scattering_fluxes
from exojax.rt.flux import direct_beam_fluxes


def compare_stellar_reports(trial, reference):
    """Report signed flux/heating differences normalized by incident flux."""
    incident = reference["incident_flux_cgs"]
    difference = {name: trial["boundary_flux_cgs"][name] - reference["boundary_flux_cgs"][name]
                  for name in Fluxes._fields}
    heating = trial["layer_heating_cgs"] - reference["layer_heating_cgs"]
    return dict(boundary_flux_difference_cgs=difference,
        layer_heating_difference_cgs=heating,
        max_absolute_boundary_difference_over_incident={
            name: float(np.max(np.abs(value))) / incident for name, value in difference.items()},
        max_absolute_layer_heating_difference_over_incident=float(np.max(np.abs(heating))) / incident,
        total_atmosphere_absorption_change_over_incident=float(heating.sum()) / incident)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("manifest", "observation", "state", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() in {args.manifest.resolve(), args.observation.resolve(), args.state.resolve()}:
        parser.error("The output must be separate from every input")
    jax.config.update("jax_enable_x64", True)
    started = perf_counter()
    code = [Path(__file__), Path(__file__).with_name("_rce_wasp18b_audit.py"),
            Path(__file__).with_name("_rce_wasp18b_scattering.py"),
            Path(__file__).with_name("_rce_wasp18b.py"), Path(__file__).with_name("_rce_broadband.py")]
    import exojax.rt.direct_sfm as direct_module
    import exojax.rt.toon as toon_module
    import exojax.rt.twostream as twostream_module
    import exojax.rt.flux as flux_module
    code += [Path(module.__file__) for module in (direct_module, toon_module, twostream_module, flux_module)]
    report = dict(status="loading", started_utc=datetime.now(timezone.utc).isoformat(),
        certifies_p7=False, physical_validation_complete=False,
        temperature_and_composition_fixed=True, thermal_photon_scattering_included=False,
        scope="Stellar-only direct plus diffuse two-stream scattering on the supplied fixed column; no RCE re-equilibration.",
        source_sha256={str(path.resolve()): _sha256(path) for path in code},
        conventions=dict(direction="Layers and boundaries are ordered top to bottom; net = upward - downward - incident.",
            heating="Net upward flux at the lower boundary minus that at the upper boundary; positive means absorption.",
            illumination="Horizontal incident flux; divide by mu0 only for the beam-normal layer-source API.",
            boundaries="Zero diffuse stellar incidence at the top; black bottom with zero stellar source; bottom-reaching light remains explicit.",
            angular_scattering="Rayleigh and Thomson approximated by g=0 two-stream closures, without higher angular moments or polarization."))

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(_serializable(report), indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    try:
        source = json.loads(args.state.read_text())
        state = source["state"] if "state" in source else source["stages"][-1]["state"]
        if not all(state[key] for key in ("converged", "domain_valid", "physics_valid")):
            raise ValueError("Use a converged valid input RCE state")
        manifest, observation = load_manifest(args.manifest), load_emission_candidate(args.observation)
        if not np.array_equal(source["parameters"], manifest.config["parameters"]):
            raise ValueError("State and manifest parameters differ")
        if source.get("observation_sha256", observation.sha256) != observation.sha256:
            raise ValueError("State and observation hashes differ")
        report.update(asset_sha256=manifest.sha256, provenance=provenance(),
            input_state=dict(path=str(args.state.resolve()), sha256=_sha256(args.state)),
            input_manifest=str(args.manifest.resolve()), observation_sha256=observation.sha256,
            state_asset_hashes_match_current_manifest=source.get("asset_sha256") == manifest.sha256)
        print("Preparing metadata-only resolved column", flush=True)
        column = prepare_column(manifest, observation, stream_resolved=True)
        temperature, bottom = jnp.asarray(state["temperature"]), jnp.asarray(state["bottom_temperature"])
        parameters = jnp.asarray(source["parameters"])
        if not bool(column.temperature_valid(temperature, bottom, parameters)):
            raise ValueError("State lies outside the current column's temperature domain")
        chemistry = jax.jit(column.chemistry)(jnp.append(temperature, bottom), *parameters[:2])
        if not bool(jnp.all(chemistry.diagnostics.valid)):
            raise ValueError("Invalid recomputed chemistry")
        report.update(parameters=parameters, temperature_K=temperature, bottom_temperature_K=bottom,
            included_line_species=sorted(column.ckd), pressure_boundaries_bar=column.boundaries,
            chemistry_diagnostics=chemistry.diagnostics._asdict(), mu0=column.settings.mu0,
            status="absorption", setup_seconds=perf_counter() - started)
        save()
        print("Computing the fixed-column CKD absorption depth", flush=True)
        begin = perf_counter()
        absorption = jax.jit(column.optical_depth)(temperature, chemistry)
        absorption_host = np.asarray(absorption)
        if not np.all(np.isfinite(absorption_host) & (absorption_host >= 0.)):
            raise ValueError("Invalid absorption depth")
        report["absorption_seconds"] = perf_counter() - begin
        components = scattering_optical_depth(column, chemistry)
        scattering = jnp.asarray(sum(components.values()))[:, None, :]
        report["first_scattering_bound"] = audit_scattering(column, temperature, chemistry,
                                                           absorption_depth=absorption_host)
        horizontal = irradiation_flux(column.star_bands[None, :], column.a_over_rstar, jnp.exp(parameters[2]))
        reference_direct = direct_beam_fluxes(absorption, horizontal, column.settings.mu0)
        zero = jnp.zeros_like(reference_direct)
        reference = integrated_stellar_report(Fluxes(zero, zero, reference_direct, -reference_direct),
                                             column.g_weights, column.band_widths)
        report.update(absorption_only=reference, status="scattering")
        save()
        begin = perf_counter()
        cases, checks = {}, {}
        for closure in ("quadrature", "hemispheric_mean"):
            print(f"Evaluating {closure} stellar scattering", flush=True)
            no_scattering = stellar_scattering_fluxes(absorption, jnp.zeros_like(scattering),
                horizontal, column.settings.mu0, closure=closure)
            scale = float(jnp.max(horizontal))
            maximum_zero_error = max(float(jnp.max(jnp.abs(left - right))) for left, right in zip(
                no_scattering, (zero, zero, reference_direct, -reference_direct))) / scale
            if maximum_zero_error > 5e-14:
                raise ValueError("Zero scattering differs from the P7 direct-beam convention")
            conservative = stellar_scattering_fluxes(jnp.zeros_like(absorption), scattering,
                horizontal, column.settings.mu0, closure=closure)
            conservative_values = integrated_stellar_report(conservative, column.g_weights, column.band_widths)
            conservative_net = np.asarray(conservative.net)
            maximum_energy_error = float(np.max(np.abs(conservative_net - conservative_net[:1]))) / scale
            if maximum_energy_error > 5e-12:
                raise ValueError("The conservative scattering limit does not preserve stellar energy")
            trial = stellar_scattering_fluxes(absorption, scattering, horizontal,
                                            column.settings.mu0, closure=closure)
            cases[closure] = integrated_stellar_report(trial, column.g_weights, column.band_widths)
            cases[closure]["minus_absorption_only"] = compare_stellar_reports(cases[closure], reference)
            checks[closure] = dict(zero_scattering_max_error_over_max_incident_density=maximum_zero_error,
                conservative_max_boundary_net_variation_over_max_incident_density=maximum_energy_error,
                conservative_reflected_fraction=conservative_values["reflected_fraction"],
                conservative_bottom_reaching_fraction=conservative_values["bottom_reaching_fraction"],
                passed=True)
        report.update(cases=cases, limiting_checks=checks,
            hemispheric_mean_minus_quadrature=compare_stellar_reports(cases["hemispheric_mean"], cases["quadrature"]),
            scattering_seconds=perf_counter() - begin, status="complete_diagnostic_only",
            device_memory_stats=jax.devices()[0].memory_stats(),
            sources=dict(two_stream_direct_beam="https://arxiv.org/html/1804.04961v2",
                         closure_conservation="https://arxiv.org/abs/1405.0026"),
            limitations=[
                "This is a fixed-column stellar-heating diagnostic, not a scattering RCE solution or an observed-spectrum error bound.",
                "Scattering of thermal photons, other scatterers and clouds are excluded.",
                "Rayleigh/Thomson phase functions are replaced by g=0 two-stream closures; higher moments and polarization are absent.",
                "The difference between the two closures is a sensitivity diagnostic, not a rigorous error envelope.",
                "Cross-section-fit errors, CKD correlations, band averaging and omitted absorbers remain unbounded.",
                "The first-scattering bound applies to the declared fixed absorption model; the three-molecule baseline bound is approximately 3%, not a small-error certificate.",
                "Input-state convergence does not imply convergence with a different current opacity manifest.",
            ])
    except Exception as error:
        report.update(status="failed", failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        report.update(total_seconds=perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

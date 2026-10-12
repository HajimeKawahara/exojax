"""Evaluate finite final8 prior states with the unchanged bounded RCE solver.

Each of the fixed 27 points starts at the manifest seed. Chemistry and CKD
radiation are inspected even when the returned state has not converged.
No resolved spectra, implicit gradients, retries or prior changes are made.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / 'results/rce_wasp18b_final8_domain_inventory.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def finite(values):
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    return dict(finite=bool(np.all(good)), nonfinite_count=int(np.sum(~good)), shape=list(values.shape),
                minimum=float(np.min(values[good])) if np.any(good) else None,
                maximum=float(np.max(values[good])) if np.any(good) else None)


def coverage(values, limits):
    values = np.asarray(values, dtype=float)
    inside = np.isfinite(values) & (values >= limits[0]) & (values <= limits[1])
    return dict(valid=bool(np.all(inside)), range=list(limits), outside_indices=np.flatnonzero(~inside).tolist(),
        lower_margin=float(np.min(values) - limits[0]) if np.all(np.isfinite(values)) else None,
        upper_margin=float(limits[1] - np.max(values)) if np.all(np.isfinite(values)) else None)


def source_metadata():
    """Reuse only pinned line/PF metadata, not the historical partial inventory."""
    inventory = read(INVENTORY)
    result = {}
    for name, old in inventory['species'].items():
        molecular = old.get('molecular_source')
        if molecular:
            for key in ('definition', 'partition'):
                entry = molecular[key]
                if sha(entry['path']) != entry['sha256']:
                    raise ValueError(f'Line source metadata changed: {name}/{key}')
            result[name] = dict(partition_temperature_K=molecular['actual_partition_temperature_K'],
                declared_linelist_maximum_K=molecular['declared_linelist_temperature_K'],
                source_files=[molecular['definition'], molecular['partition']])
        else:
            pf = inventory['atomic_partition']
            if sha(pf['table']['path']) != pf['table']['sha256']:
                raise ValueError('Atomic partition table changed')
            result[name] = dict(partition_temperature_K=pf['temperature_range_K'],
                declared_linelist_maximum_K=None, source_files=[pf['table']],
                completeness='No declared atomic line-completeness temperature.')
    return result


def column_domains(column, sources):
    """Read current loaded coordinates, including metadata-only resolved tables."""
    limits = lambda x: [float(np.min(x)), float(np.max(x))]
    species = {}
    for name, opacity in column.ckd.items():
        ckd, lbl = opacity.ckd_info, column.lbl[name].diffgrid_info
        species[name] = dict(ckd_temperature_K=limits(ckd.T_grid), ckd_pressure_bar=limits(ckd.P_grid),
            lbl_temperature_K=limits(lbl.temperature_grid), lbl_pressure_bar=limits(lbl.pressure_grid),
            lbl_native_pressure_matches=bool(np.allclose(lbl.pressure_grid, column.pressure, rtol=1e-12, atol=0)), **sources[name])
    return dict(temperature_K=list(column.temperature_range), pressure_bar=np.asarray(column.pressure).tolist(),
        chemistry_pressure_bar=np.asarray(column.nodes).tolist(),
        chemistry_pressure_range_bar=list(column.chemistry.pressure_range), species=species,
        cia={'-'.join(pair): dict(temperature_K=limits(table.tcia), wavenumber_cm1=limits(table.nucia),
                               outside_spectral_range='zero', pressure_axis=None)
             for pair, table in column.cia.items()},
        broadband_cm1=limits(column.band_edges), resolved_cm1=limits(column.nu),
        bottom_scope='Gas opacity is evaluated at layer centers only; the bottom node supplies chemistry and blackbody emission.',
        spectral_scope='Prepared finite CIA/scattering support and stellar tail are unchanged; this is not a missing-opacity bound.')


def state_domains(temperature, bottom, domains):
    temperature = np.asarray(temperature, dtype=float)
    nodes = np.append(temperature, bottom)
    pressure = domains['pressure_bar']
    result = dict(temperature=coverage(nodes, domains['temperature_K']),
        chemistry_pressure=coverage(domains['chemistry_pressure_bar'], domains['chemistry_pressure_range_bar']),
        species={}, cia={})
    checks = [result['temperature']['valid'], result['chemistry_pressure']['valid']]
    for name, source in domains['species'].items():
        row = {kind: coverage(temperature if 'temperature' in kind else pressure, source[kind])
               for kind in ('ckd_temperature_K', 'lbl_temperature_K', 'ckd_pressure_bar', 'lbl_pressure_bar')}
        row['partition_temperature_K'] = coverage(temperature, source['partition_temperature_K'])
        checks.extend(item['valid'] for item in row.values())
        checks.append(source['lbl_native_pressure_matches'])
        maximum = source['declared_linelist_maximum_K']
        row['layers_above_declared_linelist_temperature'] = (
            np.flatnonzero(temperature > maximum).tolist() if maximum is not None else None)
        row['declared_linelist_maximum_K'] = maximum
        result['species'][name] = row
    for name, source in domains['cia'].items():
        result['cia'][name] = coverage(temperature, source['temperature_K'])
        checks.append(result['cia'][name]['valid'])
    result['numerical_domain_valid'] = all(checks)
    return result


def summarize(rows, planned):
    return dict(planned=planned, finished=len(rows),
        converged=sum(bool(row['state']['converged']) for row in rows),
        finite_valid_returned_states=sum(bool(row['finite_valid_returned_state']) for row in rows),
        unresolved_cases=[row['case']['name'] for row in rows if not row['state']['converged']],
        review_cases=[row['case']['name'] for row in rows if not row['finite_valid_returned_state']],
        statuses=dict(Counter(row['status_name'] for row in rows)))


def audit_returned(column, state, parameters, domains, entropy_fn, depth_fn, flux_fn):
    import jax
    import jax.numpy as jnp
    from _rce_broadband import integrate_broadband_fluxes
    from _rce_wasp18b_audit import stellar_flux_audit

    result = dict(domain=state_domains(state.temperature, state.bottom_temperature, domains),
        solver_field_finiteness={key: finite(getattr(state, key)) for key in
            ('temperature', 'bottom_temperature', 'radiative_flux', 'convective_flux',
             'flux_residual', 'stability_residual', 'scaled_residual')},
        chemistry_status='not_evaluated_outside_domain', radiation_status='not_evaluated',
        finite_valid_returned_state=False)
    if not result['domain']['numerical_domain_valid'] or not bool(column.temperature_valid(
            state.temperature, state.bottom_temperature, parameters)):
        return result
    entropy = jax.block_until_ready(entropy_fn(jnp.append(state.temperature, state.bottom_temperature), *parameters[:2]))
    chemistry = entropy.chemistry
    checks = {key: finite(getattr(chemistry, key)) for key in chemistry._fields if key != 'diagnostics'}
    diagnostic_checks = {key: finite(value) for key, value in chemistry.diagnostics._asdict().items()}
    result.update(chemistry_status='evaluated', chemistry=chemistry.diagnostics._asdict(),
        chemistry_field_finiteness=checks, chemistry_diagnostic_finiteness=diagnostic_checks,
        thermodynamics=dict(valid=entropy.valid, status=entropy.status, specific_entropy=entropy.specific_entropy,
                            specific_entropy_finiteness=finite(entropy.specific_entropy)),
        opacity_species_mole_fractions={name: chemistry.x[:, column.chemistry.species.index(name)] for name in column.ckd})
    if not (bool(np.all(entropy.valid)) and bool(np.all(chemistry.diagnostics.valid))
            and bool(np.all(chemistry.diagnostics.converged)) and bool(chemistry.diagnostics.in_domain)
            and all(item['finite'] for item in (*checks.values(), *diagnostic_checks.values()))):
        result['radiation_status'] = 'not_evaluated_invalid_chemistry_or_thermodynamics'
        return result
    depth = jax.block_until_ready(depth_fn(state.temperature, chemistry))
    flux = jax.block_until_ready(flux_fn(state.temperature, state.bottom_temperature, parameters, chemistry))
    broadband = jax.block_until_ready(integrate_broadband_fluxes(flux, column.g_weights, column.band_widths))
    stellar = stellar_flux_audit(column, state.temperature, parameters, chemistry, absorption_depth=depth)
    radiation = {key: finite(value) for key, value in flux._asdict().items()}
    depth_stats = finite(depth)
    result.update(radiation_status='evaluated', absorption_depth=depth_stats, spectral_flux_finiteness=radiation,
                  broadband_flux=broadband._asdict(), stellar_flux_audit=stellar)
    result['finite_valid_returned_state'] = bool(state.domain_valid and state.physics_valid
        and all(item['finite'] for item in result['solver_field_finiteness'].values())
        and depth_stats['finite'] and depth_stats['minimum'] >= 0
        and all(item['finite'] for item in radiation.values())
        and all(finite(value)['finite'] for value in broadband)
        and all(finite(value)['finite'] for value in stellar['boundary_flux_cgs'].values()))
    return result


def baseline_checks(row, baseline):
    return dict(temperature=bool(np.allclose(row['state']['temperature'], baseline['temperature_K'], rtol=1e-9, atol=1e-6)),
        bottom=bool(np.isclose(row['state']['bottom_temperature'], baseline['bottom_temperature_K'], rtol=1e-9, atol=1e-6)),
        flux=all(np.allclose(row['broadband_flux'][key], baseline['broadband_flux'][key], rtol=1e-9, atol=1e-3)
                 for key in ('upward', 'downward', 'incident', 'net')),
        mask=row['state']['convective_mask'] == baseline['state']['convective_mask'],
        converged=bool(row['state']['converged']), finite_valid=bool(row['finite_valid_returned_state']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    plan = read(args.plan)
    for entry in (plan['driver'], *plan['protected']):
        if sha(entry['path']) != entry['sha256']:
            raise ValueError(f'Prepared input changed: {entry["path"]}')
    if plan['sources'] != source_metadata() or plan['driver']['sha256'] != sha(__file__):
        raise ValueError('Prior driver or source metadata changed')
    if any(os.path.lexists(p) for p in (args.output, args.output.with_name(args.output.name + '.tmp'))):
        raise FileExistsError('Preserve existing prior diagnostics')
    sys.path[:0] = [str(ROOT / 'examples'), str(ROOT / 'src'), '/home/kawahara/exogibbs/src']
    import jax
    import jax.numpy as jnp
    from _rce_observation import load_emission_candidate
    from _rce_wasp18b_assets import load_manifest
    from _rce_wasp18b_continuation import PreparedMaskContinuation
    from exojax.atm.rce_device import RceStatus
    import rce_wasp18b_forward as forward
    from rce_wasp18b_validation import make_cases, serializable
    from rce_wasp18b_explicit_ckd import ExplicitCKDContext

    jax.config.update('jax_enable_x64', True)
    if jax.default_backend() != 'gpu':
        raise ValueError('Submit this finite calculation on the GPU')
    report = dict(schema_version='exojax.final8-finite-prior/1', status='loading', started_utc=now(),
        plan_sha256=sha(args.plan), cases=plan['cases'], runs=[], current_case=None, current_stages=[],
        resolved_spectra_evaluated=False, p7_complete=False,
        scope='Finite returned states from independent fixed-seed attempts. Solver failure is not proof of absent roots or an invalid prior.',
        limitations=['Stage states and flags are stored; chemistry/radiation diagnostics refer to the final returned state.',
                     'Individual rejected line-search trial vectors are not saved; DOMAIN_STEP_FAILED is distinct from returned-state domain validity.',
                     'Line-list completeness limits are recorded separately from numerical interpolation validity.'])
    context = ExplicitCKDContext()

    def save():
        report.update(updated_utc=now(), summary=summarize(report['runs'], len(report['cases'])),
                      execution_adapter=context.evidence())
        temporary = args.output.with_name(args.output.name + '.tmp')
        temporary.write_text(json.dumps(serializable(report), indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)

    save()
    try:
        with context:
            manifest = load_manifest(Path(plan['manifest']))
            observation = load_emission_candidate(Path(plan['observation']))
            if serializable(make_cases(manifest.config, 'prior')) != plan['cases']:
                raise ValueError('The exact 27-point prior changed')
            runtime = forward.provenance()
            if any(runtime[key] != value for key, value in plan['runtime'].items()):
                raise ValueError('Prepared scientific source/runtime changed')
            column = forward.prepare_column(manifest, observation, stream_resolved=True)
            column.check_response_coverage(700)
            domains = column_domains(column, plan['sources'])
            nominal = jnp.asarray(manifest.config['parameters'], dtype=float)
            solver = PreparedMaskContinuation(column, nominal)
            entropy_fn = jax.jit(column.entropy)
            depth_fn, flux_fn = context.bind(column.optical_depth, column), context.bind(column.fluxes, column)
            report.update(configuration=manifest.config, asset_sha256=manifest.sha256,
                observation_sha256=observation.sha256, provenance=runtime, domains=domains,
                initial_temperature_K=column.solver_arguments[2], initial_bottom_temperature_K=column.solver_arguments[3],
                initial_convective_mask=solver.initial_state[2], solver_options=solver.solver_options,
                maximum_stages=solver.max_stages, solve_compile_seconds=solver.compile_seconds, status='running')
            baseline = read(plan['baseline'])
            if (report['configuration'] != baseline['configuration'] or manifest.sha256 != baseline['asset_sha256']
                    or observation.sha256 != baseline['observation_sha256']):
                raise ValueError('The prior and reviewed baseline inputs differ')
            for case in report['cases']:
                report.update(current_case=case['name'], current_stages=[])
                save()
                parameters = jnp.asarray(case['parameters'], dtype=float)
                def on_stage(stage):
                    report['current_stages'].append(stage)
                    save()
                returned = solver.run(parameters, on_stage=on_stage)
                state = returned.state
                row = dict(case=case, state=state._asdict(), status_name=RceStatus(int(state.status)).name,
                    stop_reason=returned.stop_reason, stages=returned.stages,
                    **audit_returned(column, state, parameters, domains, entropy_fn, depth_fn, flux_fn))
                row = serializable(row)
                report['runs'].append(row)
                report['current_stages'] = []
                if len(report['runs']) == 1:
                    report['baseline_reproduction'] = baseline_checks(row, baseline) if 'broadband_flux' in row else {'finite_valid': False}
                    if not all(report['baseline_reproduction'].values()):
                        report.update(status='needs_baseline_review', current_case=None)
                        save()
                        return 0
                save()
                print(f'{case["name"]}: {row["status_name"]}; finite_valid={row["finite_valid_returned_state"]}', flush=True)
            report.update(status='complete_finite_diagnostic', current_case=None)
        return 0
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['finished_utc'] = now()
        save()


if __name__ == '__main__':
    raise SystemExit(main())

"""Screen the existing WASP-18b column with reproducible, incremental reports.

Choose one suite: ``prior`` evaluates the 27 lower/nominal/upper parameter
combinations, ``initial`` changes the initial temperature, and ``quadrature``
doubles the angular or within-band Planck quadrature. ``vertical`` compares
8/16/32 layers at stored pressure centers; ``depth`` retains the first
30/31/32 native layers. Neither pressure suite interpolates opacity or adds
new pressure information. ``inner_refinement`` compares 32/62/122 layers by
subdividing the interior layers on a fixed log-pressure opacity model. Its
endpoint layers remain unchanged; pressure-opacity accuracy is not certified.
``opacity_pressure`` keeps all 32 physical layers and restores 9/17/32
native pressure nodes in the opacity tables. It tests interpolation at held-out
native nodes, without certifying accuracy between the full 32 native nodes.
``opacity_temperature`` similarly restores 5/9/16 stored temperature nodes,
keeping both endpoint temperatures and the native pressure/RT grids fixed.
These diagnostics do not certify P7 acceptance.
``--stream-resolved`` opts into bounded-memory spectra, including exact native
pressure subsets, without interpolating the prepared opacity arrays.
``--mask-continuation`` permits bounded mask updates within each point; every
point still begins at its own explicit column seed and retains all stages.
Set JAX_PLATFORMS/CUDA_VISIBLE_DEVICES externally before starting this script.
"""

import argparse
from collections import Counter
from collections.abc import Mapping
from copy import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from _rce_broadband import (
    band_planck_flux, bottom_leakage_fraction, integrate_broadband_fluxes,
)
from _rce_observation import load_emission_candidate
from _rce_wasp18b import SIGMA_SB, Wasp18bColumn
from _rce_wasp18b_assets import load_manifest
from _rce_wasp18b_continuation import PreparedMaskContinuation
from exojax.atm.rce_device import RceStatus
from rce_wasp18b_forward import compare_runs, physical_inputs, prepare_column, provenance


def serializable(value):
    """Preserve invalid numerical diagnostics as JSON null, never NaN tokens."""
    if isinstance(value, Mapping):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        return serializable(np.asarray(value).tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_cases(config, suite, initial_temperatures=(2000.0, 2500.0, 4000.0)):
    """Return the exact ordered cases without changing the declared prior."""
    nominal = np.asarray(config["parameters"], dtype=float)
    lower, upper = (np.asarray(config[key], dtype=float)
                    for key in ("prior_lower", "prior_upper"))
    if (nominal.shape != (3,) or lower.shape != (3,) or upper.shape != (3,)
            or not np.all(np.isfinite([nominal, lower, upper]))
            or not np.all((lower < upper) & (nominal >= lower) & (nominal <= upper))):
        raise ValueError("Require three finite nominal parameters inside ordered prior bounds.")
    if suite == "prior":
        if not np.all((nominal > lower) & (nominal < upper)):
            raise ValueError("The 27-point suite requires a distinct interior nominal value on each axis.")
        points = [tuple(nominal)]
        points.extend(point for point in product(*zip(lower, nominal, upper))
                      if point != tuple(nominal))
        return [{"name": f"prior_{index:02d}", "parameters": list(point),
                 "settings": {}, "initial_temperature_K": None}
                for index, point in enumerate(points)]
    if suite == "initial":
        if not initial_temperatures or not np.all(
                np.isfinite(initial_temperatures) & (np.asarray(initial_temperatures) > 0)):
            raise ValueError("Initial temperatures must be finite and positive.")
        return [{"name": f"initial_{index:02d}_{temperature:g}K",
                 "parameters": nominal.tolist(), "settings": {},
                 "initial_temperature_K": float(temperature)}
                for index, temperature in enumerate(initial_temperatures)]
    if suite == "quadrature":
        settings = config["column"]["settings"]
        cases = [("baseline", {}),
                 ("angular_doubled", {"nstream": 2 * settings.get("nstream", 8)}),
                 ("planck_doubled", {"planck_order": 2 * settings.get("planck_order", 8)})]
        return [{"name": name, "parameters": nominal.tolist(), "settings": changes,
                 "initial_temperature_K": None} for name, changes in cases]
    if suite in ("vertical", "depth"):
        boundaries = np.asarray(config["column"]["pressure_boundaries_bar"], dtype=float)
        grids = native_pressure_grids(boundaries, suite)
        return [dict(grid, parameters=nominal.tolist(), settings={}, initial_temperature_K=None)
                for grid in grids]
    if suite in ("inner_refinement", "opacity_pressure", "opacity_temperature"):
        make_grids = {"inner_refinement": inner_refinement_grids,
                      "opacity_pressure": opacity_pressure_grids,
                      "opacity_temperature": opacity_temperature_grids}[suite]
        grids = make_grids(config["column"]["pressure_boundaries_bar"])
        return [dict(grid, parameters=nominal.tolist(), settings={}, initial_temperature_K=None)
                for grid in grids]
    raise ValueError(f"Unknown validation suite: {suite}")


def native_pressure_grids(boundaries, suite):
    """Derive exact subsets of the existing 32-layer logarithmic grid.

    For selected native center j, the next boundary index is 2*j+1 minus
    the previous boundary index. Thus geometric centers are unchanged while
    the coarse 8/16-layer meshes have alternating log-pressure widths.
    """
    boundaries = np.asarray(boundaries, dtype=float)
    if (boundaries.shape != (33,) or not np.all(np.isfinite(boundaries))
            or np.any(boundaries <= 0) or np.any(np.diff(boundaries) <= 0)):
        raise ValueError("Native pressure suites require 33 increasing positive boundaries.")
    steps = np.diff(np.log(boundaries))
    if not np.allclose(steps, steps[0], rtol=1e-11, atol=1e-13):
        raise ValueError("Native pressure suites require the original logarithmic 32-layer grid.")
    source_pressure = np.sqrt(boundaries[:-1] * boundaries[1:])
    if suite == "vertical":
        selections = [np.arange(1, 32, 4), np.arange(0, 32, 2), np.arange(32)]
    elif suite == "depth":
        selections = [np.arange(count) for count in (30, 31, 32)]
    else:
        raise ValueError(f"Unknown native pressure suite: {suite}")
    grids = []
    for selected in selections:
        if suite == "depth":
            boundary_indices = np.arange(selected.size + 1)
        else:
            boundary_indices = [0]
            for index in selected:
                boundary_indices.append(2 * int(index) + 1 - boundary_indices[-1])
            boundary_indices = np.asarray(boundary_indices)
        if (np.any(np.diff(boundary_indices) <= 0) or boundary_indices[0] != 0
                or boundary_indices[-1] > 32
                or (suite == "vertical" and boundary_indices[-1] != 32)):
            raise ValueError("Invalid native pressure subset boundaries.")
        chosen_boundaries = boundaries[boundary_indices]
        centers = np.sqrt(chosen_boundaries[:-1] * chosen_boundaries[1:])
        if not np.allclose(centers, source_pressure[selected], rtol=1e-12, atol=0.0):
            raise ValueError("Selected boundaries do not preserve native pressure centers.")
        grids.append({
            "name": f"{suite}_{selected.size:02d}_layers",
            "source_pressure_indices": selected.tolist(),
            "source_boundary_indices": boundary_indices.tolist(),
            "source_pressure_grid_bar": source_pressure.tolist(),
            "pressure_boundaries_bar": chosen_boundaries.tolist(),
            "pressure_centers_bar": source_pressure[selected].tolist(),
            "grid_scope": ("Exact stored pressure slices; vertical meshes are nonuniform in log P. "
                           "No opacity interpolation, new pressure nodes, or spectral refinement."),
            "initial_state_scope": ("Temperature and initial mask select the recorded native centers. "
                                    "The mask is only an initial guess for the new connections. "
                                    "A cropped bottom seed uses log T interpolation in log P of the "
                                    "original seed; the unchanged bottom retains its original value."),
        })
    return grids


def subset_diffgrid(opacity, indices):
    """Copy a validated Diffgrid object and slice its stored layer arrays."""
    indices = np.asarray(indices)
    info = opacity.diffgrid_info
    if (indices.ndim != 1 or indices.size < 2 or indices.dtype.kind not in "iu"
            or indices[0] < 0 or indices[-1] >= len(info.pressure_grid)
            or np.any(np.diff(indices) <= 0)):
        raise ValueError("Choose at least two increasing native pressure indices.")
    if hasattr(opacity, "pressure_subset"):
        return opacity.pressure_subset(indices)
    subset = copy(opacity)
    subset.aux = dict(opacity.aux)
    subset.user_meta = dict(opacity.user_meta)
    subset.user_meta["validation_pressure_subset"] = indices.tolist()
    subset._apply_diffgrid_info(replace(
        info, pressure_grid=info.pressure_grid[indices],
        log_cross_section_grid=info.log_cross_section_grid[indices],
        log_cross_section_derivative_grid=info.log_cross_section_derivative_grid[indices],
    ))
    return subset


def inner_refinement_grids(boundaries):
    """Keep both endpoint layers; bisect the inner 30 native layers twice."""
    native = native_pressure_grids(boundaries, "vertical")[-1]
    source = np.asarray(native["pressure_centers_bar"])
    original = np.asarray(boundaries, dtype=float)
    current, cases = original.copy(), []
    for factor in (1, 2, 4):
        centers = np.sqrt(current[:-1] * current[1:])
        cases.append(dict(
            name=f"inner_refinement_{len(centers):03d}_layers", inner_split_factor=factor,
            pressure_resample=factor != 1,
            source_pressure_grid_bar=source.tolist(),
            pressure_boundaries_bar=current.tolist(), pressure_centers_bar=centers.tolist(),
            comparison_boundaries_bar=original.tolist(),
            grid_scope="Inner layers refined within the fixed log-pressure opacity model; both endpoint layers unchanged. Neither pressure-opacity interpolation accuracy nor endpoint-layer convergence is certified.",
            initial_state_scope="Interpolate log T in log P from the same native seed. Project each connection mask using its log-pressure midpoint and the original center-to-center/bottom connections (right interval at ties); the mask is only an initialization.",
        ))
        if factor != 4:
            current = np.sort(np.concatenate((current, np.sqrt(current[1:-2] * current[2:-1]))))
    return cases


def opacity_pressure_grids(boundaries):
    """Keep RT geometry fixed while restoring held-out native opacity nodes."""
    native = native_pressure_grids(boundaries, "vertical")[-1]
    cases = []
    for step in (4, 2, 1):
        indices = np.unique(np.r_[np.arange(0, 32, step), 31])
        cases.append(dict(
            name=f"opacity_pressure_{len(indices):02d}_nodes",
            opacity_pressure_indices=indices.tolist(),
            source_pressure_grid_bar=native["pressure_centers_bar"],
            pressure_boundaries_bar=list(boundaries),
            comparison_boundaries_bar=list(boundaries),
            grid_scope="Opacity pressure nodes restored from 9 to 17 to 32; both endpoints, all 32 physical layers, temperature nodes and spectral grids unchanged. This compares held-out native teacher values, not unknown values between the full native nodes.",
            initial_state_scope="Identical native initial temperature, bottom temperature and connection mask in every case; independently reconverged.",
        ))
    return cases


def opacity_pressure_column(base, case):
    """Slice CKD in memory and resample the same LBL nodes during streaming."""
    stream = getattr(base, "resolved_stream", None)
    if stream is None:
        raise ValueError("Opacity pressure refinement requires --stream-resolved")
    indices = np.asarray(case["opacity_pressure_indices"])
    source = np.asarray(next(iter(base.lbl.values())).diffgrid_info.pressure_grid)
    if (source.shape != (32,) or indices.ndim != 1 or indices.dtype.kind not in "iu"
            or indices.size < 2 or indices[0] != 0 or indices[-1] != 31
            or np.any(indices[1:] <= indices[:-1])
            or not np.array_equal(case["pressure_boundaries_bar"], np.asarray(base.boundaries))
            or not np.allclose(source, base.pressure, rtol=1e-12, atol=0)
            or not np.allclose(source, case["source_pressure_grid_bar"], rtol=1e-12, atol=0)):
        raise ValueError("Opacity pressure refinement requires the unchanged native32 grid and both opacity endpoints")
    for name, opacity in base.ckd.items():
        if (not np.array_equal(opacity.ckd_info.P_grid, source)
                or not np.array_equal(base.lbl[name].diffgrid_info.pressure_grid, source)):
            raise ValueError("CKD and resolved native pressure nodes must match exactly")
    if np.array_equal(indices, np.arange(32)):
        return base
    ckd = {}
    for name, opacity in base.ckd.items():
        selected = copy(opacity)
        info = opacity.ckd_info
        selected.ckd_info = replace(info, P_grid=info.P_grid[indices],
                                    log_kggrid=info.log_kggrid[:, indices])
        ckd[name] = selected
    lbl = {name: opacity.pressure_subset(indices).pressure_resample(source)
           for name, opacity in base.lbl.items()}
    variant = Wasp18bColumn(
        base.observation, base.stellar, ckd, lbl, base.cia, base.boundaries, base.settings,
        isotope_fractions=base.isotope_fractions, temperature_initial=base.solver_arguments[2],
        bottom_temperature_initial=base.solver_arguments[3], solver_options=dict(base.solver_options),
    )
    variant.resolved_stream = stream.pressure_subset(indices).pressure_resample(source)
    return variant


def opacity_temperature_grids(boundaries):
    """Restore stored inverse-temperature knots without moving any RT layer."""
    native_pressure_grids(boundaries, "vertical")
    cases = []
    for step in (4, 2, 1):
        indices = np.unique(np.r_[np.arange(0, 16, step), 15])
        cases.append(dict(
            name=f"opacity_temperature_{len(indices):02d}_nodes",
            opacity_temperature_indices=indices.tolist(),
            pressure_boundaries_bar=list(boundaries), comparison_boundaries_bar=list(boundaries),
            grid_scope="Opacity temperature nodes restored from 5 to 9 to 16; both endpoint temperatures, native pressure nodes, 32 physical layers and spectral grids unchanged. This compares held-out native teacher values, not unknown values between the full native temperature nodes.",
            initial_state_scope="Identical native initial temperature, bottom temperature and connection mask in every case; independently reconverged.",
        ))
    return cases


def opacity_temperature_column(base, case):
    """Select identical physical temperatures in ascending CKD and descending LBL."""
    stream = getattr(base, "resolved_stream", None)
    if stream is None:
        raise ValueError("Opacity temperature refinement requires --stream-resolved")
    indices = np.asarray(case["opacity_temperature_indices"])
    source = np.asarray(next(iter(base.lbl.values())).diffgrid_info.temperature_grid)
    if (source.shape != (16,) or indices.ndim != 1 or indices.dtype.kind not in "iu"
            or indices.size < 2 or indices[0] != 0 or indices[-1] != 15
            or np.any(indices[1:] <= indices[:-1]) or np.any(np.diff(source) >= 0)
            or np.shape(base.pressure) != (32,)
            or not np.array_equal(case["pressure_boundaries_bar"], np.asarray(base.boundaries))):
        raise ValueError("Opacity temperature refinement requires native32 layers and both native16 temperature endpoints")
    ckd_indices = len(source) - 1 - indices[::-1]
    for name, opacity in base.ckd.items():
        if (not np.array_equal(opacity.ckd_info.T_grid, source[::-1])
                or not np.array_equal(base.lbl[name].diffgrid_info.temperature_grid, source)):
            raise ValueError("CKD and resolved native temperature values must match exactly")
    if np.array_equal(indices, np.arange(16)):
        return base
    ckd = {}
    for name, opacity in base.ckd.items():
        selected = copy(opacity)
        info = opacity.ckd_info
        selected.ckd_info = replace(info, T_grid=info.T_grid[ckd_indices],
                                    log_kggrid=info.log_kggrid[ckd_indices])
        ckd[name] = selected
    lbl = {name: opacity.temperature_subset(indices) for name, opacity in base.lbl.items()}
    variant = Wasp18bColumn(
        base.observation, base.stellar, ckd, lbl, base.cia, base.boundaries, base.settings,
        isotope_fractions=base.isotope_fractions, temperature_initial=base.solver_arguments[2],
        bottom_temperature_initial=base.solver_arguments[3], solver_options=dict(base.solver_options),
    )
    variant.resolved_stream = stream.temperature_subset(indices)
    return variant


def inner_refinement_column(base, case):
    """Resample metadata now and coefficient chunks only during streaming."""
    stream = getattr(base, "resolved_stream", None)
    if stream is None:
        raise ValueError("Interior pressure refinement requires --stream-resolved")
    boundaries = np.asarray(case["pressure_boundaries_bar"])
    source = np.asarray(base.pressure)
    if (not np.array_equal(boundaries[[0, 1, -2, -1]], np.asarray(base.boundaries)[[0, 1, -2, -1]])
            or not np.allclose(source, case["source_pressure_grid_bar"], rtol=1e-12, atol=0)):
        raise ValueError("Interior refinement must retain the native endpoint layers and source grid")
    pressure = np.sqrt(boundaries[:-1] * boundaries[1:])
    refined_stream = stream.pressure_resample(pressure)
    lbl = {name: opacity.pressure_resample(pressure) for name, opacity in base.lbl.items()}
    initial = case["initial_temperature_K"]
    seed = np.asarray(base.solver_arguments[2])
    temperature = (np.exp(np.interp(np.log(pressure), np.log(source), np.log(seed)))
                   if initial is None else np.full(len(pressure), initial))
    if initial is None:
        temperature[[0, -1]] = seed[[0, -1]]
    bottom = base.solver_arguments[3] if initial is None else initial
    solver_options = dict(base.solver_options)
    mask = solver_options.get("convective_mask_initial")
    if mask is not None:
        original_nodes = np.append(source, np.asarray(base.boundaries)[-1])
        nodes = np.append(pressure, boundaries[-1])
        midpoint = .5 * (np.log(nodes[:-1]) + np.log(nodes[1:]))
        connection = np.searchsorted(np.log(original_nodes), midpoint, side="right") - 1
        if np.shape(mask) != source.shape or np.any(connection < 0) or np.any(connection >= len(source)):
            raise ValueError("Invalid original connection mask for interior refinement")
        solver_options["convective_mask_initial"] = np.asarray(mask)[connection].copy()
    variant = Wasp18bColumn(
        base.observation, base.stellar, base.ckd, lbl, base.cia,
        boundaries, replace(base.settings, **case["settings"]),
        isotope_fractions=base.isotope_fractions, temperature_initial=temperature,
        bottom_temperature_initial=bottom, solver_options=solver_options,
    )
    variant.resolved_stream = refined_stream
    return variant


def variant_column(base, case):
    """Reuse loaded opacity arrays; change only the stated numerical control."""
    if "opacity_pressure_indices" in case:
        return opacity_pressure_column(base, case)
    if "opacity_temperature_indices" in case:
        return opacity_temperature_column(base, case)
    if case.get("pressure_resample", False):
        return inner_refinement_column(base, case)
    initial = case["initial_temperature_K"]
    boundaries = np.asarray(case.get("pressure_boundaries_bar", base.boundaries))
    indices = np.asarray(case.get("source_pressure_indices", np.arange(base.pressure.size)))
    unchanged_grid = (np.array_equal(indices, np.arange(base.pressure.size))
                      and np.array_equal(boundaries, np.asarray(base.boundaries)))
    stream = getattr(base, "resolved_stream", None)
    if stream is not None and not unchanged_grid:
        stream = stream.pressure_subset(indices)
    if initial is None and not case["settings"] and unchanged_grid:
        return base
    if not np.allclose(np.asarray(base.pressure)[indices],
                       np.sqrt(boundaries[:-1] * boundaries[1:]), rtol=1e-12, atol=0.0):
        raise ValueError("Column variant must retain stored opacity pressure centers.")
    lbl = base.lbl if unchanged_grid else {
        name: subset_diffgrid(opacity, indices) for name, opacity in base.lbl.items()
    }
    solver_options = dict(base.solver_options)
    mask = solver_options.get("convective_mask_initial")
    if mask is not None:
        # Coarse connections differ from native ones. This sliced mask is only
        # an initialization; the final solver must verify complementarity.
        solver_options["convective_mask_initial"] = np.asarray(mask)[indices].copy()
    temperature = np.asarray(base.solver_arguments[2])
    bottom = base.solver_arguments[3] if initial is None else initial
    if initial is None and boundaries[-1] != np.asarray(base.boundaries)[-1]:
        bottom = float(np.exp(np.interp(
            np.log(boundaries[-1]),
            np.log(np.append(np.asarray(base.pressure), np.asarray(base.boundaries)[-1])),
            np.log(np.append(temperature, base.solver_arguments[3])),
        )))
    variant = Wasp18bColumn(
        base.observation, base.stellar, base.ckd, lbl, base.cia,
        boundaries, replace(base.settings, **case["settings"]),
        isotope_fractions=base.isotope_fractions,
        temperature_initial=(temperature[indices] if initial is None
                             else np.full(indices.size, initial)),
        bottom_temperature_initial=bottom,
        solver_options=solver_options,
    )
    variant.resolved_stream = stream
    return variant


class PreparedValidationRun:
    """Compile each calculation once per column and perform no warm repeats."""

    def __init__(self, column, parameters, response_resolution=None, *,
                 mask_continuation=False):
        self.column = column
        self.response_resolution = response_resolution
        column.check_response_coverage(response_resolution)
        started = perf_counter()
        self.continuation = None
        if mask_continuation:
            self.continuation = PreparedMaskContinuation(column, parameters)
            self.solve = None
            self.solve_compile_seconds = self.continuation.compile_seconds
        else:
            self.solve = jax.jit(column.solve).lower(jnp.asarray(parameters, dtype=float)).compile()
            self.solve_compile_seconds = perf_counter() - started
        self.chemistry = jax.jit(column.chemistry)
        self.resolved_stream = getattr(column, "resolved_stream", None)
        self.resolved_tables, self.spectrum = None, None
        if self.resolved_stream is None:
            self.resolved_tables = column.resolved_opacity_tables()
            self.spectrum = jax.jit(lambda t, b, p, c, tables: column.spectrum(
                t, b, p, c, resolving_power=response_resolution, resolved_tables=tables))
        self.fluxes = jax.jit(column.fluxes)
        self.stellar_absorption = (jax.jit(column.optical_depth)
                                   if column.settings.stellar_scattering else None)
        self.compiled_radiation = None
        self.compiled_chemistry = None
        check_temperatures = jnp.array([1500.0, 2500.0, 3500.0, 4500.0])
        integral = band_planck_flux(check_temperatures, column.band_nodes,
                                   column.band_weights) @ column.band_widths
        self.blackbody = serializable({
            "temperature_K": check_temperatures,
            "relative_error": integral / (SIGMA_SB * check_temperatures**4) - 1,
        })

    def run(self, parameters, on_state=None):
        column = self.column
        parameters = jnp.asarray(parameters, dtype=float)
        started = perf_counter()
        continuation_report = {}
        if self.continuation is None:
            state = jax.block_until_ready(self.solve(parameters))
        else:
            history = []
            metadata = {
                "max_stages": self.continuation.max_stages,
                "solver_options": self.continuation.solver_options,
                "initial_state_policy": "Fixed column seed for every point; within-point continuation only",
            }

            def save_stage(stage):
                history.append(stage)
                if on_state is not None:
                    on_state(serializable({
                        "mode": "rce", "parameters": parameters, "success": False,
                        "state": stage["state"], "status_name": stage["status_name"],
                        "mask_continuation": dict(metadata, stop_reason="running", stages=history),
                    }))

            continued = self.continuation.run(parameters, on_stage=save_stage)
            state = continued.state
            continuation_report["mask_continuation"] = dict(
                metadata, stop_reason=continued.stop_reason, stages=continued.stages,
            )
        elapsed = perf_counter() - started
        nodes = jnp.append(state.temperature, state.bottom_temperature)
        result = {
            "mode": "rce", "parameters": parameters, "success": False,
            "state": state._asdict(), "solve_seconds": elapsed,
            "status_name": RceStatus(int(state.status)).name,
            "max_energy_residual_cgs": jnp.max(jnp.abs(state.flux_residual)),
            "max_stability_residual": jnp.max(state.stability_residual),
            "temperature_min_K": jnp.min(nodes), "temperature_max_K": jnp.max(nodes),
            "temperature_domain_K": column.temperature_range,
            **continuation_report,
        }
        if column.settings.stellar_scattering:
            from _rce_wasp18b_audit import stellar_scattering_description

            result["stellar_scattering"] = stellar_scattering_description(column.settings)
        if on_state is not None:
            on_state(serializable(result))
        if not bool(column.temperature_valid(state.temperature, state.bottom_temperature, parameters)):
            return serializable(dict(result, failure="outside_temperature_or_parameter_domain"))
        if self.compiled_chemistry is None:
            started = perf_counter()
            self.compiled_chemistry = self.chemistry.lower(nodes, *parameters[:2]).compile()
            result["chemistry_compile_seconds"] = perf_counter() - started
        chemistry = jax.block_until_ready(self.compiled_chemistry(nodes, *parameters[:2]))
        result["chemistry"] = chemistry.diagnostics._asdict()
        if not bool(state.converged):
            return serializable(dict(result, failure="rce_not_converged"))
        if not bool(jnp.all(chemistry.diagnostics.valid)):
            return serializable(dict(result, failure="invalid_chemistry"))
        arguments = (state.temperature, state.bottom_temperature, parameters, chemistry)
        if self.compiled_radiation is None:
            started = perf_counter()
            self.compiled_radiation = (
                (self.spectrum.lower(*arguments, self.resolved_tables).compile()
                 if self.resolved_stream is None else None),
                self.fluxes.lower(*arguments).compile(),
            )
            result["radiation_compile_seconds"] = perf_counter() - started
        spectrum_fn, flux_fn = self.compiled_radiation
        started = perf_counter()
        if self.resolved_stream is None:
            prediction = jax.block_until_ready(spectrum_fn(*arguments, self.resolved_tables))
        else:
            top_flux = self.resolved_stream.top_flux(column, *arguments)
            prediction = jax.block_until_ready(self.resolved_stream.prediction(
                column, top_flux, self.response_resolution))
            result["resolved_streaming"] = dict(self.resolved_stream.provenance,
                timing_scope="radiation_seconds includes chunk verification, disk reads, "
                             "transfer and any first-use chunk JIT compilation; "
                             "radiation_compile_seconds covers CKD radiation only")
        broadband = jax.block_until_ready(integrate_broadband_fluxes(
            flux_fn(*arguments), column.g_weights, column.band_widths))
        result["radiation_seconds"] = perf_counter() - started
        if not bool(jnp.all(jnp.isfinite(prediction)) & jnp.all(jnp.isfinite(broadband.net))):
            return serializable(dict(result, failure="nonfinite_radiation"))
        result.update(
            success=True, pressure_bar=column.pressure, temperature_K=state.temperature,
            pressure_boundaries_bar=column.solver_arguments[1],
            bottom_temperature_K=state.bottom_temperature, prediction=prediction,
            broadband_flux=broadband._asdict(),
            bottom_incident_fraction=bottom_leakage_fraction(broadband),
            bottom_incident_over_internal_flux=broadband.incident[-1] / column.internal_flux,
            response_resolution=self.response_resolution, blackbody_integral=self.blackbody,
        )
        if self.stellar_absorption is not None:
            from _rce_wasp18b_audit import stellar_flux_audit

            result["stellar_scattering"]["stellar_flux_audit"] = stellar_flux_audit(
                column, state.temperature, parameters, chemistry,
                absorption_depth=self.stellar_absorption(state.temperature, chemistry))
        return serializable(result)


def compare_inner_runs(current, reference):
    """Compare radiation/convection on all 33 unchanged native boundaries."""
    result = compare_runs(current, reference)
    boundaries = np.asarray(current["case"]["comparison_boundaries_bar"])
    if not np.array_equal(boundaries, reference["case"]["comparison_boundaries_bar"]):
        raise ValueError("Interior comparisons require the same native boundaries")
    selected = []
    for run in (current, reference):
        grid = np.asarray(run["pressure_boundaries_bar"])
        indices = np.searchsorted(grid, boundaries)
        if np.any(indices >= len(grid)) or not np.array_equal(grid[indices], boundaries):
            raise ValueError("Interior run is missing a native comparison boundary")
        selected.append(indices)
    changes = {key: np.asarray(current["broadband_flux"][key])[selected[0]]
               - np.asarray(reference["broadband_flux"][key])[selected[1]]
               for key in ("upward", "downward", "incident", "net")}
    changes["convective"] = (np.asarray(current["state"]["convective_flux"])[selected[0]]
                              - np.asarray(reference["state"]["convective_flux"])[selected[1]])
    changes["total_net"] = changes["net"] + changes["convective"]
    result["common_boundary_flux"] = dict(
        pressure_bar=boundaries, change_cgs=changes,
        max_absolute_change_cgs={key: np.max(np.abs(value)) for key, value in changes.items()},
    )
    result["scope"] = (current["case"].get("grid_scope", "Interior RT/RCE layer refinement")
                       + " Spectral budgets alone do not certify temperature/flux convergence.")
    return serializable(result)


def checkpoint(path, report):
    """Atomically replace the report after every status change and solved case."""
    report["updated_utc"] = datetime.now(timezone.utc).isoformat()
    runs = report["runs"]
    report["summary"] = {
        "planned": len(report["cases"]), "finished": len(runs),
        "successful": sum(bool(run.get("success")) for run in runs),
        "failed": sum(not bool(run.get("success")) for run in runs),
        "statuses": dict(Counter(run.get("status_name", "EXCEPTION") for run in runs)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(serializable(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True,
                        choices=("prior", "initial", "quadrature", "vertical", "depth", "inner_refinement", "opacity_pressure", "opacity_temperature"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-temperatures", type=float, nargs="+", default=[2000., 2500., 4000.])
    parser.add_argument("--response-resolution", type=float)
    parser.add_argument("--stream-resolved", action="store_true",
                        help="Stream prepared resolved chunks, including exact native pressure subsets.")
    parser.add_argument("--mask-continuation", action="store_true",
                        help="Use bounded within-point mask continuation and record every stage.")
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    report = {
        "schema_version": "exojax.rce-wasp18b-validation/1", "suite": args.suite,
        "status": "loading", "p7_physical_acceptance": "not_certified",
        "scope": ("Finite diagnostics of the supplied opacity model; native pressure subsets or explicit interior log-pressure resampling, "
                  "without new opacity information or prior certification. Grid suites compare "
                  "successive successful cases and report every failed case."),
        "manifest_path": str(args.manifest.resolve()),
        "stream_resolved": args.stream_resolved,
        "mask_continuation": args.mask_continuation,
        "observation_path": str(args.observation.resolve()),
        "cases": [], "runs": [],
    }
    checkpoint(args.output, report)
    try:
        if args.suite in ("inner_refinement", "opacity_pressure", "opacity_temperature") and not args.stream_resolved:
            raise ValueError("Opacity/pressure refinement requires --stream-resolved")
        manifest = load_manifest(args.manifest)
        observation = load_emission_candidate(args.observation)
        cases = make_cases(manifest.config, args.suite, args.initial_temperatures)
        report.update(cases=cases, configuration=manifest.config, asset_sha256=manifest.sha256,
                      observation_sha256=observation.sha256, row_indices=observation.row_indices,
                      uncertainty=observation.uncertainty, provenance=provenance(), status="preparing")
        checkpoint(args.output, report)
        base = prepare_column(manifest, observation, stream_resolved=args.stream_resolved)
        report["temperature_domain_K"] = base.temperature_range
        evaluator = None
        reference = None
        for case in cases:
            report.update(status="running", current_case=case["name"])
            checkpoint(args.output, report)
            column = base if args.suite == "prior" else variant_column(base, case)
            if evaluator is None or args.suite != "prior":
                evaluator = PreparedValidationRun(
                    column, case["parameters"], args.response_resolution,
                    mask_continuation=args.mask_continuation,
                )
            def save_state(state_report):
                report["last_solved_state"] = dict(state_report, case=case)
                checkpoint(args.output, report)

            result = evaluator.run(case["parameters"], on_state=save_state)
            result.update(case=case, settings=asdict(column.settings), solver_options=column.solver_options,
                          initial_temperature_K=serializable(column.solver_arguments[2]),
                          bottom_temperature_initial_K=serializable(column.solver_arguments[3]),
                          solve_compile_seconds=evaluator.solve_compile_seconds,
                          observation_sha256=observation.sha256, row_indices=observation.row_indices,
                          uncertainty=serializable(observation.uncertainty),
                          physical_inputs=physical_inputs(manifest, column, args.response_resolution))
            result = serializable(result)
            if result["success"] and args.suite != "prior":
                if reference is None:
                    reference = result
                else:
                    result["comparison_reference_case"] = reference["case"]["name"]
                    result["comparison"] = (compare_inner_runs(result, reference)
                                            if args.suite in ("inner_refinement", "opacity_pressure", "opacity_temperature") else compare_runs(result, reference))
                if args.suite in ("vertical", "depth", "inner_refinement", "opacity_pressure", "opacity_temperature"):
                    reference = result
            report["runs"].append(result)
            report["last_solved_state"] = None
            checkpoint(args.output, report)
            print(f"{case['name']}: {result['status_name']}; success={result['success']}", flush=True)
            if args.suite != "prior":
                del evaluator
                evaluator = None
                del column
                jax.clear_caches()
                gc.collect()
        report.update(status="finished", current_case=None)
        checkpoint(args.output, report)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
        report.update(status="interrupted", failure=f"{type(error).__name__}: {error}")
        checkpoint(args.output, report)
        print(f"FAILED: {report['failure']}", flush=True)
        return 1
    print(f"Saved {len(report['runs'])} cases: {args.output}", flush=True)
    return int(any(not run["success"] for run in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())

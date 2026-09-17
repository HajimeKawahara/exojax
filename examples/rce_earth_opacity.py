"""Prepare HITRAN H2O line CKD tables for the forward Earth RCE example.

ExoJAX evaluates Voigt profiles within 25 cm-1 of each line and subtracts
the Lorentz value at that cutoff, as required when adding the MT_CKD water
continuum. Air broadening approximates the N2 background; line self
broadening is omitted. The continuum is added separately in the RCE model.
Only the main water isotopologue is included, with HITRAN line intensities
retaining their terrestrial isotopic abundance convention.
"""

import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time

import jax
import jax.numpy as jnp
import numpy as np

from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.opacity.ckd.api import OpaCKD
from exojax.opacity.ckd.contracts import CKDTableInfo
from exojax.opacity.ckd.core import compute_ckd_from_xsmatrix, gauss_legendre_grid
from exojax.opacity.lpf.lpf import vvoigt
from exojax.special.faddeeva import asymptotic_wofz


LINE_CUTOFF = 25.0  # cm-1
LINE_BATCH = 128
G_INTERVALS = np.array([0.0, 0.9, 0.99, 0.999, 0.9999, 1.0])


def download_water(data_dir, nu_max=30000.0):
    """Cache the public HITRAN response, recording its source and checksum."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "water.par"
    metadata_path = data_dir / "water.metadata.json"
    requested_range = [0.0, max(30000.0, nu_max) + LINE_CUTOFF]
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        cached_range = metadata["requested_wavenumber_range_cm-1"]
        if cached_range[0] <= requested_range[0] and cached_range[1] >= requested_range[1]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
                raise ValueError("The cached HITRAN file does not match its checksum.")
            return path, metadata

    import requests

    response = requests.get(
        "https://hitran.org/lbl/api",
        params={"iso_ids_list": 1, "numin": requested_range[0], "numax": requested_range[1]},
        timeout=120,
    )
    response.raise_for_status()
    lines = response.text.splitlines()
    if not lines or any(len(line) != 160 or line[:3] != " 11" for line in lines):
        raise ValueError("Expected HITRAN 160-character H2-16O records from the public API.")
    metadata = {
        "source_url": response.url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "bytes": len(response.content),
        "line_count": len(lines),
        "requested_wavenumber_range_cm-1": requested_range,
        "actual_wavenumber_range_cm-1": [float(lines[0][3:15]), float(lines[-1][3:15])],
        "molecule": "H2O",
        "hitran_global_isotopologue_id": 1,
        "release": "Live HITRANonline response; release not identified in the response",
    }
    path.write_bytes(response.content)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    # RADIS detects modification times, but avoid a stale parsed cache explicitly.
    path.with_suffix(".h5").unlink(missing_ok=True)
    return path, metadata


@jax.jit
def _chunk_cross_sections(nu_grid, line_data, states, core_offsets=None):
    """Evaluate a fixed line batch while keeping one atmospheric state in memory."""
    nu, logsij, elower, gamma_air, n_air, einstein_a = line_data
    detuning = nu_grid[None, :] - nu[:, None]

    def at_state(state):
        temperature, pressure, partition_ratio, molmass = state
        strength = line_strength(temperature, logsij, nu, elower, partition_ratio, 296.0)
        gamma = gamma_hitran(pressure, temperature, 0.0, n_air, gamma_air, gamma_air)
        gamma = gamma + gamma_natural(einstein_a)
        sigma = doppler_sigma(nu, temperature, molmass)
        if core_offsets is None:
            profile = vvoigt(detuning, sigma, gamma)
        else:
            # This is the same core/wing evaluation as ExoJAX's Voigt routine.
            # Compute its expensive core only near each line, avoiding a large
            # Faddeeva core calculation that would be discarded in the wings.
            factor = 1.0 / (jnp.sqrt(2.0) * sigma[:, None])
            safe_detuning = jnp.where(jnp.abs(detuning) < 16.0 * sigma[:, None],
                                      16.0 * sigma[:, None], detuning)
            profile = factor * jnp.real(asymptotic_wofz(
                factor * safe_detuning, factor * gamma[:, None]
            )) / jnp.sqrt(jnp.pi)
            spacing = nu_grid[1] - nu_grid[0]
            nearest = jnp.rint((nu - nu_grid[0]) / spacing).astype(int)
            indices = nearest[:, None] + core_offsets[None, :]
            core_detuning = nu_grid[jnp.clip(indices, 0, nu_grid.size - 1)] - nu[:, None]
            core = vvoigt(core_detuning, sigma, gamma)
            indices = jnp.where((indices >= 0) & (indices < nu_grid.size), indices, nu_grid.size)
            profile = profile.at[jnp.arange(nu.size)[:, None], indices].set(core, mode="drop")
        pedestal = gamma / (jnp.pi * (LINE_CUTOFF**2 + gamma**2))
        profile = jnp.where(
            jnp.abs(detuning) <= LINE_CUTOFF,
            jnp.maximum(profile - pedestal[:, None], 0.0),
            0.0,
        )
        return strength @ profile

    return jax.lax.map(at_state, states)


class WaterLineOpacity:
    """Small example adapter for ExoJAX's line strength and Voigt routines."""

    def __init__(self, parfile, temperatures, strength_cutoff=1.0e-28):
        from exojax.database.hitemp.api import MdbHitemp

        # RADIS writes a parsed cache beside the .par file. Isolate that cache
        # so independent table or spectrum calculations can run concurrently.
        with TemporaryDirectory(prefix="exojax-water-") as directory:
            local_parfile = Path(directory) / "water.par"
            shutil.copyfile(parfile, local_parfile)
            self.mdb = MdbHitemp(
                str(Path(directory) / "H2O"),
                parfile=str(local_parfile),
                nurange=[0.0, 40000.0],
                Ttyp=296.0,
                isotope=1,
                gpu_transfer=False,
                engine="pytables",
            )
        self.molmass = float(self.mdb.molmass)
        strengths = np.stack([
            np.asarray(line_strength(
                temperature, self.mdb.logsij0, self.mdb.nu_lines, self.mdb.elower,
                self.mdb.qr_interp(1, temperature, 296.0), 296.0,
            ))
            for temperature in temperatures
        ])
        keep = np.max(strengths, axis=0) >= strength_cutoff
        self.metadata = {
            "strength_cutoff_cm": strength_cutoff,
            "selection_temperatures_K": np.asarray(temperatures).tolist(),
            "original_line_count": int(keep.size),
            "retained_line_count": int(np.count_nonzero(keep)),
            "discarded_strength_fraction_by_temperature": (
                np.sum(strengths[:, ~keep], axis=1) / np.sum(strengths, axis=1)
            ).tolist(),
            "line_cutoff_cm-1": LINE_CUTOFF,
            "pedestal": "Lorentz profile at +/-25 cm-1 subtracted from each retained line",
            "broadening": "HITRAN air broadening as an N2 proxy; line self broadening omitted",
            "isotopologue": "H2-16O; HITRAN terrestrial abundance convention",
            "continuum": "Not included in line table; add MT_CKD separately",
        }
        self.line_data = np.stack([
            np.asarray(getattr(self.mdb, name))[keep]
            for name in ["nu_lines", "logsij0", "elower", "gamma_air", "n_air", "A"]
        ])

    def cross_sections(self, nu_grid, temperatures, pressures):
        """Return line cross sections for paired states, shape (Nstate, Nnu)."""
        temperatures, pressures = np.broadcast_arrays(temperatures, pressures)
        partition_ratios = {
            temperature: float(self.mdb.qr_interp(1, temperature, 296.0))
            for temperature in np.unique(temperatures)
        }
        states = np.array([
            [temperature, pressure, partition_ratios[temperature], self.molmass]
            for temperature, pressure in zip(temperatures.ravel(), pressures.ravel())
        ])
        return self._cross_sections(np.asarray(nu_grid), states)

    def _cross_sections(self, nu_grid, states):
        nu_grid = np.asarray(nu_grid)
        batch_size = 5000
        if nu_grid.size > 1 and np.allclose(np.diff(nu_grid), nu_grid[1] - nu_grid[0]):
            batch_size = min(batch_size, max(2, int(round(LINE_CUTOFF / (nu_grid[1] - nu_grid[0])))))
        if nu_grid.size > batch_size:
            return jnp.concatenate([
                self._cross_sections(nu_grid[start:start + batch_size], states)
                for start in range(0, nu_grid.size, batch_size)
            ], axis=1)
        core_offsets = None
        if nu_grid.size > 1 and np.allclose(np.diff(nu_grid), nu_grid[1] - nu_grid[0]):
            max_sigma = 3.0415595e-7 * np.sqrt(np.max(states[:, 0]) / self.molmass) * (nu_grid[-1] + LINE_CUTOFF)
            half_width = int(np.ceil(16.0 * max_sigma / (nu_grid[1] - nu_grid[0]))) + 2
            half_width = 2 ** int(np.ceil(np.log2(half_width)))
            core_offsets = np.arange(-half_width, half_width + 1)
        line_nu = self.line_data[0]
        first, last = np.searchsorted(
            line_nu, [nu_grid[0] - LINE_CUTOFF, nu_grid[-1] + LINE_CUTOFF]
        )
        result = jnp.zeros((states.shape[0], nu_grid.size))
        for start in range(first, last, LINE_BATCH):
            selected = self.line_data[:, start:min(start + LINE_BATCH, last)]
            padded_size = max(16, 2 ** int(np.ceil(np.log2(selected.shape[1]))))
            padded = np.tile(np.array([1000.0, -np.inf, 0.0, 0.05, 0.7, 0.0])[:, None], (1, padded_size))
            padded[:, :selected.shape[1]] = selected
            result = result + _chunk_cross_sections(nu_grid, padded, states, core_offsets)
        return result


def spectral_bands(nu_min=20.0, nu_max=30000.0):
    """Use 25 cm-1 thermal bands and 100 cm-1 shortwave bands."""
    edges = [nu_min]
    while edges[-1] < nu_max:
        step = 25.0 if edges[-1] < 4000.0 else 100.0
        edges.append(min(edges[-1] + step, 4000.0 if edges[-1] < 4000.0 else nu_max, nu_max))
    return np.column_stack((edges[:-1], edges[1:]))


def _g_quadrature(ng, split_g):
    if split_g:
        if ng % 5:
            raise ValueError("Split g quadrature requires Ng divisible by five.")
        base_g, base_weights = gauss_legendre_grid(ng // 5)
        widths = np.diff(G_INTERVALS)
        return ((G_INTERVALS[:-1, None] + widths[:, None] * base_g).ravel(),
                (widths[:, None] * base_weights).ravel())
    return gauss_legendre_grid(ng)


def prepare_table(opacity, source, output, bands, temperatures, pressures, dnu=None, ng=40,
                  split_g=True, resolution=1.0e6, state_batch=9, check_ng=0):
    """Compute a standard table and optionally a second quadrature from the same spectra."""
    tmesh, pmesh = np.meshgrid(temperatures, pressures, indexing="ij")
    ggrid, weights = _g_quadrature(ng, split_g)
    query_grid = ggrid
    if check_ng:
        check_ggrid, check_weights = _g_quadrature(check_ng, split_g)
        query_grid = jnp.concatenate((ggrid, check_ggrid))
    stored_ng = ng + check_ng
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial.npz")
    fingerprint = hashlib.sha256(json.dumps({
        "source": source, "settings": opacity.metadata, "dnu": dnu, "ng": ng,
        "resolution": resolution, **({"check_ng": check_ng} if check_ng else {}),
        "split_g": split_g, "cdf": "midpoint", "bands": np.asarray(bands).tolist(),
        "temperatures": np.asarray(temperatures).tolist(), "pressures": np.asarray(pressures).tolist(),
    }, sort_keys=True).encode()).hexdigest()
    table = np.full((len(temperatures), len(pressures), stored_ng, len(bands)), np.nan)
    completed = 0
    if partial.exists():
        with np.load(partial, allow_pickle=False) as data:
            if str(data["fingerprint"]) == fingerprint:
                table = data["table"]
                completed = int(data["completed"])
                print(f"Resuming {output} after {completed} bands", flush=True)
    start = time.monotonic()
    for index, (lower, upper) in enumerate(bands):
        if index < completed:
            continue
        count = int(np.ceil((upper - lower) / dnu if dnu is not None else
                            resolution * (upper - lower) / lower))
        nu_grid = lower + (np.arange(count) + 0.5) * (upper - lower) / count
        log_k = np.empty((tmesh.size, stored_ng))
        for state_start in range(0, tmesh.size, state_batch):
            selection = slice(state_start, min(state_start + state_batch, tmesh.size))
            xs = opacity.cross_sections(nu_grid, tmesh.ravel()[selection], pmesh.ravel()[selection])
            # Shift the core's i/N ranks to the midpoints (i+1/2)/N of equal
            # spectral bins, retaining the unshifted physical quadrature grid.
            log_k[selection] = np.asarray(compute_ckd_from_xsmatrix(xs, query_grid - 0.5 / count))
            if count > 50000:
                print(f"Band {index + 1}/{len(bands)}, states {selection.stop}/{tmesh.size}; {time.monotonic() - start:.1f} s", flush=True)
        table[..., index] = log_k.reshape(len(temperatures), len(pressures), stored_ng)
        if not np.all(np.isfinite(table[..., index])):
            raise ValueError(f"Nonfinite opacity in band {lower:g}-{upper:g} cm-1.")
        temporary = partial.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, table=table, completed=index + 1, fingerprint=fingerprint)
        temporary.replace(partial)
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(bands):
            print(f"Band {index + 1}/{len(bands)}: {lower:g}-{upper:g} cm-1; {time.monotonic() - start:.1f} s", flush=True)

    ckd = OpaCKD.load_only()
    ckd.Ng = ng
    ckd.band_width = float(bands[0, 1] - bands[0, 0])
    ckd.band_spacing = "linear"
    ckd.nu_bands = jnp.asarray(np.mean(bands, axis=1))
    ckd.band_edges = jnp.asarray(bands)
    ckd.molmass = opacity.molmass
    ckd.ckd_info = CKDTableInfo(
        jnp.asarray(table[:, :, :ng]), ggrid, weights, jnp.asarray(temperatures), jnp.asarray(pressures),
        ckd.nu_bands, ckd.band_edges,
    )
    ckd._expected_base_meta = {
        "fingerprint_version": 2,
        "class_name": "rce_earth_opacity.WaterLineOpacity",
        "source": source,
        "settings": {
            **opacity.metadata, "spectral_step_cm-1": dnu,
            "minimum_resolving_power": resolution if dnu is None else None,
            "split_g_quadrature": split_g, "empirical_cdf": "midpoint bins",
            "g_intervals": G_INTERVALS.tolist() if split_g else [0.0, 1.0],
        },
    }
    ckd.ready = True
    ckd.save_tables(str(output), overwrite=True)
    if check_ng:
        check = copy.copy(ckd)
        check.Ng = check_ng
        check.ckd_info = replace(ckd.ckd_info, log_kggrid=jnp.asarray(table[:, :, ng:]),
                                 ggrid=check_ggrid, weights=check_weights)
        check.save_tables(str(output.with_name(f"{output.stem}_ng{check_ng}.npz")), overwrite=True)
    partial.unlink(missing_ok=True)
    print(f"Saved {output}; total {time.monotonic() - start:.1f} s", flush=True)
    return ckd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(".database/rce_earth"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--nu-min", type=float, default=20.0)
    parser.add_argument("--nu-max", type=float, default=30000.0)
    parser.add_argument("--dnu", type=float, default=None, help="Override resolving power with fixed spectral spacing in cm-1.")
    parser.add_argument("--resolution", type=float, default=1.0e6)
    parser.add_argument("--state-batch", type=int, default=9)
    parser.add_argument("--strength-cutoff", type=float, default=1.0e-28)
    parser.add_argument("--ng", type=int, default=40)
    parser.add_argument("--check-ng", type=int, default=0,
                        help="Also save this quadrature using the same line spectra (e.g. 80).")
    parser.add_argument("--split-g", action=argparse.BooleanOptionalAction, default=True,
                        help="Resolve the high-k tail with five g subintervals.")
    args = parser.parse_args()
    positive_values = [args.nu_min, args.nu_max, args.resolution]
    if args.dnu is not None:
        positive_values.append(args.dnu)
    if (any(not np.isfinite(value) or value <= 0.0 for value in positive_values)
            or args.nu_min >= args.nu_max or args.ng < 1
            or args.state_batch < 1 or args.check_ng < 0):
        parser.error("Require increasing positive wavenumbers and positive sampling parameters.")
    if args.nu_max + LINE_CUTOFF > 40000.0:
        parser.error("Require nu_max + 25 <= 40000 cm-1 for the line database range.")
    if not np.isfinite(args.strength_cutoff) or args.strength_cutoff < 0.0:
        parser.error("Require a finite, nonnegative line-strength cutoff.")
    if args.split_g and (args.ng % 5 or args.check_ng % 5):
        parser.error("Split g quadrature requires Ng and nonzero check-Ng divisible by five.")
    jax.config.update("jax_enable_x64", True)
    temperatures = np.arange(80.0, 341.0, 20.0)
    pressures = np.geomspace(1.0e-4, 1.0, 9)
    parfile, source = download_water(args.data_dir, args.nu_max)
    opacity = WaterLineOpacity(parfile, temperatures, args.strength_cutoff)
    print(json.dumps(opacity.metadata, indent=2), flush=True)
    output = args.output or args.data_dir / "water_ckd.npz"
    prepare_table(opacity, source, output, spectral_bands(args.nu_min, args.nu_max), temperatures,
                  pressures, args.dnu, args.ng, args.split_g, args.resolution, args.state_batch, args.check_ng)


if __name__ == "__main__":
    main()

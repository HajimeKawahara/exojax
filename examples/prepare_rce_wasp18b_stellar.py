"""Append a recorded Planck tail to a pinned PHOENIX surface spectrum.

All native samples are retained exactly. The Planck amplitude matches the
integrated native flux in a finite wavelength window. A smooth correction
strictly beyond the last native sample connects its endpoint to that tail.
No bolometric renormalization or change to the physical core is performed.
"""

import argparse
import copy
import json
from pathlib import Path
import tempfile

import numpy as np

from _rce_wasp18b_assets import (
    SURFACE_FLUX_UNIT, _sha256, load_manifest, load_stellar_spectrum,
)


VERSION = "exojax.phoenix-planck-tail/1"
REFERENCE = "https://arxiv.org/html/1606.05474v2#S2.SS5"
H_CGS, C_CGS, K_CGS = 6.62607015e-27, 2.99792458e10, 1.380649e-16
SIGMA_CGS = 2 * np.pi**5 * K_CGS**4 / (15 * H_CGS**3 * C_CGS**2)


def planck_surface_flux(nu_cm1, temperature):
    """Return pi B per cm-1 in erg s-1 cm-2 (cm-1)-1, using full Planck."""
    nu = np.asarray(nu_cm1, dtype=float)
    if (not np.isfinite(temperature) or temperature <= 0
            or not np.all(np.isfinite(nu)) or np.any(nu <= 0)):
        raise ValueError("Planck temperature and wavenumbers must be finite and positive")
    x = H_CGS * C_CGS * nu / (K_CGS * temperature)
    # This form also avoids overflow in exp(x) in the Wien limit.
    return 2 * np.pi * H_CGS * C_CGS**2 * nu**3 * np.exp(-x) / (-np.expm1(-x))


def extend_spectrum(spectrum, *, temperature=None, fit_window=(4.5, 5.5),
                    max_wavelength=100.0, blend_width=0.5, tail_samples=4096):
    """Fit an integrated-flux amplitude and append a continuous positive tail.

    Wavelength parameters are in micrometres. The endpoint correction is a
    cubic smoothstep of wavelength over ``blend_width``, starting exactly at
    the final native flux and ending at the fitted Planck function. Thus no
    native sample, including the observed SOSS interval, is replaced.
    """
    nu = np.asarray(spectrum.nu_cm1, dtype=float)
    flux = np.asarray(spectrum.surface_flux_per_cm1, dtype=float)
    if (nu.ndim != 1 or nu.size < 2 or flux.shape != nu.shape
            or not np.all(np.isfinite(nu)) or np.any(nu <= 0)
            or np.any(np.diff(nu) <= 0) or not np.all(np.isfinite(flux))
            or np.any(flux < 0) or not np.any(flux > 0)):
        raise ValueError("Native spectrum requires increasing positive coordinates and finite nonnegative flux")
    temperature_source = "PHOENIX PHXTEFF" if temperature is None else "explicit override"
    if temperature is None:
        temperature = spectrum.metadata.get("teff_K")
    if temperature is None or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("A positive PHOENIX header temperature or --temperature is required")
    window = np.asarray(fit_window, dtype=float)
    if (window.shape != (2,) or not np.all(np.isfinite(window))
            or window[0] <= 0 or window[1] <= window[0]):
        raise ValueError("fit_window must contain two increasing positive wavelengths")
    join = 1e4 / nu[0]
    actual_window = window.copy()
    # The public PHOENIX endpoint is 5.499975 um, described as 5.5 um.
    if actual_window[1] > join and np.isclose(actual_window[1], join, rtol=1e-5):
        actual_window[1] = join
    if actual_window[0] < 1e4 / nu[-1] or actual_window[1] > join:
        raise ValueError("fit_window must lie within the native spectrum")
    if actual_window[0] >= actual_window[1]:
        raise ValueError("fit_window has no native wavelength interval")
    if (not np.isfinite(max_wavelength) or max_wavelength < 100
            or not np.isfinite(blend_width) or blend_width <= 0
            or max_wavelength <= join + blend_width):
        raise ValueError("max_wavelength must be at least 100 um and beyond the positive blend width")
    if isinstance(tail_samples, bool) or not isinstance(tail_samples, (int, np.integer)) or tail_samples < 32:
        raise ValueError("tail_samples must be an integer of at least 32")
    lower, upper = 1e4 / actual_window[::-1]
    fit_nu = np.concatenate(([lower], nu[(nu > lower) & (nu < upper)], [upper]))
    if fit_nu.size < 4:
        raise ValueError("fit_window must contain at least two interior native samples")
    native_fit = float(np.trapz(np.interp(fit_nu, nu, flux), fit_nu))
    planck_fit = float(np.trapz(planck_surface_flux(fit_nu, temperature), fit_nu))
    amplitude = native_fit / planck_fit
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("The integrated fitting flux must be positive and finite")
    endpoint_ratio = float(flux[0] / (amplitude * planck_surface_flux(nu[0], temperature)))
    # Include the blend endpoint explicitly and retain the native join once.
    tail_wave = np.unique(np.concatenate((
        np.geomspace(join, max_wavelength, tail_samples + 1)[1:],
        [join + blend_width],
    )))
    tail_nu = (1e4 / tail_wave)[::-1]
    t = np.clip((tail_wave - join) / blend_width, 0, 1)
    smoothstep = t * t * (3 - 2 * t)
    correction = 1 + (endpoint_ratio - 1) * (1 - smoothstep)
    tail_flux = (amplitude * planck_surface_flux(1e4 / tail_wave, temperature)
                 * correction)[::-1]
    output_nu = np.concatenate((tail_nu, nu))
    output_flux = np.concatenate((tail_flux, flux))
    native_integral = float(np.trapz(flux, nu))
    tail_integral = float(np.trapz(
        np.concatenate((tail_flux, flux[:1])), np.concatenate((tail_nu, nu[:1]))))
    total = native_integral + tail_integral
    metadata = {
        "version": VERSION, "reference": REFERENCE,
        "method": "Full Planck tail with integrated-flux amplitude and an exterior cubic smoothstep join",
        "flux_unit": SURFACE_FLUX_UNIT,
        "temperature_K": float(temperature),
        "temperature_source": temperature_source,
        "requested_fit_window_um": window.tolist(),
        "fit_window_um": actual_window.tolist(),
        "fit_integral_erg_s_cm2": native_fit,
        "amplitude": amplitude,
        "join_wavelength_um": float(join),
        "blend_end_wavelength_um": float(join + blend_width),
        "endpoint_to_fitted_planck_ratio": endpoint_ratio,
        "native_samples_preserved": int(nu.size),
        "tail_samples": int(tail_nu.size),
        "wavelength_range_um": [float(1e4 / nu[-1]), float(max_wavelength)],
        "normalization": "None; native surface flux unchanged, only Planck tail amplitude fitted",
        "native_integral_erg_s_cm2": native_integral,
        "tail_integral_erg_s_cm2": tail_integral,
        "total_integral_erg_s_cm2": total,
        "tail_fraction_of_total": tail_integral / total,
        "total_over_sigma_temperature4": total / (SIGMA_CGS * temperature**4),
        "limitations": "Finite 100 um or requested endpoint; no UV extension, parameter interpolation, or bolometric rescaling. Join choices require sensitivity checks.",
    }
    return output_nu, output_flux, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Input pinned manifest, preserved unchanged.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--fit-window", type=float, nargs=2, default=(4.5, 5.5), metavar=("LOW_UM", "HIGH_UM"))
    parser.add_argument("--max-wavelength", type=float, default=100.0)
    parser.add_argument("--blend-width", type=float, default=0.5)
    parser.add_argument("--tail-samples", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    if manifest.config.get("stellar", {}).get("format") != "phoenix-fits":
        parser.error("Input manifest must identify the native phoenix-fits spectrum")
    spectrum = load_stellar_spectrum(manifest)
    nu, flux, metadata = extend_spectrum(
        spectrum, temperature=args.temperature, fit_window=args.fit_window,
        max_wavelength=args.max_wavelength, blend_width=args.blend_width,
        tail_samples=args.tail_samples,
    )
    native_spec = manifest.config["stellar"]
    metadata["native_stellar_specification"] = native_spec
    metadata["native_sha256"] = {
        key: manifest.sha256[key]
        for key in (native_spec["wavelength_asset"], native_spec["flux_asset"])
    }
    metadata["input_manifest"] = {"path": str(manifest.path), "sha256": manifest.sha256["__manifest__"]}
    metadata["preparation_sha256"] = _sha256(Path(__file__))
    output = args.output_dir.expanduser().resolve()
    paths = {"stellar_extended": output / "stellar_spectrum.npz",
             "stellar_preparation": output / "stellar_preparation.json"}
    working = output / "stellar_manifest.json"
    if set(paths).intersection(manifest.paths):
        parser.error("Input manifest already declares stellar preparation output asset names")
    for path in (*paths.values(), working):
        if path == manifest.path or path in manifest.paths.values():
            parser.error("Outputs must be separate from the input manifest and assets")
        if path.exists():
            parser.error(f"Refusing to replace an existing artifact: {path}")
    if args.dry_run:
        print(json.dumps({"output_manifest": str(working), "metadata": metadata}, indent=2, allow_nan=False))
        return
    output.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(manifest.config)
    config["readiness"] = (
        "Explicit stellar Planck tail prepared; stellar sensitivity, CIA applicability, "
        "omitted absorbers and numerical convergence audits remain pending."
    )
    pending = [item for item in config.get("missing_requirements", [])
               if item != "Stellar surface spectrum covering the complete broadband grid."]
    pending.append("Stellar tail fitting-window/temperature and broadband endpoint sensitivity.")
    config["missing_requirements"] = pending
    # Resolve all existing paths because the new manifest has a new directory.
    for key, path in manifest.paths.items():
        config["files"][key]["path"] = str(path)
    # Publish complete artifacts individually and the usable manifest last.
    with tempfile.TemporaryDirectory(dir=output) as staging:
        staging = Path(staging)
        np.savez_compressed(staging / paths["stellar_extended"].name,
                            nu_cm1=nu, surface_flux_per_cm1=flux,
                            metadata_json=json.dumps(metadata, sort_keys=True, allow_nan=False))
        (staging / paths["stellar_preparation"].name).write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for key, path in paths.items():
            staged = staging / path.name
            config["files"][key] = {"path": str(path), "sha256": _sha256(staged),
                                     "source": "examples/prepare_rce_wasp18b_stellar.py"}
        config["stellar"] = {"format": "npz", "version": VERSION,
                              "asset": "stellar_extended", "flux_unit": SURFACE_FLUX_UNIT,
                              "metadata": metadata}
        staged_manifest = staging / working.name
        staged_manifest.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for path in paths.values():
            (staging / path.name).replace(path)
        staged_manifest.replace(working)
    print(f"Prepared stellar spectrum: {working}")


if __name__ == "__main__":
    main()

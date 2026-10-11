"""Audit molecular width recipes on a fixed column without computing opacity.

Uses small deterministic windows of existing transition caches. Sample fallback
fractions are exact for those rows, not estimates of whole-line-list fractions.
Neither this diagnostic nor coefficient ratios bound radiative-transfer errors.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np


DATABASES = {
    "H2O": ("H2O/1H2-16O/POKAZATEL", ["00200-00300", "04000-04100", "10000-10100"]),
    "CO": ("CO/12C-16O/Li2015", [None]),
    "CO2": ("CO2/12C-16O2/UCL-4000", ["00000-01000", "04000-05000", "10000-11000"]),
    "TiO": ("TiO/48Ti-16O/Toto", [None]),
    "SiO": ("SiO/28Si-16O/SiOUVenIR", [None]),
}
SOURCES = {
    "diet": "https://doi.org/10.1016/j.jqsrt.2017.01.028",
    "single_J0_and_high_J_rule": "https://doi.org/10.3847/1538-4365/ad9b19",
    "H2O": "https://exomol.com/data/molecules/H2O/",
    "CO": "https://exomol.com/data/molecules/CO/",
    "CO2": "https://exomol.com/data/molecules/CO2/",
    "TiO": "https://exomol.com/data/molecules/TiO/",
    "SiO": "https://exomol.com/data/molecules/SiO/",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipes(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        row = line.split()
        if row:
            key = (row[0], *map(int, row[3:]))
            result[key] = (float(row[1]), float(row[2]))
    if not result or any(v[0] <= 0 or not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Require positive finite recipe widths and finite exponents")
    return result


def defaults(path):
    lines = Path(path).read_text().splitlines()
    return tuple(float(next(line for line in lines if label in line).split()[0])
                 for label in ("Default value of Lorentzian half-width", "Default value of temperature exponent"))


def current_mapping(rows, recipe, default):
    """Reproduce the pinned legacy RADIS a0/a1 lookup, including fallback."""
    codes = {key[0] for key in recipe}
    alpha = np.full(len(rows), default[0])
    exponent = np.full(len(rows), default[1])
    explicit = np.zeros(len(rows), dtype=bool)
    if codes not in ({"a0"}, {"a1"}, {"a0", "a1"}):
        return alpha, exponent, explicit
    for code in ("a0", "a1"):
        for key, value in recipe.items():
            if key[0] != code:
                continue
            chosen = np.asarray(rows.jlower) == key[1]
            if code == "a1":
                chosen &= np.asarray(rows.jupper) == key[2]
            alpha[chosen], exponent[chosen], explicit[chosen] = value[0], value[1], True
    return alpha, exponent, explicit


def native_mapping(rows, path, default, radis):
    """Call the installed production mapper directly without loading an Mdb."""
    signature = inspect.signature(radis.MdbExomol.set_broadening_coef)
    if "species" in signature.parameters:
        raise ValueError("This audit pins the legacy production RADIS mapper; review a changed backend")
    state = SimpleNamespace(alpha_ref_def=default[0], n_Texp_def=default[1],
                            broad_file=str(path), broadf=True, verbose=0)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        radis.MdbExomol.set_broadening_coef(state, rows.copy(), add_columns=False)
    return np.asarray(state.alpha_ref), np.asarray(state.n_Texp), [str(item.message) for item in messages]


def documented_a0_tail(rows, recipe, default):
    """Apply the published highest-J continuation after the same a0/a1 map.

    This is an explicit alternative prescription, never a production mutation.
    Signed/unsigned m0 semantics need a separate transition mapping review.
    """
    if not {key[0] for key in recipe} <= {"a0", "a1"}:
        return None
    alpha, exponent, _ = current_mapping(rows, recipe, default)
    maximum = max(key[1] for key in recipe if key[0] == "a0")
    tail = np.asarray(rows.jlower) > maximum
    alpha[tail], exponent[tail] = recipe[("a0", maximum)]
    return alpha, exponent


def width(alpha, exponent, temperature, pressure):
    return np.asarray(alpha)[None, :] * (296. / np.asarray(temperature)[:, None])**np.asarray(exponent)[None, :] * np.asarray(pressure)[:, None]


def recipe_comparison(h2, he, temperature, pressure, fractions):
    """Compare shared published keys, retaining actual partial pressures."""
    fractions = np.asarray(fractions)
    if fractions.shape != (len(temperature), 3) or np.any(fractions < 0) or np.any(fractions.sum(axis=1) > 1 + 1e-12):
        raise ValueError("Use nonnegative H/He/H2 fractions of all particles without renormalization")
    keys = sorted(set(h2) & set(he))
    a2, n2 = np.array([h2[key] for key in keys]).T
    ae, ne = np.array([he[key] for key in keys]).T
    gamma2, gammae = width(a2, n2, temperature, pressure), width(ae, ne, temperature, pressure)
    known = fractions[:, 2, None] * gamma2 + fractions[:, 1, None] * gammae
    ratio = known / gamma2
    return dict(keys=[list(key) for key in keys],
                gamma_H2_pure_cm1=gamma2.tolist(), gamma_He_pure_cm1=gammae.tolist(),
                gamma_known_H2_He_cm1=known.tolist(), known_over_pure_H2=ratio.tolist(),
                ratio_range=[float(ratio.min()), float(ratio.max())],
                H_coefficient=None, renormalized_fractions=False,
                scope="Shared recipe keys only; no missing-H coefficient or radiative bound. CO2 m0 keys are not mapped onto transitions here.")


def sample_cache(path, window):
    import pandas as pd
    path = Path(path)
    before = path.stat()
    frames, selections = [], []
    with pd.HDFStore(path, "r") as store:
        count = int(store.get_storer("df").nrows)
        starts = sorted({0, max(0, count // 2 - window // 2), max(0, count - window)})
        stop_previous = 0
        for start in starts:
            start = max(start, stop_previous)
            stop = min(count, start + window)
            if start >= stop:
                continue
            frame = store.select("df", start=start, stop=stop,
                                 columns=["nu_lines", "jlower", "jupper", "Sij0", "elower"])
            digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
            selections.append(dict(start=start, stop=stop, sample_sha256=digest))
            frames.append(frame)
            stop_previous = stop
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Cache changed while reading: {path}")
    frame = pd.concat(frames)
    selected = ((frame.nu_lines > 200) & (frame.nu_lines < 50000) & (frame.Sij0 > 0)
                & np.isfinite(frame.to_numpy()).all(axis=1))
    return frame[selected], dict(path=str(path), size=before.st_size, mtime_ns=before.st_mtime_ns,
        total_cache_rows=count, windows=selections, selected_rows=int(selected.sum()),
        selection="200 < nu < 50000; positive stored reference strength; finite mandatory values. No whole-file hash or whole-list fallback estimate.")


def run(args):
    import pandas as pd
    import radis.api.exomolapi as radis
    state = json.loads(args.state.read_text())
    temperature = np.asarray(state["temperature_K"])
    pressure = np.asarray(state["pressure_bar"])
    fractions = np.asarray(state["fractions_H_He_H2"])
    result = dict(status="coefficient_diagnostic_only", certifies_p7=False, sources=SOURCES,
        code_sha256={str(Path(__file__).resolve()): sha256(__file__), str(Path(radis.__file__)): sha256(radis.__file__)},
        state_path=str(args.state.resolve()), state_sha256=sha256(args.state),
        temperature_K=temperature.tolist(), pressure_bar=pressure.tolist(),
        fractions_H_He_H2=fractions.tolist(), fractions_min=fractions.min(axis=0).tolist(), fractions_max=fractions.max(axis=0).tolist(),
        scope="Fixed Fe-column coefficient inventory; no new opacity, RCE, flux or P1 calculation.",
        pressure_convention="Native ExoJAX P in bar with .def cm^-1/bar labels, raw .broad values unchanged. The 2025 recipe paper labels atm; shared H2/He ratios cancel a common conversion, absolute-unit calibration remains outside this audit.",
        limitations=["No atomic-H coefficients were provided for these five molecules in the checked ExoMol listings.",
            "Positive collision widths do not give monotone monochromatic opacity or a flux/P1 error bound.",
            "High-temperature power laws and high-J continuation are model prescriptions, not laboratory accuracy bounds.",
            "Deterministic sample counts are not statistical estimates for the complete line lists."], species={})
    for molecule, (relative, bands) in DATABASES.items():
        directory = args.database_root / relative
        isotope, dataset = directory.parts[-2:]
        definition = directory / f"{isotope}__{dataset}.def"
        h2path = directory / f"{isotope}__H2.broad"
        hepath = directory / f"{isotope}__He.broad"
        if not hepath.exists():
            hepath = args.additional_he / f"{molecule}_He.broad"
        default = defaults(definition)
        h2, he = recipes(h2path), recipes(hepath)
        frames, provenance = [], []
        for band in bands:
            suffix = "" if band is None else f"__{band}"
            frame, record = sample_cache(directory / f"{isotope}__{dataset}{suffix}.trans.h5", args.rows_per_window)
            frames.append(frame); provenance.append(record)
        rows = pd.concat(frames, ignore_index=True)
        alpha, exponent, explicit = current_mapping(rows, h2, default)
        native_alpha, native_exponent, warning = native_mapping(rows, h2path, default, radis)
        if not np.array_equal(alpha, native_alpha) or not np.array_equal(exponent, native_exponent):
            raise ValueError(f"Audit mapping differs from production RADIS: {molecule}")
        unique, counts = np.unique(np.stack((alpha, exponent), axis=1), axis=0, return_counts=True)
        gammas = width(unique[:, 0], unique[:, 1], temperature, pressure)
        entry = dict(default_alpha_n=list(default), current_broadening_codes=sorted({key[0] for key in h2}),
            input_sha256={str(path): sha256(path) for path in (definition, h2path, hepath)},
            current_mapper_exact_match=True, mapper_warnings=warning,
            transition_sample=provenance, sampled_selected_rows=len(rows),
            fallback_rows=int((~explicit).sum()), fallback_fraction=float((~explicit).mean()),
            sampled_lower_J_range=[float(rows.jlower.min()), float(rows.jlower.max())],
            current_unique_alpha_n_count=np.column_stack((unique, counts)).tolist(),
            current_gamma_cm1_by_layer_and_unique_pair=gammas.tolist(),
            published_shared_recipes=recipe_comparison(h2, he, temperature, pressure, fractions))
        pub2, pube = documented_a0_tail(rows, h2, default), documented_a0_tail(rows, he, default)
        if pub2 is not None and pube is not None:
            current = width(alpha, exponent, temperature, pressure)
            public2, publice = width(*pub2, temperature, pressure), width(*pube, temperature, pressure)
            pure_ratio = public2 / current
            mixed_ratio = (fractions[:, 2, None] * public2 + fractions[:, 1, None] * publice) / current
            groups, counts = np.unique(np.column_stack((rows.jlower, rows.jupper, alpha, exponent, *pub2, *pube)), axis=0, return_counts=True)
            entry["documented_a0_tail_on_sample"] = dict(
                columns=["J_lower", "J_upper", "current_alpha", "current_n", "published_H2_alpha", "published_H2_n", "published_He_alpha", "published_He_n", "count"],
                groups=np.column_stack((groups, counts)).tolist(),
                pure_H2_over_current_min_by_layer=pure_ratio.min(axis=1).tolist(),
                pure_H2_over_current_max_by_layer=pure_ratio.max(axis=1).tolist(),
                known_H2_He_over_current_min_by_layer=mixed_ratio.min(axis=1).tolist(),
                known_H2_He_over_current_max_by_layer=mixed_ratio.max(axis=1).tolist())
        if molecule in ("TiO", "SiO"):
            if set(h2) != {("a0", 0)} or set(he) != {("a0", 0)}:
                raise ValueError("Review changed one-entry broadening recipe")
            public2 = width([h2[("a0", 0)][0]], [h2[("a0", 0)][1]], temperature, pressure)
            publice = width([he[("a0", 0)][0]], [he[("a0", 0)][1]], temperature, pressure)
            entry["single_entry_prescription"] = dict(
                meaning="Published J=0-only entry applies to every J; production legacy mapper instead defaults for J>0.",
                J0_coefficient_H2_He=[h2[("a0", 0)], he[("a0", 0)]],
                pure_H2_published_over_current_by_layer_and_pair=(public2 / gammas).tolist(),
                known_mixture_published_over_current_by_layer_and_pair=((fractions[:, 2, None] * public2 + fractions[:, 1, None] * publice) / gammas).tolist())
        result["species"][molecule] = entry
        print(f"{molecule}: {len(rows)} sampled transitions, fallback {entry['fallback_fraction']:.8f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-root", type=Path, default=Path("/home/kawahara/.database"))
    parser.add_argument("--state", type=Path, default=Path("results/rce_wasp18b_fe_broadening_audit.json"))
    parser.add_argument("--additional-he", type=Path, required=True)
    parser.add_argument("--rows-per-window", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows_per_window < 1:
        parser.error("rows-per-window must be positive")
    if args.output.exists():
        parser.error("Use a new output path")
    args.output.write_text(json.dumps(run(args), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

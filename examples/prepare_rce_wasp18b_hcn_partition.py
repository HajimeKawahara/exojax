"""Prepare an isolated finite-state HCN/HNC partition asset without downloads.

All supplied Harris states, including unknown isomer assignments, contribute to
the combined LTE sum. Evaluation through 4500 K does not extend the published
4000 K line-list completeness claim. No transition file or opacity is generated.
"""

import argparse
import bz2
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from exojax.utils.constants import hcperk


STEM = "1H-12C-14N__Harris"
SCHEMA = "exojax.rce.hcn-hnc-finite-state/1"
SOURCE_SHA256 = {
    "def": "d9ffced767368cfb92609141e0cd1cd65ea6598ce47d36128dd54c72199e31a6",
    "pf": "028f4b42b3ef9bf9527b172b9b89a7dd47259d8b1bf556de0fd201684968b4f4",
    "states.bz2": "631f775e2d8096461333837c5bb4881a11533d851c774df3da1d1698c556516a",
}
ASSUMPTIONS = [
    "One combined internal-LTE state ensemble, multiplied by n(HCN)+n(HNC) exactly once.",
    "Individual FastChem isomer populations are not separately imposed on line intensities.",
    "Unknown isomer labels are retained, not discarded or assigned by guesswork.",
    "The published line-list completeness temperature remains 4000 K; finite-state evaluation to 4500 K is conditional.",
    "No bound on missing high-J/high-energy lines, electronic transitions, continua, line shapes or re-equilibrated RCE error.",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_states(path):
    """Read the pinned Harris layout; statistical weights already include spin."""
    with bz2.open(path, "rt") as handle:
        rows = np.loadtxt(handle, usecols=(0, 1, 2, 3, 7), dtype=str, ndmin=2)
    ids, energy, weight, angular = (rows[:, i].astype(float) for i in range(4))
    labels = rows[:, 4]
    if (not np.array_equal(ids, np.arange(1, len(ids)+1))
            or not np.all(np.isfinite([energy, weight, angular]))
            or np.any(energy < 0) or np.any(weight <= 0) or np.any(angular < 0)
            or not set(labels).issubset({"0", "1", "*"})
            or not {"0", "1"}.issubset(labels)):
        raise ValueError("Invalid Harris states or isomer labels")
    return energy, weight, labels


def partition_parts(temperature, energy, weight, labels):
    """Direct positive Boltzmann sums, all with the common HCN energy zero."""
    temperature = np.asarray(temperature, dtype=float)
    if (temperature.ndim != 1 or not np.all(np.isfinite(temperature))
            or np.any(temperature <= 0)):
        raise ValueError("Partition temperatures must be finite positive and one-dimensional")
    masks = [labels == label for label in ("0", "1", "*")]
    output = []
    for value in temperature:
        terms = weight * np.exp(-hcperk * energy / value)
        output.append([terms.sum(), *(terms[mask].sum() for mask in masks)])
    return np.asarray(output)


def verify_asset(directory):
    """Reject changed original/derived bytes or a different generating script."""
    directory = Path(directory)
    record = json.loads((directory / "direct_partition.json").read_text())
    if (record["schema"] != SCHEMA or record["generator_sha256"] != sha256(__file__)
            or record["c2_cm_K"] != float(hcperk)):
        raise ValueError("Changed finite-state partition generator or convention")
    originals = {"def": f"{STEM}.def", "pf": f"{STEM}.pf.original", "states.bz2": f"{STEM}.states.bz2"}
    if (set(record["files"]) != {*originals.values(), f"{STEM}.pf"}
            or any(record["files"][name] != record["original_sha256"][ext]
                   for ext, name in originals.items())):
        raise ValueError("Incomplete or inconsistent finite-state partition provenance")
    for name, expected in record["files"].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise ValueError("Changed finite-state partition asset")
    return record


def prepare_asset(source, output_root, *, expected_sha256=SOURCE_SHA256):
    """Copy small originals and derive a new PF; preserve/reuse identical output."""
    source, output_root = Path(source), Path(output_root)
    files = {ext: source / f"{STEM}.{ext}" for ext in SOURCE_SHA256}
    if {ext: sha256(path) for ext, path in files.items()} != expected_sha256:
        raise ValueError("Official Harris metadata/state hashes differ from the pinned sources")
    directory = output_root / "HCN/1H-12C-14N/Harris"
    if (directory / "direct_partition.json").exists():
        record = verify_asset(directory)
        if record["original_sha256"] != expected_sha256:
            raise ValueError("Existing finite-state asset has different original sources")
        return directory, record
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Incomplete nonempty partition output; use a new output root")
    energy, weight, labels = read_states(files["states.bz2"])
    original = np.loadtxt(files["pf"], ndmin=2)
    if (original.shape[1] != 2 or not np.all(np.isfinite(original))
            or np.any(original <= 0) or np.any(np.diff(original[:, 0]) <= 0)
            or original[0, 0] > 296. or original[-1, 0] != 4000.):
        raise ValueError("Require a positive ordered source PF covering 296 through 4000 K")
    temperatures = np.arange(1., 4501.)
    values = partition_parts(temperatures, energy, weight, labels)[:, 0]
    checks = np.unique(np.r_[296., original[original[:, 0] >= 1500., 0]])
    direct = partition_parts(checks, energy, weight, labels)[:, 0]
    difference = direct / np.interp(checks, *original.T) - 1.
    if np.max(abs(difference)) > 1e-5:
        raise ValueError("Direct state sum does not reproduce the supplied PF within 1e-5")
    directory.mkdir(parents=True, exist_ok=True)
    for ext, source_path in files.items():
        suffix = ".original" if ext == "pf" else ""
        shutil.copyfile(source_path, directory / (source_path.name + suffix))
    np.savetxt(directory / f"{STEM}.pf", np.column_stack((temperatures, values)), fmt="%.17g")
    record = dict(schema=SCHEMA, generator_sha256=sha256(__file__), c2_cm_K=float(hcperk),
        original_sha256=expected_sha256, state_count=len(energy), label_counts=dict(Counter(labels)),
        temperature_range_K=[1., 4500.], temperature_step_K=1.,
        declared_line_list_maximum_temperature_K=4000.,
        original_pf_temperature_range_K=original[[0, -1], 0].tolist(),
        reference_reproduction=dict(checked_temperatures=len(checks),
            max_abs_relative_difference=float(np.max(abs(difference))), tolerance=1e-5),
        assumptions=ASSUMPTIONS,
        source_urls={ext: f"https://exomol.com/db/HCN/1H-12C-14N/Harris/{STEM}.{ext}" for ext in files},
        references=["https://doi.org/10.1093/mnras/stt2011",
                    "https://doi.org/10.1111/j.1365-2966.2005.09960.x"],
        files={path.name: sha256(path) for path in sorted(directory.iterdir())})
    (directory / "direct_partition.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    return directory, record


def normalization_diagnostics(directory, temperature, hcn_fraction, hnc_fraction):
    """Compare partial assigned-state weights with the supplied chemical ratio."""
    record = verify_asset(directory)
    temperature = np.asarray(temperature, dtype=float)
    if np.any((temperature < record["temperature_range_K"][0])
              | (temperature > record["temperature_range_K"][1])):
        raise ValueError("Temperature outside finite-state partition asset")
    hcn, hnc = np.asarray(hcn_fraction), np.asarray(hnc_fraction)
    if (hcn.shape != temperature.shape or hnc.shape != temperature.shape
            or not np.all(np.isfinite([hcn, hnc])) or np.any(hcn < 0) or np.any(hnc < 0)):
        raise ValueError("Require aligned finite nonnegative isomer number fractions")
    parts = partition_parts(temperature, *read_states(Path(directory) / f"{STEM}.states.bz2"))
    ratio = parts[:, 2] / parts[:, 1]
    derived = np.loadtxt(Path(directory) / f"{STEM}.pf")
    return dict(partition_asset=record, temperature_K=temperature.tolist(),
        combined_number_fraction=(hcn+hnc).tolist(),
        hcn_number_fraction=hcn.tolist(), hnc_number_fraction=hnc.tolist(),
        direct_partition_sum=parts[:, 0].tolist(),
        unassigned_partition_fraction=(parts[:, 3]/parts[:, 0]).tolist(),
        assigned_state_hnc_over_hcn=ratio.tolist(),
        chemical_hnc_over_hcn=[float(b/a) if a > 0 else None for a, b in zip(hcn, hnc)],
        chemical_over_assigned_state_ratio=[float(b/a/r) if a > 0 else None for a, b, r in zip(hcn, hnc, ratio)],
        pf_interpolation_relative_difference=(np.interp(temperature, *derived.T)/parts[:, 0]-1.).tolist(),
        scope="Assigned-state ratio omits unknown assignments and is not a complete physical isomer ratio or an error bound.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    directory, record = prepare_asset(args.source_directory, args.output_root)
    print(json.dumps(dict(directory=str(directory.resolve()), partition=record), indent=2))


if __name__ == "__main__":
    main()

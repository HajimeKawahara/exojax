"""Standalone consumer example for PreMODIT LBD HDF5 schema 1.0.

Dependencies: numpy and h5py. No ExoJAX, JAX, RADIS or PyExoCross import.
This reads prepared LBD files, not ExoMolOP cross-section tables. The producer
must validate the artifact; this example checks its basic numerical contract.
All arrays are loaded into host memory once. The returned LSD still requires
Voigt convolution and the R / nu_grid factor to become a cross section.

Usage:
    python read_premodit_lbd.py sample.premodit.h5

From Python:
    arrays, metadata = load_lbd("sample.premodit.h5")
    lsd, ngamma, nsigma = evaluation_inputs(arrays, metadata, 1000.0, 1.0)

The temperature is in K; pressure_bar is the broadener pressure in bar.
Generate a file with export_premodit_lbd.py or exojax.opacity.save_lbd.
"""

import argparse
import json
import re

import h5py
import numpy as np


def load_lbd(path):
    """Return independent NumPy arrays and metadata, closing the HDF5 file."""
    with h5py.File(path, "r") as handle:
        if (handle.attrs.get("format") != "exojax-premodit-lbd"
                or not re.fullmatch(r"1\.\d+", str(handle.attrs.get("schema_version")))):
            raise ValueError("Expected PreMODIT LBD schema major version 1.")
        group = handle["premodit"]
        partition = handle["partition_function"]
        coeff = group["lbd_coeff"]
        if coeff.attrs["encoding"] != "log0_linear_derivatives":
            raise ValueError("Unsupported LBD coefficient encoding.")
        if coeff.attrs["axes"] != "derivative_order,wavenumber,broadening_pair,elower":
            raise ValueError("Unsupported LBD axis order.")
        units = {
            "nu_grid": "cm^-1", "elower_grid": "cm^-1",
            "ngamma_ref_grid": "1", "n_Texp_grid": "1",
        }
        for name, unit in units.items():
            if group[name].attrs["units"] != unit:
                raise ValueError(f"Unsupported units for {name}.")
        if (group.attrs["temperature_unit"], group.attrs["pressure_unit"],
                group.attrs["pressure_reference"]) != ("K", "bar", 1.0):
            raise ValueError("Expected temperatures in K and reference pressure 1 bar.")
        if (partition["temperature"].attrs["units"], partition["Q"].attrs["units"],
                partition.attrs["interpolation"]) != ("K", "1", "linear_Q_in_T"):
            raise ValueError("Expected linear Q(T) interpolation with temperature in K.")
        if handle["mol_mass"].attrs["units"] != "u":
            raise ValueError("Expected isotopologue mass in atomic mass units.")
        if group["multi_index_uniqgrid"].attrs["index_base"] != 0:
            raise ValueError("Expected zero-based broadening indices.")
        arrays = {name: group[name][()] for name in (
            "lbd_coeff", "nu_grid", "elower_grid", "ngamma_ref_grid",
            "n_Texp_grid", "multi_index_uniqgrid",
        )}
        arrays["pf_temperature"] = partition["temperature"][()]
        arrays["pf_Q"] = partition["Q"][()]
        metadata = {
            name: np.asarray(handle[name].asstr()[()]).item()
            for name in ("mol_name", "key_iso_ll", "DOI", "Date_ID")
        }
        metadata.update(
            mol_mass=float(handle["mol_mass"][()]),
            premodit=dict(group.attrs),
            partition_function=dict(partition.attrs),
            provenance=json.loads(handle["provenance_json"].asstr()[()]),
        )

    mode = metadata["premodit"]["diffmode"]
    pairs = arrays["multi_index_uniqgrid"]
    expected = (mode + 1, arrays["nu_grid"].size, len(pairs), arrays["elower_grid"].size)
    if mode not in (0, 1, 2) or arrays["lbd_coeff"].shape != expected:
        raise ValueError("LBD shape does not match diffmode or coordinate arrays.")
    if pairs.shape != (expected[2], 2) or pairs.dtype.kind not in "iu":
        raise ValueError("Expected an integer (Nbroad, 2) broadening index array.")
    limits = [arrays["ngamma_ref_grid"].size, arrays["n_Texp_grid"].size]
    if np.any(pairs < 0) or np.any(pairs >= limits):
        raise ValueError("Broadening indices are outside their coordinate grids.")
    return arrays, metadata


def evaluation_inputs(arrays, metadata, temperature, pressure_bar):
    """Return LSD (cm/molecule), normalized Lorentz HWHM and Doppler sigma.

    Shapes are (Nnu, Nbroad), (Nbroad,), and scalar, respectively.
    No Voigt convolution, wavelength resampling or abundance scaling is done.
    Use the artifact's reference kernel and valid inner wavenumber range when
    comparing cross sections. Input pressure is in bar, not Pa.
    """
    p = metadata["premodit"]
    if not np.isfinite(temperature) or not p["Tmin"] <= temperature <= p["Tmax"]:
        raise ValueError("Temperature is outside the LBD design range.")
    if temperature <= 0 or not np.isfinite(pressure_bar) or pressure_bar < 0:
        raise ValueError("Expected positive temperature and nonnegative pressure.")
    tq, q = arrays["pf_temperature"], arrays["pf_Q"]
    if (tq.ndim != 1 or tq.size < 2 or q.shape != tq.shape
            or not np.all(np.isfinite(tq)) or not np.all(np.diff(tq) > 0)
            or not np.all(np.isfinite(q)) or not np.all(q > 0)):
        raise ValueError("Invalid partition-function table.")
    if not (tq[0] <= temperature <= tq[-1] and tq[0] <= p["Tref"] <= tq[-1]):
        raise ValueError("Temperature or Tref is outside the partition-function table.")
    qr = np.interp(temperature, tq, q) / np.interp(p["Tref"], tq, q)
    c2 = p["c2"]  # hc/k_B in cm K, recorded by the producer.
    bias = -c2 * arrays["elower_grid"] * (1.0 / temperature - 1.0 / p["Tref"])
    coeff = arrays["lbd_coeff"]
    dt = 1.0 / temperature - 1.0 / p["Twt"]
    # Keep C0 in log space until combining it with the Boltzmann factor.
    lsd = np.exp(coeff[0] + bias).sum(axis=-1)
    if p["diffmode"] >= 1:
        lsd += (np.exp(bias) * coeff[1] * dt).sum(axis=-1)
    if p["diffmode"] == 2:
        lsd += (np.exp(bias) * coeff[2] * (0.5 * dt**2)).sum(axis=-1)
    nu = arrays["nu_grid"]
    stimulated = np.expm1(-c2 * nu / temperature) / np.expm1(-c2 * nu / p["Tref"])
    lsd *= (stimulated / qr)[:, None]

    indices = arrays["multi_index_uniqgrid"]
    gamma = arrays["ngamma_ref_grid"][indices[:, 0]]
    exponent = arrays["n_Texp_grid"][indices[:, 1]]
    ngamma = gamma * (temperature / p["Tref_broadening"]) ** (-exponent) * pressure_bar
    nsigma = p["doppler_constant"] * np.sqrt(temperature / metadata["mol_mass"]) * p["R"]
    return lsd, ngamma, nsigma


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Prepared LBD file using schema 1.0")
    args = parser.parse_args()
    arrays, metadata = load_lbd(args.path)
    print(metadata["key_iso_ll"], metadata["Date_ID"])
    print("LBD shape:", arrays["lbd_coeff"].shape)
    print("Wavenumber range (cm^-1):", arrays["nu_grid"][[0, -1]])

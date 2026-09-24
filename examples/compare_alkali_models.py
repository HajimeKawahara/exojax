"""Compare table wings with matched sub-Voigt and Voigt Na/K resonance doublets.

Use local CDS data; no atomic line-list download is required. See
documents/tutorials/alkali_models.rst for the data and approximation details.
"""

from argparse import ArgumentParser
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from exojax.database.core.broadening import doppler_sigma
from exojax.opacity import OpaAlkaliTable
from exojax.opacity.alkali import subvoigt
from exojax.opacity.lpf.lpf import voigt
from exojax.utils.constants import bar_cgs, kB


def compare(data_path, species="Na", temperature=1000.0, density=1.0e19):
    """Return a figure at an exact tabulated temperature and perturber density."""
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("density must be finite and positive.")
    model, broadener, detuning, cutoff = {
        "Na": ("allard2019_na_h2", "H2", 30.0, 5000.0),
        "K": ("allard2024_k_he", "He", 20.0, 1600.0),
    }[species]
    pressure = density * kB * temperature / bar_cgs
    core_transition = (20.0, 30.0)
    # A small initial grid lets us inspect the source line metadata.
    opa = OpaAlkaliTable(
        np.array([17000.0 if species == "Na" else 13000.0]), data_path, model=model,
        vmr_perturber=1.0, core_transition=core_transition,
    )
    if density > opa.density_max:
        raise ValueError(f"density must not exceed {opa.density_max:g} cm-3.")
    profiles = []
    for component in ("D1", "D2"):
        matches = [p for p in opa.profiles[component] if p.temperature == temperature]
        if not matches:
            available = [p.temperature for p in opa.profiles[component]]
            raise ValueError(f"Choose a tabulated temperature for {component}: {available}.")
        profiles.append(matches[0])
    centers = np.array([1.0e8 / p.wavelength for p in profiles])
    # Retain both wings; each table component is zero outside its own support.
    lower, upper = centers.min() - 3000.0, centers.max() + 3000.0
    grid = np.unique(np.concatenate([
        np.linspace(lower, upper, 4001),
        *(center + np.linspace(-100.0, 100.0, 10001) for center in centers),
    ]))
    grid = grid[(grid >= lower) & (grid <= upper)]
    opa = OpaAlkaliTable(
        grid, data_path, model=model, vmr_perturber=1.0,
        core_transition=core_transition,
    )
    table_xs = np.asarray(opa.xsvector(temperature, pressure))
    if not np.all(np.isfinite(table_xs)):
        raise ValueError("The requested state produced nonfinite table cross sections.")
    voigt_xs, subvoigt_xs = np.zeros_like(grid), np.zeros_like(grid)
    for center, profile, weight_ratio in zip(centers, profiles, (1.0, 0.5)):
        density_ratio = density / profile.density
        natural_width = 2.0 * profile.normalization * center**2 * weight_ratio
        gamma = profile.width * density_ratio + natural_width
        shift = profile.shift * density_ratio
        sigma = doppler_sigma(center, temperature, opa.mass)
        offset = grid - center - shift
        voigt_xs += profile.normalization * np.asarray(voigt(offset, sigma, gamma))
        subvoigt_xs += profile.normalization * np.asarray(subvoigt(
            offset, sigma, gamma, temperature, detuning, cutoff,
        ))

    figure, axes = plt.subplots(
        2, 2, figsize=(11, 6), sharex="col", height_ratios=[2, 1], layout="constrained",
    )
    ratio = np.divide(
        table_xs, subvoigt_xs, out=np.full_like(grid, np.nan), where=subvoigt_xs > 0.0,
    )
    for column, title in enumerate(("Resonance doublet and wings", "Core approximation and joins")):
        top, bottom = axes[:, column]
        for values, label, color in (
            (voigt_xs, "Voigt", "C0"),
            (subvoigt_xs, "Sub-Voigt", "C1"),
            (table_xs, "Allard table wings + Voigt core", "C2"),
        ):
            top.semilogy(grid, np.where(values > 0.0, values, np.nan), label=label, color=color)
        bottom.semilogy(grid, np.where(ratio > 0.0, ratio, np.nan), color="C2")
        bottom.axhline(1.0, color="0.5", linewidth=0.8)
        top.set_title(title)
        for axis in (top, bottom):
            axis.grid(alpha=0.2)
        bottom.set_xlabel(r"Vacuum wavenumber (cm$^{-1}$)")
    for center in centers:
        for sign in (-1.0, 1.0):
            edges = sorted(center + sign * np.asarray(core_transition))
            for axis in axes[:, 1]:
                axis.axvspan(*edges, color="0.5", alpha=0.15)
        axes[0, 1].axvline(center, color="0.5", linewidth=0.5, linestyle=":")
    axes[0, 0].legend(fontsize="small")
    axes[0, 0].set_ylabel(r"Cross section (cm$^2$ per ground-state atom)")
    axes[1, 0].set_ylabel("Table hybrid / sub-Voigt")
    core_min, core_max = centers.min() - 60.0, centers.max() + 60.0
    axes[1, 1].set_xlim(core_min, core_max)
    in_core = (grid >= core_min) & (grid <= core_max)
    local_xs = np.concatenate([values[in_core] for values in (table_xs, subvoigt_xs, voigt_xs)])
    axes[0, 1].set_ylim(local_xs[local_xs > 0.0].min() * 0.7, local_xs.max() * 2.0)
    local_ratio = ratio[in_core & np.isfinite(ratio) & (ratio > 0.0)]
    axes[1, 1].set_ylim(local_ratio.min() * 0.9, local_ratio.max() * 1.1)
    figure.suptitle(
        f"{species} I D1 + D2, {broadener} only: T = {temperature:g} K, "
        f"n = {density:.1e} cm$^{{-3}}$ (P = {pressure:.3g} bar)"
    )
    return figure


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path, help="Local CDS archive or data directory.")
    parser.add_argument("--species", choices=("Na", "K"), default="Na")
    parser.add_argument("--temperature", type=float, default=1000.0, help="Exact tabulated temperature in K.")
    parser.add_argument("--density", type=float, default=1.0e19, help="Perturber number density in cm-3.")
    parser.add_argument("--output", type=Path, default=Path("alkali_models.png"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    figure = compare(args.data_path, args.species, args.temperature, args.density)
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()

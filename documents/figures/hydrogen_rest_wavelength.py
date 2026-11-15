"""Plot Balmer wavelength corrections and a common-grid H-beta profile.

Run from the repository with PYTHONPATH=src JAX_PLATFORMS=cpu.
References: SDSS for H-alpha; NIST ASD unresolved vacuum Ritz wavelengths
for n=4..10. These are representative multiplets, not fine-structure components.
https://www.sdss4.org/dr12/spectro/spectro_basics/
https://physics.nist.gov/asd
"""

from __future__ import annotations

from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exojax.database import AdbHydrogen
from exojax.opacity import OpaDirect
from exojax.utils.constants import c


def main():
    """Compare zero-velocity LTE cross sections at 8,000 K and zero pressure."""
    jax.config.update("jax_enable_x64", True)
    wavelength_ref = 4862.691
    velocity = np.linspace(30.0, -30.0, 3001)
    wavelength = wavelength_ref * (1 + velocity / c)
    wavenumber = 1e8 / wavelength
    spectra, centers = {}, {}
    for mode in ("reference", "rydberg"):
        adb = AdbHydrogen(n_upper_min=4, n_upper_max=4, wavelength_model=mode)
        spectra[mode] = np.asarray(OpaDirect(adb, wavenumber).xsvector(8000.0, 0.0))
        centers[mode] = 1e8 / float(adb.nu_lines[0])
    peak = spectra["reference"].max()
    np.testing.assert_allclose(centers["reference"], wavelength_ref, rtol=1e-12)
    reference = 1e8 / AdbHydrogen(wavelength_model="reference").nu_lines
    rydberg = 1e8 / AdbHydrogen(wavelength_model="rydberg").nu_lines
    offsets = c * (rydberg / reference - 1)

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.7, 1])
    ax = fig.add_subplot(grid[0, 0])
    bias_ax = fig.add_subplot(grid[0, 1])
    text_ax = fig.add_subplot(grid[1, :])
    for mode, color, label in (
        ("reference", "C0", "Reference: 4862.691 A (default)"),
        ("rydberg", "C1", f"Rydberg: {centers['rydberg']:.6f} A"),
    ):
        ax.plot(velocity, spectra[mode] / peak, color=color, lw=2, label=label)
        ax.axvline(c * (centers[mode] / wavelength_ref - 1), color=color, ls=":")
    ax.set(xlim=(-30, 30), ylim=(0, 1.15), ylabel="Cross section / reference peak",
           xlabel="Velocity from 4862.691 A [km/s] (redward positive)",
           title="H-beta: same grid; zero physical velocity")
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.03, 0.08, "T = 8,000 K; P = 0\nThermal + natural broadening",
            transform=ax.transAxes, fontsize=10)
    names = ["H-alpha (n=3)", "H-beta (n=4)", "H-gamma (n=5)", "H-delta (n=6)",
             "H-epsilon (n=7)", "H8 (n=8)", "H9 (n=9)", "H10 (n=10)"]
    bias_ax.barh(names, offsets, color=["C2"] + ["C1"] * 7, height=0.65)
    for index, offset in enumerate(offsets):
        bias_ax.text(offset + 0.06, index, f"+{offset:.3f}", va="center", fontsize=10)
    bias_ax.invert_yaxis()
    bias_ax.set(xlim=(0, offsets.max() + 0.65),
                xlabel="Rydberg center redward of reference [km/s]",
                title="All eight supported Balmer lines")
    bias_ax.grid(axis="x", alpha=0.2)
    bias_ax.set_axisbelow(True)
    text_ax.axis("off")
    equations = [
        (0.98, r"Rydberg option: $\tilde{\nu}_{2n}^{\rm R}=R_{\rm H}(1/2^2-1/n^2)$,"
         r"  $R_{\rm H}=109677.5834\ {\rm cm}^{-1}$", 15),
        (0.72, r"Reference default: $\tilde{\nu}_{2n}^{\rm ref}=10^8/\lambda_{{\rm tab},n}\ {\rm cm}^{-1}$,"
         r"  $\lambda_{{\rm tab},n}$ in angstrom; $n=3,\ldots,10$", 15),
        (0.45, r"$\tilde{E}_u=\tilde{E}_l+\tilde{\nu},\qquad A_{ul}\propto f_{lu}\tilde{\nu}^{\,2},"
         r"\qquad \Delta v=c(\lambda_{\rm R}/\lambda_{\rm ref}-1)$", 17),
        (0.17, "Effective level gaps and radiative coefficients use the selected wavelength. "
         "Neither option resolves fine structure.", 11),
    ]
    for y, label, size in equations:
        text_ax.text(0, y, label, fontsize=size, va="top", transform=text_ax.transAxes)
    fig.suptitle("Balmer rest wavelengths: reference values versus Rydberg approximation")
    fig.supxlabel("H-alpha: SDSS representative vacuum center, 6564.614 A; "
                  "n=4..10: NIST ASD unresolved vacuum Ritz wavelengths\n"
                  "https://www.sdss4.org/dr12/spectro/spectro_basics/    "
                  "https://physics.nist.gov/asd", fontsize=9)
    destination = Path(__file__).with_suffix(".png")
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    print(f"Wrote {destination}")
    for name, offset in zip(names, offsets):
        print(f"{name}: Rydberg center {offset:+.6f} km/s from reference")


if __name__ == "__main__":
    main()

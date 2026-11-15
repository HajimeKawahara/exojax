"""Plot the H-alpha wavelength correction on a common physical grid.

Run from the repository with PYTHONPATH=src JAX_PLATFORMS=cpu.
Reference: https://www.sdss4.org/dr12/spectro/spectro_basics/
The reference is a representative unresolved vacuum wavelength, not an
individual fine-structure component. Higher Balmer lines retain Rydberg values.
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
    wavelength_ref = 6564.614
    velocity = np.linspace(30.0, -30.0, 3001)
    wavelength = wavelength_ref * (1 + velocity / c)
    wavenumber = 1e8 / wavelength
    spectra, centers = {}, {}
    for mode in ("reference", "rydberg"):
        adb = AdbHydrogen(n_upper_max=3, wavelength_model=mode)
        spectra[mode] = np.asarray(OpaDirect(adb, wavenumber).xsvector(8000.0, 0.0))
        centers[mode] = 1e8 / float(adb.nu_lines[0])
    peak = spectra["reference"].max()
    offset = c * (centers["rydberg"] / wavelength_ref - 1)

    fig, (ax, text_ax) = plt.subplots(
        1, 2, figsize=(12, 5.5), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.1, 1]},
    )
    for mode, color, label in (
        ("reference", "C0", "Reference: 6564.614 A (default)"),
        ("rydberg", "C1", f"Rydberg: {centers['rydberg']:.6f} A"),
    ):
        ax.plot(velocity, spectra[mode] / peak, color=color, lw=2, label=label)
        ax.axvline(c * (centers[mode] / wavelength_ref - 1), color=color, ls=":")
    ax.set(xlim=(-30, 30), ylim=(0, 1.15), ylabel="Cross section / reference peak",
           xlabel="Velocity from 6564.614 A [km/s] (redward positive)",
           title="Same wavelength grid; zero physical velocity")
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.03, 0.08, "T = 8,000 K; P = 0\nThermal + natural broadening",
            transform=ax.transAxes, fontsize=10)
    text_ax.axis("off")
    equations = [
        (0.96, "Original: wavelength_model='rydberg'", 12),
        (0.86, r"$\tilde{\nu}_{2n}=R_{\rm H}(1/2^2-1/n^2)$", 16),
        (0.75, r"$R_{\rm H}=109677.5834\ {\rm cm}^{-1}$", 14),
        (0.62, "Default: wavelength_model='reference'", 12),
        (0.52, r"$\tilde{\nu}_{23}=10^8/6564.614\ {\rm cm}^{-1}$", 16),
        (0.42, "Only H-alpha changes; higher Balmer lines retain Rydberg values.", 10),
        (0.29, r"$\tilde{E}_u=\tilde{E}_l+\tilde{\nu},\quad A_{ul}\propto f_{lu}\tilde{\nu}^{\,2}$", 16),
        (0.19, "Effective level gap and radiative coefficients stay consistent.", 10),
        (0.07, rf"$c(\lambda_{{\rm R}}/\lambda_{{\rm ref}}-1)=+{offset:.4f}\ {{\rm km\,s}}^{{-1}}$", 15),
    ]
    for y, label, size in equations:
        text_ax.text(0, y, label, fontsize=size, va="top", transform=text_ax.transAxes)
    fig.suptitle("H-alpha rest wavelength: representative vacuum center versus Rydberg approximation")
    fig.supxlabel("Wavelength reference: SDSS, Understanding the Optical Spectra, air/vacuum table\n"
                  "https://www.sdss4.org/dr12/spectro/spectro_basics/", fontsize=9)
    destination = Path(__file__).with_suffix(".png")
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    print(f"Wrote {destination}; Rydberg center is {offset:.6f} km/s redward of reference.")


if __name__ == "__main__":
    main()

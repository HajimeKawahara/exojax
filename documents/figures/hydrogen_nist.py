"""Compare NIST Balmer centers with the removed Rydberg approximation.

Run with an editable ExoJAX installation; AdbNist downloads/caches H I through
RADIS on first use. The left panel changes only the center, holding widths fixed.
"""

from __future__ import annotations

from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exojax.database import AdbNist
from exojax.opacity import OpaDirect
from exojax.utils.constants import c


def main():
    jax.config.update("jax_enable_x64", True)
    adb = AdbNist("H_I", nurange=(15000.0, 26400.0))
    upper_n = np.arange(3, 11)
    np.testing.assert_array_equal(adb.glower, np.full(8, 8.0))
    np.testing.assert_array_equal(adb.gupper, 2 * upper_n**2)
    reference = adb.nu_lines.copy()
    rydberg = 109677.5834 * (1 / 2**2 - 1 / upper_n**2)
    offset = c * (reference / rydberg - 1)
    adb.masking(np.arange(8) == 0)
    velocity = np.linspace(-35, 35, 2001)
    grid = reference[0] / (1 + velocity[::-1] / c)
    # Illustrative fixed HWHM: this is not a hydrogen damping prescription.
    opa = OpaDirect(adb, grid, atomic_broadening=lambda T, P: np.array([0.001]))
    nist_profile = np.asarray(opa.line_profiles(8000.0))[0]
    old_profile = np.asarray(opa.line_profiles(
        8000.0, velocity=c * (1 - rydberg[0] / reference[0])
    ))[0]

    fig = plt.figure(figsize=(11, 6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.1])
    left, right = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    left.plot(velocity[::-1], nist_profile / nist_profile.max(), label="NIST center")
    left.plot(velocity[::-1], old_profile / nist_profile.max(), "--",
              label="Rydberg center, same width")
    left.axvline(0, color="0.6", lw=0.8)
    left.set(xlabel="Velocity from NIST H-alpha [km/s]", ylabel="Profile / NIST peak",
             title="H-alpha: 8,000 K; common wavelength grid")
    left.legend(fontsize=9)
    bars = right.bar(upper_n, offset, color="#3679a8")
    right.bar_label(bars, fmt="%.2f", fontsize=9)
    right.set(xlabel="Upper principal quantum number (lower n=2)",
              ylabel="Rydberg redward offset [km/s]", ylim=(0, 3.5),
              xticks=upper_n, title="All eight formerly supported Balmer lines")
    equations = fig.add_subplot(gs[1, :])
    equations.axis("off")
    equations.text(0.02, 0.72,
        r"$\widetilde\nu_{\rm NIST}=\widetilde E_u-\widetilde E_l,\quad"
        r"\lambda_{\rm vac}[\AA]=10^8/\widetilde\nu_{\rm NIST}[{\rm cm}^{-1}]$",
        fontsize=16)
    equations.text(0.02, 0.24,
        r"$\widetilde\nu_{\rm R}=R_{\rm H}(1/4-1/n^2),\quad"
        r"R_{\rm H}=109677.5834\ {\rm cm}^{-1},\quad"
        r"\Delta v=c(\lambda_{\rm R}/\lambda_{\rm NIST}-1)$", fontsize=16)
    fig.suptitle("Use the existing AdbNist H I data instead of a second hydrogen database")
    fig.supxlabel("Source: NIST ASD via RADIS; unresolved n-shell entries; retrieved 2026-11-16",
                  fontsize=9)
    output = Path(__file__).with_suffix(".png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    for n, nu, dv in zip(upper_n, reference, offset):
        print(f"n={n}: vacuum={1e8/nu:.8f} A; Rydberg offset={dv:.5f} km/s")
    print(output)


if __name__ == "__main__":
    main()

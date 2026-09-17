"""Compare line-profile shapes with shared illustrative widths; no database needed."""

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from exojax.opacity.alkali import subvoigt
from exojax.opacity.lpf.lpf import voigt

jax.config.update("jax_enable_x64", True)
T, sigmaD, gammaL = 1200.0, 0.03, 0.1  # K, cm-1, cm-1 (Lorentz HWHM)
offset = np.geomspace(1.0e-3, 1.0e4, 2000)  # positive side of symmetric profiles
reference = np.asarray(voigt(offset, sigmaD, gammaL))
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey="row",
                         height_ratios=[2, 1], layout="constrained")
for column, (species, a, b) in enumerate((("Na I", 30.0, 5000.0), ("K I", 20.0, 1600.0))):
    profile = np.asarray(subvoigt(offset, sigmaD, gammaL, T, a, b))
    detuning = a * (T / 500.0)**0.6
    upper, lower = axes[:, column]
    upper.loglog(offset, reference, label="Voigt", color="C0")
    upper.loglog(offset, np.where(profile > 0, profile, np.nan),
                 label="Sub-Voigt", color="C1")
    lower.semilogx(offset, profile / reference, color="C1")
    lower.axhline(1.0, color="0.5", linewidth=0.8)
    for ax in (upper, lower):
        ax.axvline(detuning, color="0.4", linestyle="--", linewidth=0.9)
        ax.axvline(9000.0, color="0.4", linestyle=":", linewidth=0.9)
        ax.grid(alpha=0.2)
        ax.set_xlim(offset[0], offset[-1])
    upper.set(title=rf"{species}: $D={detuning:.1f}\ \mathrm{{cm}}^{{-1}}$", ylim=(1e-14, 10))
    upper.legend()
    lower.set(xlabel=r"Distance from line center $|\nu-\nu_0|$ (cm$^{-1}$)", ylim=(0, 4))
axes[0, 0].set_ylabel(r"Line profile $\phi$ (cm)")
axes[1, 0].set_ylabel("Sub-Voigt / Voigt")
axes[1, 0].text(0.04, 0.89, "Core ratio = 1/0.998", transform=axes[1, 0].transAxes)
fig.suptitle(r"Voigt and sub-Voigt profiles: shared illustrative widths"
             "\n" + rf"$T={T:g}$ K, $\sigma_D={sigmaD:g}$ cm$^{{-1}}$, $\gamma_L={gammaL:g}$ cm$^{{-1}}$")
fig.savefig(Path(__file__).resolve().parents[1] / "documents/userguide/alkali_profiles.png", dpi=160)
plt.close(fig)

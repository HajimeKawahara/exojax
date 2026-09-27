"""Forecast local constraints from CO emission near 2.3 microns.

Use the bundled ExoMol CO lines and a small HITRAN H2-H2 CIA table. Run
``python examples/spectral_sensitivity.py`` to regenerate the tutorial
figures; no downloads are needed.
"""

from argparse import ArgumentParser
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from exojax.database.cia.api import CdbCIA
from exojax.database.exomol.api import MdbExomol
from exojax.opacity import OpaCIA, OpaDirect
from exojax.rt import ArtEmisPure
from exojax.test.data import get_testdata_filename
from exojax.utils.information import linear_gaussian_diagnostics


PARAMETER_LABELS = (r"$T_0$", r"$\log_{10} q_{\rm CO}$", r"$\alpha$")
COLORS = ("#3969ac", "#e58606", "#008a78")


def make_observed_spectrum(samples_per_bin=16):
    """Return a differentiable emission model and its bin-center wavenumbers."""
    # BEGIN MODEL SETUP
    # Midpoint quadrature for 256 uniform, 0.125 cm-1 observation bins.
    size = 256 * samples_per_bin
    nu_grid = 4330.0 + (np.arange(size) + 0.5) * 32.0 / size
    art = ArtEmisPure(
        pressure_top=1.0e-4,
        pressure_btm=100.0,
        nlayer=24,
        nu_grid=nu_grid,
        nstream=4,
    )
    # Parse in a temporary copy so RADIS caches never modify installed data.
    relative = "CO/12C-16O/SAMPLE"
    with TemporaryDirectory(prefix="exojax-sensitivity-") as temporary:
        target = Path(temporary) / relative
        shutil.copytree(get_testdata_filename(relative), target)
        mdb = MdbExomol(
            str(target), [4329.0, 4363.0], crit=0.0,
            broadf_download=False, gpu_transfer=True, engine="pytables",
        )
    opa = OpaDirect(mdb, nu_grid)
    cia_path = Path(__file__).with_name("spectral_sensitivity_data") / "H2-H2_2011_4320-4370.cia"
    cia = OpaCIA(CdbCIA(str(cia_path), nu_grid), nu_grid)
    gravity = 10.0**4.4  # cm s-2
    vmr_h2, mean_molecular_weight = 0.855, 2.33
    # END MODEL SETUP

    # BEGIN OBSERVED SPECTRUM
    def observed_spectrum(theta):
        temperature_0, log10_q_co, alpha = theta
        temperature = temperature_0 * (art.pressure / 1.0) ** alpha
        dtau = art.opacity_profile_xs(
            opa.xsmatrix(temperature, art.pressure),
            10.0**log10_q_co, mdb.molmass, gravity,
        )
        dtau += art.opacity_profile_cia(
            cia.logacia_matrix(temperature), temperature,
            vmr_h2, vmr_h2, mean_molecular_weight, gravity,
        )
        flux = art.run(dtau, temperature) / 1.0e4
        return flux.reshape(256, samples_per_bin).mean(axis=1)
    # END OBSERVED SPECTRUM

    return observed_spectrum, nu_grid.reshape(256, samples_per_bin).mean(axis=1)


def plot_sensitivities(wavelength, flux, jacobian, noise_std, prior_std):
    """Plot the spectrum and the dimensionless, observation-space Jacobian."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    axes[0].plot(wavelength, flux, color="#333333", lw=1.5, label=r"Fine: $\Delta\tilde\nu=0.125$ cm$^{-1}$")
    coarse_wavelength = 1.0e4 / (1.0e4 / wavelength).reshape(-1, 8).mean(axis=1)
    axes[0].plot(coarse_wavelength, flux.reshape(-1, 8).mean(axis=1),
                 color=COLORS[1], lw=1.5, drawstyle="steps-mid",
                 label=r"Coarse: $\Delta\tilde\nu=1$ cm$^{-1}$")
    axes[0].set(ylabel=r"$F_{\tilde\nu}/10^4$", title=r"CO + H$_2$-H$_2$ CIA emission")
    axes[0].set_ylim(top=flux.max() * 1.2)
    axes[0].legend(fontsize=9, loc="upper right")
    scaled_jacobian = np.asarray(jacobian) * prior_std[None, :] / noise_std[:, None]
    for index, (label, color) in enumerate(zip(PARAMETER_LABELS, COLORS)):
        axes[1].plot(wavelength, scaled_jacobian[:, index], color=color, label=label)
    axes[1].set(xlabel=r"Wavelength [$\mu$m]", ylabel=r"$K_{ij}\,\sigma_{a,j}/\sigma_{e,i}$")
    axes[1].axhline(0.0, color="0.6", lw=0.6)
    axes[1].set_ylim(top=scaled_jacobian.max() * 1.25)
    axes[1].legend(ncol=3, loc="upper left")
    for axis in axes:
        axis.set_xlim(wavelength.min(), wavelength.max())
        axis.grid(alpha=0.2)
    return fig


def plot_constraints(results, prior_std):
    """Show marginal errors and the least constrained coarse-bin combination."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    positions = np.arange(3)
    for index, (name, result) in enumerate(results.items()):
        axes[0].bar(
            positions + (index - 0.5) * 0.35,
            np.asarray(result["posterior_std"]) / prior_std,
            width=0.35,
            color=COLORS[index],
            label=name,
        )
    axes[0].axhline(1.0, color="0.4", ls="--", label="Prior")
    axes[0].set(xticks=positions, xticklabels=PARAMETER_LABELS, ylim=(0.005, 1.3), yscale="log",
                ylabel=r"$\sigma_{\rm post}/\sigma_a$", title="Marginal uncertainty")
    axes[0].legend(fontsize=8)

    coarse = results["Coarse bins"]
    mode = np.asarray(coarse["parameter_modes"])[-1].copy()
    # Singular vectors have an arbitrary sign; choose a consistent display.
    if mode[1] < 0.0:
        mode *= -1.0
    axes[1].bar(positions, mode, color=COLORS)
    axes[1].axhline(0.0, color="0.4", lw=0.7)
    axes[1].set(xticks=positions, xticklabels=PARAMETER_LABELS, ylim=(-1, 1),
                ylabel="Coefficient in prior-scaled coordinates",
                title="Least constrained mode: coarse bins")

    covariance = np.asarray(coarse["posterior_cov"])
    std = np.asarray(coarse["posterior_std"])
    correlation = covariance / np.outer(std, std)
    mesh = axes[2].imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    for row in range(3):
        for column in range(3):
            value = correlation[row, column]
            axes[2].text(column, row, f"{value:.2f}", ha="center", va="center",
                         color="white" if abs(value) > 0.6 else "black")
    axes[2].set(xticks=positions, xticklabels=PARAMETER_LABELS,
                yticks=positions, yticklabels=PARAMETER_LABELS,
                title="Posterior correlation: coarse bins")
    fig.colorbar(mesh, ax=axes[2], shrink=0.75)
    return fig


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "documents/tutorials/spectral_sensitivity_files",
        help="Directory for the two tutorial PNG figures.",
    )
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    started = perf_counter()
    observed_spectrum, nu_bins = make_observed_spectrum()

    # BEGIN DIAGNOSTICS
    theta0 = jnp.array([1200.0, -2.3, 0.1])
    prior_std = jnp.array([100.0, 0.5, 0.03])  # K, dex, dimensionless
    noise_std = jnp.full(nu_bins.size, 0.03)  # fixed error on the scaled flux
    jacobian = jax.jit(jax.jacfwd(observed_spectrum))(theta0)
    fine = linear_gaussian_diagnostics(jacobian, noise_std, prior_std)

    # Average the same measurements, propagating their independent errors.
    coarse_jacobian = jacobian.reshape(-1, 8, 3).mean(axis=1)
    coarse_noise = jnp.sqrt((noise_std.reshape(-1, 8)**2).sum(axis=1)) / 8
    coarse = linear_gaussian_diagnostics(
        coarse_jacobian, coarse_noise, prior_std
    )
    # END DIAGNOSTICS

    results = {"Fine bins": fine, "Coarse bins": coarse}
    for name, result in results.items():
        print(f"{name}: posterior std [K, dex, dimensionless] = {np.asarray(result['posterior_std'])}")
        print(f"  information = {float(result['information_bits']):.3f} bits, "
              f"degrees of freedom = {float(result['degrees_of_freedom']):.3f}")
        print(f"  singular values = {np.asarray(result['singular_values'])}")
    print(f"Setup and diagnostics: {perf_counter() - started:.2f} s on {jax.default_backend()} (including JIT)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "spectrum_and_jacobian.png": plot_sensitivities(
            1.0e4 / nu_bins, np.asarray(jax.jit(observed_spectrum)(theta0)),
            jacobian, np.asarray(noise_std), np.asarray(prior_std),
        ),
        "binning_and_degeneracy.png": plot_constraints(results, np.asarray(prior_std)),
    }
    for filename, figure in figures.items():
        path = args.output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

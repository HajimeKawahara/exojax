"""Forecast local spectral constraints with an offline emission model.

The two absorbers have illustrative Gaussian cross sections, not molecular
line lists. Run ``python examples/spectral_sensitivity.py`` to regenerate
the figures in the spectral-sensitivity tutorial; no downloads are needed.
"""

from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from exojax.rt import ArtEmisPure
from exojax.utils.information import linear_gaussian_diagnostics


PARAMETER_LABELS = (r"$T_0$", r"$\log_{10} q_A$", r"$\log_{10} q_B$")
COLORS = ("#3969ac", "#e58606", "#008a78")


def make_observed_spectrum():
    """Return a differentiable emission model and its bin-center wavenumbers."""
    # Midpoints of 960 uniform wavenumber cells, grouped into 120 top-hat bins.
    nu_grid = jnp.asarray(4000.0 + (np.arange(960) + 0.5) * 1000.0 / 960)
    art = ArtEmisPure(
        pressure_top=1.0e-5,
        pressure_btm=10.0,
        nlayer=32,
        nu_grid=nu_grid,
    )

    def gaussian(center, width):
        return jnp.exp(-0.5 * ((nu_grid - center) / width) ** 2)

    # An identical broad band creates an abundance degeneracy. Separate narrow
    # features at larger wavenumbers can break it when they are observed.
    shared_band = gaussian(4400.0, 90.0)
    cross_section_a = 1.0e-20 * (shared_band + gaussian(4820.0, 15.0))
    cross_section_b = 1.0e-20 * (shared_band + gaussian(4920.0, 15.0))
    gravity = 1.0e3  # cm s-2
    molecular_mass = 28.0  # atomic mass units, the same for both toy absorbers
    continuum_dtau = 0.01 * jnp.asarray(art.dParr)[:, None] * 1.0e6 / gravity

    # BEGIN OBSERVED SPECTRUM
    def observed_spectrum(theta):
        temperature_0, log10_q_a, log10_q_b = theta
        temperature = temperature_0 * (art.pressure / 1.0) ** 0.1
        dtau = continuum_dtau + art.opacity_profile_xs(
            cross_section_a, 10.0**log10_q_a, molecular_mass, gravity
        ) + art.opacity_profile_xs(
            cross_section_b, 10.0**log10_q_b, molecular_mass, gravity
        )
        flux = art.run(dtau, temperature) / 1.0e4
        return flux.reshape(-1, 8).mean(axis=1)
    # END OBSERVED SPECTRUM

    return observed_spectrum, np.asarray(nu_grid.reshape(-1, 8).mean(axis=1))


def plot_sensitivities(wavelength, flux, jacobian, noise_std, prior_std):
    """Plot the spectrum and the dimensionless, observation-space Jacobian."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    axes[0].plot(wavelength, flux, color="#333333", lw=1.8)
    axes[0].set(ylabel=r"$F_{\tilde\nu}/10^4$", title="Toy emission spectrum, binned in wavenumber")
    scaled_jacobian = np.asarray(jacobian) * prior_std[None, :] / noise_std[:, None]
    for index, (label, color) in enumerate(zip(PARAMETER_LABELS, COLORS)):
        axes[1].plot(wavelength, scaled_jacobian[:, index], color=color, label=label)
    axes[1].set(xlabel=r"Wavelength [$\mu$m]", ylabel=r"$K_{ij}\,\sigma_{a,j}/\sigma_{e,i}$")
    axes[1].axhline(0.0, color="0.6", lw=0.6)
    axes[1].legend(ncol=3, loc="upper left")
    for axis in axes:
        axis.axvspan(2.0, 1.0e4 / 4700.0, color="#e58606", alpha=0.12)
        axis.set_xlim(2.0, 2.5)
        axis.grid(alpha=0.2)
    axes[0].text(2.018, 0.96, "Distinct features", transform=axes[0].get_xaxis_transform(), va="top")
    axes[0].text(2.18, 0.96, "Shared band / retained coverage", transform=axes[0].get_xaxis_transform(), va="top")
    return fig


def plot_constraints(results, prior_std):
    """Show marginal errors, a weak mode, and the retained-band correlation."""
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
    axes[0].set(xticks=positions, xticklabels=PARAMETER_LABELS, ylim=(0, 1.08),
                ylabel=r"$\sigma_{\rm post}/\sigma_a$", title="Marginal uncertainty")
    axes[0].legend(fontsize=8)

    restricted = results["Shared band only"]
    mode = np.asarray(restricted["parameter_modes"])[-1].copy()
    # Singular vectors have an arbitrary sign; choose a consistent display.
    if mode[1] < 0.0:
        mode *= -1.0
    axes[1].bar(positions, mode, color=COLORS)
    axes[1].axhline(0.0, color="0.4", lw=0.7)
    axes[1].set(xticks=positions, xticklabels=PARAMETER_LABELS, ylim=(-1, 1),
                ylabel="Coefficient in prior-scaled coordinates",
                title="Weakest mode: shared band only")

    covariance = np.asarray(restricted["posterior_cov"])
    std = np.asarray(restricted["posterior_std"])
    correlation = covariance / np.outer(std, std)
    mesh = axes[2].imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    for row in range(3):
        for column in range(3):
            value = correlation[row, column]
            axes[2].text(column, row, f"{value:.2f}", ha="center", va="center",
                         color="white" if abs(value) > 0.6 else "black")
    axes[2].set(xticks=positions, xticklabels=PARAMETER_LABELS,
                yticks=positions, yticklabels=PARAMETER_LABELS,
                title="Posterior correlation: shared band only")
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
    observed_spectrum, nu_bins = make_observed_spectrum()

    # BEGIN DIAGNOSTICS
    theta0 = jnp.array([1200.0, -3.0, -3.0])
    prior_std = jnp.array([100.0, 0.5, 0.5])  # K, dex, dex
    noise_std = jnp.full(nu_bins.size, 0.003)  # fixed error on the scaled flux
    jacobian = jax.jacfwd(observed_spectrum)(theta0)
    full = linear_gaussian_diagnostics(jacobian, noise_std, prior_std)

    # Select the same measured channels and retain their original uncertainties.
    keep = nu_bins < 4700.0
    restricted = linear_gaussian_diagnostics(
        jacobian[keep], noise_std[keep], prior_std
    )
    # END DIAGNOSTICS

    results = {"Full coverage": full, "Shared band only": restricted}
    for name, result in results.items():
        print(f"{name}: posterior std [K, dex, dex] = {np.asarray(result['posterior_std'])}")
        print(f"  information = {float(result['information_bits']):.3f} bits, "
              f"degrees of freedom = {float(result['degrees_of_freedom']):.3f}")
        print(f"  singular values = {np.asarray(result['singular_values'])}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "spectrum_and_jacobian.png": plot_sensitivities(
            1.0e4 / nu_bins, np.asarray(observed_spectrum(theta0)),
            jacobian, np.asarray(noise_std), np.asarray(prior_std),
        ),
        "coverage_and_degeneracy.png": plot_constraints(results, np.asarray(prior_std)),
    }
    for filename, figure in figures.items():
        path = args.output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

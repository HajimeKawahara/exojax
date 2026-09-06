"""Compute small teaching spectra from the distributed opacity tables.

Run from a checkout with ``python examples/starter_opacity.py`` after the
data release is published. To use locally prepared data, add
``--data-root documents/_build/opacity-data``. No line list is loaded.
"""

from argparse import ArgumentParser
from pathlib import Path

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from exojax.opacity import OpaCKD, OpaDiffgrid
from exojax.provider.starter import fetch_starter_opacity
from exojax.rt import ArtEmisPure, ArtTransPure


def h2o_transmission(opa, temperature):
    """Return transit depth in ppm for an isothermal, H2O-only atmosphere.

    ``temperature`` is in K. The H2/He background sets the mean molecular
    weight; this teaching model includes only water line absorption.
    """
    art = ArtTransPure(
        pressure_top=1.0e-5,
        pressure_btm=10.0,
        nlayer=30,
        integration="simpson",
        warn_no_nu_grid=False,
    )
    temperatures = art.constant_profile(temperature)
    mass_mixing_ratio = art.constant_mmr_profile(1.0e-3)
    mean_molecular_weight = art.constant_profile(2.33)
    radius_btm = 6.9e9  # cm
    stellar_radius = 6.957e10  # cm
    gravity = 2478.57  # cm s-2

    cross_section = opa.xstensor_ckd(temperatures, art.pressure)
    optical_depth = art.opacity_profile_xs_ckd(
        cross_section, mass_mixing_ratio, opa.molmass, gravity
    )
    normalized_squared_radius = art.run_ckd(
        optical_depth,
        temperatures,
        mean_molecular_weight,
        radius_btm,
        gravity,
        opa.ckd_info.weights,
    )
    return 1.0e6 * normalized_squared_radius * (radius_btm / stellar_radius) ** 2


def co_atmosphere(opa):
    """Reconstruct the fixed pressure grid used for the CO starter table."""
    return ArtEmisPure(
        nu_grid=opa.nu_grid,
        pressure_top=0.1,
        pressure_btm=10.0,
        nlayer=16,
        rtsolver="fbased2st",
        nstream=2,
    )


def co_emission(opa, temperature):
    """Return CO-only emission flux per wavenumber.

    ``temperature`` is the temperature at 1 bar in K. All layer
    temperatures must remain in the table's 500--1500 K range.
    """
    art = co_atmosphere(opa)
    temperatures = temperature * (art.pressure / 1.0) ** 0.08
    mass_mixing_ratio = art.constant_mmr_profile(1.0e-3)
    gravity = 1.0e5  # cm s-2
    cross_section = opa.xsmatrix(temperatures)
    optical_depth = art.opacity_profile_xs(
        cross_section, mass_mixing_ratio, opa.molmass, gravity
    )
    return art.run(optical_depth, temperatures)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Local directory containing h2o-ckd-v1 and co-diffgrid-v1.",
    )
    parser.add_argument(
        "--output", type=Path, help="Save the figure instead of displaying it."
    )
    args = parser.parse_args()

    def dataset_path(dataset_id):
        if args.data_root is not None:
            return args.data_root / dataset_id
        return fetch_starter_opacity(dataset_id)

    # BEGIN LOAD TABLES
    h2o_directory = dataset_path("h2o-ckd-v1")
    h2o = OpaCKD.from_external("exomolop", str(h2o_directory / "h2o_ckd.h5"))
    co_directory = dataset_path("co-diffgrid-v1")
    co = OpaDiffgrid.from_saved_opa(str(co_directory / "co_diffgrid.npz"))
    co.check_pressure_grid(np.asarray(co_atmosphere(co).pressure))
    # END LOAD TABLES

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True)
    h2o_wavelength = 1.0e4 / np.asarray(h2o.nu_bands)
    for temperature in (900.0, 1100.0, 1300.0):
        transit_depth = h2o_transmission(h2o, temperature)
        axes[0].plot(
            h2o_wavelength[::-1], transit_depth[::-1], label=f"{temperature:g} K"
        )
    axes[0].set(
        xlabel="Wavelength [micrometer]",
        ylabel="Transit depth [ppm]",
        title="H2O line absorption",
    )
    axes[0].legend()

    # BEGIN TEMPERATURE DERIVATIVE
    flux, flux_derivative = jax.jvp(
        lambda temperature: co_emission(co, temperature),
        (jnp.asarray(1000.0),),
        (jnp.asarray(1.0),),
    )
    # END TEMPERATURE DERIVATIVE
    co_wavelength = 1.0e4 / np.asarray(co.nu_grid)
    axes[1].plot(co_wavelength[::-1], flux[::-1])
    axes[1].set(
        xlabel="Wavelength [micrometer]",
        ylabel=r"$F_{\tilde\nu}$ [erg s$^{-1}$ cm$^{-2}$ / cm$^{-1}$]",
        title="CO line emission, T(1 bar) = 1000 K",
    )
    axes[2].plot(co_wavelength[::-1], flux_derivative[::-1])
    axes[2].set(
        xlabel="Wavelength [micrometer]",
        ylabel=r"$\partial F_{\tilde\nu}/\partial T_0$ [flux units / K]",
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

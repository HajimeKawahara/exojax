"""Dry gray radiative-convective equilibrium with a black lower boundary.

Run ``JAX_PLATFORMS=cpu MPLBACKEND=Agg python examples/rce_gray.py``.
No opacity database is needed. The script saves ``rce_gray.png``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from exojax.atm.atmprof import pressure_layer_logspace_from_boundaries
from exojax.atm.rce import reconstruct_boundary_temperature, solve_rce
from exojax.rt.flux import direct_beam_fluxes, rtrun_emis_pureabs_ibased_linsap_fluxes
from exojax.rt.rtransfer import initialize_gaussian_quadrature


def gray_column(nlayer=48, nangle=4, irradiation_temperature=200.0):
    """Return pressure centers, boundaries, and a converged gray RCE result.

    Prescribed thermal depth grows as P squared, allowing deep dry convection.
    Stellar absorption is proportional to pressure. Irradiation_temperature
    defines the incident horizontal flux as sigma*T**4, not a stellar surface
    temperature. All fluxes are in erg/s/cm2; pressures are in bar.
    """
    sigma = 5.670374419e-5
    internal_temperature = 150.0
    pressure, dp, _, boundaries = pressure_layer_logspace_from_boundaries(
        -2.0, 2.0, nlayer
    )
    dtau = jnp.diff(100.0 * (boundaries / boundaries[-1]) ** 2)
    stellar = direct_beam_fluxes(0.1 * dp, sigma * irradiation_temperature**4, 0.5)
    mus, weights = initialize_gaussian_quadrature(nangle)
    upper_fraction = (pressure - boundaries[:-1]) / dp

    def radiative_flux(temperature, bottom_temperature):
        boundary_temperature = reconstruct_boundary_temperature(
            pressure, boundaries, temperature, bottom_temperature
        )
        up, down = rtrun_emis_pureabs_ibased_linsap_fluxes(
            dtau, sigma * boundary_temperature**4, mus, weights,
            source_center=sigma * temperature**4, upper_fraction=upper_fraction,
        )
        return up - down - stellar

    initial = (
        internal_temperature**4 * (0.5 + 75.0 * (pressure / boundaries[-1])**2)
        + irradiation_temperature**4
    ) ** 0.25
    result = solve_rce(
        pressure, boundaries, initial, float(initial[-1]),
        sigma * internal_temperature**4, radiative_flux,
        adiabatic_gradient=2.0 / 7.0,
    )
    return pressure, boundaries, result


def main():
    import matplotlib.pyplot as plt

    jax.config.update("jax_enable_x64", True)
    pressure, boundaries, result = gray_column()
    if not result.converged:
        raise RuntimeError(f"RCE did not converge: {result.status}")
    print(f"Status: {result.status}; Newton steps: {result.iterations}")
    print(f"Bottom temperature: {result.bottom_temperature:.3f} K")
    print(f"Maximum energy residual: {np.max(np.abs(result.flux_residual)):.3e} erg/s/cm2")
    figure, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
    axes[0].plot(np.append(result.temperature, result.bottom_temperature),
                 np.append(pressure, boundaries[-1]), color="black")
    axes[0].scatter(np.append(result.temperature[1:], result.bottom_temperature)[result.convective_mask],
                    np.append(pressure[1:], boundaries[-1])[result.convective_mask],
                    label="Convective connections", color="tab:orange", s=12)
    axes[0].set(xlabel="Temperature (K)", ylabel="Pressure (bar)", yscale="log")
    axes[0].invert_yaxis()
    axes[0].legend()
    axes[1].plot(result.radiative_flux, boundaries, label="Net radiation")
    axes[1].plot(result.convective_flux, boundaries, label="Convection")
    axes[1].plot(result.radiative_flux + result.convective_flux, boundaries,
                 "--", color="black", label="Total")
    axes[1].set_xlabel(r"Upward flux (erg s$^{-1}$ cm$^{-2}$)")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig("rce_gray.png", dpi=160)
    print("Saved rce_gray.png")


if __name__ == "__main__":
    main()

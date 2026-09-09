"""Forward spectral RCE for a saturated N2--H2O ocean planet.

Prepare HITRAN line opacity with rce_earth_opacity.py first. This example
solves for the temperature profile and ocean temperature, then saves the
outgoing spectrum and the energy budget. See the forward RCE tutorial for
the data sources and the deliberately simplified Earth-analogue assumptions.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from exojax.atm.rce import reconstruct_boundary_temperature, solve_rce
from exojax.opacity import OpaCKD
from exojax.opacity.ckd.core import gauss_legendre_grid
from exojax.rt.flux import (
    direct_beam_fluxes,
    integrate_ckd_flux,
    rtrun_emis_pureabs_ibased_linsap_fluxes,
)
from exojax.rt.layeropacity import layer_optical_depth, layer_optical_depth_ckd
from exojax.rt.planck import piB, piBarr

from rce_earth_continuum import load_continuum
from rce_earth_physics import (
    MOLAR_MASS_H2O,
    MOLAR_MASS_N2,
    convective_gradient,
    saturation_vapor_pressure,
    water_vmr,
)


class EarthColumn:
    """Fixed-total-pressure N2 column with an ocean and a cold trap.

    Fluxes use erg/s/cm2 internally (1000 erg/s/cm2 = 1 W/m2).
    HITRAN air widths and the MT_CKD foreign continuum approximate N2
    collisions. The line table neglects self broadening; the continuum
    includes both self and foreign contributions at the current water VMR.
    """

    def __init__(self, opacity, continuum, nlayer=24, solar_constant=1361.0,
                 albedo=0.20, nangle=4):
        self.opacity = opacity
        self.continuum = continuum
        self.solar_constant = solar_constant
        self.albedo = albedo
        self.boundaries = jnp.geomspace(1.0e-4, 1.0, nlayer + 1)
        self.pressure = jnp.sqrt(self.boundaries[:-1] * self.boundaries[1:])
        self.dp = jnp.diff(self.boundaries)
        self.upper_fraction = (self.pressure - self.boundaries[:-1]) / self.dp
        self.mus, self.angular_weights = gauss_legendre_grid(nangle)
        self.g_weights = opacity.ckd_info.weights

        # Transparent tails complete the bolometric integral. H2O line data
        # cover the intervening thermal and solar absorption bands.
        edges = np.asarray(opacity.band_edges)
        if (edges[0, 0] > 20.0 or edges[-1, 1] < 30000.0
                or not np.all(edges[:, 1] > edges[:, 0])
                or not np.allclose(edges[1:, 0], edges[:-1, 1], rtol=0.0, atol=1.0e-10)):
            raise ValueError("The RCE table must continuously cover 20--30000 cm-1.")
        tail = np.geomspace(edges[-1, 1], 1.0e5, 17)
        self.band_edges = jnp.asarray(np.vstack((
            [[1.0e-3, edges[0, 0]]], edges,
            np.column_stack((tail[:-1], tail[1:])),
        )))
        self.nu = jnp.mean(self.band_edges, axis=1)
        self.widths = jnp.diff(self.band_edges, axis=1).ravel()
        self.absorbed_solar = 1000.0 * (1.0 - albedo) * solar_constant / 4.0
        solar_shape = piB(5772.0, self.nu)
        self.stellar_top = (
            self.absorbed_solar * solar_shape / jnp.sum(solar_shape * self.widths)
        )

    def composition(self, temperature, bottom_temperature):
        return water_vmr(self.pressure, temperature, bottom_temperature,
                         self.boundaries[-1])

    def adiabat(self, temperature, bottom_temperature):
        return convective_gradient(self.pressure, temperature, bottom_temperature,
                                   self.boundaries[-1])

    def optical_depth(self, temperature, bottom_temperature):
        vmr = self.composition(temperature, bottom_temperature)[:-1]
        mean_mass = 1000.0 * (
            MOLAR_MASS_N2 * (1.0 - vmr) + MOLAR_MASS_H2O * vmr
        )
        line_sigma = self.opacity.xstensor_ckd(temperature, self.pressure)
        line_sigma = jnp.pad(line_sigma, ((0, 0), (0, 0), (1, 16)))
        continuum_sigma = self.continuum(temperature, self.pressure, vmr, self.nu)
        return layer_optical_depth_ckd(
            self.dp, line_sigma + continuum_sigma[:, None, :],
            vmr, mean_mass, 980.665,
        )

    def spectral_fluxes(self, temperature, bottom_temperature):
        """Return upward thermal, downward thermal, and direct solar spectra."""
        dtau = self.optical_depth(temperature, bottom_temperature)
        boundary_temperature = reconstruct_boundary_temperature(
            self.pressure, self.boundaries, temperature, bottom_temperature
        )
        upward, downward = rtrun_emis_pureabs_ibased_linsap_fluxes(
            dtau, piBarr(boundary_temperature, self.nu)[:, None, :],
            self.mus, self.angular_weights,
            source_center=piBarr(temperature, self.nu)[:, None, :],
            upper_fraction=self.upper_fraction,
        )
        stellar = direct_beam_fluxes(dtau, self.stellar_top[None, :], 0.5)
        return upward, downward, stellar

    def radiative_flux(self, temperature, bottom_temperature):
        up, down, stellar = self.spectral_fluxes(temperature, bottom_temperature)
        return integrate_ckd_flux(up - down - stellar, self.g_weights, self.widths)

    def valid_state(self, temperature, bottom_temperature):
        tmin, tmax = np.asarray(self.opacity.ckd_info.T_grid)[[0, -1]]
        pmin, pmax = np.asarray(self.opacity.ckd_info.P_grid)[[0, -1]]
        return bool(
            np.all((temperature >= tmin) & (temperature <= tmax))
            and np.all((self.pressure >= pmin) & (self.pressure <= pmax))
            and 273.16 <= bottom_temperature <= tmax
            and float(saturation_vapor_pressure(bottom_temperature)) < 1.0e4
        )

    def solve(self, surface_initial=288.0, flux_atol=1.0e-6):
        """Solve the spectral energy balance; surface_initial is only a guess."""
        # A cold, dry upper atmosphere is possible without CO2 or ozone.
        # This is an initial guess; its temperatures and convective mask evolve.
        initial = jnp.maximum(120.0, surface_initial * self.pressure**0.23)
        return solve_rce(
            self.pressure, self.boundaries, initial, surface_initial,
            0.0, self.radiative_flux, adiabatic_gradient=self.adiabat,
            convective_mask_initial=np.asarray(self.pressure > 0.05),
            valid_state=self.valid_state, flux_atol=flux_atol, flux_rtol=0.0,
        )


def save_result(column, result, output):
    """Save the solved profile, band spectra, and a reproducible diagnostic plot."""
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    temperature = np.append(result.temperature, result.bottom_temperature)
    pressure = np.append(column.pressure, column.boundaries[-1])
    vmr = np.asarray(column.composition(result.temperature, result.bottom_temperature))
    up, down, stellar = column.spectral_fluxes(result.temperature, result.bottom_temperature)
    spectra = [np.asarray(jnp.sum(f * column.g_weights[None, :, None], axis=1)) / 1000.0
               for f in (up, down, stellar)]
    np.savez(output / "rce_earth.npz", pressure_bar=pressure, temperature_K=temperature,
             water_vmr=vmr, boundaries_bar=column.boundaries,
             convective_mask=result.convective_mask,
             radiative_flux_W_m2=result.radiative_flux / 1000.0,
             convective_flux_W_m2=result.convective_flux / 1000.0,
             flux_residual_W_m2=result.flux_residual / 1000.0,
             gradient_residual=result.gradient_residual,
             solar_constant_W_m2=column.solar_constant, albedo=column.albedo,
             opacity_metadata=json.dumps(column.opacity._expected_base_meta),
             opacity_T_grid_K=column.opacity.ckd_info.T_grid,
             opacity_P_grid_bar=column.opacity.ckd_info.P_grid,
             ggrid=column.opacity.ckd_info.ggrid, g_weights=column.g_weights,
             continuum_metadata=json.dumps(getattr(column.continuum, "metadata", {})),
             band_edges_cm1=column.band_edges, upward_W_m2_cm=spectra[0],
             downward_W_m2_cm=spectra[1], stellar_W_m2_cm=spectra[2])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].plot(temperature, pressure, color="black")
    axes[0].scatter(temperature[1:][result.convective_mask],
                    pressure[1:][result.convective_mask], color="tab:orange", s=12,
                    label="Convective connections")
    axes[0].set(xlabel="Temperature (K)", ylabel="Pressure (bar)", yscale="log")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8)
    for flux, label in ((result.radiative_flux, "Net radiation"),
                        (result.convective_flux, "Convection"),
                        (result.radiative_flux + result.convective_flux, "Total")):
        axes[1].plot(flux / 1000.0, column.boundaries, label=label)
    axes[1].set(xlabel=r"Upward flux (W m$^{-2}$)", yscale="log")
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)
    thermal = np.asarray(column.nu) < 3000.0
    axes[2].plot(np.asarray(column.nu)[thermal], spectra[0][0, thermal], label="TOA emission (bands)")
    axes[2].plot(np.asarray(column.nu)[thermal],
                 np.asarray(piB(result.bottom_temperature, column.nu))[thermal] / 1000.0,
                 "--", color="gray", label="Ocean blackbody")
    axes[2].set(xlabel=r"Wavenumber (cm$^{-1}$)",
                ylabel=r"Flux density (W m$^{-2}$ / cm$^{-1}$)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "rce_earth.png", dpi=160)
    plt.close(fig)


def save_line_spectrum(column, result, parfile, output, dnu=0.02):
    """Recompute a resolved thermal spectrum from lines at the converged state.

    This is a separate line-by-line calculation, not an inversion of the
    sorted k distribution. It uses the same local-line/continuum convention.
    """
    import matplotlib.pyplot as plt
    from rce_earth_opacity import WaterLineOpacity

    metadata = column.opacity._expected_base_meta
    if hashlib.sha256(Path(parfile).read_bytes()).hexdigest() != metadata["source"]["sha256"]:
        raise ValueError("The line spectrum must use the HITRAN file used for the CKD table.")
    opacity = WaterLineOpacity(
        parfile, metadata["settings"]["selection_temperatures_K"],
        metadata["settings"]["strength_cutoff_cm"],
    )
    temperature = jnp.asarray(result.temperature)
    boundary_temperature = reconstruct_boundary_temperature(
        column.pressure, column.boundaries, temperature, result.bottom_temperature
    )
    vmr = column.composition(temperature, result.bottom_temperature)[:-1]
    mean_mass = 1000.0 * (MOLAR_MASS_N2 * (1.0 - vmr) + MOLAR_MASS_H2O * vmr)
    wavenumbers, fluxes = [], []
    for lower in np.arange(500.0, 2500.0, 25.0):
        count = int(np.ceil(25.0 / dnu))
        nu = lower + (np.arange(count) + 0.5) * 25.0 / count
        cross_section = opacity.cross_sections(nu, temperature, column.pressure)
        cross_section += column.continuum(temperature, column.pressure, vmr, nu)
        dtau = layer_optical_depth(column.dp, cross_section, vmr, mean_mass, 980.665)
        up, _ = rtrun_emis_pureabs_ibased_linsap_fluxes(
            dtau, piBarr(boundary_temperature, nu), column.mus, column.angular_weights,
            source_center=piBarr(temperature, nu), upper_fraction=column.upper_fraction,
        )
        wavenumbers.append(nu)
        fluxes.append(np.asarray(up[0]) / 1000.0)
        if (lower - 500.0) % 500.0 == 0:
            print(f"Resolved emission: {lower:g}--{lower + 25:g} cm-1", flush=True)
    nu, flux = np.concatenate(wavenumbers), np.concatenate(fluxes)
    np.savez(Path(output) / "rce_earth_line_spectrum.npz", wavenumber_cm1=nu,
             upward_W_m2_cm=flux, spectral_step_cm1=25.0 / count)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(nu, flux, linewidth=0.4, label="Line-by-line TOA emission")
    up, _, _ = column.spectral_fluxes(temperature, result.bottom_temperature)
    band_flux = jnp.sum(up[0] * column.g_weights[:, None], axis=0) / 1000.0
    ax.plot(column.nu, band_flux, color="tab:orange", label="CKD band means")
    ax.plot(nu, piB(result.bottom_temperature, nu) / 1000.0, "--", color="gray",
            label="Ocean blackbody")
    ax.set(xlim=(500.0, 2500.0), xlabel=r"Wavenumber (cm$^{-1}$)",
           ylabel=r"Flux density (W m$^{-2}$ / cm$^{-1}$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / "rce_earth_line_spectrum.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True, help="HITRAN CKD table")
    parser.add_argument("--continuum", type=Path, required=True, help="MT_CKD 4.3 netCDF")
    parser.add_argument("--output", type=Path, default=Path("rce_earth_output"))
    parser.add_argument("--nlayer", type=int, default=24)
    parser.add_argument("--surface-initial", type=float, default=288.0)
    parser.add_argument("--solar-constant", type=float, default=1361.0, help="W/m2")
    parser.add_argument("--albedo", type=float, default=0.20,
                        help="Prescribed planetary albedo for the cloud-free analogue")
    parser.add_argument("--flux-atol", type=float, default=1.0e-6,
                        help="Absolute energy-balance tolerance in erg/s/cm2")
    parser.add_argument("--line-data", type=Path,
                        help="Also recompute 500--2500 cm-1 emission from this HITRAN .par")
    parser.add_argument("--line-dnu", type=float, default=0.02,
                        help="Sampling step of the separate line spectrum in cm-1")
    args = parser.parse_args()
    if (not np.all(np.isfinite([args.line_dnu, args.solar_constant, args.albedo, args.flux_atol]))
            or args.line_dnu <= 0 or args.solar_constant <= 0 or not 0 <= args.albedo < 1
            or args.nlayer < 2 or args.flux_atol <= 0):
        parser.error("Require finite positive spacing, solar flux and tolerance, Nlayer >= 2, and 0 <= albedo < 1")
    jax.config.update("jax_enable_x64", True)
    opacity = OpaCKD.from_saved_tables(str(args.table))
    column = EarthColumn(opacity, load_continuum(args.continuum), nlayer=args.nlayer,
                         solar_constant=args.solar_constant, albedo=args.albedo)
    result = column.solve(surface_initial=args.surface_initial, flux_atol=args.flux_atol)
    print(f"Status: {result.status}; Newton steps: {result.iterations}")
    print(f"Ocean temperature: {result.bottom_temperature:.3f} K")
    print(f"Maximum energy residual: {np.max(np.abs(result.flux_residual)) / 1000:.3e} W/m2")
    if not result.converged:
        raise RuntimeError(f"RCE did not converge: {result.status}")
    save_result(column, result, args.output)
    if args.line_data is not None:
        save_line_spectrum(column, result, args.line_data, args.output, args.line_dnu)
    print(f"Saved profile, spectra, and figure to {args.output}")


if __name__ == "__main__":
    main()

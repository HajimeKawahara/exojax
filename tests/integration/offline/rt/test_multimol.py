"""Named PreMODIT mixtures through emission transfer, without database downloads."""

import jax
import jax.numpy as jnp
import numpy as np

from exojax.atm.atmconvert import vmr_to_mmr
from exojax.database.contracts import Lines, MDBMeta, MDBSnapshot
from exojax.opacity.multimol import build_premodit
from exojax.rt import ArtEmisPure
from exojax.rt.layeropacity import layer_optical_depth
from exojax.rt.multimol import layer_optical_depth_multi
from exojax.utils.grids import wavenumber_grid


def _synthetic_snapshots():
    snapshots = {}
    for name, mass, shift in (("H2O", 18.0, 0.0), ("CO", 28.0, 0.7)):
        snapshots[name] = MDBSnapshot(
            meta=MDBMeta(
                dbtype="exomol",
                molmass=mass,
                T_gQT=np.array([300.0, 1000.0, 2000.0]),
                gQT=np.array([1.0, 2.0, 4.0]),
            ),
            lines=Lines(
                nu_lines=np.array([4350.0, 4354.0, 4358.0]) + shift,
                elower=np.array([20.0, 350.0, 900.0]),
                line_strength_ref_original=np.array([2.0e-22, 4.0e-22, 3.0e-22]),
            ),
            n_Texp=np.full(3, 0.5),
            alpha_ref=np.full(3, 0.06),
        )
    return snapshots


def test_named_emission_gradients_and_vmr_conversion():
    """Exercise the tutorial's builder, mixture, emission, and gradient chain."""
    nu_grid, _, _ = wavenumber_grid(
        22920.0, 23000.0, 64, unit="AA", xsmode="premodit"
    )
    opas = build_premodit(
        _synthetic_snapshots(),
        nu_grid,
        auto_trange=(500.0, 1500.0),
        broadening_resolution={"mode": "manual", "value": 0.2},
    )
    art = ArtEmisPure(
        nu_grid=nu_grid, pressure_top=1.0e-3, pressure_btm=10.0, nlayer=3
    )
    gravity = 1.0e5

    def spectrum(T0, log_mmr):
        temperature = T0 * (art.pressure / 1.0) ** 0.05
        mmr = {name: 10.0**value for name, value in log_mmr.items()}
        dtau = layer_optical_depth_multi(
            opas, temperature, art.pressure, art.dParr, mmr=mmr, gravity=gravity
        )
        return art.run(dtau, temperature)

    log_mmr = {"H2O": -3.0, "CO": -3.5}
    flux = jax.jit(spectrum)(1000.0, log_mmr)
    assert flux.shape == (64,)
    assert np.all(np.isfinite(flux))
    assert np.all(flux > 0.0)

    def mean_flux(T0, abundances):
        return jnp.mean(spectrum(T0, abundances))

    value, (dtemperature, dabundances) = jax.jit(
        jax.value_and_grad(mean_flux, argnums=(0, 1))
    )(1000.0, log_mmr)
    np.testing.assert_allclose(value, flux.mean(), rtol=1.0e-12)
    derivatives = [dtemperature, *(dabundances[name] for name in log_mmr)]
    assert np.all(np.isfinite(derivatives))
    assert np.all(np.abs(derivatives) > 0.0)
    differences = [(mean_flux(1000.1, log_mmr) - mean_flux(999.9, log_mmr)) / 0.2]
    for name in log_mmr:
        upper = {**log_mmr, name: log_mmr[name] + 1.0e-4}
        lower = {**log_mmr, name: log_mmr[name] - 1.0e-4}
        differences.append(
            (mean_flux(1000.0, upper) - mean_flux(1000.0, lower)) / 2.0e-4
        )
    np.testing.assert_allclose(derivatives, differences, rtol=1.0e-5, atol=1.0e-10)

    # The mean mass includes transparent H2/He as well as the line absorbers.
    vmr = {"H2O": 1.0e-4, "CO": jnp.array([2.0e-5, 3.0e-5, 4.0e-5])}
    background = 1.0 - vmr["H2O"] - vmr["CO"]
    mean_mass = background * (0.85 * 2.0 + 0.15 * 4.0) + sum(
        vmr[name] * opa.molmass for name, opa in opas.items()
    )
    mmr = {
        name: vmr_to_mmr(vmr[name], opa.molmass, mean_mass)
        for name, opa in opas.items()
    }
    temperature = 1000.0 * (art.pressure / 1.0) ** 0.05
    actual = layer_optical_depth_multi(
        opas, temperature, art.pressure, art.dParr, mmr=mmr, gravity=gravity
    )
    expected = sum(
        layer_optical_depth(
            art.dParr, opa.xsmatrix(temperature, art.pressure),
            vmr[name], mean_mass, gravity,
        )
        for name, opa in opas.items()
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-15)
    np.testing.assert_allclose(
        art.run(actual, temperature), art.run(expected, temperature), rtol=1.0e-12
    )

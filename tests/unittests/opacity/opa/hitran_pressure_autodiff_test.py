"""Pressure tracing must work for both HITRAN opacity matrix APIs."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.opacity import OpaDirect, OpaModit


@pytest.mark.parametrize("calculator", [OpaDirect, OpaModit])
def test_hitran_xsmatrix_pressure_jit_and_gradient(calculator):
    with jax.experimental.enable_x64():
        nu_lines = np.array([1000.3, 1000.7])
        mdb = SimpleNamespace(
            dbtype="hitran",
            gpu_transfer=True,
            nu_lines=nu_lines,
            dev_nu_lines=jnp.asarray(nu_lines),
            logsij0=jnp.log(jnp.array([1.0e-20, 2.0e-20])),
            elower=jnp.array([0.0, 100.0]),
            n_air=jnp.array([0.5, 0.7]),
            gamma_air=jnp.array([0.05, 0.08]),
            gamma_self=jnp.array([0.1, 0.2]),
            A=jnp.array([1.0, 2.0]),
            molmass=28.0,
            qr_interp_lines=lambda temperature, reference: jnp.full(
                2, temperature / reference
            ),
        )
        nu_grid = np.geomspace(1000.0, 1001.0, 64)
        temperatures = jnp.array([800.0, 1000.0])
        pressures = jnp.array([0.3, 1.0])
        kwargs = {}
        if calculator is OpaModit:
            kwargs = dict(
                Tarr_list=[temperatures * 0.8, temperatures * 1.2], Parr=pressures
            )
        opa = calculator(mdb, nu_grid, **kwargs)

        expected = opa.xsmatrix(temperatures, pressures)
        actual = jax.jit(opa.xsmatrix)(temperatures, pressures)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=0.0)

        def signal(pressure_scale):
            return (
                jnp.sum(opa.xsmatrix(temperatures, pressures * pressure_scale)) * 1.0e20
            )

        derivative = jax.jit(jax.grad(signal))(1.0)
        step = 1.0e-5
        finite_difference = (signal(1.0 + step) - signal(1.0 - step)) / (2.0 * step)
        assert np.isfinite(derivative)
        np.testing.assert_allclose(derivative, finite_difference, rtol=1.0e-5)

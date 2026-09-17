"""Differentiate a synthetic CKD mixture through existing radiative transfer."""

import jax
import jax.numpy as jnp
import numpy as np

from exojax.opacity.ckd.api import OpaCKD
from exojax.opacity.ckd.contracts import CKDTableInfo
from exojax.opacity.ckd.mixing import mix_ckd_rorr, validate_ckd_mixture_tables
from exojax.rt import ArtEmisPure, ArtTransPure
from exojax.rt.layeropacity import layer_optical_depth_ckd


def test_multilayer_emission_converges_to_independent_spectral_reference():
    bands = jnp.array([1000.0, 2000.0])
    temperature = jnp.array([650.0, 900.0, 1200.0])
    layer_scale = jnp.array([0.05, 0.2, 0.8])
    emission = ArtEmisPure(nlayer=3, nu_grid=bands, rtsolver="ibased", nstream=4)

    def distributions(g):
        first = (0.02 + 2.0 * g**3)[:, None] * jnp.array([1.0, 1.5])
        second = (0.03 + 0.7 * g**2)[:, None] * jnp.array([2.0, 0.5])
        return jnp.stack([first, second])

    # The Cartesian product defines synthetic spectral samples with independent
    # absorbers, not a validation of random overlap for real molecular lines.
    # Common layer scaling preserves every mixed rank through the atmosphere.
    nreference = 64
    reference_species = distributions((jnp.arange(nreference) + 0.5) / nreference)
    reference_pairs = (
        reference_species[0, :, None, :] + reference_species[1, None, :, :]
    ).reshape(nreference**2, 2)
    reference_dtau = layer_scale[:, None, None] * reference_pairs[None, :, :]
    reference_flux = emission.run_ckd(
        reference_dtau,
        temperature,
        jnp.full(nreference**2, 1.0 / nreference**2),
        bands,
    )

    relative_errors = []
    for ng in [8, 16]:
        samples, weights = np.polynomial.legendre.leggauss(ng)
        weights = jnp.asarray(weights / 2)
        species = distributions(jnp.asarray((samples + 1) / 2))
        dtau = species[:, None, :, :] * layer_scale[None, :, None, None]
        flux = emission.run_ckd(
            mix_ckd_rorr(dtau, weights), temperature, weights, bands
        )
        relative_errors.append(jnp.abs(flux / reference_flux - 1.0))

    assert np.all(relative_errors[1] < relative_errors[0])
    assert np.max(relative_errors[1]) < 3e-3


def test_temperature_pressure_and_abundance_gradients_through_radiative_transfer():
    ggrid, weights = np.polynomial.legendre.leggauss(4)
    weights = jnp.asarray(weights / 2)
    bands = jnp.array([1000.0, 2000.0])
    tables = np.sort(
        np.random.default_rng(34).normal(-58.0, 1.2, (2, 2, 2, 4, 2)), axis=3
    )
    opas = []
    for table in tables:
        opa = OpaCKD.load_only()
        opa.ready, opa.Ng = True, 4
        opa.nu_bands = bands
        opa.band_edges = jnp.array([[900.0, 1100.0], [1900.0, 2100.0]])
        opa.ckd_info = CKDTableInfo(
            log_kggrid=jnp.asarray(table),
            ggrid=jnp.asarray((ggrid + 1) / 2),
            weights=weights,
            T_grid=jnp.array([400.0, 1200.0]),
            P_grid=jnp.array([0.01, 10.0]),
            nu_bands=bands,
            band_edges=opa.band_edges,
        )
        opas.append(opa)
    validate_ckd_mixture_tables(opas)
    emission = ArtEmisPure(nlayer=2, nu_grid=bands, rtsolver="ibased", nstream=4)
    transit = ArtTransPure(nlayer=2, pressure_top=0.05, pressure_btm=1.0, nu_grid=bands)

    def model(parameters):
        temperature = jnp.array([650.0, 950.0]) * jnp.exp(parameters[0])
        pressure = jnp.array([0.05, 1.0]) * jnp.exp(parameters[1])
        dpressure = jnp.array([0.08, 1.5]) * jnp.exp(parameters[1])
        vmr = jnp.exp(parameters[2:])
        mmw = 2.3 * (1.0 - vmr.sum()) + jnp.dot(vmr, jnp.array([18.0, 44.0]))
        species = jnp.stack([
            layer_optical_depth_ckd(
                dpressure, opa.xstensor_ckd(temperature, pressure), vmr[i], mmw, 1000.0
            )
            for i, opa in enumerate(opas)
        ])
        mixed = mix_ckd_rorr(species, weights)
        flux = emission.run_ckd(mixed, temperature, weights, bands)
        radius = transit.run_ckd(
            mixed, temperature, jnp.full_like(temperature, mmw), 7e9, 1000.0, weights
        )
        return jnp.stack([jnp.log(flux.sum()), radius.sum()])

    parameters = jnp.array([0.0, 0.0, np.log(0.03), np.log(0.07)])
    forward = jax.jit(jax.jacfwd(model))(parameters)
    reverse = jax.jit(jax.jacrev(model))(parameters)
    step = 1e-5
    finite_difference = jnp.stack([
        (model(parameters + step * direction) - model(parameters - step * direction))
        / (2 * step)
        for direction in jnp.eye(4)
    ], axis=1)

    assert np.all(np.isfinite(forward))
    assert np.all(np.abs(forward) > 1e-8)
    np.testing.assert_allclose(forward, reverse, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(forward, finite_difference, rtol=2e-6, atol=1e-9)

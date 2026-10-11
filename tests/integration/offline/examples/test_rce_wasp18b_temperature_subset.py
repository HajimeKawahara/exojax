"""Stored Hermite-knot selection agrees with full-array transfer and P1."""

from copy import copy
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from tests.integration.offline.examples.test_rce_wasp18b_streaming import fixture_manifest, modules


@pytest.mark.parametrize("step", [4, 2, 1])
def test_temperature_subset_stream_matches_direct_tables(modules, tmp_path, step):
    temperature = 1/np.linspace(1/4500., 1/1500., 16)[::-1]
    manifest, original, chemistry = fixture_manifest(modules, tmp_path, temperature=temperature)
    metadata, plan = modules.stream.load_streaming_opacities(manifest)
    indices = np.unique(np.r_[np.arange(0, 16, step), 15])
    selected = plan.temperature_subset(indices)
    full, streamed = copy(original), copy(original)
    full.lbl = {}
    for name, opacity in original.lbl.items():
        own = copy(opacity)
        info = opacity.diffgrid_info
        own._apply_diffgrid_info(replace(info,
            temperature_grid=info.temperature_grid[indices],
            inverse_temperature_grid=info.inverse_temperature_grid[indices],
            log_cross_section_grid=info.log_cross_section_grid[:, indices],
            log_cross_section_derivative_grid=info.log_cross_section_derivative_grid[:, indices]))
        full.lbl[name] = own
    streamed.lbl = {name: opacity.temperature_subset(indices) for name, opacity in metadata.items()}
    arguments = (jnp.array([2337., 3189.]), 3400., jnp.array([0., .6, np.log(.5)]), chemistry)
    expected = full.fluxes(*arguments, resolved=True).upward[0]
    actual = selected.top_flux(streamed, *arguments)
    np.testing.assert_allclose(actual, expected, rtol=2e-12)
    np.testing.assert_allclose(selected.prediction(streamed, actual, 300.),
                               full.spectrum(*arguments, resolving_power=300.), rtol=2e-12)
    if step == 1:
        np.testing.assert_array_equal(actual, plan.top_flux(streamed, *arguments))
    else:
        with pytest.raises(ValueError, match="temperature selection"):
            plan.top_flux(streamed, *arguments)
    assert plan.temperature_indices is None and "temperature_subset" not in plan.provenance
    assert selected.provenance["file_sha256"] == plan.provenance["file_sha256"]
    assert selected.provenance["all_chunk_coefficients_verified"]
    np.testing.assert_array_equal(selected.temperature_subset([0, len(indices)-1]).temperature_indices,
                                  [0, 15])
    for invalid in ([0], [1, 0], [0, 16], [0., 15.]):
        with pytest.raises(ValueError, match="temperature indices"):
            plan.temperature_subset(invalid)

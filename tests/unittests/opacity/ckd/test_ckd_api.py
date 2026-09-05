"""Unit tests for CKD API grid setting."""

import pytest
import numpy as np
import jax.numpy as jnp
from exojax.opacity.ckd.api import OpaCKD
from exojax.rt import ArtAbsPure
from exojax.rt.layeropacity import layer_optical_depth_ckd


class MockBaseOpa:
    """Mock base opacity calculator for testing."""
    def __init__(self):
        self.nu_grid = np.linspace(1000.0, 2000.0, 1000)


class PairedMockBaseOpa:
    """Base opacity mock whose xsmatrix inputs are layer-aligned."""

    def __init__(self):
        self.nu_grid = jnp.linspace(1000.0, 1003.0, 4)

    def xsmatrix(self, T_array, P_array):
        cross_section = T_array + 10.0 * P_array
        return jnp.broadcast_to(
            cross_section[:, None], (cross_section.size, self.nu_grid.size)
        )


class SparseMockBaseOpa:
    """Base opacity mock with an uncovered spectral band."""

    def __init__(self):
        self.nu_grid = jnp.array([1000.0, 1003.0])

    def xsmatrix(self, T_array, P_array):
        return jnp.full(
            (T_array.size, self.nu_grid.size), 2.0e-20, dtype=jnp.float32
        )


def test_opa_ckd_init():
    """Test basic OpaCKD initialization."""
    mock_base_opa = MockBaseOpa()
    
    # Test initialization
    opa_ckd = OpaCKD(mock_base_opa, Ng=16, band_width=100.0)
    
    # Check basic attributes
    assert opa_ckd.method == "ckd"
    assert opa_ckd.Ng == 16
    assert opa_ckd.band_spacing == "log"  # Default
    assert len(opa_ckd.nu_bands) > 0
    assert opa_ckd.ready == False
    
    # Check that band_edges are now available (new feature)
    assert hasattr(opa_ckd, 'band_edges')
    assert opa_ckd.band_edges.shape == (len(opa_ckd.nu_bands), 2)


def test_opa_ckd_custom_bands():
    """Test OpaCKD with custom band settings."""
    mock_base_opa = MockBaseOpa()
    
    # Test with different band width and spacing
    opa_ckd = OpaCKD(mock_base_opa, band_width=200.0, band_spacing="linear")
    
    # Check that bands were auto-generated with the specified parameters
    assert opa_ckd.band_width == 200.0
    assert opa_ckd.band_spacing == "linear"
    assert len(opa_ckd.nu_bands) > 0
    assert hasattr(opa_ckd, 'band_edges')


def test_precompute_tables_uses_cartesian_temperature_pressure_grid():
    opa_ckd = OpaCKD(
        PairedMockBaseOpa(), Ng=2, band_width=10.0, band_spacing="linear"
    )
    T_grid = jnp.array([1.0, 2.0])
    P_grid = jnp.array([10.0, 20.0])

    opa_ckd.precompute_tables(T_grid, P_grid)

    expected = T_grid[:, None] + 10.0 * P_grid[None, :]
    actual = jnp.exp(opa_ckd.ckd_info.log_kggrid[:, :, :, 0])
    assert jnp.allclose(actual, expected[:, :, None])


def test_precompute_tables_uses_opacity_floor_for_empty_band():
    opa_ckd = OpaCKD(
        SparseMockBaseOpa(), Ng=2, band_width=1.0, band_spacing="linear"
    )

    opa_ckd.precompute_tables(jnp.array([1000.0]), jnp.array([1.0]))

    actual = opa_ckd.xsarray_ckd(1000.0, 1.0)[:, 1]
    assert jnp.all(actual < 1.0e-37)


@pytest.mark.parametrize("gravity", [1000.0, 100.0])
def test_empty_band_is_transparent_in_radiative_transfer(gravity):
    opa_ckd = OpaCKD(
        SparseMockBaseOpa(), Ng=2, band_width=1.0, band_spacing="linear"
    )
    opa_ckd.precompute_tables(jnp.array([1000.0]), jnp.array([1.0]))
    xs_ckd = opa_ckd.xstensor_ckd(
        jnp.array([1000.0, 1000.0]), jnp.array([1.0, 1.0])
    )
    art = ArtAbsPure(
        pressure_top=1.0e-8,
        pressure_btm=100.0,
        nlayer=2,
        nu_grid=opa_ckd.nu_bands,
    )

    dtau_ckd = layer_optical_depth_ckd(
        jnp.array([50.0, 50.0]), xs_ckd, jnp.ones(2), 2.3, gravity
    )
    transmission = art.run_ckd(
        dtau_ckd,
        art.pressure_boundary[-1],
        jnp.ones_like(opa_ckd.nu_bands),
        1.0,
        None,
        opa_ckd.ckd_info.weights,
    )

    assert transmission[1] > 1.0 - 1.0e-7

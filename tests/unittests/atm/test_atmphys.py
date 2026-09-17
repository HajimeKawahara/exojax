import jax.numpy as jnp
import pytest

from exojax.atm import atmprof
from exojax.atm.atmphys import AmpAmcloud
from exojax.utils.constants import bar_cgs, kB, m_u


class _DummyPdb:
    condensate_substance_density = 0.84

    def saturation_pressure(self, temperatures):
        return jnp.array([10.0, 1.0e4])


def test_calc_ammodel_rw_uses_cgs_density_once():
    # The deep layer separates a missing bar conversion from double subtraction.
    pressures = jnp.array([100.0, 1000.0])
    temperatures = jnp.full_like(pressures, 300.0)
    mean_molecular_weight = 2.22
    gravity = 2478.6
    target_velocity = 1.0e-2
    scale_height = atmprof.pressure_scale_height(
        gravity, temperatures[0], mean_molecular_weight
    )
    amp = AmpAmcloud(
        _DummyPdb(),
        bkgatm="H2",
        size_min=1.0e-7,
        size_max=1.0e-3,
        nsize=4000,
    )

    rw, _ = amp.calc_ammodel_rw(
        pressures,
        temperatures,
        mean_molecular_weight=mean_molecular_weight,
        molecular_mass_condensate=mean_molecular_weight,
        gravity=gravity,
        fsed=1.0,
        Kzz=target_velocity * scale_height,
        MMR_base=1.0,
    )

    rho_atm = (
        mean_molecular_weight
        * m_u
        * pressures[1]
        * bar_cgs
        / (kB * temperatures[1])
    )
    dynamic_viscosity = amp.dynamic_viscosity(temperatures)[1]
    expected_rw = jnp.sqrt(
        9.0
        * dynamic_viscosity
        * target_velocity
        / (2.0 * gravity * (_DummyPdb.condensate_substance_density - rho_atm))
    )

    assert rw[1] == pytest.approx(expected_rw, rel=5.0e-3)


def test_calc_ammodel_rw_with_layer_dependent_kzz():
    pressures = jnp.array([100.0, 1000.0])
    temperatures = jnp.full_like(pressures, 300.0)
    mean_molecular_weight = 2.22
    gravity = 2478.6
    scale_height = atmprof.pressure_scale_height(
        gravity, temperatures[0], mean_molecular_weight
    )
    amp = AmpAmcloud(
        _DummyPdb(),
        bkgatm="H2",
        size_min=1.0e-7,
        size_max=1.0e-3,
        nsize=4000,
    )

    def calculate_rw(Kzz):
        return amp.calc_ammodel_rw(
            pressures,
            temperatures,
            mean_molecular_weight=mean_molecular_weight,
            molecular_mass_condensate=mean_molecular_weight,
            gravity=gravity,
            fsed=1.0,
            Kzz=Kzz,
            MMR_base=1.0,
        )[0]

    Kzz = jnp.array([1.0e-3, 1.0e-2]) * scale_height
    rw = calculate_rw(Kzz)
    expected_rw = jnp.array(
        [calculate_rw(layer_Kzz)[i] for i, layer_Kzz in enumerate(Kzz)]
    )

    assert rw.shape == pressures.shape
    assert jnp.array_equal(rw, expected_rw)

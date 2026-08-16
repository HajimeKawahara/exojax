import jax.numpy as jnp
import pytest

from exojax.atm import viscosity
from exojax.atm import atmprof
from exojax.atm import vterm
from exojax.atm.atmphys import AmpAmcloud
from exojax.atm.amclouds import sigmag_from_effective_radius
from exojax.atm.amclouds import effective_radius
from exojax.atm.amclouds import get_rg
from exojax.utils.constants import bar_cgs, kB, m_u


class _DummyPdb:
    condensate_substance_density = 0.84

    def saturation_pressure(self, temperatures):
        return jnp.array([10.0, 1.0e4])


def test_viscosity():
    T = 1000.0  # K
    assert viscosity.eta_Rosner_H2(T) == pytest.approx(0.0001929772857173383)


def test_pressure_scale_height_for_Earth():
    g = 980.0  # cm^2/s
    T = 300.0  # K
    mu = 28.8
    ref = 883764.8664527453

    assert atmprof.pressure_scale_height(g, T, mu) == pytest.approx(ref)


def test_terminal_velocity():
    g = 980.0
    rho_cloud = 1.0
    rho = 1.29 * 1.0e-3  # g/cm3
    vfactor, Tr = viscosity.calc_vfactor(atm="Air")
    eta = viscosity.eta_Rosner(300.0, vfactor)
    r = jnp.logspace(-5, 0, 70)
    vfall = vterm.terminal_velocity(r, g, eta, rho_cloud, rho)
    assert jnp.mean(vfall) == pytest.approx(328.12296)


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


def _am01_test_param_set():
    rw = 1.0e-4
    fsed = 2.0
    alpha = 2.0
    sigmag = 2.0

    rg_ref = 2.0695821e-05  # computed from get_rg
    reff_ref = 6.879041e-05  # computed from effective_radius
    return rw, fsed, alpha, sigmag, rg_ref, reff_ref


def test_get_rg():
    rw, fsed, alpha, sigmag, rg_ref, _ = _am01_test_param_set()
    assert get_rg(rw, fsed, alpha, sigmag) == pytest.approx(rg_ref)


def test_effective_radius():
    _, _, _, sigmag, rg_ref, reff_ref = _am01_test_param_set()
    assert effective_radius(rg_ref, sigmag) == pytest.approx(reff_ref)


def test_sigmag_from_effective_radius():
    rw, fsed, alpha, sigmag_ref, rg, reff = _am01_test_param_set()
    val = sigmag_from_effective_radius(reff, fsed, rw, alpha)
    assert val == pytest.approx(sigmag_ref)


if __name__ == "__main__":
    test_sigmag_from_effective_radius()

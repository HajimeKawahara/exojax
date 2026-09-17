import pytest

from exojax.atm import viscosity


def test_viscosity():
    T = 1000.0  # K
    assert viscosity.eta_Rosner_H2(T) == pytest.approx(0.0001929772857173383)

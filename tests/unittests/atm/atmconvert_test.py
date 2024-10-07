import pytest
from exojax.atm.atmconvert import mmr_to_vmr
from exojax.atm.atmconvert import vmr_to_mmr
from exojax.atm.atmconvert import mmr_to_density


def test_mmr2vmr():
    # Test case 1
    mmr = 0.02
    molecular_mass = 18.01528  # H2O
    mean_molecular_weight = 28.97  # Earth's atmosphere
    expected_vmr = 0.03216158727480228
    assert mmr_to_vmr(mmr, molecular_mass, mean_molecular_weight) == pytest.approx(
        expected_vmr, rel=1e-3
    )


def test_vmr2mmr():
    # Test case 1
    vmr = 0.03216158727480228
    molecular_mass = 18.01528  # H2O
    mean_molecular_weight = 28.97  # Earth's atmosphere
    expected_mmr = 0.02
    assert vmr_to_mmr(vmr, molecular_mass, mean_molecular_weight) == pytest.approx(
        expected_mmr, rel=1e-3
    )


def test_mmr_to_density_g_per_L():
    mmr = 0.02
    molmass = 18.01528  # H2O
    Parr = 1.0  # bar
    Tarr = 300.0  # K
    expected_density = 0.014444939134608615
    assert mmr_to_density(mmr, molmass, Parr, Tarr, unit="g/L") == pytest.approx(
        expected_density, rel=1e-3
    )


def test_mmr_to_density_g_per_cm3():
    mmr = 0.02
    molmass = 18.01528  # H2O
    Parr = 1.0  # bar
    Tarr = 300.0  # K
    expected_density = (
        1.4444939134608615e-05
    )
    assert mmr_to_density(mmr, molmass, Parr, Tarr, unit="g/cm3") == pytest.approx(
        expected_density, rel=1e-3
    )


def test_mmr_to_density_invalid_unit():
    mmr = 0.02
    molmass = 18.01528  # H2O
    Parr = 1.0  # bar
    Tarr = 300.0  # K
    with pytest.raises(ValueError, match="unit is not correct"):
        mmr_to_density(mmr, molmass, Parr, Tarr, unit="invalid_unit")

if __name__ == "__main__":
    test_mmr2vmr()
    test_vmr2mmr()
    test_mmr_to_density_g_per_L()
    test_mmr_to_density_g_per_cm3()
    test_mmr_to_density_invalid_unit()
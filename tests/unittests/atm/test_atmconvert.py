import pytest
from exojax.atm.atmconvert import mmr_to_vmr
from exojax.atm.atmconvert import vmr_to_mmr

def test_mmr2vmr():
    # Test case 1
    mmr = 0.02
    molecular_mass = 18.01528  # H2O
    mean_molecular_weight = 28.97  # Earth's atmosphere
    expected_vmr = 0.03216158727480228
    assert mmr_to_vmr(mmr, molecular_mass, mean_molecular_weight) == pytest.approx(expected_vmr, rel=1e-3)

def test_vmr2mmr():
    # Test case 1
    vmr = 0.03216158727480228
    molecular_mass = 18.01528  # H2O
    mean_molecular_weight = 28.97  # Earth's atmosphere
    expected_mmr = 0.02
    assert vmr_to_mmr(vmr, molecular_mass, mean_molecular_weight) == pytest.approx(expected_mmr, rel=1e-3)


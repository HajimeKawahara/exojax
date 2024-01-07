from exojax.spec.unitconvert import wav2nu
from exojax.spec.unitconvert import nu2wav

import pytest
import numpy as np

def test_wav2nu_single():
    wav = 20000.0  # AA
    assert wav2nu(wav, unit="AA") == pytest.approx(5000.0)

def test_wav2nu_check_nus_is_in_ascending():
    wav_ascending = np.array([10000.0,20000.0])  # AA
    wav_descending = np.array([20000.0,10000.0])  # AA
    
    nus_ref = np.array([5000.0,10000.0])  # cm-1
    
    assert np.all(wav2nu(wav_ascending, unit="AA") == nus_ref)
    assert np.all(wav2nu(wav_descending, unit="AA") == nus_ref)
    
def test_nu2wav_single():
    nus_single = 5000.0
    assert nu2wav(nus_single) == pytest.approx(20000.0)

def test_nu2wav_ascending():
    nus_ascending = np.array([5000.0,10000.0])  # cm-1
    wav_ascending = np.array([10000.0,20000.0])  # AA
    wav_descending = np.array([20000.0,10000.0])  # AA
    
    assert np.all(nu2wav(nus_ascending, wavelength_order="ascending") == wav_ascending)
    assert np.all(nu2wav(nus_ascending, wavelength_order="descending") == wav_descending)


def test_nu2wav_when_nus_input_is_ascending_or_single_then_value_error():
    nus_descending = np.array([10000.0,5000.0])  # cm-1
    nus_disorder = np.array([10000.0,20000.0, 5000.0])  # cm-1
    
    with pytest.raises(ValueError):
        nu2wav(nus_descending, wavelength_order="ascending")
    with pytest.raises(ValueError):
        nu2wav(nus_descending, wavelength_order="descending")
    with pytest.raises(ValueError):
        nu2wav(nus_disorder, wavelength_order="ascending")
    with pytest.raises(ValueError):
        nu2wav(nus_disorder, wavelength_order="descending")


if __name__ == "__main__":
    test_wav2nu_single()
    test_wav2nu_check_nus_is_in_ascending()
    test_nu2wav_when_nus_input_is_ascending_or_single_then_value_error()
from exojax.utils.grids import wav2nu
from exojax.utils.grids import nu2wav

import pytest
import numpy as np


def test_wav2nu_single():
    wav = 20000.0  # AA
    assert wav2nu(wav, unit="AA") == pytest.approx(5000.0)


def test_wav2nu_check_nus_is_in_ascending():
    wav_ascending = np.array([10000.0, 20000.0])  # AA
    wav_descending = np.array([20000.0, 10000.0])  # AA

    nus_ref = np.array([5000.0, 10000.0])  # cm-1

    assert np.all(wav2nu(wav_ascending, unit="AA") == nus_ref)
    assert np.all(wav2nu(wav_descending, unit="AA") == nus_ref)


def test_nu2wav_single():
    nus_single = 5000.0
    assert nu2wav(nus_single) == pytest.approx(20000.0)


def test_nu2wav_single_with_values():
    nus_single = 5000.0
    _, vals = nu2wav(nus_single, values=1.0)
    assert vals == 1.0 


def test_wav2nus_single_with_values():
    wav_single = 20000.0
    _, vals = nu2wav(wav_single, values=1.0)
    assert vals == 1.0 


def test_nu2wav_ascending():
    nus_ascending = np.array([5000.0, 10000.0])  # cm-1
    wav_ascending = np.array([10000.0, 20000.0])  # AA
    wav_descending = np.array([20000.0, 10000.0])  # AA

    assert np.all(nu2wav(nus_ascending, wavelength_order="ascending") == wav_ascending)
    assert np.all(
        nu2wav(nus_ascending, wavelength_order="descending") == wav_descending
    )


def test_nu2wav_when_nus_input_is_ascending_or_single_then_value_error():
    nus_descending = np.array([10000.0, 5000.0])  # cm-1
    nus_disorder = np.array([10000.0, 20000.0, 5000.0])  # cm-1

    with pytest.raises(ValueError):
        nu2wav(nus_descending, wavelength_order="ascending")
    with pytest.raises(ValueError):
        nu2wav(nus_descending, wavelength_order="descending")
    with pytest.raises(ValueError):
        nu2wav(nus_disorder, wavelength_order="ascending")
    with pytest.raises(ValueError):
        nu2wav(nus_disorder, wavelength_order="descending")


def test_nu2wav_ascending_with_values():
    nus_ascending = np.array([5000.0, 10000.0])  # cm-1
    vals = np.array([1.0, 2.0])

    _, vals_as = nu2wav(nus_ascending, wavelength_order="ascending", values=vals)
    _, vals_des = nu2wav(nus_ascending, wavelength_order="descending", values=vals)

    assert np.all(vals_as == vals[::-1])
    assert np.all(vals_des == vals)


def test_wav2nu_with_values():
    wav_ascending = np.array([10000.0, 20000.0])  # AA
    wav_descending = np.array([20000.0, 10000.0])  # AA
    vals = np.array([1.0, 2.0])

    _, vals_as = wav2nu(wav_ascending, unit="AA", values=vals)
    _, vals_des = wav2nu(wav_descending, unit="AA", values=vals)

    assert np.all(vals_as == vals[::-1])
    assert np.all(vals_des == vals)


if __name__ == "__main__":
    test_wav2nu_single()
    test_wav2nu_check_nus_is_in_ascending()
    test_nu2wav_when_nus_input_is_ascending_or_single_then_value_error()
    test_nu2wav_ascending_with_values()
    test_wav2nu_with_values()
    test_nu2wav_single_with_values()
    test_wav2nus_single_with_values()
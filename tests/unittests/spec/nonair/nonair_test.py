import pytest
from exojax.database.nonair  import temperature_exponent_nonair, gamma_nonair
from exojax.database.nonair  import  nonair_coeff_CO_in_H2
import numpy as np

def test_n_Texp_nonair_CO_H2():
    m = np.array([2])
    res = temperature_exponent_nonair(m, nonair_coeff_CO_in_H2)

    ans_by_hand = (0.64438 + 0.49261 * 2 - 0.0748 * 4 + 0.0032 * 8) / (
        1.0 + 0.69861 * 2 - 0.09569 * 4 + 0.003 * 8 + 5.7852E-5 * 16)

    assert res[0] == pytest.approx(ans_by_hand)

def test_gamma_nonair_CO_H2():
    m = np.array([2])
    res = gamma_nonair(m, nonair_coeff_CO_in_H2)

    ans_by_hand = (0.08228 -0.07411 * 2 + 0.10795 * 4 + 0.00211 * 8) / (
        1.0  -1.0 * 2 + 1.53458 * 4 + 0.03054 * 8 + 6.9468E-5 * 16)

    assert res[0] == pytest.approx(ans_by_hand)


if __name__ == "__main__":
    m = np.array([2, 4])
    res = (temperature_exponent_nonair(m, nonair_coeff_CO_in_H2))
    print(res)

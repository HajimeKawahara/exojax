"""Unit tests for exojax.atm.amclouds.(cloud base relationals)
"""

import pytest
import numpy as np


def _default_cloud_setting():
    from exojax.atm.atmprof import pressure_layer_logspace
    from exojax.atm.psat import psat_enstatite_AM01
    from exojax.utils.zsol import nsol

    Parr, dParr, k = pressure_layer_logspace(
        log_pressure_top=-4.0, log_pressure_btm=6.0, nlayer=100
    )
    alpha = 0.097
    T0 = 1200.0
    Tarr = T0 * (Parr) ** alpha
    n = nsol()  # solar abundance
    MolMR_enstatite = np.min([n["Mg"], n["Si"], n["O"] / 3])
    P_enstatite = psat_enstatite_AM01(Tarr)
    return Parr, Tarr, MolMR_enstatite, P_enstatite


def test_get_pressure_at_cloud_base():
    """test get_pressure_at_cloud_base"""

    Parr, Tarr, MolMR_enstatite, P_enstatite = _default_cloud_setting()
    from exojax.atm.amclouds import smooth_index_base_pressure
    from exojax.atm.amclouds import get_pressure_at_cloud_base

    smooth_index = smooth_index_base_pressure(Parr, P_enstatite, MolMR_enstatite)

    Pbase_enstatite = get_pressure_at_cloud_base(Parr, smooth_index)

    assert Pbase_enstatite == pytest.approx(104.62701, 1.0e-3)


def test_get_value_at_cloud_base_value_is_temperature():
    """test get_value_at_cloud_base using value = temperatures"""
    from exojax.atm.amclouds import smooth_index_base_pressure
    from exojax.utils.indexing import get_value_at_smooth_index

    Parr, Tarr, MolMR_enstatite, P_enstatite = _default_cloud_setting()
    smooth_index = smooth_index_base_pressure(Parr, P_enstatite, MolMR_enstatite)
    Tbase_enstatite = get_value_at_smooth_index(Tarr, smooth_index)
    ref = 1884.2233
    assert Tbase_enstatite == pytest.approx(ref)


if __name__ == "__main__":
    test_get_pressure_at_cloud_base()
    test_get_value_at_cloud_base_value_is_temperature()

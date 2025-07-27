from exojax.opacity.base import OpaCalc
import pytest
def _set_wavenumber_grid():
    from exojax.utils.grids import wavenumber_grid
    nu_grid = wavenumber_grid(1900.0,2300.0,10000,"modit")
    return nu_grid

def test_opacalc():
    nu_grid = _set_wavenumber_grid()
    opa = OpaCalc(nu_grid)


def test_opacalc_check_alias_invalid():
    nu_grid = _set_wavenumber_grid()
    opa = OpaCalc(nu_grid)
    opa.alias = "invalid"
    with pytest.raises(ValueError):
        opa.set_aliasing()
    
    
def test_opacalc_set_alias_left_right_from_cutwing():
    nu_grid = _set_wavenumber_grid()    
    opa = OpaCalc(nu_grid)
    opa.set_filter_length_oneside_from_cutwing()
    
    assert len(nu_grid) == opa.filter_length_oneside
    assert len(nu_grid) == opa.filter_length_oneside

def test_opacalc_set_alias_left_right_from_cutwing_half():
    nu_grid = _set_wavenumber_grid()    
    opa = OpaCalc(nu_grid)
    opa.cutwing = 0.5
    opa.set_filter_length_oneside_from_cutwing()
    
    assert int(len(nu_grid)/2) == opa.filter_length_oneside
    assert int(len(nu_grid)/2) == opa.filter_length_oneside


if __name__ == "__main__":
    test_opacalc()
    test_opacalc_check_alias_invalid()  
    test_opacalc_set_alias_left_right_from_cutwing()
    test_opacalc_set_alias_left_right_from_cutwing_half()
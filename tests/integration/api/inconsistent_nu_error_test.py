from exojax.database.api  import MdbHitran
from exojax.opacity import OpaPremodit
from exojax.utils.grids import wavenumber_grid
from exojax.utils.checkarray import is_outside_range
import numpy as np
import pytest

def test_raise_error_in_premodit_when_no_mdblines_in_nu_grid():
    wavelength_start = 7100.0 #AA
    wavelength_end = 7450.0 #AA

    wavenumber_start = 7100.0 #cm-1
    wavenumber_end = 7450.0 #cm-1

    N=40000
    mdb_water = MdbHitran("H2O", nurange=[1.e8/wavelength_end, 1.e8/wavelength_start], isotope=0)
    nus, wav, res = wavenumber_grid(wavenumber_start, wavenumber_end, N, xsmode="premodit")

    print("opa nu range", nus[0],nus[-1])
    print("mdb nu range", np.min(mdb_water.nu_lines), np.max(mdb_water.nu_lines))
    print(is_outside_range(mdb_water.nu_lines,nus[0],nus[-1]))

    with pytest.raises(ValueError):
        opa = OpaPremodit(mdb_water, nu_grid=nus, allow_32bit=True, auto_trange=[150.0,250.0])


def test_no_error_in_premodit_when_mdblines_in_nu_grid():
    wavelength_start = 7100.0 #AA
    wavelength_end = 7450.0 #AA

    wavenumber_start = 1.e8/wavelength_end
    wavenumber_end = 1.e8/wavelength_start 

    N=40000
    mdb_water = MdbHitran("H2O", nurange=[1.e8/wavelength_end, 1.e8/wavelength_start], isotope=0)
    nus, wav, res = wavenumber_grid(wavenumber_start, wavenumber_end, N, xsmode="premodit")

    print("opa nu range", nus[0],nus[-1])
    print("mdb nu range", np.min(mdb_water.nu_lines), np.max(mdb_water.nu_lines))
    print(is_outside_range(mdb_water.nu_lines,nus[0],nus[-1]))

    opa = OpaPremodit(mdb_water, nu_grid=nus, allow_32bit=True, auto_trange=[150.0,250.0])

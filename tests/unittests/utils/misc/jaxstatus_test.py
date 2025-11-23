import pytest
from exojax.utils.jaxstatus import check_jax64bit
from exojax.test.emulate_mdb import mock_mdb
from exojax.opacity import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.rt import ArtEmisPure
from jax import config


def test_check_raise_valueerror_when_32bit():
    config.update("jax_enable_x64", False)
    allow_32bit = False
    with pytest.raises(ValueError):
        check_jax64bit(allow_32bit)


def test_check_raise_error_premodit_when_32bit():
    config.update("jax_enable_x64", False)
    db = "exomol"
    diffmode = 0
    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtEmisPure(pressure_top=1.e-8,
                      pressure_btm=1.e2,
                      nlayer=100,
                      nu_grid=nu_grid)
    art.change_temperature_range(400.0, 1500.0)

    mdb = mock_mdb(db)

    with pytest.raises(ValueError):
        opa = OpaPremodit(mdb=mdb,
                          nu_grid=nu_grid,
                          diffmode=diffmode,
                          auto_trange=[art.Tlow, art.Thigh],
                          broadening_resolution={
                              "mode": "single",
                              "value": None
                          })


if __name__ == "__main__":
    test_check_raise_valueerror_when_32bit()
    test_check_raise_error_premodit_when_32bit()
import pytest
from exojax.utils.memuse import premodit_devmemory_use
from exojax.utils.memuse import device_memory_use
from exojax.test.emulate_mdb import mock_mdb
from exojax.opacity import OpaPremodit
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.rt import ArtEmisPure
from jax import config


def test_memuse_premodit():
    ngrid_nu_grid = 70000
    ngrid_broadpar = 10
    nlayer = 200
    nfree = 10
    ngrid_elower = 10
    mem, case = premodit_devmemory_use(ngrid_nu_grid,
                                       ngrid_broadpar,
                                       ngrid_elower,
                                       nlayer=nlayer,
                                       nfree=nfree,
                                       precision="FP64")
    assert mem == 44800000000


def test_device_memory_use_premodit_art_opa():
    config.update("jax_enable_x64", True)
    db = "exomol"
    diffmode = 0
    nlayer = 100
    nu_grid, wav, res = mock_wavenumber_grid()
    art = ArtEmisPure(pressure_top=1.e-8,
                      pressure_btm=1.e2,
                      nlayer=nlayer,
                      nu_grid=nu_grid)
    art.change_temperature_range(400.0, 1500.0)

    mdb = mock_mdb(db)
    opa = OpaPremodit(mdb=mdb,
                      nu_grid=nu_grid,
                      diffmode=diffmode,
                      auto_trange=[art.Tlow, art.Thigh],
                      broadening_resolution={
                          "mode": "manual",
                          "value": 0.2
                      })
    nfree = 10
    nbroad_ref = 6
    nfp64 = 8
    nelower_ref = 283
    # CASE 0
    memuse = device_memory_use(opa, art=art, nfree=nfree)
    assert memuse == len(nu_grid) * nbroad_ref * nlayer * nfree * nfp64 * 4
    # CASE 1
    memuse = device_memory_use(opa)
    assert memuse == len(nu_grid) * nbroad_ref * nelower_ref * nfp64 * 2


if __name__ == "__main__":
    test_memuse_premodit()
    test_device_memory_use_premodit_art_opa()
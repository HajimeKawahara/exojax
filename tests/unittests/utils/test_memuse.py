from types import SimpleNamespace

from exojax.utils.memuse import premodit_devmemory_use
from exojax.utils.memuse import device_memory_use


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
    nlayer = 100
    nu_grid = range(20)
    nfree = 10
    nbroad_ref = 6
    nfp64 = 8
    nelower_ref = 283
    art = SimpleNamespace(nlayer=nlayer)
    opa = SimpleNamespace(
        method="premodit",
        nu_grid=nu_grid,
        ngrid_broadpar=nbroad_ref,
        ngrid_elower=nelower_ref,
    )
    # CASE 0
    memuse = device_memory_use(opa, art=art, nfree=nfree)
    assert memuse == len(nu_grid) * nbroad_ref * nlayer * nfree * nfp64 * 4
    # CASE 1
    memuse = device_memory_use(opa)
    assert memuse == len(nu_grid) * nbroad_ref * nelower_ref * nfp64 * 2

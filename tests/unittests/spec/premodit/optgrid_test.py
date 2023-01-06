from exojax.spec.optgrid import optelower


def test_optelower():
    from exojax.utils.grids import wavenumber_grid
    from exojax.test.emulate_mdb import mock_mdbExomol
    Nx = 15000
    nu_grid, wav, res = wavenumber_grid(22800.0,
                                        23100.0,
                                        Nx,
                                        unit='AA',
                                        xsmode="premodit")
    Tmax = 1000.0  #K
    Pmin = 0.1  #bar
    mdb = mock_mdbExomol()
    Eopt = optelower(mdb, nu_grid, Tmax, Pmin)
    assert Eopt == 7500.0


if __name__ == "__main__":
    test_optelower()
from exojax.spec.optgrid import optelower


def test_optelower():
    from exojax.utils.grids import wavenumber_grid
    from exojax.test.emulate_mdb import mock_mdbExomol
    Nx = 20000
    nu_grid, wav, reso = wavenumber_grid(22800.0,
                                     24000.0,
                                     Nx,
                                     unit='AA',
                                     xsmode="modit")
    Tmax = 2500.0  #K
    Pmin = 0.1  #bar
    mdb = mock_mdbExomol()
    Eopt = optelower(mdb, nu_grid, Tmax, Pmin)
    assert Eopt == 24200.0


if __name__ == "__main__":
    test_optelower()
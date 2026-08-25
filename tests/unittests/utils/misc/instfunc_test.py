import numpy as np
from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.utils.instfunc import resolution_eslin
from exojax.utils.instfunc import resolution_eslog
from exojax.utils.instfunc import nx_even_from_resolution_eslog
import pytest

from exojax.utils.grids import wavenumber_grid





def test_nx_from_resolution_eslog():
    nu0 = 4000.0
    nu1 = 4500.0
    resolution = 849010.2113833647
    Nx = nx_even_from_resolution_eslog(nu0, nu1, resolution)

    assert Nx == 100000
    assert (
        nx_even_from_resolution_eslog(
            nu0, nu1, resolution, definition="log"
        )
        == Nx
    )


def test_nx_from_pointwise_resolution_eslog():
    nu0 = 1.0
    nu1 = 2.0
    requested_resolution = 10.0

    Nx = nx_even_from_resolution_eslog(
        nu0,
        nu1,
        requested_resolution,
        definition="pointwise",
    )
    nus = np.logspace(np.log10(nu0), np.log10(nu1), Nx)
    previous_even_nus = np.logspace(
        np.log10(nu0), np.log10(nu1), Nx - 2
    )

    assert Nx == 10
    assert (
        resolution_eslog(nus, definition="pointwise")
        >= requested_resolution
    )
    assert (
        resolution_eslog(previous_even_nus, definition="pointwise")
        < requested_resolution
    )


def test_resolution_to_gaussian_std():
    resolution = 10**5
    beta = resolution_to_gaussian_std(resolution)
    assert beta == pytest.approx(1.2731013507066515)


def test_resolution_eslin():
    nus = np.linspace(1000, 2000, 1000)
    ref = (999.0000000000146, 1500.0, 1998.000000000029)
    assert np.all(resolution_eslin(nus) == pytest.approx(ref))


def test_resolution_eslog():
    nu0 = 4000.0
    nu1 = 4500.0
    Nx = 100000
    nus = np.logspace(np.log10(nu0), np.log10(nu1), Nx)
    assert resolution_eslog(nus) == pytest.approx(849010.2113833647)
    assert resolution_eslog(nus, definition="log") == pytest.approx(
        849010.2113833647
    )
    assert resolution_eslog(nus, definition="pointwise") == pytest.approx(
        np.min(nus[:-1] / np.diff(nus))
    )


@pytest.mark.parametrize(
    "function, arguments",
    [
        (resolution_eslog, (np.array([1.0, 2.0]),)),
        (nx_even_from_resolution_eslog, (1.0, 2.0, 100.0)),
    ],
)
def test_unknown_eslog_resolution_definition(function, arguments):
    with pytest.raises(ValueError, match="definition"):
        function(*arguments, definition="unknown")


if __name__ == "__main__":
    test_resolution_to_gaussian_std()
    test_resolution_eslin()
    test_resolution_eslog()
    test_nx_from_resolution_eslog()
    test_nx_from_pointwise_resolution_eslog()

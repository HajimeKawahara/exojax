import numpy as np

import pytest

from exojax.postproc.ckd import sample_ckd_bands_at_wavelengths
from exojax.postproc.ckd import validate_ckd_band_coverage
from exojax.postproc.ckd import validate_ckd_sampling_inputs
from exojax.postproc.ckd import wavenumber_range_with_radial_velocity
from exojax.utils.constants import c


def test_sample_ckd_bands_at_exact_wavelengths():
    nu_bands = np.array([1000.0, 2000.0, 3000.0])
    spectrum_bands = np.array([1.0, 2.0, 3.0])
    wavelength_nm = np.array([1.0e7 / 1000.0, 1.0e7 / 2000.0, 1.0e7 / 3000.0])

    sampled = sample_ckd_bands_at_wavelengths(
        nu_bands, spectrum_bands, wavelength_nm, unit="nm"
    )

    np.testing.assert_allclose(np.asarray(sampled), spectrum_bands)


def test_sample_ckd_bands_preserves_wavelength_order():
    nu_bands = np.array([1000.0, 2000.0, 3000.0])
    spectrum_bands = np.array([1.0, 2.0, 3.0])
    wavelength_nm = np.array([1.0e7 / 3000.0, 1.0e7 / 2000.0, 1.0e7 / 1000.0])

    sampled = sample_ckd_bands_at_wavelengths(
        nu_bands, spectrum_bands, wavelength_nm, unit="nm"
    )

    np.testing.assert_allclose(np.asarray(sampled), [3.0, 2.0, 1.0])


def test_sample_ckd_bands_accepts_descending_band_order():
    nu_bands = np.array([3000.0, 2000.0, 1000.0])
    spectrum_bands = np.array([3.0, 2.0, 1.0])
    wavelength_nm = np.array([1.0e7 / 1000.0, 1.0e7 / 3000.0])

    sampled = sample_ckd_bands_at_wavelengths(
        nu_bands, spectrum_bands, wavelength_nm, unit="nm"
    )

    np.testing.assert_allclose(np.asarray(sampled), [1.0, 3.0])


def test_sample_ckd_bands_uses_response_sampling_rv_convention():
    nu_bands = np.linspace(1000.0, 3000.0, 5)
    spectrum_bands = nu_bands.copy()
    wavelength_nm = np.array([1.0e7 / 2000.0])
    radial_velocity = 30.0

    sampled = sample_ckd_bands_at_wavelengths(
        nu_bands, spectrum_bands, wavelength_nm, radial_velocity=radial_velocity
    )

    expected = 2000.0 * (1.0 + radial_velocity / c)
    np.testing.assert_allclose(np.asarray(sampled), [expected], rtol=1.0e-6)


def test_sample_ckd_bands_supports_micron_wavelengths():
    nu_bands = np.array([1000.0, 2000.0, 3000.0])
    spectrum_bands = np.array([1.0, 2.0, 3.0])
    wavelength_um = np.array([1.0e4 / 2000.0])

    sampled = sample_ckd_bands_at_wavelengths(
        nu_bands, spectrum_bands, wavelength_um, unit="um"
    )

    np.testing.assert_allclose(np.asarray(sampled), [2.0])


@pytest.mark.parametrize(
    "nu_bands, spectrum_bands, wavelength, message",
    [
        ([[1000.0, 2000.0]], [1.0, 2.0], [5000.0], "nu_bands"),
        ([1000.0, 2000.0], [[1.0, 2.0]], [5000.0], "spectrum_bands"),
        ([1000.0, 2000.0], [1.0, 2.0], [[5000.0]], "wavelength"),
        ([], [], [5000.0], "at least one"),
        ([1000.0, 2000.0], [1.0], [5000.0], "length"),
        ([1000.0, 2000.0], [1.0, 2.0], [], "wavelength"),
        ([1000.0, np.nan], [1.0, 2.0], [5000.0], "finite"),
        ([1000.0, 2000.0], [1.0, np.nan], [5000.0], "finite"),
        ([1000.0, 2000.0], [1.0, 2.0], [np.nan], "finite"),
        ([0.0, 2000.0], [1.0, 2.0], [5000.0], "positive"),
        ([1000.0, 2000.0], [1.0, 2.0], [0.0], "positive"),
        ([1000.0, 1000.0], [1.0, 2.0], [5000.0], "unique"),
    ],
)
def test_validate_ckd_sampling_inputs_rejects_invalid_static_grids(
    nu_bands, spectrum_bands, wavelength, message
):
    with pytest.raises(ValueError, match=message):
        validate_ckd_sampling_inputs(nu_bands, spectrum_bands, wavelength)


def test_wavenumber_range_with_radial_velocity_expands_range():
    nu_values = np.array([1000.0, 2000.0])
    nu_min, nu_max = wavenumber_range_with_radial_velocity(
        nu_values, radial_velocity_min=-200.0, radial_velocity_max=100.0
    )

    expected_min = 1000.0 * (1.0 - 200.0 / c)
    expected_max = 2000.0 * (1.0 + 100.0 / c)
    np.testing.assert_allclose([nu_min, nu_max], [expected_min, expected_max])


@pytest.mark.parametrize(
    "nu_values, rv_min, rv_max, message",
    [
        ([[1000.0, 2000.0]], 0.0, 0.0, "one-dimensional"),
        ([], 0.0, 0.0, "at least one"),
        ([1000.0, np.nan], 0.0, 0.0, "finite"),
        ([0.0, 1000.0], 0.0, 0.0, "positive"),
        ([1000.0], 0.0, np.nan, "finite"),
        ([1000.0], -c, 0.0, "positive"),
    ],
)
def test_wavenumber_range_with_radial_velocity_validates_inputs(
    nu_values, rv_min, rv_max, message
):
    with pytest.raises(ValueError, match=message):
        wavenumber_range_with_radial_velocity(nu_values, rv_min, rv_max)


def test_validate_ckd_band_coverage_accepts_covering_centers():
    validate_ckd_band_coverage(
        nu_bands=np.array([900.0, 1000.0, 1100.0, 1200.0]),
        nu_range=(950.0, 1150.0),
    )


def test_validate_ckd_band_coverage_accepts_covering_edges():
    validate_ckd_band_coverage(
        nu_bands=np.array([1000.0, 1100.0]),
        nu_range=(950.0, 1150.0),
        band_edges=np.array([[950.0, 1050.0], [1050.0, 1150.0]]),
    )


def test_validate_ckd_band_coverage_accepts_overlapping_edges():
    validate_ckd_band_coverage(
        nu_bands=np.array([1000.0, 1100.0]),
        nu_range=(975.0, 1125.0),
        band_edges=np.array([[950.0, 1060.0], [1040.0, 1150.0]]),
    )


@pytest.mark.parametrize(
    "nu_bands, nu_range, message",
    [
        ([[900.0, 1000.0]], (950.0, 1150.0), "one-dimensional"),
        ([], (950.0, 1150.0), "at least one"),
        ([900.0, np.nan], (950.0, 1150.0), "finite"),
        ([900.0, 1000.0], (950.0, np.nan), "finite"),
        ([0.0, 1000.0], (950.0, 1150.0), "positive"),
        ([900.0, 1000.0], (0.0, 1150.0), "positive"),
        ([900.0, 1000.0], (950.0,), "two-element"),
        ([1000.0, 1100.0], (950.0, 1150.0), "do not cover"),
    ],
)
def test_validate_ckd_band_coverage_rejects_invalid_or_uncovered_ranges(
    nu_bands, nu_range, message
):
    with pytest.raises(ValueError, match=message):
        validate_ckd_band_coverage(nu_bands, nu_range)


@pytest.mark.parametrize(
    "band_edges, message",
    [
        ([[950.0, 1050.0]], "shape"),
        ([[950.0, np.nan], [1050.0, 1150.0]], "finite"),
        ([[0.0, 1050.0], [1050.0, 1150.0]], "positive"),
        ([[950.0, 950.0], [1050.0, 1150.0]], "positive widths"),
        ([[960.0, 1050.0], [1050.0, 1150.0]], "do not cover"),
        ([[950.0, 1040.0], [1060.0, 1150.0]], "continuously cover"),
    ],
)
def test_validate_ckd_band_coverage_rejects_invalid_or_uncovered_edges(
    band_edges, message
):
    with pytest.raises(ValueError, match=message):
        validate_ckd_band_coverage(
            nu_bands=np.array([1000.0, 1100.0]),
            nu_range=(950.0, 1150.0),
            band_edges=np.array(band_edges),
        )

import copy
import pickle

import jax
import numpy as np
import pytest

from exojax.rt.common import ArtCommon
from exojax.rt.reflect import ArtAbsPure, OpartReflectPure
from exojax.rt.trans import ArtTransPure


def test_legacy_pressure_arguments_remain_representative_values():
    art = ArtCommon(
        pressure_top=1.0e-2,
        pressure_btm=1.0,
        nlayer=3,
        warn_no_nu_grid=False,
    )

    np.testing.assert_allclose(art.pressure, [1.0e-2, 1.0e-1, 1.0])
    np.testing.assert_allclose(
        art.pressure_boundary,
        [
            1.0e-2 / np.sqrt(10.0),
            1.0e-2 * np.sqrt(10.0),
            1.0e-1 * np.sqrt(10.0),
            np.sqrt(10.0),
        ],
    )
    np.testing.assert_allclose(art.dParr, np.diff(art.pressure_boundary))
    assert art.pressure_top_boundary == art.pressure_boundary[0]
    assert art.pressure_btm_boundary == art.pressure_boundary[-1]


def test_gravity_profile_obeys_inverse_square_law(monkeypatch):
    art = ArtCommon(
        pressure_top=1.0e-2,
        pressure_btm=1.0,
        nlayer=2,
        warn_no_nu_grid=False,
    )
    normalized_height = np.array([0.8, 0.2])
    normalized_radius_lower = np.array([1.2, 1.0])
    monkeypatch.setattr(
        art,
        "atmosphere_height",
        lambda *_: (normalized_height, normalized_radius_lower),
    )

    gravity_btm = 100.0
    gravity = art.gravity_profile(
        temperature=np.ones(2),
        mean_molecular_weight=np.ones(2),
        radius_btm=1.0,
        gravity_btm=gravity_btm,
    )

    normalized_radius_layer = normalized_radius_lower + 0.5 * normalized_height
    expected = gravity_btm / normalized_radius_layer**2
    assert gravity.shape == (2, 1)
    np.testing.assert_allclose(np.asarray(gravity[:, 0]), expected)


def test_from_pressure_boundaries_defines_canonical_grid():
    pressure_top_boundary = 3.7e-8
    pressure_btm_boundary = 987.654321
    art = ArtCommon.from_pressure_boundaries(
        pressure_top_boundary=pressure_top_boundary,
        pressure_btm_boundary=pressure_btm_boundary,
        nlayer=4,
        warn_no_nu_grid=False,
    )

    assert art.pressure_top_boundary == pressure_top_boundary
    assert art.pressure_btm_boundary == pressure_btm_boundary
    np.testing.assert_allclose(art.dParr, np.diff(art.pressure_boundary))
    np.testing.assert_allclose(
        art.pressure,
        np.sqrt(art.pressure_boundary[:-1] * art.pressure_boundary[1:]),
    )
    np.testing.assert_allclose(
        art.pressure_decrease_rate,
        art.pressure_boundary[0] / art.pressure_boundary[1],
    )
    assert art.pressure_top == art.pressure[0]
    assert art.pressure_btm == art.pressure[-1]


def test_exact_factory_reproduces_an_equivalent_legacy_grid():
    legacy = ArtCommon(
        pressure_top=1.0e-4,
        pressure_btm=10.0,
        nlayer=6,
        warn_no_nu_grid=False,
    )
    exact = ArtCommon.from_pressure_boundaries(
        pressure_top_boundary=legacy.pressure_top_boundary,
        pressure_btm_boundary=legacy.pressure_btm_boundary,
        nlayer=legacy.nlayer,
        warn_no_nu_grid=False,
    )

    np.testing.assert_allclose(exact.pressure_boundary, legacy.pressure_boundary)
    np.testing.assert_allclose(exact.pressure, legacy.pressure)
    np.testing.assert_allclose(exact.dParr, legacy.dParr)
    np.testing.assert_allclose(
        exact.pressure_decrease_rate, legacy.pressure_decrease_rate
    )


def test_from_pressure_boundaries_supports_one_layer():
    art = ArtCommon.from_pressure_boundaries(
        pressure_top_boundary=1.0e-4,
        pressure_btm_boundary=1.0,
        nlayer=1,
        warn_no_nu_grid=False,
    )

    np.testing.assert_allclose(art.pressure_boundary, [1.0e-4, 1.0])
    np.testing.assert_allclose(art.pressure, [1.0e-2])
    np.testing.assert_allclose(art.dParr, [1.0 - 1.0e-4])


def test_from_pressure_boundaries_rejects_grid_unstable_in_active_jax_dtype():
    with jax.experimental.disable_x64():
        with pytest.raises(ValueError, match="active JAX dtype"):
            ArtCommon.from_pressure_boundaries(
                pressure_top_boundary=1.0e-40,
                pressure_btm_boundary=1.0,
                nlayer=1,
                warn_no_nu_grid=False,
            )

    with jax.experimental.enable_x64():
        art = ArtCommon.from_pressure_boundaries(
            pressure_top_boundary=1.0e-40,
            pressure_btm_boundary=1.0,
            nlayer=1,
            warn_no_nu_grid=False,
        )
        np.testing.assert_array_equal(
            art.pressure_boundary[[0, -1]], [1.0e-40, 1.0]
        )


@pytest.mark.parametrize("nlayer", [1, 4])
def test_from_pressure_boundaries_survives_pressure_profile_reinitialization(
    nlayer,
):
    art = ArtCommon.from_pressure_boundaries(
        pressure_top_boundary=3.7e-8,
        pressure_btm_boundary=987.654321,
        nlayer=nlayer,
        warn_no_nu_grid=False,
    )
    expected_boundary = art.pressure_boundary.copy()

    art.init_pressure_profile()

    np.testing.assert_array_equal(art.pressure_boundary, expected_boundary)
    np.testing.assert_allclose(art.dParr, np.diff(expected_boundary))
    assert np.all(np.isfinite(art.pressure))


def test_exact_pressure_grid_survives_copy_and_pickle():
    art = ArtCommon.from_pressure_boundaries(
        pressure_top_boundary=3.7e-8,
        pressure_btm_boundary=987.654321,
        nlayer=4,
        warn_no_nu_grid=False,
    )

    copies = [copy.copy(art), copy.deepcopy(art), pickle.loads(pickle.dumps(art))]
    for copied_art in copies:
        copied_art.init_pressure_profile()
        np.testing.assert_array_equal(
            copied_art.pressure_boundary[[0, -1]], [3.7e-8, 987.654321]
        )
        np.testing.assert_allclose(
            copied_art.dParr, np.diff(copied_art.pressure_boundary)
        )


def test_from_pressure_boundaries_passes_constructor_arguments_to_new():
    class ArtWithCustomNew(ArtCommon):
        new_arguments = None

        def __new__(cls, pressure_top, pressure_btm, nlayer, token):
            cls.new_arguments = (pressure_top, pressure_btm, nlayer, token)
            return super().__new__(cls)

        def __init__(self, pressure_top, pressure_btm, nlayer, token):
            self.token = token
            super().__init__(
                pressure_top,
                pressure_btm,
                nlayer,
                warn_no_nu_grid=False,
            )

    art = ArtWithCustomNew.from_pressure_boundaries(
        pressure_top_boundary=1.0e-4,
        pressure_btm_boundary=1.0,
        nlayer=1,
        token="sentinel",
    )

    assert ArtWithCustomNew.new_arguments == (0.01, 0.01, 1, "sentinel")
    assert art.token == "sentinel"
    np.testing.assert_array_equal(art.pressure_boundary, [1.0e-4, 1.0])


def test_from_pressure_boundaries_cleans_state_after_constructor_failure():
    class FailingArt(ArtCommon):
        allocated_instance = None

        def __new__(cls, **kwargs):
            cls.allocated_instance = super().__new__(cls)
            return cls.allocated_instance

        def __init__(self, **kwargs):
            raise RuntimeError("constructor failed")

    with pytest.raises(RuntimeError, match="constructor failed"):
        FailingArt.from_pressure_boundaries(
            pressure_top_boundary=1.0e-4,
            pressure_btm_boundary=1.0,
            nlayer=1,
        )

    assert not hasattr(
        FailingArt.allocated_instance, "_pressure_boundary_specification"
    )


def test_one_layer_grid_supports_geometry_and_opacity():
    art = ArtTransPure.from_pressure_boundaries(
        pressure_top_boundary=1.0e-4,
        pressure_btm_boundary=1.0,
        nlayer=1,
        warn_no_nu_grid=False,
    )
    temperature = np.array([1000.0])
    mean_molecular_weight = np.array([2.3])
    radius_btm = 7.0e9
    gravity_btm = 2.5e3

    height, radius_lower = art.atmosphere_height(
        temperature,
        mean_molecular_weight,
        radius_btm,
        gravity_btm,
    )
    gravity = art.gravity_profile(
        temperature,
        mean_molecular_weight,
        radius_btm,
        gravity_btm,
    )
    dtau = art.opacity_profile_xs(
        xs=np.ones((1, 2)) * 1.0e-25,
        mixing_ratio=np.array([1.0e-3]),
        molmass=2.3,
        gravity=gravity,
    )

    assert height.shape == (1,)
    assert radius_lower.shape == (1,)
    assert gravity.shape == (1, 1)
    assert dtau.shape == (1, 2)
    assert np.all(np.isfinite(np.asarray(height)))
    assert np.all(np.isfinite(np.asarray(dtau)))


def test_pressure_boundary_grids_join_exactly():
    junction_pressure = 3.7
    atmosphere = ArtCommon.from_pressure_boundaries(
        pressure_top_boundary=1.0e-6,
        pressure_btm_boundary=junction_pressure,
        nlayer=4,
        warn_no_nu_grid=False,
    )
    lower_model = ArtCommon.from_pressure_boundaries(
        pressure_top_boundary=junction_pressure,
        pressure_btm_boundary=1.0e3,
        nlayer=3,
        warn_no_nu_grid=False,
    )

    assert atmosphere.pressure_btm_boundary == junction_pressure
    assert lower_model.pressure_top_boundary == junction_pressure
    assert atmosphere.pressure_boundary[-1] == lower_model.pressure_boundary[0]


@pytest.mark.parametrize(
    "pressure_top_boundary, pressure_btm_boundary",
    [
        (0.0, 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
        (np.nan, 1.0),
        (np.array([1.0e-4]), 1.0),
        (1.0e-4, np.array([1.0])),
    ],
)
def test_from_pressure_boundaries_rejects_invalid_bounds(
    pressure_top_boundary, pressure_btm_boundary
):
    with pytest.raises(ValueError):
        ArtCommon.from_pressure_boundaries(
            pressure_top_boundary=pressure_top_boundary,
            pressure_btm_boundary=pressure_btm_boundary,
            nlayer=2,
            warn_no_nu_grid=False,
        )


@pytest.mark.parametrize("legacy_keyword", ["pressure_top", "pressure_btm"])
def test_from_pressure_boundaries_rejects_legacy_pressure_kwargs(legacy_keyword):
    with pytest.raises(TypeError, match="cannot be used"):
        ArtCommon.from_pressure_boundaries(
            pressure_top_boundary=1.0e-4,
            pressure_btm_boundary=1.0,
            nlayer=2,
            warn_no_nu_grid=False,
            **{legacy_keyword: 0.1},
        )


def test_from_pressure_boundaries_is_inherited_by_art_subclasses():
    trans = ArtTransPure.from_pressure_boundaries(
        pressure_top_boundary=1.0e-6,
        pressure_btm_boundary=10.0,
        nlayer=3,
        integration="trapezoid",
        warn_no_nu_grid=False,
    )
    absorption = ArtAbsPure.from_pressure_boundaries(
        pressure_top_boundary=1.0e-6,
        pressure_btm_boundary=10.0,
        nlayer=3,
        nu_grid=np.array([1000.0]),
    )

    assert trans.integration == "trapezoid"
    np.testing.assert_allclose(trans.pressure_boundary[[0, -1]], [1.0e-6, 10.0])
    np.testing.assert_allclose(
        absorption.pressure_boundary[[0, -1]], [1.0e-6, 10.0]
    )


def test_from_pressure_boundaries_accepts_required_opart_keyword():
    class FakeOpacityLayer:
        nu_grid = np.array([1000.0])

    opalayer = FakeOpacityLayer()
    opart = OpartReflectPure.from_pressure_boundaries(
        pressure_top_boundary=1.0e-5,
        pressure_btm_boundary=1.0,
        nlayer=2,
        opalayer=opalayer,
    )

    assert opart.opalayer is opalayer
    np.testing.assert_allclose(opart.pressure_boundary[[0, -1]], [1.0e-5, 1.0])

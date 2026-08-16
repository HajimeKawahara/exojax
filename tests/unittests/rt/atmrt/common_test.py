import numpy as np

from exojax.rt.common import ArtCommon


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

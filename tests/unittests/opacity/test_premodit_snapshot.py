import numpy as np

from exojax.database.contracts import Lines, MDBMeta, MDBSnapshot
from exojax.opacity.premodit.api import _MDBLikeFromSnapshot


def test_snapshot_restores_scalar_isotope():
    meta = MDBMeta(
        dbtype="hitran",
        molmass=16.0,
        T_gQT=np.array([296.0, 400.0]),
        gQT=np.array([[1.0, 1.0], [1.1, 1.1]]),
    )
    lines = Lines(
        nu_lines=np.array([1000.0]),
        elower=np.array([0.0]),
        line_strength_ref_original=np.array([1.0]),
    )
    snapshot = MDBSnapshot(
        meta=meta,
        lines=lines,
        isotope=np.array([1, 1, 1]),
        uniqiso=np.array([1]),
    )

    adapter = _MDBLikeFromSnapshot.from_snapshot(snapshot)

    assert adapter.isotope == 1
    assert adapter.uniqiso.tolist() == [1]


def test_snapshot_defaults_all_isotopes():
    meta = MDBMeta(
        dbtype="hitran",
        molmass=16.0,
        T_gQT=np.array([296.0]),
        gQT=np.array([[1.0]]),
    )
    lines = Lines(
        nu_lines=np.array([1000.0, 1001.0]),
        elower=np.array([0.0, 1.0]),
        line_strength_ref_original=np.array([1.0, 1.0]),
    )
    snapshot = MDBSnapshot(
        meta=meta,
        lines=lines,
        isotope=np.array([1, 2]),
        uniqiso=np.array([1, 2]),
    )

    adapter = _MDBLikeFromSnapshot.from_snapshot(snapshot)

    assert adapter.isotope == 0


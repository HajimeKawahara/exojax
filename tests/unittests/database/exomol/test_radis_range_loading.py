"""Keep ExoMol range selection in the cache reader without changing line data."""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from exojax.database.exomol.api import MdbExomol
from exojax.test.data import get_testdata_filename


@pytest.fixture
def local_co(tmp_path, monkeypatch):
    """Copy bundled sources; any network access is a test failure."""
    import requests

    def unexpected_download(*args, **kwargs):
        pytest.fail("The bundled ExoMol fixture attempted a download")

    monkeypatch.setattr(MdbExomol, "download", unexpected_download)
    monkeypatch.setattr(MdbExomol, "is_registered", lambda self: False)
    monkeypatch.setattr(requests.sessions.Session, "request", unexpected_download)
    source = Path(get_testdata_filename("CO")) / "12C-16O" / "SAMPLE"
    target = tmp_path / "CO" / "12C-16O" / "SAMPLE"
    target.mkdir(parents=True)
    for path in source.iterdir():
        if path.name.endswith((".bz2", ".def", ".pf", ".broad")):
            shutil.copy2(path, target / path.name)
    return target


def load(path, nurange, *, engine="pytables", **kwargs):
    return MdbExomol(
        path, nurange=nurange, engine=engine, gpu_transfer=False,
        broadf_download=False, local_databases=str(path.parents[2]), **kwargs,
    )


@pytest.mark.parametrize("bounds", ["finite", "inactive"])
def test_cache_predicate_preserves_full_load_result(local_co, monkeypatch, bounds):
    from radis.api.hdf5 import DataFileManager

    full = load(local_co, [1., 1.e9], activation=False)
    all_rows = full.df.copy()
    centers = np.unique(all_rows.nu_lines)
    lower, upper = centers[len(centers) // 4], centers[3 * len(centers) // 4]
    requested = {
        "finite": [lower, upper], "lower": [lower, np.inf],
        "upper": [-np.inf, upper], "unbounded": [-np.inf, np.inf],
        "inactive": None,
    }[bounds]
    reads = []
    original_read = DataFileManager.read

    def tracked_read(self, filename, *args, **kwargs):
        result = original_read(self, filename, *args, **kwargs)
        if str(filename).endswith(".trans.h5"):
            reads.append((kwargs.get("where"), len(result)))
        return result

    monkeypatch.setattr(DataFileManager, "read", tracked_read)
    selected = load(local_co, requested, activation=False)
    assert reads
    if requested is None:
        assert not selected.activation
        pd.testing.assert_frame_equal(selected.df, all_rows)
        assert not any(where for where, _ in reads)
        return
    mask = (all_rows.nu_lines > requested[0]) & (all_rows.nu_lines < requested[1])
    pd.testing.assert_frame_equal(selected.df, all_rows[mask])
    assert sum(count for _, count in reads) == int(mask.sum())
    expected_terms = sum(np.isfinite(requested))
    assert all(len(where or []) == expected_terms for where, _ in reads)
    full.nurange = requested
    full.df_load_mask = full.compute_load_mask(full.df)
    full.activate(full.df)
    selected.activate(selected.df)
    for field in (
        "nu_lines", "A", "elower", "gpp", "jlower", "jupper", "alpha_ref",
        "n_Texp", "line_strength_ref_original", "logsij0", "gamma_natural",
    ):
        np.testing.assert_array_equal(getattr(selected, field), getattr(full, field))
    if bounds == "finite":
        assert not np.any(selected.nu_lines == lower)
        assert not np.any(selected.nu_lines == upper)


@pytest.mark.parametrize("engine", ["vaex", None])
@pytest.mark.parametrize("empty", [False, True])
def test_vaex_activation_with_empty_filtered_chunks(local_co, monkeypatch, engine, empty):
    """RADIS extract() must not leave empty chunks for activation (#827)."""
    vaex = pytest.importorskip("vaex")
    full = load(local_co, [1., 1.e9], activation=False)
    centers = np.sort(full.df.nu_lines.to_numpy())
    # Select exactly one two-row chunk, leaving the rest outside the range.
    requested = [centers[0], centers[3]]
    if empty:
        requested = [centers[-1] + 1., centers[-1] + 2.]
    monkeypatch.setattr(vaex.dataframe.get_main_executor(), "chunk_size", 2)

    if empty:
        with pytest.raises(ValueError, match="No line found"):
            load(local_co, requested, engine=engine)
        return

    reference = load(local_co, requested)
    selected = load(local_co, requested, engine=engine)
    assert selected.engine == "vaex"
    assert len(selected.nu_lines) == 2
    for field in (
        "nu_lines", "A", "elower", "gpp", "jlower", "jupper", "alpha_ref",
        "n_Texp", "line_strength_ref_original", "logsij0", "gamma_natural",
    ):
        np.testing.assert_array_equal(getattr(selected, field), getattr(reference, field))


@pytest.mark.parametrize("lower,upper,filter_wavenumbers,lengths", [
    (1000., 1003., True, [1, 1]),
    (1000., None, True, [1, 3]),
    (None, 1003., True, [3, 1]),
    (None, None, True, [3, 3]),
    (-np.inf, np.inf, True, [3, 3]),
    (0., 0., False, [3, 3]),
])
def test_pushdown_precedes_multiple_cache_concatenation(tmp_path, monkeypatch, lower, upper, filter_wavenumbers, lengths):
    """Each cache reader must materialize only its matching subset."""
    from radis.api.hdf5 import DataFileManager
    import exojax.database.exomol.api as api

    cache_manager = DataFileManager("pytables")
    paths = [tmp_path / "first.trans.bz2", tmp_path / "second.trans.bz2"]
    frames = [pd.DataFrame({"nu_lines": [999., 1000., 1001.], "A": [1., 2., 3.]}),
              pd.DataFrame({"nu_lines": [1002., 1003., 1004.], "A": [4., 5., 6.]})]
    for path, frame in zip(paths, frames):
        cache_manager.write(cache_manager.cache_file(path), frame)
    monkeypatch.setattr(api, "supports_exomol_broadf_download", lambda: True)
    monkeypatch.setattr(api, "init_exomol_manager", lambda *args, **kwargs: None)
    mdb = MdbExomol.__new__(MdbExomol)
    mdb.path, mdb.exact_molecule_name = tmp_path, "12C-16O"
    mdb.crit, mdb.broadf, mdb.broadf_download = 0., False, False
    mdb.skip_optional_data, mdb.bkgdatm, mdb.trans_file = True, "H2", paths
    loaded_lengths = []
    original_read = DataFileManager.read

    def tracked_read(self, filename, *args, **kwargs):
        result = original_read(self, filename, *args, **kwargs)
        loaded_lengths.append(len(result))
        return result

    monkeypatch.setattr(DataFileManager, "read", tracked_read)
    selected = mdb._load_radis(lower, upper, str(tmp_path), "pytables", filter_wavenumbers=filter_wavenumbers)
    assert loaded_lengths == lengths
    reference = pd.concat(frames)
    if filter_wavenumbers:
        if lower is not None:
            reference = reference[reference.nu_lines > lower]
        if upper is not None:
            reference = reference[reference.nu_lines < upper]
    pd.testing.assert_frame_equal(selected, reference)

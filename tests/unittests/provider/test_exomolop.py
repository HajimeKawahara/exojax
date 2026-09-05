import h5py
import numpy as np
import pytest
import requests

from exojax.provider import exomolop
from exojax.provider import url as provider_url
from exojax.provider.exomolop import download_exomolop_h5, load_ckd


class _FakeResponse:
    def __init__(self, *, status_code=200, text="", chunks=(), error=None):
        self.status_code = status_code
        self.text = text
        self._chunks = chunks
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def iter_content(self, chunk_size):
        del chunk_size
        yield from self._chunks


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._responses.pop(0)


def _expected_table_path(root):
    return (
        root
        / "H2O"
        / "1H2-16O"
        / "POKAZATEL"
        / "1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    )


def _legacy_dot_table_path(root):
    return (
        root
        / "H2O"
        / "1H2-16O"
        / "POKAZATEL"
        / "1H2-16O__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    )


def test_petitradtrans_ktable_filenames_defaults_and_override():
    filenames = provider_url.petitRADTRANS_ktable_filenames("1H2-16O", "POKAZATEL")

    assert filenames == [
        "1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.petitRADTRANS.h5",
        "1H2-16O__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5",
        "1H2-16O__POKAZATEL.R1000_0.1-250mu.ktable.petitRADTRANS.h5",
    ]
    assert provider_url.petitRADTRANS_ktable_filenames(
        "1H2-16O", "POKAZATEL", extension=".custom.h5"
    ) == ["1H2-16O__POKAZATEL.custom.h5"]


def test_load_ckd_returns_api_axis_order_and_replaces_zeros(tmp_path):
    table_path = tmp_path / "synthetic.h5"
    pressures = np.array([0.1, 1.0])
    temperatures = np.array([500.0, 1000.0, 1500.0])
    bin_centers = np.array([1000.0, 1100.0])
    samples = np.array([0.25, 0.75])
    weights = np.array([0.5, 0.5])
    kcoeff = np.arange(
        pressures.size * temperatures.size * bin_centers.size * samples.size,
        dtype=float,
    ).reshape(pressures.size, temperatures.size, bin_centers.size, samples.size)

    with h5py.File(table_path, "w") as handle:
        handle.create_dataset("mol_name", data=np.array([b"H2O"]))
        handle.create_dataset("mol_mass", data=np.array([18.0]))
        handle.create_dataset("bin_centers", data=bin_centers)
        handle.create_dataset("samples", data=samples)
        handle.create_dataset("weights", data=weights)
        handle.create_dataset("t", data=temperatures)
        handle.create_dataset("p", data=pressures)
        handle.create_dataset("kcoeff", data=kcoeff)

    (
        xsgrid,
        loaded_samples,
        loaded_weights,
        loaded_temperatures,
        loaded_pressures,
        loaded_wavenumber,
        molecule,
        molmass,
    ) = load_ckd(table_path)

    assert xsgrid.shape == (
        temperatures.size,
        pressures.size,
        samples.size,
        bin_centers.size,
    )
    assert molecule == "H2O"
    assert molmass == 18.0
    np.testing.assert_allclose(loaded_samples, samples)
    np.testing.assert_allclose(loaded_weights, weights)
    np.testing.assert_allclose(loaded_temperatures, temperatures)
    np.testing.assert_allclose(loaded_pressures, pressures)
    np.testing.assert_allclose(loaded_wavenumber, bin_centers)

    tiny = np.finfo(xsgrid.dtype).tiny
    assert xsgrid[0, 0, 0, 0] == tiny
    assert xsgrid[1, 0, 1, 0] == kcoeff[0, 1, 0, 1]
    assert xsgrid[2, 1, 0, 1] == kcoeff[1, 2, 1, 0]


def test_load_ckd_infers_molecule_name_when_dataset_is_absent(tmp_path):
    table_dir = tmp_path / "H2O" / "1H2-16O" / "POKAZATEL"
    table_dir.mkdir(parents=True)
    table_path = table_dir / "synthetic.h5"

    with h5py.File(table_path, "w") as handle:
        handle.create_dataset("mol_mass", data=np.array([18.0]))
        handle.create_dataset("bin_centers", data=np.array([1000.0]))
        handle.create_dataset("samples", data=np.array([0.5]))
        handle.create_dataset("weights", data=np.array([1.0]))
        handle.create_dataset("t", data=np.array([500.0]))
        handle.create_dataset("p", data=np.array([0.1]))
        handle.create_dataset("kcoeff", data=np.ones((1, 1, 1, 1)))

    *_, molecule, molmass = load_ckd(table_path)

    assert molecule == "H2O"
    assert molmass == 18.0


def test_load_ckd_accepts_integer_kcoeff(tmp_path):
    table_path = tmp_path / "integer.h5"
    kcoeff = np.array([[[[0, 2], [3, 4]]]], dtype=np.int64)

    with h5py.File(table_path, "w") as handle:
        handle.create_dataset("mol_name", data=np.array([b"H2O"]))
        handle.create_dataset("mol_mass", data=np.array([18.0]))
        handle.create_dataset("bin_centers", data=np.array([1000.0, 1100.0]))
        handle.create_dataset("samples", data=np.array([0.25, 0.75]))
        handle.create_dataset("weights", data=np.array([0.5, 0.5]))
        handle.create_dataset("t", data=np.array([500.0]))
        handle.create_dataset("p", data=np.array([0.1]))
        handle.create_dataset("kcoeff", data=kcoeff)

    xsgrid, *_ = load_ckd(table_path)

    assert np.issubdtype(xsgrid.dtype, np.floating)
    assert xsgrid[0, 0, 0, 0] == np.finfo(xsgrid.dtype).tiny
    assert xsgrid[0, 0, 1, 0] == 2.0


def test_download_exomolop_h5_does_not_leave_final_file_on_http_error(
    tmp_path, monkeypatch
):
    root = tmp_path / "ckd"
    table_dir = root / "H2O" / "1H2-16O" / "POKAZATEL"
    expected = _expected_table_path(root)
    error = requests.HTTPError("404 Client Error")
    session = _FakeSession(
        [
            _FakeResponse(status_code=403),
            _FakeResponse(error=error),
            _FakeResponse(error=error),
            _FakeResponse(error=error),
        ]
    )

    monkeypatch.setattr(provider_url, "url_ExoMol", lambda: "https://example.test/db/")
    monkeypatch.setattr(exomolop.requests, "Session", lambda: session)

    with pytest.raises(RuntimeError) as excinfo:
        download_exomolop_h5(table_dir)

    assert "Failed to download any ExoMolOP CKD table candidate" in str(excinfo.value)
    assert (
        "1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.petitRADTRANS.h5"
        in str(excinfo.value)
    )
    assert (
        "1H2-16O__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
        in str(excinfo.value)
    )
    assert (
        "1H2-16O__POKAZATEL.R1000_0.1-250mu.ktable.petitRADTRANS.h5"
        in str(excinfo.value)
    )
    assert "HTTPError: 404 Client Error" in str(excinfo.value)
    download_url = session.calls[1][0][0]
    assert download_url.endswith(
        "1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    )
    assert not expected.exists()
    assert not expected.with_name(expected.name + ".part").exists()
    assert not _legacy_dot_table_path(root).exists()
    assert not _legacy_dot_table_path(root).with_name(
        _legacy_dot_table_path(root).name + ".part"
    ).exists()


def test_download_exomolop_h5_replaces_existing_empty_file(tmp_path, monkeypatch):
    root = tmp_path / "ckd"
    table_dir = root / "H2O" / "1H2-16O" / "POKAZATEL"
    expected = _expected_table_path(root)
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"")
    session = _FakeSession(
        [
            _FakeResponse(status_code=403),
            _FakeResponse(chunks=[b"abc", b"", b"def"]),
        ]
    )

    monkeypatch.setattr(provider_url, "url_ExoMol", lambda: "https://example.test/db/")
    monkeypatch.setattr(exomolop.requests, "Session", lambda: session)

    downloaded = download_exomolop_h5(table_dir)

    assert downloaded == expected
    assert expected.read_bytes() == b"abcdef"
    assert not expected.with_name(expected.name + ".part").exists()


def test_download_exomolop_h5_falls_back_to_legacy_dot_filename(
    tmp_path, monkeypatch
):
    root = tmp_path / "ckd"
    table_dir = root / "H2O" / "1H2-16O" / "POKAZATEL"
    legacy = _legacy_dot_table_path(root)
    error = requests.HTTPError("404 Client Error")
    session = _FakeSession(
        [
            _FakeResponse(status_code=403),
            _FakeResponse(error=error),
            _FakeResponse(chunks=[b"legacy"]),
        ]
    )

    monkeypatch.setattr(provider_url, "url_ExoMol", lambda: "https://example.test/db/")
    monkeypatch.setattr(exomolop.requests, "Session", lambda: session)

    downloaded = download_exomolop_h5(table_dir)

    assert downloaded == legacy
    assert legacy.read_bytes() == b"legacy"
    assert session.calls[1][0][0].endswith(
        "1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    )
    assert session.calls[2][0][0].endswith(
        "1H2-16O__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    )


def test_download_exomolop_h5_uses_existing_legacy_dot_file(tmp_path, monkeypatch):
    root = tmp_path / "ckd"
    table_dir = root / "H2O" / "1H2-16O" / "POKAZATEL"
    legacy = _legacy_dot_table_path(root)
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"existing")
    session = _FakeSession([])

    monkeypatch.setattr(provider_url, "url_ExoMol", lambda: "https://example.test/db/")
    monkeypatch.setattr(exomolop.requests, "Session", lambda: session)

    downloaded = download_exomolop_h5(table_dir)

    assert downloaded == legacy
    assert legacy.read_bytes() == b"existing"
    assert session.calls == []


def test_download_exomolop_h5_tries_all_matching_listing_links(tmp_path, monkeypatch):
    root = tmp_path / "ckd"
    table_dir = root / "H2O" / "1H2-16O" / "POKAZATEL"
    legacy = _legacy_dot_table_path(root)
    error = requests.HTTPError("stale listing link")
    listing = """
    <html>
      <a href="bad/1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.petitRADTRANS.h5">new</a>
      <a href="ok/1H2-16O__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5">legacy</a>
    </html>
    """
    session = _FakeSession(
        [
            _FakeResponse(text=listing),
            _FakeResponse(error=error),
            _FakeResponse(chunks=[b"legacy"]),
        ]
    )

    monkeypatch.setattr(provider_url, "url_ExoMol", lambda: "https://example.test/db/")
    monkeypatch.setattr(exomolop.requests, "Session", lambda: session)

    downloaded = download_exomolop_h5(table_dir)

    assert downloaded == legacy
    assert legacy.read_bytes() == b"legacy"
    assert session.calls[1][0][0] == (
        "https://example.test/db/H2O/1H2-16O/POKAZATEL/bad/"
        "1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    )
    assert session.calls[2][0][0] == (
        "https://example.test/db/H2O/1H2-16O/POKAZATEL/ok/"
        "1H2-16O__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    )

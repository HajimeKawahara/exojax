"""ExoMolHR downloads are reusable only for identical queries."""

import io
import zipfile
from unittest.mock import MagicMock

import pytest

from exojax.provider.exomolhr import _fetch_opacity_zip


def _query():
    return dict(
        wvmin=0,
        wvmax=None,
        numin=1000.0,
        numax=2000.0,
        T=1200,
        Smin=1.0e-30,
        iso="12C-16O",
    )


@pytest.fixture
def session():
    session = MagicMock()
    archive = None

    def get(url, *, params=None, **kwargs):
        nonlocal archive
        response = MagicMock()
        if params is not None:
            csv_name = f"timestamp__{params['iso']}__{float(params['T']):.1f}K.csv"
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as zf:
                zf.writestr(csv_name, repr(params))
            archive = buffer.getvalue()
            response.text = '<a href="/download/?archive_name=opacity.zip">Download</a>'
        else:
            response.__enter__.return_value = response
            response.iter_content.return_value = [archive]
        return response

    session.get.side_effect = get
    return session


@pytest.mark.parametrize(
    "changed",
    [
        {"wvmin": 100},
        {"wvmax": 5000},
        {"numin": 1500.0},
        {"numax": 2500.0},
        {"T": 1400},
        {"Smin": 1.0e-25},
        {"iso": "13C-16O"},
    ],
)
def test_cache_matches_every_query_parameter(tmp_path, session, changed):
    first = _fetch_opacity_zip(**_query(), out_dir=tmp_path, session=session)
    first_contents = first.read_text()
    assert session.get.call_count == 2
    assert _fetch_opacity_zip(**_query(), out_dir=tmp_path, session=session) == first
    assert session.get.call_count == 2

    second_query = dict(_query(), **changed)
    second = _fetch_opacity_zip(**second_query, out_dir=tmp_path, session=session)
    assert session.get.call_count == 4
    assert first != second
    assert first.read_text() == first_contents
    assert second.read_text() != first_contents
    assert (
        _fetch_opacity_zip(**second_query, out_dir=tmp_path, session=session) == second
    )
    assert session.get.call_count == 4


def test_legacy_cache_without_query_identity_is_not_reused(tmp_path, session):
    legacy = tmp_path / "timestamp__12C-16O__1200.0K.csv"
    legacy.write_text("unknown query")
    downloaded = _fetch_opacity_zip(**_query(), out_dir=tmp_path, session=session)
    assert session.get.call_count == 2
    assert downloaded != legacy
    assert legacy.read_text() == "unknown query"

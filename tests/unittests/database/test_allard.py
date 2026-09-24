"""Allard readers: published sample values and local archive formats."""

from io import BytesIO
import tarfile

import numpy as np
import pytest

from exojax.database.alkali import load_allard2019, read_allard_profile


# Two rows from Allard et al. (2019), CDS J/A+A/628/A120, opacity.tar.gz,
# TABLES_D2_NaH2_2017/tableD2_NaH2_1000_1e21_FS17.omg.
# The original has 91 rows; only the row-count header is reduced here.
# https://cdsarc.cds.unistra.fr/ftp/J/A+A/628/A120/
D2_SAMPLE = """Date of calculation: Tue Jan 2 14:53:20 2018
TK: 1000.00
lambda: 5891.58
absorption oscillator strength fabs 0.64 cgs
 radiator perturber mass: 23.00 2.00
omega cm-1 1st order.. until 15order
end
5891.582 1.E+21 7.60885796 5.6367712E-13
2 15
21.9821136 1.69550044
100 0.0010622583 0.00776389262 0.027981409 0.0680600108 0.124747936
0.183110854 0.223896832 0.234446474 0.21457295 0.174363462
0.127373158 0.0844920747 0.0513202983 0.0287437682 0.0149341305
1000 7.4352412e-05 0.000562513778 0.0021305834 0.00538006631
0.0101871374 0.0154273795 0.0194637878 0.0210421451 0.0198991932
0.01672264 0.0126443166 0.00868908956 0.00547202709 0.0031801403 0.00171572783
"""


def test_published_d2_sample(tmp_path):
    path = tmp_path / "table.omg"
    path.write_text(D2_SAMPLE)
    profile = read_allard_profile(path)
    assert profile.temperature == 1000
    assert profile.wavelength == 5891.582
    assert profile.width == 21.9821136
    assert profile.shift == 1.69550044
    np.testing.assert_array_equal(profile.offsets, [100, 1000])
    assert profile.coefficients.shape == (2, 15)
    ratio = 1e19 / profile.density
    sigma = (profile.normalization * np.exp(-profile.volume * ratio)
             * (profile.coefficients @ ratio ** np.arange(1, 16)))
    # Independent reference: D2/EXAMPLE_D2/sigma_out.omg, n(H2)=1e19.
    np.testing.assert_allclose(sigma, [5.969561e-18, 4.189274e-19], rtol=1e-6)


def test_header_variations_and_surplus_rows(tmp_path):
    text = D2_SAMPLE.replace("TK: 1000.00", "  TK:    1.000D+03")
    text = text.replace("until 15order", "until 99 order (descriptive only)")
    text = text.replace("1.E+21", "1.D+21")
    # Na D1 500/725 K tables have complete rows beyond the declared count.
    text += "2000 " + " ".join(["-1e-6"] * 15) + "\n"
    path = tmp_path / "table.omg"
    path.write_text(text)
    profile = read_allard_profile(path)
    assert profile.temperature == 1000
    assert profile.coefficients.shape == (2, 15)
    np.testing.assert_array_equal(profile.offsets, [100, 1000])


@pytest.mark.parametrize("text", [
    D2_SAMPLE.replace("end\n", ""),
    D2_SAMPLE.replace("TK:", "temperature:"),
    D2_SAMPLE.replace("2 15\n", "3 15\n"),
    D2_SAMPLE.rsplit(" ", 1)[0],
    D2_SAMPLE.replace("1000 7.4352412e-05", "100 7.4352412e-05"),
    D2_SAMPLE.replace("7.60885796", "nan"),
    D2_SAMPLE.replace("21.9821136", "-21.9821136"),
])
def test_malformed_profile_rejected(tmp_path, text):
    path = tmp_path / "broken.omg"
    path.write_text(text)
    with pytest.raises(ValueError, match="Invalid Allard profile"):
        read_allard_profile(path)


def _collection():
    """Small synthetic collection for exercising discovery, not the physics."""
    files = {}
    for temperature in [1500, 1000]:
        text = D2_SAMPLE.replace("TK: 1000.00", f"TK: {temperature}")
        files[f"TABLES_D2_NaH2_2017/tableD2_NaH2_{temperature}.omg"] = text.encode()
        d1 = text.replace("5891.582", "5897.558")
        files[f"TABLES_D1_NaH2_2017/T{temperature}/table_nearwing_{temperature}.omg"] = d1.encode()
        files[f"TABLES_D1_NaH2_2017/T{temperature}/redwing_NaH2_D1_21_{temperature}.omg"] = b"-4000 1e-19\n-100 1e-16\n"
    return files


def _tar(files):
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, BytesIO(content))
    return buffer.getvalue()


@pytest.mark.parametrize("archive", [False, True])
def test_load_local_collection(tmp_path, archive):
    files = _collection()
    # The original archive includes identical copies in example directories.
    files["D2/tableD2_NaH2_1000.omg"] = files["TABLES_D2_NaH2_2017/tableD2_NaH2_1000.omg"]
    if archive:
        path = tmp_path / "opacity.tar.gz"
        path.write_bytes(_tar({"ALLARD_NaH2/tables.tar.gz": _tar(files)}))
    else:
        path = tmp_path
        for name, content in files.items():
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    profiles = load_allard2019(path)
    assert list(profiles) == ["D1", "D2"]
    assert [profile.temperature for profile in profiles["D1"]] == [1000, 1500]
    assert [profile.temperature for profile in profiles["D2"]] == [1000, 1500]
    np.testing.assert_array_equal(profiles["D1"][0].red_offsets, [-4000, -100])
    assert profiles["D2"][0].red_offsets is None


@pytest.mark.parametrize("problem,match", [
    ("species", "Not an Allard 2019"),
    ("grid", "matching temperature grids"),
    ("red", "Missing Na-H2 D1 far-red-wing"),
    ("conflict", "Conflicting Na-H2 D2 tables"),
])
def test_invalid_collection(tmp_path, problem, match):
    files = _collection()
    key = "TABLES_D2_NaH2_2017/tableD2_NaH2_1000.omg"
    if problem == "species":
        files[key] = files[key].replace(b"5891.582", b"7667.02")
    elif problem == "grid":
        del files[key]
    elif problem == "red":
        del files["TABLES_D1_NaH2_2017/T1000/redwing_NaH2_D1_21_1000.omg"]
    else:
        files["D2/tableD2_NaH2_1000.omg"] = files[key].replace(b"7.60885796", b"7.6")
    path = tmp_path / "opacity.tar.gz"
    path.write_bytes(_tar(files))
    with pytest.raises(ValueError, match=match):
        load_allard2019(path)

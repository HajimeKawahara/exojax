"""K--He import and opacity checks against the corrected CDS distribution."""

from io import BytesIO
import tarfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exojax.database.alkali_k import load_allard2024
from exojax.opacity.allard import OpaAlkaliTable, density_expansion
from exojax.utils.constants import kB


@pytest.fixture(autouse=True)
def precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


# Two rows from Allard et al. (2024), CDS J/A+A/683/A188,
# D2/tableD2_KHe_1000_1e21.omg. Only the declared row count is reduced.
# https://cdsarc.cds.unistra.fr/ftp/J/A+A/683/A188/
D2_SAMPLE = """Date of calculation: Sat Nov 26 11:38:22 2016
TK: 1000.00
lambda: 7667.02
absorption oscillator strength fabs 0.68 cgs
 radiator perturber mass: 39.00 4.00
omega cm-1 1st order.. until 16order
end
7667.02 1.E+21 6.56725899 5.99918533E-13
2 16
18.4064602 -4.70989447
100 0.00196526234 0.0102945333 0.0307252305 0.0625077257
0.0957171648 0.117282483 0.119707073 0.104673105
0.0800443577 0.0543820925 0.0332370071 0.0184592532
0.00939421747 0.00441178353 0.00192344804 0.000782546621
1000 9.46208301e-05 0.000613781143 0.001988606 0.00430183306
0.00698773327 0.00908896544 0.00985895531 0.0091716493
0.00746888642 0.00540805225 0.00352492948 0.00208885855
0.00113471444 0.000568958618 0.000264877201 0.000115074647
"""
TEMPERATURES = [500, 800, 1000, 1500, 2000, 2500, 3000]
CORRECTED = "D2/tableD2_KHe_800_1e21_2025.omg"


def _collection():
    """Synthetic other temperatures/components exercise loading only."""
    files = {}
    for temperature in reversed(TEMPERATURES):
        text = D2_SAMPLE.replace("TK: 1000.00", f"TK: {temperature}")
        suffix = "_2025" if temperature == 800 else ""
        files[f"D2/tableD2_KHe_{temperature}_1e21{suffix}.omg"] = text.encode()
        # Keep the synthetic D1 wing outside the D2 sample's blue wing.
        d1 = text.replace("7667.02", "7701.1")
        d1 = d1.replace("\n100 ", "\n-1000 ").replace("\n1000 ", "\n-100 ")
        files[f"D1/tableD1_KHe_{temperature}_1e21.omg"] = d1.encode()
    return files


def _archive(files):
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, BytesIO(content))
    return buffer.getvalue()


def _write_collection(path, files):
    for name, content in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return path


@pytest.mark.parametrize("archive", [False, True])
def test_corrected_collection_and_reference_sample(tmp_path, archive):
    files = _collection()
    duplicate = "D2/EXAMPLE_D2/tableD2_KHe_1000_1e21.omg"
    files[duplicate] = files["D2/tableD2_KHe_1000_1e21.omg"]
    # Superseded 800 K data must never replace the corrected table.
    files["D2/tableD2_KHe_800_1e21.omg"] = b"superseded data\n"
    if archive:
        path = tmp_path / "khe.tar.gz"
        path.write_bytes(_archive({"tables.tar.gz": _archive(files)}))
    else:
        path = _write_collection(tmp_path, files)
    profiles = load_allard2024(path)
    for component in ["D1", "D2"]:
        assert [profile.temperature for profile in profiles[component]] == TEMPERATURES
        assert all(profile.red_offsets is None for profile in profiles[component])
    profile = profiles["D2"][2]
    assert profile.wavelength == 7667.02
    assert profile.width == 18.4064602
    assert profile.coefficients.shape == (2, 16)
    sigma = density_expansion(profile.coefficients, 0.01,
                              profile.volume, profile.normalization)
    # Independent published EXAMPLE_D2/sigma_out.omg at n(He)=1e19 cm-3.
    np.testing.assert_allclose(sigma, [1.163652e-17, 5.671898e-19], rtol=1e-6)


@pytest.mark.parametrize("problem,match", [
    ("old_only", "Missing corrected"),
    ("incomplete", "all seven temperatures"),
    ("wrong_perturber", "Not an Allard 2024"),
    ("wrong_temperature", "Not an Allard 2024"),
    ("conflict", "Conflicting K--He"),
])
def test_invalid_collection(tmp_path, problem, match):
    files = _collection()
    if problem == "old_only":
        files[CORRECTED.replace("_2025", "")] = files.pop(CORRECTED)
    elif problem == "incomplete":
        del files["D1/tableD1_KHe_3000_1e21.omg"]
    elif problem == "wrong_perturber":
        files[CORRECTED] = files[CORRECTED].replace(b"39.00 4.00", b"39.00 2.00")
    elif problem == "wrong_temperature":
        files[CORRECTED] = files[CORRECTED].replace(b"TK: 800", b"TK: 1000")
    else:
        files["EXAMPLE/" + CORRECTED] = files[CORRECTED].replace(b"6.56725899", b"6.5")
    with pytest.raises(ValueError, match=match):
        load_allard2024(_write_collection(tmp_path, files))


def test_k_opacity_reference_and_partial_pressure(tmp_path):
    path = _write_collection(tmp_path, _collection())
    grid = 1e8 / 7667.02 + np.array([100.0, 1000.0])
    opacity = OpaAlkaliTable(grid, path, model="allard2024_k_he")
    mixture = OpaAlkaliTable(grid, path, model="allard2024_k_he", vmr_perturber=0.1)
    pressure = 1e19 * kB * 1000.0 / 1e6
    sigma = opacity.xsvector(1000.0, pressure)
    assert (opacity.species, opacity.perturber) == ("K", "He")
    np.testing.assert_allclose(sigma, [1.163652e-17, 5.671898e-19], rtol=1e-5)
    np.testing.assert_allclose(mixture.xsvector(1000.0, pressure * 10), sigma, rtol=1e-6)
    np.testing.assert_allclose(jax.jit(opacity.xsvector)(1000.0, pressure), sigma, rtol=1e-6)
    derivative = jax.grad(lambda p: jnp.log(opacity.xsvector(1000.0, p)).sum())(pressure)
    assert np.isfinite(derivative)
    assert derivative > 0
    matrix = opacity.xsmatrix(jnp.array([1000.0, 1200.0]), jnp.array([pressure, pressure]))
    assert matrix.shape == (2, 2)
    assert np.all(np.isfinite(matrix))

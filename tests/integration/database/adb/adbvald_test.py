from pathlib import Path

import pytest

from exojax.database import AdbSepVald, AdbVald


DATABASE_DIR = Path(__file__).resolve().parents[4] / ".database"
VALD2600_PATH = DATABASE_DIR / "vald2600.gz"
VALD4214450_PATH = DATABASE_DIR / "vald4214450.gz"

requires_vald2600 = pytest.mark.skipif(
    not VALD2600_PATH.is_file(),
    reason="requires locally provisioned .database/vald2600.gz",
)
requires_vald4214450 = pytest.mark.skipif(
    not VALD4214450_PATH.is_file(),
    reason="requires locally provisioned .database/vald4214450.gz",
)


@requires_vald2600
def test_adb_vald():
    adbV = AdbVald(VALD2600_PATH, nurange=[9660.0, 9570.0])
    assert adbV.atomicmass[0] == 55.847


@requires_vald2600
def test_adb_vald_interp():
    adbV = AdbVald(VALD2600_PATH, nurange=[9660.0, 9570.0])
    T = 1000.0
    qt_284 = adbV.QT_interp_284(T)
    assert qt_284[76] == 15.7458  # Fe I


@requires_vald4214450
def test_adb_sepvald():
    adbV = AdbVald(VALD4214450_PATH, nurange=[9660.0, 9570.0], crit=1e-100)
    # The criterion avoids errors caused by zero broadening for weak lines.
    asdb = AdbSepVald(adbV)
    assert asdb.atomicmass[asdb.ielem == 26][0] == 55.847


if __name__ == "__main__":
    test_adb_vald()
    test_adb_vald_interp()
    test_adb_sepvald()

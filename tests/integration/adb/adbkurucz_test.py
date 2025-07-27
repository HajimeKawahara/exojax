import os
from exojax.database.moldb  import AdbKurucz

filepath_Kurucz = '.database/gf2600.all'
if not os.path.isfile(filepath_Kurucz):
    import urllib.request
    try:
        url = "http://kurucz.harvard.edu/linelists/gfall/gf2600.all"
        urllib.request.urlretrieve(url, filepath_Kurucz)
    except:
        print('could not connect ', url)

def test_adb_kurucz():
    adbK = AdbKurucz(filepath_Kurucz, nurange=[9660., 9570.])
    assert adbK.atomicmass[0] == 55.847

def test_adb_kurucz_interp():
    adbK = AdbKurucz(filepath_Kurucz, nurange=[9660., 9570.])
    T = 1000.0
    qt_284 = adbK.QT_interp_284(T)
    assert qt_284[76] == 15.7458 #Fe I

if __name__ == "__main__":
    test_adb_kurucz()
    test_adb_kurucz_interp()

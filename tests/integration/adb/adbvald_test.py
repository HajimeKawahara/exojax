import os
from exojax.database.moldb  import AdbVald, AdbSepVald

import urllib.request
from exojax.utils.url import url_developer_data

filepath_VALD3 = '.database/vald2600.gz'
path_ValdLineList = '.database/vald4214450.gz'
if not os.path.isfile(filepath_VALD3):
    try:
        url = url_developer_data()+'vald2600.gz'
        urllib.request.urlretrieve(url, filepath_VALD3)
    except:
        print('could not connect ', url_developer_data())
if not os.path.isfile(path_ValdLineList):
    try:
        url = url_developer_data()+'vald4214450.gz'
        urllib.request.urlretrieve(url, path_ValdLineList)
    except:
        print('could not connect ', url_developer_data())

def test_adb_vald():
    adbV = AdbVald(filepath_VALD3, nurange=[9660., 9570.])
    assert adbV.atomicmass[0] == 55.847

def test_adb_vald_interp():
    adbV = AdbVald(filepath_VALD3, nurange=[9660., 9570.])
    T = 1000.0
    qt_284 = adbV.QT_interp_284(T)
    assert qt_284[76] == 15.7458 #Fe I

def test_adb_sepvald():
    adbV = AdbVald(path_ValdLineList,  nurange=[9660., 9570.], crit = 1e-100) 
    #The crit is defined just in case some weak lines may cause an error that results in a gamma of 0... (220219)
    asdb = AdbSepVald(adbV)
    assert asdb.atomicmass[asdb.ielem==26][0] == 55.847

if __name__ == "__main__":
    test_adb_vald()
    test_adb_vald_interp()
    test_adb_sepvald()

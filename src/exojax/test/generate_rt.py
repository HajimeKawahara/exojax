import numpy as np
from exojax.test.emulate_mdb import mock_wavenumber_grid
import matplotlib.pyplot as plt
from exojax.test.data import TESTDATA_CO_EXOMOL_LPF_EMISSION_REF
from exojax.test.data import TESTDATA_CO_HITEMP_LPF_EMISSION_REF
from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
from exojax.test.data import TESTDATA_CO_HITEMP_MODIT_EMISSION_REF
    
from exojax.test.emulate_mdb import mock_mdb
from exojax.spec.opacalc import OpaDirect
from exojax.spec.opacalc import OpaModit
from exojax.spec.atmrt import ArtEmisPure

from jax.config import config

config.update("jax_enable_x64", True)

testdata_modit={}
testdata_modit["exomol"]=TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF
testdata_modit["hitemp"]=TESTDATA_CO_HITEMP_MODIT_EMISSION_REF

testdata_lpf={}
testdata_lpf["exomol"]=TESTDATA_CO_EXOMOL_LPF_EMISSION_REF
testdata_lpf["hitemp"]=TESTDATA_CO_HITEMP_LPF_EMISSION_REF

def gendata_rt_modit(db):

    nu_grid, wav, res = mock_wavenumber_grid()

    art = ArtEmisPure(nu_grid,
                      pressure_top=1.e-8,
                      pressure_btm=1.e2,
                      nlayer=100)
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_mmr_profile(0.1)
    gravity = 2478.57
    #gravity = art.constant_gravity_profile(2478.57) #gravity can be profile

    mdb = mock_mdb(db)
    #mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
    #mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
    opa = OpaModit(mdb=mdb,
                   nu_grid=nu_grid,
                   Tarr_list=Tarr,
                   Parr=art.pressure,
                   dit_grid_resolution=0.2)
    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_lines(xsmatrix, mmr_arr, opa.mdb.molmass,
                                     gravity)
    F0 = art.run(dtau, Tarr)
    np.savetxt(testdata_modit[db], np.array([nu_grid, F0]).T, delimiter=",")
    
    return nu_grid, F0


def gendata_rt_lpf(db):
    nu_grid, wav, res = mock_wavenumber_grid()

    art = ArtEmisPure(nu_grid,
                      pressure_top=1.e-8,
                      pressure_btm=1.e2,
                      nlayer=100)
    art.change_temperature_range(400.0, 1500.0)
    Tarr = art.powerlaw_temperature(1300.0, 0.1)
    mmr_arr = art.constant_mmr_profile(0.1)
    gravity = 2478.57
    #gravity = art.constant_gravity_profile(2478.57) #gravity can be profile

    mdb = mock_mdb(db)
    #mdb = api.MdbExomol('.database/CO/12C-16O/Li2015',nu_grid,inherit_dataframe=False,gpu_transfer=False)
    #mdb = api.MdbHitemp('CO', art.nu_grid, gpu_transfer=False, isotope=1)
    opa = OpaDirect(mdb=mdb, nu_grid=nu_grid)

    xsmatrix = opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_lines(xsmatrix, mmr_arr, opa.mdb.molmass,
                                     gravity)
    F0 = art.run(dtau, Tarr)
    np.savetxt(testdata_lpf[db], np.array([nu_grid, F0]).T, delimiter=",")
    
    return nu_grid, F0


if __name__ == "__main__":
    nus, F0_exomol = gendata_rt_modit("exomol")
    nus, F0_hitemp = gendata_rt_modit("hitemp")
    nus, F0_exomol_lpf = gendata_rt_lpf("exomol")
    nus, F0_hitemp_lpf = gendata_rt_lpf("hitemp")

    fig = plt.figure()
    fig.add_subplot(211)
    plt.plot(nus, F0_exomol)
    plt.plot(nus, F0_hitemp)
    plt.plot(nus, F0_exomol_lpf, ls="dashed")
    plt.plot(nus, F0_hitemp_lpf, ls="dashed")
    
    fig.add_subplot(212)
    plt.plot(nus, 1.0 - F0_exomol / F0_exomol_lpf, label="diff exomol")
    plt.plot(nus, 1.0 - F0_hitemp / F0_hitemp_lpf, label="diff hitemp")
    plt.legend()
    plt.show()
    
    import matplotlib.pyplot as plt
    plt.plot(nus, F0_exomol_lpf)
    plt.plot(nus, F0_exomol, ls="dashed")
    plt.show()


    print(
        "to include the generated files in the package, move .txt to exojax/src/exojax/data/testdata/"
    )

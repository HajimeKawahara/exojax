"""generate test data

"""
from exojax.spec import moldb
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA, gen_wavenumber_grid

def gendata_exomol():
    Nx=1500
    nus, wav, res = gen_wavenumber_grid(29200.0,30000., Nx, unit='AA')
    nus, wav, res = nugrid(29200.0,30000., Nx, unit='AA')

    mdb = moldb.MdbExomol('.database/CO/12C-16O/Li2015', nus, crit=1.e-30)
    print(len(mdb.A))

if __name__ == "__main__":
    gendata_exomol()

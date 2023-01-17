import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.spec.initspec import init_modit
from exojax.test.emulate_mdb import mock_mdbExomol
from jax.config import config

config.update("jax_enable_x64", True)

def gendata_rt_modit_exomol():
    """generate a sample adiative transfered spectrum using MODIT/exomol

    Returns:
        _type_: _description_
    """
    import jax.numpy as jnp
    from exojax.spec import rtransfer as rt
    from exojax.spec.modit import exomol
    from exojax.spec.modit import xsmatrix
    from exojax.spec.rtransfer import dtauM
    from exojax.spec.rtransfer import rtrun
    from exojax.spec.planck import piBarr
    from exojax.spec.modit import set_ditgrid_matrix_exomol
    from exojax.test.data import TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF

    mdb = mock_mdbExomol()
    Parr, dParr, k = rt.pressure_layer(NP=100)
    T0_in = 1300.0
    alpha_in = 0.1
    Tarr = T0_in * (Parr)**alpha_in
    molmass = mdb.molmass
    MMR = 0.1
    nus, wav, res = wavenumber_grid(22900.0,
                                    23100.0,
                                    15000,
                                    unit='AA',
                                    xsmode="modit")
    cont_nu, index_nu, R, pmarray = init_modit(mdb.nu_lines, nus)
    def fT(T0, alpha):
        return T0[:, None] * (Parr[None, :])**alpha[:, None]

    dgm_ngammaL = set_ditgrid_matrix_exomol(mdb, fT, Parr, R, molmass, 0.2,
                                            np.array([T0_in]),
                                            np.array([alpha_in]))

    g = 2478.57
    SijM, ngammaLM, nsigmaDl = exomol(mdb, Tarr, Parr, R, molmass)
    xsm = xsmatrix(cont_nu, index_nu, R, pmarray, nsigmaDl, ngammaLM, SijM,
                   nus, dgm_ngammaL)
    dtau = dtauM(dParr, jnp.abs(xsm), MMR * np.ones_like(Parr), molmass, g)
    sourcef = piBarr(Tarr, nus)
    F0 = rtrun(dtau, sourcef)
    np.savetxt(TESTDATA_CO_EXOMOL_MODIT_EMISSION_REF,np.array([nus,F0]).T,delimiter=",")
    return nus, F0

if __name__ == "__main__":
    nus, F0 = gendata_rt_modit_exomol()

    print(
        "to include the generated files in the package, move .txt to exojax/src/exojax/data/testdata/"
    )

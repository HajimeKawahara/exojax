import pandas as pd
import pkgutil
from io import BytesIO
import numpy as np
from exojax.utils.grids import wavenumber_grid
from exojax.postproc import response
from exojax.database import molinfo 
from exojax.database import contdb 
from exojax.database.exomol.api import MdbExomol
from exojax.opacity import OpaDirect
from exojax.opacity import OpaCIA
from exojax.rt import ArtEmisPure
from exojax.postproc.specop import SopRotation
from exojax.postproc.specop import SopInstProfile
from exojax.utils.instfunc import resolution_to_gaussian_std
import jax.numpy as jnp


def test_jaxopt_spectrum(fig=False):
    np.random.seed(1)
    specdata = pkgutil.get_data("exojax", "data/testdata/spectrum.txt")
    dat = pd.read_csv(BytesIO(specdata), delimiter=",", names=("wav", "flux"))
    wavd = dat["wav"].values
    flux = dat["flux"].values
    nusd = jnp.array(1.0e8 / wavd[::-1])
    sigmain = 0.05
    norm = 40000
    nflux = flux / norm + np.random.normal(0, sigmain, len(wavd))

    # wavenumber grid setting
    Nx = 1500
    nus, wav, resolution = wavenumber_grid(
        np.min(wavd) - 5.0, np.max(wavd) + 5.0, Nx, unit="AA", xsmode="premodit"
    )

    instrument_resolution = 100000.0
    beta_inst = resolution_to_gaussian_std(instrument_resolution)
    mmw = 2.33  # mean molecular weight
    mmrH2 = 0.74
    molmassH2 = molinfo.molmass_isotope("H2")
    vmrH2 = mmrH2 * mmw / molmassH2  # VMR
    Mp = 33.2  # fixing mass...

    # art
    art = ArtEmisPure(nu_grid=nus, pressure_top=1.0e-8, pressure_btm=1.0e2, nlayer=100)

    # mdb/cdb
    mdbCO = MdbExomol(
        ".database/CO/12C-16O/Li2015", nus, crit=1.0e-46, gpu_transfer=True
    )
    cdbH2H2 = contdb.CdbCIA(".database/H2-H2_2011.cia", nus)

    # opa
    opa = OpaDirect(mdbCO, nus)
    opacia = OpaCIA(cdbH2H2, nus)

    # spectral operators
    vsini_max = 100.0
    sos_rot = SopRotation(nus, vsini_max)
    sos_ip = SopInstProfile(nus, vsini_max)

    def model_c(params, boost, nu1):
        Rp, RV, MMR_CO, T0, alpha, vsini = params * boost
        g = 2478.57730044555 * Mp / Rp**2  # gravity
        u1 = 0.0
        u2 = 0.0
        # T-P model//
        Tarr = art.powerlaw_temperature(T0, alpha)

        def obyo(nusd):
            # CO
            xsm_CO = opa.xsmatrix(Tarr, art.pressure)
            mmr_profile = art.constant_profile(MMR_CO)
            dtaumCO = art.opacity_profile_xs(xsm_CO, mmr_profile, mdbCO.molmass, g)
            # CIA
            logacia = opacia.logacia_matrix(Tarr)
            dtaucH2H2 = art.opacity_profile_cia(logacia, Tarr, vmrH2, vmrH2, mmw, g)

            dtau = dtaumCO + dtaucH2H2
            F0 = art.run(dtau, Tarr) / norm

            Frot = sos_rot.rigid_rotation(F0, vsini, u1, u2)
            Frot_ip = sos_ip.ipgauss(Frot, beta_inst)
            mu = sos_ip.sampling(Frot_ip, RV, nusd)
            return mu

        model = obyo(nu1)
        return model

    import jaxopt

    boost = np.array([1.0, 10.0, 0.1, 1000.0, 1.0e-3, 10.0])
    initpar = np.array([0.8, 9.0, 0.1, 1200.0, 0.1, 17.0]) / boost

    def objective(params):
        f = nflux - model_c(params, boost, nusd)
        g = jnp.dot(f, f)
        return g

    gd = jaxopt.GradientDescent(fun=objective, maxiter=1000, stepsize=1.0e-4)
    resolution = gd.run(init_params=initpar)
    params, state = resolution
    model = model_c(params, boost, nusd)
    resid = np.sqrt(np.sum((nflux - model) ** 2) / len(nflux))

    if fig:
        import matplotlib.pyplot as plt

        plt.plot(nusd, nflux)
        plt.plot(nusd, model, ls="dashed")
        plt.show()
    print(resid)
    assert resid < 0.05


if __name__ == "__main__":
    test_jaxopt_spectrum(fig=True)

from jax import config 
config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import math
import os

from exojax.atm.atmprof import pressure_layer_logspace
from exojax.utils.grids import wavenumber_grid
from exojax.database.multimol  import MultiOpa
from exojax.database import contdb 
from exojax.rt.layeropacity import layer_optical_depth, layer_optical_depth_CIA
from exojax.rt import ArtEmisPure

from petitRADTRANS.radtrans import Radtrans
from exojax.database import molinfo 
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.config import petitradtrans_config_parser
petitradtrans_config_parser.set_input_data_path(r'~/database/petitRADTRANS/input_data')

from exojax.utils.instfunc import resolution_to_gaussian_std
from exojax.utils.grids import velocity_grid
from exojax.postproc import response



def run_exojax(path_data, ld_min, ld_max, mols, db, T0, alpha, logg, logvmr):
    Parr, dParr, k = pressure_layer_logspace(log_pressure_top=-3., nlayer=200)
    ONEARR = np.ones_like(Parr)


    R = 900000.
    ld_min = ld_min - 5.
    ld_max = ld_max + 5.
    nu_min = 1.0e8 / ld_max
    nu_max = 1.0e8 / ld_min
    Nx = math.ceil(R * np.log(nu_max / nu_min)) + 1 # ueki
    Nx = math.ceil(Nx / 2.) * 2 # make even

    nus, wav, res = wavenumber_grid(ld_min, ld_max, Nx, unit="AA", xsmode="premodit")
    nus = [nus]
    wav = [wav]
    res = [res]


    mul = MultiOpa(molmulti=[mols], dbmulti=[db], database_root_path=path_data)
    multimdb = mul.multimdb(nus, crit=1.e-30, Ttyp=1000.)
    multiopa = mul.multiopa_premodit(multimdb, nus, auto_trange=[500.,1500.], dit_grid_resolution=1.0)

    cdbH2H2 = []
    cdbH2He = []
    for k in range(len(nus)):
        cdbH2H2.append(contdb.CdbCIA(os.path.join(path_data, 'H2-H2_2011.cia'), nus[k]))
        cdbH2He.append(contdb.CdbCIA(os.path.join(path_data, 'H2-He_2011.cia'), nus[k]))

    molmass_list, molmassH2, molmassHe = mul.molmass()


    def frun(T0, alpha, logg, logvmr):
        Tarr = T0 * (Parr)**alpha
        Tarr = np.clip(Tarr, 500, None)

        g = 10.**logg # cgs

        vmr = jnp.power(10., jnp.array(logvmr))
        vmrH2 = (1. - jnp.sum(vmr)) * 6./7.
        vmrHe = (1. - jnp.sum(vmr)) * 1./7.
        mmw = jnp.sum(vmr*jnp.array(molmass_list)) + vmrH2*molmassH2 + vmrHe*molmassHe
        mmr = jnp.multiply(vmr, jnp.array(molmass_list)) / mmw

        mu_ibased = []
        mu_fbased = []
        for k in range(len(nus)):
            art = ArtEmisPure(pressure_top=1.e-3,
                              pressure_btm=1.e2,
                              nlayer=200,
                              nu_grid=nus[k],
                              rtsolver="ibased",
                              nstream=6)

            dtaum = []
            for i in range(len(mul.masked_molmulti[k])):
                xsm = multiopa[k][i].xsmatrix(Tarr, Parr)
                xsm = jnp.abs(xsm)
                dtaum.append(layer_optical_depth(dParr, xsm, mmr[mul.mols_num[k][i]]*ONEARR, molmass_list[mul.mols_num[k][i]], g))

            dtau = sum(dtaum)

            if(len(cdbH2H2[k].nucia) > 0):
                dtaucH2H2 = layer_optical_depth_CIA(nus[k], Tarr, Parr, dParr, vmrH2, vmrH2, mmw, g, cdbH2H2[k].nucia, cdbH2H2[k].tcia, cdbH2H2[k].logac)
                dtau = dtau + dtaucH2H2
            if(len(cdbH2He[k].nucia) > 0):
                dtaucH2He = layer_optical_depth_CIA(nus[k], Tarr, Parr, dParr, vmrH2, vmrHe, mmw, g, cdbH2He[k].nucia, cdbH2He[k].tcia, cdbH2He[k].logac)
                dtau = dtau + dtaucH2He

            F0_ibased = art.run(dtau, Tarr)

            art.rtsolver = "fbased2st"
            art.nstream = 2
            F0_fbased = art.run(dtau, Tarr)

            mu_ibased.append(F0_ibased)
            mu_fbased.append(F0_fbased)

        return mu_ibased, mu_fbased

    f_ibased, f_fbased = frun(T0, alpha, logg, logvmr)

    return nus[0], f_ibased[0], f_fbased[0]



def run_petit(ld_min, ld_max, mols, mols_exojax, T0, alpha, logg, logvmr):
    radtrans = Radtrans(pressures = np.logspace(-3, 2, 200),
                        line_species = mols,
                        gas_continuum_contributors = ['H2-H2', 'H2-He'],
                        wavelength_boundaries = [(ld_min - 5.)*1e-4, (ld_max + 5.)*1e-4],
                        line_opacity_mode = 'lbl')

    pressures_bar = radtrans.pressures * 1e-6
    temperatures = T0*(pressures_bar)**alpha
    temperatures = np.clip(temperatures, 500, None)


    molmass_list = []
    for i in range(len(mols_exojax)):
        molmass_list.append(molinfo.molmass(mols_exojax[i]))
    molmassH2=molinfo.molmass("H2")
    molmassHe=molinfo.molmass("He", db_HIT=False)

    vmr = jnp.power(10., jnp.array(logvmr))
    vmrH2 = (1. - jnp.sum(vmr)) * 6./7.
    vmrHe = (1. - jnp.sum(vmr)) * 1./7.
    mmw = jnp.sum(vmr*jnp.array(molmass_list)) + vmrH2*molmassH2 + vmrHe*molmassHe
    mmr = jnp.multiply(vmr, jnp.array(molmass_list)) / mmw
    mmrH2 = vmrH2 * molmassH2 / mmw
    mmrHe = vmrHe * molmassHe / mmw

    mass_fractions = {}
    mass_fractions['H2'] = mmrH2 * np.ones_like(temperatures)
    mass_fractions['He'] = mmrHe * np.ones_like(temperatures)
    for i in range(len(mols)):
        mass_fractions[mols[i]] = mmr[i] * np.ones_like(temperatures)


    mean_molar_masses = mmw * np.ones_like(temperatures)
    reference_gravity = 1e1**(logg)

    frequencies, flux, _ = radtrans.calculate_flux(temperatures=temperatures,
                                                   mass_fractions=mass_fractions,
                                                   mean_molar_masses=mean_molar_masses,
                                                   reference_gravity=reference_gravity,
                                                   frequencies_to_wavelengths=False)

    ld = cst.c/frequencies/1e-4 # [um]
    nus = 1.0e4 / ld # [cm^{-1}]
    f = flux * cst.c #[erg cm^{-2} s^{-1} Hz^{-1}] => [erg/s/cm^2/cm^{-1}]

    nus = nus[::-1]
    f = f[::-1]

    return nus, f



path_data = "/home/kawashima/database"
ld_min = 14600.
ld_max = 16500.
mols_exojax = ['H2O', 'CH4']
db_exojax = ['ExoMol', 'HITEMP']
mols_petit = ['1H2-16O__POKAZATEL', '12C-1H4__Hargreaves']

T0 = 995.56
alpha = 0.09
logg = 5.01
logvmr = [-2.93, -3.06]

nus1, f1_ibased, f1_fbased = run_exojax(path_data, ld_min, ld_max, mols_exojax, db_exojax, T0, alpha, logg, logvmr)
nus2, f2 = run_petit(ld_min, ld_max, mols_petit, mols_exojax, T0, alpha, logg, logvmr)



Rinst = 7000.
nu_min = 1.0e8 / ld_max
nu_max = 1.0e8 / ld_min
Nx = math.ceil(Rinst * np.log(nu_max / nu_min)) + 1 # ueki
Nx = math.ceil(Nx / 2.) * 2 # make even
nusd, wav, res = wavenumber_grid(ld_min, ld_max, Nx, unit="AA", xsmode="premodit")

beta_inst = resolution_to_gaussian_std(Rinst)
res_calc = 900000.
vsini_max = 100.0
vr_array = velocity_grid(res_calc, vsini_max)

f1_ibased_obs = response.ipgauss_sampling(nusd, nus1, f1_ibased, beta_inst, 0., vr_array)
f1_fbased_obs = response.ipgauss_sampling(nusd, nus1, f1_fbased, beta_inst, 0., vr_array)
if len(nus2) % 2 == 1:
    nus2 = nus2[1:]
    f2 = f2[1:]
f2_obs = response.ipgauss_sampling(nusd, nus2, f2, beta_inst, 0., vr_array)



import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40,8), gridspec_kw={'height_ratios': [3, 1]})
plt.subplots_adjust(hspace=0)
norm = np.median(f1_ibased_obs)
f1_ibased_obs = f1_ibased_obs / norm
f1_fbased_obs = f1_fbased_obs / norm
f2_obs = f2_obs / norm
ax1.plot(1.0e8/nusd, f1_ibased_obs, label="ExoJAX, intensity-based", c='C0')
ax1.plot(1.0e8/nusd, f1_fbased_obs, label="ExoJAX, flux-based", c='C1')
ax1.plot(1.0e8/nusd, f2_obs, label="petitRADTRANS", c='C2')

ax2.plot(1.0e8/nusd, f1_ibased_obs - f2_obs, "+", color="C0", markersize=5)
#ax2.plot(1.0e8/nusd, f1_fbased_obs - f2_obs, "+", color="C1", markersize=5)

ax1.set_ylabel("Normalized Flux", fontsize=15)
ax2.set_xlabel("Wavelength [$\AA$]", fontsize=15)
ax2.set_ylabel("Residual", fontsize=15)

ax1.set_xlim(np.min(1.0e8/nusd), np.max(1.0e8/nusd))
ax2.set_xlim(np.min(1.0e8/nusd), np.max(1.0e8/nusd))

ax1.xaxis.set_ticks_position('both')
ax1.yaxis.set_ticks_position('both')
ax2.xaxis.set_ticks_position('both')
ax2.yaxis.set_ticks_position('both')

ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())

ax2.patch.set_alpha(0)
ax1.tick_params(labelbottom=False, labeltop=True)

ax1.legend()
#plt.show()
plt.savefig("output/wide_R7000.png", bbox_inches='tight', dpi=300)

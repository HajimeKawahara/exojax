"""test for psudo line generator (plg) on CH4 lines.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from exojax.spec import plg
from exojax.spec import moldb, contdb, molinfo
from exojax.spec.setrt import gen_wavenumber_grid
from exojax.spec import initspec
import time
import copy

def test_plg_methane():
    #Parameter to make elower grid
    Nelower = 7
    #Ncrit = 0 #10
    Tgue = 1300.
    errTgue = 500.

    wls, wll, nugrid_res = 16448.0, 16452.0, 0.05
    nus, wav, reso = gen_wavenumber_grid(wls, wll, int((wll-wls)/nugrid_res), unit="AA", xsmode="modit")
    mdb_orig = moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/', nus, \
                               crit=0, Ttyp=Tgue)
    mdb = copy.deepcopy(mdb_orig)
    cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)
    molmassCH4=molinfo.molmass("CH4")
    print("Nline=",len(mdb_orig.A))

    #To save computation time, let us use only the middle 1 Å width to optimize coefTgue
    assess_width = 1. #Note that too narrow (e.g., 1. Å) might cause a ValueError. (#Cause unspecified...)
    nusc, wavc, resoc = gen_wavenumber_grid( (wll+wls-assess_width)/2, (wll+wls+assess_width)/2, 20, unit="AA", xsmode="modit")
    mdbc = moldb.MdbExomol('.database/CH4/12C-1H4/YT10to10/', nusc)

    ts = time.time()
    coefTgue = plg.optimize_coefTgue(Tgue, nusc, mdbc, molmassCH4, Nelower, errTgue)
    te = time.time()
    print(te-ts, "sec for", len(mdbc.A), "lines,  coefTgue =", coefTgue)

    #Initialization of modit
    cnu,indexnu,R,pmarray=initspec.init_modit(mdb.nu_lines,nus)
    cnu_orig = copy.deepcopy(cnu)
    indexnu_orig = copy.deepcopy(indexnu)

    #make grid of gamma and index_gamma
    gammaL_set=mdb.alpha_ref+mdb.n_Texp*(1j) #complex value
    gammaL_set_unique=np.unique(gammaL_set)
    Ngamma=np.shape(gammaL_set_unique)[0]
    index_gamma=np.zeros_like(mdb.alpha_ref,dtype=int)
    alpha_ref_grid=gammaL_set_unique.real
    n_Texp_grid=gammaL_set_unique.imag
    for j,a in enumerate(gammaL_set_unique):
        index_gamma=np.where(gammaL_set==a,j,index_gamma)

    ts = time.time()
    qlogsij0, qcnu, num_unique, elower_grid, frozen_mask, nonzeropl_mask = plg.plg_elower_addcon(\
        index_gamma, Ngamma, cnu, indexnu, nus, mdb, Tgue, errTgue, \
        Nelower=Nelower, reshape=False, coefTgue=coefTgue)
    te = time.time()
    print(te-ts, "sec")

    Nnugrid = len(nus)
    mdb, cnu, indexnu = plg.gather_lines(mdb, Ngamma, Nnugrid, Nelower, nus, cnu, indexnu, qlogsij0, qcnu, elower_grid, alpha_ref_grid, n_Texp_grid, frozen_mask, nonzeropl_mask)

    from exojax.spec import rtransfer as rt
    from exojax.spec import modit
    from exojax.spec.modit import set_ditgrid_matrix_exomol
    from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA
    from exojax.spec import planck, response

    #atm
    NP=100
    Parr, dParr, k=rt.pressure_layer(NP=NP)
    mmw=2.33 #mean molecular weight
    mmrH2=0.74
    molmassH2=molinfo.molmass("H2")
    vmrH2=(mmrH2*mmw/molmassH2) #VMR

    Pref=1.0 #bar
    ONEARR=np.ones_like(Parr)

    # Precomputing gdm_ngammaL
    fT = lambda T0,alpha: T0[:,None]*(Parr[None,:]/Pref)**alpha[:,None]
    T0_test=np.array([Tgue-300.,Tgue+300.,Tgue-300.,Tgue+300.])
    alpha_test=np.array([0.2,0.2,0.05,0.05])
    res=0.2
    dgm_ngammaL=set_ditgrid_matrix_exomol(mdb,fT,Parr,R,molmassCH4,res,T0_test,alpha_test)
    dgm_ngammaL_orig = set_ditgrid_matrix_exomol(mdb_orig, fT, Parr, R, molmassCH4, res, T0_test, alpha_test)

    #a core driver
    def frun(Tarr,MMR_,Mp,Rp,u1,u2,RV,vsini):
        g=2478.57730044555*Mp/Rp**2
        SijM_,ngammaLM_,nsigmaDl_=modit.exomol(mdb,Tarr,Parr,R,molmassCH4)
        xsm_=modit.xsmatrix(cnu,indexnu,R,pmarray,nsigmaDl_,ngammaLM_,SijM_,nus,dgm_ngammaL)
        #abs is used to remove negative values in xsv
        dtaum=dtauM(dParr,jnp.abs(xsm_),MMR_*ONEARR,molmassCH4,g)
        #CIA
        dtaucH2H2=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrH2,mmw,g,cdbH2H2.nucia,cdbH2H2.tcia,cdbH2H2.logac)
        dtau=dtaum+dtaucH2H2
        sourcef = planck.piBarr(Tarr,nus)
        F0=rtrun(dtau,sourcef)
        Frot=response.rigidrot(nus,F0,vsini,u1,u2)
        #mu=response.ipgauss_sampling(nusd,nus,Frot,beta_inst,RV)
        mu=Frot
        return mu
    def frun_orig(Tarr, MMR_, Mp, Rp, u1, u2, RV, vsini):
        g = 2478.57730044555*Mp/Rp**2
        SijM_, ngammaLM_, nsigmaDl_ = modit.exomol(mdb_orig, Tarr, Parr, R, molmassCH4)
        xsm_ = modit.xsmatrix(cnu_orig, indexnu_orig, R, pmarray, nsigmaDl_, ngammaLM_, SijM_, nus, dgm_ngammaL_orig)
        #abs is used to remove negative values in xsv
        dtaum = dtauM(dParr, jnp.abs(xsm_), MMR_*ONEARR, molmassCH4, g)
        #CIA
        dtaucH2H2 = dtauCIA(nus, Tarr, Parr, dParr, vmrH2, vmrH2, mmw, g, cdbH2H2.nucia, cdbH2H2.tcia, cdbH2H2.logac)
        dtau = dtaum+dtaucH2H2
        sourcef  =  planck.piBarr(Tarr, nus)
        F0 = rtrun(dtau, sourcef)
        Frot = response.rigidrot(nus, F0, vsini, u1, u2)
        #mu = response.ipgauss_sampling(nusd, nus, Frot, beta_inst, RV)
        mu = Frot
        return mu

    #Example with Luhman 16 A parameters (from Kawahara+2022)
    MpBd = 33.2
    RpBd = 0.89
    MMR_rough = 0.0059 #(referring to MMR_CO)
    Tarr = Tgue*(Parr/Pref)**0.1
    mu=frun(Tarr,MMR_=MMR_rough,Mp=MpBd,Rp=RpBd,u1=0.0,u2=0.0,RV=10.0,vsini=0.1)
    mu_orig = frun_orig(Tarr,MMR_=MMR_rough,Mp=MpBd,Rp=RpBd,u1=0.0,u2=0.0,RV=10.0,vsini=0.1)
    assert abs(np.mean(mu/mu_orig)-1) < 0.01

if __name__ == '__main__':
    test_plg_methane()

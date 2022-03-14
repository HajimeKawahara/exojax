import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from exojax.spec.lpf import xsmatrix
from exojax.spec.exomol import gamma_exomol
from exojax.spec.hitran import SijT, doppler_sigma, gamma_natural, gamma_hitran
from exojax.spec.hitrancia import read_cia, logacia
from exojax.spec.rtransfer import rtrun, dtauM, dtauCIA, nugrid
from exojax.spec import planck, response
from exojax.spec.lpf import xsvector
from exojax.spec import molinfo
from exojax.utils.constants import RJ, pc, Rs, c
from exojax.spec import rtransfer as rt
from exojax.spec import moldb, contdb
from exojax.spec import make_numatrix0
from exojax.spec import initspec
import jax.numpy as jnp
from jax import vmap, jit

def jaxopt_spectrum_test():

    dat=pd.read_csv("spectrum.txt",delimiter=",",names=("wav","flux"))
    wavd=dat["wav"].values
    flux=dat["flux"].values
    nusd=jnp.array(1.e8/wavd[::-1])
    sigmain=0.05
    norm=40000
    nflux=flux/norm+np.random.normal(0,sigmain,len(wavd))
    
    NP=100
    Parr, dParr, k=rt.pressure_layer(NP=NP)
    Nx=1500
    nus,wav,res=nugrid(np.min(wavd)-5.0,np.max(wavd)+5.0,Nx,unit="AA")
    
    R=100000.
    beta=c/(2.0*np.sqrt(2.0*np.log(2.0))*R)
    
    molmassCO=molinfo.molmass("CO")
    mmw=2.33 #mean molecular weight
    mmrH2=0.74
    molmassH2=molinfo.molmass("H2")
    vmrH2=(mmrH2*mmw/molmassH2) #VMR
    
    Mp = 33.2 #fixing mass...
    mdbCO=moldb.MdbExomol('.database/CO/12C-16O/Li2015',nus,crit=1.e-46)
    cdbH2H2=contdb.CdbCIA('.database/H2-H2_2011.cia',nus)
    numatrix_CO=make_numatrix0(nus,mdbCO.nu_lines)
    numatrix_CO=initspec.init_lpf(mdbCO.nu_lines,nus)
    Pref=1.0 #bar
    ONEARR=np.ones_like(Parr)
    ONEWAV=jnp.ones_like(nflux)

    def model_c(params,boost,nu1):
        Rp,RV,MMR_CO,T0,alpha,vsini=params*boost
        g=2478.57730044555*Mp/Rp**2 #gravity                                        
        u1=0.0
        u2=0.0
        #T-P model//                                                                
        Tarr = T0*(Parr/Pref)**alpha
        
        #line computation CO                                                        
        qt_CO=vmap(mdbCO.qr_interp)(Tarr)
        
        def obyo(nusd,nus,numatrix_CO,mdbCO,cdbH2H2):
            #CO                                                                     
            SijM_CO=jit(vmap(SijT,(0,None,None,None,0)))            (Tarr,mdbCO.logsij0,mdbCO.dev_nu_lines,mdbCO.elower,qt_CO)
            gammaLMP_CO = jit(vmap(gamma_exomol,(0,0,None,None)))            (Parr,Tarr,mdbCO.n_Texp,mdbCO.alpha_ref)
            gammaLMN_CO=gamma_natural(mdbCO.A)
            gammaLM_CO=gammaLMP_CO+gammaLMN_CO[None,:]
            
            sigmaDM_CO=jit(vmap(doppler_sigma,(None,0,None)))            (mdbCO.dev_nu_lines,Tarr,molmassCO)
            xsm_CO=xsmatrix(numatrix_CO,sigmaDM_CO,gammaLM_CO,SijM_CO)
            dtaumCO=dtauM(dParr,xsm_CO,MMR_CO*ONEARR,molmassCO,g)
            #CIA                                                                    
            dtaucH2H2=dtauCIA(nus,Tarr,Parr,dParr,vmrH2,vmrH2,                          mmw,g,cdbH2H2.nucia,cdbH2H2.tcia,cdbH2H2.logac)
            dtau=dtaumCO+dtaucH2H2
            sourcef = planck.piBarr(Tarr,nus)
            F0=rtrun(dtau,sourcef)/norm
            
            Frot=response.rigidrot(nus,F0,vsini,u1,u2)
            #Frot=rigidrotx(nus,F0,vsini,u1,u2)
            mu=response.ipgauss_sampling(nusd,nus,Frot,beta,RV)
            return mu
        
        model=obyo(nu1,nus,numatrix_CO,mdbCO,cdbH2H2)
        return model

    import jaxopt
    boost=np.array([1.0,10.0,0.1,1000.0,1.e-3,10.0])
    initpar=np.array([0.8,9.0,0.1,1200.0,0.1,17.0])/boost
    def objective(params):
        f=nflux-model_c(params,boost,nusd)
        g=jnp.dot(f,f)
        return g

    gd = jaxopt.GradientDescent(fun=objective, maxiter=1000,stepsize=1.e-4)
    res = gd.run(init_params=initpar)
    params, state = res

    print(np.sum(params*boost))

if __name__=="__main__":
    jaxopt_spectrum_test()

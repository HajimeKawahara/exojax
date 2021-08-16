import numpy as np
import tqdm
import jax.numpy as jnp
from jax import vmap
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('bmh')
from exojax.spec import make_numatrix0
from exojax.spec.lpf import xsvector as lpf_xsvector
from exojax.spec.modit import xsvector as modit_xsvector
from exojax.spec import initspec
from exojax.spec import xsection as lpf_xsection
from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
from exojax.spec import rtcheck, moldb
from exojax.spec.dit import set_ditgrid
from exojax.spec.hitran import normalized_doppler_sigma


def comperr(Nnu,plotfig=False):

    nus=np.logspace(np.log10(1.e7/2700),np.log10(1.e7/2100.),Nnu,dtype=np.float64)

#    nus=np.logspace(np.log10(3000),np.log10(6000.0),Nnu,dtype=np.float64)
    mdbCO=moldb.MdbHit('/home/kawahara/exojax/data/CO/05_hit12.par',nus)
    
    Mmol=28.010446441149536
    Tref=296.0
    Tfix=1000.0
    Pfix=1.e-3 #
    
    #USE TIPS partition function
    Q296=np.array([107.25937215917970,224.38496958496091,112.61710362499998,\
                   660.22969049609367,236.14433662109374,1382.8672147421873])
    Q1000=np.array([382.19096582031250,802.30952197265628,402.80326733398437,\
                    2357.1041210937501,847.84866308593757,4928.7215078125000])
    qr=Q1000/Q296
    
    qt=np.ones_like(mdbCO.isoid,dtype=np.float64)
    for idx,iso in enumerate(mdbCO.uniqiso):
        mask=mdbCO.isoid==iso
        qt[mask]=qr[idx]
        
    Sij=SijT(Tfix,mdbCO.logsij0,mdbCO.nu_lines,mdbCO.elower,qt)
    gammaL = gamma_hitran(Pfix,Tfix,Pfix, mdbCO.n_air, mdbCO.gamma_air, mdbCO.gamma_self)
    #+ gamma_natural(A) #uncomment if you inclide a natural width
    sigmaD=doppler_sigma(mdbCO.nu_lines,Tfix,Mmol)
    
    cnu,indexnu,R,dLarray=initspec.init_modit(mdbCO.nu_lines,nus)
    nsigmaD=normalized_doppler_sigma(Tfix,Mmol,R)
    ngammaL=gammaL/(mdbCO.nu_lines/R)
    ngammaL_grid=set_ditgrid(ngammaL)
    
    xs_modit_lp=modit_xsvector(cnu,indexnu,R,dLarray,nsigmaD,ngammaL,Sij,nus,ngammaL_grid)
    wls_modit = 100000000/nus
    
    #ref (direct)
    d=10000
    ll=mdbCO.nu_lines
    xsv_lpf_lp=lpf_xsection(nus,ll,sigmaD,gammaL,Sij,memory_size=30)


    dif=xs_modit_lp/xsv_lpf_lp-1.
    med=np.median(dif)
    iju=22940.
    ijd=26400.
    limu,limd=np.searchsorted(wls_modit[::-1],[iju,ijd])
    std=np.std(dif[::-1][limu:limd])
    
    return med,std,R,ijd,iju,wls_modit,xs_modit_lp,xsv_lpf_lp,dif


if __name__=="__main__":
    import matplotlib
    m,std,R,ijd,iju,wls_modit,xs_modit_lp,xsv_lpf_lp,dif=comperr(200000)
    m1,std1,R1,ijd1,iju1,wls_modit1,xs_modit_lp1,xsv_lpf_lp1,dif1=comperr(3000000)

    print(m,std,R)
    print(m1,std1,R1)



    
    #PLOT
    plotfig=True
    if plotfig:
        matplotlib.rcParams['agg.path.chunksize'] = 100000
        llow=2300.4
        lhigh=2300.7
        tip=2.0
        fig=plt.figure(figsize=(12,3))
        ax=plt.subplot2grid((12, 1), (0, 0),rowspan=8)
        plt.plot(wls_modit1,xsv_lpf_lp1,label="Direct",color="C0",alpha=0.3,markersize=3)
        plt.plot(wls_modit,xs_modit_lp,color="C1",lw=1,alpha=0.3,label="R="+str(R))
        plt.plot(wls_modit1,xs_modit_lp1,color="C2",lw=1,alpha=0.5,label="R="+str(R1),ls="dashed")

        plt.xlim(ijd,iju)    
#        plt.xlim(llow*10-tip,lhigh*10+tip)    

        plt.ylim(1.1e-35,1.e-17)
#        plt.ylim(1.e-27,3.e-17)
        plt.yscale("log")
        
#        plt.xlim(llow*10.0-tip,lhigh*10.0+tip)
        plt.legend(loc="upper right")
        plt.ylabel("       cross section",fontsize=10)
        #plt.text(22986,3.e-21,"$P=10^{-3}$ bar")
        plt.xlabel('wavelength [$\AA$]')
        
        ax=plt.subplot2grid((12, 1), (8, 0),rowspan=4)
        plt.plot(wls_modit,np.abs((dif)*100),alpha=0.2,color="C1",label="R="+str(R))
        plt.plot(wls_modit1,np.abs((dif1)*100),alpha=0.5,color="C2",label="R="+str(R1))
        
        plt.ylabel("difference (%)",fontsize=10)
        plt.xlim(ijd,iju)
#        plt.xlim(llow*10-tip,lhigh*10+tip)    

        plt.ylim(0.1,10000.0)
#        plt.ylim(-10*100*std,10*100*std)
        plt.yscale("log")
        plt.xlabel('wavelength [$\AA$]')
        plt.legend(loc="upper left")
        
        plt.savefig("fig/comparison_modit.png", bbox_inches="tight", pad_inches=0.0)
        plt.savefig("fig/comparison_modit.pdf", bbox_inches="tight", pad_inches=0.0)
        plt.show()

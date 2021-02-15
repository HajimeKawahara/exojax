"""plotting tool for atmospheric structure

   * --

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plottau(nus,dtauM,Tarr=None,Parr=None,unit=None,mode=None,vmin=-3,vmax=3):
    """plot tau

       This function gives the color map of log10(tau) (or log10(dtau)), optionally w/ a T-P profile.

    Args:
       nus: wavenumber
       dtauM: dtau matrix
       Tarr: temperature profile
       Parr: perssure profile 
       unit: x-axis unit=um (wavelength microns), nm  = (wavelength nm), AA  = (wavelength Angstrom), 
       mode: mode=None (lotting tau), mode=dtau (plotting delta tau for each layer)
       vmin: color value min (default=-3)
       vmax: color value max (default=3)
    """
    if mode=="dtau":
        ltau=np.log10(dtauM)
    else:
        ltau=np.log10(np.cumsum(dtauM,axis=0))
        
    fig=plt.figure(figsize=(20,3))
    ax=plt.subplot2grid((1, 20), (0, 3),colspan=18)
    if unit=="um":
        c=ax.imshow(ltau[:,::-1],vmin=vmin,vmax=vmax,cmap="RdYlBu_r",alpha=0.9,extent=[1.e4/nus[-1],1.e4/nus[0],np.log10(Parr[-1]),np.log10(Parr[0])])
        plt.xlabel("wavelength ($\mu \mathrm{m}$)")
    elif unit=="nm":
        c=ax.imshow(ltau[:,::-1],vmin=vmin,vmax=vmax,cmap="RdYlBu_r",alpha=0.9,extent=[1.e7/nus[-1],1.e7/nus[0],np.log10(Parr[-1]),np.log10(Parr[0])])
        plt.xlabel("wavelength (nm)")
    elif unit=="AA":
        c=ax.imshow(ltau[:,::-1],vmin=vmin,vmax=vmax,cmap="RdYlBu_r",alpha=0.9,extent=[1.e8/nus[-1],1.e8/nus[0],np.log10(Parr[-1]),np.log10(Parr[0])])
        plt.xlabel("wavelength ($\AA$)")
    else:
        c=ax.imshow(ltau,vmin=vmin,vmax=vmax,cmap="RdYlBu_r",alpha=0.9\
           ,extent=[nus[0],nus[-1],np.log10(Parr[-1]),np.log10(Parr[0])])
        plt.xlabel("wavenumber ($\mathrm{cm}^{-1}$)")
        
    plt.colorbar(c,shrink=0.8)
    plt.ylabel("log10 (P (bar))")
    ax.set_aspect(0.2/ax.get_data_ratio())
    if Tarr is not None and Parr is not None:
        ax=plt.subplot2grid((1, 20), (0, 0),colspan=2)
        plt.plot(Tarr,np.log10(Parr),color="gray")
        plt.xlabel("temperature (K)")
        plt.ylabel("log10 (P (bar))")
        plt.gca().invert_yaxis()
        plt.ylim(np.log10(Parr[-1]),np.log10(Parr[0]))
        ax.set_aspect(1.45/ax.get_data_ratio())


def plotcf(nus,dtauM,Tarr,Parr,dParr,unit=None,cmap="magma"):
    from exojax.spec.planck import piBarr
    """plot contribution function

       This function gives the color map of log10(cf), optionally w/ a T-P profile.

    Args:
       nus: wavenumber
       dtauM: dtau matrix
       Tarr: temperature profile
       Parr: perssure profile 
       dParr: perssure difference profile 
       unit: x-axis unit=um (wavelength microns), nm  = (wavelength nm), AA  = (wavelength Angstrom), 
       mode: mode=None (lotting tau), mode=dtau (plotting delta tau for each layer)
       cmap: colormap
    """

    hcperk=1.4387773538277202
    tau=np.cumsum(dtauM,axis=0)
    cf=np.exp(-tau)*dtauM*Parr[:,None]/dParr[:,None]
#nus**3/(np.exp(hcperk*nus/Tarr[:,None])-1.0)\

    
    fig=plt.figure(figsize=(20,3))
    ax=plt.subplot2grid((1, 20), (0, 3),colspan=18)
    if unit=="um":
        c=ax.imshow(cf[:,::-1],cmap=cmap,alpha=0.9,extent=[1.e4/nus[-1],1.e4/nus[0],np.log10(Parr[-1]),np.log10(Parr[0])])
        plt.xlabel("wavelength ($\mu \mathrm{m}$)")
    elif unit=="nm":
        c=ax.imshow(cf[:,::-1],cmap=cmap,alpha=0.9,extent=[1.e7/nus[-1],1.e7/nus[0],np.log10(Parr[-1]),np.log10(Parr[0])])
        plt.xlabel("wavelength (nm)")
    elif unit=="AA":
        c=ax.imshow(cf[:,::-1],cmap=cmap,alpha=0.9,extent=[1.e8/nus[-1],1.e8/nus[0],np.log10(Parr[-1]),np.log10(Parr[0])])
        plt.xlabel("wavelength ($\AA$)")
    else:
        c=ax.imshow(cf,cmap=cmap,alpha=0.9\
           ,extent=[nus[0],nus[-1],np.log10(Parr[-1]),np.log10(Parr[0])])
        plt.xlabel("wavenumber ($\mathrm{cm}^{-1}$)")
        
    plt.colorbar(c,shrink=0.8)
    plt.ylabel("log10 (P (bar))")
    ax.set_aspect(0.2/ax.get_data_ratio())
    if Tarr is not None and Parr is not None:
        ax=plt.subplot2grid((1, 20), (0, 0),colspan=2)
        plt.plot(Tarr,np.log10(Parr),color="gray")
        plt.xlabel("temperature (K)")
        plt.ylabel("log10 (P (bar))")
        plt.gca().invert_yaxis()
        plt.ylim(np.log10(Parr[-1]),np.log10(Parr[0]))
        ax.set_aspect(1.45/ax.get_data_ratio())

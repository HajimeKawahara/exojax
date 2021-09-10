"""Exojax File Format
   
   Note:
       EFF contains .json and main file (normally hdf5 for vaex)

   - emif: Exomol MODIT Input Format

"""
import pathlib
import vaex
import json

def save_exomol_mif(output,nu_grid,cnu,indexnu,logsij0,A,alpha_ref,n_Texp,elower,extension=".hdf5"):
    """Save inputs as MODIT Input File Format for ExoMol (emif)

    Args:

    """
    outpath=pathlib.Path(output)
    mainf=outpath.with_suffix(extension)
    #INFO
    info = {}
    info["exojax_file_format"]=[{"format":"emif","main_file":str(mainf)}]
    info["lines"]=[{"Nlines":str(len(cnu))}]
    info["wavenumber_grid"]=[
        {"start":str(nu_grid[0]),"end":str(nu_grid[-1]),"N":str(len(nu_grid))}
        ]
    with open(outpath.with_suffix(".json"), mode='wt', encoding='utf-8') as file:
        json.dump(info, file, ensure_ascii=False, indent=2)    
    #main data file
    mif=vaex.from_arrays(cnu=cnu
                         ,indexnu=indexnu
                         ,logsij0=logsij0
                         ,A=A
                         ,alpha_ref=alpha_ref
                         ,n_Texp=n_Texp
                         ,elower=elower
    )    
    mif.export(mainf)
    
    return None

if __name__ == "__main__":
    import numpy as np
    import jax.numpy as jnp
    from exojax.spec import initspec
    from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
    from exojax.spec import rtcheck, moldb
    from exojax.spec.dit import set_ditgrid

    nus=np.logspace(np.log10(4000),np.log10(4500.0),3000000,dtype=np.float64)
    mdb=moldb.MdbExomol('/home/kawahara/exojax/examples/bd/.database/CO/12C-16O/Li2015',nus)
    cnu,indexnu,R,pmarray=initspec.init_modit(mdb.nu_lines,nus)
    save_exomol_mif("CO.mif",nus,cnu,indexnu,mdb.logsij0,mdb.A,mdb.alpha_ref,mdb.n_Texp,mdb.elower)

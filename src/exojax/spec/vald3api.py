"""API for VALD 3

"""

import numpy as np
import pandas as pd

def read_ExStellar(stellarf):
    """VALD IO for "Extract stellar" file (long format)
    Note:
        See https://www.astro.uu.se/valdwiki/select_output

    Args:
        
    Returns:
        line data in pandas DataFrame
        
    """
    dat = pd.read_csv(stellarf, sep=",",skiprows=3,\
    names=("species","wav_lines","elowereV","vmic","loggf","rad_damping","stark_damping","waals_damping","lande","depth","reference"))
    return dat
    
    
def read_ExAll(allf):
    """VALD IO for "Extract all" file (long format)
    Note:
        See https://www.astro.uu.se/valdwiki/presformat_output

    Args:
        Elm Ion
        WL_vac(AA)
        log gf*
        E_low(eV):  lower excitation potential
        J lo:  lower rotational quantum number
        E_up(eV):  upper excitation potential
        J up:  upper rotational quantum number
        Lande lower
        Lande upper
        Lande mean
        Damping Rad.
        Damping Stark
        Damping Waals

    Returns:
        line data in pandas DataFrame
        
    """
    dat = pd.read_csv(allf, sep=",", skiprows=1, \
    names=("species","wav_lines","loggf","elowereV","jlower","euppereV", "jupper", "landelower", "landeupper", "landemean", "rad_damping","stark_damping","waals_damping")\
                     )

    dat = dat[dat.species.str.startswith("'")] #Remove rows not starting with "'"
    dat = dat[dat.species.str.startswith("' ").map({False: True, True: False})] #Remove rows of Reference
    dat = dat[dat.species.str.startswith("'_").map({False: True, True: False})] #Remove rows of Reference
    
    dat = dat[ dat.species.str.len()<7 ] #Remove long name (molecules e.g., TiO)
    dat = dat[dat.species.str[1:3].str.isupper().map({False: True, True: False})] #Remove names starting with successive uppercase letters (molecules e.g., CO, OH, CN)
    #dat = dat[dat.species.str.startswith("'CO").map({False: True, True: False})]
    dat = dat.reset_index(drop=True)
    
    dat = dat.astype({'wav_lines': 'float64', 'loggf': 'float64', 'elowereV': 'float64', 'jlower': 'float64', 'euppereV': 'float64'})

    return dat
    

def pickup_param(ExAll):
    """ extract transition parameters from VALD3 line list and insert the same DataFrame.
    
    Args:
        ExAll: VALD3 line list (DataFrame): Output of read_ExAll
    
    Returns:
        A:  Einstein coefficient in [s-1]
        nu_lines:  transition waveNUMBER in [cm-1] (#NOT frequency in [s-1])
        elower: lower excitation potential [cm-1] (#converted from eV)
        gupper: upper statistical weight
        jlower: lower J (rotational quantum number, total angular momentum)
        jupper: upper J
        ielem:  atomic number (e.g., Fe=26)
        iion:  ionized level (e.g., neutral=1, singly)
        vdWdamp:  van der Waals damping parameters
        gamRad: gamma(HWHM) of radiation damping
       
    Note:
    
    """
    ccgs = 2.99792458e10 #[cm/s]
    ecgs = 4.80320450e-10 #[esu]=[dyn^0.5*cm] !elementary charge
    mecgs  = 9.10938356e-28 #[g] !electron mass
    
    # Correspondence between Atomic Number and Element Symbol
    PeriodicTable = np.zeros([119], dtype=object) #PeriodicTable = np.empty([100], dtype=object)
    PeriodicTable[:] = [' 0', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    # insert new columns in VALD line list
    ExAll["nu_lines"] = 1.e8 / ExAll["wav_lines"] #[cm-1]<-[AA]
    ExAll["elower"] = ExAll["elowereV"] * 8065.541
    ExAll["eupper"] = ExAll["euppereV"] * 8065.541
    ExAll["gupper"] = ExAll["jupper"]*2+1
    ExAll["glower"] = ExAll["jlower"]*2+1
    #notes4Tako#どうせ比をとるので電子の縮退度等の係数は落ちる.(MyLog2017.rtf)
    ExAll["A"] = 10**ExAll["loggf"] / ExAll["gupper"] * (ccgs*ExAll["nu_lines"])**2 \
        *8*np.pi**2*ecgs**2 / (mecgs*ccgs**3)
    
    A=ExAll["A"].to_numpy()
    nu_lines=ExAll["nu_lines"].to_numpy()
    elower=ExAll["elower"].to_numpy()
    gupper=ExAll["gupper"].to_numpy()
    jlower=ExAll["jlower"].to_numpy()
    jupper=ExAll["jupper"].to_numpy()
    vdWdamp=ExAll["waals_damping"].to_numpy()
    gamRad=ExAll["rad_damping"].to_numpy()

    ielem = np.zeros(len(ExAll), dtype='int') #atomic number (e.g., Fe=26)
    iion = np.zeros(len(ExAll), dtype='int') #e.g., neutral=1, singly ionized=2, ...
    for i, sp in enumerate(ExAll['species']):
        symbol = sp.strip("'").split(' ')[0]
        ielem[i] = np.where(PeriodicTable==symbol)[0][0] if (symbol in PeriodicTable) else 0
        iion[i] = int(sp.strip("'").split(' ')[-1])

    return A, nu_lines, elower, gupper, jlower, jupper, ielem, iion, vdWdamp, gamRad

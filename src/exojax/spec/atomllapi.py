"""API for VALD 3."""

import numpy as np
import pandas as pd
from exojax.utils.constants import ccgs, ecgs, mecgs, eV2wn
import io
import vaex
import pkgutil
from io import BytesIO

# Correspondence between Atomic Number and Element Symbol
PeriodicTable = np.zeros([119], dtype=object)
PeriodicTable[:] = [' 0', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
                    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']


def read_ExStellar(stellarf):
    """VALD IO for "Extract stellar" file (long format)

    Note:
        Deprecated to use! It's incomplete and untested. #210817
        See https://www.astro.uu.se/valdwiki/select_output

    Args:
        stellarf: file path

    Returns:
        line data in pandas DataFrame
    """
    dat = pd.read_csv(stellarf, sep=',', skiprows=3, names=('species', 'wav_lines', 'elowereV', 'vmic',
                      'loggf', 'rad_damping', 'stark_damping', 'waals_damping', 'lande', 'depth', 'reference'))
    return dat


def read_ExAll(allf):
    """IO for linelists downloaded from VALD3 with a query of "Long format" in
    the format of "Extract All" or "Extract Element".

    Note:
        About input linelists obtained from VALD3 (http://vald.astro.uu.se/):
            VALD data access is free but requires registration through the Contact form (http://vald.astro.uu.se/~vald/php/vald.php?docpage=contact.html).
        After the registration, you can login and choose the "Extract Element" mode.
        For example, if you want the Fe I linelist, the request form should be filled as:
            Starting wavelength :    1500
            Ending wavelength :    100000
            Element [ + ionization ] :    Fe 1
            Extraction format :    Long format
            Retrieve data via :    FTP
            Linelist configuration :    Default
            Unit selection:    Energy unit: eV - Medium: vacuum - Wavelength unit: angstrom - VdW syntax: default
        Please assign the fullpath of the output file sent by VALD ([user_name_at_VALD].[request_number_at_VALD].gz) to the variable "allf".
        See https://www.astro.uu.se/valdwiki/presformat_output for the detail of the format.

    Args:
        allf: fullpath to the input VALD linelist (See Notes above for more details.)
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
        line data in vaex DataFrame
    """
    dat = pd.read_csv(allf, sep=',', skiprows=1, names=('species', 'wav_lines', 'loggf', 'elowereV', 'jlower', 'euppereV',
                      'jupper', 'landelower', 'landeupper', 'landemean', 'rad_damping', 'stark_damping', 'waals_damping'))  # convert=False)
    colWL = dat.iat[0, 0][13:22]

    # Remove rows not starting with "'"
    dat = dat[dat.species.str.startswith("'")]
    dat = dat[dat.species.str.startswith("' ").map(
        {False: True, True: False})]  # Remove rows of Reference
    dat = dat[dat.species.str.startswith("'_").map(
        {False: True, True: False})]  # Remove rows of Reference

    # Remove long name (molecules e.g., TiO)
    dat = dat[dat.species.str.len() < 7]
    # Remove names starting with successive uppercase letters (molecules e.g., CO, OH, CN)
    dat = dat[dat.species.str.slice(start=1, stop=3).str.isupper().map({
        False: True, True: False})]
    # Remove highly ionized ions (iion > 3, for which the partition function is not reported in Barklem+2016)
    dat = dat.drop(dat.index[np.where(np.array(
        list(map(lambda x: int(x[1].strip("'")), dat.species.str.split(' ')))) > 3)[0]])
    for i, sp in enumerate(dat.species):
        symbol = sp.strip("'").split(' ')[0]
        ielem = np.where(PeriodicTable == symbol)[
            0][0] if (symbol in PeriodicTable) else 0
        iion = int(sp.strip("'").split(' ')[-1])
        dat.species.values[i] = ielem*100+iion-1
    dat = dat.reset_index(drop=True)
    dat = dat.astype('float64')
    #dat = dat.astype({'wav_lines': 'float64', 'loggf': 'float64', 'elowereV': 'float64', 'jlower': 'float64', 'euppereV': 'float64'})
    if colWL == 'WL_air(A)':
        # If wavelength is in air, it will be corrected (Note that wavelengths of transitions short of 2000 Angstroems are actually in vacuum and not in air.)
        dat.iloc[:, 1] = np.where(
            dat.iloc[:, 1] > 2000, air_to_vac(dat.iloc[:, 1]), dat.iloc[:, 1])
    dat = vaex.from_pandas(dat)
    dat.export_hdf5(allf.with_suffix('.hdf5'))

    return dat


def read_kurucz(kuruczf):
    """Input Kurucz line list (http://kurucz.harvard.edu/linelists/)

    Args:
        kuruczf: file path

    Returns:
        A:  Einstein coefficient in [s-1]
        nu_lines:  transition waveNUMBER in [cm-1] (#NOT frequency in [s-1])
        elower: lower excitation potential [cm-1] (#converted from eV)
        eupper: upper excitation potential [cm-1] (#converted from eV)
        gupper: upper statistical weight
        jlower: lower J (rotational quantum number, total angular momentum)
        jupper: upper J
        ielem:  atomic number (e.g., Fe=26)
        iion:  ionized level (e.g., neutral=1, singly)
        gamRad: log of gamma of radiation damping (s-1) #(https://www.astro.uu.se/valdwiki/Vald3Format)
        gamSta: log of gamma of Stark damping (s-1)
        gamvdW:  log of (van der Waals damping constant / neutral hydrogen number) (s-1)
    """
    with open(kuruczf) as f:
        lines = f.readlines()
    wlnmair, loggf, species, elower, jlower, labellower, eupper, jupper, labelupper, \
        gamRad, gamSta, gamvdW, ref, \
        NLTElower, NLTEupper, isonum, hyperfrac, isonumdi, isofrac, \
        hypershiftlower, hypershiftupper, hyperFlower, hypernotelower, hyperFupper, hypternoteupper, \
        strenclass, auto, landeglower, landegupper, isoshiftmA \
        = \
        np.zeros(len(lines)), np.zeros(len(lines)), np.array(['']*len(lines), dtype=object), np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), \
        np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), \
        np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), \
        np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(len(lines)), \
        np.zeros(len(lines)), np.zeros(len(lines)), np.zeros(
            len(lines)), np.zeros(len(lines)), np.zeros(len(lines))
    ielem, iion = np.zeros(len(lines), dtype=int), np.zeros(
        len(lines), dtype=int)

    for i, line in enumerate(lines):
        wlnmair[i] = float(line[0:11])
        loggf[i] = float(line[11:18])
        species[i] = str(line[18:24])
        ielem[i] = int(species[i].split('.')[0])
        iion[i] = int(species[i].split('.')[1])+1
        elower[i] = float(line[24:36])
        jlower[i] = float(line[36:41])
        eupper[i] = float(line[52:64])
        jupper[i] = float(line[64:69])
        gamRad[i] = float(line[80:86])
        gamSta[i] = float(line[86:92])
        gamvdW[i] = float(line[92:98])

    elower_inverted = np.where((eupper-elower) > 0,  elower,  eupper)
    eupper_inverted = np.where((eupper-elower) > 0,  eupper,  elower)
    jlower_inverted = np.where((eupper-elower) > 0,  jlower,  jupper)
    jupper_inverted = np.where((eupper-elower) > 0,  jupper,  jlower)
    elower = elower_inverted
    eupper = eupper_inverted
    jlower = jlower_inverted
    jupper = jupper_inverted

    wlaa = np.where(wlnmair < 200, wlnmair*10, air_to_vac(wlnmair*10))
    nu_lines = 1e8 / wlaa[::-1]  # [cm-1]<-[AA]
    loggf = loggf[::-1]
    ielem = ielem[::-1]
    iion = iion[::-1]
    elower = elower[::-1]
    eupper = eupper[::-1]
    jlower = jlower[::-1]
    jupper = jupper[::-1]
    glower = jlower*2+1
    gupper = jupper*2+1
    A = 10**loggf / gupper * (ccgs*nu_lines)**2 \
        * (8*np.pi**2*ecgs**2) / (mecgs*ccgs**3)
    gamRad = gamRad[::-1]
    gamSta = gamSta[::-1]
    gamvdW = gamvdW[::-1]

    return A, nu_lines, elower, eupper, gupper, jlower, jupper, ielem, iion, gamRad, gamSta, gamvdW


def pickup_param(ExAll):
    """extract transition parameters from VALD3 line list and insert the same
    DataFrame.

    Args:
        ExAll: VALD3 line list as pandas DataFrame (Output of read_ExAll)

    Returns:
        A:  Einstein coefficient in [s-1]
        nu_lines:  transition waveNUMBER in [cm-1] (#NOT frequency in [s-1])
        elower: lower excitation potential [cm-1] (#converted from eV)
        eupper: upper excitation potential [cm-1] (#converted from eV)
        gupper: upper statistical weight
        jlower: lower J (rotational quantum number, total angular momentum)
        jupper: upper J
        ielem:  atomic number (e.g., Fe=26)
        iion:  ionized level (e.g., neutral=1, singly)
        gamRad: log of gamma of radiation damping (s-1) #(https://www.astro.uu.se/valdwiki/Vald3Format)
        gamSta: log of gamma of Stark damping (s-1)
        vdWdamp:  log of (van der Waals damping constant / neutral hydrogen number) (s-1)
    """
    # insert new columns in VALD line list
    ExAll = ExAll.astype({'wav_lines': 'float64', 'loggf': 'float64',
                         'elowereV': 'float64', 'jlower': 'float64', 'euppereV': 'float64'})
    ExAll['nu_lines'] = 1.e8 / ExAll['wav_lines']  # [cm-1]<-[AA]
    ExAll = ExAll.iloc[::-1].reset_index(drop=True)  # Sort by wavenumber
    ExAll['elower'] = ExAll['elowereV'] * eV2wn
    ExAll['eupper'] = ExAll['euppereV'] * eV2wn
    ExAll['gupper'] = ExAll['jupper']*2+1
    ExAll['glower'] = ExAll['jlower']*2+1
    # notes4Tako#どうせ比をとるので電子の縮退度等の係数は落ちる.(MyLog2017.rtf)
    ExAll['A'] = 10**ExAll['loggf'] / ExAll['gupper'] * (ccgs*ExAll['nu_lines'])**2 \
        * (8*np.pi**2*ecgs**2) / (mecgs*ccgs**3)

    A = ExAll['A'].to_numpy()
    nu_lines = ExAll['nu_lines'].to_numpy()
    elower = ExAll['elower'].to_numpy()
    eupper = ExAll['eupper'].to_numpy()
    gupper = ExAll['gupper'].to_numpy()
    jlower = ExAll['jlower'].to_numpy()
    jupper = ExAll['jupper'].to_numpy()
    gamRad = ExAll['rad_damping'].to_numpy()
    gamSta = ExAll['stark_damping'].to_numpy()
    vdWdamp = ExAll['waals_damping'].to_numpy()

    ielem = np.zeros(len(ExAll), dtype='int')  # atomic number (e.g., Fe=26)
    # e.g., neutral=1, singly ionized=2, ...
    iion = np.zeros(len(ExAll), dtype='int')
    for i, sp in enumerate(ExAll['species']):
        ielem[i] = int(str(int(sp))[:2])
        iion[i] = int(str(int(sp))[2:])+1

    return A, nu_lines, elower, eupper, gupper, jlower, jupper, ielem, iion, gamRad, gamSta, vdWdamp


def vac_to_air(wlvac):
    """Convert wavelengths [AA] in vacuum into those in air.

    * See http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

    Args:
        wlvac:  wavelength in vacuum [Angstrom]
        n:  Refractive Index in dry air at 1 atm pressure and 15ºC with 0.045% CO2 by volume (Birch and Downs, 1994, Metrologia, 31, 315)

    Returns:
        wlair:  wavelengthe in air [Angstrom]
    """
    s = 1e4 / wlvac
    n = 1. + 0.0000834254 + 0.02406147 / \
        (130 - s*s) + 0.00015998 / (38.9 - s*s)
    wlair = wlvac / n
    return wlair


def air_to_vac(wlair):
    """Convert wavelengths [AA] in air into those in vacuum.

    * See http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

    Args:
        wlair:  wavelengthe in air [Angstrom]
        n:  Refractive Index in dry air at 1 atm pressure and 15ºC with 0.045% CO2 by volume (Birch and Downs, 1994, Metrologia, 31, 315)

    Returns:
        wlvac:  wavelength in vacuum [Angstrom]
    """
    s = 1e4 / wlair
    n = 1. + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s*s) + \
        0.0001599740894897 / (38.92568793293 - s*s)
    wlvac = wlair * n
    return wlvac


def load_atomicdata():
    """load atomic data and solar composition.

    * See  Asplund et al. 2009, Gerevesse et al. 1996

    Returns:
        ipccd (pd.DataFrame): table of atomic data

    Note:
        atomic.txt is in data/atom
    """
    ipccc = ('ielem', 'ionizationE1', 'dam1', 'dam2',
             'solarA', 'mass', 'ionizationE2')
    adata = pkgutil.get_data('exojax', 'data/atom/atomic.txt')
    ipccd = pd.read_csv(BytesIO(adata), sep='\s+', skiprows=1,
                        usecols=[1, 2, 3, 4, 5, 6, 7], names=ipccc)
    return ipccd


def make_ielem_to_index_of_ipccd():
    """index conversion for atomll.uspecies_info (Preparation for LPF)

    Returns:
        ielem_to_index_of_ipccd: jnp.array to convert ielem into index of ipccd
    """
    import jax.numpy as jnp
    ielemarr = jnp.array(load_atomicdata()['ielem'])

    # jnp.array for conversin from ielem to the index of ipccd
    ielem_to_index_of_ipccd = np.zeros(np.max(ielemarr)+1, dtype='int')
    for i in range(np.max(ielemarr)+1):
        if (i in ielemarr):
            ielem_to_index_of_ipccd[i] = np.where(ielemarr == i)[0][0]
    ielem_to_index_of_ipccd = jnp.array(ielem_to_index_of_ipccd)
    return ielem_to_index_of_ipccd


ielem_to_index_of_ipccd = make_ielem_to_index_of_ipccd()


def load_ionization_energies():
    """Load atomic ionization energies.

    Returns:
        df_ionE (pd.DataFrame): table of ionization energies

    Note:
        NIST_Atomic_Ionization_Energies.txt is in data/atom
    """
    fn_IonE = pkgutil.get_data(
        'exojax', 'data/atom/NIST_Atomic_Ionization_Energies.txt')
    df_ionE = pd.read_csv(BytesIO(fn_IonE), sep='|', skiprows=6, header=0)
    return df_ionE


def pick_ionE(ielem, iion, df_ionE):
    """Pick up ionization energy of a specific atomic species.

    Args:
        ielem (int): atomic number (e.g., Fe=26)
        iion (int): ionized level (e.g., neutral=1, singly ionized=2, etc.)
        df_ionE (pd.DataFrame): table of ionization energies

    Returns:
        ionE (float): ionization energy

    Note:
        NIST_Atomic_Ionization_Energies.txt is in data/atom
    """
    def f_droppare(x): return x.str.replace('(', '', regex=True).str.replace(')', '', regex=True).str.replace(
        '[', '', regex=True).str.replace(']', '', regex=True).str.replace('                                      ', '0', regex=True)
    ionE = float(f_droppare(df_ionE[(df_ionE['At. num '] == ielem) & (
        df_ionE[' Ion Charge '] == iion-1)]['      Ionization Energy (a) (eV)      ']))
    return ionE


def load_pf_Barklem2016():
    """load a table of the partition functions for 284 atomic species.

    * See Table 8 of Barklem & Collet (2016); https://doi.org/10.1051/0004-6361/201526961

    Returns:
        pfTdat (pd.DataFrame): steps of temperature (K)
        pfdat (pd.DataFrame): partition functions for 284 atomic species
    """
    pfT_str = 'T[K]   1.00000e-05   1.00000e-04   1.00000e-03   1.00000e-02   1.00000e-01   1.50000e-01   2.00000e-01   3.00000e-01   5.00000e-01   7.00000e-01   1.00000e+00   1.30000e+00   1.70000e+00   2.00000e+00   3.00000e+00   5.00000e+00   7.00000e+00   1.00000e+01   1.50000e+01   2.00000e+01   3.00000e+01   5.00000e+01   7.00000e+01   1.00000e+02   1.30000e+02   1.70000e+02   2.00000e+02   2.50000e+02   3.00000e+02   5.00000e+02   7.00000e+02   1.00000e+03   1.50000e+03   2.00000e+03   3.00000e+03   4.00000e+03   5.00000e+03   6.00000e+03   7.00000e+03   8.00000e+03   9.00000e+03   1.00000e+04'
    pffdata = pkgutil.get_data(
        'exojax', 'data/atom/barklem_collet_2016_pff.txt')

    # T label for grid QT
    pfTdat = pd.read_csv(io.StringIO(pfT_str), sep='\s+')
    pfdat = pd.read_csv(BytesIO(pffdata), sep='\s+',
                        comment='#', names=pfTdat.columns)
    return pfTdat, pfdat


def partfn_Fe(T):
    """Partition function of Fe I from Irwin_1981.

    Args:
       T: temperature

    Returns:
       partition function Q
    """
    # Irwin_1981
    a = np.zeros(6)
    a[0] = -1.15609527e3
    a[1] = 7.46597652e2
    a[2] = -1.92865672e2
    a[3] = 2.49658410e1
    a[4] = -1.61934455e0
    a[5] = 4.21182087e-2

    Qln = 0.0
    for i, a in enumerate(a):
        Qln = Qln + a*np.log(T)**i
    Q = np.exp(Qln)
    return Q

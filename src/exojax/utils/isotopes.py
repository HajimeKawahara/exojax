import numpy as np
from exojax.utils import isodata
from exojax.utils import molname

import pkgutil
from io import BytesIO
import pandas as pd


def molarmass_hitran():
    """molar mass info from HITRAN_molparam.txt

    
    Returns:
        dict:  mean_molmass, molmass_isotope, abundance_isotope

    Examples:

        >>> path = pkgutil.get_data('exojax', 'data/atom/HITRAN_molparam.txt')
        >>> mean_molmass, molmass_isotope, abundance_isotope = read_HITRAN_molparam(path)
        >>> molmass_isotope["CO"][0] # molar mass for CO HITRAN isotope number = 1
        >>> abundance_isotope["CO"][0] # relative abundance for CO HITRAN isotope number = 1
        >>> mean_molmass["CO"] #mean molar mass of CO
        
    """
    path = pkgutil.get_data('exojax', 'data/atom/HITRAN_molparam.txt')
    df = pd.read_csv(BytesIO(path), sep="\s{2,}", engine="python", skiprows=1, \
                     names=["# Iso","Abundance","Q(296K)","gj","Molar Mass(g)"])
    mean_molmass = {}
    molmass_isotope = {}
    abundance_isotope = {}
    #exact_isotope_number = {}
    e = 0
    for i in range(len(df)):
        if ("(" in df["# Iso"][i]):
            molname = df["# Iso"][i].split()[0]
            tot = 0.0
            tot_abd = 0.0
            molmass_isotope[molname] = []
            abundance_isotope[molname] = []
            #exact_isotope_number[molname] = []
        else:
            tot = tot + df["Abundance"][i] * df["Molar Mass(g)"][i]
            tot_abd = tot_abd + df["Abundance"][i]
            molmass_isotope[molname].append(df["Molar Mass(g)"][i])
            abundance_isotope[molname].append(df["Abundance"][i])
            #isotope_lazy_tag = df["# Iso"][i]
            #exact_isotope_number[molname].append(isotope_lazy_tag)
        if (i == len(df) - 1 or "(" in df["# Iso"][i + 1]):
            mean_molmass[molname] = tot / tot_abd
    return mean_molmass, molmass_isotope, abundance_isotope

def exact_isotope_name_from_isotope(simple_molecule_name, isotope):
    """exact isotope name from isotope (number)

    Args:
        simple_molecular_name (str): simple molecular name such as CO
        isotope (int): isotope number starting from 1

    Returns:
        str: exact isotope name such as (12C)(16O)
    """
    from radis.db.molparam import MolParams
    mp = MolParams()
    return mp.get(simple_molecule_name, isotope, "isotope_name")


def get_isotope(atom, isolist):
    """get isotope info.

    Args:
       atom: simple atomic symbol, such as "H", "Fe"
       isolist: isotope list

    Return:
       iso: isotope list, such as "1H", "2H"
       mass_number: mass_number list
       abundance: abundance list
    """
    iso = []
    mass_number = []
    abundance = []
    for j, isol in enumerate(isolist['isotope']):
        if molname.e2s(isol) == atom:
            iso.append(isolist['isotope'][j])
            mass_number.append(isolist['mass_number'][j])
            abundance.append(float(isolist['abundance'][j]))
    return iso, mass_number, abundance


def get_stable_isotope(atom, isolist):
    """get isotope info.

    Args:
       atom: simple atomic symbol, such as "H", "Fe"
       isolist: isotope list

    Return:
       iso: stabel isotope such as "1H", "2H"
       mass_number: mass_number
       abundance: abundance
    """
    iso, mass_number, abundance = get_isotope(atom, isolist)
    jmax = np.nanargmax(abundance)
    return iso[jmax], mass_number[jmax], abundance[jmax]


if __name__ == '__main__':
    isolist = isodata.read_mnlist()
    print(get_isotope('H', isolist))
    print(get_stable_isotope('H', isolist))

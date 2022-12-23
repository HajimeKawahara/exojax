"""Isotope

These are the same isotope of CO.

- 16O-13C-17O (ExoMol)
- (16O)(13C)(17O) (HITRAN)
- 637 (HITRAN)
- 6 (isotope number, defined in HITRAN_molparam.txt, starting from 1)

Notes:
    ExoJAX follows the definition of isotope number used in HITRAN, which starts from 1, 
    but isotope_number = 0 implies the mean value of all of the isotopes. 

"""

import numpy as np
from exojax.utils import isodata
from exojax.utils import molname
import pkgutil
from io import BytesIO
import pandas as pd


def exact_molecule_name_to_isotope_number(exact_molecule_name):
    """Convert exact molecule name to isotope number

    Args:
        exact_molecule_name (str): exact exomol, hitran, molecule name such as 12C-16O,  (12C)(16O)

    Returns:
        int: isotope number 
    """
    from radis.db.molparam import isotope_name_dict

    #check exomol exact name
    keys = [
        k for k, v in isotope_name_dict.items() if v == exact_molecule_name
    ]
    if len(keys) == 0:
        #check hitran exact name
        exact_hitran_molecule_name = exact_molname_exomol_to_hitran(
            exact_molecule_name)
        keys = [
            k for k, v in isotope_name_dict.items()
            if v == exact_hitran_molecule_name
        ]
    if len(keys) == 1:
        isotope_number = keys[0][1]
    else:
        print("No isotope number identified")
        print("Report at https://github.com/HajimeKawahara/exojax/issues.")
        return None

    return isotope_number


def exact_molname_exomol_to_hitran(exact_exomol_molecule_name):
    """Convert exact_molname used in ExoMol to those in HITRAN

    Args:
        exact_exomol_molecule_name (str): exact exomol molecule name such as 12C-16O

    Returns:
        str: exact exomol molecule name such as (12C)(16O)
    """
    return "(" + exact_exomol_molecule_name.replace("-", ")(") + ")"


def molmass_hitran():
    """molar mass info from HITRAN_molparam.txt

    
    Returns:
        dict:  molmass_isotope, abundance_isotope

    Examples:

        >>> path = pkgutil.get_data('exojax', 'data/atom/HITRAN_molparam.txt')
        >>> mean_molmass, molmass_isotope, abundance_isotope = read_HITRAN_molparam(path)
        >>> molmass_isotope["CO"][1] # molar mass for CO isotope number = 1
        >>> abundance_isotope["CO"][1] # relative abundance for CO isotope number = 1
        >>> molmass_isotope["CO"][0] # mean molar mass for CO isotope number = 1
        
    """
    path = pkgutil.get_data('exojax', 'data/atom/HITRAN_molparam.txt')
    df = pd.read_csv(BytesIO(path), sep="\s{2,}", engine="python", skiprows=1, \
                     names=["# Iso","Abundance","Q(296K)","gj","Molar Mass(g)"])
    molmass_isotope = {}
    abundance_isotope = {}
    for i in range(len(df)):
        if ("(" in df["# Iso"][i]):
            molname = df["# Iso"][i].split()[0]
            tot = 0.0
            tot_abd = 0.0
            molmass_isotope[molname] = []
            abundance_isotope[molname] = [1.0]
        else:
            tot = tot + df["Abundance"][i] * df["Molar Mass(g)"][i]
            tot_abd = tot_abd + df["Abundance"][i]
            molmass_isotope[molname].append(df["Molar Mass(g)"][i])
            abundance_isotope[molname].append(df["Abundance"][i])
        if (i == len(df) - 1 or "(" in df["# Iso"][i + 1]):
            molmass_isotope[molname].insert(0, tot / tot_abd)
    return molmass_isotope, abundance_isotope


def exact_hitran_isotope_name_from_isotope(simple_molecule_name, isotope):
    """exact isotope name from isotope (number)

    Args:
        simple_molecular_name (str): simple molecular name such as CO
        isotope (int): isotope number starting from 1

    Returns:
        str: HITRAN exact isotope name such as (12C)(16O)
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

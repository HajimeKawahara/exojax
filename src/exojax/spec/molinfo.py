import warnings
from exojax.utils.isotopes import molmass_hitran
from exojax.utils.molname import exact_molname_exomol_to_simple_molname
from exojax.utils.molname import exact_molecule_name_to_isotope_number
from exojax.utils.molname import exact_molname_hitran_to_simple_molname

def isotope_molmass(exact_molecule_name):
    """isotope molecular mass 

    Args:
        exact_molecule_name (str): exact exomol, hitran, molecule name such as 12C-16O,  (12C)(16O)

    Returns:
        float or None: molecular mass g/mol
    """
    molmass_isotope, abundance_isotope = molmass_hitran()
    molnumber, isotope_number = exact_molecule_name_to_isotope_number(exact_molecule_name)
    try:
        simple_molecule_name = exact_molname_exomol_to_simple_molname(exact_molecule_name)
        return molmass_isotope[simple_molecule_name][isotope_number]
    except:
        pass
    
    try:
        simple_molecule_name = exact_molname_hitran_to_simple_molname(exact_molecule_name)
        return molmass_isotope[simple_molecule_name][isotope_number]
    except:
        warnings.warn("exact molecule name is not Exomol nor HITRAN form.")
        warnings.warn("No molmass available", UserWarning)
        return None


def molmass_isotope(simple_molecule_name, db_HIT=True):
    """provide molecular mass for the major isotope from the simple molecular name.

    Args:
       molecule: molecular name e.g. CO2, He
       db_HIT: if True, use the molecular mass considering the natural terrestrial abundance and mass of each isotopologue provided by HITRAN (https://hitran.org/docs/iso-meta/)

    Returns: 
       molecular mass

    Example:
       >>> from exojax.spec.moinfo import mean_molmass
       >>> print(molmass("H2"))
       >>> 2.01588
       >>> print(molmass("CO2"))
       >>> 44.0095
       >>> print(molmass("He"))
       >>> 4.002602
       >>> print(molmass("air"))
       >>> 28.97
    """
    molmass_isotope, abundance_isotope = molmass_hitran()

    if simple_molecule_name == 'air' or simple_molecule_name == 'Air':
        return 28.97

    if simple_molecule_name in molmass_isotope and db_HIT:
        molmass = molmass_isotope[simple_molecule_name][1]
    else:
        if (db_HIT):
            warn_msg = "db_HIT is set as True, but the molecular name '%s' does not exist in the HITRAN database. So set db_HIT as False. For reference, all the available molecules in the HITRAN database are as follows:" % simple_molecule_name
            warnings.warn(warn_msg, UserWarning)
            print(list(molmass_isotope.keys()))

        molmass = mean_molmass_manual(simple_molecule_name)

    return molmass

#deprecated
molmass = molmass_isotope

def mean_molmass_manual(simple_molecule_name):
    """molecular mass for major isotope given manually

    Args:
        simple_molecule_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    em = 0.0
    tot = 0.0
    listmol = list(simple_molecule_name)
    ignore = False
    for k, i in enumerate(listmol):
        if ignore:
            ignore = False
        elif i.isdigit():
            tot = tot + em * (int(i) - 1)
        else:
            if k + 1 < len(listmol):
                if listmol[k + 1].islower():
                    em = EachMass[listmol[k] + listmol[k + 1]]
                    ignore = True
                else:
                    em = EachMass[i]
            else:
                em = EachMass[i]

            tot = tot + em
    mean_molmass = tot
    return mean_molmass


EachMass = {
    'H': 1.00794,
    'He': 4.002602,
    'Li': 6.941,
    'Be': 9.012182,
    'B': 10.811,
    'C': 12.0107,
    'N': 14.0067,
    'O': 15.9994,
    'F': 18.9984032,
    'Ne': 20.1797,
    'Na': 22.98976928,
    'Mg': 24.305,
    'Al': 26.9815386,
    'Si': 28.0855,
    'P': 30.973762,
    'S': 32.065,
    'Cl': 35.453,
    'Ar': 39.948,
    'K': 39.0983,
    'Ca': 40.078,
    'Sc': 44.955912,
    'Ti': 47.867,
    'V': 50.9415,
    'Cr': 51.9961,
    'Mn': 54.938045,
    'Fe': 55.845,
    'Co': 58.933195,
    'Ni': 58.6934,
    'Cu': 63.546,
    'Zn': 65.409,
    'Ga': 69.723,
    'Ge': 72.64,
    'As': 74.9216,
    'Se': 78.96,
    'Br': 79.904,
    'Kr': 83.798,
    'Rb': 85.4678,
    'Sr': 87.62,
    'Y': 88.90585,
    'Zr': 91.224,
    'Nb': 92.90638,
    'Mo': 95.94,
    'Tc': 98.9063,
    'Ru': 101.07,
    'Rh': 102.9055,
    'Pd': 106.42,
    'Ag': 107.8682,
    'Cd': 112.411,
    'In': 114.818,
    'Sn': 118.71,
    'Sb': 121.760,
    'Te': 127.6,
    'I': 126.90447,
    'Xe': 131.293,
    'Cs': 132.9054519,
    'Ba': 137.327,
    'La': 138.90547,
    'Ce': 140.116,
    'Pr': 140.90465,
    'Nd': 144.242,
    'Pm': 146.9151,
    'Sm': 150.36,
    'Eu': 151.964,
    'Gd': 157.25,
    'Tb': 158.92535,
    'Dy': 162.5,
    'Ho': 164.93032,
    'Er': 167.259,
    'Tm': 168.93421,
    'Yb': 173.04,
    'Lu': 174.967,
    'Hf': 178.49,
    'Ta': 180.9479,
    'W': 183.84,
    'Re': 186.207,
    'Os': 190.23,
    'Ir': 192.217,
    'Pt': 195.084,
    'Au': 196.966569,
    'Hg': 200.59,
    'Tl': 204.3833,
    'Pb': 207.2,
    'Bi': 208.9804,
    'Po': 208.9824,
    'At': 209.9871,
    'Rn': 222.0176,
    'Fr': 223.0197,
    'Ra': 226.0254,
    'Ac': 227.0278,
    'Th': 232.03806,
    'Pa': 231.03588,
    'U': 238.02891,
    'Np': 237.0482,
    'Pu': 244.0642,
    'Am': 243.0614,
    'Cm': 247.0703,
    'Bk': 247.0703,
    'Cf': 251.0796,
    'Es': 252.0829,
    'Fm': 257.0951,
    'Md': 258.0951,
    'No': 259.1009,
    'Lr': 262,
    'Rf': 267,
    'Db': 268,
    'Sg': 271,
    'Bh': 270,
    'Hs': 269,
    'Mt': 278,
    'Ds': 281,
    'Rg': 281,
    'Cn': 285,
    'Nh': 284,
    'Fl': 289,
    'Mc': 289,
    'Lv': 292,
    'Ts': 294,
    'Og': 294
}

if __name__ == '__main__':
    print(molmass_isotope('H2'))
    print(molmass_isotope('CO2'))
    print(molmass_isotope('He'))
    print(molmass_isotope('air'))
    print(molmass_isotope('CO2', db_HIT=True))
    print(molmass_isotope('He', db_HIT=True))

"""molecular name conversion

- CO -> 12C-16O : simple_molname_to_exact_exomol_stable
- (12C)(16O) -> 12C-16O : exact_molname_hitran_to_exomol
- (12C)(16O) -> CO : exact_molname_hitran_to_simple_molname
- 12C-16O -> (12C)(16O) : exact_molname_exomol_to_hitran
- 12C-16O -> CO exact_molname_exomol_to_simple_molname
- CO+isotope -> (12C)(16O)  or 12C-16O :exact_exact_molecule_name_from_isotope
- To get the recommended ExoMol database, use radis.api.exomolapi.get_exomol_database_list("CO2","12C-16O2")

"""
from radis.db.classes import get_molecule
import re
import warnings

def exact_molecule_name_from_isotope(simple_molecule_name,
                                           isotope,
                                           dbtype="hitran"):
    """exact isotope name from isotope (number)

    Args:
        simple_molecular_name (str): simple molecular name such as CO
        isotope (int): isotope number starting from 1
        dbtype (str): "hitran" or "exomol" 

    Returns:
        str: HITRAN exact isotope name such as (12C)(16O) for dbtype="hitran", 12C-16O for "exomol"
    """
    from radis.db.molparam import MolParams
    mp = MolParams()
    exact_molname = mp.get(simple_molecule_name, isotope, "isotope_name")
    if dbtype == "hitran":
        return exact_molname
    elif dbtype == "exomol":
        return exact_molname_hitran_to_exomol(exact_molname)

def exact_molecule_name_to_isotope_number(exact_molecule_name):
    """Convert exact molecule name to isotope number

    Args:
        exact_molecule_name (str): exact exomol, hitran, molecule name such as 12C-16O,  (12C)(16O)

    Returns:
        int: molecular number, isotope number (or None, None) 
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
        print("HITRAN exact name=", exact_hitran_molecule_name)
        keys = [
            k for k, v in isotope_name_dict.items()
            if v == exact_hitran_molecule_name
        ]
    if len(keys) == 1:
        molnumber = keys[0][0]
        isotope_number = keys[0][1]
    else:
        warnings.warn("No isotope number identified.", UserWarning)
        return None, None

    return molnumber, isotope_number


def exact_molname_exomol_to_simple_molname(exact_exomol_molecule_name):
    """convert the exact molname (used in ExoMol) to the simple molname.

    Args:
       exact_exomol_molecule_name: the exact exomol molecule name

    Returns:
       simple molname

    Examples:
       >>> print(exact_molname_exomol_to_simple_molname("12C-1H4"))
       >>> CH4
       >>> print(exact_molname_exomol_to_simple_molname("23Na-16O-1H"))
       >>> NaOH
       >>> print(exact_molname_exomol_to_simple_molname("HeH_p"))
       >>> HeH_p
       >>> print(exact_molname_exomol_to_simple_molname("trans-31P2-1H-2H")) #not working 
       >>> Warning: Exact molname  trans-31P2-1H-2H cannot be converted to simple molname
       >>> trans-31P2-1H-2H
    """

    try:
        t = exact_exomol_molecule_name.split('-')
        molname_simple = ''
        for ele in t:
            alp = ''.join(re.findall(r'\D', ele))
            num0 = re.split('[A-Z]', ele)[1]
            if num0.isdigit():
                num = num0
            else:
                num = ''
            molname_simple = molname_simple + alp + num
        return molname_simple
    except:
        print('Warning: Exact molname ', exact_exomol_molecule_name,
              'cannot be converted to simple molname')
        return exact_exomol_molecule_name


def exact_molname_hitran_to_simple_molname(exact_hitran_molecule_name):
    """convert exact hitran molname (16C)(13C)(17O) to simple molname, CO2.

    Args:
        exact_hitran_molecule_name (str): exact_hitran_molecule_name, such as (16C)(13C)(17O) 

    Returns:
        str: simple molecue name, such as CO2
    """
    molnum, isonum = exact_molecule_name_to_isotope_number(
        exact_hitran_molecule_name)
    return get_molecule(molnum)


def exact_molname_exomol_to_hitran(exact_exomol_molecule_name):
    """Convert exact_molname used in ExoMol to those in HITRAN

    Args:
        exact_exomol_molecule_name (str): exact exomol molecule name such as 12C-16O

    Returns:
        str: exact exomol molecule name such as (12C)(16O)
    """
    component = exact_exomol_molecule_name.split("-")
    hitran_exact_name = ""
    hydrolist = ["1H", "1H2", "2H", "2H2"]
    replacelist = ["H", "H2", "D", "D2"]
    for c in component:
        if c in hydrolist:
            ind = hydrolist.index(c)
            hitran_exact_name = hitran_exact_name + replacelist[ind]
        elif c[-1].isdigit():
            hitran_exact_name = hitran_exact_name + "(" + c[:-1] + ")" + c[-1]
        else:
            hitran_exact_name = hitran_exact_name + "(" + c + ")"
    return hitran_exact_name


def exact_molname_hitran_to_exomol(exact_molecule_name_hitran):
    """Convert exact_molname used in HITRAN to those in ExoMol

    Args:
        exact_exomol_molecule_name (str): exact exomol molecule name such as (12C)(16O)

    Returns:
        str: exact exomol molecule name such as 12C-16O
    """

    from collections import defaultdict
    import re
    matches = re.findall(r'\((.*?)\)(\d*)', exact_molecule_name_hitran)
    result = defaultdict(int)

    for item, freq in matches:
        if freq == '':
            freq = 1
        result[item] += int(freq)

    # Format the string, exclude 1 from the counts
    result_string = '-'.join([
        f'{key}{value}' if value > 1 else key for key, value in result.items()
    ])

    return result_string


def e2s(exact_exomol_molecule_name):

    warnings.warn(
        "e2s will be replaced to exact_molname_exomol_to_simple_molname.",
        FutureWarning)
    return exact_molname_exomol_to_simple_molname(exact_exomol_molecule_name)


def split_simple(molname_simple):
    """split simple molname.

    Args: 
       molname_simple: simple molname

    Return: 
       atom list
       number list

    Example:

       >>> split_simple("Fe2O3")
       >>> (['Fe', 'O'], ['2', '3'])
    """

    atom_list = []
    num_list = []
    tmp = None
    num = ''
    for ch in molname_simple:
        if ch.isupper():
            if tmp is not None:
                atom_list.append(tmp)
                num_list.append(num)
                num = ''
            tmp = ch
        elif ch.islower():
            tmp = tmp + ch
        elif ch.isdigit():
            num = ch

    atom_list.append(tmp)
    num_list.append(num)

    return atom_list, num_list


def simple_molname_to_exact_exomol_stable(molname_simple):
    """convert the simple molname to the exact molname (used in ExoMol) using
    stable isotopes.

    Args:
        molname_simple: simple molname, such as CO

    Return:
        exact exomol molecule name such as 12C-16O
    """
    if molname_simple == "H3O_p":
        return "1H3-16O_p"

    from exojax.utils import isotopes, isodata
    isolist = isodata.read_mnlist()

    atom_list, num_list = split_simple(molname_simple)
    molname_exact = ''
    for j, atm in enumerate(atom_list):
        iso = isotopes.get_stable_isotope(atm, isolist)
        molname_exact = molname_exact + iso[0] + num_list[j]
        if j < len(atom_list) - 1:
            molname_exact = molname_exact + '-'
    return molname_exact


if __name__ == '__main__':

    print(simple_molname_to_exact_exomol_stable('Fe2O3'))
    print(simple_molname_to_exact_exomol_stable('CH4'))
    print(simple_molname_to_exact_exomol_stable('NaOH'))
    print(simple_molname_to_exact_exomol_stable('H3O_p'))

    print(e2s('12C-1H4'))
    print(e2s('23Na-16O-1H'))
    print(e2s('HeH_p'))
    print(e2s('trans-31P2-1H-2H'))  # not working

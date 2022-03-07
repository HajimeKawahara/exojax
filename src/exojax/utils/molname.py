import re


def e2s(molname_exact):
    """convert the exact molname (used in ExoMol) to the simple molname.

    Args:
       molname_exact: the exact molname

    Returns:
       simple molname

    Examples:
       >>> print(e2s("12C-1H4"))
       >>> CH4
       >>> print(e2s("23Na-16O-1H"))
       >>> NaOH
       >>> print(e2s("HeH_p"))
       >>> HeH_p
       >>> print(e2s("trans-31P2-1H-2H")) #not working 
       >>> Warning: Exact molname  trans-31P2-1H-2H cannot be converted to simple molname
       >>> trans-31P2-1H-2H
    """

    try:
        t = molname_exact.split('-')
        molname_simple = ''
        for ele in t:
            alp = ''.join(re.findall(r'\D', ele))
            num0 = re.split('[A-Z]', ele)[1]
            if num0.isdigit():
                num = num0
            else:
                num = ''
            molname_simple = molname_simple+alp+num
        return molname_simple
    except:
        print('Warning: Exact molname ', molname_exact,
              'cannot be converted to simple molname')
        return molname_exact


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
            tmp = tmp+ch
        elif ch.isdigit():
            num = ch

    atom_list.append(tmp)
    num_list.append(num)

    return atom_list, num_list


def s2e_stable(molname_simple):
    """convert the simple molname to the exact molname (used in ExoMol) using
    stable isotopes.

    Args:
       molname_simple: simple molname

    Return:
       exact molname
    """
    from exojax.utils import isotopes, isodata
    isolist = isodata.read_mnlist()

    atom_list, num_list = split_simple(molname_simple)
    molname_exact = ''
    for j, atm in enumerate(atom_list):
        iso = isotopes.get_stable_isotope(atm, isolist)
        molname_exact = molname_exact+iso[0]+num_list[j]
        if j < len(atom_list)-1:
            molname_exact = molname_exact+'-'
    return molname_exact


if __name__ == '__main__':

    print(s2e_stable('Fe2O3'))
    print(s2e_stable('CH4'))
    print(s2e_stable('NaOH'))

    print(e2s('12C-1H4'))
    print(e2s('23Na-16O-1H'))
    print(e2s('HeH_p'))
    print(e2s('trans-31P2-1H-2H'))  # not working

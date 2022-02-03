import numpy as np
import pandas as pd
from exojax.utils import isodata
from exojax.utils import molname


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

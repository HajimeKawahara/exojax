"""solar abundance.

- Solar abundance data
- AAG21 = Asplund, M., Amarsi, A. M., & Grevesse, N. 2021, arXiv:2105.01661
"""

import numpy as np
from exojax import data
from exojax.spec.molinfo import element_mass


AAG21 = {
    "H": [12.00, 0.00, 8.22, 0.04],
    "He": [10.914, 0.013, 1.29, 0.18],
    "Li": [0.96, 0.06, 3.25, 0.04],
    "Be": [1.38, 0.09, 1.32, 0.03],
    "B": [2.70, 0.20, 2.79, 0.04],
    "C": [8.46, 0.04, 7.39, 0.04],
    "N": [7.83, 0.07, 6.26, 0.06],
    "O": [8.69, 0.04, 8.39, 0.04],
    "F": [4.40, 0.25, 4.42, 0.06],
    "Ne": [8.06, 0.05, -1.12, 0.18],
    "Na": [6.22, 0.03, 6.27, 0.04],
    "Mg": [7.55, 0.03, 7.53, 0.02],
    "Al": [6.43, 0.03, 6.43, 0.03],
    "Si": [7.51, 0.03, 7.51, 0.01],
    "P": [5.41, 0.03, 5.43, 0.03],
    "S": [7.12, 0.03, 7.15, 0.02],
    "Cl": [5.31, 0.20, 5.23, 0.06],
    "Ar": [6.38, 0.10, -0.50, 0.18],
    "K": [5.07, 0.03, 5.08, 0.04],
    "Ca": [6.30, 0.03, 6.29, 0.03],
    "Sc": [3.14, 0.04, 3.04, 0.03],
    "Ti": [4.97, 0.05, 4.90, 0.03],
    "V": [3.90, 0.08, 3.96, 0.03],
    "Cr": [5.62, 0.04, 5.63, 0.02],
    "Mn": [5.42, 0.06, 5.47, 0.03],
    "Fe": [7.46, 0.04, 7.46, 0.02],
    "Co": [4.94, 0.05, 4.87, 0.02],
    "Ni": [6.20, 0.04, 6.20, 0.03],
    "Cu": [4.18, 0.05, 4.25, 0.06],
    "Zn": [4.56, 0.05, 4.61, 0.02],
    "Ga": [3.02, 0.05, 3.07, 0.03],
    "Ge": [3.62, 0.10, 3.58, 0.04],
    "As": [np.nan, np.nan, 2.30, 0.04],
    "Se": [np.nan, np.nan, 3.34, 0.03],
    "Br": [np.nan, np.nan, 2.54, 0.06],
    "Kr": [3.12, 0.10, -2.27, 0.18],
    "Rb": [2.32, 0.08, 2.37, 0.03],
    "Sr": [2.83, 0.06, 2.88, 0.03],
    "Y": [2.21, 0.05, 2.15, 0.02],
    "Zr": [2.59, 0.04, 2.53, 0.02],
    "Nb": [1.47, 0.06, 1.42, 0.04],
    "Mo": [1.88, 0.09, 1.93, 0.04],
    "Ru": [1.75, 0.08, 1.77, 0.02],
    "Rh": [0.78, 0.11, 1.04, 0.02],
    "Pd": [1.57, 0.10, 1.65, 0.02],
    "Ag": [0.96, 0.10, 1.20, 0.04],
    "Cd": [np.nan, np.nan, 1.71, 0.03],
    "In": [0.80, 0.20, 0.76, 0.02],
    "Sn": [2.02, 0.10, 2.07, 0.06],
    "Sb": [np.nan, np.nan, 1.01, 0.06],
    "Te": [np.nan, np.nan, 2.18, 0.03],
    "I": [np.nan, np.nan, 1.55, 0.08],
    "Xe": [2.22, 0.05, -1.95, 0.18],
    "Cs": [np.nan, np.nan, 1.08, 0.03],
    "Ba": [2.27, 0.05, 2.18, 0.02],
    "La": [1.11, 0.04, 1.17, 0.01],
    "Ce": [1.58, 0.04, 1.58, 0.01],
    "Pr": [0.75, 0.05, 0.76, 0.01],
    "Nd": [1.42, 0.04, 1.45, 0.01],
    "Sm": [0.95, 0.04, 0.94, 0.01],
    "Eu": [0.52, 0.04, 0.52, 0.01],
    "Gd": [1.08, 0.04, 1.05, 0.01],
    "Tb": [0.31, 0.10, 0.31, 0.01],
    "Dy": [1.10, 0.04, 1.13, 0.01],
    "Ho": [0.48, 0.11, 0.47, 0.01],
    "Er": [0.93, 0.05, 0.93, 0.01],
    "Tm": [0.11, 0.04, 0.12, 0.01],
    "Yb": [0.85, 0.11, 0.92, 0.01],
    "Lu": [0.10, 0.09, 0.09, 0.01],
    "Hf": [0.85, 0.05, 0.71, 0.01],
    "Ta": [np.nan, np.nan, -0.15, 0.04],
    "W": [0.79, 0.11, 0.65, 0.04],
    "Re": [np.nan, np.nan, 0.26, 0.02],
    "Os": [1.35, 0.12, 1.35, 0.02],
    "Ir": [np.nan, np.nan, 1.32, 0.02],
    "Pt": [np.nan, np.nan, 1.61, 0.02],
    "Au": [0.91, 0.12, 0.81, 0.05],
    "Hg": [np.nan, np.nan, 1.17, 0.18],
    "Tl": [0.92, 0.17, 0.77, 0.05],
    "Pb": [1.95, 0.08, 2.03, 0.03],
    "Bi": [np.nan, np.nan, 0.65, 0.0],
    "U": [np.nan, np.nan, -0.54, 0.03],
}


def nsol(database="AAG21"):
    """provides solar abundance dictionary.

    Args:
        database: name of database, default to AAG21.

    Note:
        AAG21   Asplund, M., Amarsi, A. M., & Grevesse, N. 2021, arXiv:2105.01661
        AG89 	Anders E. & Grevesse N. (1989, Geochimica et Cosmochimica Acta 53, 197) (Photospheric, using Table 2)
        AGSS09 	Asplund M., Grevesse N., Sauval A.J. & Scott P. (2009, ARAA, 47, 481) (Photospheric, using Table 1)
        F92 	Feldman U.(1992, Physica Scripta 46, 202)
        AE82 	Anders E. & Ebihara (1982, Geochimica et Cosmochimica Acta 46, 2363)
        GS98 	Grevesse, N. & Sauval, A.J. (1998, Space Science Reviews 85, 161)
        WAM00 	Wilms J., Allen A. & McCray R. (2000, ApJ 542, 914)
        L03	Lodders K (2003, ApJ 591, 1220) (Photospheric, using Table 1)
        LPG09photo 	Lodders K., Palme H., Gail H.P. (2009, Landolt-Barnstein, New Series, vol VI/4B, pp 560-630) (Photospheric, using Table 4)
        LPG09proto 	Lodders K., Palme H., Gail H.P. (2009, Landolt-Barnstein, New Series, vol VI/4B, pp 560-630) (Proto-solar, using Table 10)

    Returns:
        number ratio of elements for solar abundance

    Example:
        >>>  nsun=nsol()
        >>>  print(nsun["Fe"])
        >>>  2.6602622265852853e-05
    """
    available_databases = _available_abundance_databases()
    if database not in available_databases:
        raise ValueError(
            f"database {database} is not available. Available databases are {available_databases.keys()}"
        )

    print("Database for solar abundance = ", database)
    print(available_databases[database])
    if database == "AAG21":
        return _nsol_aag21()
    else:
        return _nsol_from_xspec(database)


def _available_abundance_databases():
    database_available = {}
    database_available["AAG21"] = (
        "Asplund, M., Amarsi, A. M., & Grevesse, N. 2021, arXiv:2105.01661"
    )
    database_available["AG89"] = (
        "Anders E. & Grevesse N. (1989, Geochimica et Cosmochimica Acta 53, 197) (Photospheric, using Table 2)"
    )
    database_available["AGSS09"] = (
        "Asplund M., Grevesse N., Sauval A.J. & Scott P. (2009, ARAA, 47, 481) (Photospheric, using Table 1)"
    )
    database_available["F92"] = "Feldman U.(1992, Physica Scripta 46, 202)"
    database_available["AE82"] = (
        "Anders E. & Ebihara (1982, Geochimica et Cosmochimica Acta 46, 2363)"
    )
    database_available["GS98"] = (
        "Grevesse, N. & Sauval, A.J. (1998, Space Science Reviews 85, 161)"
    )
    database_available["WAM00"] = "Wilms J., Allen A. & McCray R. (2000, ApJ 542, 914)"
    database_available["L03"] = (
        "Lodders K (2003, ApJ 591, 1220) (Photospheric, using Table 1)"
    )
    database_available["LPG09photo"] = (
        "Lodders K., Palme H., Gail H.P. (2009, Landolt-Barnstein, New Series, vol VI/4B, pp 560-630) (Photospheric, using Table 4)"
    )
    database_available["LPG09proto"] = (
        "Lodders K., Palme H., Gail H.P. (2009, Landolt-Barnstein, New Series, vol VI/4B, pp 560-630) (Proto-solar, using Table 10)"
    )
    return database_available


def _nsol_from_xspec(database):
    """

    Notes:
        reference: https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node116.html
    """
    import pkg_resources
    import pandas as pd

    filename = "data/abundance/xspec_abundance.txt"
    file_path = pkg_resources.resource_filename("exojax", filename)
    df = pd.read_csv(file_path, comment="#", delimiter=",")
    nsol = df.set_index("El")[database].to_dict()
    total_sum = sum(nsol.values())
    nsol = {key: value / total_sum for key, value in nsol.items()}

    return nsol


def _nsol_aag21():
    """provides solar abundance dictionary from AAG21.

    Args:
        database: name of database.

    Returns:
        number ratio of elements for solar abundance

    Example:
        >>>  nsun=nsol()
        >>>  print(nsun["Fe"])
        >>>  2.6602622265852853e-05
    """
    # compute total number
    allab = 0.0
    for atm in AAG21:
        val = AAG21[atm]
        if val[0] is np.nan:
            allab = allab + 10 ** val[2]
        else:
            allab = allab + 10 ** val[0]

    nsun = {}
    for atm in AAG21:
        val = AAG21[atm]
        if val[0] is np.nan:
            nsun[atm] = 10 ** AAG21[atm][2] / allab
        else:
            nsun[atm] = 10 ** AAG21[atm][0] / allab

    return nsun


def mass_fraction(atom, number_ratio_elements):
    """mass fraction of atom

    Notes:
        X = mass fraction of hydrogen
        Y = mass fraction of helium
        Z = mass fraction of metals
        For definition, see https://en.wikipedia.org/wiki/Metallicity#Mass_fraction

    Args:
        atom: atom name, such as "H", "He", "C", "O", "Fe", etc.
        number_ratio_elements: element number ratio of abundance, when n = nsol(), X, Y, Z are solar abundance Xsol. Ysol, Zsol.

    Returns:
        mass fraction of atom
    """
    weighted_sum_mass = _sum_mass_weighted_elements(number_ratio_elements)
    return element_mass[atom] * number_ratio_elements[atom] / weighted_sum_mass


def mass_fraction_XYZ(number_ratio_elements):
    """mass fraction of hydrogen, helium, metals, i.e. well known symbols in astronomy X, Y, Z

    Notes:
        X = mass fraction of hydrogen
        Y = mass fraction of helium
        Z = mass fraction of metals
        For definition, see https://en.wikipedia.org/wiki/Metallicity#Mass_fraction

    Args:
        number_ratio_elements: element number ratio of abundance, when n = nsol(), X, Y, Z are solar abundance Xsol. Ysol, Zsol.

    Returns:
        float: X, Y, Z (mass fraction of H, He, metals)
    """

    X = mass_fraction("H", number_ratio_elements)
    Y = mass_fraction("He", number_ratio_elements)
    Z = 1.0 - X - Y

    return X, Y, Z


def _sum_mass_weighted_elements(number_ratio_elements):
    sum_element = sum(
        [
            element_mass[atom] * number_ratio_elements[atom]
            for atom in number_ratio_elements
        ]
    )
    return sum_element

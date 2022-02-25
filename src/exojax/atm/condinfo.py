"""condensate information.

- LF98 Lodders and Fegley (1998)
"""

# condensate density g/cm3 from LF98 Table 1.18 (p37)
conddensity = {
    'Fe_solid': 7.875,
    'Si_solid': 2.33,
    'FeO': 5.987,
    'Fe2O3': 5.275,
    'Fe3O4': 5.200,
    'FeSiO4': 4.393,
    'Al2O3': 3.987,
    'SiO2': 2.648,
    'TiO2': 4.245,
    'MgSiO3': 3.194,
    'Mg2SiO4': 3.214,
}

name2formula = {
    'ferrous oxide': 'FeO',
    'hematite': 'Fe2O3',
    'magnetite': 'Fe3O4',
    'fayalite': 'FeSiO4',
    'corundum': 'Al2O3',
    'quartz': 'SiO2',
    'rutile': 'TiO2',
    'enstatite': 'MgSiO3',
    'forstelite': 'Mg2SiO4',
}

if __name__ == '__main__':
    print(conddensity['Fe_solid'])

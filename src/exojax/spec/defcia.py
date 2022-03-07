"""Definition of Default cia dataset for autospec."""

HITRAN_DEFCIA = \
    {
        'H2-CH4 (equilibrium)': 'H2-CH4_eq_2011.cia',
        'H2-CH4 (normal)': 'H2-CH4_norm_2011.cia',
        'H2-H2': 'H2-H2_2011.cia',
        'H2-H': 'H2-H_2011.cia',
        'H2-He': 'H2-He_2011.cia',
        'He-H': 'He-H_2011.cia',
        'N2-H2': 'N2-H2_2011.cia',
        'N2-He': 'N2-He_2018.cia',
        'N2-N2': 'N2-N2_2018.cia',
        'N2-air': 'N2-air_2018.cia',
        'N2-H2O': 'N2-H2O_2018.cia',
        'O2-CO2': 'O2-CO2_2011.cia',
        'O2-N2': 'O2-N2_2018.cia',
        'O2-O2': 'O2-O2_2018b.cia',
        'O2-air': 'O2-Air_2018.cia',
        'CO2-CO2': 'CO2-CO2_2018.cia',
        'CO2-H2': 'CO2-H2_2018.cia',
        'CO2-He': 'CO2-He_2018.cia',
        'CO2-CH4': 'CO2-CH4_2018.cia',
        'CH4-He': 'CH4-He_2018.cia'
    }


def ciafile(interaction):
    """provide ciafile from interaction.

    Args:
       interaction: e.g. H2-H2

    Returns:
       cia file name
    """
    try:
        return HITRAN_DEFCIA[interaction]
    except:
        return None


def interaction2mols(interaction):
    """provide 2 molecules from interaction.

    Args:
       interaction: e.g. H2-H2

    Returns:
       mol1, mol2

    Examples:

    >>> from exojax.spec.defcia import interaction2mols
    >>> print(interaction2mols("H2-CH4 (equilibrium)"))
    >>> ('H2', 'CH4')
    >>> print(interaction2mols("CH4-He"))
    >>> ('CH4', 'He')
    """
    mm = interaction.split(' ')[0]
    mm = mm.split('-')
    return mm[0], mm[1]


if __name__ == '__main__':
    print(interaction2mols('H2-CH4 (equilibrium)'))
    print(interaction2mols('CH4-He'))
    print(ciafile('H2-H2'))

"""Definition of Default dataset for autospec."""

from exojax.utils.molname import e2s, s2e_stable
from exojax.utils.recexomol import get_exomol_database_list

HITRAN_DEFMOL = \
    {
        'CO': '05_hit12.par',
        'CH4': '06_hit12.par'
    }

HITEMP_DEFMOL = \
    {
        'N2O': '04_HITEMP2019.par.bz2',
        'CO': '05_HITEMP2019.par.bz2',
        'CH4': '06_HITEMP2020.par.bz2',
        'NO': '08_HITEMP2019.par.bz2',
        'NO2': '10_HITEMP2019.par.bz2',
        'OH': '13_HITEMP2020.par.bz2',
    }

EXOMOL_DEFMOL = \
    {
        '12C-16O': 'Li2015',
        '16O-1H': 'MoLLIST',
        '14N-1H3': 'CoYuTe',
        '14N-16O': 'NOname',
        '56Fe-1H': 'MoLLIST',
        '1H2-32S': 'AYT2',
        '28Si-16O2': 'OYT3',
        '12C-1H4': 'YT34to10',
        '1H-12C-14N': 'Harris',
        '12C2-1H2': 'aCeTY',
        '48Ti-16O': 'Toto',
        '12C-16O2': 'UCL-4000',
        '52Cr-1H': 'MoLLIST',
        '1H2-16O': 'POKAZATEL',
        '51V-16O': 'VOMYT',
        '12C-14N': 'Trihybrid'
    }


def search_molfile(database, molecules):
    """name identifier of molecular databases.

    Args:
       database: molecular database (HITRAN,HITEMP,ExoMol)
       molecules: molecular name such as (CO, 12C-16O)

    Returns:
       identifier
    """
    if database == 'ExoMol':
        try:
            # online
            try:
                rec = get_exomol_database_list(e2s(molecules), molecules)
                # default (recommended) database by ExoMol
                exomol_defmol = rec[1]
                print('Recommendated database by ExoMol: ', exomol_defmol)
                return e2s(molecules)+'/'+molecules+'/'+exomol_defmol
            except:
                molname_exact = s2e_stable(molecules)
                print('Warning:', molecules, 'is interpreted as', molname_exact)
                rec = get_exomol_database_list(molecules, molname_exact)
                # default (recommended) database by ExoMol
                exomol_defmol = rec[1]
                print('Recommendated database by ExoMol: ', exomol_defmol)
                return molecules+'/'+molname_exact+'/'+exomol_defmol

        except:
            # offline
            try:
                print('Warning: try off-line mode in defmol.py.')
                try:
                    print('Recommendated database by defmol.py: ',
                          EXOMOL_DEFMOL[molname_exact])
                    return e2s(molecules)+'/'+molecules+'/'+EXOMOL_DEFMOL[molecules]
                except:
                    molname_exact = s2e_stable(molecules)
                    print('Warning:', molecules,
                          'is interpreted as', molname_exact)
                    print('Recommendated database by defmol.py: ',
                          EXOMOL_DEFMOL[molname_exact])
                    return molecules+'/'+molname_exact+'/'+EXOMOL_DEFMOL[molname_exact]
            except:
                print('No recommendation found.')
                return None

    elif database == 'HITRAN':
        try:
            return HITRAN_DEFMOL[molecules]
        except:
            return None

    elif database == 'HITEMP':
        try:
            return HITEMP_DEFMOL[molecules]
        except:
            return None


if __name__ == '__main__':
    print(search_molfile('ExoMol', '12C-16O'))
    print(search_molfile('ExoMol', 'CO'))
    print(search_molfile('ExoMol', 'CN'))
    print(search_molfile('HITRAN', 'CO'))
    print(search_molfile('HITEMP', 'CO'))

from exojax.utils.molname import simple_molname_to_exact_exomol_stable
from exojax.utils.molname import split_simple
from exojax.utils.molname import exact_molname_exomol_to_simple_molname
from exojax.utils.molname import exact_molname_exomol_to_hitran
from exojax.utils.molname import exact_molname_hitran_to_exomol
from exojax.utils.molname import exact_molname_hitran_to_simple_molname
from exojax.utils.molname import exact_molecule_name_from_isotope
from exojax.utils.molname import exact_molecule_name_to_isotope_number
import numpy as np


def test_exact_molecule_name_to_isotope_number():
    eemn = "12C-16O"
    molnum, isonum = exact_molecule_name_to_isotope_number(eemn)
    assert isonum == 1
    eemn = "16O-13C-17O"
    molnum, isonum = exact_molecule_name_to_isotope_number(eemn)
    assert isonum == 6
    eemn = "1H2-16O"
    molnum, isonum = exact_molecule_name_to_isotope_number(eemn)
    assert isonum == 1
    eemn = "12C-16O2"
    molnum, isonum = exact_molecule_name_to_isotope_number(eemn)


def test_exact_isotope_name_from_isotope():
    simple_molecule_name = "CO"
    isotope = 1
    assert exact_molecule_name_from_isotope(simple_molecule_name,
                                                  isotope) == "(12C)(16O)"

    simple_molecule_name = "H2O"
    isotope = 5
    assert exact_molecule_name_from_isotope(simple_molecule_name,
                                                  isotope) == "HD(18O)"


def test_exact_molname_hitran_to_simple_molname():
    emen = "(16O)(13C)(17O)"
    sn = exact_molname_hitran_to_simple_molname(emen)
    assert sn == "CO2"


def test_exact_molname_exomol_to_simple_molname():
    assert exact_molname_exomol_to_simple_molname('12C-1H4') == "CH4"
    assert exact_molname_exomol_to_simple_molname('23Na-16O-1H') == "NaOH"
    assert exact_molname_exomol_to_simple_molname('HeH_p') == "HeH_p"
    assert exact_molname_exomol_to_simple_molname(
        "trans-31P2-1H-2H") == "trans-31P2-1H-2H"


def test_exact_molname_exomol_to_hitran():
    eemn = "16O-13C-17O"
    ehmn = exact_molname_exomol_to_hitran(eemn)
    assert ehmn == "(16O)(13C)(17O)"


def test_exact_molname_hitran_to_exomol():
    ehmn = "(12C)(16O)"
    eemn = exact_molname_hitran_to_exomol(ehmn)
    assert eemn == "12C-16O"

    ehmn = "(12C)(16O)2"
    eemn = exact_molname_hitran_to_exomol(ehmn)
    assert eemn == "12C-16O2"
    

def test_s2estable():
    EXOMOL_SIMPLE2EXACT = \
        {
            'CO': '12C-16O',
            'OH': '16O-1H',
            'NH3': '14N-1H3',
            'NO': '14N-16O',
            'FeH': '56Fe-1H',
            'H2S': '1H2-32S',
            'SiO': '28Si-16O',
            'CH4': '12C-1H4',
            'HCN': '1H-12C-14N',
            'C2H2': '12C2-1H2',
            'TiO': '48Ti-16O',
            'CO2': '12C-16O2',
            'CrH': '52Cr-1H',
            'H2O': '1H2-16O',
            'VO': '51V-16O',
            'CN': '12C-14N',
            'PN': '31P-14N',
        }

    check = True
    for i in EXOMOL_SIMPLE2EXACT:
        assert simple_molname_to_exact_exomol_stable(i) == EXOMOL_SIMPLE2EXACT[i]
    assert simple_molname_to_exact_exomol_stable("H3O_p") == "1H3-16O_p"


def test_split_simple():
    assert np.all(split_simple("Fe2O3") == (['Fe', 'O'], ['2', '3']))


if __name__ == '__main__':
    #test_s2estable()
    #test_exact_molecule_name_to_isotope_number()
    test_exact_molname_hitran_to_exomol()
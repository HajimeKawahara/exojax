from exojax.utils.mollabel import format_molecule, format_molecules_list, format_molecules_lists
from exojax.utils.mollabel import molecule_color, molecules_color_list, molecules_color_lists


def test_format_molecule():
    assert format_molecule("H2O") == "$\\mathrm{H_2O}$"
    assert format_molecule("CH4") == "$\\mathrm{CH_4}$"
    assert format_molecule("CO") == "$\\mathrm{CO}$"
    assert format_molecule("NH3") == "$\\mathrm{NH_3}$"


def test_format_molecules_list():
    assert format_molecules_list(
        ["H2O", "CH4",
         "CO"]) == ["$\\mathrm{H_2O}$", "$\\mathrm{CH_4}$", "$\\mathrm{CO}$"]
    assert format_molecules_list(
        ["NH3", "NO2"]) == ["$\\mathrm{NH_3}$", "$\\mathrm{NO_2}$"]


def test_format_molecules_lists():
    assert format_molecules_lists([["H2O", "CH4", "CO"], [
        "H2O", "CH4", "CO"
    ]]) == [["$\\mathrm{H_2O}$", "$\\mathrm{CH_4}$", "$\\mathrm{CO}$"],
            ["$\\mathrm{H_2O}$", "$\\mathrm{CH_4}$", "$\\mathrm{CO}$"]]
    assert format_molecules_lists(
        [["NH3", "NO2"],
         ["H2O", "CH4"]]) == [["$\\mathrm{NH_3}$", "$\\mathrm{NO_2}$"],
                              ["$\\mathrm{H_2O}$", "$\\mathrm{CH_4}$"]]

def test_molecules_color():
    assert molecule_color("H2O") == "C1"

def test_molecules_color_list():
    assert molecules_color_list(["H2O","CH4"]) == ["C1","C6"]

def test_molecules_color_lists():
    simple_molecule_name_lists = [["H2O", "CH4", "CO"], ["H2O", "NH3", "CO"]]
    assert molecules_color_lists(simple_molecule_name_lists) == [['C1', 'C6', 'C5'], ['C1', 'C11', 'C5']]

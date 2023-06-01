import re


def molecule_color(simple_molecule_name):
    """return the individual matplotlib color label (CN) for a given molecule simple name  

    Args:
        simple_molecule_name (str): simple molecule name, "H2O"

    Examples:
        >>> format_molecule_color("H2O")
        "C1"
    

    Returns:
        str: CN label, such as C1 for "H2O" based on HITRAN identifier. If the molecule does not exist in the HITRAN identifiers, return gray
    """
    from radis.db.classes import get_molecule_identifier
    try:
        i = get_molecule_identifier(simple_molecule_name)
        color = "C"+str(i)
    except:
        color = "gray"
    return color

def molecules_color_list(simple_molecule_name_list):
    """return the individual matplotlib color label (CN) for a given molecule simple name list

    Args:
        simple_molecule_name_list (array): simple molecule name list such as ["H2O","CO"]

    Returns:
        str: CN label, such as C1 for "H2O"
    """
    
    return [molecule_color(molecule) for molecule in simple_molecule_name_list]

def molecules_color_lists(simple_molecule_name_lists):
    return [[molecule_color(molecule) for molecule in molecules] for molecules in simple_molecule_name_lists]


def format_molecule(simple_molecule_name):
    """Format a given molecule string with subscript numbers and convert it to LaTeX syntax.
    
    This function takes in a molecule string, where elements are represented by their symbols and
    the number of atoms is represented by a subscript number following the symbol. The function
    converts the molecule string to a LaTeX string, with subscript numbers formatted correctly
    and the string surrounded by LaTeX math mode and "mathrm" syntax.

    Args:
        molecule (str): A string representation of a molecule.

    Returns:
        str: A LaTeX-formatted string representation of the molecule.

    Examples:
        >>> format_molecule("H2O")
        "$\\mathrm{H_2O}$"
        >>> format_molecule("CH4")
        "$\\mathrm{CH_4}$"
        >>> format_molecule("CO")
        "$\\mathrm{CO}$"
    """
    formatted_molecule = re.sub(r'([A-Z][a-z]?)([0-9]+)', r'\1_\2', simple_molecule_name)
    latex_molecule = f"$\\mathrm{{{formatted_molecule}}}$"
    return latex_molecule

def format_molecules_list(molecules):
    """
    Format a list of molecule strings with subscript numbers and convert them to LaTeX syntax.
    
    This function takes in a list of molecule strings, where elements are represented by their symbols and
    the number of atoms is represented by a subscript number following the symbol. The function
    converts each molecule string to a LaTeX string, with subscript numbers formatted correctly
    and the string surrounded by LaTeX math mode and "mathrm" syntax.

    Args:
        molecules (list of str): A list of string representations of molecules.

    Returns:
        list of str: A list of LaTeX-formatted string representations of the molecules.

    Examples:
        >>> format_molecules_list(["H2O", "CH4", "CO"])
        ["$\\mathrm{H_2O}$", "$\\mathrm{CH_4}$", "$\\mathrm{CO}$"]
    """
    return [format_molecule(molecule) for molecule in molecules]

def format_molecules_lists(molecules_lists):
    """
    Format a list of lists of molecule strings with subscript numbers and convert them to LaTeX syntax.
    
    This function takes in a list of lists of molecule strings, where elements are represented by their symbols and
    the number of atoms is represented by a subscript number following the symbol. The function
    converts each molecule string to a LaTeX string, with subscript numbers formatted correctly
    and the string surrounded by LaTeX math mode and "mathrm" syntax.

    Args:
        molecules_lists (list of list of str): A list of lists of string representations of molecules.

    Returns:
        list of list of str: A list of lists of LaTeX-formatted string representations of the molecules.

    Examples:
        >>> format_molecules_lists([["H2O", "CH4", "CO"], ["H2O", "CH4", "CO"]])
        [["$\\mathrm{H_2O}$", "$\\mathrm{CH_4}$", "$\\mathrm{CO}$"], ["$\\mathrm{H_2O}$", "$\\mathrm{CH4}$", "$\\mathrm{CO}$"]]
    """
    return [[format_molecule(molecule) for molecule in molecules] for molecules in molecules_lists]


if __name__ == "__main__":
    simple_molecule_name_list = ["H2O", "CH4", "CO"]

    for molecule in simple_molecule_name_list:
        print(format_molecule(molecule))
    print(format_molecules_list(simple_molecule_name_list))
    
    print(molecules_color_list(simple_molecule_name_list))
    
    simple_molecule_name_lists = [["H2O", "CH4", "CO"], ["H2O", "NH3", "CO"]]
    print(format_molecules_lists(simple_molecule_name_lists))
    print(molecules_color_lists(simple_molecule_name_lists))

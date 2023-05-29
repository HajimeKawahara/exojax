import re

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

def replace_molecules_with_color_indices(molecules_lists):
    """
    Replace a list of lists of molecule strings with strings of 'C' followed by an index.
    
    This function takes in a list of lists of molecule strings and replaces each molecule with a 
    unique string of 'C' followed by an index.

    Args:
        molecules_lists (list of list of str): A list of lists of string representations of molecules.

    Returns:
        list of list of str: A list of lists of strings, where each string is 'C' followed by an index.

    Examples:
        >>> replace_molecules_with_indices([["H2O", "CH4", "CO"], ["H2S", "H2O", "CO"]])
        [["C0", "C1", "C2"], ["C3", "C0", "C2"]]
    """
    molecule_to_index = {}
    next_index = 0
    result = []
    for molecules in molecules_lists:
        result.append([])
        for molecule in molecules:
            if molecule not in molecule_to_index:
                molecule_to_index[molecule] = next_index
                next_index += 1
            result[-1].append("C" + str(molecule_to_index[molecule]))
    return result

if __name__ == "__main__":
    molecules = ["H2O", "CH4", "CO"]

    for molecule in molecules:
        print(format_molecule(molecule))
    print(format_molecules_list(molecules))
    
    molecules_s = [["H2O", "CH4", "CO"], ["H2O", "NH3", "CO"]]
    print(format_molecules_lists(molecules_s))
    print(replace_molecules_with_color_indices(molecules_s))
import re

def format_molecule(simple_molecule_name):
    """Format a given molecule string with subscript numbers and convert it to LaTeX syntax.
    
    This function takes in a molecule string, where elements are represented by their symbols and
    the number of atoms is represented by a subscript number following the symbol. The function
    converts the molecule string to a LaTeX string, with subscript numbers formatted correctly
    and the string surrounded by LaTeX math mode and "mathrm" syntax.

    Args:
        simple_molecule_name (str): A string representation of a molecule.

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

if __name__ == "__main__":
    molecules = ["H2O", "CH4", "CO"]

    for molecule in molecules:
        print(format_molecule(molecule))


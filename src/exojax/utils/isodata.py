"""Mass number and abundance list.

- Taken from https://www.chem.ualberta.ca/~massspec/atomic_mass_abund.pdf The isotopic mass data is from G. Audi, A. H. Wapstra Nucl. Phys A. 1993, 565, 1-65 and G. Audi, A. H. Wapstra Nucl. Phys A. 1995, 595, 409-480.  The percent natural abundance data is from the 1997 report of the IUPAC Subcommittee for Isotopic Abundance Measurements by K.J.R. Rosman, P.D.P. Taylor Pure Appl. Chem. 1999, 71, 1593-1607.
"""
import pandas as pd
import io

mn = '''
'''


def read_mnlist():
    """loading mass number list.

    Returns:
        dictionary of mass number, keys="isotope","mass_number","abundance"
    """
    mnlist = pd.read_csv(io.StringIO(mn), delimiter=',',
                         comment='#', usecols=(0, 1, 2))
    return mnlist

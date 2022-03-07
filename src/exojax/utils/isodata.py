"""Mass number and abundance list."""
import pandas as pd
import pkgutil
from io import BytesIO


def read_mnlist():
    """loading mass number list.

    Note:
        this code reads data/atom/iso_mn.txt, taken from https://www.chem.ualberta.ca/~massspec/atomic_mass_abund.pdf The isotopic mass data is from G. Audi, A. H. Wapstra Nucl. Phys A. 1993, 565, 1-65 and G. Audi, A. H. Wapstra Nucl. Phys A. 1995, 595, 409-480.  The percent natural abundance data is from the 1997 report of the IUPAC Subcommittee for Isotopic Abundance Measurements by K.J.R. Rosman, P.D.P. Taylor Pure Appl. Chem. 1999, 71, 1593-1607.

    Returns:
        dictionary of mass number, keys="isotope","mass_number","abundance"
    """
    mn = pkgutil.get_data('exojax', 'data/atom/iso_mn.txt')
    mnlist = pd.read_csv(BytesIO(mn), delimiter=',',
                         comment='#', usecols=(0, 1, 2))
    return mnlist

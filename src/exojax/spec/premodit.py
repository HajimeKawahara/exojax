"""Line profile computation using PremoDIT = Precomputation of LSD version of MODIT

"""
import numpy as np
from exojax.spec.lsd import npgetix
from exojax.utils.constants import hcperk

def make_initial_LSD(nu_grid, nu_lines, Tmax, elower, interval_contrast_lsd=1.0):
    """make initial LSD to compute the power spectrum of the LSD

    Args:
        nu_lines: wavenumber list of lines [Nline] (should be numpy F64)
        nu_grid: wavenumenr grid [Nnugrid] (should be numpy F64)
        Tmax: max temperature you will use.
        elower: E lower
        interval_contrast_lsd: interval contrast of line strength between upper and lower E lower grid

    Returns:
        contribution nu
        index nu
        contribution E lower
        index E lower


    """
    elower_grid=make_elower_grid(Tmax, elower, interval_contrast=interval_contrast_lsd)
    cont_inilsd_elower, index_inilsd_elower = npgetix(elower, elower_grid)
    cont_inilsd_nu, index_inilsd_nu = npgetix(nu_lines, nu_grid)
    return cont_inilsd_nu, index_inilsd_nu, cont_inilsd_elower, index_inilsd_elower



def compute_dElower(T,interval_contrast=0.1):
    """ compute a grid interval of Elower given the grid interval of line strength

    Args: 
        T: temperature in Kelvin
        interval_contrast: putting c = grid_interval_line_strength, then, the contrast of line strength between the upper and lower of the grid becomes c-order of magnitude.

    Returns:
        Required dElower
    """
    return interval_contrast*np.log(10.0)*T/hcperk

    
def make_elower_grid(Tmax, elower, interval_contrast):
    """compute E_lower grid given interval_contrast of line strength

    Args: 
        Tmax: max temperature in Kelvin
        elower: E_lower
        interval_contrast: putting c = grid_interval_line_strength, then, the contrast of line strength between the upper and lower of the grid becomes c-order of magnitude.

    Returns:
       grid of E_lower given interval_contrast

    """
    dE = compute_dElower(Tmax,interval_contrast)
    min_elower=np.min(elower)
    max_elower=np.max(elower)
    Ng_elower = int((max_elower - min_elower)/dE)+2
    return min_elower + np.arange(Ng_elower)*dE
    


    
    
def test_determine_initial_nugrid():
    print("test")
    
if __name__ == "__main__":
    print("premodit")
    print(compute_dElower(1000.0,interval_contrast=1.0))

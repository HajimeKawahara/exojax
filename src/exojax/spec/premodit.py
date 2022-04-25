"""Line profile computation using PremoDIT = Precomputation of LSD version of MODIT

"""
import numpy as np
from exojax.spec.lsd import npgetix
from exojax.utils.constants import hcperk

def make_initial_LSD(Tmax, nu_lines, elower):
    """

    Args:
       Tmax: max temperature you will use.
       blur_scale: log wavenumber scale to be blurred

    """

    
    c_elower, i_elower = npgetix(elower, elower_grid)

    q_lines=np.log(nu_lines)
    initial_q_grid = determine_initial_qgrid(q_lines, blur_scale)
    c_q_lines, i_q_lines = npgetix(q_lines, intial_q_grid)


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
    
def determine_initial_qgrid(q_lines, blur_scale):
    """ detemine the interval of the inital q grid

    Args:
        q_lines: the line center in the form of q = ln(nu [cm-1]), 
        blur_scale: 

    """
    print()



    
    
def test_determine_initial_nugrid():
    print("test")
    
if __name__ == "__main__":
    print("premodit")
    test_make_elower_grid()

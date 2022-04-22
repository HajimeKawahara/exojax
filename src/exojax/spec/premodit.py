"""Line profile computation using PremoDIT = Precomputation of LSD version of MODIT

"""
from exojax.spec.lsd import npgetix
def check_elower_grid_density(nu_lines, elower, elower_grid, blur_scale):
    """

    Args:
       blur_scale: log wavenumber scale to be blurred

    """
    c_elower, i_elower = npgetix(elower, elower_grid)

    initial_nu_grid = determine_initial_nugrid(nu_lines, blur_scale)
    c_nu_lines, i_nu_lines = npgetix(nu_lines, intial_nu_grid)
    
    
def determine_initial_nugrid(nu_lines, blur_scale):
    print("test")
    
    
def test_elower_grid_density():
    print("test")

def test_determine_initial_nugrid():
    print("test")
    
if __name__ == "__main__":
    print("premodit")

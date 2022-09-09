import numpy as np
from exojax.spec.lpf import 


def generate_voigt_shape_filter(spectral_resolution, filter_length):
    
    return

def check_filter_condition(shape_filter):
    """_summary_

    Args:
        shape_filter (_type_): _description_
    """
    if np.mod(len(shape_filter),2) == 0: 
        raise ValueError("shape filter length must be odd.")
        
def test_generate_voigt_shape_filter():
    return

if __name__ == "__main__":
    test_generate_voigt_shape_filter()
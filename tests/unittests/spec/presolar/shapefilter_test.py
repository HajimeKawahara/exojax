import numpy as np
from exojax.spec.lpf import voigt
from exojax.spec.presolar import compute_filter_length


def test_compute_filter_length():
    # example 50cm-1 tail cut at 4000cm-1
    spectral_resolution = 10**6
    filter_length = compute_filter_length(50.0, 4000.0, spectral_resolution)
    assert filter_length == 25001
    
def generate_voigt_shape_filter(spectral_resolution, filter_length):

    return


def check_filter_condition(shape_filter):
    """_summary_

    Args:
        shape_filter (_type_): _description_
    """
    if np.mod(len(shape_filter), 2) == 0:
        raise ValueError("shape filter length must be odd.")


def test_generate_voigt_shape_filter():
    return


if __name__ == "__main__":
    test_compute_filter_length()
from exojax.spec.api import _convert_proper_isotope

def test__convert_proper_isotope():
    assert _convert_proper_isotope(0) is None
    assert _convert_proper_isotope(1) == "1"
    assert _convert_proper_isotope(None) is None
    
if __name__ == "__main__":
    test__convert_proper_isotope()
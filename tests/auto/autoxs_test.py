import numpy
from exojax.spec import AutoXS
import pytest

@pytest.mark.parametrize(
    "nus", [
        (numpy.linspace(1900.0,2300.0,40000,dtype=numpy.float64)),
        (numpy.logspace(numpy.log10(1900.0),numpy.log10(2300.0),40000,dtype=numpy.float64))
    ]
)

def test_methods(nus):
    autoxs=AutoXS(nus,"ExoMol","CO",xsmode="DIT") #using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.  
    xsv0=autoxs.xsection(1000.0,1.0) #cross section for 1000K, 1bar (cm2)

    autoxs=AutoXS(nus,"ExoMol","CO",xsmode="LPF") #using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.  
    xsv1=autoxs.xsection(1000.0,1.0) #cross section for 1000K, 1bar (cm2)
    dif=(numpy.sum((xsv0-xsv1)**2))
    assert dif<1.e-36 


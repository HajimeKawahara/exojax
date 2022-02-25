"""tests for autoxs using different methods LPFDIT/MODIT."""

import numpy
from exojax.spec import AutoXS
import pytest


@pytest.mark.parametrize(
    'nus', [
        (numpy.linspace(1900.0, 2300.0, 40000, dtype=numpy.float64)),
        (numpy.logspace(numpy.log10(1900.0), numpy.log10(
            2300.0), 40000, dtype=numpy.float64))
    ]
)
def test_dit(nus):
    autoxs = AutoXS(nus, 'ExoMol', 'CO', xsmode='DIT')
    xsv0 = autoxs.xsection(1000.0, 1.0)

    autoxs = AutoXS(nus, 'ExoMol', 'CO', xsmode='LPF')
    xsv1 = autoxs.xsection(1000.0, 1.0)
    dif = (numpy.sum((xsv0-xsv1)**2))
    assert dif < 1.e-36


def test_modit():
    nus2 = numpy.logspace(numpy.log10(1900.0), numpy.log10(
        2300.0), 40000, dtype=numpy.float64)
    autoxs = AutoXS(nus2, 'ExoMol', 'CO', xsmode='MODIT')
    xsv0 = autoxs.xsection(1000.0, 1.0)
    autoxs = AutoXS(nus2, 'ExoMol', 'CO', xsmode='LPF')
    xsv1 = autoxs.xsection(1000.0, 1.0)
    dif = (numpy.sum((xsv0-xsv1)**2))
    assert dif < 1.e-36

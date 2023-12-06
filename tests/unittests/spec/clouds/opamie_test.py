"""opacity for mie test
"""
from exojax.test.emulate_pdb import mock_PdbPlouds
import numpy as np
import pytest


def test_interp_xsection():
    pdb = mock_PdbPlouds(nurange=[12000.0, 15000.0])
    pdb.load_miegrid()
    print(np.shape(pdb.miegrid))
    print(np.shape(pdb.refraction_index_wavenumber))


if __name__ == "__main__":
    test_interp_xsection()
"""test for partfn_Fe.

- Test polynomial expansion of the partition function of iron by Irwin1981
"""

import pytest
import numpy as np
from exojax.database.core_atom.pf import partfn_Fe


def test_partfn_Fe():
    tabulated = 5.62138956e0  # Table 1 of Irwin (1981)
    diff = np.log(partfn_Fe(16000)) - tabulated
    assert diff < 1e-8


if __name__ == '__main__':
    test_partfn_Fe()

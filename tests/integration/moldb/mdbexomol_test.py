import pytest
from exojax.spec.api import MdbExomol


def test_moldb_exomol():
    mdb = MdbExomol(".database/CO/12C-16O/Li2015",nurange=[4200.0,4300.0],crit=1.e-30)


if __name__ == "__main__":
    test_moldb_exomol()
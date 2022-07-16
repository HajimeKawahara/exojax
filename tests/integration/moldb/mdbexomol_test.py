import pytest
from exojax.spec.moldf import MdbExomol


def test_moldb_exomol():
    mdb = MdbExomol("CO/12C-16O/Li2015")


if __name__ == "__main__":
    test_moldb_exomol()
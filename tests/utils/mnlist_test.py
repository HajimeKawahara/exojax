import pytest
from exojax.utils.isodata import read_mnlist


def test_mnlist():
    arr=read_mnlist()
    assert arr["abundance"][0]==99.9885

if __name__ == '__main__':
    test_mnlist()

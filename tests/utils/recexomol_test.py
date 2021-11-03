import pytest
from exojax.utils.recexomol import get_exomol_database_list

def test_get_recexomol():
    db, db0=get_exomol_database_list("CO", "12C-16O")
    assert db0=="Li2015"

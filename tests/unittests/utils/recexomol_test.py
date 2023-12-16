import pytest
from exojax.utils.recexomol import get_exomol_database_list

# when the exomol server is down it fails 
def get_recexomol():
    db, db0 = get_exomol_database_list('CO', '12C-16O')
    assert db0 == 'Li2015'

if __name__ == "__main__":
    get_recexomol()
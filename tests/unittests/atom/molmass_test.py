"""test for molmass calculation"""

from exojax.spec.molinfo import molmass


if __name__ == '__main__':
    print(molmass('air'))
    print(molmass('CO2'))
    print(molmass('He'))
    print(molmass('CO2',db_HIT=True))
    print(molmass('He',db_HIT=True))

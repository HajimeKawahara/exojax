import pytest
from exojax.utils.molname import s2e_stable

def test_s2estable():
    EXOMOL_SIMPLE2EXACT= \
        {\
         "CO":"12C-16O",\
         "OH":"16O-1H",\
         "NH3":"14N-1H3",\
         "NO":"14N-16O",
         "FeH":"56Fe-1H",
         "H2S":"1H2-32S",
         "SiO":"28Si-16O",
         "CH4":"12C-1H4",
         "HCN":"1H-12C-14N",
         "C2H2":"12C2-1H2",
         "TiO":"48Ti-16O",
         "CO2":"12C-16O2",
         "CrH":"52Cr-1H",
         "H2O":"1H2-16O",
         "VO":"51V-16O",
         "CN":"12C-14N",
         "PN":"31P-14N",
        }

    check=True
    for i in EXOMOL_SIMPLE2EXACT:
        assert s2e_stable(i)==EXOMOL_SIMPLE2EXACT[i]
            
    
if __name__ == "__main__":
    test_s2estable()

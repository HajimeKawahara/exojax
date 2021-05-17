"""Definition of Default dataset for autospec

"""

from exojax.utils.molname import e2s

HITRAN_DEFMOL= \
{\
"CO":"05_hit12.par",\
"CH4":"06_hit12.par"\
} 

HITEMP_DEFMOL= \
{\
"N2O":"04_HITEMP2019.par.bz2",\
"CO":"05_HITEMP2019.par.bz2",\
"CH4":"06_HITEMP2020.par.bz2",\
"NO":"08_HITEMP2019.par.bz2",\
"NO2":"10_HITEMP2019.par.bz2",\
"OH":"13_HITEMP2020.par.bz2",\
} 

EXOMOL_DEFMOL= \
{\
 "12C-16O":"Li2015",\
 "16O-1H":"MoLLIST",
 "14N-1H3":"CoYuTe",
 "14N-16O":"NOname",
 "56Fe-1H":"MoLLIST",
 "1H2-32S":"AYT2",
 "28Si-16O2":"OYT3",
 "12C-1H4":"YT34to10",
 "1H-12C-14N":"Harris",
 "12C2-1H2":"aCeTY",
 "48Ti-16O":"Toto",
 "12C-16O2":"UCL-4000",
 "52Cr-1H":"MoLLIST",
 "1H2-16O":"POKAZATEL",
}

EXOMOL_SIMPLE2EXACT= \
{\
 "CO":"12C-16O",\
 "OH":"16O-1H",\
 "NH3":"14N-1H3",\
 "NO":"14N-16O",
 "FeH":"56Fe-1H",
 "H2S":"1H2-32S",
 "SiO":"28Si-16O2",
 "CH4":"12C-1H4",
 "HCN":"1H-12C-14N",
 "C2H2":"12C2-1H2",
 "TiO":"48Ti-16O",
 "CO2":"12C-16O2",
 "CrH":"52Cr-1H",
 "H2O":"1H2-16O",
}


def search_molfile(database,molecules):
    """name identifier of molecular databases

    Args:
       database: molecular database (HITRAN,HITEMP,ExoMol)
       molecules: molecular name such as (CO, 12C-16O)
    Returns:
       identifier

    """
    if database=="ExoMol":
        try:
            try:
                return e2s(molecules)+"/"+molecules+"/"+EXOMOL_DEFMOL[molecules]
            except:
                molname_exact=EXOMOL_SIMPLE2EXACT[molecules]
                print("Warning:",molecules,"is interpreted as",molname_exact)
                return molecules+"/"+molname_exact+"/"+EXOMOL_DEFMOL[molname_exact]
        except:
            return None

    elif database=="HITRAN":
        try:
            return HITRAN_DEFMOL[molecules]
        except:
            return None

    elif database=="HITEMP":
        try:
            return HITEMP_DEFMOL[molecules]
        except:
            return None

if __name__ == "__main__":
    print(search_molfile("ExoMol","12C-16O"))
    print(search_molfile("ExoMol","CO"))
    print(search_molfile("HITRAN","CO"))
    print(search_molfile("HITEMP","CO"))

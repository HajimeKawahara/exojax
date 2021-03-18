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

#12C-16O/Li2015/12C-16O__Li2015/
EXOMOL_DEFMOL= \
{\
 "12C-16O":"Li2015",\
 "16O-1H":"MoLLIST",
 "14N-1H3":"CoYuTe",
 "14N-16O":"NOname",
 "56Fe-1H":"MoLLIST",
}

EXOMOL_SIMPLE2EXACT= \
{\
 "CO":"12C-16O",\
 "OH":"16O-1H",\
 "NH3":"14N-1H3",\
 "NO":"14N-16O",
 "FeH":"56Fe-1H",
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

def hitran_dbpath(simple_molname):
    _hitran_dbpath = {
    "H2O": "H2O/01_hit12.par",
    "CH4": "CH4/06_hit12.par",
    "CO": "CO/05_hit12.par",
    "NH3": "NH3/11_hit12.par",
    "H2S": "H2S/31_hit12.par"
    }
    return _hitran_dbpath[simple_molname]

def hitemp_dbpath(simple_molname):
    _hitemp_dbpath = {
    "H2O": "H2O/01_HITEMP2010",
    "CH4": "CH4/06_HITEMP2020/06_HITEMP2020.par.bz2",
    "CO": "CO/05_HITEMP2019/05_HITEMP2019.par.bz2"
    }
    return _hitemp_dbpath[simple_molname]

def exomol_dbpath(simple_molname, databasepath=".database"):
    from exojax.utils.molname import simple_molname_to_exact_exomol_stable
    from radis.api.exomolapi import get_exomol_database_list
    exact_molname_exomol_stable = simple_molname_to_exact_exomol_stable(
        simple_molname)
    mlist, recommended = get_exomol_database_list(simple_molname,
                                                  exact_molname_exomol_stable)
    dbpath = databasepath + "/" + simple_molname + "/" + recommended
    return dbpath

def generate_database_directories(mols, db):
    dbpath_lookup = {"ExoMol": exomol_dbpath, "HITRAN": hitran_dbpath, "HITEMP": hitemp_dbpath, "exomol": exomol_dbpath, "hitran": hitran_dbpath, "hitemp": hitemp_dbpath}
    db_dir = []
    for mol_k, db_k in zip(mols, db):
        db_dir_k = []
        for mol_i, db_i in zip(mol_k, db_k):
            if db_i not in dbpath_lookup:
                raise ValueError(f"Unsupported database: {db_i}")

            dbpath_func = dbpath_lookup[db_i]
            dbpath = dbpath_func(mol_i)
            
            if dbpath is None:
                raise ValueError("db_dir not specified")

            db_dir_k.append(dbpath)

        db_dir.append(db_dir_k)

    return db_dir


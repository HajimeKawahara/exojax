import numpy as np
from exojax.spec import api
import os


def database_path_hitran(simple_molname):
    _hitran_dbpath = {
    "H2O": "H2O/01_hit12.par",
    "CH4": "CH4/06_hit12.par",
    "CO": "CO/05_hit12.par",
    "NH3": "NH3/11_hit12.par",
    "H2S": "H2S/31_hit12.par"
    }
    return _hitran_dbpath[simple_molname]

def database_path_hitemp(simple_molname):
    _hitemp_dbpath = {
    "H2O": "H2O/01_HITEMP2010",
    "CH4": "CH4/06_HITEMP2020/06_HITEMP2020.par.bz2",
    "CO": "CO/05_HITEMP2019/05_HITEMP2019.par.bz2"
    }
    return _hitemp_dbpath[simple_molname]

def database_path_exomol(simple_molname, root_database_path=".database"):
    from exojax.utils.molname import simple_molname_to_exact_exomol_stable
    from radis.api.exomolapi import get_exomol_database_list
    exact_molname_exomol_stable = simple_molname_to_exact_exomol_stable(
        simple_molname)
    mlist, recommended = get_exomol_database_list(simple_molname,
                                                  exact_molname_exomol_stable)
    dbpath = root_database_path + "/" + simple_molname + "/" + recommended
    return dbpath

def generate_database_directories(molecule_list, database_list):
    """generate database directory array

    Args:
        molecule_list (_type_): simple molecular list, e.g. mols = [["H2O","CH4","CO","NH3","H2S"]]
        database_list (_type_): database list db = [["ExoMol","HITEMP","HITEMP","HITRAN","HITRAN"]]


    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: database directory list
    """
    dbpath_lookup = {"ExoMol": database_path_exomol, "HITRAN": database_path_hitran, "HITEMP": database_path_hitemp, "exomol": database_path_exomol, "hitran": database_path_hitran, "hitemp": database_path_hitemp}
    db_dir = []
    for mol_k, db_k in zip(molecule_list, database_list):
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


########################
# REFACTORING NOTES (HK): CURRENTLY, api.Mdb is repreatedly called. We can replace it to the saved mdbs.
########################
def select_multimols_from_wavenumber_grid(path_data, mols, db, db_dir, nu_grid, crit=0., Ttyp=1000.):
    """select current multimols from wavenumber grid

    Args:
        path_data (_type_): _description_
        mols (_type_): _description_
        db (_type_): _description_
        db_dir (_type_): _description_
        nu_grid (_type_): _description_
        crit (_type_, optional): _description_. Defaults to 0..
        Ttyp (_type_, optional): _description_. Defaults to 1000..

    Returns:
        _type_: _description_
    """
    mdb = []
    for k, mol in enumerate(mols):
        mdb_k = []
        mask = np.ones_like(mol, dtype=bool)

        for i, item in enumerate(mol):
            try:
                if db[k][i] in ["ExoMol","exomol"]:
                    mdb_k.append(api.MdbExomol(os.path.join(path_data, db_dir[k][i]), nu_grid[k], crit=crit, Ttyp=Ttyp, gpu_transfer=False))
                elif db[k][i] in ["HITRAN", "hitran"]:
                    mdb_k.append(api.MdbHitran(os.path.join(path_data, db_dir[k][i]), nu_grid[k], crit=crit, Ttyp=Ttyp, gpu_transfer=False, isotope=1))
                elif db[k][i] in ["HITEMP", "hitemp"]:
                    mdb_k.append(api.MdbHitemp(os.path.join(path_data,db_dir[k][i]),nu_grid[k],crit=crit,Ttyp=Ttyp,gpu_transfer=False, isotope=1))
                    #mask_n = mdb_k[i].n_air > 0.01
                    #mdb_k[i] = apply_mask(mdb_k[i], mask_n)

            except:
                mask[i] = False
    
        mols[k] = np.array(mols[k])[mask].tolist()
        db[k] = np.array(db[k])[mask].tolist()
        db_dir[k] = np.array(db_dir[k])[mask].tolist()

        mdb.append(mdb_k)

    return mols, db, db_dir, mdb

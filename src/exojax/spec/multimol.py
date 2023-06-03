import numpy as np
from exojax.spec import api
import os


class MultiMol():
    """multimols molecular database of ExoMol.
    """
    def __init__(self, molmulti, dbmulti):
        self.molmulti = molmulti
        self.dbmulti = dbmulti
        self.generate_database_directories()

    def generate_database_directories(self):
        """generate database directory array

        Args:
            molecule_list (_type_): simple molecular list, e.g. mols = [["H2O","CH4","CO","NH3","H2S"]]
            database_list (_type_): database list db = [["ExoMol","HITEMP","HITEMP","HITRAN","HITRAN"]]


        Raises:
            ValueError: _description_
            ValueError: _description_

        """
        dbpath_lookup = {
            "ExoMol": database_path_exomol,
            "HITRAN": database_path_hitran12,
            "HITEMP": database_path_hitemp,
            "exomol": database_path_exomol,
            "hitran": database_path_hitran12,
            "hitemp": database_path_hitemp
        }
        self.db_dirs = []
        for mol_k, db_k in zip(self.molmulti, self.dbmulti):
            db_dir_k = []
            for mol_i, db_i in zip(mol_k, db_k):
                if db_i not in dbpath_lookup:
                    raise ValueError(f"Unsupported database: {db_i}")

                dbpath_func = dbpath_lookup[db_i]
                dbpath = dbpath_func(mol_i)

                if dbpath is None:
                    raise ValueError("db_dirs not specified")

                db_dir_k.append(dbpath)

            self.db_dirs.append(db_dir_k)
        #return db_dirs

    ########################
    # REFACTORING NOTES (HK): CURRENTLY, api.Mdb is repreatedly called. We can replace it to the saved mdbs.
    ########################
    def generate_multimdb(self, path_data, nu_grid, crit=0., Ttyp=1000.):
        """select current multimols from wavenumber grid

        Args:
            path_data (_type_): _description_
            nu_grid (_type_): _description_
            crit (_type_, optional): _description_. Defaults to 0..
            Ttyp (_type_, optional): _description_. Defaults to 1000..

        Returns:
            lists of mdb: multi mdb
            list: masked molecular list
        """
        multimdb = []
        masked_mols = self.molmulti[:]
        for k, mol in enumerate(self.molmulti):
            mdb_k = []
            mask = np.ones_like(mol, dtype=bool)

            for i, item in enumerate(mol):
                try:
                    if self.dbmulti[k][i] in ["ExoMol", "exomol"]:
                        mdb_k.append(
                            api.MdbExomol(os.path.join(path_data,
                                                       self.db_dirs[k][i]),
                                          nu_grid[k],
                                          crit=crit,
                                          Ttyp=Ttyp,
                                          gpu_transfer=False))
                    elif self.dbmulti[k][i] in ["HITRAN", "hitran"]:
                        mdb_k.append(
                            api.MdbHitran(os.path.join(path_data,
                                                       self.db_dirs[k][i]),
                                          nu_grid[k],
                                          crit=crit,
                                          Ttyp=Ttyp,
                                          gpu_transfer=False,
                                          isotope=1))
                    elif self.dbmulti[k][i] in ["HITEMP", "hitemp"]:
                        mdb_k.append(
                            api.MdbHitemp(os.path.join(path_data,
                                                       self.db_dirs[k][i]),
                                          nu_grid[k],
                                          crit=crit,
                                          Ttyp=Ttyp,
                                          gpu_transfer=False,
                                          isotope=1))
                except:
                    mask[i] = False

            masked_mols[k] = np.array(self.molmulti[k])[mask].tolist()
            multimdb.append(mdb_k)

        return multimdb, masked_mols


def database_path_hitran12(simple_molname):
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

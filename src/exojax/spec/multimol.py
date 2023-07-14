import numpy as np
from exojax.spec import api
import os


class MultiMol():
    """multiple molecular database handler

        Notes:
            MultiMol provides an easy way to generate multiple mdb (multimdb) and multiple opa (multiopa) 
            for multiple molecules/wavenumber segements.

        Attributes:
            molmulti: multiple simple molecule names [n_wavenumber_segments, n_molecules], such as [["H2O","CO"],["H2O"],["CO"]]
            dbmulti: multiple database names, such as [["HITEMP","EXOMOL"],["HITEMP","HITRAN12"]]]
            masked_molmulti: masked multiple simple molecule names [n_wavenumber_segments, n_molecules], such as [["H2O","CO"],["H2O"],[False]] Note that "False" is assigned when the code fails to get mdb because, for example, there are no transition lines for the specified condition.
            database_root_path: database root path
            db_dirs: database directories
            mols_unique: the list of the unique molecules, 
            mols_num: the same shape as self.masked_molmulti but gives indices of mols_unique

        Methods:
            multimdb: return multiple mdb 
            multiopa_premodit: return multiple opa for premodit
            molmass: return molecular mass list

    """
    def __init__(self, molmulti, dbmulti, database_root_path=".database"):
        """initialization

        Args:
            molmulti (_type_): multiple simple molecule names [n_wavenumber_segments, n_molecules], such as [["H2O","CO"],["H2O"],["CO"]]
            dbmulti (_type_): multiple database names, such as [["HITEMP","EXOMOL"],["HITEMP","HITRAN12"]]],
            database_root_path (str, optional): _description_. Defaults to ".database".
        """
        self.molmulti = molmulti
        self.dbmulti = dbmulti
        self.database_root_path = database_root_path
        self.generate_database_directories()
        

    def generate_database_directories(self):
        """generate database directory array
        """
        dbpath_lookup = {
            "ExoMol": database_path_exomol,
            "HITRAN12": database_path_hitran12,
            "HITEMP": database_path_hitemp,
            "exomol": database_path_exomol,
            "hitran12": database_path_hitran12,
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

    

    def multimdb(self, nu_grid_list, crit=0., Ttyp=1000.):
        """select current multimols from wavenumber grid

        Notes:
            multimdb() also generates self.masked_molmulti (masked molmulti), self.mols_unique (unique molecules), 
            and self.mols_num (same shape as self.masked_molmulti but gives indices of self.mols_unique)

        Args:
            nu_grid_list (_type_): _description_
            crit (_type_, optional): _description_. Defaults to 0..
            Ttyp (_type_, optional): _description_. Defaults to 1000..

        Returns:
            lists of mdb: multi mdb
        """
        _multimdb = []
        self.masked_molmulti = self.molmulti[:]
        for k, mol in enumerate(self.molmulti):

            mdb_k = []
            mask = np.ones_like(mol, dtype=bool)

            for i, item in enumerate(mol):
                try:
                    if self.dbmulti[k][i] in ["ExoMol", "exomol"]:
                        mdb_k.append(
                            api.MdbExomol(os.path.join(self.database_root_path,
                                                       self.db_dirs[k][i]),
                                          nu_grid_list[k],
                                          crit=crit,
                                          Ttyp=Ttyp,
                                          gpu_transfer=False))
                    elif self.dbmulti[k][i] in ["HITRAN12", "hitran12"]:
                        mdb_k.append(
                            api.MdbHitran(os.path.join(self.database_root_path,
                                                       self.db_dirs[k][i]),
                                          nu_grid_list[k],
                                          crit=crit,
                                          Ttyp=Ttyp,
                                          gpu_transfer=False,
                                          isotope=1))
                    elif self.dbmulti[k][i] in ["HITEMP", "hitemp"]:
                        mdb_k.append(
                            api.MdbHitemp(os.path.join(self.database_root_path,
                                                       self.db_dirs[k][i]),
                                          nu_grid_list[k],
                                          crit=crit,
                                          Ttyp=Ttyp,
                                          gpu_transfer=False,
                                          isotope=1))
                except:
                    mask[i] = False

            self.masked_molmulti[k] = np.array(self.molmulti[k])[mask].tolist()
            _multimdb.append(mdb_k)
            self.derive_unique_molecules()
        
        return _multimdb

    def derive_unique_molecules(self):
        """derive unique molecules in masked_molmulti and set self.mols_unique and self.mols_num

        Notes:
            self.mols_unique is the list of the unique molecules, 
            and self.mols_num has the same shape as self.masked_molmulti but gives indices of self.mols_unique


        """
        self.mols_unique = []
        self.mols_num = []
        for k in range(len(self.masked_molmulti)):
            mols_num_k = []
            for i in range(len(self.masked_molmulti[k])):
                if self.masked_molmulti[k][i] in self.mols_unique:
                    mols_num_k.append(
                        self.mols_unique.index(self.masked_molmulti[k][i]))
                else:
                    self.mols_unique.append(self.masked_molmulti[k][i])
                    mols_num_k.append(
                        self.mols_unique.index(self.masked_molmulti[k][i]))
            self.mols_num.append(mols_num_k)


    def multiopa_premodit(self,
                          multimdb,
                          nu_grid_list,
                          auto_trange,
                          diffmode=2,
                          dit_grid_resolution=0.2):
        """multiple opa for PreMODIT

        Args:
            multimdb (): multimdb
            nu_grid_list (): wavenumber grid list
            auto_trange (optional): temperature range [Tl, Tu], in which line strength is within 1 % prescision. Defaults to None.
            diffmode (int, optional): _description_. Defaults to 2.
            dit_grid_resolution (float, optional): force to set broadening_parameter_resolution={mode:manual, value: dit_grid_resolution}), ignores broadening_parameter_resolution.
            
        Returns:
            _type_: _description_
        """
        from exojax.spec.opacalc import OpaPremodit
        multiopa = []
        for k in range(len(multimdb)):
            opa_k = []
            for i in range(len(multimdb[k])):
                opa_i = OpaPremodit(mdb=multimdb[k][i],
                                    nu_grid=nu_grid_list[k],
                                    diffmode=diffmode,
                                    auto_trange=auto_trange,
                                    dit_grid_resolution=dit_grid_resolution)
                opa_k.append(opa_i)
            multiopa.append(opa_k)

        return multiopa

    def molmass(self):
        """return molecular mass list and H and He 

        Returns:
            molmass_list: molecular mass list for self.mols_unique
            molmassH2: molecular mass for hydorogen 
            molmassHe: molecular mass for helium
        """
        from exojax.spec import molinfo

        molmass_list = []
        for i in range(len(self.mols_unique)):
            molmass_list.append(molinfo.molmass(self.mols_unique[i]))
        molmassH2 = molinfo.molmass("H2")
        molmassHe = molinfo.molmass("He", db_HIT=False)

        return molmass_list, molmassH2, molmassHe

    

def database_path_hitran12(simple_molecule_name):
    """HITRAN12 default data path

    Args:
        simple_molecule_name (str): simple molecule name "H2O" 

    Returns:
        str: HITRAN12 default data path, such as "H2O/01_hit12.par" for "H2O"
    """
    from radis.db.classes import get_molecule_identifier
    ihitran = get_molecule_identifier(simple_molecule_name)
    return simple_molecule_name + "/" + str(ihitran).zfill(2) + "_hit12.par"


def database_path_hitemp(simple_molname):
    """default HITEMP path based on https://hitran.org/hitemp/

    Args:
        simple_molecule_name (str): simple molecule name "H2O" 

    Returns:
        str: HITEMP default data path, such as "H2O/01_HITEMP2010" for "H2O"
    """
    _hitemp_dbpath = {
        "H2O": "H2O/01_HITEMP2010",
        "CO2": "CO2/02_HITEMP2010",
        "N2O": "N2O/04_HITEMP2019/04_HITEMP2019.par.bz2",
        "CO": "CO/05_HITEMP2019/05_HITEMP2019.par.bz2",
        "CH4": "CH4/06_HITEMP2020/06_HITEMP2020.par.bz2",
        "NO": "NO/08_HITEMP2019/08_HITEMP2019.par.bz2",
        "NO2": "NO2/10_HITEMP2019/10_HITEMP2019.par.bz2",
        "OH": "OH/13_HITEMP2020/13_HITEMP2020.par.bz2"
    }
    return _hitemp_dbpath[simple_molname]


def database_path_exomol(simple_molecule_name):
    """default ExoMol path  

    Args:
        simple_molecule_name (str): simple molecule name "H2O" 

    Returns:
        str: Exomol default data path
    """
    from exojax.utils.molname import simple_molname_to_exact_exomol_stable
    from radis.api.exomolapi import get_exomol_database_list
    exact_molname_exomol_stable = simple_molname_to_exact_exomol_stable(
        simple_molecule_name)
    mlist, recommended = get_exomol_database_list(simple_molecule_name,
                                                  exact_molname_exomol_stable)
    dbpath = simple_molecule_name + "/" + exact_molname_exomol_stable + "/" + recommended
    return dbpath

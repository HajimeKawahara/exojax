import os
import traceback

import numpy as np

from exojax.database.contracts import MDBSnapshot
from exojax.opacity import OpaPremodit

# Lazily imported RADIS-backed classes to reduce import-time pressure.
MdbExomol = None
MdbHitran = None
MdbHitemp = None
mock_mdbExomol = None


def _load_mdb_exomol():
    global MdbExomol
    if MdbExomol is None:
        from exojax.database.exomol.api import MdbExomol as _MdbExomol

        MdbExomol = _MdbExomol
    return MdbExomol


def _load_mdb_hitran():
    global MdbHitran
    if MdbHitran is None:
        from exojax.database.hitran.api import MdbHitran as _MdbHitran

        MdbHitran = _MdbHitran
    return MdbHitran


def _load_mdb_hitemp():
    global MdbHitemp
    if MdbHitemp is None:
        from exojax.database.hitemp.api import MdbHitemp as _MdbHitemp

        MdbHitemp = _MdbHitemp
    return MdbHitemp


def _load_mock_mdb_exomol():
    global mock_mdbExomol
    if mock_mdbExomol is None:
        from exojax.test.emulate_mdb import mock_mdbExomol as _mock_mdbExomol

        mock_mdbExomol = _mock_mdbExomol
    return mock_mdbExomol


class MultiMDBCollection(list):
    """List-like container for selected MDB instances.

    Provides ``to_snapshot`` so downstream code can switch to the snapshot
    strategy without losing backwards compatibility with list semantics.
    """

    payload_kind = "mdb"

    def __init__(self, nested_mdbs):
        super().__init__(nested_mdbs)

    def to_snapshot(self):
        snapshot_rows = []
        for seg in self:
            seg_snapshots = []
            for mdb in seg:
                if not hasattr(mdb, "to_snapshot"):
                    raise AttributeError(
                        f"{type(mdb).__name__} does not implement to_snapshot()."
                    )
                seg_snapshots.append(mdb.to_snapshot())
            snapshot_rows.append(seg_snapshots)
        return MultiMDBSnapshot(snapshot_rows)


class MultiMDBSnapshot(list):
    """List-like container holding MDBSnapshot payloads."""

    payload_kind = "snapshot"

    def __init__(self, nested_snapshots):
        super().__init__(nested_snapshots)

    def to_snapshot(self):
        """Allow idempotent chaining."""
        return self


class MultiMol:
    """multiple molecular database and opacity calculator handler (multi Mdb/Opa Listing)

    Notes:
        MultiMol provides an easy way to generate multiple mdb (multiapi.mdb) and multiple opa (multiopa)
        for multiple molecules/wavenumber segments/stitching.

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
        """initialization of multimol

        Args:
            molmulti (nested list): multiple simple molecule names, such as [["H2O","CO"],["H2O"],["CO"]]
            dbmulti (nested list): multiple database names, such as [["HITEMP","EXOMOL"],["HITEMP"],["HITRAN12"]]
            database_root_path (str, optional): database root path. Defaults to ".database".
        """
        self.molmulti = molmulti
        self.dbmulti = dbmulti

        if self._check_structure(self.molmulti, self.dbmulti):
            pass
        else:
            print("molmulti=", molmulti, "dbmulti=", dbmulti)
            raise ValueError("molmulti and dbmulti have different structures")

        self.database_root_path = database_root_path
        self.generate_database_directories()

    def _check_structure(self, a, b):
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            return all(
                self._check_structure(sub_a, sub_b) for sub_a, sub_b in zip(a, b)
            )
        return not isinstance(a, list) and not isinstance(b, list)

    def _prepare_nu_grid_list(self, nu_grid_input):
        """Normalize nu_grid input to match the segment structure."""
        if isinstance(nu_grid_input, list):
            grids = nu_grid_input
        elif isinstance(nu_grid_input, tuple):
            grids = list(nu_grid_input)
        else:
            grids = [nu_grid_input]

        if len(grids) != len(self.molmulti):
            raise ValueError(
                "nu_grid_list must have the same number of segments as molmulti "
                f"(expected {len(self.molmulti)}, got {len(grids)})"
            )
        return grids

    def generate_database_directories(self):
        """generate database directory array"""
        dbpath_lookup = {
            "ExoMol": lambda mol: database_path_exomol(mol, self.database_root_path),
            "HITRAN12": database_path_hitran12,
            "HITEMP": database_path_hitemp,
            "exomol": lambda mol: database_path_exomol(mol, self.database_root_path),
            "hitran12": database_path_hitran12,
            "hitemp": database_path_hitemp,
            "SAMPLE": database_path_sample,
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

    def multimdb(self, nu_grid_list, crit=0.0, Ttyp=1000.0):
        """select current multimols from wavenumber grid

        Notes:
            multimdb() also generates self.masked_molmulti (masked molmulti), self.mols_unique (unique molecules),
            and self.mols_num (same shape as self.masked_molmulti but gives indices of self.mols_unique)

        Args:
            nu_grid_list (list): list of wavelength grids
            crit (float, optional): line strength criterion. Defaults to 0..
            Ttyp (float, optional): Typical temperature. Defaults to 1000..

        Returns:
            lists of mdb: multi mdb
        """
        nu_grid_segments = self._prepare_nu_grid_list(nu_grid_list)

        _multimdb = []
        self.masked_molmulti = self.molmulti[:]
        for k, mol in enumerate(self.molmulti):

            mdb_k = []
            mask = np.ones_like(mol, dtype=bool)

            for i, simple_molecule_name in enumerate(mol):
                print("Sets mdb for ", simple_molecule_name)
                try:
                    if self.dbmulti[k][i] in ["ExoMol", "exomol"]:
                        mdb_exomol_class = _load_mdb_exomol()
                        mdb_k.append(
                            mdb_exomol_class(
                                os.path.join(
                                    self.database_root_path, self.db_dirs[k][i]
                                ),
                                nu_grid_segments[k],
                                crit=crit,
                                Ttyp=Ttyp,
                                gpu_transfer=False,
                                broadf_download=False,
                            )
                        )
                    elif self.dbmulti[k][i] in ["HITRAN12", "hitran12"]:
                        mdb_hitran_class = _load_mdb_hitran()
                        mdb_k.append(
                            mdb_hitran_class(
                                os.path.join(
                                    self.database_root_path, self.db_dirs[k][i]
                                ),
                                nu_grid_segments[k],
                                crit=crit,
                                Ttyp=Ttyp,
                                gpu_transfer=False,
                                isotope=1,
                            )
                        )
                    elif self.dbmulti[k][i] in ["HITEMP", "hitemp"]:
                        mdb_hitemp_class = _load_mdb_hitemp()
                        mdb_k.append(
                            mdb_hitemp_class(
                                os.path.join(
                                    self.database_root_path, self.db_dirs[k][i]
                                ),
                                nu_grid_segments[k],
                                crit=crit,
                                Ttyp=Ttyp,
                                gpu_transfer=False,
                                isotope=1,
                            )
                        )
                    elif self.dbmulti[k][i] in ["SAMPLE"]:
                        mock_mdb_exomol_func = _load_mock_mdb_exomol()
                        mdb_k.append(mock_mdb_exomol_func(simple_molecule_name))

                except Exception as e:
                    if "No line found in " in e.args:
                        print(
                            self.molmulti[k][i],
                            self.dbmulti[k][i],
                            "in the range of",
                            e.args[1],
                            e.args[2],
                            "will be ignored due to no available lines found",
                        )
                        mask[i] = False
                    else:
                        print(traceback.format_exc())
                        exit()

            self.masked_molmulti[k] = np.array(self.molmulti[k])[mask].tolist()
            _multimdb.append(mdb_k)
            self.derive_unique_molecules()

        return MultiMDBCollection(_multimdb)

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
                        self.mols_unique.index(self.masked_molmulti[k][i])
                    )
                else:
                    self.mols_unique.append(self.masked_molmulti[k][i])
                    mols_num_k.append(
                        self.mols_unique.index(self.masked_molmulti[k][i])
                    )
            self.mols_num.append(mols_num_k)

    def multiopa_premodit(
        self,
        multimdb,
        nu_grid_list,
        auto_trange,
        nstitch_list=None,
        diffmode=0,
        dit_grid_resolution=0.2,
        allow_32bit=False,
    ):
        """multiple opa for PreMODIT

        Args:
            multimdb (): multimdb
            nu_grid_list (): wavenumber grid list
            auto_trange (list): temperature range [Tl, Tu], in which line strength is within 1 % prescision. Defaults to None.
            nstitch_list (list): The list of the number of nu-stitching segments for nu_grid_list (same structure). If None, no nu-stitching.
            diffmode (int, optional): _description_. Defaults to 0.
            dit_grid_resolution (float, optional): force to set broadening_parameter_resolution={mode:manual, value: dit_grid_resolution}), ignores broadening_parameter_resolution.

        Returns:
            _type_: _description_
        """

        nu_grid_segments = self._prepare_nu_grid_list(nu_grid_list)

        if nstitch_list is not None:
            self._check_structure(nu_grid_segments, nstitch_list)
            self.nstitch_list = nstitch_list
        else:
            self.nstitch_list = [1] * len(nu_grid_segments)
        del nstitch_list

        multiopa = []
        for k_nuseg in range(len(multimdb)):
            opa_k = []
            for i_mol in range(len(multimdb[k_nuseg])):
                opa_i = self.store_single_opa(
                    multimdb[k_nuseg][i_mol],
                    nu_grid_segments[k_nuseg],
                    auto_trange,
                    diffmode,
                    dit_grid_resolution,
                    allow_32bit,
                    self.nstitch_list[k_nuseg],
                )
                opa_k.append(opa_i)
            multiopa.append(opa_k)

        return multiopa

    def store_single_opa(
        self,
        multimdb_each,
        nu_grid_list_seg,
        auto_trange,
        diffmode,
        dit_grid_resolution,
        allow_32bit,
        nstitch,
    ):
        opa_kwargs = {
            "diffmode": diffmode,
            "auto_trange": auto_trange,
            "dit_grid_resolution": dit_grid_resolution,
            "allow_32bit": allow_32bit,
            "nstitch": nstitch,
        }

        if isinstance(multimdb_each, MDBSnapshot):
            return OpaPremodit.from_snapshot(multimdb_each, nu_grid_list_seg, **opa_kwargs)
        if hasattr(multimdb_each, "to_snapshot"):
            return OpaPremodit.from_mdb(multimdb_each, nu_grid_list_seg, **opa_kwargs)

        # Legacy path for custom MDB implementations without snapshot support.
        return OpaPremodit(
            mdb=multimdb_each,
            nu_grid=nu_grid_list_seg,
            **opa_kwargs,
        )

    def molmass(self):
        """return molecular mass list and H and He

        Returns:
            molmass_list: molecular mass list for self.mols_unique
            molmassH2: molecular mass for hydorogen
            molmassHe: molecular mass for helium
        """
        from exojax.database import molinfo 

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
    from exojax.database._common.radis_adapter import get_molecule_identifier

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
        "CO2": "CO2/02_HITEMP2024/02_HITEMP2024.par.bz2",
        "N2O": "N2O/04_HITEMP2019/04_HITEMP2019.par.bz2",
        "CO": "CO/05_HITEMP2019/05_HITEMP2019.par.bz2",
        "CH4": "CH4/06_HITEMP2020/06_HITEMP2020.par.bz2",
        "NO": "NO/08_HITEMP2019/08_HITEMP2019.par.bz2",
        "NO2": "NO2/10_HITEMP2019/10_HITEMP2019.par.bz2",
        "OH": "OH/13_HITEMP2020/13_HITEMP2020.par.bz2",
    }
    return _hitemp_dbpath[simple_molname]


def database_path_exomol(simple_molecule_name, database_root_path=None):
    """default ExoMol path

    Args:
        simple_molecule_name (str): simple molecule name "H2O"
        database_root_path (str, optional): base directory that already
            contains molecule/exact folders. Used to detect offline datasets.

    Returns:
        str: Exomol default data path
    """
    from exojax.utils.molname import simple_molname_to_exact_exomol_stable

    exact_molname_exomol_stable = simple_molname_to_exact_exomol_stable(
        simple_molecule_name
    )

    dataset_name = _discover_local_exomol_dataset(
        simple_molecule_name, exact_molname_exomol_stable, database_root_path
    )
    if dataset_name is None:
        dataset_name = _query_recommended_exomol_dataset(
            simple_molecule_name, exact_molname_exomol_stable
        )

    return f"{simple_molecule_name}/{exact_molname_exomol_stable}/{dataset_name}"


def _discover_local_exomol_dataset(simple_molecule_name, exact_name, root_path):
    """Return the first locally available dataset under the provided root."""
    if root_path is None:
        return None

    base_dir = os.path.join(root_path, simple_molecule_name, exact_name)
    if not os.path.isdir(base_dir):
        return None

    try:
        candidates = sorted(
            [
                entry
                for entry in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, entry))
            ]
        )
    except OSError:
        return None

    if not candidates:
        return None

    non_sample = [cand for cand in candidates if cand.upper() != "SAMPLE"]
    if non_sample:
        return non_sample[0]
    return candidates[0]


def _query_recommended_exomol_dataset(simple_molecule_name, exact_name):
    """Ask RADIS for the recommended dataset, propagating actionable errors."""
    from exojax.database._common.radis_adapter import get_exomol_database_list_func

    try:
        get_exomol_database_list = get_exomol_database_list_func()
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "radis.api.exomolapi is required to locate ExoMol data. "
            "Install RADIS or place the dataset under database_root_path."
        ) from exc

    from urllib.error import URLError

    try:
        _, recommended = get_exomol_database_list(simple_molecule_name, exact_name)
    except URLError as exc:
        raise RuntimeError(
            "Unable to reach ExoMol servers to determine the recommended dataset. "
            "Provide the files locally (e.g., database_root_path/CO/12C-16O/<dataset>)."
        ) from exc

    return recommended


def database_path_sample(simple_molname):
    """default SAMPLE (emulated mdb)

    Args:
        simple_molecule_name (str): simple molecule name "CO" or "H2O"

    Returns:
        str: HITEMP default data path, such as "H2O/01_HITEMP2010" for "H2O"
    """
    _sample_dbpath = {
        "H2O": "H2O/1H2-16O/SAMPLE",
        "CO": "CO/12C-16O/SAMPLE",
    }
    return _sample_dbpath[simple_molname]

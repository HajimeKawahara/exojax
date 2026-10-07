import os
import warnings

# Lazily imported RADIS-backed classes to reduce import-time pressure.
MdbExomol = None
MdbHitran = None
MdbHitemp = None


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
        masked_molmulti: Names with selected lines in each segment. This legacy
            list describes line availability, not the atmospheric composition.
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
        if not self._check_structure(molmulti, dbmulti):
            raise ValueError("molmulti and dbmulti have different structures")
        self.molmulti = [list(row) for row in molmulti]
        self.dbmulti = [list(row) for row in dbmulti]

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

    def _load_single_mdb(self, database, directory, nu_grid, crit, Ttyp):
        """Apply common loading options and each provider's fixed defaults."""
        options = {"crit": crit, "Ttyp": Ttyp, "gpu_transfer": False}
        if database in ("ExoMol", "exomol"):
            mdb_class = _load_mdb_exomol()
            options["broadf_download"] = False
        elif database in ("HITRAN12", "hitran12"):
            mdb_class = _load_mdb_hitran()
            options["isotope"] = 1
        elif database in ("HITEMP", "hitemp"):
            mdb_class = _load_mdb_hitemp()
            options["isotope"] = 1
        else:
            raise ValueError(f"Unsupported database: {database}")
        return mdb_class(
            os.path.join(self.database_root_path, directory), nu_grid, **options
        )

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

        mdb_rows = []
        masked_molmulti = []
        segments = zip(self.molmulti, self.dbmulti, self.db_dirs, nu_grid_segments)
        for k, (molecules, databases, directories, nu_grid) in enumerate(segments):
            mdb_row = []
            selected_names = []
            for name, database, directory in zip(molecules, databases, directories):
                print("Sets mdb for ", name)
                try:
                    mdb = self._load_single_mdb(database, directory, nu_grid, crit, Ttyp)
                except ValueError as e:
                    if e.args and e.args[0] == "No line found in ":
                        warnings.warn(
                            f"{name} ({database}) has no "
                            f"selected lines in segment {k}; omitted from the legacy "
                            "MDB list, not from atmospheric composition.",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue
                    else:
                        raise
                mdb_row.append(mdb)
                selected_names.append(name)

            masked_molmulti.append(selected_names)
            mdb_rows.append(mdb_row)

        self.masked_molmulti = masked_molmulti
        self.derive_unique_molecules()
        return MultiMDBCollection(mdb_rows)

    def derive_unique_molecules(self):
        """derive unique molecules in masked_molmulti and set self.mols_unique and self.mols_num

        Notes:
            self.mols_unique is the list of the unique molecules,
            and self.mols_num has the same shape as self.masked_molmulti but gives indices of self.mols_unique


        """
        molecule_indices = {}
        mols_num = []
        for molecules in self.masked_molmulti:
            mols_num.append([
                molecule_indices.setdefault(name, len(molecule_indices))
                for name in molecules
            ])
        self.mols_unique = list(molecule_indices)
        self.mols_num = mols_num

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
        """Compatibility wrapper for the opacity-layer nested-list builder."""
        warnings.warn(
            "MultiMol.multiopa_premodit is deprecated. Use "
            "exojax.opacity.multimol.multiopa_premodit for legacy lists, or "
            "exojax.opacity.build_premodit for named species.",
            DeprecationWarning,
            stacklevel=2,
        )
        from exojax.opacity.multimol import multiopa_premodit

        grids = self._prepare_nu_grid_list(nu_grid_list)
        result = multiopa_premodit(
            multimdb,
            grids,
            auto_trange,
            nstitch_list=nstitch_list,
            diffmode=diffmode,
            dit_grid_resolution=dit_grid_resolution,
            allow_32bit=allow_32bit,
        )
        self.nstitch_list = [1] * len(grids) if nstitch_list is None else list(nstitch_list)
        return result

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
        """Compatibility wrapper for constructing a single PreMODIT opacity."""
        warnings.warn(
            "MultiMol.store_single_opa is deprecated. Use "
            "OpaPremodit.from_mdb or OpaPremodit.from_snapshot.",
            DeprecationWarning,
            stacklevel=2,
        )
        from exojax.opacity.multimol import multiopa_premodit

        return multiopa_premodit(
            [[multimdb_each]],
            [nu_grid_list_seg],
            diffmode=diffmode,
            auto_trange=auto_trange,
            dit_grid_resolution=dit_grid_resolution,
            allow_32bit=allow_32bit,
            nstitch_list=[nstitch],
        )[0][0]

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

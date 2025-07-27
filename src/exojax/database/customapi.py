import copy
import pathlib
import pkgutil
import warnings
from io import BytesIO

import numpy as np
import pandas as pd

from exojax.database.api  import _set_engine
from exojax.database.dbmanager  import HargreavesDatabaseManager
from exojax.database.hitran  import line_strength_numpy
from exojax.database.qstate  import branch_to_number
from exojax.utils.constants import Tref_original, ccgs, hcperk
from exojax.utils.molname import e2s

HARGREAVES_URL = "https://content.cld.iop.org/journals/1538-3881/140/4/919/revision1/aj357217t5_mrt.txt"

def set_wavenum(nurange):
    """Set the wavenumber range for the database.

    Args:
        nurange (list): Wavenumber range for the database
    Returns:
        tuple: Minimum and maximum wavenumber
    """
    if nurange is None:
        wavenum_min = 0.0
        wavenum_max = 0.0
        warnings.warn("nurange=None.", UserWarning)
    else:
        wavenum_min, wavenum_max = np.min(nurange), np.max(nurange)
    if wavenum_min == -np.inf:
        wavenum_min = None
    if wavenum_max == np.inf:
        wavenum_max = None
    return wavenum_min, wavenum_max

def Einstein_coeff_from_line_strength(nu_lines, Sij, elower, g, Q, T):
    """Einstein coefficient from line strength

    Args:
        nu_lines (float): line center wavenumber (cm-1)
        Sij (float): line strength (cm)
        elower (float): elower
        g (float): upper state statistical weight
        Q (float): partition function
        T (float): temperature

    Returns:
        Einstein coefficient (s-1)
    """    
    Aij = - ((Sij * 8.0 * np.pi * ccgs * nu_lines**2 * Q) 
            / g
            / np.exp(-hcperk * elower / T) 
            / np.expm1(-hcperk * nu_lines / T))
    return Aij

class MdbHargreaves(HargreavesDatabaseManager):
    """molecular database of Hargreaves et al. (2010) in ExoMol form.
    
    Attributes:
        path (str): Path to the local database file
        database (str): Name of the database
        exact_molecule_name (str): Exact name of the molecule
        simple_molecule_name (str): Simple name of the molecule
        engine (str): Computational engine to use
        QTref_original (float): Original partition function at reference temperature
        QTref_raw (float): Raw partition function at reference temperature
        df (pd.DataFrame): Dataframe containing the converted data
    """
    def __init__(
            self,
            path,
            nurange=[-np.inf, np.inf],
            QTref_original=215.3488,
            QTref_raw=11691.1386, #5534.78,
            scale_intensity=1/3,
            engine=None,
            verbose=True,
            download=False,
            cache=True,
            ):
        """
        Args:
            path (str): Path to the local database file
            nurange (list): Wavenumber range for the database
            QTref_original (float): Partition function at reference temperature (Tref_original=296.0 K)
            QTref_raw (float): Partition function at reference temperature (Tref_raw=2200.0 K) of Hargreaves line list (default value is from Dulick et al. 2003)
            scale_intensity (float): correction for line intensity adopted in Sonora (see Marley et al. 2021; Morley et al. 2024)
            engine (str): Computational engine to use
            verbose (bool): Verbosity flag
            download (bool): if True, file will be downloaded from HARGREAVES_URL to path_to_database/FeH/Hargreaves2010/FeH-aj357217t5_mrt.h5. The default is False.
            cache (bool): Cache setting for file management
        """
        self.path = pathlib.Path(path).expanduser()
        self.database = str(self.path.stem)
        self.exact_molecule_name = self.path.parents[0].stem
        self.simple_molecule_name = e2s(self.exact_molecule_name)
        wavenum_min, wavenum_max = set_wavenum(nurange)
        self.nurange = [wavenum_min, wavenum_max]
        self.engine = _set_engine(engine)

        self.QTref_original = QTref_original
        self.QTref_raw = QTref_raw
        self.scale_intensity = scale_intensity

        if download:
            super().__init__(
                name="Hargreaves2010-{molecule}",
                molecule=self.simple_molecule_name,
                local_databases=self.path,
                url=HARGREAVES_URL,
                engine=self.engine,
                verbose=verbose,
            )

            assert cache  # cache only used for cache='regen' or cache='force' modes, cache=False is not expected
            
            local_files, urlnames = self.get_filenames()
    
            if cache == "regen":  
                self.remove_local_files(local_files)  
            
            self.check_deprecated_files(  
                self.get_existing_files(local_files),  
                auto_remove=True if cache != "force" else False,  
            )  
            
            get_main_files = True  
            if local_files and not self.get_missing_files(local_files):  
                get_main_files = False  
            
            # download files  
            if get_main_files:  
                for urlname, local_file in zip(urlnames, local_files):
                    self.parse_to_local_file(urlname, local_file) 
            
            clean_cache_files = True
            if get_main_files and clean_cache_files:
                    self.clean_download_files()

            # load the original data of Hargreaves et al. (2010)
            # columns: "wavenumber", "intensity", "e_lower", "einsteinA", "j_lower", "branch", "omega"
            df_raw = self.load(local_files, output=self.engine)
        else:
            Harg2010 = pkgutil.get_data("exojax", "data/opacity/FeH_Hargreaves2010.csv")
            df_raw = pd.read_csv(BytesIO(Harg2010), sep=",", comment="#")

        # convert to exomol format
        # columns: "A", "nu_lines", "elower", "jlower", "jupper", "Sij0", "gup"
        df = self.convert_to_exomol(df_raw)

        self.df = df
    
    def convert_to_exomol(self, df_raw):
        """Convert the raw (original) dataframe to ExoMol format."""
        # check if the wavenumber range of the raw data is within the specified range
        nurange_raw = [df_raw["wavenumber"].min(), df_raw["wavenumber"].max()]
        # Normalize self.nurange to handle None values
        nurange_min = self.nurange[0] if self.nurange[0] is not None else -np.inf
        nurange_max = self.nurange[1] if self.nurange[1] is not None else np.inf
        if (nurange_raw[1] < nurange_min) or (nurange_max < nurange_raw[0]):
            raise ValueError(f"No line found in {self.nurange} cm-1")

        df = self._convert_to_exomol(df_raw)
        return df

    def _convert_to_exomol(self, df_raw):
        """Convert the raw (original) dataframe to ExoMol format.
        Args:
            df_raw (pd.DataFrame): Original dataframe of Hargreaves et al. (2010)

        Returns:
            pd.DataFrame: Converted dataframe in ExoMol format
        """
        df_raw_sorted = df_raw.sort_values(by=["wavenumber"]) # ascending order

        nu_lines = df_raw_sorted["wavenumber"].values

        # elower for unassigned lines it is simply an average of all the lower state energies of the assigned lines (Hargreaves et al. 2010)
        elower = df_raw_sorted["e_lower"].values 

        # scale_intensity: correction for line intensity adopted in Sonora (see Marley et al. 2021; Morley et al. 2024)
        intensity_raw = self.scale_intensity * df_raw_sorted["intensity"].values
        A_raw = self.scale_intensity * df_raw_sorted["einsteinA"].values

        jlower = df_raw_sorted["j_lower"].values
        jlower[np.isnan(jlower)] = np.nanmean(df_raw_sorted["j_lower"].unique()) #set constant value

        branch = branch_to_number(df_raw_sorted["branch"].values, fillvalue=1) #set R branch
        jupper = jlower + branch

        gns = 2 #nuclear spin degeneracy of FeH
        gupper = gns * (2*jupper + 1)

        # convert line strength to Tref_original=296.0 K
        Tref_raw = 2200 # K
        #QTref_raw = 5534.78 # Dulick et al. 2003
        qr = self.QTref_original / self.QTref_raw
        Sij0 = line_strength_numpy(Tref_original, intensity_raw, nu_lines, elower, qr, Tref=Tref_raw)

        A = A_raw.copy()
        #A_est = Einstein_coeff_from_line_strength(nu_lines, Sij0, elower, g, self.QTref_original, Tref_original)
        A_est = Einstein_coeff_from_line_strength(nu_lines, intensity_raw, elower, gupper, self.QTref_raw, Tref_raw)
        A[np.isnan(A)] = A_est[np.isnan(A)] #set estimated values

        df_new = pd.DataFrame({
            "A": A,
            "nu_lines": nu_lines,
            "elower": elower,
            "jlower": jlower,
            "jupper": jupper,
            "Sij0": Sij0,
            "gup": gupper
        })

        return df_new
    
    def activate_with_exomol(self, mdb_exomol, extend=True):
        """Activate the Hargreaves database with an existing ExoMol database.
        Args:
            mdb_exomol (MdbExomol): An existing ExoMol database to extend
            extend (bool): If True, extend the existing database with Hargreaves data
        Returns:
            MdbExomol: A new ExoMol database with Hargreaves data activated
        """
        if extend:
            # MoLLIST (ExoMol) + Hargreaves
            if not hasattr(mdb_exomol, "df"):
                raise ValueError("The mdb_exomol must have a df attribute. Do not activate MdbExomol.")
            df_activate = pd.concat([mdb_exomol.df, self.df], ignore_index=True)
            df_activate = df_activate.sort_values(by=["nu_lines"])  # sort by nu_lines
        else:
            # Hargreaves only
            df_activate = self.df
        # avoid to modify the original mdb_exomol
        mdb_exomol_cp = copy.deepcopy(mdb_exomol)
        mdb_exomol_cp.df = df_activate
        mdb_exomol_cp.df_load_mask = mdb_exomol_cp.compute_load_mask(df_activate) # need nurange
        mdb_exomol_cp.activate(df_activate)
        return mdb_exomol_cp


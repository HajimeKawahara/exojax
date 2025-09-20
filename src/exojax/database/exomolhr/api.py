import numpy as np
from exojax.utils.molname import e2s
from exojax.utils.url import url_lists_exomolhr
from exojax.database.molinfo import isotope_molmass
from exojax.database._common.isotope_functions import _list_isotopologues
from exojax.database.exomolhr._downloader import _load_exomolhr_csv
from exojax.database.exomolhr._downloader import _list_exomolhr_molecules
from exojax.database.exomolhr._downloader import _fetch_opacity_zip


EXOMOLHR_HOME, EXOMOLHR_API_ROOT, EXOMOLHR_DOWNLOAD_ROOT = url_lists_exomolhr()

list_exomolhr_molecules = _list_exomolhr_molecules
list_exomolhr_isotopes = _list_isotopologues
class XdbExomolHR:
    """XdbExomolHR class for ExomolHR database

    Warnings:
        XdbExomolHR is not MDB.

    Notes:
        The ExomolHR database (eXtra db) is emprical high-res line strengths/info for a given single temperature.
        Xdb is a database that does not belong to regular types of ExoJAX databases.

    Attributes:
        simple_molecule_name: simple molecule name
        nurange: nu range [min,max] (cm-1)
        nu_lines (nd array): line center (cm-1)
        Sij0 (nd array): line strength at T=Tref (cm)
        logsij0 (jnp array): log line strength at T=Tref
        A (jnp array): Einstein A coeeficient
        gamma_natural (DataFrame or jnp array): gamma factor of the natural broadening
        elower (DataFrame or jnp array): the lower state energy (cm-1)
        gpp (DataFrame or jnp array): statistical weight
        jlower (DataFrame or jnp array): J_lower
        jupper (DataFrame or jnp array): J_upper
        n_Texp (DataFrame or jnp array): temperature exponent
        dev_nu_lines (jnp array): line center in device (cm-1)


    """

    def __init__(
        self,
        exact_molecule_name,
        nurange,
        temperature,
        crit=1.0e-40,
        gpu_transfer=True,
        activation=True,
        inherit_dataframe=False,
        local_databases="./opacity_zips",
    ):
        """Molecular database for ExomolHR.

        Args:
            path: path for Exomol data directory/tag. For instance, "/home/CO/12C-16O/Li2015"
            nurange: wavenumber range list (cm-1) [min,max] or wavenumber grid, if None, it starts as the nonactive mode
            temperature: temperature in Kelvin
            crit: line strength lower limit for extraction
            gpu_transfer: if True, some attributes will be transfered to jnp.array. False is recommended for PreMODIT.
            inherit_dataframe: if True, it makes self.df attribute available, which needs more DRAM when pickling.
            activation: if True, the activation of mdb will be done when initialization, if False, the activation won't be done and it makes self.df attribute available.
            local_databases: path for local databases, default is "./opacity_zips"

        """
        self.dbtype = "exomolhr"
        self.exact_molecule_name = exact_molecule_name
        self.gpu_transfer = gpu_transfer
        self.crit = crit
        self.temperature = temperature
        self.local_databases = local_databases

        self.simple_molecule_name = e2s(self.exact_molecule_name)
        self.molmass = isotope_molmass(self.exact_molecule_name)
        self.activation = activation
        self.wavenum_min, self.wavenum_max = np.min(nurange), np.max(nurange)
        self.nurange = nurange

        self.fetch_data()

        df = _load_exomolhr_csv(self.csv_path)
        if self.activation:
            self.activate(df)
        if inherit_dataframe or not self.activation:
            print("DataFrame (self.df) available.")
            self.df = df

    def fetch_data(self):
        self.csv_path = _fetch_opacity_zip(
            wvmin=0,
            wvmax=None,
            numin=self.wavenum_min,
            numax=self.wavenum_max,
            T=self.temperature,
            Smin=self.crit,
            iso=self.exact_molecule_name,
            out_dir=self.local_databases,
        )
        print("Downloaded and unzipped to", self.csv_path)

    def attributes_from_dataframes(self, df_masked):
        """Generates attributes from (usually masked) data frame for Exomol

        Args:
            df_masked (DataFrame): (masked) data frame

        Raises:
            ValueError: _description_
        """

        if len(df_masked) == 0:
            raise ValueError("No line found in ", self.nurange, "cm-1")

        self._attributes_from_dataframes(df_masked)

    def _attributes_from_dataframes(self, df_masked):
        """
        Notes:
            df_masked["S"] is the line strength at T (self.Ttyp)
        """
        self.A = df_masked["A"].values
        self.nu_lines = df_masked["nu"].values
        self.elower = df_masked['E"'].values
        self.jlower = df_masked['J"'].values
        self.jupper = df_masked["J'"].values
        self.line_strength = df_masked["S"].values
        self.logsij0 = np.log(self.line_strength)
        self.gpp = df_masked["g'"].values

    def activate(self, df, mask=None):
        """Activates of moldb for Exomol,  including making attributes, computing broadening parameters, natural width, and transfering attributes to gpu arrays when self.gpu_transfer = True

        Notes:
            activation includes, making attributes, computing broadening parameters, natural width,
            and transfering attributes to gpu arrays when self.gpu_transfer = True

        Args:
            df: DataFrame
            mask: mask of DataFrame to be used for the activation, if None, no additional mask is applied.

        Note:
            self.df_load_mask is always applied when the activation.

        Examples:

            >>> # we would extract the line with delta nu = 2 here
            >>> mdb = api.MdbExomolHR(emf, nus, optional_quantum_states=True, activation=False)
            >>> load_mask = (mdb.df["v_u"] - mdb.df["v_l"] == 2)
            >>> mdb.activate(mdb.df, load_mask)


        """
        if mask is not None:
            self.attributes_from_dataframes(df[mask])
        else:
            self.attributes_from_dataframes(df)


if __name__ == "__main__":
    mols = _list_exomolhr_molecules()  # downloads live HTML
    print(f"Currently {len(mols)} molecules are available:")
    print(", ".join(mols))
    iso_dict = _list_isotopologues(mols)
    print(iso_dict)

    from exojax.test.emulate_mdb import mock_wavenumber_grid

    nus, wav, res = mock_wavenumber_grid()

    csv_path = _fetch_opacity_zip(
        wvmin=0,
        wvmax=None,
        numin=0,
        numax=4000,
        T=1300,
        Smin=1e-40,
        iso="12C-16O2",
        out_dir="opacity_zips",
    )
    print("Downloaded and unzipped to", csv_path)

    df = _load_exomolhr_csv(csv_path)
    print(df.head())

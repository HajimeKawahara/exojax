import os
import re
import pathlib
import requests
import logging
import time
import numpy as np
from urllib.parse import urljoin
from typing import Sequence
from typing import Iterable
from bs4 import BeautifulSoup
import zipfile
import pandas as pd
import concurrent.futures as _cf


from exojax.utils.molname import e2s
from exojax.spec.molinfo import isotope_molmass
from exojax.utils.url import url_lists_exomolhr

EXOMOLHR_HOME, EXOMOLHR_API_ROOT, EXOMOLHR_DOWNLOAD_ROOT = url_lists_exomolhr()
_ISO_INPUT_RE = re.compile(r"^\d+[A-Z][a-z]?-\d+[A-Z][a-z]?.*$")  # e.g. 27Al-35Cl


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

        df = load_exomolhr_csv(self.csv_path)
        if self.activation:
            self.activate(df)
        if inherit_dataframe or not self.activation:
            print("DataFrame (self.df) available.")
            self.df = df

    def fetch_data(self):
        self.csv_path = fetch_opacity_zip(
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


def fetch_opacity_zip(  # noqa: WPS211 (a few branches are fine here)
    *,
    wvmin: float,
    wvmax: float | None,
    numin: float,
    numax: float,
    T: int,
    Smin: float,
    iso: str,
    out_dir: str | os.PathLike = ".",
    session: requests.Session | None = None,
    chunk: int = 1 << 19,  # 512 kB blocks
) -> pathlib.Path:
    """Return a local ExoMolHR CSV—downloaded only if necessary.

    The function skips the network step when a file with the same physics
    (identical *iso* and *T*) is already present in *out_dir*; only the
    timestamp differs between downloads.

    Args:
        wvmin / wvmax : float | None
            Viewer wavenumber limits (nm). Pass ``None`` to omit *wvmax*.
        numin / numax : float
            Line‑list limits in cm⁻¹ requested from the service.
        T : int
            Temperature [K].
        Smin : float
            Lower cut‑off for line strength (cm molecule⁻¹).
        iso : str
            Isotopologue tag, e.g. ``"12C-16O2"``.
        out_dir : str | os.PathLike, optional
            Directory for ZIP/CSV files (default ``"."``).
        session : requests.Session | None, optional
            Re‑use a ``requests.Session`` if supplied.
        chunk : int, optional
            Streaming block size in bytes (default 512 kB).

    Returns:
        pathlib.Path
            Path to the local CSV file.

    Raises:
        RuntimeError
            When the expected download link is missing or HTTP fails.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. reuse if the same physics file already exists
    # ------------------------------------------------------------------
    csv_suffix = f"__{iso}__{float(T):.1f}K.csv"  # 1200  -> 1200.0K
    existing = sorted(out_dir.glob(f"*{csv_suffix}"))
    if existing:  # return the most recent (arbitrary choice)
        return existing[-1]

    # ------------------------------------------------------------------
    # 1. build query and fetch HTML page
    # ------------------------------------------------------------------
    sess = session or requests.Session()
    query = {
        "wvmin": wvmin,
        **({} if wvmax is None else {"wvmax": wvmax}),
        "numin": numin,
        "numax": numax,
        "T": T,
        "Smin": Smin,
        "iso": iso,
    }
    html_resp = sess.get(EXOMOLHR_API_ROOT, params=query, timeout=120)
    html_resp.raise_for_status()

    # ------------------------------------------------------------------
    # 2. locate the ZIP link (may be relative)
    # ------------------------------------------------------------------
    soup = BeautifulSoup(html_resp.text, "html.parser")
    dl_tag = soup.find("a", href=re.compile(r"download/\?archive_name=.*\.zip"))
    if dl_tag is None:
        raise RuntimeError("No download link found – HTML layout may have changed.")

    dl_url = urljoin("https://www.exomol.com", dl_tag["href"])
    zip_name = re.search(r"archive_name=([^&]+\.zip)", dl_url).group(1)
    zip_path = out_dir / zip_name

    # ------------------------------------------------------------------
    # 3. download ZIP (streaming) unless it is already present
    # ------------------------------------------------------------------
    if not zip_path.exists():
        with sess.get(dl_url, stream=True, timeout=600) as resp, zip_path.open(
            "wb"
        ) as fh:
            resp.raise_for_status()
            for block in resp.iter_content(chunk):
                fh.write(block)

    # ------------------------------------------------------------------
    # 4. extract the single CSV and return its path
    # ------------------------------------------------------------------
    with zipfile.ZipFile(zip_path) as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            print("Check available molecules at https://www.exomol.com/exomolhr")
            raise RuntimeError("No CSV file found in the archive.")
        if len(csv_members) > 1:
            raise RuntimeError(f"Multiple CSV files found: {csv_members}")

        csv_name = csv_members[0]
        csv_path = out_dir / csv_name
        if not csv_path.exists():  # avoid overwriting if already extracted
            zf.extract(csv_name, path=out_dir)

    return csv_path


def load_exomolhr_csv(csv_path: str | pathlib.Path) -> pd.DataFrame:
    """Load a CSV file from an ExoMolHR ZIP archive into a DataFrame."""
    # ------------------------------------------------------------------
    # 3. read CSV into DataFrame
    # ------------------------------------------------------------------
    # The header contains embedded quotes (e.g. Gtot', "Gtot""")
    # Using the default C engine is fine; we only need to tell pandas
    # that double quotes inside a quoted field are escaped by doubling.
    df = pd.read_csv(
        csv_path,
        engine="python",  # more forgiving with odd quoting
        quotechar='"',
        doublequote=True,
        skipinitialspace=True,  # trims spaces after commas
    )

    # Optional cleanup: strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    return df


def list_exomolhr_molecules(
    html_source: str | bytes | pathlib.Path | None = None,
    *,
    session: requests.Session | None = None,
) -> Sequence[str]:
    """Return the list of molecule formulas shown on the ExoMolHR landing page.

    The function can work in three modes:

    1. **Online**  `html_source is None`
       → download *https://www.exomol.com/exomolhr/* live.
    2. **From file** `html_source` is a `pathlib.Path` or filename
       → read the saved HTML.
    3. **From string/bytes**  `html_source` is raw HTML content
       → parse directly.

    Args:
        html_source : str | bytes | pathlib.Path | None, optional
            Where to get the HTML.  Pass ``None`` (default) to fetch online.
        session : requests.Session | None, optional
            Re-use a session if you call repeatedly.

    Returns: Sequence[str]
        Formulas in the order they appear in the table
        (duplicates are removed).

    Raises
    ------
    RuntimeError
        If the molecule table cannot be located in the HTML.
    """
    # ------------------------------------------------------------------
    # 1. obtain the HTML text
    # ------------------------------------------------------------------
    if html_source is None:
        sess = session or requests.Session()
        resp = sess.get(EXOMOLHR_HOME, timeout=60)
        resp.raise_for_status()
        html_text = resp.text
    elif isinstance(html_source, (bytes, str)):
        # already HTML content
        html_text = (
            html_source.decode() if isinstance(html_source, bytes) else html_source
        )
    else:
        # assume a filesystem path
        html_text = pathlib.Path(html_source).read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # 2. parse and extract formulas
    # ------------------------------------------------------------------
    soup = BeautifulSoup(html_text, "html.parser")
    rows = soup.select("#dataTable tbody tr")
    if not rows:
        raise RuntimeError("Could not find the molecule table (id='dataTable').")

    formulas: list[str] = []
    for row in rows:
        first_td = row.find("td")
        if not first_td:
            continue
        formula = first_td.get_text(strip=True).replace(
            "\u200b", ""
        )  # strip zero-width spaces
        if formula and formula not in formulas:
            formulas.append(formula)

    return formulas


def _fetch_isos_for_one(
    molecule: str,
    *,
    session: requests.Session,
    timeout: float = 120.0,
) -> list[str]:
    """Return a list of isotopologue tags for *one* molecule.

    The HTML for a single-molecule query contains a group of check-boxes
    like:

        <input class="form-check-input iso-checkbox"
               type="checkbox" name="iso" value="27Al-35Cl">
               ...
        <input ... value="27Al-37Cl">

    We grab every checkbox with ``name="iso"`` whose value looks like
    ``{mass number}{Element}-{mass number}{Element}``.

    Notes
    -----
    * The server occasionally throttles rapid, repeated requests.  A small
      “courtesy sleep” keeps us polite.
    * If no isotopologue check-boxes are found **or** the pattern does not
      match the expected “27Al-35Cl” style, an empty list is returned
      instead of raising – this keeps the whole mapping operation robust.
    """
    url = f"{EXOMOLHR_HOME}?molecule={molecule}"
    time.sleep(0.3)  # be gentle with ExoMol servers

    r = session.get(url, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    candidates = soup.select('input[name="iso"]')
    return [
        inp["value"] for inp in candidates if _ISO_INPUT_RE.match(inp.get("value", ""))
    ]


# helper -------------------------------------------------------------
def _slug(molecule: str) -> str:
    """Turn H3+  -> H3_p   and  H3O+ -> H3O_p  (no change otherwise)."""
    return molecule.replace("+", "_p")


# fetch one molecule -------------------------------------------------
def _fetch_isos_for_one(
    molecule: str,
    *,
    session: requests.Session,
    timeout: float = 120.0,
) -> list[str]:
    url = EXOMOLHR_HOME  # "https://www.exomol.com/exomolhr/"
    resp = session.get(url, params={"molecule": _slug(molecule)}, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    iso_inputs = soup.select("input.iso-checkbox")

    wanted_class = f"{_slug(molecule)}-checkbox"
    tags = [
        inp["value"].strip()
        for inp in iso_inputs
        if wanted_class in inp.get("class", [])
    ]
    # preserve order, remove dups
    seen = set()
    return [t for t in tags if not (t in seen or seen.add(t))]


def list_isotopologues(
    simple_molecule_list: Iterable[str],
    *,
    max_workers: int | None = None,
) -> dict[str, list[str]]:
    """Return {molecule: [iso₁, iso₂, …]} for the given molecules.

    Args:
        simple_molecule_list: list of simple molecule names, e.g. [AlCl, AlH, AlO, C2, C2H2, CaH, CH4, CN, CO2]
        max_workers: number of workers for parallel processing

    Returns:
        dict[str, list[str]]: dictionary of isotopologues (simple_molecule_name:[list of exact_molecule_name]) for each molecule
        e.g. {'H2O': ['1H2-16O'], 'C2H2': ['12C2-1H2'], 'H3O+': ['1H3-16O_p']}
    """
    simple_molecule_list = list(dict.fromkeys(simple_molecule_list))
    iso_map: dict[str, list[str]] = {}

    with requests.Session() as sess, _cf.ThreadPoolExecutor(
        max_workers=max_workers
    ) as pool:
        fut_to_mol = {
            pool.submit(_fetch_isos_for_one, m, session=sess): m
            for m in simple_molecule_list
        }
        for fut in _cf.as_completed(fut_to_mol):
            mol = fut_to_mol[fut]
            try:
                iso_map[mol] = fut.result()
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to fetch %s: %s", mol, exc)
                iso_map[mol] = []

    return iso_map


if __name__ == "__main__":
    mols = list_exomolhr_molecules()  # downloads live HTML
    print(f"Currently {len(mols)} molecules are available:")
    print(", ".join(mols))
    iso_dict = list_isotopologues(mols)
    print(iso_dict)

    from exojax.test.emulate_mdb import mock_wavenumber_grid

    nus, wav, res = mock_wavenumber_grid()

    csv_path = fetch_opacity_zip(
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

    df = load_exomolhr_csv(csv_path)
    print(df.head())

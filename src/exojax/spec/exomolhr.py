import os
import re
import pathlib
import requests
import numpy as np
from urllib.parse import urljoin
from bs4 import BeautifulSoup  # pip install beautifulsoup4  (pure‑Python)
import zipfile
import pandas as pd



from exojax.utils.molname import e2s
from exojax.spec.molinfo import isotope_molmass


class MdbExomolHR:
    def __init__(
        self,
        path,
        nurange,
        Ttyp=1000.0,
        bkgdatm="H2",
        gpu_transfer=True,
        activation=True,
        local_databases="./",
    ):
        self.dbtype = "exomolhr"
        self.path = pathlib.Path(path).expanduser()
        self.exact_molecule_name = self.path.parents[0].stem
        self.database = str(self.path.stem)
        self.bkgdatm = bkgdatm
        # molecbroad = self.exact_molecule_name + '__' + self.bkgdatm
        self.gpu_transfer = gpu_transfer
        self.Ttyp = Ttyp
        self.simple_molecule_name = e2s(self.exact_molecule_name)
        self.molmass = isotope_molmass(self.exact_molecule_name)
        self.activation = activation
        wavenum_min, wavenum_max = np.min(nurange), np.max(nurange)


EXOMOLHR_API_ROOT = "https://www.exomol.com/exomolhr/get-data/"  # <- base for HTML
EXOMOLHR_DOWNLOAD_ROOT = "https://www.exomol.com/exomolhr/get-data/download/"

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
        with sess.get(dl_url, stream=True, timeout=600) as resp, zip_path.open("wb") as fh:
            resp.raise_for_status()
            for block in resp.iter_content(chunk):
                fh.write(block)

    # ------------------------------------------------------------------
    # 4. extract the single CSV and return its path
    # ------------------------------------------------------------------
    with zipfile.ZipFile(zip_path) as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError("No CSV file found in the archive.")
        if len(csv_members) > 1:
            raise RuntimeError(f"Multiple CSV files found: {csv_members}")

        csv_name = csv_members[0]
        csv_path = out_dir / csv_name
        if not csv_path.exists():  # avoid overwriting if already extracted
            zf.extract(csv_name, path=out_dir)

    return csv_path

def load_exomolhr_csv(csv_path: str | pathlib.Path) -> pd.DataFrame:
    """Load a CSV file from an ExoMolHR ZIP archive into a DataFrame.
    """
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


if __name__ == "__main__":
    csv_path = fetch_opacity_zip(
        wvmin=0,
        wvmax=None,
        numin=0,
        numax=2000,
        T=1300,
        Smin=1e-40,
        iso="12C-16O2",
        out_dir="opacity_zips",
    )
    print("Downloaded and unzipped to", csv_path)


    df = load_exomolhr_csv(csv_path)
    
    print(df.head())
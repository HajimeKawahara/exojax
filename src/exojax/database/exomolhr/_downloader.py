import os
import zipfile
from urllib.parse import urljoin
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pathlib
import re
from collections.abc import Sequence
from typing import Optional
from typing import Union
    
from exojax.utils.url import url_lists_exomolhr

EXOMOLHR_HOME, EXOMOLHR_API_ROOT, EXOMOLHR_DOWNLOAD_ROOT = url_lists_exomolhr()

def _fetch_opacity_zip(  # noqa: WPS211 (a few branches are fine here)
    *,
    wvmin: float,
    wvmax: Optional[float],
    numin: float,
    numax: float,
    T: int,
    Smin: float,
    iso: str,
    out_dir: Union[str, os.PathLike] = ".",
    session: Optional[requests.Session] = None,
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



def _load_exomolhr_csv(csv_path: Union[str, pathlib.Path]) -> pd.DataFrame:
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


def _list_exomolhr_molecules(
    html_source: Optional[Union[str, bytes, pathlib.Path]] = None,
    *,
    session: Optional[requests.Session] = None,
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

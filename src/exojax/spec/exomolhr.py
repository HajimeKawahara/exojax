import os
import re
import pathlib
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup   # pip install beautifulsoup4  (pure‑Python)

EXOMOLHR_API_ROOT = (
    "https://www.exomol.com/exomolhr/get-data/"        # <- base for HTML
)
EXOMOLHR_DOWNLOAD_ROOT = (
    "https://www.exomol.com/exomolhr/get-data/download/"
)

def fetch_opacity_zip(
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
    chunk: int = 1 << 19,            # 512 kB blocks
) -> pathlib.Path:
    """Download an ExoMolHR opacity archive and return the local filename.
    Only the final ZIP is written to disk; the HTML page is kept in memory.

    Args:
        wvmin / wvmax : float | None
            Viewer wavenumber limits (nm) – pass None to omit.
        numin / numax : float
            Line‑list limits in cm⁻¹ required by the service.
        T : int
            Temperature [K] used by ExoMolHR.
        Smin : float
            Lower cutoff for line strength (cm molecule⁻¹).
        iso : str
            Isotopologue tag, e.g. ``"12C-16O2"``.
        out_dir : pathlike
            Where to write the ZIP.  Created if necessary.
        session : requests.Session | None
            Provide a session to reuse TCP connections & cookies.
        chunk : int
            Streaming chunk size in bytes.

    Returns:
        pathlib.Path: Path to the downloaded ZIP.

    Raises:
        RuntimeError
        If no download link is found or HTTP fails.
    """
    sess = session or requests.Session()

    # --- 1. build query and fetch HTML page ---------------------------------
    query = {
        "wvmin": wvmin,
        "wvmax": "" if wvmax is None else wvmax,
        **({} if wvmax is None else {"wvmax": wvmax}),
        "numin": numin,
        "numax": numax,
        "T": T,
        "Smin": Smin,
        "iso": iso,
    }
    html_resp = sess.get(EXOMOLHR_API_ROOT, params=query, timeout=120)
    html_resp.raise_for_status()

    # --- 2. parse HTML to find “…download/?archive_name=YYYYMMDDhhmmss.zip” -
    soup = BeautifulSoup(html_resp.text, "html.parser")
    dl_tag = soup.find("a", href=re.compile(r"download/\?archive_name=.*\.zip"))
    if dl_tag is None:
        raise RuntimeError("No download link found – HTML layout may have changed?")

    dl_url = dl_tag["href"]  # already absolute
    dl_url = urljoin("https://www.exomol.com", dl_url)
    zip_name = re.search(r"archive_name=([^&]+\.zip)", dl_url).group(1)

    # --- 3. stream ZIP to disk ----------------------------------------------
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / zip_name

    with sess.get(dl_url, stream=True, timeout=600) as r, open(local_path, "wb") as f:
        r.raise_for_status()
        for block in r.iter_content(chunk):
            f.write(block)

    return local_path

if __name__ == "__main__":
    path = fetch_opacity_zip(
    wvmin=0, wvmax=None,
    numin=0, numax=2000,
    T=1200,
    Smin=1e-40,
    iso="12C-16O2",
    out_dir="opacity_zips",
)
    print("Downloaded to", path)
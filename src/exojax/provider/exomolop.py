from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import h5py
import numpy as np
import requests

from exojax.utils.molname import e2s


def load_ckd(path: Path):
    """Load a correlated-k opacity file and return metadata and the cross-section grid."""
    with h5py.File(path, "r") as fh5:
        molecule = fh5["mol_name"][()][0].decode("utf-8")
        mol_mass = float(fh5["mol_mass"][()][0])
        wavenumber = fh5["bin_centers"][:]  # cm-1
        samples = fh5["samples"][:]  # g-ordinates
        weights = fh5["weights"][:]
        temperatures = fh5["t"][:]  # K
        pressures = fh5["p"][:]  # bar
        kcoeff = np.array(fh5["kcoeff"])

    # reshape to (T, P, g, wavenumber)
    xsgrid = np.swapaxes(kcoeff, 0, 1)
    xsgrid = np.swapaxes(xsgrid, 2, 3)

    # Clip negative values
    tiny = np.finfo(xsgrid.dtype).tiny
    xsgrid = np.where(xsgrid == 0, tiny, xsgrid)

    return xsgrid, samples, weights, temperatures, pressures, wavenumber, molecule, mol_mass


def download_exomolop_h5(path, extension=".R1000_0.3-50mu.ktable.petitRADTRANS.h5"):
    """Download ExoMol opacity files in h5 format from ExoMol website.

    download_exomolop_h5(".database/CO/12C-16O/Li2015") should download
    https://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015.R1000_0.3-50mu.ktable.petitRADTRANS.h5

    one can know the recommended source using radis
    radis.api.exomolapi.get_exomol_database_list("CO", isotope_full_name="12C-16O")

    """
    from exojax.provider.url import url_ExoMol

    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    url_exomol_db = url_ExoMol()
    path = Path(path).expanduser()
    exact_molecule_name = path.parents[0].stem
    database = str(path.stem)
    simple_molecule_name = e2s(exact_molecule_name)
    base_url = (
        url_exomol_db + f"{simple_molecule_name}/{exact_molecule_name}/{database}/"
    )
    filename = f"{exact_molecule_name}__{database}{extension}"
    full_url = base_url + filename
    print(f"Downloading from {full_url}")

    dest = out_dir / filename
    if dest.exists():
        print(f"{dest} already exists. Skip downloading.")
        return dest

    # Obtain HTML for the directory listing and locate the target file link.
    session = requests.Session()
    headers = {"User-Agent": "exojax downloader"}
    response = session.get(base_url, timeout=120, headers=headers)

    if response.status_code == 403:
        # Some directories disallow listing; fall back to direct file URL.
        download_url = full_url
    else:
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        link_tag = soup.find("a", href=lambda href: href and href.endswith(filename))
        if link_tag is None:
            raise RuntimeError(f"Could not find {filename} at {base_url}")
        download_url = urljoin(base_url, link_tag["href"])

    # Stream download to avoid loading the entire file in memory.
    with session.get(
        download_url, stream=True, timeout=600, headers=headers
    ) as resp, dest.open(
        "wb"
    ) as fh:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)

    print(f"Saved to {dest}")
    return dest


if __name__ == "__main__":
    download_exomolop_h5(".database/CO/12C-16O/Li2015")

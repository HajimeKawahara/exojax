from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import h5py
import numpy as np
import requests

from exojax.utils.molname import e2s


def infer_molecule_name_from_path(path: Path):
    """Infer a simple molecule name from an ExoMolOP table path."""
    path = Path(path)
    try:
        return e2s(path.parent.parent.name)
    except Exception:
        return path.parent.parent.name


def _decode_scalar_text_dataset(dataset):
    """Decode a scalar or one-element byte/string HDF5 dataset."""
    value = np.asarray(dataset[()]).reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_ckd(path: Path):
    """Load a correlated-k opacity file and return metadata and the cross-section grid."""
    path = Path(path)
    with h5py.File(path, "r") as fh5:
        if "mol_name" in fh5:
            molecule = _decode_scalar_text_dataset(fh5["mol_name"])
        else:
            molecule = infer_molecule_name_from_path(path)
        mol_mass = float(fh5["mol_mass"][()][0])
        wavenumber = fh5["bin_centers"][:]  # cm-1
        samples = fh5["samples"][:]  # g-ordinates
        weights = fh5["weights"][:]
        temperatures = fh5["t"][:]  # K
        pressures = fh5["p"][:]  # bar
        kcoeff = np.asarray(fh5["kcoeff"], dtype=float)

    # reshape to (T, P, g, wavenumber)
    xsgrid = np.swapaxes(kcoeff, 0, 1)
    xsgrid = np.swapaxes(xsgrid, 2, 3)

    # Replace exact zeros so downstream log/positivity checks stay well-defined.
    tiny = np.finfo(xsgrid.dtype).tiny
    xsgrid = np.where(xsgrid == 0, tiny, xsgrid)

    return xsgrid, samples, weights, temperatures, pressures, wavenumber, molecule, mol_mass



def download_exomolop_h5(path, extension=None):
    """Download ExoMol opacity files in h5 format from ExoMol website.

    download_exomolop_h5(".database/CO/12C-16O/Li2015") should first try
    https://www.exomol.com/db/CO/12C-16O/Li2015/12C-16O__Li2015__R1000_0.3-50mu.ktable.petitRADTRANS.h5

    one can know the recommended source using radis
    radis.api.exomolapi.get_exomol_database_list("CO", isotope_full_name="12C-16O")

    """
    from exojax.provider.url import petitRADTRANS_ktable_filenames, url_ExoMol

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
    filenames = petitRADTRANS_ktable_filenames(
        exact_molecule_name, database, extension=extension
    )

    for filename in filenames:
        dest = out_dir / filename
        if dest.exists():
            if dest.stat().st_size == 0:
                print(f"{dest} is empty. Re-downloading.")
                dest.unlink()
            else:
                print(f"{dest} already exists. Skip downloading.")
                return dest

    # Obtain HTML for the directory listing and locate the target file link.
    session = requests.Session()
    headers = {"User-Agent": "exojax downloader"}
    response = session.get(base_url, timeout=120, headers=headers)

    if response.status_code == 403:
        # Some directories disallow listing; fall back to direct file URL.
        download_candidates = [
            (filename, base_url + filename) for filename in filenames
        ]
    else:
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        download_candidates = []
        for filename in filenames:
            link_tag = soup.find("a", href=lambda href: href and href.endswith(filename))
            if link_tag is not None:
                download_candidates.append((filename, urljoin(base_url, link_tag["href"])))
        if not download_candidates:
            raise RuntimeError(
                f"Could not find any of {', '.join(filenames)} at {base_url}"
            )

    # Stream to a temporary file first so HTTP errors cannot leave a final
    # zero-byte .h5 file that later looks like a valid local cache.
    errors = []
    for filename, download_url in download_candidates:
        dest = out_dir / filename
        tmp_dest = dest.with_name(dest.name + ".part")
        if tmp_dest.exists():
            tmp_dest.unlink()
        print(f"Downloading from {download_url}")
        try:
            with session.get(
                download_url, stream=True, timeout=600, headers=headers
            ) as resp:
                resp.raise_for_status()
                with tmp_dest.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)

            if tmp_dest.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty CKD table from {download_url}")
            tmp_dest.replace(dest)
        except Exception as exc:
            if tmp_dest.exists():
                tmp_dest.unlink()
            errors.append((download_url, exc))
            continue

        print(f"Saved to {dest}")
        return dest

    if errors:
        details = "; ".join(
            f"{url} -> {type(exc).__name__}: {exc}" for url, exc in errors
        )
        raise RuntimeError(
            "Failed to download any ExoMolOP CKD table candidate: " + details
        ) from errors[-1][1]
    raise RuntimeError(f"No ExoMolOP download candidates found at {base_url}")


if __name__ == "__main__":
    download_exomolop_h5(".database/CO/12C-16O/Li2015")

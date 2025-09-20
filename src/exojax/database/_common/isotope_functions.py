import numpy as np
import concurrent.futures as _cf
import logging
import requests
from bs4 import BeautifulSoup
from exojax.utils.url import url_lists_exomolhr
from typing import Optional
    

EXOMOLHR_HOME, EXOMOLHR_API_ROOT, EXOMOLHR_DOWNLOAD_ROOT = url_lists_exomolhr()


def _convert_proper_isotope(isotope):
    """covert isotope (int) to proper type for df

    Args:
        isotope (int or other type): isotope

    Returns:
        str: proper isotope type
    """
    if isotope == 0:
        return None
    elif isotope is not None and type(isotope) == int:
        return str(isotope)
    elif isotope is None:
        return isotope
    else:
        raise ValueError("Invalid isotope type")


def _isotope_index_from_isotope_number(isotope, uniqiso):
    """isotope index given HITRAN/HITEMP isotope number

    Args:
        isotope (int): isotope number
        uniqiso (nd int array): unique isotope array

    Returns:
        int: isotope_index for T_gQT and gQT
    """
    isotope_index = np.where(uniqiso == isotope)[0][0]
    return isotope_index


def _list_isotopologues(
    max_workers: Optional[int] = None

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


def _slug(molecule: str) -> str:
    """Turn H3+  -> H3_p   and  H3O+ -> H3O_p  (no change otherwise)."""
    return molecule.replace("+", "_p")


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

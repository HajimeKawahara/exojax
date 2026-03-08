"""RADIS dependency boundary for ExoJAX internals.

This module centralizes RADIS imports so exojax modules do not import RADIS
directly. The goal is dependency isolation only; behavior is unchanged.
"""
import warnings
from packaging import version


def get_radis_version():
    """Return ``radis.__version__``."""
    from radis import __version__ as radis_version

    return radis_version


def supports_exomol_broadf_download():
    """Return whether backend supports ExoMol ``broadf_download``."""
    return version.parse(get_radis_version()) >= version.parse("0.16")


def exomol_init_needs_bkgdatm():
    """Return whether ExoMol manager init expects ``bkgdatm`` argument."""
    return not supports_exomol_broadf_download()


def warn_if_exomol_broadf_download_unsupported():
    """Emit the legacy warning message for backends without broadf_download."""
    if supports_exomol_broadf_download():
        return
    radis_version = get_radis_version()
    print("radis==", radis_version)
    msg = "The current version of radis does not support broadf_download (requires >=0.16)."
    warnings.warn(msg, UserWarning)


def exomol_broadening_mode():
    """Return ExoMol broadening mode key used by ExoJAX call sites."""
    radis_version = get_radis_version()
    if version.parse(radis_version) <= version.parse("0.14"):
        return "compute_broadening"
    if version.parse(radis_version) <= version.parse("0.15.2"):
        return "set_broadening_coef_legacy"
    return "set_broadening_coef_species"


def get_auto_memory_mapping_engine():
    """Return RADIS auto-selected memory mapping engine."""
    from radis.api.dbmanager import get_auto_MEMORY_MAPPING_ENGINE

    return get_auto_MEMORY_MAPPING_ENGINE()


def get_exomol_mdb_class():
    """Return RADIS common-API ExoMol MDB class."""
    from radis.api.exomolapi import MdbExomol

    return MdbExomol


def get_hitran_database_manager_class():
    """Return RADIS HITRAN database manager class."""
    from radis.api.hitranapi import HITRANDatabaseManager

    return HITRANDatabaseManager


def get_hitemp_database_manager_class():
    """Return RADIS HITEMP database manager class."""
    from radis.api.hitempapi import HITEMPDatabaseManager

    return HITEMPDatabaseManager


def get_hit2df_func():
    """Return ``radis.api.hitranapi.hit2df`` for lazy call-sites."""
    from radis.api.hitranapi import hit2df

    return hit2df


def get_exomol_database_list_func():
    """Return ``radis.api.exomolapi.get_exomol_database_list`` lazily."""
    from radis.api.exomolapi import get_exomol_database_list

    return get_exomol_database_list


def get_molecule(molecule_identifier):
    """Return RADIS simple molecule name from identifier."""
    from radis.db.classes import get_molecule as _get_molecule

    return _get_molecule(molecule_identifier)


def get_molecule_identifier(simple_molecule_name):
    """Return RADIS HITRAN molecule identifier from simple name."""
    from radis.db.classes import get_molecule_identifier as _get_molecule_identifier

    return _get_molecule_identifier(simple_molecule_name)


def get_partition_function_value(molecule_identifier, isotope, temperature):
    """Return partition function value for molecule/isotope at temperature."""
    from radis.levels.partfunc import PartFuncTIPS

    partfunc = PartFuncTIPS(molecule_identifier, isotope)
    return partfunc.at(T=temperature)


def get_isotope_name(simple_molecule_name, isotope):
    """Return exact isotope name for molecule/isotope pair."""
    from radis.db.molparam import MolParams

    molparams = MolParams()
    return molparams.get(simple_molecule_name, isotope, "isotope_name")


def get_isotope_name_dict():
    """Return ``radis.db.molparam.isotope_name_dict`` lazily."""
    from radis.db.molparam import isotope_name_dict

    return isotope_name_dict

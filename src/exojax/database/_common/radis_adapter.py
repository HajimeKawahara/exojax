"""RADIS dependency boundary for ExoJAX internals.

This module centralizes RADIS imports so exojax modules do not import RADIS
directly. The goal is dependency isolation only; behavior is unchanged.
"""


def get_radis_version():
    """Return ``radis.__version__``."""
    from radis import __version__ as radis_version

    return radis_version


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

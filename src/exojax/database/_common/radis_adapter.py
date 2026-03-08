"""RADIS backend implementation layer for ExoJAX database APIs.

This module is the current backend boundary between ExoJAX database code and
RADIS-specific behavior. In addition to import indirection, this layer is the
preferred home for backend-specific logic such as:
- manager construction/init details
- backend capability checks
- backend compatibility shims and legacy-path handling

ExoJAX call sites should prefer these adapter helpers over embedding RADIS
constructor/version semantics directly.
"""
import warnings
from pathlib import Path
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


def init_exomol_manager(
    instance,
    *,
    path,
    local_databases,
    molecule,
    nurange,
    engine,
    crit,
    broadf,
    broadf_download,
    skip_optional_data,
    bkgdatm,
):
    """Initialize ExoMol backend manager on an existing instance.

    This keeps backend/version-specific constructor kwargs localized in the
    adapter while preserving inheritance-based architecture.
    """
    exomol_manager_class = get_exomol_mdb_class()
    if not exomol_init_needs_bkgdatm():
        exomol_manager_class.__init__(
            instance,
            path,
            local_databases=local_databases,
            molecule=molecule,
            name="EXOMOL-{molecule}",
            nurange=nurange,
            engine=engine,
            crit=crit,
            broadf=broadf,
            broadf_download=broadf_download,
            cache=True,
            skip_optional_data=skip_optional_data,
        )
    else:
        exomol_manager_class.__init__(
            instance,
            path,
            local_databases=local_databases,
            molecule=molecule,
            name="EXOMOL-{molecule}",
            nurange=nurange,
            engine=engine,
            crit=crit,
            bkgdatm=bkgdatm,  # uses radis <= 0.15.2
            broadf=broadf,
            cache=True,
            skip_optional_data=skip_optional_data,
        )


def init_hitran_manager(
    instance,
    *,
    molecule,
    local_databases,
    engine,
    nonair_broadening,
):
    """Initialize HITRAN backend manager on an existing instance."""
    hitran_manager_class = get_hitran_database_manager_class()
    extra_params = "all" if nonair_broadening else None
    hitran_manager_class.__init__(
        instance,
        molecule=molecule,
        name="HITRAN-{molecule}",
        local_databases=local_databases,
        engine=engine,
        verbose=True,
        parallel=True,
        extra_params=extra_params,
    )


def init_hitemp_manager(
    instance,
    *,
    molecule,
    local_databases,
    engine,
):
    """Initialize HITEMP backend manager on an existing instance.

    Handles RADIS databank-name collisions for pre-registered environments.
    """
    hitemp_manager_class = get_hitemp_database_manager_class()
    db_name = f"HITEMP-{molecule}"
    try:
        hitemp_manager_class.__init__(
            instance,
            molecule=molecule,
            name=db_name,
            local_databases=local_databases,
            engine=engine,
            verbose=True,
            chunksize=100000,
            parallel=True,
        )
    except ValueError as exc:
        if "already registered in radis.json" not in str(exc):
            raise
        local_tag = Path(local_databases).expanduser().resolve().name or "local"
        db_name = f"HITEMP-{molecule}-{local_tag}"
        hitemp_manager_class.__init__(
            instance,
            molecule=molecule,
            name=db_name,
            local_databases=local_databases,
            engine=engine,
            verbose=True,
            chunksize=100000,
            parallel=True,
        )


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

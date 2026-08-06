"""Tests for the deprecated :mod:`exojax.spec` compatibility namespace."""

import importlib
from importlib.util import find_spec
from types import ModuleType


LEGACY_MODULE_NAMES = frozenset(
    {
        "api",
        "atmrt",
        "atomll",
        "atomllapi",
        "chord",
        "contdb",
        "customapi",
        "dbmanager",
        "dit",
        "ditkernel",
        "exomol",
        "exomolhr",
        "generate_elower_grid_trange",
        "hitran",
        "hitranapi",
        "hitrancia",
        "hminus",
        "initspec",
        "layeropacity",
        "lbd",
        "lbderror",
        "limb_darkening",
        "lpf",
        "lpffilter",
        "lsd",
        "make_numatrix",
        "mie",
        "modit",
        "moldb",
        "molinfo",
        "multimol",
        "nonair",
        "opacalc",
        "opacont",
        "opart",
        "optgrid",
        "pardb",
        "planck",
        "premodit",
        "profconv",
        "qstate",
        "rayleigh",
        "response",
        "rtlayer",
        "rtransfer",
        "set_ditgrid",
        "specop",
        "spin_rotation",
        "toon",
        "twostream",
    }
)
LEGACY_OBJECT_NAMES = frozenset({"OpaDirect", "OpaModit", "OpaPremodit"})
LEGACY_NAMES = LEGACY_MODULE_NAMES | LEGACY_OBJECT_NAMES


def test_all_advertised_compatibility_names_resolve():
    spec = importlib.import_module("exojax.spec")
    assert len(LEGACY_MODULE_NAMES) == 50
    assert len(LEGACY_NAMES) == 53
    assert LEGACY_MODULE_NAMES <= set(spec._MODULE_ALIASES)
    assert LEGACY_NAMES <= set(spec.__all__)

    unresolved = []
    for name in spec.__all__:
        try:
            getattr(spec, name)
        except (AttributeError, ImportError) as exc:
            unresolved.append((name, str(exc)))

    assert unresolved == []


def test_all_old_submodule_imports():
    spec = importlib.import_module("exojax.spec")
    failures = []
    module_names = LEGACY_MODULE_NAMES | set(spec._MODULE_ALIASES)
    for old_name in module_names:
        try:
            imported = importlib.import_module(f"exojax.spec.{old_name}")
        except ImportError as exc:
            failures.append((old_name, str(exc)))
            continue

        if not isinstance(imported, ModuleType):
            failures.append((old_name, f"resolved to {type(imported)!r}"))
        elif imported is not getattr(spec, old_name):
            failures.append((old_name, "attribute and import resolved differently"))

    assert failures == []

    lpf = importlib.import_module("exojax.spec.lpf")
    assert find_spec("exojax.spec.lpf") is lpf.__spec__


def test_all_compatibility_targets_import():
    spec = importlib.import_module("exojax.spec")
    failures = []
    for old_name, targets in spec._MODULE_ALIASES.items():
        for target in targets:
            try:
                importlib.import_module(target)
            except Exception as exc:  # report every broken compatibility target
                failures.append((old_name, target, f"{type(exc).__name__}: {exc}"))

    assert failures == []


def test_star_import_forwards_public_and_renamed_symbols():
    namespace = {}
    exec("from exojax.spec.atomll import *", namespace)

    from exojax.database.core_atom.broadening import gamma_vald3
    from exojax.database.core_atom.line_strength import line_strength_atom

    assert namespace["gamma_vald3"] is gamma_vald3
    assert namespace["Sij0"] is line_strength_atom


def test_split_database_modules_forward_to_current_implementations():
    from exojax.database._common.commonapi import MdbCommonHitempHitran
    from exojax.database._common.isotope_functions import _list_isotopologues
    from exojax.database.cia.io import read_cia
    from exojax.database.core.abscoeff import interp_logacia_matrix
    from exojax.database.core.broadening import gamma_exomol
    from exojax.database.core.line_strength import line_strength
    from exojax.database.core_atom.broadening import gamma_vald3
    from exojax.database.core_atom.io import read_ExAll
    from exojax.database.core_atom.line_strength import line_strength_atom
    from exojax.database.core_atom.pf import partfn_Fe
    from exojax.database.exomolhr.api import list_exomolhr_molecules
    from exojax.database.hargreaves.api import MdbHargreaves
    from exojax.database.hargreaves.api import _set_wavenum_hargreaves
    from exojax.database.molinfo.qstate import m_transition_state
    from exojax.database.vald.api import AdbVald
    from exojax.opacity.modit.dit import xsvector
    from exojax.rt.emis import OpartEmisPure
    from exojax.provider.exomolhr import _fetch_opacity_zip, _load_exomolhr_csv
    from exojax.spec import api, atomll, atomllapi, customapi, dbmanager, dit
    from exojax.spec import exomol, exomolhr
    from exojax.spec import hitran, hitrancia, moldb, opart, qstate

    assert api.MdbCommonHitempHitran is MdbCommonHitempHitran
    assert atomll.gamma_vald3 is gamma_vald3
    assert atomll.Sij0 is line_strength_atom
    assert atomllapi.read_ExAll is read_ExAll
    assert atomllapi.partfn_Fe is partfn_Fe
    assert customapi.set_wavenum is _set_wavenum_hargreaves
    assert dbmanager.MdbHargreaves is MdbHargreaves
    assert dit.xsvector is xsvector
    assert exomol.gamma_exomol is gamma_exomol
    assert exomolhr.fetch_opacity_zip is _fetch_opacity_zip
    assert exomolhr.load_exomolhr_csv is _load_exomolhr_csv
    assert exomolhr.list_exomolhr_molecules is list_exomolhr_molecules
    assert exomolhr.list_isotopologues is _list_isotopologues
    assert hitran.line_strength is line_strength
    assert hitrancia.read_cia is read_cia
    assert hitrancia.interp_logacia_matrix is interp_logacia_matrix
    assert moldb.AdbVald is AdbVald
    assert opart.OpartEmisPure is OpartEmisPure
    assert qstate.m_transition_state is m_transition_state


def test_root_and_renamed_aliases_are_current_implementations():
    from exojax.database.core.broadening import doppler_sigma, gamma_natural
    from exojax.database.core.broadening import normalized_doppler_sigma
    from exojax.database.core.line_strength import line_strength
    from exojax.opacity.lpf.lpf import hjert, voigt, voigtone, vvoigt
    from exojax.opacity.lpf.make_numatrix import make_numatrix0
    from exojax.postproc.specop import SopCommonConv
    from exojax.spec import doppler_sigma as legacy_doppler_sigma
    from exojax.spec import gamma_natural as legacy_gamma_natural
    from exojax.spec import hjert as legacy_hjert
    from exojax.spec import line_strength as legacy_line_strength
    from exojax.spec import make_numatrix0 as legacy_make_numatrix0
    from exojax.spec import normalized_doppler_sigma as legacy_normalized_doppler_sigma
    from exojax.spec import specop
    from exojax.spec import voigt as legacy_voigt
    from exojax.spec import voigtone as legacy_voigtone
    from exojax.spec import vvoigt as legacy_vvoigt

    assert legacy_line_strength is line_strength
    assert legacy_doppler_sigma is doppler_sigma
    assert legacy_gamma_natural is gamma_natural
    assert legacy_normalized_doppler_sigma is normalized_doppler_sigma
    assert legacy_hjert is hjert
    assert legacy_voigt is voigt
    assert legacy_voigtone is voigtone
    assert legacy_vvoigt is vvoigt
    assert legacy_make_numatrix0 is make_numatrix0
    assert specop.SopCommon is SopCommonConv


def test_class_aliases_are_the_current_classes():
    from exojax.opacity import OpaDirect, OpaModit, OpaPremodit
    from exojax.opacity.base import OpaCalc
    from exojax.spec import opacalc
    from exojax.spec import OpaDirect as LegacyOpaDirect
    from exojax.spec import OpaModit as LegacyOpaModit
    from exojax.spec import OpaPremodit as LegacyOpaPremodit

    assert LegacyOpaDirect is OpaDirect
    assert LegacyOpaModit is OpaModit
    assert LegacyOpaPremodit is OpaPremodit
    assert opacalc.OpaCalc is OpaCalc


def test_rt_opart_compatibility_shim_exports_all_classes():
    from exojax.rt import OpartEmisPure, OpartEmisScat
    from exojax.rt import OpartReflectEmis, OpartReflectPure
    from exojax.rt.opart import OpartEmisPure as ShimOpartEmisPure
    from exojax.rt.opart import OpartEmisScat as ShimOpartEmisScat
    from exojax.rt.opart import OpartReflectEmis as ShimOpartReflectEmis
    from exojax.rt.opart import OpartReflectPure as ShimOpartReflectPure

    assert ShimOpartEmisPure is OpartEmisPure
    assert ShimOpartEmisScat is OpartEmisScat
    assert ShimOpartReflectEmis is OpartReflectEmis
    assert ShimOpartReflectPure is OpartReflectPure

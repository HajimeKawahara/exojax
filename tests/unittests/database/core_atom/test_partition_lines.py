"""Atomic partition functions must match line strengths and selected species."""

import importlib

import jax
import numpy as np
import pytest

from exojax.database.core.line_strength import line_strength
from exojax.database.core_atom.pf import partfn_Fe
from exojax.utils.constants import ccgs, hcperk


@pytest.fixture(params=["vald", "kurucz"])
def make_adb(request, monkeypatch, tmp_path):
    api = importlib.import_module(f"exojax.database.{request.param}.api")
    transitions = (
        [1.0e6, 2.0e6, 3.0e6], [1000.0, 1001.0, 1002.0],
        [0.0, 10.0, 20.0], [1000.0, 1011.0, 1022.0],
        [3.0, 4.0, 4.0], [0.0, 0.5, 0.5], [1.0, 1.5, 1.5],
        [26, 26, 11], [1, 2, 1],
        [8.0, 8.1, 8.2], [-6.0, -6.1, -6.2], [-7.0, -7.1, -7.2],
    )

    def read_transitions(_):
        return tuple(np.array(values) for values in transitions)

    if request.param == "vald":
        monkeypatch.setattr(api, "_load_vald_dataframe", lambda *args: None)
        monkeypatch.setattr(api, "pickup_param", read_transitions)
        adb_class = api.AdbVald
    else:
        monkeypatch.setattr(api, "read_kurucz", read_transitions)
        adb_class = api.AdbKurucz
    return lambda **kwargs: adb_class(tmp_path / "synthetic.lines", **kwargs)


@pytest.mark.parametrize("irwin", [False, True])
def test_atomic_partition_ratio_reference_and_masking(make_adb, irwin):
    adb = make_adb(Irwin=irwin)
    for reference in (adb.Tref, 1100.0):
        np.testing.assert_array_equal(adb.qr_interp_lines(reference, reference), np.ones(3))
    ratios = jax.jit(adb.qr_interp_lines)(1800.0, adb.Tref)
    assert ratios.shape == (3,)
    assert np.all(np.isfinite(ratios))
    np.testing.assert_array_equal(adb.line_masses, [55.847, 55.847, 22.98981])

    mask = np.array([False, True, True])
    adb.masking(mask)
    np.testing.assert_array_equal(adb.line_masses, [55.847, 22.98981])
    np.testing.assert_allclose(adb.qr_interp_lines(1800.0, adb.Tref), ratios[mask])


def test_atomic_irwin_line_strength_uses_consistent_partition_function(make_adb):
    barklem = make_adb()
    irwin = make_adb(Irwin=True)
    np.testing.assert_array_equal(irwin.Sij0[1:], barklem.Sij0[1:])

    for temperature in (irwin.Tref, 1800.0):
        partition = np.array([
            np.interp(temperature, barklem.T_gQT, barklem.gQT_284species[index])
            for index in np.asarray(barklem.QTmask)
        ])
        partition[0] = float(partfn_Fe(temperature))
        expected = (
            -np.asarray(irwin.A) * np.asarray(irwin.gupper)
            * np.exp(-hcperk * np.asarray(irwin.elower) / temperature)
            * np.expm1(-hcperk * irwin.nu_lines / temperature)
            / (8.0 * np.pi * ccgs * irwin.nu_lines**2 * partition)
        )
        actual = line_strength(
            temperature, irwin.logsij0, irwin.nu_lines, irwin.elower,
            irwin.qr_interp_lines(temperature, irwin.Tref), irwin.Tref,
        )
        baseline = line_strength(
            temperature, barklem.logsij0, barklem.nu_lines, barklem.elower,
            barklem.qr_interp_lines(temperature, barklem.Tref), barklem.Tref,
        )
        np.testing.assert_allclose(actual, expected, rtol=1.0e-11, atol=0.0)
        np.testing.assert_array_equal(actual[1:], baseline[1:])
    assert not np.isclose(actual[0] / baseline[0], 1.0, rtol=1.0e-3)

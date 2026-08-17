from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from exojax.database.vald.api import AdbSepVald
from exojax.opacity.modit.modit import vald_all


def test_vald_all_uses_shared_partition_function_reference():
    number_of_species = 2
    number_of_lines = 1
    adb = SimpleNamespace(
        nu_lines=jnp.array([1000.0, 1001.0]),
        gQT_284species=jnp.ones((284, 2)),
        T_gQT=jnp.array([100.0, 2000.0]),
        QTmask=jnp.array([0, 1]),
        QTref_284=jnp.ones(284),
        ielem=jnp.array([26, 1]),
        iion=jnp.ones(number_of_species, dtype=int),
        atomicmass=jnp.array([56.0, 1.0]),
        ionE=jnp.array([7.9, 13.6]),
        dev_nu_lines=jnp.array([1000.0, 1001.0]),
        logsij0=jnp.log(jnp.full(number_of_species, 1.0e-20)),
        elower=jnp.zeros(number_of_species),
        eupper=jnp.ones(number_of_species),
        gamRad=jnp.zeros(number_of_species),
        gamSta=jnp.zeros(number_of_species),
        vdWdamp=jnp.full(number_of_species, -7.0),
        Tref=296.0,
    )
    asdb = AdbSepVald(adb)
    temperatures = jnp.array([1000.0, 1200.0])
    partial_pressure = jnp.full_like(temperatures, 1.0e-3)

    results = vald_all(
        asdb,
        temperatures,
        partial_pressure,
        partial_pressure,
        partial_pressure,
        1.0e5,
    )

    for result in results:
        assert result.shape == (number_of_species, len(temperatures), number_of_lines)
        assert np.all(np.isfinite(result))

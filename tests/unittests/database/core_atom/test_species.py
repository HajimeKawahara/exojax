import jax.numpy as jnp
import numpy as np

from exojax.database.core_atom.line_strength import line_strength_atom
from exojax.database.core_atom.misc import get_VMR_uspecies
from exojax.database.core_atom.misc import ielemion_to_FastChemSymbol
from exojax.database.core_atom.pf import partfn_Fe
from exojax.utils.constants import Tref_original


def test_atomic_species_helpers():
    assert ielemion_to_FastChemSymbol(26, 1) == "Fe"

    vmr = get_VMR_uspecies(jnp.array([[26, 1], [26, 2]]))
    np.testing.assert_allclose(vmr[1] / vmr[0], 1.0e-10)


def test_line_strength_atom_irwin_partition_function():
    inputs = (
        np.ones(1),
        np.ones(1),
        np.full(1, 1000.0),
        np.zeros(1),
        np.ones(77),
        np.array([76]),
    )

    barklem = line_strength_atom(*inputs)
    irwin = line_strength_atom(*inputs, Irwin=True)

    np.testing.assert_allclose(irwin, barklem / partfn_Fe(Tref_original))

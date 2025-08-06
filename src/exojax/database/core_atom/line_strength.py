import numpy as np
import jax.numpy as jnp
from exojax.database import core_atom.io
from exojax.utils.constants import Tref_original
from exojax.utils.constants import ccgs
from exojax.utils.constants import hcperk


def line_strength_atom(A, gupper, nu_lines, elower, QTref_284, QTmask, Irwin=False):
    """Reference Line Strength in Tref=296K, S0.

    Args:
        A: Einstein coefficient (s-1)
        gupper: the upper state statistical weight
        nu_lines: line center wavenumber (cm-1)
        elower: elower
        QTref_284: partition function Q(Tref)
        QTmask: mask to identify a rows of QTref_284 to apply for each line
        Irwin: if True(1), the partition functions of Irwin1981 is used, otherwise those of Barklem&Collet2016

    Returns:
        Sij(T): Line strength (cm)
    """

    # Assign Q(Tref) for each line
    QTref = np.zeros_like(QTmask, dtype=float)
    for i, mask in enumerate(QTmask):
        QTref[i] = QTref_284[mask]

    # Use Irwin_1981 for Fe I (mask==76)  #test211013Tako
    if Irwin == True:
        QTref[jnp.where(QTmask == 76)[0]] = core_atom.io.partfn_Fe(Tref_original)

    S0 = (
        -A
        * gupper
        * np.exp(-hcperk * elower / Tref_original)
        * np.expm1(-hcperk * nu_lines / Tref_original)
        / (8.0 * np.pi * ccgs * nu_lines**2 * QTref)
    )

    return S0

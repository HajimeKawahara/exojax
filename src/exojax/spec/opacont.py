"""opacity continuum calculator class

Notes:
    Opa does not assume any T-P structure, no fixed T, P, mmr grids.

"""
from exojax.utils.grids import nu2wav
from exojax.spec.hitrancia import logacia
import jax.numpy as jnp

__all__ = ['OpaCIA']


class OpaCont():
    """Common Opacity Calculator Class
    """
    __slots__ = [
        "opainfo",
    ]

    def __init__(self):
        self.method = None  # which opacity cont method is used
        self.ready = False  # ready for opacity computation


class OpaCIA(OpaCint):
    """Opacity Continuum Calculator Class for CIA

    """
    def __init__(self, cdb, nu_grid):
        self.method = "cia"
        self.warning = True
        self.nu_grid = nu_grid
        self.wav = nu2wav(self.nu_grid, unit="AA")
        self.cdb = cdb

    def logcc_vector(self, T):
        temperature = jnp.array([T])
        return logacia(temperature, self.nu_grid, self.cdb.nucia, self.cdb.tcia,
                       self.cdb.logac)[0]

    def logcc_matrix(self, temperature):
        return logacia(temperature, self.nu_grid, self.cdb.nucia, self.cdb.tcia,
                       self.cdb.logac)

"""Continuum database (CDB) class.

* CdbCIA is the CDB for CIA
"""

from pathlib import Path
import jax.numpy as jnp
import numpy as np
from exojax.database.cia.io import read_cia
from exojax.provider.hitrancia import fetch_hitran_cia

__all__ = ["CdbCIA"]


class CdbCIA:
    def __init__(self, path, nurange=[-np.inf, np.inf], margin=10.0):
        """Continuum database for hitrancia.

        Args:
            path (str): path for hitrancia file, e.g. ".database/H2-H2_2011.cia"
            nurange: wavenumber range list (cm-1) or wavenumber array
            margin: margin for nurange (cm-1)

        Example::
            ciaH2H2 = CdbCIA(".database/H2-H2_2011.cia", nurange=[4050.0, 4150.0])

        """
        self.nurange = [np.min(nurange), np.max(nurange)]
        self.margin = margin

        if not Path(path).exists():
            fetch_hitran_cia(Path(path))

        self.nucia, self.tcia, ac = read_cia(
            path, self.nurange[0] - self.margin, self.nurange[1] + self.margin
        )
        self.logac = jnp.array(np.log10(ac))
        self.tcia = jnp.array(self.tcia)
        self.nucia = jnp.array(self.nucia)


if __name__ == "__main__":
    ciaH2H2 = CdbCIA(".database/H2-H2_2011.cia", nurange=[4050.0, 4150.0])
    print(ciaH2H2.tcia)

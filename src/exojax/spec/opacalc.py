"""opacity calculator class

"""

__all__ = ['OpaPremodit']

from exojax.spec import initspec
from exojax.utils.grids import wavenumber_grid
from exojax.utils.instfunc import nx_from_resolution_eslog
from exojax.utils.grids import nu2wav
from exojax.utils.instfunc import resolution_eslog
import numpy as np


class OpaCalc():
    """Common Opacity Calculator Class
    """
    __slots__ = [
        "opainfo",
    ]

    def __init__(self):
        self.opainfo = None


class OpaPremodit(OpaCalc):
    """Opacity Calculator Class for PreMODIT

    Attributes:
        opainfo: information set used in PreMODIT

    """
    def __init__(self, mdb=None, nu_grid=None, diffmode=2):
        super().__init__()

        #default setting
        self.dit_grid_resolution = 0.1
        self.diffmode = diffmode
        self.warning = True
        #need to refine
        #self.set_dET(Tlow, Thigh, precision)
        #self.Twt = 610.0
        self.Twt = 1000.0
        self.Tref = 500.0
        self.dE = 1500.0

        # initialize mdb and nu_grid
        if mdb is not None and nu_grid is not None:
            self.nu_grid = nu_grid
            self.wav = nu2wav(self.nu_grid, unit="AA")
            self.resolution = resolution_eslog(nu_grid)
            self.setmdb(mdb)

    def set_nu_grid(self, x0, x1, unit, resolution=700000, Nx=None):
        if Nx is None:
            Nx = nx_from_resolution_eslog(x0, x1, resolution)
        if np.mod(Nx, 2) == 1:
            Nx = Nx + 1
        self.nu_grid, self.wav, self.resolution = wavenumber_grid(
            x0, x1, Nx, unit=unit, xsmode="premodit")

    def set_gamma_and_n_Texp(self, mdb):
        if mdb.dbtype == "hitran":
            print("gamma_air and temperature exponent are used.")
            self.gamma_ref = mdb.gamma_air
            self.n_Texp = mdb.n_air
        elif mdb.dbtype == "exomol":
            self.gamma_ref = mdb.alpha_ref
            self.n_Texp = mdb.n_Texp

    def setmdb(self, mdb):
        print("Set mdb. opainfo is now available.")
        mdb.change_reference_temperature(self.Tref)
        self.dbtype = mdb.dbtype
        self.set_gamma_and_n_Texp(mdb)
        self.opainfo = initspec.init_premodit(
            mdb.nu_lines,
            self.nu_grid,
            mdb.elower,
            self.gamma_ref,
            self.n_Texp,
            mdb.line_strength_ref,
            self.Twt,
            Tref=self.Tref,
            dE=self.dE,
            dit_grid_resolution=self.dit_grid_resolution,
            diffmode=self.diffmode,
            warning=self.warning)

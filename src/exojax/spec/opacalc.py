"""opacity calculator class

"""

__all__ = ['OpaPremodit']

from exojax.spec import initspec
from exojax.utils.grids import wavenumber_grid
from exojax.utils.instfunc import nx_from_resolution_eslog
from exojax.utils.grids import nu2wav
from exojax.utils.instfunc import resolution_eslog
import numpy as np
import warnings


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
    def __init__(self,
                 mdb,
                 nu_grid,
                 diffmode=2,
                 auto_params=None,
                 manual_params=None):
        """initialization of OpaPremodit

        Note:
            If auto_params nor manual_params is not given in arguments, 
            self.manual_setting or self.auto_setting is required. 

        Args:
            mdb (mdb class): mdbExomol, mdbHitemp, mdbHitran
            nu_grid (): wavenumber grid (cm-1)
            diffmode (int, optional): _description_. Defaults to 2.
            auto_params (dictionary, optional): _description_. Defaults to None.
            manual_params (dictionary, optional): _description_. Defaults to None.
        """
        super().__init__()

        #default setting
        self.diffmode = diffmode
        self.warning = True
        self.nu_grid = nu_grid
        self.wav = nu2wav(self.nu_grid, unit="AA")
        self.resolution = resolution_eslog(nu_grid)
        self.mdb = mdb

        if auto_params is not None:
            self.auto_setting()
        elif manual_params is not None:
            self.manual_setting(manual_params["Twt"], manual_params["Tref"],
                                manual_params["dE"])
        else:
            print("OpaPremodit: init w/o params setting")

    def manual_setting(self, Twt, Tref, dE):
        """setting PreMODIT parameters by manual

        Args:
            Twt (float): Temperature for weight (K)
            Tref (float): reference temperature (K)
            dE (float): E lower grid interval (cm-1)
        """
        print("OpaPremodit: params manually set.")
        self.Twt = Twt
        self.Tref = Tref
        self.dE = dE
        self.apply_params()

    def auto_setting(self, Tmin, Tmax, precision):
        print("OpaPremodit: params automatically set.")
        assert False

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

    def apply_params(self):
        self.mdb.change_reference_temperature(self.Tref)
        self.dbtype = self.mdb.dbtype
        self.set_gamma_and_n_Texp(self.mdb)
        self.opainfo = initspec.init_premodit(
            self.mdb.nu_lines,
            self.nu_grid,
            self.mdb.elower,
            self.gamma_ref,
            self.n_Texp,
            self.mdb.line_strength_ref,
            self.Twt,
            Tref=self.Tref,
            dE=self.dE,
            #dit_grid_resolution=self.dit_grid_resolution,
            diffmode=self.diffmode,
            warning=self.warning)

"""opacity calculator class

"""

__all__ = ['OpaPremodit']

from exojax.spec import initspec
from exojax.spec.lbderror import optimal_params
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
        self.opaclass = None  # which opacity lass is used
        self.ready = False  # ready for opacity computation


class OpaPremodit(OpaCalc):
    """Opacity Calculator Class for PreMODIT

    Attributes:
        opainfo: information set used in PreMODIT

    """
    def __init__(self,
                 mdb,
                 nu_grid,
                 diffmode=2,
                 auto_trange=None,
                 manual_params=None,
                 dit_grid_resolution=0.2):
        """initialization of OpaPremodit

        Note:
            If auto_trange nor manual_params is not given in arguments, 
            use manual_setting()
            or provide self.dE, self.Tref, self.Twt and apply self.apply_params()
            

        Args:
            mdb (mdb class): mdbExomol, mdbHitemp, mdbHitran
            nu_grid (): wavenumber grid (cm-1)
            diffmode (int, optional): _description_. Defaults to 2.
            auto_trange (optional): temperature range [Tl, Tu], in which line strength is within 1 % prescision. Defaults to None.
            manual_params (optional): premodit param set [dE, Tref, Twt]. Defaults to None.
        """
        super().__init__()

        #default setting
        self.opaclass = "premodit"
        self.diffmode = diffmode
        self.warning = True
        self.nu_grid = nu_grid
        self.wav = nu2wav(self.nu_grid, unit="AA")
        self.resolution = resolution_eslog(nu_grid)
        self.mdb = mdb
        self.dit_grid_resolution = dit_grid_resolution
        if auto_trange is not None:
            self.auto_setting(auto_trange[0], auto_trange[1])
        elif manual_params is not None:
            self.manual_setting(manual_params[0], manual_params[1], manual_params[2])
        else:
            print("OpaPremodit: init w/o params setting")
            print("Call self.apply_params() to complete the setting.")
            

    def auto_setting(self, Tl, Tu):
        print("OpaPremodit: params automatically set.")
        self.dE, self.Tref, self.Twt = optimal_params(Tl, Tu, self.diffmode)
        self.Tmax = Tu
        self.apply_params()

    def manual_setting(self, dE, Tref, Twt):
        """setting PreMODIT parameters by manual

        Args:
            dE (float): E lower grid interval (cm-1)
            Tref (float): reference temperature (K)
            Twt (float): Temperature for weight (K)
        """
        print("OpaPremodit: params manually set.")
        self.Twt = Twt
        self.Tref = Tref
        self.dE = dE
        self.Tmax = np.max([Twt,Tref])
        self.apply_params()

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
            Tmax=self.Tmax,
            dE=self.dE,
            dit_grid_resolution=self.dit_grid_resolution,
            diffmode=self.diffmode,
            warning=self.warning)
        self.ready = True

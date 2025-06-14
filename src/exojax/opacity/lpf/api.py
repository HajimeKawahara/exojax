import numpy as np
from jax import jit, vmap

from exojax.opacity.base import OpaCalc
from exojax.opacity import initspec
from exojax.utils.constants import Tref_original
from exojax.utils.grids import nu2wav


class OpaDirect(OpaCalc):
    def __init__(self, mdb, nu_grid, wavelength_order="descending"):
        """initialization of OpaDirect (LPF)



        Args:
            mdb (mdb class): mdbExomol, mdbHitemp, mdbHitran
            nu_grid (): wavenumber grid (cm-1)
        """
        super().__init__(nu_grid)

        # default setting
        self.method = "lpf"
        self.warning = True
        self.wavelength_order = wavelength_order
        self.wav = nu2wav(
            self.nu_grid, wavelength_order=self.wavelength_order, unit="AA"
        )
        self.mdb = mdb
        self.apply_params()

    def __eq__(self, other):
        """eq method for OpaDirect, definied by comparing all the attributes and important status

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not isinstance(other, OpaDirect):
            return False

        eq_attributes = (
            (self.mdb == other.mdb)
            and (self.wavelength_order == other.wavelength_order)
            and all(self.nu_grid == other.nu_grid)
        )

        return eq_attributes

    def __ne__(self, other):
        return not self.__eq__(other)

    def apply_params(self):
        self.dbtype = self.mdb.dbtype
        self.opainfo = initspec.init_lpf(self.mdb.nu_lines, self.nu_grid)
        self.ready = True

    def xsvector(self, T, P, Pself=0.0):
        """cross section vector

        Args:
            T (float): temperature
            P (float): pressure in bar
            Pself (float, optional): self pressure for HITEMP/HITRAN. Defaults to 0.0.

        Returns:
            1D array: cross section in cm2
        """
        from exojax.database.hitran import doppler_sigma, gamma_natural
        from exojax.database.exomol import gamma_exomol
        from exojax.database.hitran import gamma_hitran, line_strength
        from exojax.opacity.lpf.lpf import xsvector as xsvector_lpf

        numatrix = self.opainfo

        if self.mdb.dbtype == "hitran":
            qt = self.mdb.qr_interp(self.mdb.isotope, T, Tref_original)
            gammaL = gamma_hitran(
                P, T, Pself, self.mdb.n_air, self.mdb.gamma_air, self.mdb.gamma_self
            ) + gamma_natural(self.mdb.A)
        elif self.mdb.dbtype == "exomol":
            qt = self.mdb.qr_interp(T, Tref_original)
            gammaL = gamma_exomol(
                P, T, self.mdb.n_Texp, self.mdb.alpha_ref
            ) + gamma_natural(self.mdb.A)
        sigmaD = doppler_sigma(self.mdb.nu_lines, T, self.mdb.molmass)
        Sij = line_strength(
            T, self.mdb.logsij0, self.mdb.nu_lines, self.mdb.elower, qt, Tref_original
        )
        return xsvector_lpf(numatrix, sigmaD, gammaL, Sij)

    def xsmatrix(self, Tarr, Parr):
        """cross section matrix

        Notes:
            Currently Pself is regarded to be zero for HITEMP/HITRAN

        Args:
            Tarr (): tempearture array in K
            Parr (): pressure array in bar

        Raises:
            ValueError: _description_

        Returns:
            jnp.array : cross section matrix (Nlayer, N_wavenumber)
        """
        from exojax.database.hitran import doppler_sigma, gamma_natural
        from exojax.database.atomll import gamma_vald3, interp_QT_284
        from exojax.database.exomol import gamma_exomol
        from exojax.database.hitran import gamma_hitran, line_strength
        from exojax.opacity.lpf.lpf import xsmatrix as xsmatrix_lpf

        numatrix = self.opainfo
        vmaplinestrengh = jit(vmap(line_strength, (0, None, None, None, 0, None)))
        if self.mdb.dbtype == "hitran":
            vmapqt = vmap(self.mdb.qr_interp, (None, 0, None))
            qt = vmapqt(self.mdb.isotope, Tarr, Tref_original)
            vmaphitran = jit(vmap(gamma_hitran, (0, 0, 0, None, None, None)))
            gammaLM = vmaphitran(
                Parr,
                Tarr,
                np.zeros_like(Parr),
                self.mdb.n_air,
                self.mdb.gamma_air,
                self.mdb.gamma_self,
            ) + gamma_natural(self.mdb.A)
            SijM = vmaplinestrengh(
                Tarr,
                self.mdb.logsij0,
                self.mdb.nu_lines,
                self.mdb.elower,
                qt,
                Tref_original,
            )
            sigmaDM = jit(vmap(doppler_sigma, (None, 0, None)))(
                self.mdb.nu_lines, Tarr, self.mdb.molmass
            )
        elif self.mdb.dbtype == "exomol":
            vmapqt = vmap(self.mdb.qr_interp, (0, None))
            qt = vmapqt(Tarr, Tref_original)
            vmapexomol = jit(vmap(gamma_exomol, (0, 0, None, None)))
            gammaLMP = vmapexomol(Parr, Tarr, self.mdb.n_Texp, self.mdb.alpha_ref)
            gammaLMN = gamma_natural(self.mdb.A)
            gammaLM = gammaLMP + gammaLMN[None, :]
            SijM = vmaplinestrengh(
                Tarr,
                self.mdb.logsij0,
                self.mdb.nu_lines,
                self.mdb.elower,
                qt,
                Tref_original,
            )
            sigmaDM = jit(vmap(doppler_sigma, (None, 0, None)))(
                self.mdb.nu_lines, Tarr, self.mdb.molmass
            )
        elif (self.mdb.dbtype == "kurucz") or (self.mdb.dbtype == "vald"):
            qt_284 = vmap(interp_QT_284, (0, None, None))(
                Tarr, self.mdb.T_gQT, self.mdb.gQT_284species
            )
            qt_K = qt_284[:, self.mdb.QTmask]  # e.g., qt_284[:,76] #Fe I
            qr_K = qt_K / self.mdb.QTref_284[self.mdb.QTmask]
            vmapvald3 = jit(
                vmap(
                    gamma_vald3,
                    (
                        0,
                        0,
                        0,
                        0,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ),
                )
            )
            PH, PHe, PHH = (
                Parr * self.mdb.vmrH,
                Parr * self.mdb.vmrHe,
                Parr * self.mdb.vmrHH,
            )
            gammaLM = vmapvald3(
                Tarr,
                PH,
                PHH,
                PHe,
                self.mdb.ielem,
                self.mdb.iion,
                self.mdb.dev_nu_lines,
                self.mdb.elower,
                self.mdb.eupper,
                self.mdb.atomicmass,
                self.mdb.ionE,
                self.mdb.gamRad,
                self.mdb.gamSta,
                self.mdb.vdWdamp,
                1.0,
            )
            SijM = vmaplinestrengh(
                Tarr,
                self.mdb.logsij0,
                self.mdb.nu_lines,
                self.mdb.elower,
                qr_K.T,
                Tref_original,
            )
            sigmaDM = jit(vmap(doppler_sigma, (None, 0, None)))(
                self.mdb.nu_lines, Tarr, self.mdb.atomicmass
            )

        return xsmatrix_lpf(numatrix, sigmaDM, gammaLM, SijM)

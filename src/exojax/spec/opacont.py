"""opacity continuum calculator class

Notes:
    Opa does not assume any T-P structure, no fixed T, P, mmr grids.

"""

from exojax.spec.hitrancia import interp_logacia_vector
from exojax.spec.hitrancia import interp_logacia_matrix
from exojax.spec.mie import mie_lognormal_pymiescatt
from exojax.spec.hminus import log_hminus_continuum
from exojax.spec.rayleigh import xsvector_rayleigh_gas
import warnings
import jax.numpy as jnp
from jax import vmap
import numpy as np

__all__ = ["OpaCIA", "Opahminus", "OpaRayleigh", "OpaMie"]


class OpaCont:
    """Common Opacity Calculator Class"""

    __slots__ = [
        "opainfo",
    ]

    def __init__(self):
        self.method = None  # which opacity cont method is used
        self.ready = False  # ready for opacity computation


class OpaCIA(OpaCont):
    """Opacity Continuum Calculator Class for CIA"""

    def __init__(self, cdb, nu_grid):
        """initialization of opacity calcluator for CIA

        Args:
            cdb (_type_): Continuum database
            nu_grid (_type_): _wavenumber grid
        """
        self.method = "cia"
        self.warning = True
        self.nu_grid = nu_grid
        self.cdb = cdb
        self.ready = True

    def logacia_vector(self, T):
        return interp_logacia_vector(
            T, self.nu_grid, self.cdb.nucia, self.cdb.tcia, self.cdb.logac
        )

    def logacia_matrix(self, temperatures):
        return interp_logacia_matrix(
            temperatures, self.nu_grid, self.cdb.nucia, self.cdb.tcia, self.cdb.logac
        )


class OpaHminus(OpaCont):
    def __init__(self, nu_grid):
        self.method = "hminus"
        self.warning = True
        self.nu_grid = nu_grid
        self.ready = True

    def logahminus_matrix(self, temperatures, number_density_e, number_density_h):
        """absorption coefficient (cm-1) matrix of H- continuum

        Args:
            temperatures (_type_): temperature array
            number_density_e (_type_): number density of electron in cgs
            number_density_h (_type_): number density of H in cgs

        Returns:
            log10(absorption coefficient in cm-1 ) [Nlayer,Nnu]

        """
        return log_hminus_continuum(
            self.nu_grid, temperatures, number_density_e, number_density_h
        )


class OpaRayleigh(OpaCont):
    def __init__(self, nu_grid, molname):
        """sets opa

        Args:
            nu_grid (float, array): wavenumber grid
            molname (str): gas molecule name, such as "N2"
        """
        self.method = "rayleigh"
        self.nu_grid = nu_grid
        self.molname = molname
        self.set_auto_polarizability()
        self.set_auto_king_factor()
        self.check_ready()

    def set_auto_polarizability(self):
        from exojax.atm.polarizability import polarizability

        try:
            self.polarizability = polarizability[self.molname]
        except:
            self.polarizability = None
            warnings.warn(
                "No polarizability found. Set opa.polarizability by yourself."
            )

    def set_auto_king_factor(self):
        from exojax.atm.polarizability import king_correction_factor

        try:
            self.king_factor = king_correction_factor[self.molname]
        except:
            self.king_factor = 1.0
            warnings.warn(
                "No king correction factor found. Applied to 1. you can modify by setting opa.king_factor."
            )

    def check_ready(self):
        if self.polarizability is None:
            print("no opa.polarizability. Not ready for OpaRayleigh yet.")
            self.ready = False
        else:
            print("Ready for OpaRayleigh.")
            self.ready = True

    def xsvector(self):
        """computes cross section vector of the Rayleigh scattering

        Returns:
            float, array: Rayleigh scattring cross section vector [Nnus] in cm2
        """
        return xsvector_rayleigh_gas(
            self.nu_grid, self.polarizability, king_factor=self.king_factor
        )


class OpaMie(OpaCont):
    def __init__(
        self,
        pdb,
        nu_grid,
    ):
        self.method = "mie"
        self.nu_grid = nu_grid
        self.pdb = pdb
        self.ready = True

    def mieparams_vector(self, rg, sigmag):
        """interpolate the Mie parameters vector (Nnu: wavenumber direction) from Miegrid

        Args:
            rg (float): rg parameter in the lognormal distribution of condensate size, defined by (9) in AM01
            sigmag (float): sigmag parameter in the lognormal distribution of condensate size, defined by (9) in AM01

        Notes:
            AM01 = Ackerman and Marley 2001
            Volume extinction coefficient (1/cm) for the number density N can be computed by beta_extinction = N*beta0_extinction/N0

        Returns:
            sigma_extinction, extinction cross section (cm2) = volume extinction coefficient (1/cm) normalized by the reference numbver density N0.
            sigma_scattering, scattering cross section (cm2) = volume extinction coefficient (1/cm) normalized by the reference numbver density N0.
            asymmetric factor, (mean g)
        """

        # loads grid (mieparams in cgs)
        sigexg, sigscg, gg = (
            self.pdb.mieparams_cgs_at_refraction_index_wavenumber_from_miegrid(
                rg, sigmag
            )
        )

        # interpolation
        sigma_extinction = jnp.interp(
            self.nu_grid, self.pdb.refraction_index_wavenumber, sigexg
        )
        sigma_scattering = jnp.interp(
            self.nu_grid, self.pdb.refraction_index_wavenumber, sigscg
        )
        asymmetric_factor = jnp.interp(
            self.nu_grid, self.pdb.refraction_index_wavenumber, gg
        )

        return sigma_extinction, sigma_scattering, asymmetric_factor

    def mieparams_matrix(self, rg_layer, sigmag_layer):
        """interpolate the Mie parameters matrix (Nlayer x Nnu) from Miegrid
        Args:
            rg_layer (1d array): layer rg parameters  in the lognormal distribution of condensate size, defined by (9) in AM01
            sigmag_layer (1d array): layer sigmag parameters in the lognormal distribution of condensate size, defined by (9) in AM01

        Notes:
            AM01 = Ackerman and Marley 2001
            Volume extinction coefficient (1/cm) for the number density N can be computed by beta_extinction = N*beta0_extinction/N0

        Returns:
            sigma_extinction matrix, extinction cross section (cm2) = volume extinction coefficient (1/cm) normalized by the reference number density N0
            omega0  matrix, single scattering albedo
            g  matrix, asymmetric factor (mean g)
        """

        f = vmap(self.mieparams_vector, (0, 0), 0)
        return f(rg_layer, sigmag_layer)

    def mieparams_vector_direct_from_pymiescatt(self, rg, sigmag):
        """compute the Mie parameters vector (Nnu: wavenumber direction) from pymiescatt direclty (slow), i.e. no need of Miegrid, the unit is in cgs

        Args:
            rg (float): rg parameter in the lognormal distribution of condensate size, defined by (9) in AM01
            sigmag (float): sigmag parameter in the lognormal distribution of condensate size, defined by (9) in AM01

        Notes:
            AM01 = Ackerman and Marley 2001
            Volume extinction coefficient (1/cm) for the number density N can be computed by beta_extinction = N*beta0_extinction/N0

        Returns:
            sigma_extinction, extinction cross section (cm2) = volume extinction coefficient (1/cm) normalized by the reference numbver density N0.
            sigma_scattering, scattering cross section (cm2) = volume extinction coefficient (1/cm) normalized by the reference numbver density N0.
            asymmetric factor, (mean g)
        """
        from exojax.spec.mie import auto_rgrid
        from tqdm import tqdm

        # restrict wavenumber grid
        nind = len(self.pdb.refraction_index_wavenumber)
        numin = np.min(self.nu_grid)
        imin = np.searchsorted(self.pdb.refraction_index_wavenumber, numin)
        imin = np.max([imin - 1, 0])
        numax = np.max(self.nu_grid)
        imax = np.searchsorted(self.pdb.refraction_index_wavenumber, numax)
        imax = np.min([imax + 1, nind])

        refraction_index_wavenumber_restricted = self.pdb.refraction_index_wavenumber[
            imin:imax
        ]
        nind = len(refraction_index_wavenumber_restricted)
        refraction_index_restricted = self.pdb.refraction_index[imin:imax]

        # generates grid
        convfactor_to_cgs = (
            1.0e-8 / self.pdb.N0
        )  # conversion to cgs(1/Mega meter to 1/cm)

        sigexg = np.zeros(nind)
        sigscg = np.zeros(nind)
        gg = np.zeros(nind)

        cm2nm = 1.0e7
        rg_nm = rg * cm2nm
        rgrid = auto_rgrid(rg_nm, sigmag)
        for ind_m, m in enumerate(tqdm(refraction_index_restricted)):
            coeff = mie_lognormal_pymiescatt(
                m,
                refraction_index_wavenumber_restricted[ind_m],
                sigmag,
                rg_nm,
                self.pdb.N0,
                rgrid,
            )

            sigexg[ind_m] = coeff[0] * convfactor_to_cgs
            sigscg[ind_m] = coeff[1] * convfactor_to_cgs
            gg[ind_m] = coeff[3]

        # interpolation
        sigma_extinction = np.interp(
            self.nu_grid, refraction_index_wavenumber_restricted, sigexg
        )
        sigma_scattering = np.interp(
            self.nu_grid, refraction_index_wavenumber_restricted, sigscg
        )
        asymmetric_factor = np.interp(
            self.nu_grid, refraction_index_wavenumber_restricted, gg
        )

        return sigma_extinction, sigma_scattering, asymmetric_factor

    def mieparams_matrix_direct_from_pymiescatt(self, rg_layer, sigmag_layer):
        """compute the Mie parameters matrix (Nlayer x Nnu) from pymiescatt direclty (slow), i.e. no need of Miegrid
        Args:
            rg_layer (1d array): layer rg parameters  in the lognormal distribution of condensate size, defined by (9) in AM01
            sigmag_layer (1d array): layer sigmag parameters in the lognormal distribution of condensate size, defined by (9) in AM01

        Notes:
            AM01 = Ackerman and Marley 2001
            Volume extinction coefficient (1/cm) for the number density N can be computed by beta_extinction = N*beta0_extinction/N0

        Returns:
            sigma_extinction matrix, extinction cross section (cm2) = volume extinction coefficient (1/cm) normalized by the reference number density N0
            omega0  matrix, single scattering albedo
            g  matrix, asymmetric factor (mean g)
        """

        f = vmap(self.mieparams_vector_direct_from_pymiescatt, (0, 0), 0)
        return f(rg_layer, sigmag_layer)

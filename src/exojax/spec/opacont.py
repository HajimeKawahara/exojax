"""opacity continuum calculator class

Notes:
    Opa does not assume any T-P structure, no fixed T, P, mmr grids.

"""
from exojax.utils.grids import nu2wav
from exojax.spec.hitrancia import interp_logacia_vector
from exojax.spec.hitrancia import interp_logacia_matrix
import jax.numpy as jnp
from jax import vmap

__all__ = ["OpaCIA"]


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

    def logacia_matrix(self, temperature):
        return interp_logacia_matrix(
            temperature, self.nu_grid, self.cdb.nucia, self.cdb.tcia, self.cdb.logac
        )


class OpaHminus(OpaCont):
    def __init__(self):
        self.method = "hminus"
        ValueError("Not implemented yet")


class OpaRayleigh(OpaCont):
    def __init__(self):
        self.method = "rayleigh"
        ValueError("Not implemented yet")


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
        """computes the Mie parameter vector (Nnu: wavenumber direction)

        Args:
            rg (float): rg parameter in the lognormal distribution of condensate size, defined by (9) in AM01
            sigmag (float): sigmag parameter in the lognormal distribution of condensate size, defined by (9) in AM01

        Notes:
            AM01 = Ackerman and Marley 2001

        Returns:
            beta0_extinction vector, volume extinction coefficient (1/cm) normalized by the reference number density N0
            omega0  vector, single scattering albedo
            g  vector, asymmetric factor (mean g)
        """
        dtau_g, w_g, g_g = self.pdb.mieparams_at_refraction_index_wavenumber(rg, sigmag)
        dtau = jnp.interp(self.nu_grid, self.pdb.refraction_index_wavenumber, dtau_g)
        w = jnp.interp(self.nu_grid, self.pdb.refraction_index_wavenumber, w_g)
        g = jnp.interp(self.nu_grid, self.pdb.refraction_index_wavenumber, g_g)

        return dtau, w, g

    def mieparams_matrix(self, rg_layer, sigmag_layer):
        """computes the Mie parameter matrix (Nlayer x Nnu)
        Args:
            rg_layer (1d array): layer rg parameters  in the lognormal distribution of condensate size, defined by (9) in AM01
            sigmag_layer (1d array): layer sigmag parameters in the lognormal distribution of condensate size, defined by (9) in AM01

        Notes:
            AM01 = Ackerman and Marley 2001

        Returns:
            beta0_extinction matrix, volume extinction coefficient (1/cm) normalized by the reference number density N0
            omega0  matrix, single scattering albedo
            g  matrix, asymmetric factor (mean g)
        """

        f = vmap(self.mieparams_vector, (0, 0), 0)
        return f(rg_layer, sigmag_layer)

"""opacity continuum calculator class

Notes:
    Opa does not assume any T-P structure, no fixed T, P, mmr grids.

"""
from exojax.utils.grids import nu2wav
from exojax.spec.hitrancia import interp_logacia_vector
from exojax.spec.hitrancia import interp_logacia_matrix

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


class OpaCIA(OpaCont):
    """Opacity Continuum Calculator Class for CIA

    """
    def __init__(self, cdb, nu_grid, wavelength_order="descending"):
        """
        _summary_

        Args:
            cdb (_type_): Continuum database 
            nu_grid (_type_): _wavenumber grid
            wavelength_order (str, optional): _description_. Defaults to "descending".
        """
        self.method = "cia"
        self.warning = True
        self.nu_grid = nu_grid
        #self.wavelength_order = wavelength_order
        #self.wav = nu2wav(self.nu_grid,
        #                  wavelength_order=self.wavelength_order,
        #                  unit="AA")
        self.cdb = cdb
        self.ready = True

    def logacia_vector(self, T):
        return interp_logacia_vector(T, self.nu_grid, self.cdb.nucia,
                                     self.cdb.tcia, self.cdb.logac)

    def logacia_matrix(self, temperature):
        return interp_logacia_matrix(temperature, self.nu_grid, self.cdb.nucia,
                                     self.cdb.tcia, self.cdb.logac)

class OpaHminus(OpaCont):
    def __init__(self):
        self.method = "hminus"
        ValueError("Not implemented yet")

class OpaRayleigh(OpaCont):
    def __init__(self):
        self.method = "rayleigh"
        ValueError("Not implemented yet")

class OpaMie(OpaCont):
    def __init__(self):
        self.method = "mie"

    #def xsvector()



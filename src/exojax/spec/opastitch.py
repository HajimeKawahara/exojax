from exojax.spec.opacalc import OpaPremodit
from exojax.utils.grids import nu2wav
from exojax.utils.instfunc import resolution_eslog
from exojax.signal.ola import overlap_and_add
from exojax.signal.ola import ola_output_length
from exojax.signal.ola import overlap_and_add_matrix
import numpy as np
import jax.numpy as jnp

class OpaPremoditStitch:

    def __init__(
        self,
        mdb,
        nu_grid,
        ndiv,
        cutwing=1.0,
        diffmode=0,
        broadening_resolution={"mode": "manual", "value": 0.2},
        auto_trange=None,
        manual_params=None,
        dit_grid_resolution=None,
        allow_32bit=False,
        wavelength_order="descending",
        version_auto_trange=2,
    ):
        self.mdb = mdb
        self.nu_grid = nu_grid
        self.ndiv = ndiv
        self.cutwing = cutwing
        self.method = "premodit_stitch"
        self.diffmode = diffmode
        self.broadening_resolution = broadening_resolution
        self.auto_trange = auto_trange
        self.manual_params = manual_params
        self.dit_grid_resolution = dit_grid_resolution
        self.allow_32bit = allow_32bit
        self.wavelength_order = wavelength_order
        self.wav_all = nu2wav(
            self.nu_grid, wavelength_order=self.wavelength_order, unit="AA"
        )
        self.resolution = resolution_eslog(nu_grid)
        self.mdb = mdb
        self.version_auto_trange = version_auto_trange

        self.check_nu_grid_reducible()
        self.nu_grid_list = np.array_split(nu_grid, self.ndiv)
        self.set_opa_list()
        self.set_ola_lengths_from_opa_list_zero()
    
    def set_opa_list(self):
        self.opa_list = []
        for nu_grid in self.nu_grid_list:
            self.opa_list.append(
                OpaPremodit(
                    self.mdb,
                    nu_grid,
                    diffmode=self.diffmode,
                    broadening_resolution=self.broadening_resolution,
                    auto_trange=self.auto_trange,
                    manual_params=self.manual_params,
                    dit_grid_resolution=self.dit_grid_resolution,
                    allow_32bit=self.allow_32bit,
                    alias="open",
                    cutwing=self.cutwing,
                    wavelength_order=self.wavelength_order,
                    version_auto_trange=self.version_auto_trange,
                )
            )
    
    def set_ola_lengths_from_opa_list_zero(self):
        self.filter_length_oneside = self.opa_list[0].filter_length_oneside
        self.filter_length = self.opa_list[0].filter_length
        self.div_length = self.opa_list[0].div_length

    def check_nu_grid_reducible(self):
        if len(self.nu_grid) % self.ndiv != 0:
            msg = (
                "nu_grid_all length = "
                + str(len(self.nu_grid))
                + " cannot be divided by stitch="
                + str(self.ndiv)
            )
            raise ValueError(msg)

    def xsvector(self, T, P):
        xsv_matrix = []
        for opa in self.opa_list:
            xsv_matrix.append(opa.xsvector(T, P))
        xsv_matrix = jnp.vstack(xsv_matrix)

        output_length = ola_output_length(self.ndiv, self.div_length, self.filter_length)
        xsv_ola_stitch = overlap_and_add(xsv_matrix,output_length, self.div_length)
        return xsv_ola_stitch[self.filter_length_oneside:-self.filter_length_oneside]

    def xsmatrix(self, Tarr, Parr):
        xsm_matrix = []
        for opa in self.opa_list:
            xsm_matrix.append(opa.xsmatrix(Tarr, Parr))
        xsm_matrix = jnp.array(xsm_matrix)

        output_length = ola_output_length(self.ndiv, self.div_length, self.filter_length)
        xsmatrix_ola_stitch = overlap_and_add_matrix(xsm_matrix, output_length, self.div_length)
        return xsmatrix_ola_stitch[:,self.filter_length_oneside:-self.filter_length_oneside]


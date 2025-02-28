"""opastitch -- opa with nu stitching
"""

from exojax.spec.opacalc import OpaPremodit
from exojax.utils.grids import nu2wav
from exojax.utils.instfunc import resolution_eslog
from exojax.signal.ola import overlap_and_add
from exojax.signal.ola import ola_output_length
from exojax.signal.ola import overlap_and_add_matrix
from jax.lax import scan
from jax.lax import dynamic_slice
from jax import vmap
import numpy as np
from jax.lax import fori_loop
import jax.numpy as jnp


class OpaPremoditStitch:
    """premodit with nu stitching"""

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
        """initializes the premodit with nu stitching

        Args:
            mdb (mdb): molecular database
            nu_grid (array): (original) wavenumber grid, should be reducilbe by ndiv
            ndiv (int): the number of division (stitching)
            cutwing (float, optional): wingcut for the convolution used in open cross section. Defaults to 1.0.
            diffmode (int, optional): _description_. Defaults to 0.
            broadening_resolution (dict, optional): definition of the broadening parameter resolution. Default to {"mode": "manual", value: 0.2}. See Note.
            auto_trange (optional): temperature range [Tl, Tu], in which line strength is within 1 % prescision. Defaults to None.
            manual_params (optional): premodit parameter set [dE, Tref, Twt]. Defaults to None.
            dit_grid_resolution (float, optional): force to set broadening_parameter_resolution={mode:manual, value: dit_grid_resolution}), ignores broadening_parameter_resolution.
            allow_32bit (bool, optional): If True, allow 32bit mode of JAX. Defaults to False.
            wavlength order: wavelength order: "ascending" or "descending"
            version_auto_trange: version of the default elower grid trange (degt) file, Default to 2 since Jan 2024.
        """

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

        # uses OpePremodit to sets parameters
        self.set_opa_list()
        self.set_common_parameters_from_opa_list_zero()
        self.sets_opainfo_list()
        #del self.opa_list

        self._sets_capable_opacalculators()

    def set_opa_list(self):
        """set opa_list from nu_grid_list"""
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

    def set_common_parameters_from_opa_list_zero(self):
        """set filter_length, filter_length_oneside, div_length from opa_list[0]"""
        self.filter_length_oneside = self.opa_list[0].filter_length_oneside
        self.filter_length = self.opa_list[0].filter_length
        self.div_length = self.opa_list[0].div_length
        self.Tref_broadening = self.opa_list[0].Tref_broadening
        self.Twt = self.opa_list[0].Twt
        self.Tref = self.opa_list[0].Tref

    def check_nu_grid_reducible(self):
        """check if nu_grid is reducible by ndiv

        Raises:
            ValueError: if nu_grid is not reducible by ndiv
        """
        if len(self.nu_grid) % self.ndiv != 0:
            msg = (
                "nu_grid_all length = "
                + str(len(self.nu_grid))
                + " cannot be divided by stitch="
                + str(self.ndiv)
            )
            raise ValueError(msg)
            
    def sets_opainfo_list(self):

        self.lbd_coeff_jnp_array = []
        self.nu_grid_extended_jnp_array = []
        for opa in self.opa_list:
            (
                lbd_coeff,
                multi_index_uniqgrid,
                elower_grid,
                ngamma_ref_grid,
                n_Texp_grid,
                R,
                _,
            ) = opa.opainfo
            self.lbd_coeff_jnp_array.append(lbd_coeff)
            self.nu_grid_extended_jnp_array.append(opa.nu_grid_extended)
            
        self.lbd_coeff_jnp_array = jnp.array(self.lbd_coeff_jnp_array)
        self.nu_grid_extended_jnp_array = jnp.array(self.nu_grid_extended_jnp_array)
        
        self.multi_index_uniqgrid = multi_index_uniqgrid
        self.elower_grid = elower_grid
        self.ngamma_ref_grid = ngamma_ref_grid
        self.n_Texp_grid = n_Texp_grid            
        self.R = R

    def _sets_capable_opacalculators(self):
        """sets capable open opacalculators"""
        # opa calculators for PreMODIT
        from exojax.spec.premodit import xsvector_nu_open_zeroth
        from exojax.spec.premodit import xsvector_nu_open_first
        from exojax.spec.premodit import xsvector_nu_open_second
        from exojax.spec.premodit import xsmatrix_nu_open_zeroth
        from exojax.spec.premodit import xsmatrix_nu_open_first
        from exojax.spec.premodit import xsmatrix_nu_open_second

        self.xsvector_nu_open = {
            0: xsvector_nu_open_zeroth,
            1: xsvector_nu_open_first,
            2: xsvector_nu_open_second,
        }
        self.xsmatrix_nu_open = {
            0: xsmatrix_nu_open_zeroth,
            1: xsmatrix_nu_open_first,
            2: xsmatrix_nu_open_second,
        }

    def xsvector(self, T, P):
        """cross section vector with stitching

        Args:
            T (float): temperature in K
            P (float): pressure in bar

        Returns:
            array: cross section vector [Nnus]
        """
        from exojax.spec import normalized_doppler_sigma

        nsigmaD = normalized_doppler_sigma(T, self.mdb.molmass, self.R)

        if self.mdb.dbtype == "hitran":
            qt = self.mdb.qr_interp(self.mdb.isotope, T, self.Tref)
        elif self.mdb.dbtype == "exomol":
            qt = self.mdb.qr_interp(T, self.Tref)

        xsvector_nu_func = self.xsvector_nu_open[self.diffmode]
        output_length = ola_output_length(
            self.ndiv, self.div_length, self.filter_length
        )
        
        def floop(icarry, lbd_coeff):
            nu_grid_each = dynamic_slice(self.nu_grid, (icarry,), (self.div_length,))
        
            xsv_nu = xsvector_nu_func(
                T,
                P,
                nsigmaD,
                lbd_coeff,
                self.Tref,
                self.R,
                nu_grid_each,
                self.elower_grid,
                self.multi_index_uniqgrid,
                self.ngamma_ref_grid,
                self.n_Texp_grid,
                qt,
                self.Tref_broadening,
                self.filter_length_oneside,
                self.Twt,
            )

            return icarry + self.div_length, xsv_nu
        _, xsv_matrix = scan(floop, 0, self.lbd_coeff_jnp_array)
        xsv_matrix = xsv_matrix/self.nu_grid_extended_jnp_array
        xsv_ola_stitch = overlap_and_add(xsv_matrix, output_length, self.div_length)
        
        return xsv_ola_stitch[self.filter_length_oneside : -self.filter_length_oneside]

    def xsmatrix(self, Tarr, Parr):
        """cross section marix with stitching
        
        Args:
            Tarr (array): temperature array in K [Nlayer]
            Parr (array): pressure array in bar [Nlayer]

        Returns:
            2D array: cross section matrix [Nlayer, Nnus]
        """
        

        if self.mdb.dbtype == "hitran":
            qtarr = vmap(self.mdb.qr_interp, (None, 0, None))(
                self.mdb.isotope, Tarr, self.Tref
            )
        elif self.mdb.dbtype == "exomol":
            qtarr = vmap(self.mdb.qr_interp, (0, None))(Tarr, self.Tref)

        
        xsmatrix_nu_func = self.xsmatrix_nu_open[self.diffmode]
        output_length = ola_output_length(
            self.ndiv, self.div_length, self.filter_length
        )
        
        def floop(icarry, lbd_coeff):
            nu_grid_each = dynamic_slice(self.nu_grid, (icarry,), (self.div_length,))
        
            xsm_nu = xsmatrix_nu_func(
                Tarr,
                Parr,
                self.Tref,
                self.R,
                lbd_coeff,
                nu_grid_each,
                self.ngamma_ref_grid,
                self.n_Texp_grid,
                self.multi_index_uniqgrid,
                self.elower_grid,
                self.mdb.molmass,
                qtarr,
                self.Tref_broadening,
                self.filter_length_oneside,
                Twt=None
            )          
            

            return icarry + self.div_length, xsm_nu
        _, xsm_matrix = scan(floop, 0, self.lbd_coeff_jnp_array)
        xsm_matrix = xsm_matrix/self.nu_grid_extended_jnp_array[:,jnp.newaxis,:]
        xsmatrix_ola_stitch = overlap_and_add_matrix(
            xsm_matrix, output_length, self.div_length
        )
        return xsmatrix_ola_stitch[:, self.filter_length_oneside : -self.filter_length_oneside]
        
        
    def xsvector_for_loop(self, T, P):
        """cross section vector with stitching using for loop

        Args:
            T (float): temperature in K
            P (float): pressure in bar

        Returns:
            array: cross section vector [Nnus]
        """
        xsv_matrix = []
        for opa in self.opa_list:
            xsv_matrix.append(opa.xsvector(T, P))
        xsv_matrix = jnp.vstack(xsv_matrix)
        output_length = ola_output_length(
            self.ndiv, self.div_length, self.filter_length
        )
        xsv_ola_stitch = overlap_and_add(xsv_matrix, output_length, self.div_length)
        return xsv_ola_stitch[self.filter_length_oneside : -self.filter_length_oneside]

    def xsmatrix_for_loop(self, Tarr, Parr):
        """cross section matrix with stitching using for loop

        Args:
            Tarr (array): temperature array in K [Nlayer]
            Parr (array): pressure array in bar [Nlayer]

        Returns:
            2D array: cross section matrix [Nlayer, Nnus]
        """
        xsm_matrix = []
        for opa in self.opa_list:
            xsm_matrix.append(opa.xsmatrix(Tarr, Parr))
        xsm_matrix = jnp.array(xsm_matrix)

        output_length = ola_output_length(
            self.ndiv, self.div_length, self.filter_length
        )
        xsmatrix_ola_stitch = overlap_and_add_matrix(
            xsm_matrix, output_length, self.div_length
        )
        return xsmatrix_ola_stitch[
            :, self.filter_length_oneside : -self.filter_length_oneside
        ]

import warnings
import jax.numpy as jnp
import numpy as np

from exojax.atm.atmprof import (
    atmprof_gray,
    atmprof_Guillot,
    atmprof_powerlow,
    normalized_layer_height,
)
from exojax.atm.idealgas import number_density
from exojax.rt.layeropacity import (
    layer_optical_depth,
    layer_optical_depth_ckd,
    layer_optical_depth_clouds_lognormal,
)
from exojax.utils.constants import logkB, logm_ucgs


class ArtCommon:
    """Common Atmospheric Radiative Transfer"""

    @property
    def pressure_top_boundary(self):
        """Pressure at the top boundary of the atmospheric grid in bar."""
        return self.pressure_boundary[0]

    @property
    def pressure_btm_boundary(self):
        """Pressure at the bottom boundary of the atmospheric grid in bar."""
        return self.pressure_boundary[-1]

    @staticmethod
    def _pressure_profile_from_exact_boundaries(
        pressure_top_boundary,
        pressure_btm_boundary,
        nlayer,
        reference_point,
    ):
        from exojax.atm.atmprof import pressure_layer_logspace_from_boundaries

        grid_arguments = {
            "log_pressure_top_boundary": np.log10(pressure_top_boundary),
            "log_pressure_btm_boundary": np.log10(pressure_btm_boundary),
            "nlayer": nlayer,
            "reference_point": reference_point,
        }
        (
            _,
            _,
            pressure_decrease_rate,
            pressure_boundary,
        ) = pressure_layer_logspace_from_boundaries(
            **grid_arguments,
            numpy=True,
        )
        pressure_boundary = pressure_boundary.copy()
        pressure_boundary[0] = pressure_top_boundary
        pressure_boundary[-1] = pressure_btm_boundary
        pressure = (
            pressure_boundary[:-1] ** (1.0 - reference_point)
            * pressure_boundary[1:] ** reference_point
        )
        dpressure = np.diff(pressure_boundary)

        from jax import config

        active_dtype = np.dtype(np.float64 if config.x64_enabled else np.float32)
        smallest_normal = np.finfo(active_dtype).tiny
        with np.errstate(over="ignore", under="ignore"):
            active_boundary = pressure_boundary.astype(active_dtype)
            active_pressure = pressure.astype(active_dtype)
            active_dpressure = np.diff(active_boundary)
            active_decrease_rate = np.asarray(
                pressure_decrease_rate, dtype=active_dtype
            )
        active_grid_values = (
            active_boundary,
            active_pressure,
            active_dpressure,
            active_decrease_rate,
        )
        if any(
            not np.all(np.isfinite(value)) or np.any(value < smallest_normal)
            for value in active_grid_values
        ) or np.any(active_dpressure <= 0.0):
            raise ValueError(
                "Pressure grid cannot be represented by the active JAX dtype."
            )
        if active_decrease_rate >= 1.0:
            raise ValueError(
                "Pressure grid cannot be represented by the active JAX dtype."
            )
        return (
            pressure,
            dpressure,
            pressure_decrease_rate,
            pressure_boundary,
        )

    @classmethod
    def from_pressure_boundaries(
        cls,
        pressure_top_boundary,
        pressure_btm_boundary,
        nlayer,
        **kwargs,
    ):
        """Initialize an atmospheric model from its exact pressure boundaries.

        Args:
            pressure_top_boundary (float): Pressure at the top boundary in bar.
            pressure_btm_boundary (float): Pressure at the bottom boundary in bar.
            nlayer (int): Number of atmospheric layers.
            **kwargs: Additional arguments passed to the subclass constructor.

        Returns:
            ArtCommon: An instance whose pressure grid spans the specified
                boundaries exactly.

        Notes:
            ``pressure`` contains representative layer pressures, while
            ``pressure_boundary`` contains the physical layer boundaries.
            Subclass-specific required arguments, such as ``opalayer``, must be
            supplied as keyword arguments.
        """
        if "pressure_top" in kwargs or "pressure_btm" in kwargs:
            raise TypeError(
                "pressure_top and pressure_btm cannot be used with "
                "from_pressure_boundaries."
            )
        if type(nlayer) is not int:
            raise ValueError("Number of the layer should be integer")
        if nlayer < 1:
            raise ValueError("Number of the layer should be positive")

        if np.ndim(pressure_top_boundary) != 0 or np.ndim(
            pressure_btm_boundary
        ) != 0:
            raise ValueError("Pressure boundaries should be scalar values.")
        try:
            pressure_top_boundary = float(pressure_top_boundary)
            pressure_btm_boundary = float(pressure_btm_boundary)
        except (TypeError, ValueError) as err:
            raise ValueError("Pressure boundaries should be scalar values.") from err
        if not np.isfinite(pressure_top_boundary) or not np.isfinite(
            pressure_btm_boundary
        ):
            raise ValueError("Pressure boundaries should be finite.")
        if pressure_top_boundary <= 0.0 or pressure_btm_boundary <= 0.0:
            raise ValueError("Pressure boundaries should be positive.")
        if pressure_btm_boundary <= pressure_top_boundary:
            raise ValueError(
                "Pressure at the bottom boundary should be higher than that "
                "at the top boundary."
            )

        log_pressure_top_boundary = np.log10(pressure_top_boundary)
        log_pressure_btm_boundary = np.log10(pressure_btm_boundary)
        dlog_pressure = (
            log_pressure_btm_boundary - log_pressure_top_boundary
        ) / nlayer
        constructor_kwargs = {
            "pressure_top": 10.0
            ** (log_pressure_top_boundary + 0.5 * dlog_pressure),
            "pressure_btm": 10.0
            ** (log_pressure_btm_boundary - 0.5 * dlog_pressure),
            "nlayer": nlayer,
            **kwargs,
        }

        if cls.__new__ is object.__new__:
            instance = cls.__new__(cls)
        else:
            instance = cls.__new__(cls, **constructor_kwargs)
        if not isinstance(instance, cls):
            raise TypeError(f"{cls.__name__}.__new__ did not return an instance.")

        missing = object()
        previous_specification = getattr(
            instance, "_pressure_boundary_specification", missing
        )
        instance._pressure_boundary_specification = (
            pressure_top_boundary,
            pressure_btm_boundary,
        )
        try:
            type(instance).__init__(instance, **constructor_kwargs)
        except BaseException:
            if previous_specification is missing:
                if hasattr(instance, "_pressure_boundary_specification"):
                    del instance._pressure_boundary_specification
            else:
                instance._pressure_boundary_specification = previous_specification
            raise
        instance._pressure_boundary_specification = (
            pressure_top_boundary,
            pressure_btm_boundary,
        )
        return instance

    def __init__(
        self, pressure_top, pressure_btm, nlayer, nu_grid=None, warn_no_nu_grid=True
    ):
        """initialization of art

        Args:
            pressure_top (float): Representative pressure of the top layer in bar.
            pressure_btm (float): Representative pressure of the bottom layer in bar.
            nlayer (int): # of atmospheric layers
            nu_grid (nd.array, optional): wavenumber grid in cm-1
            warn_no_nu_grid (bool, optional): Warn when ``nu_grid`` is not given.
        """
        self.artinfo = None
        self.method = None  # which art is used
        self.ready = False  # ready for art computation
        self.Tlow = 0.0
        self.Thigh = jnp.inf
        self.reference_point = 0.5  # ref point (r) for pressure layers

        if nu_grid is None and warn_no_nu_grid:
            warnings.warn(
                "nu_grid is not given. specify nu_grid when using 'run' ", UserWarning
            )
        self.nu_grid = nu_grid

        self.pressure_top = pressure_top
        self.pressure_btm = pressure_btm
        self.nlayer = nlayer
        self.check_pressure()
        self.log_pressure_btm = np.log10(self.pressure_btm)
        self.log_pressure_top = np.log10(self.pressure_top)
        self.init_pressure_profile()

        self.fguillot = 0.25

    def atmosphere_height(
        self, temperature, mean_molecular_weight, radius_btm, gravity_btm
    ):
        """atmosphere height and radius

        Args:
            temperature (1D array): temparature profile (Nlayer)
            mean_molecular_weight (float/1D array):
                mean molecular weight profile (float/Nlayer)
            radius_btm (float):
                the bottom radius of the atmospheric layer
            gravity_btm (float): the bottom gravity in cm/s2 at radius_btm,
                i.e. G M_p/radius_btm**2

        Returns:
            1D array: height normalized by radius_btm (Nlayer)
            1D array: layer radius r_n normalized by radius_btm (Nlayer)

        Notes:
            Our definitions of the radius_lower, radius_layer, and height are as follows:
            n=0,1,...,N-1
            radius_lower[N-1] = radius_btm (i.e. R0)
            radius_lower[n-1] = radius_lower[n] + height[n]
            "normalized" means physical length divided by radius_btm


        """
        normalized_height, normalized_radius_lower = normalized_layer_height(
            temperature,
            self.pressure_decrease_rate,
            mean_molecular_weight,
            radius_btm,
            gravity_btm,
        )
        return normalized_height, normalized_radius_lower

    def constant_gravity_profile(self, value):
        return value * np.array([np.ones_like(self.pressure)]).T

    def gravity_profile(
        self, temperature, mean_molecular_weight, radius_btm, gravity_btm
    ):
        """gravity layer profile assuming hydrostatic equilibrium

        Args:
            temperature (1D array): temparature profile (Nlayer)
            mean_molecular_weight (float/1D array):
                mean molecular weight profile (float/Nlayer)
            radius_btm (float): the bottom radius of the atmospheric layer
            gravity_btm (float):
                the bottom gravity in cm/s2 at radius_btm,
                i.e. G M_p/radius_btm**2

        Returns:
            2D array:
                gravity in cm/s2 (Nlayer, 1), suitable for the input of opacity_profile_lines
        """
        normalized_height, normalized_radius_lower = self.atmosphere_height(
            temperature, mean_molecular_weight, radius_btm, gravity_btm
        )
        normalized_radius_layer = normalized_radius_lower + 0.5 * normalized_height
        return jnp.array([gravity_btm / normalized_radius_layer**2]).T

    def constant_profile(self, value):
        return value * np.ones_like(self.pressure)

    def constant_mmr_profile(self, value):
        return self.constant_profile(value)

    def opacity_profile_lines(self, xs, mixing_ratio, molmass, gravity):
        raise ValueError(
            "opacity_profile_lines was removed. Use opacity_profile_xs instead"
        )

    def opacity_profile_xs(self, xs, mixing_ratio, molmass, gravity):
        """opacity profile (delta tau) from cross section matrix or vector, molecular line/Rayleigh scattering

        Args:
            xs (3D array/2D array): cross section matrix/ cross section tensor for CKD 
                i.e. xsmatrix (Nlayer, N_wavenumber) or xstensor_ckd (Nlayer, Ng, Nbands) 
            mixing_ratio (1D array): mass mixing ratio (Nlayer,), Nlayer, (or volume mixing ratio profile)
            molmass (float): molecular mass (or mean molecular weight)
            gravity (float/1D profile): constant or 1d profile of gravity in cgs

        Returns:
            dtau: opacity profile, whose element is optical depth in each layer.
        """
        return layer_optical_depth(
            self.dParr, jnp.abs(xs), mixing_ratio, molmass, gravity
        )

    def opacity_profile_xs_ckd(self, xstensor_ckd, mixing_ratio, molmass, gravity):
        """Compute opacity profile from CKD cross section tensor.

        Args:
            xstensor_ckd (3D array): CKD cross section tensor (Nlayer, Ng, Nbands)
            mixing_ratio (1D array): mass mixing ratio (Nlayer,)
            molmass (float): molecular mass
            gravity (float/1D profile): gravity in cgs

        Returns:
            3D array: CKD optical depth tensor (Nlayer, Ng, Nbands)
        """
        return layer_optical_depth_ckd(
            self.dParr, jnp.abs(xstensor_ckd), mixing_ratio, molmass, gravity
        )
    
    

    def opacity_profile_cloud_lognormal(
        self,
        extinction_coefficient,
        condensate_substance_density,
        mmr_condensate,
        rg,
        sigmag,
        gravity,
    ):
        """
        opacity profile (delta tau) from extinction coefficient assuming the AM cloud model with a lognormal cloud distribution
        Args:
            extinction coefficient: extinction coefficient  in cgs (cm-1) [N_layer, N_nus]
            condensate_substance_density: condensate substance density (g/cm3)
            mmr_condensate: Mass mixing ratio (array) of condensate [Nlayer]
            rg: rg parameter in the lognormal distribution of condensate size, defined by (9) in AM01
            sigmag:sigmag parameter (geometric standard deviation) in the lognormal distribution of condensate size, defined by (9) in AM01, must be sigmag > 1
            gravity: gravity (cm/s2)

        Returns:
            2D array: optical depth matrix, dtau  [N_layer, N_nus]

        """

        return layer_optical_depth_clouds_lognormal(
            self.dParr,
            extinction_coefficient,
            condensate_substance_density,
            mmr_condensate,
            rg,
            sigmag,
            gravity,
        )

    def opacity_profile_cia(
        self, logacia_matrix, temperature, vmr1, vmr2, mmw, gravity
    ):
        """opacity profile (delta tau) from collision-induced absorption

        Args:
            logacia_matrix (_type_): _description_
            temperature (_type_): _description_
            vmr1 (_type_): _description_
            vmr2 (_type_): _description_
            mmw (_type_): _description_
            gravity (_type_): _description_

        Returns:
            _type_: _description_
        """
        if jnp.ndim(mmw) == 1:
            mmw = mmw[:, None]
        if jnp.ndim(gravity) == 1:
            gravity = gravity[:, None]

        narr = number_density(self.pressure, temperature)
        lognarr1 = jnp.log10(vmr1 * narr)  # log number density
        lognarr2 = jnp.log10(vmr2 * narr)  # log number density
        logg = jnp.log10(gravity)
        ddParr = self.dParr / self.pressure
        return (
            10
            ** (
                logacia_matrix
                + lognarr1[:, None]
                + lognarr2[:, None]
                + logkB
                - logg
                - logm_ucgs
            )
            * temperature[:, None]
            / mmw
            * ddParr[:, None]
        )

    def check_pressure(self):
        if self.pressure_btm < self.pressure_top:
            raise ValueError(
                "Pressure at bottom should be higher than that at top atmosphere."
            )
        if type(self.nlayer) is not int:
            raise ValueError("Number of the layer should be integer")

    def init_pressure_profile(self):
        pressure_boundary_specification = getattr(
            self, "_pressure_boundary_specification", None
        )
        if pressure_boundary_specification is not None:
            (
                self.pressure,
                self.dParr,
                self.pressure_decrease_rate,
                self.pressure_boundary,
            ) = self._pressure_profile_from_exact_boundaries(
                pressure_top_boundary=pressure_boundary_specification[0],
                pressure_btm_boundary=pressure_boundary_specification[1],
                nlayer=self.nlayer,
                reference_point=self.reference_point,
            )
            self.pressure_top = self.pressure[0]
            self.pressure_btm = self.pressure[-1]
            self.log_pressure_top = np.log10(self.pressure_top)
            self.log_pressure_btm = np.log10(self.pressure_btm)
            return

        from exojax.atm.atmprof import (
            pressure_boundary_logspace,
            pressure_layer_logspace,
        )

        (
            self.pressure,
            self.dParr,
            self.pressure_decrease_rate,
        ) = pressure_layer_logspace(
            log_pressure_top=self.log_pressure_top,
            log_pressure_btm=self.log_pressure_btm,
            nlayer=self.nlayer,
            mode="ascending",
            reference_point=self.reference_point,
            numpy=True,
        )
        self.pressure_boundary = pressure_boundary_logspace(
            self.pressure,
            self.pressure_decrease_rate,
            reference_point=self.reference_point,
            numpy=True,
        )
        self.dParr = np.diff(self.pressure_boundary)

    def change_temperature_range(self, Tlow, Thigh):
        """temperature range to be assumed.

        Note:
            The default temperature range is self.Tlow = 0 K, self.Thigh = jnp.inf.

        Args:
            Tlow (float): lower temperature
            Thigh (float): higher temperature
        """
        self.Tlow = Tlow
        self.Thigh = Thigh

    def clip_temperature(self, temperature):
        """temperature clipping

        Args:
            temperature (array): temperature profile

        Returns:
            array: temperature profile clipped in the range of (self.Tlow-self.Thigh)
        """
        return jnp.clip(temperature, self.Tlow, self.Thigh)

    def powerlaw_temperature(self, T0, alpha):
        """powerlaw temperature profile

        Args:
            T0 (float): T at P=1 bar in K
            alpha (float): powerlaw index

        Returns:
            array: temperature profile
        """
        return self.clip_temperature(atmprof_powerlow(self.pressure, T0, alpha))

    def gray_temperature(self, gravity, kappa, Tint):
        """gray temperature profile

        Args:
            gravity: gravity (cm/s2)
            kappa: infrared opacity
            Tint: temperature equivalence of the intrinsic energy flow in K

        Returns:
            array: temperature profile

        """
        return self.clip_temperature(atmprof_gray(self.pressure, gravity, kappa, Tint))

    def guillot_temperature(self, gravity, kappa, gamma, Tint, Tirr):
        """Guillot tempearture profile

        Notes:
            Set self.fguillot (default 0.25) to change the assumption of irradiation.
            self.fguillot = 1. at the substellar point, self.fguillot = 0.5 for a day-side average
            and self.fguillot = 0.25 for an averaging over the whole planetary surface
            See Guillot (2010) Equation (29) for details.

        Args:
            gravity: gravity (cm/s2)
            kappa: thermal/IR opacity (kappa_th in Guillot 2010)
            gamma: ratio of optical and IR opacity (kappa_v/kappa_th), gamma > 1 means thermal inversion
            Tint: temperature equivalence of the intrinsic energy flow in K
            Tirr: temperature equivalence of the irradiation in K

        Returns:
            array: temperature profile

        """
        return self.clip_temperature(
            atmprof_Guillot(
                self.pressure, gravity, kappa, gamma, Tint, Tirr, self.fguillot
            )
        )

    def custom_temperature(self, np_temperature):
        """custom temperature profile from numpy ndarray

        Notes: this function is equivalen to jnp.array(np_temperature), but it is necessary for the compatibility.

        Args:
            np_temperature (numpy nd array): temperature profile

        Returns:
            array: jnp.array temperature profile
        """
        return jnp.array(np_temperature)

    def powerlaw_temperature_boundary(self, T0, alpha):
        """powerlaw temperature at the upper point (overline{T}) + TB profile

        Args:
            T0 (float): T at P=1 bar in K
            alpha (float): powerlaw index

        Returns:
            array: layer boundary temperature profile (Nlayer + 1)
        """
        return self.clip_temperature(
            atmprof_powerlow(self.pressure_boundary, T0, alpha)
        )

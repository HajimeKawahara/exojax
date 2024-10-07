""" Atmospheric MicroPhysics (amp) class 
"""

from exojax.atm.amclouds import get_rg
from exojax.atm.amclouds import find_rw
from exojax.atm.amclouds import smooth_index_base_pressure
from exojax.atm.amclouds import get_pressure_at_cloud_base
from exojax.atm.amclouds import get_value_at_smooth_index
from exojax.atm.amclouds import mixing_ratio_cloud_profile
from exojax.atm.atmprof import pressure_scale_height
from exojax.atm.atmconvert import mmr_to_vmr
from exojax.atm.vterm import terminal_velocity
from exojax.atm.viscosity import calc_vfactor
from exojax.atm.viscosity import eta_Rosner
from exojax.utils.constants import kB, m_u
import warnings
from jax import vmap
import jax.numpy as jnp

__all__ = ["AmpAmcloud"]


class AmpCloud:
    """Common Amp cloud model class"""

    def __init__(self):
        self.cloudmodel = None  # cloud model
        self.bkgatm = None

    def check_temperature_range(self, temperatures):
        _, vfactor_temperature = calc_vfactor(atm=self.bkgatm)
        vfactor_temperature = jnp.array(vfactor_temperature)
        mint = jnp.min(temperatures)
        maxt = jnp.max(temperatures)
        vmint = jnp.min(vfactor_temperature)
        vmaxt = jnp.max(vfactor_temperature)

        if mint < vmint:
            self._issue_warning(mint, vmint, "min", "smaller")

        if maxt > vmaxt:
            self._issue_warning(maxt, vmaxt, "max", "larger")

    def dynamic_viscosity(self, temperatures):
        vfactor, _ = calc_vfactor(atm=self.bkgatm)
        return eta_Rosner(temperatures, vfactor)

    def set_condensates_scale_array(self, size_min=1.0e-5, size_max=1.0e-3, nsize=1000):
        logsize_min = jnp.log10(size_min)
        logsize_max = jnp.log10(size_max)
        self.rcond_arr = jnp.logspace(logsize_min, logsize_max, nsize)  # cm

    def _issue_warning(self, temperature, limit, temp_type, comparison):
        warnings.warn(
            f"{temp_type} temperature {temperature} K is {comparison} than {temp_type}(vfactor t range) {limit} K"
        )


class AmpAmcloud(AmpCloud):
    def __init__(self, pdb, bkgatm, size_min=1.0e-5, size_max=1.0e-3, nsize=1000):
        """initialization of amp for Ackerman and Marley 2001 cloud model

        Args:
            pdb (pdb class): particulates database (pdb)
            bkgatm: background atmosphere, such as H2, Air
        """
        self.cloudmodel = "Ackerman and Marley (2001)"
        self.pdb = pdb
        self.bkgatm = bkgatm

        self.set_condensates_scale_array(size_min, size_max, nsize)

    def calc_ammodel(
        self,
        pressures,
        temperatures,
        mean_molecular_weight,
        molecular_mass_condensate,
        gravity,
        fsed,
        sigmag,
        Kzz,
        MMR_base,
        alphav=2.0,
    ):
        """computes rg and VMR of condensates based on AM01

        Args:
            pressures (array): Pressure profile of the atmosphere (bar)
            temperatures (array): Temperature profile of the atmosphere (K)
            mean_molecular_weight (float): Mean molecular weight of the atmosphere
            molecular_mass_condensate (float): Molecular mass of the condensate
            gravity (float): Gravitational acceleration (cm/s^2)
            fsed (float): Sedimentation efficiency factor
            sigmag (float): Width of the lognormal size distribution
            Kzz (array): Eddy diffusion coefficient profile (cm^2/s)
            MMR_base (float): Mass Mixing Ratio of condensate at the cloud base
            alphav (float, optional): Shape parameter for the lognormal distribution. Defaults to 2.0.

        Returns:
            rg (array): Parameter in the lognormal distribution of condensate size, defined by (9) in AM01
            MMR_condensate (array): Mass Mixing Ratio (MMR) of condensates
        """
        rw, MMR_condensate = self.calc_ammodel_rw(
            pressures,
            temperatures,
            mean_molecular_weight,
            molecular_mass_condensate,
            gravity,
            fsed,
            Kzz,
            MMR_base,
        )
        rg = get_rg(rw, fsed, alphav, sigmag)
        return rg, MMR_condensate  # , self.pdb.rhoc, self.molmass_c

    def calc_ammodel_rw(
        self,
        pressures,
        temperatures,
        mean_molecular_weight,
        molecular_mass_condensate,
        gravity,
        fsed,
        Kzz,
        MMR_base,
    ):
        """computes rw and VMR of condensates based on AM01 without sigmag

        Args:
            pressures (array): Pressure profile of the atmosphere (bar)
            temperatures (array): Temperature profile of the atmosphere (K)
            mean_molecular_weight (float): Mean molecular weight of the atmosphere
            molecular_mass_condensate (float): Molecular mass of the condensate
            gravity (float): Gravitational acceleration (cm/s^2)
            fsed (float): Sedimentation efficiency factor
            Kzz (array): Eddy diffusion coefficient profile (cm^2/s)
            MMR_base (float): Mass Mixing Ratio of condensate at the cloud base

        Returns:
            rw (array): Parameter in the lognormal distribution of condensate size, defined by (9) in AM01
            MMR_condensate (array): Mass Mixing Ratio (MMR) of condensates
        """
        # density difference
        rho = mean_molecular_weight * m_u * pressures / (kB * temperatures)
        drho = self.pdb.condensate_substance_density - rho

        # saturation pressure
        psat = self.pdb.saturation_pressure(temperatures)

        # cloud base pressure/temperature
        VMR = mmr_to_vmr(MMR_base, molecular_mass_condensate, mean_molecular_weight)

        smooth_index = smooth_index_base_pressure(pressures, psat, VMR)
        pressure_base = get_pressure_at_cloud_base(pressures, smooth_index)
        temperature_base = get_value_at_smooth_index(temperatures, smooth_index)

        # cloud scale height L
        L_cloud = pressure_scale_height(
            gravity, temperature_base, mean_molecular_weight
        )

        # viscosity
        eta_dvisc = self.dynamic_viscosity(temperatures)

        # terminal velocity
        vf_vmap = vmap(terminal_velocity, (None, None, 0, 0, 0))
        vterminal = vf_vmap(self.rcond_arr, gravity, eta_dvisc, drho, rho)

        # condensate size
        vfind_rw = vmap(find_rw, (None, 0, None), 0)
        rw = vfind_rw(self.rcond_arr, vterminal, Kzz / L_cloud)
        # MMR of condensates
        MMR_condensate = mixing_ratio_cloud_profile(
            pressures, pressure_base, fsed, MMR_base
        )
        return rw, MMR_condensate

""" Atmospheric MicroPhysics (amp) class 
"""
from exojax.atm.amclouds import get_rg
from exojax.atm.amclouds import find_rw
from exojax.atm.viscosity import calc_vfactor, eta_Rosner
from exojax.atm.amclouds import compute_cloud_base_pressure_index
from exojax.atm.amclouds import mixing_ratio_cloud_profile
from exojax.atm.vterm import terminal_velocity
from exojax.atm.atmprof import pressure_scale_height
from exojax.atm.mixratio import mmr2vmr
from exojax.utils.constants import kB, m_u
import warnings
from jax import vmap
import jax.numpy as jnp

__all__ = ["AmpCldAM"]


class AmpCloud:
    """Common Amp cloud"""

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
            pdb (_type_): particulates database (pdb)
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
        """computes rg and VMR of condenstates based on AM01 

        Args:
            pressures (_type_): _description_
            temperatures (_type_): _description_
            mean_molecular_weight (_type_): _description_
            molecular_mass_condensate: condensate molecular mass
            gravity (_type_): _description_
            fsed (_type_): _description_
            sigmag (_type_): _description_
            Kzz (_type_): _description_
            MMR_base (_type_): Mass Mixing Ratio of condensate at the cloud base
            alphav (float, optional): _description_. Defaults to 2.0.

        Returns:
            rg parameter in the lognormal distribution of condensate size, defined by (9) in AM01
            Mass Mixing Ratio (MMR) of condensates
        """
        # density difference
        rho = mean_molecular_weight * m_u * pressures / (kB * temperatures)
        drho = self.pdb.condensate_substance_density - rho

        # saturation pressure
        psat = self.pdb.saturation_pressure(temperatures)

        # cloud base pressure/temperature
        VMR = mmr2vmr(MMR_base,molecular_mass_condensate, mean_molecular_weight)
        ibase = compute_cloud_base_pressure_index(pressures, psat, VMR)
        pressure_base = pressures[ibase]
        temperature_base = temperatures[ibase]

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
        rg = get_rg(rw, fsed, alphav, sigmag)

        # MMR of condensates
        MMR_condensate = mixing_ratio_cloud_profile(pressures, pressure_base, fsed, MMR_base)

        return rg, MMR_condensate  # , self.pdb.rhoc, self.molmass_c

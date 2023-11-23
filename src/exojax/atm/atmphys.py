""" Atmospheric MicroPhysics (amp) class 
"""
from exojax.atm.amclouds import get_rg
from exojax.atm.amclouds import find_rw
from exojax.atm.viscosity import calc_vfactor, eta_Rosner
from exojax.atm.amclouds import compute_cloud_base_pressure_index
from exojax.atm.amclouds import vmr_cloud_profile
from exojax.atm.vterm import terminal_velocity
from exojax.atm.atmprof import pressure_scale_height
from exojax.utils.constants import kB, m_u

from jax import vmap
import jax.numpy as jnp

__all__ = ['AmpCldAM']


class AmpCloud():
    """Common Amp cloud
    """
    def __init__(self):
        self.cloudmodel = None # cloud model
        self.bkgatm = None

    def set_background_atmosphere(self, temperatures):
        self.vfactor, self.vfactor_trange = calc_vfactor(atm=self.bkgatm)
        self.eta_dvisc = eta_Rosner(temperatures, self.vfactor)

    def set_condensates_scale_array(self, size_min=1.e-5, size_max=1.e-3, nsize=1000):
        logsize_min = jnp.log10(size_min)
        logsize_max = jnp.log10(size_max)
        self.rcond_arr = jnp.logspace(logsize_min, logsize_max, nsize)  #cm


class AmpAmcloud(AmpCloud):
    
    def __init__(self, pdb):
        """initialization of amp for Ackerman and Marley 2001 cloud model

        Args:
            pdb (_type_): particulates database (pdb) 
        """
        self.cloudmodel = "Ackerman and Marley (2001)"
        self.pdb = pdb
        self.bkgatm = pdb.bkgatm

    def calc_ammodel(pressures, temperatures, mean_molecular_weight, gravity, fsed, sigmag, Kzz, VMR, alphav=2.0):

        # density difference 
        rho  = mean_molecular_weight * m_u * pressures / (kB * temperatures)
        drho = self.pdb.rhoc - rho
        

        # saturation pressure
        psat = self.pdb.saturation_pressure(temperatures)
        
        # cloud base pressure/temperature
        ibase = compute_cloud_base_pressure_index(pressures, psat, VMR)
        pressure_base = pressures[ibase]
        temperature_base = temperatures[ibase]

        # cloud scale height L
        L_cloud = pressure_scale_height(gravity, temperature_base, mean_molecular_weight)

        
        #terminal velocity
        vf_vmap = vmap(terminal_velocity, (None, None, 0, 0, 0))
        vterminal = vf_vmap(self.rcond_arr, gravity, self.eta_dvisc, drho, rho)
        vfind_rw = vmap(find_rw, (None, 0, None), 0)

        # condensate size
        rw = vfind_rw(self.rcond_arr, vterminal, Kzz / L_cloud)
        rg = get_rg(rw, fsed, alphav, sigmag)

        # VMR of condensates
        VMRc = vmr_cloud_profile(pressures, pressure_base, fsed, VMR)


        return rg, sigmag, VMRc #, self.pdb.rhoc, self.molmass_c 


"""Continuum database (CDB) class

   * CdbCIA is the CDB for CIA
   
"""
import numpy as np
import jax.numpy as jnp
import pathlib
from exojax.spec.hitrancia import read_cia, logacia

__all__ = ['CdbCIA']

class CdbCIA(object):
    def __init__(self,path,nurange=[-np.inf,np.inf],margin=10.0):
        """Continuum database for HITRAN CIA

        Args: 
           path: path for HITRAN cia file
           nurange: wavenumber range list (cm-1) or wavenumber array
           margin: margin for nurange (cm-1)

        """        
        #downloading
        self.nurange=[np.min(nurange),np.max(nurange)]
        self.margin = margin
        self.path = pathlib.Path(path)
        molec=str(self.path.stem)
        if not self.path.exists():
            self.download()
        self.nucia,self.tcia,ac=read_cia(path,self.nurange[0]-self.margin,self.nurange[1]+self.margin)
        #--------------------------------
        self.logac=jnp.array(np.log10(ac))
        self.tcia=jnp.array(self.tcia)
        self.nucia=jnp.array(self.nucia)
        
    def download(self):
        """Downloading HITRAN cia file

        Note:
           The download URL is written in exojax.utils.url.

        """
        import urllib.request
        import os
        from exojax.utils.url import url_HITRANCIA

        try:
            os.makedirs(str(self.path.stem), exist_ok=True)
            url = url_HITRANCIA()+self.path.name
            urllib.request.urlretrieve(url,str(self.path))
        except:
            print(url)
            print("HITRAN download failed")

            
CONST_K, CONST_C, CONST_H = 1.380649e-16, 29979245800.0, 6.62607015e-27 # cgs

def log_hminus_continuum(wavelength_um, temperature, number_density_e, number_density_h):
    """
    John (1988) H- continuum opacity
    
    Args:
       wavelength_um: wavelength in units of microns
       temperature: gas temperature [K]
       number_density_e: electron number density
       number_density_h: H atom number density
    
    Returns: 
       log10(absorption coefficient)
    """
    # first, compute the cross sections (in cm4/dyne)
    kappa_bf = bound_free_absorption(wavelength_um, temperature) 
    kappa_ff = free_free_absorption(wavelength_um, temperature) 

    electron_pressure = number_density_e * CONST_K * temperature  #//electron pressure in dyne/cm2
    hydrogen_density = number_density_h   #//hydrogen number density in cm-3

    # and now finally the absorption_coeff (in cm-1)
    absorption_coeff = (kappa_bf + kappa_ff) * electron_pressure * hydrogen_density 

    return jnp.log10(absorption_coeff)



def bound_free_absorption(wavelength_um, temperature):

    # Note: alpha has a value of 1.439e4 micron-1 K-1, the value stated in John (1988) is wrong
    # here, we express alpha using physical constants 
    alpha = CONST_C*CONST_H/CONST_K*10000.0  
    lambda_0 = 1.6419  # photo-detachment threshold

    #   //tabulated constant from John (1988)
    def f(wavelength_um):
        C_n = jnp.vstack(
            [jnp.arange(7), [0.0, 152.519, 49.534, -118.858, 92.536, -34.194, 4.982]]
        ).T

        def body_fun(val, x):
            i, C_n_i = x
            return val, val + C_n_i * jnp.power(jnp.clip(1.0/wavelength_um - 1.0/lambda_0, a_min=0, a_max=None), (i-1)/2.0)

        return lax.scan(body_fun, jnp.zeros_like(wavelength_um), C_n)[-1].sum(0)

    # first, we calculate the photo-detachment cross-section (in cm2)
    kappa_bf = (1e-18 * wavelength_um ** 3 * 
        jnp.power(jnp.clip(1.0/wavelength_um - 1.0/lambda_0, a_min=0, a_max=None), 1.5) * f(wavelength_um)
    )

    kappa_bf = jnp.where(
        (wavelength_um <= lambda_0) & (wavelength_um > 0.125),
        (0.750 * jnp.power(temperature, -2.5) * jnp.exp(alpha / lambda_0 / temperature) * 
         (1.0 - jnp.exp( -alpha / wavelength_um / temperature)) * kappa_bf), 
         0
    )
    return kappa_bf


def free_free_absorption(wavelength_um, temperature):
    # coefficients from John (1988)
    # to follow his notation (which starts at an index of 1), the 0-index components are 0
    # for wavelengths larger than 0.3645 micron
    A_n1 = jnp.array([0.0, 0.0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830])
    B_n1 = jnp.array([0.0, 0.0, 285.8270, -1158.3820, 2427.7190, -1841.4000, 444.5170])
    C_n1 = jnp.array([0.0, 0.0, -2054.2910, 8746.5230, -13651.1050, 8624.9700, -1863.8650])
    D_n1 = jnp.array([0.0, 0.0, 2827.7760, -11485.6320, 16755.5240, -10051.5300, 2095.2880])
    E_n1 = jnp.array([0.0, 0.0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880])
    F_n1 = jnp.array([0.0, 0.0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850])

    # for wavelengths between 0.1823 micron and 0.3645 micron
    A_n2 = jnp.array([0.0, 518.1021, 473.2636, -482.2089, 115.5291, 0.0, 0.0])
    B_n2 = jnp.array([0.0, -734.8666, 1443.4137, -737.1616, 169.6374, 0.0, 0.0])
    C_n2 = jnp.array([0.0, 1021.1775, -1977.3395, 1096.8827, -245.6490, 0.0, 0.0])
    D_n2 = jnp.array([0.0, -479.0721, 922.3575, -521.1341, 114.2430, 0.0, 0.0])
    E_n2 = jnp.array([0.0, 93.1373, -178.9275, 101.7963, -21.9972, 0.0, 0.0])
    F_n2 = jnp.array([0.0, -6.4285, 12.3600, -7.0571, 1.5097, 0.0, 0.0])

    coeffs1 = jnp.vstack([
        jnp.arange(7), A_n1, B_n1, C_n1, D_n1, E_n1, F_n1
    ]).T

    coeffs2 = jnp.vstack([
        jnp.arange(7), A_n2, B_n2, C_n2, D_n2, E_n2, F_n2
    ]).T

    def body_fun(val, x):
        i, A_n_i, B_n_i, C_n_i, D_n_i, E_n_i, F_n_i = x
        return val, val + (jnp.power(5040.0/temperature, (i+1)/2.0) * 
                  (wavelength_um**2 * A_n_i + B_n_i + C_n_i/wavelength_um + D_n_i/wavelength_um**2  + 
                   E_n_i/wavelength_um**3 + F_n_i/wavelength_um**4))

    kappa_ff = jnp.where(
        wavelength_um > 0.3645, 
        lax.scan(body_fun, jnp.zeros_like(wavelength_um), coeffs)[-1].sum(0) * 1e-29, 
        0
    ) + jnp.where(
        (wavelength_um >= 0.1823) & (wavelength_um <= 0.3645),
        lax.scan(body_fun, jnp.zeros_like(wavelength_um), coeffs2)[-1].sum(0) * 1e-29 ,
        0
    )

    return kappa_ff 
            

if __name__ == "__main__":
    ciaH2H2=CdbCIA("/home/kawahara/exojax/data/CIA/H2-H2_2011.cia",nurange=[4050.0,4150.0])
    print(ciaH2H2.tcia)

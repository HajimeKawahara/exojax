#!/usr/bin/env python
# coding: utf-8

# # Foward Modeling of Jupiter-like Clouds and Reflection Spectrum 

# In[1]:


import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax.config import config
config.update("jax_enable_x64", True)

# Sets an atmosphere model

# In[2]:


from exojax.utils.constants import kB, m_u
from exojax.atm.atmprof import pressure_layer_logspace
nlayer = 200
Parr, dParr, k = pressure_layer_logspace(log_pressure_top=-5., log_pressure_btm=2.0, nlayer=nlayer)
alpha = 0.097
T0 = 200.
Tarr = T0 * (Parr)**alpha

mu = 2.0  # mean molecular weight
R = kB / (mu * m_u)
rho = Parr / (R * Tarr)

g=1.e5


# `pdb` is a class for particulates databases. We here use `PdbCloud` for NH3, i.e. `pdb` for the ammonia cloud. 
# PdbCloud uses the refaction (refractive) indice given by VIRGA. The Mie parameters assuming a log-normal distribution is called `miegrid`. This can be computed pdb.generate_miegrid if you do not have it. To compute `miegrid`, we use PyMieScatt as a calculator.   
# 
# Also, `amp` is a class for atmospheric micorphysics. AmpAmcloud is the class for the Akerman and Marley 2001 cloud model (AM01). We adopt the background atmosphere to hydorogen atmosphere.

# In[3]:


from exojax.spec.pardb import PdbCloud
from exojax.atm.atmphys import AmpAmcloud

pdb_nh3 = PdbCloud("NH3")
pdb_nh3.load_miegrid()

amp_nh3 = AmpAmcloud(pdb_nh3,bkgatm="H2")
amp_nh3.check_temperature_range(Tarr)


# Sets the parameters in the AM01 cloud model. `calc_ammodel` method computes the vertical distribution of `rg` and the condensate volume mixing ratio.

# In[4]:


fsed = 10.
sigmag = 2.0
Kzz = 1.e4
VMR = 0.001
rg_layer, VMRc = amp_nh3.calc_ammodel(Parr, Tarr, mu, g, fsed=fsed, sigmag=sigmag, Kzz=Kzz, VMR=VMR)


# In[5]:




# `rg` is almost constant through the vertical distribution. So, let's set a mean value here. `miegrid_interpolated_value` method interpolates the original parameter set given by MiQ_lognormal in PyMieScatt. See https://pymiescatt.readthedocs.io/en/latest/forward.html#Mie_Lognormal. The number of the original parameters are seven, Bext, Bsca, Babs, G, Bpr, Bback, and Bratio. 

# In[6]:


rg = 8e-4
mieQpar = pdb_nh3.miegrid_interpolated_values(rg, sigmag)
#beta0, omega, g = pdb_nh3.(rg,sigmag)


# Plots the extinction coefficient for instance (index=0) and some approximation from the Kevin Heng's textbook .

# In[7]:


# approximate model (i.e. Heng's textbook)
#x = 2*np.pi*(rg*np.exp(np.log(sigmag)**2/2.0))/(pdb_nh3.refraction_index_wavelength_nm*1.e-7)
#Q0 = 10.
#qe = 5/(Q0*x**-4 + x**0.2)*0.1


# To handle the opacity for Mie scattering, we call `OpaMie` class as `opa`.  
# The mie parameters can be derived by `mieparams_vector` method, which returns $\beta_0$: the extinction coefficient of the reference number density $N_0$, $\omega$: a single scattering albedo , and $g$: the asymmetric parameter.
# The extinction coefficient given the number density $N$ can be computed by $\beta = \beta_0 N/N_0$.

# In[8]:


from exojax.utils.grids import wavenumber_grid
N = 10000
nus, wav, res = wavenumber_grid(10**3, 10**4, N, xsmode="premodit")

from exojax.spec.opacont import OpaMie
opa = OpaMie(pdb_nh3, nus)
beta0, betasct, g = opa.mieparams_vector(rg,sigmag)


# In[9]:


# plt.plot(pdb_nh3.refraction_index_wavenumber, miepar[50,:,0])

# ## Emission spectrum

# ### pure absorption
# 
# First assume that you'd like to use the emission model with pure absorption, i.e., only the cross section

# In[51]:


from exojax.spec.atmrt import ArtEmisPure
art = ArtEmisPure(nu_grid=nus, pressure_btm=1.e2, pressure_top=1.e-5, nlayer=nlayer, rtsolver="ibased", nstream=8) #intesity-based 8stream
#art = ArtEmisPure(nu_grid=nus, pressure_btm=1.e2, pressure_top=1.e-5, nlayer=nlayer, rtsolver="fbased2st") #flux-based

art.change_temperature_range(500.0, 1500.0)
Tarr = art.powerlaw_temperature(1200.0,0.1)
mmr_profile = art.constant_mmr_profile(0.01)

from exojax.utils.astrofunc import gravity_jupiter
gravity = gravity_jupiter(1.0,10.0)


# In[52]:


mean_molecular_weight = 2.0
xsmatrix = beta0[None,:] + np.zeros((len(art.pressure), len(nus)))
dtau = art.opacity_profile_xs(xsmatrix, VMRc, mean_molecular_weight, gravity)


# In[53]:


np.shape(xsmatrix), np.shape(VMRc)


# In[54]:


from exojax.plot.atmplot import plotcf

cf = plotcf(
    nus, dtau, Tarr, art.pressure, art.dParr, optarr=VMRc, leftxlabel="log10 VMR (condensate)", leftxlog=True
)


# In[55]:


F = art.run(dtau, Tarr, nu_grid=nus)
    


# ### with Scattering

# In[56]:


from exojax.spec.atmrt import ArtEmisScat
arts = ArtEmisScat(nu_grid=nus, pressure_btm=1.e2, pressure_top=1.e-5, nlayer=nlayer)
arts.change_temperature_range(500.0, 1400.0)
Tarr = arts.powerlaw_temperature(1200.0,0.1)
mmr_profile = arts.constant_mmr_profile(0.01)

artsl = ArtEmisScat(nu_grid=nus, pressure_btm=1.e2, pressure_top=1.e-5, nlayer=nlayer, rtsolver="lart_toon_hemispheric_mean")
artsl.change_temperature_range(500.0, 1400.0)



# In[49]:


#dtau = arts.opacity_profile_xs(xsmatrix, VMRc, mean_molecular_weight, gravity)

single_scattering_albedo = betasct[None,:]/beta0[None,:] + np.zeros((len(art.pressure), len(nus)))
asymmetric_parameter = g + np.zeros((len(art.pressure), len(nus)))
reflectivity_surface = np.zeros(len(nus))
Fs = arts.run(dtau,np.real(single_scattering_albedo),np.real(asymmetric_parameter),Tarr)
Fsl, Fsl2, Fslpure = artsl.run(dtau,np.real(single_scattering_albedo),np.real(asymmetric_parameter),Tarr,show=True)
    

#incoming_flux = np.zeros(len(nus))

# In[58]:


plt.plot(nus,F,label="no scattering")
plt.plot(nus,Fs,label="with scattering (fluxadd)")
plt.plot(nus,Fsl,label="with scattering (LART)")
plt.plot(nus,Fsl2,label="with scattering (LART, numpy)")
plt.plot(nus,Fslpure,label="with scattering (LART, pure)")

plt.legend()

plt.xscale("log")
#plt.yscale("log")
plt.ylabel("flux")
plt.xlabel("wavenumber (cm-1)")
plt.show()

# ## Reflection spectrum

# In[16]:


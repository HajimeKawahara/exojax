import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random
from jax import jit
import numpyro.distributions as dist
import numpyro
from exojax.spec import rtransfer as rt

NP=100
Parr, dParr, k=rt.pressure_layer(NP=NP)
#reference pressure for a T-P model                                             
Pref=1.0 #bar
ONEARR=np.ones_like(Parr)
smalla=1.0
smalldiag=smalla**2*jnp.identity(NP)

@jit
def modelcov(t,tau,a):
    fac=1.e-5
    Dt = t - jnp.array([t]).T
    amp=a
    K=amp*jnp.exp(-(Dt)**2/2/(tau**2))+amp*fac*jnp.identity(NP)
    return K


# In[16]:


ZEROARR=jnp.zeros_like(Parr)
ONEARR=jnp.ones_like(Parr)

lnParr=jnp.log10(Parr)


# In[17]:

def comp_Tarr(okey):
    okey,key=random.split(okey)
#    lnsT = numpyro.sample('lnsT', dist.Uniform(3.0,5.0),rng_key=key)
    lnsT=6.0
    sT=10**lnsT
    okey,key=random.split(okey)
#    lntaup =  numpyro.sample('lntaup', dist.Uniform(0,1),rng_key=key)
    lntaup=0.5
    taup=10**lntaup    
    cov=modelcov(lnParr,taup,sT)

    okey,key=random.split(okey)
    T0 =  numpyro.sample('T0', dist.Uniform(1000,2000),rng_key=key)
    okey,key=random.split(okey)
    Tarr=numpyro.sample("Tarr", dist.MultivariateNormal(loc=ONEARR, covariance_matrix=cov),rng_key=key)+T0

    #lnT0=3.0 #1000K
    #lnTarr=numpyro.sample("Tarr", dist.MultivariateNormal(loc=lnT0*ONEARR, covariance_matrix=cov),rng_key=key)
    #Tarr=10**lnTarr
    return Tarr

fig=plt.figure(figsize=(5,7))
okey=random.PRNGKey(20)
for i in range(0,100):
    okey,key=random.split(okey)
    Tarr=comp_Tarr(key)
    plt.plot(Tarr,Parr,alpha=0.2,color="green",rasterized=True)
plt.plot(1295*Parr**0.099,Parr,color="black",lw=1)

plt.yscale("log")
plt.xlabel("temperature (K)",fontsize=17)
plt.ylabel("pressure (bar)",fontsize=17)
plt.xlim(0,3200)
plt.ylim(Parr[0],Parr[-1])

plt.tick_params(labelsize=17)
plt.gca().invert_yaxis()
plt.savefig("prarr.png")
plt.savefig("prarr.pdf", bbox_inches="tight", pad_inches=0.0)
plt.show()
    

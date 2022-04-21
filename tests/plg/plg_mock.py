"""Pseudo line generator using mock line databases

"""


import numpy as np
from exojax.spec.initspec import init_modit
import time
import matplotlib.pyplot as plt
from exojax.spec import plg

np.random.seed(1)
Nline=10000001
Ngamma=10

logsij0=np.random.rand(Nline)
elower=(np.random.rand(Nline))**0.2*5000.+2000.0
nu_lines=np.random.rand(Nline)*10.0
index_gamma=np.random.randint(Ngamma,size=Nline)

#init modit
Nnus=1001
nus=np.linspace(0,10,Nnus)
cnu,indexnu,R,pmarray=init_modit(nu_lines,nus)

#elower setting
Ncrit=3
Nelower=11
    
ts=time.time()
qlogsij0,qcnu,num_unique,elower_grid,frozen_mask,nonzeropl_mask=plg.plg_elower_addcon(index_gamma,Ngamma,cnu,indexnu,nus,logsij0,elower,Ncrit=Ncrit,Nelower=Nelower,reshape=True)    
te=time.time()
print(te-ts,"sec")
num_unique=np.array(num_unique,dtype=float)
num_unique[num_unique<Ncrit]=None

fig=plt.figure(figsize=(10,3))
ax=fig.add_subplot(211)
c=plt.imshow(num_unique[5,:,:].T)
plt.colorbar(c,shrink=0.2)
ax.set_aspect(0.1/ax.get_data_ratio())

ax=fig.add_subplot(212)
c=plt.imshow(qlogsij0[5,:,:].T)
plt.colorbar(c,shrink=0.2)
ax.set_aspect(0.1/ax.get_data_ratio())
plt.show()

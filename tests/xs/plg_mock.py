import numpy as np
from exojax.spec.initspec import init_modit
import time
import matplotlib.pyplot as plt
from exojax.spec import plg

np.random.seed(1)
Nline=10000001
Ngamma=20

logsij0=np.random.rand(Nline)
elower=(np.random.rand(Nline))**0.2*5000.+2000.0
nus=np.random.rand(Nline)*10.0
index_gamma=np.random.randint(Ngamma,size=Nline)

#init modit
Nnus=101
nu_grid=np.linspace(0,10,Nnus)
cnu,indexnu,R,pmarray=init_modit(nus,nu_grid)

#elower setting
Ncrit=30
Nelower=11
    
ts=time.time()
qlogsij0,qcnu,num_unique,elower_grid=plg.plg_elower_addcon(index_gamma,Ngamma,cnu,indexnu,Nnus,logsij0,elower,Ncrit=Ncrit,Nelower=Nelower)
te=time.time()
print(te-ts,"sec")
num_unique=np.array(num_unique,dtype=float)
num_unique[num_unique<Ncrit]=None

fig=plt.figure(figsize=(10,3))
ax=fig.add_subplot(211)
c=plt.imshow(num_unique[0,:,:].T)
plt.colorbar(c,shrink=0.2)
ax=fig.add_subplot(212)

c=plt.imshow(qlogsij0[0,:,:].T)
plt.colorbar(c,shrink=0.2)
plt.show()

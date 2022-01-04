import numpy as np
import matplotlib.pyplot as plt
from exojax.dynamics.getE import getE
#make example by PyAstronomy
from PyAstronomy.pyasl.asl import MarkleyKESolver as MKE

m=MKE()
print(m.getE(M=0.5,e=0.3))
marr=np.linspace(0.0,4*np.pi,1000)
ea=[]
for meach in marr:
    ea.append(m.getE(M=meach,e=0.3))
ea=np.array(ea)

fig=plt.figure()
ax=fig.add_subplot(211)
plt.plot(marr,ea)
plt.plot(marr,getE(marr,0.3))

ax.set_ylabel("Eccentric anomary")

ax=fig.add_subplot(212)
plt.plot(marr,ea-getE(marr,0.3))

plt.xlabel("Mean anomary")
ax.set_ylabel("Difference")
plt.savefig("getE.png")
plt.show()

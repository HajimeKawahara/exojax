import matplotlib.pyplot as plt
import numpy as np


nus_lpf,mu_lpf=np.load("clpf.npz",allow_pickle=True)["arr_0"]
nus_modit,mu_modit=np.load("cmodit4500.npz",allow_pickle=True)["arr_0"]

fig=plt.figure(figsize=(8,4))
plt.plot(nus_modit,mu_modit,label="MODIT",color="C1")
plt.plot(nus_lpf,mu_lpf,label="DIRECT",ls="dashed",color="C0")

plt.xlabel("wavenumber (cm-1)")
plt.ylabel("spectrum")
plt.legend()
plt.savefig("compspec_luhman16A.png")
plt.show()

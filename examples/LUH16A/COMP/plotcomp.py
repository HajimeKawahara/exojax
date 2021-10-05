import matplotlib.pyplot as plt
import numpy as np

nus_lpf,mu_lpf=np.load("clpf.npz",allow_pickle=True)["arr_0"]
nus_modit,mu_modit=np.load("cmodit.npz",allow_pickle=True)["arr_0"]

print(nus_modit)
print(mu_modit)
plt.plot(nus_lpf,mu_lpf,label="DIRECT")
plt.plot(nus_modit,mu_modit,label="MODIT")
plt.legend()

plt.show()

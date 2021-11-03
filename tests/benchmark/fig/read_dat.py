import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dat=pd.read_csv("data/gpu.dat",delimiter=",")
n=dat["N"].values
t=dat["t_s"].values
std=dat["std_s"].values
plt.errorbar(n,t,yerr=std,color="C0",fmt="*",label="A:$\\Delta \\nu = 100 \\mathrm{cm^{-1}}, N_\\nu=10^4$")
plt.plot(n,t,color="C0",alpha=0.3)


dat=pd.read_csv("data/gpu2.dat",delimiter=",")
n=dat["N"].values
t=dat["t_s"].values
std=dat["std_s"].values
plt.errorbar(n,t,yerr=std,color="C1",fmt="*",label="B:$\\Delta \\nu = 1000 \\mathrm{cm^{-1}}, N_\\nu=10^5$")
plt.plot(n,t,color="C1",alpha=0.3)

dat=pd.read_csv("data/each.dat",delimiter=",")
n=dat["N"].values
t=dat["t_us"].values*1.e-6
std=dat["std_us"].values*1.e-6
#plt.errorbar(n,t,yerr=std,color="C0",fmt=".",label="$\\Delta \\nu = 100 \\mathrm{cm^{-1}}, N_\\nu=10^4$")
plt.errorbar(n,t,yerr=std,color="C0",fmt=".",label="A w/ transfer")
plt.plot(n,t,color="C0",alpha=0.15,ls="dashed")

dat=pd.read_csv("data/each2.dat",delimiter=",")
n=dat["N"].values
t=dat["t_ms"].values*1.e-3
std=dat["std_ms"].values*1.e-3
#plt.errorbar(n,t,yerr=std,color="C1",fmt=".",label="$\\Delta \\nu = 1000 \\mathrm{cm^{-1}}, N_\\nu=10^5$")
plt.errorbar(n,t,yerr=std,color="C1",fmt=".",label="B w/ transfer")
plt.plot(n,t,color="C1",alpha=0.15,ls="dashed")



xarr=np.logspace(2,5)
plt.plot(xarr,xarr*1.4e-6,alpha=0.3,ls="dotted",color="gray")
plt.text(10000,0.01,"$t \propto N_\mathrm{line}$",rotation=35,color="gray")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("$N_\mathrm{line}$",fontsize=14)
plt.ylabel("Time (sec)",fontsize=14)
#plt.axhline(0.2)
plt.legend(loc="lower right")
plt.tick_params(labelsize=14)
plt.savefig("bklpf.pdf", bbox_inches="tight", pad_inches=0.0)
plt.show()

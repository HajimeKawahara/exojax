import numpy as np
import tqdm
from exojax.spec.lbderror import single_tilde_line_strength_zeroth
from exojax.spec.lbderror import worst_tilde_line_strength_first
from exojax.spec.lbderror import worst_tilde_line_strength_second

N=30
Twtarr = np.logspace(np.log10(100.1),np.log10(2000.1),N)
Trefarr = np.logspace(np.log10(100.1),np.log10(2000.1),N)
M=100
Tarr = np.logspace(np.log10(100.), np.log10(3000.), M)
L=15
dEarr = np.linspace(100,1500,L)

dE_0th=500.
dE_1st=1000.
dE_2nd=1500.

arr=np.zeros((M,N,N,L,3))

for idE, dE in enumerate(dEarr):
    print(idE,"/",L)
    for iTref, Tref in enumerate(Trefarr):
        for iTwt, Twt in enumerate(Twtarr):
            x = single_tilde_line_strength_zeroth(1./Tarr, 1.0/Twt, 1.0/Tref, dE_0th)
            arr[:,iTwt,iTref,idE,0]=x
            x = worst_tilde_line_strength_first(Tarr, Twt, Tref, dE_1st)
            arr[:,iTwt,iTref,idE,1]=x
            x = worst_tilde_line_strength_second(Tarr, Twt, Tref, dE_2nd)
            arr[:,iTwt,iTref,idE,2]=x

np.savez("elower_grid_arr.npz",arr,Tarr,Twtarr,Trefarr,dEarr)
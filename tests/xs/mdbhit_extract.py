from exojax.spec import xsection
from exojax.spec.hitran import SijT, doppler_sigma, gamma_hitran, gamma_natural
from exojax.spec.exomol import gamma_exomol
from exojax.spec import moldb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
# Setting wavenumber bins and loading HITEMP database
wav=np.linspace(16370.0,16390.0,2000,dtype=np.float64) #AA
nus=1.e8/wav[::-1] #cm-1
ts=time.time()
mdbCO_HITEMP=moldb.MdbHit('/home/kawahara/exojax/data/CH4/06_HITEMP2020.par.bz2',nus,extract=True)
te=time.time()
print(te-ts,"sec")

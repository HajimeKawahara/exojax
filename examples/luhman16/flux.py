"""Checking Luhman 16A flux@2.3um (CIA)

   * our fiducial ref is Burgasser+ 1303.7283
   * d=2.02+-0.019pc


"""

#something wrong
#F_REF1=2.5e-16 #lambda F_lambda erg/s/cm2　
#https://iopscience.iop.org/article/10.1088/0004-637X/790/2/90/pdf

#THIS IS CONSISTENT
Fabs_REF2=2.7e-12 #erg/s/cm2/um Burgasser+ 1303.7283 @2.3um

#APPARENT from magnitude
K=9.44 #http://simbad.u-strasbg.fr/simbad/sim-id?Ident=NAME+WISE+J1049-5319A
Fa=7.14496326075512e-11 #erg/cm2/s/um  from Ks=9.44

from exojax.spec import planck
from exojax.utils.constants import RJ, pc, Rs
import numpy as np
ccgs=29979245800.0
nu0=1.e8/23000.0

Teff=1000.

#APPARENT FLUX
fac=(RJ)**2/((2.02*pc)**2)
pBnu=planck.piBarr(np.array([Teff]),nu0) #erg/cm2/s/(cm)
pBf=pBnu/ccgs #erg/cm2/s/Hz

pBmicron_app=pBnu*1.e4*fac #erg/cm2/s/micron apparent
print("Apparent flux:",pBmicron_app, Fa, "erg/cm2/s/micron") 

#ABSOLUTE FLUX
fac0=(RJ)**2/((10.0*pc)**2)
pBmicron_abs=pBnu*1.e4*fac0 #erg/cm2/s/micron absolute
print("---------------")
print("Absolute flux:",pBmicron_abs, Fabs_REF2, Fa*(2.02/10)**2,"erg/cm2/s/micron" )
print(",from Burgasser+ 1303.7283")

print("---------------")
Fapp=Fabs_REF2*(2.02/10)**-2
print("Apparent flux from Burgasser+ 1303.7283")
print(Fapp,"erg/cm2/s/micron")

print("---------------")
fac0=(RJ)**2/((10.0*pc)**2)
Ftoa=Fabs_REF2/fac0/1.e4
print("TOA flux from Burgasser+ 1303.7283")
print(Ftoa,"erg/cm2/s/cm")
print("---------------")
print("TOA flux from Burgasser+ 1303.7283")
print(Ftoa/ccgs,"erg/cm2/s/Hz")

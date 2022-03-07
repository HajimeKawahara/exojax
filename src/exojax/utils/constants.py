"""constants.

* constants in cgs or km/s
* logm_ucgs=np.log10(m_u*1.e3) where m_u = scipy.constants.m_u.
"""

# cgs
Rs = 6.957*1.e10
RJ = 7.1492*1.e9
RE = 6.3781*1.e8
au = 1.495978707*1.e13
pc = 3.0856775814913673*1.e18
G = 6.67408e-08
MJ = 1.89813*1.e30
m_u = 1.66053904e-24  # amu
kB = 1.38064852e-16
hcperk = 1.4387773538277202  # hc/kB (cm K)
ccgs = 29979245800.0  # c in cgs
logm_ucgs = -23.779750909492115  # log(m_u) in cgs unit
ecgs = 4.80320450e-10  # [esu]=[dyn^0.5*cm] #elementary charge
mecgs = 9.10938356e-28  # [g] !electron mass
eV2wn = 8065.54  # 1[eV]=8065.54[cm^-1]
hcgs = 6.62607015e-27  # Planck constant [erg*s]
Rcgs = 1.0973731568e5  # Rydberg constant [cm-1]
a0 = 5.2917720859e-9  # Bohr radius [cm]

# km/s
c = 299792.458
Gcr = 115.38055682147402 #cuberoot of Gravitaional constant in the unit of [km/s] normalized by day and Msun

if __name__ == "__main__":
    #derivation of Gcr
    from astropy.constants import G
    from astropy.constants import M_sun
    from astropy import units as u
    day = 24*3600*u.s
    Gu = (G*M_sun/day).value
    Gcr_val = Gu**(1.0/3.0)*1.e-3
    print(Gcr_val)

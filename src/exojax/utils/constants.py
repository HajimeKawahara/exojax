"""constants.

* constants in cgs or km/s
* Tref_original: reference temperature used in exojax.spec

"""

# original reference temperature used in HITRAN/HITEMP in K
Tref_original = 296.0

#temperature at water triple point  (K)
Ttp_water = 273.16 
# 0 Celsius degree in Kelvin
Tc_water = 273.15

# cgs unit
#Rs = 6.957 * 1.e10
Rs = 6.9551 * 1.e10
RJ = 7.1492 * 1.e9
RE = 6.3781 * 1.e8
au = 1.495978707 * 1.e13
pc = 3.0856775814913673 * 1.e18
G = 6.67408e-08
#MJ = 1.89813 * 1.e30  # Jovian mass
MJ = 1.8986 * 1.e30  # Jovian mass
gJ = 2478.57730044555  # Jovian gravity
m_u = 1.66053904e-24  # atomic mass unit [g]
kB = 1.38064852e-16
logkB = -15.859916868309735  # log10(kB)
hcperk = 1.4387773538277202  # hc/kB (cm K)
ccgs = 29979245800.0  # c in cgs
logm_ucgs = -23.779750909492115  # log(m_u) in cgs unit = np.log10(m_u*1.e3) where m_u = scipy.constants.m_u.
ecgs = 4.80320450e-10  # [esu]=[dyn^0.5*cm] #elementary charge
mecgs = 9.10938356e-28  # [g] !electron mass
eV2wn = 8065.54  # 1[eV]=8065.54[cm^-1]
hcgs = 6.62607015e-27  # Planck constant [erg*s]
Rcgs = 1.0973731568e5  # Rydberg constant [cm-1]
a0 = 5.2917720859e-9  # Bohr radius [cm]
bar_cgs = 1.e6 # 1 bar in cgs = 1e6 dyn/cm2


# in bar unit (1bar = 10**6 dyn/cm2 (cgs))
Patm = 1.01325  # 1 atm in bar

# in km/s unit
c = 299792.458
Gcr = 115.38055682147402  #cuberoot of Gravitaional constant in the unit of [km/s] normalized by day and Msun

# opacity factor
# opfac = bar_cgs/(m_u (g)). m_u: atomic mass unit. bar_cgs: 1 bar in cgs = 1.e6 dyn/cm2
# obtained as opfac = bar_cgs/m_u(in g) = 1.e6/(m_u(in kg)*1.e3) = 1.e3/m_u(in kg), m_u(in kg) = scipy.constants.m_u.
opfac = 6.022140858549162e+29

import numpy as np
from petitRADTRANS import Radtrans
import matplotlib.pyplot as plt
#atmosphere = Radtrans(line_species = ['CH4_main_iso'], \
atmosphere = Radtrans(line_species = ['COej_HITEMP19'], \
                      #  'H2O_main_iso', 'CO_all_iso', \
#      'CH4_main_iso', 'CO2_main_iso', 'Na', 'K', \
#      'C2H2', 'HCN_main_iso', 'NH3', 'H2S_main_iso'], \
      #rayleigh_species = ['H2', 'He'], \
      #continuum_opacities = ['H2-H2', 'H2-He'], \
      wlen_bords_micron = [2.3,2.4], \
      mode = 'lbl')

pressures = np.logspace(-10, 2, 130)
print(len(pressures))
with open('Gl229/pressures.dat', 'w') as f:                                
  for a in range(0,len(pressures)):                               
    f.write('{:e}\n'.format(pressures[a]))
    
atmosphere.setup_opa_structure(pressures)



import petitRADTRANS.nat_cst as nc

R_pl = 1.838*nc.r_jup_mean
gravity = 1e1**5.0
P0 = 0.01

kappa_IR = 0.01
gamma = 0.4
T_int = 200.
T_equ = 1500.
temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)
print(len(temperature))

data = open('Gl229/MMR/eq.dat', 'r')
line = data.read().split()
data.close()

molname = {}
for a in range(0,160):
  molname[a] = line[a+6]
  print(a, molname[a])

  
data = open('Gl229/MMR/eq.dat', 'r')
a = 0
MMW_k = np.ones_like(temperature)
MMR = [[0] * len(pressures) for j in range(160)]
for line in data:
    if line[0]=='#':
        continue
    lines = line.rstrip('\n').split()
    temperature[a] = lines[1]
    MMW_k[a] = lines[2]
    for i in range(0,160):
      MMR[i][a] = lines[i+3]
    a = a + 1
data.close()



abundances = {}
abundances['H2'] = 7.1383555382E-001 * np.ones_like(temperature)
for a in range(0,130):
  abundances['H2'][a] = MMR[1][a]

abundances['He'] = 0.24 * np.ones_like(temperature)
for a in range(0,130):
  abundances['He'][a] = MMR[2][a]

abundances['H2O_main_iso'] = 0.0 * np.ones_like(temperature)
for a in range(0,130):
  abundances['H2O_main_iso'][a] = MMR[59][a]

abundances['COej_HITEMP19'] = 0.01 * np.ones_like(temperature)
#for a in range(0,130):
#  abundances['CO_all_iso'][a] = MMR[26][a]

abundances['CO2_main_iso'] = 0.00001 * np.ones_like(temperature)
for a in range(0,130):
  abundances['CO2_main_iso'][a] = MMR[27][a]

abundances['CH4_main_iso'] = 0.000001 * np.ones_like(temperature)
for a in range(0,130):
  abundances['CH4_main_iso'][a] = MMR[19][a]

abundances['Na'] = 0.00001 * np.ones_like(temperature)
for a in range(0,130):
  abundances['Na'][a] = MMR[5][a]

abundances['K'] = 0.000001 * np.ones_like(temperature)
for a in range(0,130):
  abundances['K'][a] = MMR[3][a]

abundances['C2H2'] = 0.000001 * np.ones_like(temperature)
for a in range(0,130):
  abundances['C2H2'][a] = MMR[30][a]

abundances['HCN_main_iso'] = 0.000001 * np.ones_like(temperature)
for a in range(0,130):
  abundances['HCN_main_iso'][a] = MMR[13][a]

abundances['NH3'] = 0.000001 * np.ones_like(temperature)
for a in range(0,130):
  abundances['NH3'][a] = MMR[61][a]

abundances['H2S_main_iso'] = 0.000001 * np.ones_like(temperature)
for a in range(0,130):
  abundances['H2S_main_iso'][a] = MMR[92][a]

  
MMW = 2.33 * np.ones_like(temperature)
for a in range(0,130):
  MMW[a] = MMW_k[a]

plt.plot(temperature,pressures)
plt.yscale("log")
plt.gca().invert_yaxis()
plt.show()
plt.clf()

atmosphere.calc_flux(temperature, abundances, gravity, MMW, contribution = True)
print(atmosphere.contr_em)

plt.rcParams['figure.figsize'] = (10, 6)

wlen_mu = nc.c/atmosphere.freq/1e-4
X, Y = np.meshgrid(wlen_mu, pressures)
plt.contourf(X,Y,atmosphere.contr_em,30,cmap=plt.cm.bone_r)

plt.yscale('log')
plt.xscale('log')
plt.ylim([1e2,1e-10])
#plt.xlim()

plt.xlabel('Wavelength (microns)')
plt.ylabel('P (bar)')
plt.title('Emission contribution function')
plt.gca().invert_yaxis()
plt.xlim(2.3800,2.3900)
plt.show()
plt.clf()


with open('Gl229/Gl229B_spectrum_CO.dat', 'w') as f:
  for a in range(0,len(atmosphere.freq)):                               
    f.write('{:e} {:e}\n'.format(nc.c/atmosphere.freq[a]/1e-4, atmosphere.flux[a]))



import pylab as plt
plt.rcParams['figure.figsize'] = (10, 6)

plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6)

plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
plt.show()
plt.savefig('Gl229/emission-HR.pdf',bbox_inches='tight')
plt.clf()



plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6)

plt.xlim([2.3,2.3025])
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
plt.show()
plt.savefig('Gl229/emission-HR-zoom.pdf',bbox_inches='tight')
plt.clf()


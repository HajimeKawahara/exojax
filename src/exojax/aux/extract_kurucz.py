import numpy as np
import math
import struct
import os
import argparse
import pf_bc as pf
from scipy import interpolate
from astropy import constants as const
from astropy import units as u

def linestrength_normal(gf,nui,hatnu,QT,T):
	#line strength in the unit of cm2/s/species. see Sharps & Burrows equation(1)
	#all quantities should be converted to the cgs unit
	#gf   : g(statistical weight) * f(oscillator strength)
	#nui  : initial wavenumber in cm-1
	#hatnu: line position in cm-1
	#QT: partition function
	#T   : temperature
	
	eesu=const.e.esu
	ee=(eesu*eesu).to(u.g*u.cm*u.cm*u.cm/u.s/u.s).value
	me=const.m_e.cgs.value
	c=const.c.cgs.value
	h=const.h.cgs.value
	k=const.k_B.cgs.value
	
	fac0=np.pi*ee/me/c
	fac1=-h*c*nui/k/T
	fac2=-h*c*hatnu/k/T
	Snorm=fac0*gf/QT*np.exp(fac1)*(-np.expm1(fac2))
	return Snorm

def linestrength_hitran_zero(gf,nui,hatnu,QT0,T0):
	#line strength used in HITRAN form (cm/species). see Sharps & Burrows equation(11)
	#QT0=Q(T=296 K)
	#print "REFERENCE TEMPERATURE=",T0
	c=const.c.cgs.value
	Snorm=linestrength_normal(gf,nui,hatnu,QT0,T0)
	Sh0=np.array(Snorm/c)
	
		#cleanup
	mask=(Sh0<0.0)
	Sh0[mask]=0.0
			
	return Sh0

# This script downloads the Kurucz gfnew files and produces the binary and *.param files for HELIOS-K

# Date: May 2019
# Author: Simon Grimm
# Adopted by S. K. Nugroho for Py4CATS


#choose if the file contains wavenumber or not
Wavenumber = 1


#filename="gfall08oct17.dat"
#filename="gfallvac08oct17.dat"

filename="gfallwn08oct17.dat"
#filename="gf2600.all"
#filename="hyper1900.all"



elt0=[
[  100, "H"  , 1.00794],
[  200, "He" , 4.002602],
[  300, "Li" , 6.941],
[  400, "Be" , 9.012182],
[  500, "B"  , 10.811],
[  600, "C"  , 12.011],
[  700, "N"  , 14.00674],
[  800, "O"  , 15.9994],
[  900, "F"  , 18.9984032],
[ 1000, "Ne" , 20.1797],
[ 1100, "Na" , 22.989768],
[ 1200, "Mg" , 24.3050],
[ 1300, "Al" , 26.981539],
[ 1400, "Si" , 28.0855],
[ 1500, "P"  , 30.973762],
[ 1600, "S"  , 32.066],
[ 1700, "Cl" , 35.4527],
[ 1800, "Ar" , 39.948],
[ 1900, "K"  , 39.0983],
[ 2000, "Ca" , 40.078],
[ 2100, "Sc" , 44.955910],
[ 2200, "Ti" , 47.88],
[ 2300, "V"  , 50.9415],
[ 2400, "Cr" , 51.9961],
[ 2500, "Mn" , 54.93805],
[ 2600, "Fe" , 55.847],
[ 2700, "Co" , 58.93320],
[ 2800, "Ni" , 58.6934],
[ 2900, "Cu" , 63.546],
[ 3000, "Zn" , 65.39],
[ 3100, "Ga" , 69.723],
[ 3200, "Ge" , 72.61],
[ 3300, "As" , 74.92159],
[ 3400, "Se" , 78.96],
[ 3500, "Br" , 79.904],
[ 3600, "Kr" , 83.80],
[ 3700, "Rb" , 85.4678],
[ 3800, "Sr" , 87.62],
[ 3900, "Y"  , 88.90585],
[ 4000, "Zr" , 91.224],
[ 4100, "Nb" , 92.90638],
[ 4200, "Mo" , 95.94],
[ 4300, "Tc" , 97.9072],
[ 4400, "Ru" ,101.07],
[ 4500, "Rh" ,102.90550],
[ 4600, "Pd" ,106.42],
[ 4700, "Ag" ,107.8682],
[ 4800, "Cd" ,112.411],
[ 4900, "In" ,114.818],
[ 5000, "Sn" ,118.710],
[ 5100, "Sb" ,121.757],
[ 5200, "Te" ,127.60],
[ 5300, "I"  ,126.90447],
[ 5400, "Xe" ,131.29],
[ 5500, "Cs" ,132.90543],
[ 5600, "Ba" ,137.327],
[ 5700, "La" ,138.9055],
[ 5800, "Ce" ,140.115],
[ 5900, "Pr" ,140.90765],
[ 6000, "Nd" ,144.24],
[ 6100, "Pm" ,144.9127],
[ 6200, "Sm" ,150.36],
[ 6300, "Eu" ,151.965],
[ 6400, "Gd" ,157.25],
[ 6500, "Tb" ,158.92534],
[ 6600, "Dy" ,162.50],
[ 6700, "Ho" ,164.93032],
[ 6800, "Er" ,167.26],
[ 6900, "Tm" ,168.93421],
[ 7000, "Yb" ,173.04],
[ 7100, "Lu" ,174.967],
[ 7200, "Hf" ,178.49],
[ 7300, "Ta" ,180.9479],
[ 7400, "W"  ,183.84],
[ 7500, "Re" ,186.207],
[ 7600, "Os" ,190.23],
[ 7700, "Ir" ,192.22],
[ 7800, "Pt" ,195.08],
#[ 7900, "Au" ,196.96654],
#[ 8000, "Hg" ,200.59],
#[ 8100, "Tl" ,204.3833],
#[ 8200, "Pb" ,207.2],
#[ 8300, "Bi" ,208.98037],
#[ 8400, "Po" ,208.9824],
#[ 8500, "At" ,209.9871],
#[ 8600, "Rn" ,222.0176],
#[ 8700, "Fr" ,223.0197],
#[ 8800, "Ra" ,226.0254],
#[ 8900, "Ac" ,227.0278],
#[ 9000, "Th" ,232.0381],
#[ 9100, "Pa" ,231.03588],
#[ 9200, "U"  ,238.0289],
#[ 9300, "Np" ,237.0482],
#[ 9400, "Pu" ,244.0642],
#[ 9500, "Am" ,243.0614],
#[ 9600, "Cu" ,247.0703],
#[ 9700, "Bk" ,247.0703],
#[ 9800, "Cf" ,251.0796],
#[ 9900, "Es" ,252.0830],
#[10000, "Fm" ,257.0951],
#[10100, "Md" ,258.0984],
#[10200, "No" ,259.1011],
#[10300, "Lr" ,262.1098],
#[10400, "Rf" ,261.1089],
#[10500, "Db" ,262.1144],
#[10600, "Sg" ,263.1186],
#[10700, "Bh" ,264.12],
#[10800, "Hs" ,265.1306],
#[10900, "Mt" ,268.00],
#[11000, "Ds" ,268.00],
#[11100, "Rg" ,272.00],
#[11200, "Cn" ,277.00],
#[11300, "Uut" ,0.00],
#[11400, "Fl" ,289.00],
#[11500, "Uup" ,0.00],
#[11600, "Lv" ,289.00],
#[11700, "Uus" ,294.00],
#[11800, "Uuo" ,293.00]
]

eesu=const.e.esu
ee=(eesu*eesu).to(u.g*u.cm*u.cm*u.cm/u.s/u.s).value
me=const.m_e.cgs.value
c=const.c.cgs.value

def main(Download):

	#for i in range(18,19):
	for i in range(2,78):
		for j in range(0,2):
			if int(elt0[i][0])/100==33 or int(elt0[i][0])/100==34 or int(elt0[i][0])/100== 37 or int(elt0[i][0])/100== 51 or int(elt0[i][0])/100== 52 or int(elt0[i][0])/100== 55 or int(elt0[i][0])/100== 78:
				if j==1:
					continue
				else:
					processLineList(i, j, Download)
			elif int(elt0[i][0])/100==35 or int(elt0[i][0])/100==36 or int(elt0[i][0])/100==43 or int(elt0[i][0])/100== 53 or int(elt0[i][0])/100== 54 or int(elt0[i][0])/100== 61 :
				continue
			else:
				processLineList(i, j, Download)

def processLineList(i, j, Download):
	# i molecule id 0 to 100
	# j ion id 0 to 3

	el=elt0[i]
	if (j==0):
		name = el[1] + "_I"
	if(j==1):
		name = el[1] + "_II"
		el[0] = el[0] + 1
	if(j==2):
		name = el[1] + "_III"
		el[0] = el[0] + 1

	els= "% 6.2f" % (el[0] / 100.0)

#name ="%s" % el[1]


#outname = "%s-hypr.lines" % name
	outname = "%s.lines" % name
	pfname = "%s.pf" % name

	mass = el[2]
	
	print(el[0], els, name, outname, mass)

	if(Download == 1):
		#download file

		exists = os.path.isfile("%s" % filename)
		if(exists == 0):
			com = "wget http://kurucz.harvard.edu/linelists/gfnew/%s" % filename
			print(com)
			os.system(com)

	T,QT=np.loadtxt(pfname,unpack= True)
	T0=3000.
	QT0= interpolate.interp1d(T,QT,kind="cubic")(T0)
	
	
	pf_file = open(pfname,"w")
	for ii in range(len(T)):
		pf_file.write("%g %g\n" % (T[ii], QT[ii]))
	pf_file.close()

	
	numax = 0.0
	nl = 0	
	LabelLOld =""
	LabelUOld =""
	gUPOld = -1
	gLowOld = -1
	ELowOld =  -1.0
	EUPOld = -1.0

	with open(filename) as f:
		line = f.readlines()

		position=[]
		strength=[]
		energy=[]
		airWidth=[]
		Tdep=[]
		hyper=0
		same=0
		diff=0
		for ii in range(len(line)):
			l = line[ii]
			#E in cm^-1
			#atomic line list format
		
			wn = float(l[0:11])
			wn=np.array(wn,dtype="f4")
			wl = 1.0E7/wn		#wavelenght in nm
		
			loggf = float(l[11:18])
			element = l[18:24]
			ELow = float(l[24:36])
			JLow = float(l[36:41])
			LabelL = l[42:52]
			EUP = float(l[52:64])
			JUP = float(l[64:69])
			LabelU = l[70:80]
			GammaR = (l[80:86])
			isotope = l[106:109]
			hyperFineFraction = float(l[109:115])
			ISOFraction = float(l[118:124])
			hyperShiftL = (l[124:129])
			hyperShiftU = (l[129:134])
	
			if(isotope == "   " or isotope == ""):
				isotope = "  0"
			if(GammaR == "      " or GammaR == ""):
				GammaR = "  0"
			if(hyperShiftL == "     " or hyperShiftL == ""):
				hyperShiftL = "  0"
			if(hyperShiftU == "     " or hyperShiftU == ""):
				hyperShiftU = "  0"
			
			GammaR = float(GammaR)
			hyperShiftL = float(hyperShiftL)
			hyperShiftU = float(hyperShiftU)


#			e = 4.80320425E-10      #electron charge in cgs units [statcoulomb = cm^(3/2) g^(1/2) s^-1]
#			c = 2.99792458E10       #Speed of light cm/s
#			me = 9.1093835611E-28   #mass of electron in g
			NA = 6.0221412927e23	#Avogadro Constant  1/mol


			ELow = abs(ELow)
			EUP = abs(EUP)


			#somethimes ELOW is larger than EUP
			
			if(ELow > EUP):
				t = EUP
				EUP = ELow
				ELow = t

				t = JUP
				JUP = JLow
				JLow = t
				
				t= hyperShiftU
				hyperShiftU = hyperShiftL
				hyperShiftL = t

				t = LabelU
				LabelU = LabelL
				LabelL = t
				
				#if(element == els):
				#	print("swap energies")
				
			#convert air wavelength to vacuum wavelength
			#http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
				

			gUP = 2 * JUP + 1
			gLow = 2 * JLow + 1
			
			#if(element == " 19.00"):
			if(element == els):
#				A = 8.0 * math.pi * wn * wn * (10.0**loggf) / gUP * math.pi * e * e / (me * c)
#                gamma = 2.223e13 / (wl * wl) #Gray 1976 natural broadening approximation/radiation dampening

				sameLabel = 0
				if(LabelL == LabelLOld and LabelU == LabelUOld and gUP == gUPOld and gLow == gLowOld and ELow == ELowOld and EUP == EUPOld):
					sameLabel = 1
				
				if(sameLabel == 0):
					HF = 10.0**hyperFineFraction
				else:
					HF += 10.0**hyperFineFraction
#					hyper=hyper+1
#					print (hyper)
#					print (1e8/(EUP-ELow))

				'''	
				# use this block to filer out hyperfine splits
				###################################
				hyperFineFraction = 0.0
				ISOFraction = 0.0
				wn += 0.001 * hyperShiftL
				wn -= 0.001 * hyperShiftU
				hyperShiftU = 0.0
				hyperShiftL = 0.0
				if(sameLabel == 1):
					continue
				##################################
				'''

				#print(element, wn, isotope, GammaR, 10.0**GammaR, A, gamma)
				#print(element, wn, isotope,  ELow, EUP, gLow, gUP, 10.0**loggf, 10.0**hyperFineFraction, 10.0**ISOFraction, LabelL, LabelU, hyperShiftL, hyperShiftU, sameLabel, HF)

				#if(HF > 1.001):
				#	print("***", element, wn, isotope, HF)


				LabelLOld = LabelL
				LabelUOld = LabelU
				gUPOld = gUP
				gLowOld = gLow
				EUPOld = EUP
				ELowOld = ELow
			
				
			#this must be done after the Hyperfine fraction filtering. (Old value comparison)
			ELow += 0.001 * hyperShiftL
			EUP  += 0.001 * hyperShiftU


			
			isotope = int(isotope)
			if(element == els):
				nl = nl + 1

				numax = max(numax, wn)

				wn_hyper=np.array(EUP-ELow,dtype="f4")
				wn=np.array(wn,dtype="f4")
				
				if wn_hyper==wn:
					same=same+1
				else:
					diff=diff+1
				
#				S = np.pi * ee * 10.0**loggf /(c * c * me) * 10.0**hyperFineFraction * 10.0**ISOFraction/QT0

				S=linestrength_hitran_zero(10.0**loggf,ELow,wn_hyper,QT0,T0)#* 10.0**hyperFineFraction * 10.0**ISOFraction

				#A = 8.0 * math.pi * wn * wn * 10.0**loggf / gUP * math.pi * e * e / (me * c)
				
				#print(wn, 1.0E7/wn, loggf, ELow, EUP, JLow, JUP, GammaR, isotope, element, mass, 10.0**GammaR, A, 10.0**hyperFineFraction, 10.0**ISOFraction)
				position.append(wn_hyper)
				strength.append(S)
				energy.append(ELow)
				airWidth.append(GammaR)
				Tdep.append(0)
	
		ind_sort=np.argsort(position)
		position=position[ind_sort]
		strength=strength[ind_sort]
		energy=energy[ind_sort]
		airWidth=airWidth[ind_sort]
		Tdep=Tdep[ind_sort]
		
		header=[
		'# molecule: '+str(name)+"\n", \
		'# temperature: '+str(T0)+' K'+"\n", \
		'# pressure: 1013.25 mb'+"\n", \
		'# number of lines: '+str(len(position))+"\n", \
		'# min, max line strength: '+str(min(strength))+'     '+str(max(strength))+"\n", \
		'# format: vSEan'+"\n", \
		'# min, max airbroad.  widths: '+str(min(airWidth))+'     '+str(max(airWidth))+"\n", \
		'# min, max selfbroad. widths: 0.000        0.000'+"\n", \
		'# min, max temp. exponent: '+str(min(Tdep))+'     '+str(max(Tdep))+"\n", \
		'#   position    strength      energy airWidth    Tdep'+"\n", \
		'#       cm-1   cm-1/cm-2        cm-1     cm-1'+"\n" ]
		
		g=open(outname,"w")
		for ih in header:
			g.writelines(ih)
		for iline in range(len(position)):
			g.writelines(str.format("{0:.10f}",position[iline])+" "+str(strength[iline])+" "+str(energy[iline])+" "+str(airWidth[iline])+" "+str(Tdep[iline])+"\n")
		g.close()
		print(" Lines:"+str(nl))
		print(str(same))
		print(str(diff))
		print("")
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('-D', '--Download', type=str,
		help='Download the files', default = 0)

	args = parser.parse_args()
	Download = int(args.Download)

	print("Download: %d" % Download)

	main(Download)

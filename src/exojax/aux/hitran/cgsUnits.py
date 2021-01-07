#!/usr/bin/env python3

"""
  cgsUnits
  convert (physical) data between (compatible) units

  usage:
  cgsUnits oldUnit newUnit value(s)

  NOTE:  the (physical) quantities and units supported are far from complete.
         (see the source file for quantities and units known)
"""

##############################################################################################################
#####  LICENSE issues:                                                                                   #####
#####                 This file is part of the Py4CAtS package.                                          #####
#####                 Copyright 2002 - 2019; Franz Schreier;  DLR-IMF Oberpfaffenhofen                   #####
#####                 Py4CAtS is distributed under the terms of the GNU General Public License;          #####
#####                 see the file ../license.txt in the parent directory.                               #####
##############################################################################################################

try:                      import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')

from exojax.aux.hitran.ir import c
from exojax.aux.hitran.molecules import molecules

####################################################################################################################################
#
#  concentration conversion:
#  http://cimss.ssec.wisc.edu/itwg/groups/rtwg/rtairs.html
#  To convert from specific concentration q in kg/kg to volume mixing ratio v in ppmv use the following equation:
#  v = 1e6 * q*M_air / ((1-q)*M_wv + q*M_air)
#  where M_air = 28.9644 and M_wv = 18.01528 are the molecular weights of dry air and water vapour
#  and r is their ratio Mwv/Mair= 0.62198.
#
####################################################################################################################################

cgs_units = {'length': 'cm', 'pressure': 'g/(cm*s**2)', 'temperature': 'K', 'density': '1/cm**3', 'vmr': 'pp1',
             'energy': 'erg', 'power': 'erg/s'}

# conversion factors for cgs units
# for the micro = 10^{-6} prefix the latin letter 'u' is used mimicking the greek \mu (similar to astropy.units)
# (see also https://de.wikipedia.org/wiki/Vors%C3%A4tze_f%C3%BCr_Ma%C3%9Feinheiten#Typographie)
# for wavelengths some other variants are possible, too (see source below)

pressureUnits    = {'g/cm/s**2': 1.0, 'g/(cm.s**2)': 1.0, 'g/(cm*s**2)': 1.0, 'g/(cm.s^2)': 1.0, 'g/(cm*s^2)': 1.0,
                    'dyn/cm^2': 1.0, 'dyn/cm**2': 1.0,
                    'mb': 1.e3, 'mbar': 1.e3, 'bar': 1.e6, 'hPa': 1.e3,  'atm': 1013250., 'Pa': 10., 'N/m**2': 10., 'N/m^2': 10.,
		    'torr': 1.33322e3, 'Torr': 1.33322e3}
frequencyUnits   = {'Hz': 1.0, 'kHz': 1.0e3, 'MHz': 1.0e6, 'GHz': 1.0e9, 'THz': 1.0e12, 'cm-1': c, '1/cm': c}
wavelengthUnits  = {'cm': 1.0, 'mm': 0.1, 'mue': 1.e-4, 'mum': 1.e-4, 'um': 1.e-4, 'micrometer': 1.e-4,  'nm': 1.e-7, 'A': 1.e-8}
lengthUnits      = {'km': 1.e5, 'm': 1.e2, 'dm': 10., 'inch': 2.54};  lengthUnits.update(wavelengthUnits)
areaUnits        = dict([(nam+'**2', val**2) for nam,val in list(lengthUnits.items())])
volumeUnits      = dict([(nam+'**3', val**3) for nam,val in list(lengthUnits.items())]);  volumeUnits.update({'l': 1.e3, 'hl': 1.e5})
mixingRatioUnits = {'ppv': 1.0, 'ppV': 1.0, 'pp1': 1.0, 'vmr': 1.0, '%': 1.e-2, 'ppm': 1.e-6, 'ppb': 1.e-9, 'ppt': 1.e-12}
densityUnits     = dict( [('1/'+nam+'**3', val**-3) for nam,val in list(lengthUnits.items())] +
                         [(nam+'-3', val**-3) for nam,val in list(lengthUnits.items())])
energyUnits      = {'erg': 1.0, 'g.cm**2/s**2': 1.0, 'kg.m**2/s**2': 1.e7, 'Nm': 1.e7, 'N.m': 1.e7, 'J': 1.e7, 'mJ': 1.e4}
powerUnits       = {'erg/s': 1.0, 'g.cm**2/s**3': 1.0, 'kg.m**2/s**3': 1.e7, 'W': 1.e7, 'mW': 1.e4, 'uW': 1.e1, 'nW': 1.e-2}
massUnits        = {'g': 1.0, 'kg': 1000., 'mg': 1.e-3, 'ug': 1.e-6, 'ton': 1.e6, 'amu': 1.660538782e-24}

# given here for completeness, do not mix with the other units because T conversions are additive
temperatureUnits = {'Kelvin': 0.0, 'K': 0.0, 'k': 0.0, 'C': 273.15, 'Celsius': 273.15}  # hmmm, lower case 'k' only to satisfy libradtran atmospheric data files

# a dictionary of dictionaries
# temperatureUnits: do NOT include in this list, otherwise the if-block in unitConversion fails for T
allQuantities = {'length':      lengthUnits,
                 'area':        areaUnits,
                 'volume':      volumeUnits,
                 'pressure':    pressureUnits,
                 'density':     densityUnits,
                 'mixingratio': mixingRatioUnits,
                 'frequency':   frequencyUnits,
                 'mass':        massUnits,
                 'power':       powerUnits,
                 'energy':      energyUnits}

# combine all dictionaries
allUnits={}
for units in list(allQuantities.values()):  allUnits.update(units)

# some copies for aliases
allQuantities['p'] = pressureUnits
allQuantities['wavelength'] = wavelengthUnits
for alias in ['z', 'altitude', 'height']:  allQuantities[alias] = lengthUnits

####################################################################################################################################

def cgs (unit, data=1.0):
	""" Conversion of (scalar or array) physical value(s) to its cgs base unit (e.g. 'cm' or 'g').
	    If no data are given, simply return the conversion factor to the cgs base unit.
	    If unit starts with an exclamation mark (!), convert from the base unit.
	    If unit contains an exclamation mark, convert from the unit given before the ! to the unit after the !

	    ARGUMENTS:
	    unit:    a text string like "cm" or "mb"
	    data:    optional float or list or array of physical quantities

	    RETURNS:
	    either simply the conversion factor to (from) the cgs base unit
	    or the data converted to (or from) the cgs base unit

	    EXAMPLES:
	    cgs('km')                ---> 100000.0
	    cgs('!kg',amu)           ---> 1.660538782e-27
	    cgs('mb ! atm',1013.25)  ---> 1.0

	    NOTES:
	    * the greek mu can be specified with the latin 'u' (e.g. 'um' for wavelength in micrometer)
	    * temperature conversion K <--> C not implemented (additive, not multiplicative)
	    * the list of supported/known units is far from complete
	      (for a complete conversion module see astropy.units, http://www.astropy.org)
	    """

	# some preparation
	unit = unit.strip()
	ix   = unit.find('!')
	if unit=='?':  return allUnits
	if isinstance(data,(list,tuple)):    data = np.array(data)

	try:
		if    ix==0:  return  data * (1.0/allUnits[unit[1:].strip()])  # division not supported for PairOfFloats or Interval
		elif  ix>0:   return  data * allUnits[unit[:ix].strip()] / allUnits[unit[ix+1:].strip()]
		else:         return  data * allUnits[unit.strip()]
	except KeyError:                     raise SystemExit ("ERROR --- cgsUnits.cgs:  unknown unit " + repr(unit))


####################################################################################################################################

def unitConversion (data, WHAT, old=None, new=None):
	""" Conversion of (scalar or array) physical values to different units.
	    If old (input) or new (output) unit is not given: assume cgs standard unit for this quantity. """
	# alternative approaches (see also discussion in H.P. Langtangen's book):
	# UNUM     http://pypi.python.org/pypi/Unum/4.1.0
	# ScientificPython: http://dirac.cnrs-orleans.fr/ScientificPython/
	# astropy:  http://www.astropy.org/
	what = WHAT.lower()

	# check if name is given in plural form
	if len(what)>2 and what.endswith('s'):
		if what.endswith('ies'):
			if what[:-3]+'y' in list(allQuantities.keys()):  what=what[:-3]+'y'
		else:
			if (what[:-1] in list(allQuantities.keys())) or what[:4]=='temp':  what=what[:-1]
		if not what==WHAT:
			print('cgsUnits.unitConversion (replacing plural->singular):', WHAT, '--->', what)

	if what in list(allQuantities.keys()):
		xUnits = allQuantities[what]
		if old==new:
			return data
		elif old in xUnits and new in xUnits:
			return data * xUnits[old] / xUnits[new]
		elif old in xUnits and not new:
			return data * xUnits[old]  # convert to cgs 'base' unit
		elif new in xUnits and not old:
			return data / xUnits[new]  # convert from cgs 'base' unit
		else:
			print('\n', WHAT, list(xUnits.keys()))
			raise SystemExit ('%s %s %s %s' % ('ERROR --- unitConversion:  unknown/unsupported units ',
			                                   old, ' ---> ', new))
	elif WHAT=='T' or what[:4].lower()=='temp':
		return cgsTemperature (data, old, new)
	elif WHAT in list(molecules.keys()):
		print("INFO --- unitConversion:  ignoring molecular data for ", WHAT)
	else:
		raise SystemExit ('ERROR --- unitConversion failed, unknown/unsupported quantity ' + WHAT)


####################################################################################################################################

def cgsTemperature (data, old=None, new=None):
	""" Temperature unit conversion:  additive, so the standard scheme (multiplicative) does not work. """
	if old==new:
		return data
	elif old in temperatureUnits and new in temperatureUnits:
		return data + temperatureUnits[old] - temperatureUnits[new]
	elif old in temperatureUnits and not new:
		return data - temperatureUnits[old]
	elif new in temperatureUnits and not old:
		return data + temperatureUnits[new]
	else:
		raise SystemExit ('%s %s %s %s' %
		      ('ERROR --- cgsTemperature:  unit conversion failed, unknown/unsupported units ', old, ' ---> ', new))


####################################################################################################################################


def change_frequency_units (x, xUnitOld, xUnitNew):
	""" Convert frequency <--> wavenumber <--> wavelength. """
	if xUnitOld=='cm-1':
		if   xUnitNew=='cm-1': pass
		elif xUnitNew=='THz':  x = x*c * 1e-12
		elif xUnitNew=='GHz':  x = x*c * 1e-9
		elif xUnitNew=='MHz':  x = x*c * 1e-6
		elif xUnitNew=='Hz':   x = x*c
		elif xUnitNew=='nm':   x = 10000000./x
		elif xUnitNew in ['mue','micro','mum']: x = 10000./x
		else: raise SystemExit ('ERROR: unknown/invalid unit for wavenumber/frequency/wavelength!')
	elif xUnitOld.endswith('Hz'):
		if   xUnitOld=='Hz':   pass
		elif xUnitOld=='kHz':  x = x*1e3
		elif xUnitOld=='MHz':  x = x*1e6
		elif xUnitOld=='GHz':  x = x*1e9
		elif xUnitOld=='THz':  x = x*1e12
		if xUnitNew=='cm-1':   x = x/c
		else: raise SystemExit ('ERROR: conversion ' + xUnitOld + ' --> ' + xUnitNew + ' not yet implemented!')
	elif xUnitOld in ['mue','micro','mum']:
		if   xUnitNew=='cm-1':  x = 10000./x
		else: raise SystemExit ('ERROR: conversion ' + xUnitOld + ' --> ' + xUnitNew + ' not yet implemented!')
	elif xUnitOld=='nm':
		#modified by HK (2021/1/6)
		if   xUnitNew=='cm-1':
		        x.upper=1.e7/x.lower                
		        x.lower=1.e7/x.upper
		else: raise SystemExit ("ERROR!")
	else:
		raise SystemExit ('ERROR: unknown/invalid unit for wavenumber/wavelength,frequency ' + xUnitOld)
	return x


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def nu2lambda (vData, freq='', nm=False):
	""" Convert wavenumber [cm-1] or frequency to wavelength.

	    ARGUMENTS:
	    ----------
	    vData     wavenumber(s) or frequency(ies)
	    freq      if empty:  input data in cm-1
	              for frequency give Hz | kHz | MHz | GHz | THz
	    nm        flag: return micrometer (default) or nanometer
	 """

	if isinstance(vData,(list,tuple)):  vData = np.array(vData)

	# first convert frequency to wavenumber:
	if   freq=='Hz':   vData /= c
	elif freq=='kHz':  vData *= 1e3/c
	elif freq=='MHz':  vData *= 1e6/c
	elif freq=='GHz':  vData *= 1e9/c
	elif freq=='THz':  vData *= 1e12/c
	elif freq in '1/cm cm-1 cm**-1 cm^-1'.split():  pass   # better use re for 'cm' & '-1'
	elif len(freq)>0:  raise SystemExit ('ERROR --- nu2lambda:  unknown / invalid frequency unit')
	else:              pass

	# return reciprocal and scale for appropriate length unit
	if nm:  return 1e7/vData
	else:   return 1e4/vData


def lambda2nu (lData, nm=False, freq='', delta=False):
	""" Convert wavelength [um|nm] to wavenumber [1/cm] or frequency.

	    ARGUMENTS:
	    ----------
	    lData     wavelength(s)
	    nm        flag: input micrometer (default), otherwise nanometer
	    freq      if empty:  output data in cm-1
	              for frequency give Hz | kHz | MHz | GHz | THz
	    delta     return the differences between consecutive wavelengths
	 """

	if isinstance(lData,(list,tuple)):  lData = np.array(lData)

	# return reciprocal and scale for appropriate length unit
	if nm:  vData = 1e7/lData
	else:   vData = 1e4/lData

	# first convert frequency to wavenumber:
	if   freq=='Hz':   vData /= c
	elif freq=='kHz':  vData *= 1e3/c
	elif freq=='MHz':  vData *= 1e6/c
	elif freq=='GHz':  vData *= 1e9/c
	elif freq=='THz':  vData *= 1e12/c
	elif freq in '1/cm cm-1 cm**-1 cm^-1'.split():  pass   # better use re for 'cm' & '-1'
	elif len(freq)>0:  raise SystemExit ('ERROR --- nu2lambda:  unknown / invalid frequency unit')
	else:              pass

	if isinstance(lData,np.ndarray) and len(lData)>1 and delta:   return abs(np.ediff1d(vData))
	else:                                                         return vData


####################################################################################################################################

if __name__ == "__main__":
	import sys

	args = sys.argv[1:]

	if '-h' in sys.argv or '--help' in sys.argv:
		print(__doc__%globals())
		if args[0]=="--help":  print('default units in the cgs systems:\n', cgs_units)
		raise SystemExit (" End of cgsUnits help")

	if len(args)>2:
		old, new, values = args[0], args[1], np.array(list(map(float,args[2:])))
		if old in list(temperatureUnits.keys()) and new in list(temperatureUnits.keys()):
			print('temperature: ', cgsTemperature(values, old, new))
			sys.exit()
		# find what physical variable matches old and new unit
		what=''
		for name, units in list(allQuantities.items()):
			if old in list(units.keys()) and new in list(units.keys()):  what = name;  break
		if what:
			print('%s [%s]: ' % (what,new), unitConversion(values, what, old, new))
		else:
			raise SystemExit ('%s "%s" %s "%s" %s' % ('old', old, 'and/or new', new, 'unit unknown or incompatible'))
	else:
		raise SystemExit ('need at least three inputs:  oldUnit, newUnit, value(s)')

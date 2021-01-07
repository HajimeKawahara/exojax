#!/usr/bin/env python3

"""  lines
  Read molecular spectroscopic line parameters (Hitran, Geisa, ... extract)  and convert to new pressure/temperature

  usage:
  lines [options] line_parameter_file(s)

  -h          help
  -c char     comment character(s) used in input, output file (default '#')
  -o string   output file for saving of line data (if not given: write to StdOut)

  -m string   molecule (no default, should be given in the line file, required otherwise)
  -p float    pressure (in mb,  default: p_ref of linefile, usually 1013.25mb=1atm)
  -T float    Temperature (in K, default: T_ref of linefile, usually 296K)
  -x Interval lower, upper wavenumbers (comma separated pair of floats [no blanks!],
                                        default set according to range of lines in datafile)
 --plot char  plot wavenumber vs strength or width or ...
              (default "S" for strength, other choices are one of  "EansL")

  NOTES:
  in:  The line parameter file(s) should contain a list of (preselected) lines
       that can be generated from HITRAN or GEISA database with extract.py
       i.e., the original HITRAN or GEISA data bases cannot be used as input files
       (See the documentation header of lbl2xs.py for more details)

  out: The output file(s) are formatted automatically according to the extension given for the output file(s):
       if the extension is vSL or vSLD or vSLG, then 3 or 4 columns with position, strength, Lorentz (and Doppler) width are written
       otherwise, three columns with position and strengths at T_ref and T are written
       (actually there is a fourth column with zeros to facilitate plotting of delta S as (one sided) error bar)

  plot:  this is an alternative to the plot_atlas module  (none of them is perfect!)
"""

_LICENSE_ = """\n
This file is part of the Py4CAtS package.

Authors:
Franz Schreier
DLR Oberpfaffenhofen
Copyright 2002 - 2019  The Py4CAtS authors

Py4CAtS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Py4CAtS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

##############################################################################################################
##############################################################################################################

import os
from math import sqrt, log

try:                      import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')

# prepare plotting
try:
	from matplotlib.pyplot import figure, plot, semilogy, vlines, subplot, legend, xlabel, ylabel, xlim, title, show
except ImportError as msg:
	print (str(msg) + '\nWARNING --- lines:  matplotlib not available, no quicklook!')
	#raise SystemExit ('ERROR --- lines.lines_atlas:  matplotlib not available, no quicklook!')
else:
	pass # print 'from matplotlib import figure, plot, ...'

from exojax.aux.hitran.ir import c, k, amu, C2
from exojax.aux.hitran.molecules import get_molec_data, molecules, mainMolecules
from exojax.aux.hitran.aeiou import parse_comments, readFileHeader, open_outFile
from exojax.aux.hitran.struc_array import loadStrucArray, strucArrayAddField
from exojax.aux.hitran.cgsUnits import unitConversion, cgs
from exojax.aux.hitran.pairTypes import Interval, PairOfFloats, PairOfInts

# and some math constants
ln2     = log(2.0)
sqrtLn2 = sqrt(ln2)

# translation table for long names in file header to short names used in structured array
long2short = {'position': 'v',  'strength': 'S',  'energy': 'E',  'airWidth': 'a',  'selfWidth': 's',  'Tdep': 'n',
              'iso': 'i',  'mix': 'm',  'shift': 'd',  'pShift': 'd',  'Dicke': 'N',  'narrowing': 'N'}

verbose = os.getenv('verbose',0)

####################################################################################################################################

class lineArray (np.ndarray):
	""" A subclassed numpy array of core line parameters with molec, p, T, ... attributes added.

	    Furthermore, some convenience functions are implemented:
	    *  info:      print the attributes and the minimum and maximum xs values
	    *  __eq__:    the equality test accepts 0.1% differences of pressure and 1K for temperature
	    *  truncate:  return a subset of lines within a wavenumber interval
	    *  perturb:   return the lines with one of the parameters perturbed multiplicatively or additively
	    """
	# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

	def __new__(cls, input_array, p=None, t=None, molec=None):
		# Input array is an already formed ndarray instance
		# First cast to be our class type
		obj = np.asarray(input_array).view(cls)
		# add the new attributes to the created instance
		obj.p     = p
		obj.t     = t   # cannot use capital "T" because this means 'transpose'
		obj.molec = molec
		# Finally, we must return the newly created object:
		return obj

	def __array_finalize__(self, obj):
		# see InfoArray.__array_finalize__ for comments
		if obj is None: return
		self.p     = getattr(obj, 'p', None)
		self.t     = getattr(obj, 't', None)
		self.molec = getattr(obj, 'molec', None)

	def info (self):
		""" Print a summary statistics of the line parameter array (min, max position, strength, ...). """
		mainInfo = '%-8s  %8i lines in   %f ... %f cm-1   with' % (self.molec, len(self), min(self['v']), max(self['v']))
		blanks   = len(mainInfo)*' '
		print('%s  %8.2g < S < %8.2g   (T=%5.1fK, p=%.3e)' % (mainInfo, min(self['S']), max(self['S']), self.t, self.p))
		if 'E' in self.dtype.names:  print('%s  %8g < E < %8g' % (blanks, min(self['E']), max(self['E'])))
		if 'a' in self.dtype.names:  print('%s  %8g < a < %8g' % (blanks, min(self['a']), max(self['a'])))
		if 's' in self.dtype.names:  print('%s  %8g < s < %8g' % (blanks, min(self['s']), max(self['s'])))
		if 'n' in self.dtype.names:  print('%s  %8g < n < %8g' % (blanks, min(self['n']), max(self['n'])))
		if 'd' in self.dtype.names:  print('%s  %8g < d < %8g' % (blanks, min(self['d']), max(self['d'])))  # delta shift
		if 'm' in self.dtype.names:  # line mixing
			nz  = np.nonzero(self['m'])
			nnz = np.count_nonzero(self['m'])
			print('%s  %8g < m < %8g %10i nonzero with mean %8g' %
			      (blanks, min(self['m']), max(self['m']), nnz, np.mean(self['m'][nz])))
		if 'N' in self.dtype.names:  # Dicke narrowing
			nz  = np.nonzero(self['N'])
			nnz = np.count_nonzero(self['N'])
			print ('%s  %8g < N < %8g %10i nonzero with mean %8g' %
			       (blanks, min(self['N']), max(self['N']), nnz, np.mean(self['N'][nz])))
		if 'sd' in self.dtype.names:  # speed dependence
			nz  = np.nonzero(self['sd'])
			nnz = np.count_nonzero(self['sd'])
			print ('%s  %8g < sd < %7g %10i nonzero with mean %8g' %
			       (blanks, min(self['sd']), max(self['sd']), nnz, np.mean(self['sd'][nz])))

	def __eq__(self, other):
		""" Compare line arrays including its attributes.
		    (For p and od relative differences < 0.1% are seen as 'equal') """
		return self.molec==other.molec \
			   and len(self)==len(other) \
			   and abs(self.t-other.t)<1.0 \
			   and abs(self.p-other.p)<0.001*self.p \
			   and np.allclose(self['v'],other['v'],atol=0.00001,rtol=0.0)  \
			   and np.allclose(self['S'],other['S'],atol=0.0,rtol=0.001)

	def truncate(self, vLimits):
		""" Return a subset of lines within a wavenumber interval `vLimits`."""
		if   isinstance(vLimits,(list,tuple)):               vLimits=Interval(*vLimits)
		elif isinstance(vLimits,(PairOfFloats,PairOfInts)):  vLimits=Interval(vLimits.left,vLimits.right)
		elif isinstance(vLimits,Interval):                   pass
		else:  raise SystemExit ("ERROR --- lineArray.truncate:  expected a interval or pair-of-floats")
		mask = np.logical_and(vLimits.lower<=self['v'], self['v']<=vLimits.upper)
		return self[mask]

	def strong (self, sCut):
		""" Return a subset of strong lines within strength >= `sCut`."""
		mask = self['S']>=sCut
		return self[mask]

	def perturb(self, change, what='S', additive=False):
		""" Return the lines with one of the parameters perturbed by a multiplicative scale factor or additive shift."""
		newLines = self.copy()
		if additive:
			if not (isinstance(change,(int,float)) and abs(change)>0.0):
				raise SystemExit ("ERROR --- lineArray.perturb:  non-zero shift")
			if   what=='S':  newLines['S'] += change
			elif what=='E':  newLines['E'] += change
			elif what=='a':  newLines['a'] += change
			elif what=='s':  newLines['s'] += change
			elif what=='n':  newLines['n'] += change
			elif what=='d':  newLines['d'] += change
			elif what=='m':  newLines['m'] += change
			elif what=='N':  newLines['N'] += change
			else:  raise SystemExit ("ERROR --- lineArray.perturb:  unknown or unsupported line parameter")
		else:
			if not (isinstance(change,(int,float)) and change>0.0):
				raise SystemExit ("ERROR --- lineArray.perturb:  non-positive scaling factor")
			if   what=='S':  newLines['S'] *= change
			elif what=='E':  newLines['E'] *= change
			elif what=='a':  newLines['a'] *= change
			elif what=='s':  newLines['s'] *= change
			elif what=='n':  newLines['n'] *= change
			elif what=='d':  newLines['d'] *= change
			elif what=='m':  newLines['m'] *= change
			elif what=='N':  newLines['N'] *= change
			else:  raise SystemExit ("ERROR --- lineArray.perturb:  unknown or unsupported line parameter")
		return newLines


####################################################################################################################################

def read_line_file (lineFile, xLimits=None, wingExt=0.0,  airWidth=0.1, molecule=None, commentChar='#', verbose=False):
	""" Read a simple line parameter list and return a structured numpy array with data and some attributes.

	    ARGUMENTS:
	    ----------
	    lineFile    a single file or a list thereof
	                (if it is a list of files, a dictionary of line lists will be returned!)
	    xLimits     accept only lines with position in this interval (extended by `wingExt`)
	    airWidth    default air broadening halfwidth, only used when not given in lineFile
	    wingExt     extend the line position interval xLimits (unused otherwise)
	    molecule    molecular name, only used when not given in lineFile

	    RETURNS:
	    --------
            lineArray   a structured numpy array with line position, strengths, etc.
	                and attributes (molecule, pressure, temperature) added by subclassing
			(OR a dictionary of lineArray's for a list of input files)

	    NOTE:
	    -----
	    If you want to read all/several (line) files in a directory, you can use Python's glob function, e.g.
	    dll = read_line_file(glob('*'))       # returns a Dictionary of LineLists (one per file)
	    dll = read_line_file(glob('*.vSEan'))
	"""

	#from glob import glob
	#if '*' in lineFile or '?' in lineFile:  lineFile = glob(lineFile)

	if   isinstance(lineFile, (list,tuple)):
		dictLineLists = {}
		for file in lineFile:
			lineListArray = read_line_file(file, xLimits, wingExt, airWidth, molecule, commentChar, verbose)
			dictLineLists[lineListArray.molec] = lineListArray
		return  dictLineLists

	# read entire line file and return a structured array and a dictionary of attributes
	lines    = loadStrucArray (lineFile, 'position', changeNames=long2short, verbose=verbose)
	# if there is just a single line in the dataset, a 1dim array is returned
	if lines.ndim==0:  raise SystemExit('WARNING --- lines.read_line_file: just one line, needs more work')
	#lines = np.atleast_2d (lines)

	# parse comment header and extract some attributes (returned as dictionary)
	lineAttr = parse_comments (readFileHeader (lineFile, commentChar), ['molecule','gas','format','pressure','temperature'])

	# check if molecule is specified  (and consistent if specified in file and on command line)
	if molecule and 'molecule' in lineAttr:
		if not molecule==lineAttr['molecule']:
			raise SystemExit ('ERROR --- lines.read_line_file:  inconsistent molecule specification!   ' +
			                  repr(molecule) + ' <--> ' + repr(lineAttr['molecule']))
	elif 'gas' in lineAttr:
		lineAttr['molecule'] = lineAttr.pop('gas')
	elif 'molecule' in lineAttr:
		pass
	elif molecule:
		lineAttr['molecule'] = molecule
	else:
		fileRoot = os.path.splitext(lineFile)[0]
		if os.path.sep in fileRoot:  fileRoot = os.path.split(fileRoot)[1]
		if fileRoot in list(molecules.keys()):
			lineAttr['molecule'] = molecule
		else:
			raise SystemExit ('ERROR --- lines.read_line_file:  ' + lineFile + '\nmolecule not specified!' +
			                  '\n(neither in line list header nor root of filename nor as command option)')

	# also need reference pressure and temperature
	if 'temperature' in lineAttr:
		# remove unit spec 'Kelvin'
		lineAttr['temperature'] = float(lineAttr['temperature'].split()[0])
	else:
		raise SystemExit ('ERROR --- lines.read_line_file:  reference temperature of line parameters not given!')

	if 'pressure' in lineAttr:
		try:    # remove unit spec 'millibar' and return pressure in cgs units!
			value,unit = lineAttr['pressure'].split()
			lineAttr['pressure'] = unitConversion(float(value), 'pressure', unit)
		except Exception as errMsg:
			raise SystemExit (str(errMsg) + '\nparsing pressure spec in line file failed ' + repr(lineAttr['pressure']))
	else:
		raise SystemExit ('ERROR --- lines.read_line_file:  reference pressure of line parameters not given!')

	# check if at least position and strengths are found
	if 'v' in lines.dtype.names and 'S' in lines.dtype.names:
		print('\n %-18s %8i lines in  %10f ... %10f cm-1   with  %8.2g < S < %8.2g   (T=%5.1fK)' %
		       (os.path.basename(lineFile), lines.shape[0], min(lines['v']), max(lines['v']),
		        min(lines['S']), max(lines['S']), lineAttr['temperature']))
	else:
		raise SystemExit ('ERROR --- lines.read_line_file:  Need at least line positions and strengths!')

	# lower state energy
	if 'E' in lines.dtype.names:
		if verbose:  print('%s %8g < E < %8g' % (77*' ', min(lines['E']), max(lines['E'])))
	else:
		print('WARNING --- lines.read_line_file:  no lower state energy, no S(T) conversion!')

	# set air broadening parameter to default if not found in line file
	if 'a' in lines.dtype.names:
		if verbose:  print('%s %8g < a < %8g   (p=%gmb)' %
		       (77*' ', min(lines['a']), max(lines['a']), unitConversion(lineAttr['pressure'],'p', new='mb')))
	else:
		lines = strucArrayAddField (lines, np.zeros_like(lines['v']) + airWidth,   'a')
		print('INFO --- lines.read_line_file:  air width initialized to ', airWidth)

	if verbose:
		if 's' in lines.dtype.names:  print('%s %8g < s < %8g' % (77*' ', min(lines['s']), max(lines['s'])))
		if 'n' in lines.dtype.names:  print('%s %8g < n < %8g' % (77*' ', min(lines['n']), max(lines['n'])))
		if 'm' in lines.dtype.names:  print('%s %8g < Y < %8g' % (77*' ', min(lines['m']), max(lines['m'])))  # line Mixing

	if xLimits:
		if isinstance(xLimits,(list,tuple)):  xLimits=Interval(*xLimits)
		# subset of lines requested, truncate lists
		print(' read_line_file:  xLow,xHigh specified: ', xLimits, '  (extension:  +/-', wingExt,')')
		lines   = truncate_lineList (lines,  xLimits+wingExt)

	# All data and attributes defined, finally add attributes to the numpy array by subclassing
	return lineArray (lines, p=lineAttr['pressure'], t=lineAttr['temperature'], molec=lineAttr['molecule'])


####################################################################################################################################

def truncate_lineList (lines, xLimits=None, strMin=0.0):
	""" Remove some lines, e.g. weak lines or at head and/or tail of line list. """
	# setup the mask(s)
	if xLimits and strMin:
		mask = np.logical_and (lines['v']>=xLimits.lower, lines['v']<=xLimits.upper)
		mask = np.logical_and (lines['S']>=strMin, mask)
	elif strMin:
		mask = lines['S']>=strMin
	elif xLimits:
		mask = np.logical_and (lines['v']>=xLimits.lower, lines['v']<=xLimits.upper)

	if sum(mask)<len(lines):
		print(' line data subset:  select', sum(mask), ' of', len(lines), ' lines in', xLimits.lower, xLimits.upper, '\n')
	else:
		print(' INFO --- lines.truncate_line_list:  no lines rejected!')

	return lines[mask]


####################################################################################################################################

def voigt_parameters (lines, molData, pressure=1013.25e3, temperature=296.0, verbose=False):
	""" Convert line strength and Lorentzian width to pressure [g/cm/s^2] and temperature [K] and set Doppler width. """
	strengths = line_strengths (lines.t, temperature,
	                            lines['v'], lines['S'], lines['E'],
	                            molData['NumDeg'], molData['VibFreq'], molData.get('TempExpQR',1.0))
	# Lorentz broadening
	if 'a' in lines.dtype.names and  'n' in lines.dtype.names:
		gammaL, gammaD = line_widths  (lines.p, lines.t, pressure, temperature,
	                                       molData['mass'], lines['v'], lines['a'], lines['n'])
	elif 'a' in lines.dtype.names:
		gammaL, gammaD = line_widths  (lines.p, lines.t, pressure, temperature,
	                                       molData['mass'], lines['v'], lines['a'])
	elif 'n' in lines.dtype.names:
		gammaL, gammaD = line_widths  (lines.p, lines.t, pressure, temperature,
	                                       molData['mass'], lines['v'], tempExp=lines['n'])
	else:
		gammaL, gammaD = line_widths  (lines.p, lines.t, pressure, temperature,
	                                       molData['mass'], lines['v'])

	# pack everything in a structured array
	voigtLines = np.empty(lines.size, dtype={'names': 'v S L D'.split(), 'formats': 4*[np.float]})
	voigtLines['v'] = lines['v']
	voigtLines['S'] = strengths
	voigtLines['L'] = gammaL
	voigtLines['D'] = gammaD

	if verbose:
		print ('\n %s %10.3gmb %s %9.2fK' % ('Voigt line parameters at ', cgs('!mb', pressure), 'and', temperature))
		print (' strengths     %10.2g <= S <= %10.2g' % (min(voigtLines['S']), max(voigtLines['S'])))
		print (' Lorentz width %10.3g <= L <= %10.3g' % (min(voigtLines['L']), max(voigtLines['L'])))
		print (' Gauss width   %10.3g <= D <= %10.3g' % (min(voigtLines['D']), max(voigtLines['D'])))

	# optionally pressure induced line shift
	if 'd' in lines.dtype.names:
		vShift = lines['d']*pressure/lines.p
		voigtLines['v'] += vShift
		if verbose:  print((' mean line shift  %.6f for p/pRef %.3g' % (vShift.mean(), pressure/lines.p)))

	return voigtLines


####################################################################################################################################

def line_strengths (tempRef, temp, positions, strengths, energies, numDeg, vibFreq, tempExpQR=1.0):
	""" Convert line strengths to actual p, T. """
	if abs(tempRef-temp)>0.1:
		if energies.any():
			# ratio of rotational partition function
			ratioQR  =  (tempRef/temp)**tempExpQR
			# ratio of vibrational partition function
			ratioQV  =  vibPartitionFunction (numDeg, vibFreq, tempRef) / vibPartitionFunction (numDeg, vibFreq, temp)
			# Boltzmann factor
			deltaInvTemp = C2 * (temp-tempRef) / (temp*tempRef)
			sb = np.exp(deltaInvTemp*energies)
			# stimulated emission factor
			#se = (1.0 - np.exp(-C2*positions/temp)) / (1.0 - np.exp(-C2*positions/tempRef))
			se = np.expm1(-C2*positions/temp) / np.expm1(-C2*positions/tempRef)
			# multiply all conversion factors with original line strength
			#print 'partition function ratio @ %.1fK:  rot %.3g   vib %.3g   ==>  %.4g' % (temp, ratioQR, ratioQV, ratioQV*ratioQR)
			return strengths * sb * se * ratioQR * ratioQV
		else:
			raise SystemExit ('ERROR:  line strength temperature conversion impossible, no lower state energies!')
	else:
		return strengths


def  vibPartitionFunction (degeneracy, omega, temperature):
	""" Calculate vibrational partition function as a function of temperature.
	    Norton & Rinsland: ATMOS Data Processing and Science Analysis Methods; Appl. Opt. 30,389(1991)

	    Harmonic oscillator approximation, see Eq. (7) in Bob Gamache TIPS http://dx.doi.org/10.1016/j.jqsrt.2017.03.045
	    """
	c2T        = C2 / temperature
	factors    = 1.0 / (1.0 - np.exp(-c2T*omega))**degeneracy
	qVib      = np.product(factors)
	return qVib


####################################################################################################################################

def line_widths (pressRef, tempRef, press, temp, mass, positions, airWidths=0.1, tempExp=0.5):
	""" Convert pressure (air) broadening and set Doppler broadening half widths to actual p, T. """

	# Lorentzian half widths: only air broadening (self broadening is ignored!)
	if isinstance(airWidths,float):
		print('Air widths initializing to ', airWidths)
		airWidths = np.zeros_like(positions) + airWidths
	if isinstance(tempExp,float):
		print('Air width temperature exponent initializing to ', tempExp)
		tempExp = np.zeros_like(positions) + tempExp
	gammaL = airWidths * (press/pressRef) * (tempRef/temp)**tempExp

	# Gaussian half widths (note: molecular mass in atomic mass units!)
	gammaD = positions * sqrt(2.*ln2*k*temp/(mass*amu*c*c))

	return gammaL, gammaD


####################################################################################################################################

def write_strengths (positions, strengths, strRef, temperature, tempRef, molecule, outFile=None, commentChar='#'):
	""" Write line strengths (for two temperatures) vs line positions. """
	out = open_outFile (outFile, commentChar)
	out.write ('%s %s %s\n' % (commentChar, "molecule:", molecule))
	out.write ('%s %s %8.2f %s\n' % (commentChar, "temperature T_ref:" , tempRef, "K"))
	out.write ('%s %s %8.2f %s\n' % (commentChar, "temperature T:    ", temperature, "K"))
	out.write ('%s %10s %23s\n' % (commentChar, "position", "   S(T_ref)    |S(T)- S(T_ref)|"))
	frmt = '%12f  %11.3e %10.2e 0\n'
	for v,S0,S in zip(positions, strRef, strengths):
		#out.write ( frmt % (v,S0,S) )
		if S>S0: out.write (frmt % (v,S0,S-S0))
		else:    out.write (frmt % (v,S0,S0-S))
	# close the output file (if its not stdout)
	if outFile: out.close()


def write_voigtLines (voigtLines, pressure, temperature, molecule, outFile=None, commentChar='#'):
	""" Write Voigt line parameters (strengths, Lorentz and Gaussian widths vs line positions. """
	out = open_outFile (outFile, commentChar)
	out.write ('%s %s %s\n' % (commentChar, "molecule:", molecule))
	out.write ('%s %s %-12g\n' % (commentChar, "pressure  [mb]:     " ,  unitConversion(pressure,'p', new='mb')))
	out.write ('%s %s %8.2f\n' % (commentChar, "temperature  [K]: ", temperature))
	out.write ('%s %10s  %11s %11s %11s\n' % (commentChar, "position", "strength", "gammaL", "gammaG"))
	frmt = '%12f  %11.3e %11.3g %11.3g\n'
	for line in voigtLines:  out.write (frmt % tuple(line))
	# close the output file (if its not stdout)
	if outFile: out.close()


####################################################################################################################################

def meanLineWidths (gammaL, gammaD):
	""" Evaluate mean pressure, Doppler, and combined broadening half width, return mean Voigt width. """
	# Voigt width (Whiting approximation, 1% accuracy)
	gammaV = 0.5 * (gammaL + np.sqrt(gammaL**2 + 4.*gammaD**2))
	# averages
	nLines = len(gammaL)
	meanGL = np.sum(gammaL)/nLines
	meanGD = np.sum(gammaD)/nLines
	meanGV = np.sum(gammaV)/nLines
	#print  ' mean width (L, G, V): %10g %10g %10g   y %8g' % (meanGL,meanGD,meanGV, sqrtLn2*meanGL/meanGD)
	return  meanGL, meanGD , meanGV


####################################################################################################################################

def atlas (lines, yType='S', split=False):
	""" matplotlib implementation of line parameter 'atlas', wavenumber vs strength or width or ... .

	ARGUMENTS:
	----------
	lines    a single lineArray, or a dictionary or list of lineArray's
	yType    character to select line parameter to plot on y axis, default "S" (strength)
	split    boolean flag, use subplots for individual molecules
	"""
	yText = {'S': r'Strength $S \rm\,[cm^{-1} / (molec.cm^{-2})]$',
	         'a': r'Air Width $\gamma_a \rm\,[cm^{-1}]$',
	         's': r'Self Width $\gamma_s \rm\,[cm^{-1}]$',
	         'n': r'Temperature Exponent $n$',
	         'L': r'Lorentz Width $\gamma_L \rm\,[cm^{-1}]$',
	         'E': r'Energy $E \rm\,[cm^{-1}]$',
	         'd': r'shift  $\delta \rm\,[cm^{-1}/atm]$',
	         'm': r'mix  $y \rm\,[1/atm]$',
	         'N': r'Dicke narrowing $\Omega$',
	         'sd': r'Speed dependence air broadening'
		 }
	if len(lines)==0:  raise SystemExit ('WARNING --- lines.atlas:  no lines!')

	if isinstance(lines, dict):
                # call atlas recursively
		if split:  nPlots=0
		for data in lines.values():
			if verbose:  print (data.molec, min(data['v']), max(data['v']), min(data['S']), max(data['S']))
			if split:
				nPlots+=1
				subplot(len(lines),1,nPlots)
				if nPlots==len(lines)/2:  ylabel (yText.get(yType,' '))
			atlas (data, yType, split)
			if split:  title(data.molec)
			### new
			if split:
				xLow, xHigh = xlim()
				print ('%3i. subplot with automated xlim %15.5f ... %.5f' % (nPlots, xLow, xHigh), end=' ')
				if nPlots>1:
					xMin = min(xMin, xLow)
					xMax = max(xMax, xHigh)
					xlim (xMin,xMax);  print ('  --->  corrected to ', xMin, xMax)
				else:
					xMin, xMax = xLow,xHigh;  print ()
	elif isinstance(lines, (list,tuple)):
		if split:
			raise SystemExit ("WARNING --- atlas:  splitted atlas not (yet?) supported for lists!")
		for data in lines:
			atlas (data, yType)
	elif isinstance(lines, lineArray):
		info = '%-8s   p=%.1g   T=%.1f  %i' % (lines.molec,lines.p,lines.t, len(lines))
		if yType not in lines.dtype.names:
			raise SystemExit ("ERROR -- atlas:  invalid line parameter " + yText[yType])
		if yType in 'SE':  semilogy (lines['v'], lines[yType], '+', label=info)
		else:              plot     (lines['v'], lines[yType], '+', label=info)
		if not split:      ylabel (yText.get(yType,' '))
	elif isinstance(lines, np.ndarray) and lines.dtype:  # this is also true for lineArray !!!
		if 'v' in lines.dtype.names and yType in lines.dtype.names:
			if yType in 'SE':  semilogy (lines['v'], lines[yType], 'x')
			else:              plot     (lines['v'], lines[yType], 'x')
	else:
		raise SystemExit ("ERROR --- atlas:  first argument 'lines' neither a lineArray nor a dictionary thereof")

	xlabel (r'position $\hat\nu \rm\,[cm^{-1}]$')
	if not split:
		legend(loc='best', frameon=False, fontsize='xx-small')


####################################################################################################################################

def delete_traceGasLines (dictOfLineLists):
	""" From a dictionary of core parameter line lists, remove all trace gases, i.e. return the main gases only. """
	if isinstance(dictOfLineLists,dict):
		nAll = len(dictOfLineLists)
		for mol in list(dictOfLineLists.keys()):
			if mol not in mainMolecules:  del dictOfLineLists[mol]
		print('main gases only:  deleted ', nAll-len(dictOfLineLists), ' of ', nAll, ' linelists')
		return dictOfLineLists
	else:
		raise SystemExit ('ERROR --- delete_traceGasLines:  need a dictionary(!) of line parameter arrays')


####################################################################################################################################

def _lines_ (lineFiles, outFile='', commentChar='#',  pressure=None, temperature=None, xLimits=None, airWidth=0.1,
	     lineAtlas='S', verbose=False):
	""" Read a couple of lines from vSEan file(s) and convert to new pressure|temperature and/or plot|save data."""

	# get the line data
	dictOfLineLists = read_line_file (lineFiles, xLimits, airWidth,  commentChar=commentChar, verbose=verbose)

	# extract the attributes
        #lineAttr = dict([(key,dictOfLineLists.pop(key)) for key in ['pressure','temperature']])

	#if abs(pressure-lineAttr['pressure'])/lineAttr['pressure']>0.01 and abs(temperature-lineAttr['temperature'])>1.0:
	if isinstance(pressure,(int,float)) or isinstance(temperature,(int,float)):
		if pressure: pressure=unitConversion(pressure,'p', old='mb')

		for molec,data in list(dictOfLineLists.items()):
			if not molec==data.molec:
				raise SystemExit ('%s %s %s' % ('ERROR --- _lines__:  inconsistent molec ', molec, data.molec))
			# read molecular data (mass etc.)
			molData    = get_molec_data (molec)
			if verbose:  print(molec, molData)

			# adjust line parameters to p, T
			if not pressure:
				pressure=data.p
				print(' INFO --- _lines_:  set pressure to line reference pressure', data.p, data.molec)
			if not temperature:
				temperature=data.t
				print(' INFO --- _lines_:  set temperature to line reference temperature', data.t, data.molec)
			voigtLines = voigt_parameters (data, molData, pressure, temperature, verbose)
			if verbose:  print('%10s %10.2g <S<%10.2g' % (molec, min(voigtLines['S']),max(voigtLines['S'])))

			# plot line strength vs position
			if   lineAtlas=='T': vlines (voigtLines['v'], data['S'], voigtLines['S'], label=molec)
			elif lineAtlas: semilogy (voigtLines['v'], voigtLines['S'], '+', label=molec)
			ylabel (r'$S(T)$')

			# optionally save converted line data
			if isinstance(outFile,str):
				write_voigtLines (voigtLines, pressure, temperature, molec, molec+outFile, commentChar)

		# annotate plot
		if lineAtlas:  title (r'$p$=%.2edyn/cm**2 \qquad $T$=%.1fK' % (pressure, temperature))
	else:
		if not lineAtlas=='T':  atlas (dictOfLineLists, yType=lineAtlas)

	if lineAtlas:
		legend();  show()


####################################################################################################################################

if __name__ == "__main__":

	from command_parser import parse_command, standardOptions

        # parse the command, return (ideally) one file and some options
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       dict(ID='about'),
               dict(ID='a', name='airWidth', type=float, constraint='airWidth>0.0'),
               dict(ID='m', name='molecule', type=str),
	       # option called "atlas" to avoid conflict with matplotlib function "plot"
               dict(ID='plot', name='lineAtlas', type=str, default='S', constraint='lineAtlas in "STEansdL"'),
               dict(ID='p', name='pressure', type=float, constraint='pressure>0.0'),
               dict(ID='T', name='temperature', type=float, constraint='temperature>0.0'),
	       dict(ID='x', name='xLimits', type=Interval, constraint='xLimits.lower>=0.0'),
	       dict(ID='v', name='verbose')
               ]

	lineFiles, options, commentChar, outFile = parse_command (opts,(1,99))

	if 'h'     in options:  raise SystemExit (__doc__ + "\n end of lines help")
	if 'about' in options:  raise SystemExit (_LICENSE_)

	options['verbose'] = 'verbose' in options

	if 'lineAtlas' in options:
		figure()

	_lines_ (lineFiles, outFile, commentChar, **options)

#!/usr/bin/env python3

""" hitran

 usage:
 hitran [options] line_parameter_database

 command line options:
   -h  help
   -i  int       isotope number  (default: all)
   -m  int       molecule number (default: all)
   -o  string    output file (default: standard output)
   -S  float     minimum line strength to accept (default: S=0.0)
   -x  interval  lower and upper end of wavenumber range (comma separated pair without blanks)

 Notes:   at least wavenumber range or molecule has to be selected!
          this is the 'low-level' version of the extract module
"""

##############################################################################################################
#####  LICENSE issues:                                                                                   #####
#####                 This file is part of the Py4CAtS package.                                          #####
#####                 Copyright 2002 - 2019; Franz Schreier;  DLR-IMF Oberpfaffenhofen                   #####
#####                 Py4CAtS is distributed under the terms of the GNU General Public License;          #####
#####                 see the file ../license.txt in the parent directory.                               #####
##############################################################################################################

#from string import *
from time   import clock

from exojax.aux.hitran.pairTypes import Interval

####################################################################################################################################

def bisect_first_line (db, xBegin, xEnd, iw=3,lw=15):
	""" Search and return first line of Hitran or Geisa formatted database. """
	# skip optional header (comment) section
	if 'hit' in db.name.lower():
		# hitran database, skip header section with comment lines (indicated with mol=00)
		mol  = -1
		try:
			while mol<1:  record=db.readline(); mol = int(record[:2])
		except ValueError as msg:
			if len(record.rstrip()) not in (100,160):
				print('Hitran type database, but length of first data record is not 100 or 160')
			raise SystemExit (str(msg) + '\nERROR reading header section of Hitran-type database (trying to parse molec id number)')
	else:
		# hitran database, skip header section with comment lines (indicated with mol=00)
		record=db.readline()

	# first data record with spectroscopic line transition in file:  parse it carefully!
	lenRec   = len(record)
	locFirst = db.tell() - lenRec
	try:
		recFirst = record
		xFirst   = float(recFirst[iw:lw])
	except ValueError as msg:
		if 'hit' in db.name.lower() and not len(record.rstrip()) in (100,160):
			print('Hitran type database, but length of first data records is not 100 or 160')
		elif 'geisa' in db.name.lower() and len(record.rstrip()) in (100,160):
			print('Geisa type database, but length of first data records is 100 or 160 (looks like Hitran)')
		raise RuntimeError ('ERROR --- hitran.bisect_first_line: reading first data record of database (trying to parse wavenumber)\n' + str(msg))

	# check file consistency: total number of bytes should be an multiple integer (= number of records) of (first) record length
	db.seek(0,2);  nBytes=db.tell()
	if nBytes%lenRec>0:
		print(lenRec, locFirst, nBytes, nBytes%lenRec)
		raise SystemExit ('%s %s %i %s %i %s'% (' ERROR --- hitran:  file corrupt???  Inconsistent record length!?!',
		                  '\n total number of bytes', nBytes,
				  '\n len(firstRecord)', lenRec,
				  '\n nBytes should be a integer multiple of the record length (nBytes/lenRec=numLines)'))

	# move to last record in file
	db.seek(-lenRec,2)
	locLast = db.tell()
	recLast = db.readline()
	xLast   = float(recLast[iw:lw])

	# check spectral range
	if  xFirst <= xBegin < xEnd <= xLast:
		pass
	elif  xFirst>xEnd or xLast<xBegin:
		raise SystemExit (' ERROR --- hitran.bisect_first_line: requested spectral range not in database ' \
		                  + str(xFirst) + ' --- ' + str(xLast))
	else:
		if xFirst>1.0 or xLast<20000.0:  # apparently a subset of the full hitran database
			print('%s %s %f --- %f %s %f --- %f' % (' WARNING --- hitran.bisect_first_line:  ',
			      'requested spectral range ', xBegin, xEnd, '    only partly in database ', xFirst, xLast))

	# record number of very first and last lines (locFirst, locLast are byte numbers)
	lineFirst = locFirst//lenRec
	lineLast  = (locLast-locFirst)//lenRec

	# bisecting to desired first line
	while lineLast-lineFirst > 1:
		mid=(lineLast+lineFirst)//2
		db.seek(locFirst+mid*lenRec)
		recMid=db.readline()
		if not len(recMid)==lenRec:
			raise SystemExit ('%s %s %i %s %i %s %i\n%s'% (' ERROR --- hitran:  file corrupt???  Inconsistent record length',
			                  '\n len (firstRecord)', lenRec,
					  '\n len (currentRecord)', len(recMid), ' at record #', mid, recMid))
		xMid = float(recMid[iw:lw])
		#print locFirst,locLast,lineFirst,lineLast,lenRec,lineLast*lenRec,(locLast-locFirst)//lenRec,mid*lenRec, xMid
		if   xBegin<xMid: lineLast =mid; xLast=xMid
		elif xBegin>xMid: lineFirst=mid; xFirst=xMid
		else:
			# backstep: there are possibly several lines at exactly this position
			while 1:
				db.seek(-2*lenRec,1); rec=db.readline(); xMid=float(rec[iw:lw])
				if xMid<xBegin:
					print("# first line in spectral range at record ", mid, "found in", clock(), "sec\n", rec[:67])
					return        db.readline()

	if    xMid<xBegin:      record = db.readline()
	else:                   record = recMid
	print("# first line in spectral range at record number", mid, "found in", clock(), "sec\n", record[:67])

	return record

####################################################################################################################################

def extract_All (hitran, xBegin, xHigh):
	""" Read all lines up to an upper wavenumber limit from Hitran formatted database. """
	# proceed to first requested line
	record = bisect_first_line (hitran, xBegin, xHigh)
	# initialize list if lines
	lines = []
	# collect lines
	while record:
		mol    = int(record[:2])
		if mol>0:
			wvn = float(record[3:15])
			if wvn<=xHigh:  lines.append(record)
			else:           break
		# read next record
		record = hitran.readline()

	if len(lines)>0:  print('# last  line     accepted \n', lines[-1][:67])
	if record:        print('# first line not accepted \n', record[:67])     # empty string returned at end-of-file

	return lines

####################################################################################################################################

def extract_Mol (hitran, xBegin,xHigh, getMol):
	""" Read lines of a given molecule up to an upper wavenumber limit from Hitran formatted database. """
	# proceed to first requested line
	record = bisect_first_line (hitran, xBegin, xHigh)
	# initialize list if lines
	lines = []
	# collect lines
	while record:
		mol = int(record[:2])
		wvn = float(record[3:15])
		if wvn>xHigh: break
		if mol==getMol: lines.append(record)
		# read next record
		record = hitran.readline()

	if len(lines)>0:  print('# last  line     accepted \n', lines[-1][:67])
	if record:        print('# first line not accepted \n', record[:67])     # empty string returned at end-of-file

	return lines

####################################################################################################################################

def extract_MolIso (hitran, xBegin,xHigh, getMol, getIso):
	""" Read lines of a given molecule/isotope up to an upper wavenumber limit from Hitran formatted database. """
	# proceed to first requested line
	record = bisect_first_line (hitran, xBegin, xHigh)
	# initialize list if lines
	lines = []
	# collect lines
	while record:
		mol = int(record[:2])
		iso = int(record[2:3])
		wvn = float(record[3:15])
		if wvn>xHigh: break
		if mol==getMol and iso==getIso: lines.append(record)
		# read next record
		record = hitran.readline()

	if len(lines)>0:  print('# last  line     accepted \n', lines[-1][:67])
	if record:        print('# first line not accepted \n', record[:67])     # empty string returned at end-of-file

	return lines

####################################################################################################################################

def extract_MolStr (hitran, xBegin,xHigh, getMol, strMin):
	""" Read strong lines of a given molecule up to an upper wavenumber limit from Hitran formatted database. """
	# proceed to first requested line
	record = bisect_first_line (hitran, xBegin, xHigh)
	# initialize list if lines
	lines = []
	# collect lines
	while record:
		mol = int(record[:2])
		wvn = float(record[3:15])
		Str = float(record[15:25])
		if wvn>xHigh: break
		if mol==getMol and Str>=strMin: lines.append(record)
		# read next record
		record = hitran.readline()

	if len(lines)>0:  print('# last  line     accepted \n', lines[-1][:67])
	if record:        print('# first line not accepted \n', record[:67])     # empty string returned at end-of-file

	return lines

####################################################################################################################################

def extract_MolIsoStr (hitran, xBegin,xHigh, getMol, getIso, strMin):
	""" Read strong lines of a given molecule/isotope up to an upper wavenumber limit from Hitran formatted database. """
	# proceed to first requested line
	record = bisect_first_line (hitran, xBegin, xHigh)
	# initialize list if lines
	lines = []
	# collect lines
	while record:
		mol = int(record[:2])
		iso = int(record[2:3])
		wvn = float(record[3:15])
		Str = float(record[15:25])
		if wvn>xHigh: break
		if mol==getMol and iso==getIso and Str>=strMin:  lines.append(record)
		# read next record
		record = hitran.readline()

	if len(lines)>0:  print('# last  line     accepted \n', lines[-1][:67])
	if record:        print('# first line not accepted \n', record[:67])     # empty string returned at end-of-file

	return lines


####################################################################################################################################

def extract_hitran (hFile, xLimits=None, molNr=0, isoNr=0, strMin=0.0):
	""" Read Hitran formatted database, return list of accepted lines. """

	try:
		hitran = open (hFile,'rb')  # open in binary mode, otherwise backward-seek in bisection fails!
	except IOError:
		raise SystemExit ('ERROR --- hitran:  opening Hitran data file "' + hFile + '" failed!')

	# wavenumber interval to be searched
	if isinstance(xLimits,Interval):  xBegin, xHigh = xLimits.limits()
	else:                             xBegin, xHigh = 0.0, 99999.9

	# initialize time and search first useful line
	if molNr<=0:
		if isoNr>0 or strMin>0:
			raise SystemExit ('ERROR --- hitran:  no isotope or linestrength selection without molecule specification!')
		lines = extract_All (hitran, xBegin,xHigh)
	elif not (isoNr or strMin):
		lines = extract_Mol (hitran, xBegin,xHigh, molNr)
	elif isoNr==0 and strMin>0.0:
		lines = extract_MolStr (hitran, xBegin,xHigh, molNr, strMin)
	elif strMin<=0.0:
		lines = extract_MolIso (hitran, xBegin,xHigh, molNr, isoNr)
	else:
		lines = extract_MolIsoStr (hitran, xBegin,xHigh, molNr, isoNr, strMin)

	hitran.close()
	return lines


####################################################################################################################################

if __name__ == "__main__":
	from exojax.aux.hitran.aeiou import open_outFile

	from exojax.aux.hitran.command_parser import parse_command, standardOptions
	opts = standardOptions + [  # h=help, c=commentChar, o=outFile
	       {'ID': 'i', 'name': 'isoNr',  'type': int, 'default': 0},
               {'ID': 'm', 'name': 'molNr', 'type': int, 'default': 0},
               {'ID': 'S', 'name': 'strMin', 'type': float,   'default': 0.0, 'constraint': 'strMin>=0.0'},
	       {'ID': 'x', 'name': 'xLimits', 'type': Interval, 'default': Interval(0.0,99999.9), 'constraint': 'xLimits.lower>=0.0'}
               ]

	files, options, commentChar, outFile = parse_command (opts, 1)

	for opt in opts:
		if 'name' in opt and 'type' in opt:  exec(opt['name'] + ' = ' + repr(options.get(opt['name'])))

	if 'h' in options:
		raise SystemExit (__doc__ + "\n end of hitran help")
	elif options.get('molNr',0) or options['xLimits'].size()<50000.0:
		# Read lines from HITRAN line parameter file
		lines = extract_hitran (files[0], **options)

		if len(lines)>0:
			print(len(lines), ' lines')
			# open an output file if explicitely specified (otherwise use standard output)
			out = open_outFile (outFile, commentChar='00')
			# print extracted lines
			for line in lines: out.write (line)
			# close the output file (if its not stdout)
			if outFile: out.close()
	else:
		# at least molecule or wavenumber range needed
		raise SystemExit (' ERROR: neither molecule nor wavenumber range specified!')

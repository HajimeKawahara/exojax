""" aeiou   --- advanced extended input output utilities:

awrite           Write (a) numeric array(s) to file (or fileObject, or stdout if unspecified).
cstack           Shorthand robust version of numpy.column_stack: 'paste' arrays side-by-side.
...
and many more functions.
"""

##############################################################################################################
#####  LICENSE issues:                                                                                   #####
#####                 This file is part of the Py4CAtS package.                                          #####
#####                 Copyright 2002 - 2019; Franz Schreier;  DLR-IMF Oberpfaffenhofen                   #####
#####                 Py4CAtS is distributed under the terms of the GNU General Public License;          #####
#####                 see the file ../license.txt in the parent directory.                               #####
##############################################################################################################

import os
import sys
from string import ascii_lowercase
import re
from io import IOBase  # isinstance(out,file) ---> isinstance(out,IOBase)

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numeric python failed!')

from exojax.aux.hitran.cgsUnits import unitConversion
from exojax.aux.hitran.pairTypes import PairOfInts
from exojax.aux.hitran.misc import xTruncate


####################################################################################################################################

def parse_comments (commentLines, keywords, sep=':', commentChar='#'):
	""" Scan thru list of records (typically read as file header) and search for certain keys;
	    returns a dictionary with (ideally) len(keywords) key,value(s) pairs
	    If units are specified in the key (like "name [unit]: value(s)),
	    the value(s) are converted to the standard cgs unit (e.g. mb -> g/cm/s**2). """

	# if there is just one keyword (given as a string) put it into a list nevertheless
	if isinstance(keywords, str):  keywords=[keywords]
	# get rid of leading or trailing blanks in keywords
	keywords = [keyword.strip() for keyword in keywords]

	# initialize dictionary to be returned
	hDict = {}

	# loop over all file header lines
	for record in commentLines:
		# get rid of comment character(s) and leading blanks
		record = re.sub ('^'+commentChar+'* *','',record)
		if record.count(sep)==0: continue
		key,val = record.split(sep,1)
		#print '\nkey=', repr(key), '\nval=', repr(val)

		# check if there is a unit specification [in square brackets]
		mo = re.search(r'\[.*\]',key)
		if mo:
			# only use first part as key without units
			name   = key[:mo.start()].strip()
			unit   = key[mo.start()+1:mo.end()-1]
			print('name: ', repr(name), '    unit: ', repr(unit))

			for keyword in keywords:
				if keyword==name:
					try:
						values = unitConversion (np.array(list(map(float,val.split()))), name, unit)
						hDict[name] = values
					except ValueError as errMsg:
						raise SystemExit ('ERROR --- parse_comments:  failed to parse comment line ' + repr(key)
						              + '\n                           (units given, but no numbers follow)\n' + errMsg)
		else:
			# no units given, hope values are in cgs units
			for keyword in keywords:
				if keyword==key:
					try:
						values = np.array(list(map(float,val)))
					except ValueError:
						hDict[key.strip()] = val.strip()  # simply return data as is
	return hDict


####################################################################################################################################

def join_words (wordList, sep=' '):
	""" Join / concatenate list or tuple of 'words' (not in a strict grammar sense) and return a single string.

	    (This function should work similar to the string.join function of Python 2) """

	return sep.join(wordList)


####################################################################################################################################

def grep_word (inFile, firstWord, commentChar='#'):
	""" Scan thru header section of file and search for a comment line starting with <firstWord>;
	    Return list of words. """
	# check file, open for read if not yet done
	if   isinstance(inFile, str):
		try:                     fo = open(inFile)
		except IOError as errMsg:  raise SystemExit (str(errMsg) + '\ncheck your input file!  ' + repr(inFile))
	elif isinstance(inFile, IOBase):
		fo = inFile
	else:
		raise SystemExit ('ERROR --- aeiou.grep_word:  need either a file name or a file object')
	# scan thru records and search for a line starting with "# keyword:"
	firstWord = firstWord.strip()
	while True:
		record   = fo.readline().strip()
		if len(record)==0: continue
		if not record.startswith(commentChar): break
		# get rid of comment character(s) and leading blanks
		record = re.sub ('^'+commentChar+'* *','',record)
		words = record.split()
		if len(words)==0:  continue
		if words[0]==firstWord: return words


####################################################################################################################################

def grep_from_header (inFile, keyword, sep=':', commentChar='#'):
	""" Scan thru list of records (typically read as file header), search for ONE keyword, and return its 'value'.
	    (Equivalent to parse_comments (readFileHeader(file),keyword)[keyword], but returns only the entry). """
	# check file, open for read if not yet done
	if   isinstance(inFile, str):
		try:                       fo = open(inFile)
		except IOError as errMsg:  raise SystemExit (str(errMsg) + '\ncheck your input file!  ' + repr(inFile))
	elif isinstance(inFile, IOBase):
		fo = inFile
	else:
		raise SystemExit ('ERROR --- aeiou.grep_from_header:  need either a file name or a file object')

	# scan thru records and search for a line starting with "# keyword:"
	keyword = keyword.strip()
	while True:
		record   = fo.readline().strip()
		if len(record)==0: return
		# get rid of comment character(s) and leading blanks
		record = re.sub ('^'+commentChar+'* *','',record)
		if record.count(sep)==0: continue
		key,val = record.split(sep,1)
		# comparison of given and wanted keyword case insensitive!
		if key.upper().find(keyword.upper())>-1:
			# check if there is a unit specification [in square brackets]
			mo = re.search(r'\[.*\]',key)
			if mo:
				# only use first part as key without units
				#name   = key[:mo.start()].strip()
				values = val.strip().split()
				unit   = key[mo.start()+1:mo.end()-1]
				return (np.array(list(map(float,values))), unit)
			else:
				return val.strip()


def grep_array_from_header (inFile, keyword, keyValSep=':', commentChar='#', intType=0):
	""" Scan thru list of records (typically read as file header), search for ONE keyword, and return data as numpy array. """
	stringOfData = grep_from_header (inFile, keyword, keyValSep, commentChar)
	if intType:
		data = np.fromstring(stringOfData, sep=' ', dtype=int)
	else:
		data = np.fromstring(stringOfData, sep=' ')
	return data


####################################################################################################################################


def getCommentLines (inFile, commentChar='#'):
	""" Read a tabular (xy) formatted ascii file and return list of ALL comments (without commentChar) found in file. """
	if   isinstance(inFile, str):
		try:                       fo = open(inFile)
		except IOError as errMsg:  raise SystemExit ('ERROR --- getCommentLines:  check file specification\n' + str(errMsg))
	elif isinstance(inFile, IOBase):
		fo = inFile
		if fo.tell()>0:  fo.seek(0)  # rewind to start search at begin of file
	else:
		raise SystemExit ('ERROR --- getCommentLines:  need either a file name or a file object')
	# read ALL lines/records
	records = (rec.strip() for rec in fo)
	# filter comments lines and remove leading commentChar
	comments = (re.sub ('^'+commentChar+'* *','',rec) for rec in records if rec.startswith(commentChar))
	return list(comments)


def readFileHeader (inFile, commentChar='#'):
	""" Read a tabular (xy) formatted ascii file and return list of comments (without commentChar) found in header.
	    Stops reading when the first non-comment line is found. """
	# check file, open for read if not yet done
	if   isinstance(inFile, str):
		try:                     fo = open(inFile)
		except IOError as errMsg:  raise SystemExit ('ERROR --- readFileHeader:  check file specification\n' + str(errMsg))
	elif isinstance(inFile, IOBase):
		fo = inFile
		if fo.tell()>0:  fo.seek(0)  # rewind to start search at begin of file
	else:
		raise SystemExit ('ERROR --- readFileHeader:  need either a file name or a file object')
	# initialize list of comment and read first line
	comments = []
	record   = fo.readline().strip()
	# loop over file header and move all comment records to a separate list
	while record.startswith(commentChar):
	        # get rid of comment character(s) and leading/tailing blanks
		record = re.sub ('^'+commentChar+'* *','',record)
		if len(record)>0:  comments.append(record)
		record   = fo.readline().strip()
	fo.close()
	return comments


####################################################################################################################################

def readDataAndComments (inFile, commentChar='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False):
	""" Read tabular (xy) formatted ascii file, return data as numpy array and list of comments in file header.
	    (Most options are simply passed thru to numpy.loadtxt) """
	if not os.path.isfile(inFile):
		raise SystemExit ('ERROR --- readDataAndComments:  input file not existing!?!\n' + repr(inFile))
	# note different naming convention for comment character
	comments = readFileHeader (inFile, commentChar)
	try:
		data     = np.loadtxt (inFile, comments=commentChar, delimiter=delimiter,
		                       converters=converters, skiprows=skiprows, usecols=usecols, unpack=unpack)
	except ValueError as msg:
		raise SystemExit( 'ERROR --- readDataAndComments:  reading (numeric) data failed\n' + str(msg))
	return data, comments


####################################################################################################################################

def cstack (*arrays):
	""" Shorthand robust version of numpy.column_stack: 'paste' arrays side-by-side.

            See also the numpy builtin c_
	"""
	if len(arrays)==1 and isinstance(arrays[0], (list,tuple)):
		arrays=arrays[0]  # 'oldfashioned' cstack call with explicit tuple/list of arrays
	try:
		return np.column_stack(arrays)
	except ValueError as msg:
		arrayShapes = str([a.shape for a in arrays])
		raise SystemExit ('%s\n%s\n%s' % ('ERROR --- aeiou.cstack:  array dimensions mismatch', msg, arrayShapes))


####################################################################################################################################

def commonExtension (files):
	""" Return the common extension of all files, if identical; otherwise return None. """
	extensions = [os.path.splitext(file)[1] for file in files]
	ext0 = extensions[0]
	for ext in extensions:
		if not ext==ext0: return
	return ext0


####################################################################################################################################

def open_outFile (outFile, commentChar='#'):
	""" Open output file and write job  specification (command line). """
	if outFile:
		try:
			out = open(outFile,'w')
		except IOError as errMsg:
			raise SystemExit (str(errMsg) + '\nERROR --- opening output file failed!')
		# print command line as very first record to file header
		if not (sys.argv[0].endswith('python') or 'ipython' in join_words(sys.argv)) and out.tell()<1 and not out.isatty():
			out.write (commentChar + ' ' + get_command() + '\n' + commentChar + '\n')
	else:
		out = sys.stdout
	return out


####################################################################################################################################

def get_command (maxArgs=13):
	""" Return a string with the command line used (optionally truncated). """
	# remove head, i.e. retrun only the last pathname component
	sys.argv[0] = os.path.basename(sys.argv[0])
	# get rid of too many arguments
	if len(sys.argv)<maxArgs: sysArgv = join_words(sys.argv)
	else:                     sysArgv = join_words(sys.argv[:maxArgs]) + ' ....'
	return sysArgv


####################################################################################################################################

def minmaxmean(xy, name='xy'):
	""" Print some 'statistics' of a numpy array. """
	if not isinstance(xy, np.ndarray):
		raise SystemExit ("ERROR --- minmaxmean:  not a numpy (numeric) array!")
	if len(xy.shape)>1:
		for j in range(xy.shape[1]):
			print(' %12.6g <= %s[:,%2.2i] <= %-12.6g   %s %11.5g   %s %12.6g' %
			      (min(xy[:,j]), name, j, max(xy[:,j]), 'mean', np.mean(xy[:,j]), 'norm', np.linalg.norm(xy[:,j])))
	else:
			print(' %12.6g <= %s <= %-12.6g   %s %11.5g   %s %12.6g' %
			      (min(xy), name, max(xy), 'mean', np.mean(xy), 'norm', np.linalg.norm(xy)))


####################################################################################################################################

def awrite (data, outFile=None, format='%g ', comments=None, append=False, commentChar='#'):
	""" Write (a) numeric array(s) to file (or fileObject, or stdout if unspecified).

	    data:         a numpy array or a list/tuple of numpy arrays (with consistent number of rows, i.e., identical first shape)
	    outFile:      filename or file object
	                  if unspecified: write to stdout
		          if an already opened file object, awrite does not close the file after writing
	    format:       must have one, two, or data.shape[1] specifiers  (line feed is appended if necessary)
	    comments:     a (list of) string(s) to be written as a file header in front of the data
	    append:       flag
	    commentChar:  default #

	    [awrite is similar to numpy.savetxt, but smarter, more flexible, ....
	     Note that the output file (object) is optional here, hence the second argument!]
	    """
	# check output file name, it it looks like a format specification
	if isinstance(outFile, np.ndarray):
		raise SystemExit (' ERROR --- awrite:  second argument `outFile` looks like a numpy array\n' +
		                  '                    maybe you gave several arrays and forgot to put all in a list!?!')
	elif isinstance(outFile, str) and outFile.startswith('%') \
	                           and ('f' in outFile or 'e' in outFile.lower() and 'g' in outFile.lower()):
		format  = outFile
		outFile = None
		print(' WARNING --- awrite:  second argument `outFile` looks like a format specification\n',
		   '                         reassigning this to format and resetting outFile=None')

	# check file status and open the file (if unspecified: use stdout)
	if outFile:
		if isinstance(outFile, str):
			if outFile.startswith('*') or outFile.startswith('?'):
				raise SystemExit ('ERROR --- awrite:  output file name starts with Unix/Linux wildcard')
			if outFile.startswith('.') and not outFile.startswith('../'):
				print('WARNING --- awrite:  output file name starts with single period,\n',
				      '                     file probably hidden in the directory listing!?!')
			if append:  out=open(outFile,'a')
			else:       out=open(outFile,'w')
		elif isinstance(outFile, IOBase):  # an already opened file object, check access mode
			if outFile.mode in 'aw':  out=outFile
			else:                     raise SystemExit ('ERROR --- aeiou.awrite:  file ' + repr(outFile.name) + ' opened in readmode!')
		else:
			raise SystemExit ('ERROR --- awrite:  invalid output file name, need string or fileObject')
		# print command line as first line to output file
		#f not sys.argv[0].endswith('python') and out.tell()<1 and not out.isatty():
		if not (sys.argv[0].endswith('python') or 'ipython' in join_words(sys.argv)):
			out.write (commentChar + ' ' + get_command() + '\n' + commentChar + '\n')
	else:   out = sys.stdout

	# write header section
	if isinstance(comments, dict):
		for key,com in list(comments.items()): out.write ( '%s %s: %s\n' % (commentChar, key, com) )
	elif isinstance(comments, (list,tuple)):
		for com in comments: out.write ( '%s %s\n' % (commentChar, com) )
	elif isinstance(comments, np.ndarray):
		format = commentChar+len(comments)*' %g'+'\n';  out.write ( format % tuple(comments) )
	elif isinstance(comments, int):
		out.write ( '%s %i\n' % (commentChar, comments) )
	elif isinstance(comments, float):
		out.write ( '%s %g\n' % (commentChar, comments) )
	elif isinstance(comments, str):
		out.write ( '%s %s\n' % (commentChar, comments.rstrip()) )
	else:
		if comments:  print(type(comments))

	# in case a list/tuple of arrays is given: combine into common array
	if isinstance(data, (list,tuple)):  xy = cstack(*data)
	else:
		if data.dtype.names:       xy = data.view(np.float64).reshape(-1,len(data[0]))
		else:                      xy = data

	# finally print the data
	if len(xy.shape)==1:
		if format.count('\n')==0: format = format.rstrip()+'\n'
		for i in range(xy.shape[0]): out.write (format % xy[i] )
	elif len(xy.shape)==2:
		npc = format.count('%')
		if npc==1:
			format = xy.shape[1] * format
		elif npc==2 and xy.shape[1]>2:
			f2 = format.rfind('%')
			format = format[:f2] + (xy.shape[1]-1) * (' '+format[f2:])
		elif npc!=xy.shape[1]:
			print("ERROR --- aeiou.awrite:  check format (number of format specs does'nt match number of columns in data)")
			return
		if format.count('\n')==0: format = format.rstrip()+'\n'
		for i in range(xy.shape[0]): out.write (format % tuple(xy[i,:]) )
	else:
		print('ERROR --- aeiou.awrite:   writing arrays with more than 2 dimensions not supported!')

	# only close file if it has been opened here, too
	if isinstance(outFile, IOBase):  out.close()


####################################################################################################################################

def loadxy (xyFile, usecols=(0,1), xLimits=None, verbose=False, commentChar='#'):
	""" Read a tabular two-column ascii file with loadtxt and separately return the xGrid and the yValues arrays. """
	if isinstance(usecols,(list,tuple,PairOfInts)):
		xGrid, yValues = np.loadtxt(xyFile, usecols=usecols, unpack=1, comments=commentChar)
	else:
		raise SystemExit ("ERROR --- loadxy:  expected a list, tuple, pair of 2 integers for column specification")

	if xLimits:
		xGrid, yValues = xTruncate(xGrid, yValues, xLimits)

	if verbose:
                deltaX = np.diff(xGrid)
                print (len(xGrid),  'xy pairs:\n',  min(xGrid), '< x <', max(xGrid), end='')
                if max(deltaX)-min(deltaX)<0.001*deltaX[0]:  print ('  equidistant with dx=', deltaX[0])
                else:                                        print ('  nonequidistant with ', min(deltaX), ' <= dx <=', max(deltaX))
                print (min(yValues), '< y <', max(yValues))

	return xGrid, yValues


def loadxyy (xyyFile, verbose=False, commentChar='#'):
	""" Read a tabular ascii file with loadtxt and separately return the xGrid array and the y data 'matrix'. """
	xyyy = np.loadtxt(xyyFile, comments=commentChar)
	if verbose:
		print(len(xyyy[:,0]),  'grid points:',  min(xyyy[:,0]), '< x <', max(xyyy[:,0]))
		print(xyyy.shape[1]-1, ' y columns, ',  min(xyyy[:,1:].flatten()), '< y <', max(xyyy[:,1:].flatten()))
	return xyyy[:,0], xyyy[:,1:]


####################################################################################################################################

def read_xyz_file (xyzFile, yInfo=-1, verbose=False, commentChar='#'):
	""" Read a xGrid, yGrid and a zzz "matrix" from a tabular ascii file.
	    Return first column as xGrid, yGrid given in header or as function argument, further columns as zzz matrix.

	    ARGUMENTS:
	    ----------
	    xyzFile:        the data file to read (using numpy.loadtxt)
	    yInfo:          * an integer indicating the line number in the file header where to read the y grid.
	                      default -1:  read the last header line
	                    * a numpy array (or list of floats) with length equal to zzz.shape[1]
	    verbose:        flag
	    commentChar:    default '#'

	"""
	xzzz  = np.loadtxt(xyzFile, comments=commentChar)
	xGrid =  xzzz[:,0]
	zzz   =  xzzz[:,1:]

	# read or construct the y grid
	if isinstance(yInfo, int):
		where = readFileHeader(xyzFile)[yInfo]        # assume column ID's are given in a header comment line
		where = re.sub(r'[+-][qh1248]K','',where)     # in finite difference files column ID's end with perturbations like '+1K'
		yGrid = np.array(list(map(float,where.translate(None,ascii_lowercase+'T@').split())))
	elif isinstance(yInfo, (list,tuple)):
		yGrid = np.ndarray(yInfo)
	elif isinstance(yInfo, np.ndarray):
		yGrid = yInfo
	else:
		yGrid=np.arange(float(zzz.shape[1]))

	# check monotonicity of y grid
	if not (np.all(np.ediff1d(yGrid)>0) or np.all(np.ediff1d(yGrid)<0)):
		print('WARNING --- read_xyz_file:  yGrid is not monotonically increasing/decreasing!!!')
	# check number of rows and columns
	if not (len(xGrid)==zzz.shape[0] and len(yGrid)==zzz.shape[1]):
		print('x', len(xGrid), '     z', len(yGrid), '     jac', zzz.shape)
		raise SystemExit ('ERROR --- read_xyz_file:  inconsistent array shapes!')
	if verbose:
		print(len(xGrid), min(xGrid), '< x <', max(xGrid))
		print(len(yGrid), min(yGrid), '< z <', max(yGrid))
		print(zzz.shape,  min(zzz.flatten()), '< y <', max(zzz.flatten()))

	return xGrid, yGrid, zzz


####################################################################################################################################

def uniqueNames (listOfNames):
	""" Given a list of strings, return the list of unique strings (i.e. a list without duplicates). """
	listUnique=[]
	for name in listOfNames:
		if name not in listUnique:  listUnique.append(name)
	return listUnique


def countUniqueNames (listOfNames):
	""" Given a list of strings, return the number of unique strings (i.e. without duplicates). """
	return len(uniqueNames(listOfNames))


####################################################################################################################################

def read_first_line (sFile, verbose=0):
	""" Try to determine filetype (pickle or ascii or ...) automatically from first nonblank character in file. """

	try:
		firstLine = open(sFile).readline().strip()
	except UnicodeDecodeError:
		from pickle import loads
		firstLine = loads(open(sFile,'rb').readline())
	except Exception as errMsg:
		raise SystemExit ("ERROR --- aeiou.read_first_line failed to read first line from file %s\n%s" % (sFile, errMsg))
	finally:
		if verbose:  print ("read_first_line: ", sFile, ' --->', len(firstLine), '"'+firstLine+'"')

	return firstLine

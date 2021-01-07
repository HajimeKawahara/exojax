"""
Structured arrays --- some convenience functions to read and manipulate:

loadStrucArray:           Read a tabular ascii file and return a structured array
strucArrayChangeNames:    Change field names of some entries of a structured array
strucArrayExtractFields:  Extract some field entries of a structured array
strucArrayDeleteFields:   Delete some field entries of a structured array
strucArrayAddField:       Add a new field (data array) to a structured array
dict2strucArray:          Transform a dictionary of arrays into a structured numpy array

Documentation:
  http://docs.scipy.org/doc/numpy/user/basics.rec.html
  http://www.scipy.org/Cookbook/Recarray

Example:
  atm=loadStrucArray('midLatSummer.xy', -2, changeNames={'pressure':'p', 'temperature':'T'})
"""

##############################################################################################################
#####  LICENSE issues:                                                                                   #####
#####                 This file is part of the Py4CAtS package.                                          #####
#####                 Copyright 2002 - 2019; Franz Schreier;  DLR-IMF Oberpfaffenhofen                   #####
#####                 Py4CAtS is distributed under the terms of the GNU General Public License;          #####
#####                 see the file ../license.txt in the parent directory.                               #####
##############################################################################################################

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numeric python failed!')

from exojax.aux.hitran.aeiou import grep_from_header, grep_word, getCommentLines, countUniqueNames, join_words

####################################################################################################################################

def loadStrucArray (inFile, key2names=-1, changeNames=None, commentChar='#', verbose=False):
	""" Read a tabular ascii file and return a structured array (closely related to record array, but more efficient).

	    key2names      identifies the row (record) in the file's header section (comments) where to read the field names.
	                   *  an integer specifying the row (0 for the very first, -1 for the last)
                           *  a single word identical to the very first field name
                           *  a string ending with ':' (e.g. "what:", the following words are taken as the field names)
			   Alternatively you can explicitely specify the column ID's:
	                   *  a string (or list) with names (words, exactly as many as data columns)
	    changeNames    a dictionary to translate field names found in the file header.

	    NOTE:  If the dataset has more columns than names given, the extra columns are ignored.
	           Blanks are not allowed in field names read from the file header. """
	# get column names
	if   isinstance(key2names, int):
		fieldNames = getCommentLines (inFile, commentChar='#')[key2names]
		if fieldNames.split()[0].endswith(':'):  fieldNames=fieldNames.split()[1:]
		else:                                    fieldNames=fieldNames.split()
	elif isinstance(key2names, str):
		words = key2names.split()
		if len(words)==1:
			if key2names.endswith(':'):
				fieldNames = grep_from_header (inFile, key2names[:-1])
				if isinstance(fieldNames, str):
					fieldNames = fieldNames.split()
				else:
					raise SystemExit ('ERROR --- aeiou.loadStrucArray: could not find a header line starting with '
					        + repr(key2names) + ' identifying the field (column) names in file ' + repr(inFile))
			else:
				fieldNames = grep_word (inFile, key2names)
		else:
			fieldNames = words
	elif isinstance(key2names, (list,tuple)) and all([isinstance(word, str) for word in key2names]):
		fieldNames = key2names
	else:
		raise SystemExit ('ERROR --- aeiou.loadStrucArray: could not find a header line starting with '
		                  + repr(key2names) + ' identifying the field (column) names in file ' + repr(inFile))

	# check if names are unique
	if countUniqueNames(fieldNames)<len(fieldNames):
		if verbose:  print('# unique', countUniqueNames(fieldNames), '   # fields', len(fieldNames))
		raise SystemExit ("ERROR --- aeiou.loadStrucArray:  field names are not unique (%i duplicate entries)\n           %s"
		                   % (len(fieldNames)-countUniqueNames(fieldNames), fieldNames))

	# optionally change field names
	if isinstance(changeNames,dict):
		fieldNames = [changeNames.get(name,name) for name in fieldNames]

	# simply assume all data are floats
	frmt  = [np.float]*len(fieldNames)

	# read file and return structured array
	strArray = np.loadtxt(inFile, dtype={'names': fieldNames, 'formats': frmt}, comments=commentChar)
	if verbose:  print('\n INFO:  structured array with', len(strArray), 'rows and ', len(fieldNames), 'columns (entries):',
	       join_words(fieldNames))

	return strArray


####################################################################################################################################

def strucArrayChangeNames (strucArray, changeNames=None):
	""" Change field names of some entries of a structured array. """
	oldNames =  strucArray.dtype.names
	if not isinstance(changeNames,dict):
		raise SystemExit ("ERROR --- strucArrayChangeNames:  expected a dictionary with old -> new pairs.")
	newNames = [changeNames.get(name,name) for name in oldNames]
	strucArray.dtype.names = newNames
	return strucArray


####################################################################################################################################

def strucArrayExtractFields (strucArray, extract):
	""" Extract some field entries of a structured array and return a smaller structured array. """
	if   isinstance(extract, str):           extract = extract.split(',')
	elif isinstance(extract, tuple):         extract = list(extract)
	elif isinstance(extract, list):          pass
	else:   raise SystemExit ('ERROR --- strucArrayExtractFields:  need a string or list of names')
	return strucArray[extract]


####################################################################################################################################

def strucArrayDeleteFields (strucArray, delete):
	""" Delete some field entries of a structured array and return a smaller structured array. """
	if   isinstance(delete, str):           delete = delete.split()
	elif isinstance(delete, (list,tuple)):  pass
	else:   raise SystemExit ('ERROR --- strucArrayDeleteFields:  need a string or list of names')
	oldNames =  strucArray.dtype.names
	extract  = [name for name in oldNames if name not in delete]
	return strucArray[extract]


####################################################################################################################################

def strucArrayAddField (strucArray, newField, newName):
	""" Add a new field (data array) to a structured array. """
	if isinstance(newField, np.ndarray):
		if not (len(newField.shape)==1 and newField.shape[0]==strucArray.size):
			raise SystemExit ('ERROR --- strucArrayAddField: new field must be one-dim and have same size as strucArray')
	else:
		raise SystemExit ('ERROR --- strucArrayAddField: new field must be a numpy array')

	newNames =  list(strucArray.dtype.names) + [newName]
	newFrmt  = [np.float]*len(newNames)
	newStrucArray = np.empty(strucArray.size, dtype={'names': newNames, 'formats': newFrmt})
	# copy the old fields
	for name in strucArray.dtype.names:  newStrucArray[name] = strucArray[name]
	# ... and insert the new one
	newStrucArray[newName] = newField
	return newStrucArray


####################################################################################################################################

def dict2strucArray (arrayDict, changeNames=None):
	""" Transform a dictionary containing some arrays (of equal length) into a structured numpy array. """
	# get length and keys of all arrays in dictionary
	nzz   = np.array([len(value) for value     in list(arrayDict.values()) if isinstance(value, np.ndarray)])
	names =          [key        for key,value in list(arrayDict.items())  if isinstance(value, np.ndarray)]

	if min(nzz)==max(nzz):
		# equal sized arrays, allocate structured array
		newStrucArray = np.empty(nzz[0], dtype={'names': names, 'formats': len(nzz)*[np.float]})
		# ... and copy data
		for key,values in list(arrayDict.items()):
			if isinstance(values, np.ndarray):  newStrucArray[key] = values
	else:
		raise SystemExit ('ERROR --- dict2strucArray:  dictionary contains arrays of different lengths!')

	if isinstance(changeNames,dict):  return strucArrayChangeNames(newStrucArray, changeNames)
	else:                             return newStrucArray

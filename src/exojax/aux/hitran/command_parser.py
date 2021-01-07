
##############################################################################################################
#####  LICENSE issues:                                                                                   #####
#####                 This file is part of the Py4CAtS package.                                          #####
#####                 Copyright 2002 - 2019; Franz Schreier;  DLR-IMF Oberpfaffenhofen                   #####
#####                 Py4CAtS is distributed under the terms of the GNU General Public License;          #####
#####                 see the file ../license.txt in the parent directory.                               #####
##############################################################################################################

import getopt
import sys
import os
import re
import keyword
import numpy as np

from aeiou import commonExtension
from pairTypes import Interval, PairOfInts, PairOfFloats,ListOfInts

standardOptions = [{'ID': 'h'},
               {'ID': 'c', 'type': str, 'default': '#'},
                   {'ID': 'o', 'type': str}]

####################################################################################################################################

def prepare4getopt (knownOptions):
    """ Extract short and long option id's as an input for getopt. """
    ShortOptions = ''
    LongOptions  = []
    for option in knownOptions:
        # retrieve type from default value
        if 'type' not in option and 'default' in option:  option['type'] = type(option['default'])

        #f option.has_key('ID') or option.has_key('id'):
        if 'ID' in option and 'id' in option:
            if not option['ID']==option['id']:
                raise SystemExit ("ERROR --- prepare4getopt:  duplicate, but conflicting id and ID")
        elif 'ID' in option or 'id' in option:
            ID = option.setdefault('ID',option.get('id'))
            if 'name' not in option:  option['name'] = option['ID']
            if len(ID)==1:
                if 'type' in option:  ShortOptions = ShortOptions + ID + ':'
                else:                 ShortOptions = ShortOptions + ID
            else:
                if 'type' in option:  LongOptions.append(ID+'=')
                else:                 LongOptions.append(ID)
        elif 'name' in option:
            option['ID'] = option['name']
            if 'type' in option:  LongOptions.append(ID+'=')
            else:                 LongOptions.append(ID)
        else:
            raise SystemExit ('option specification requires "ID" and/or "name"')
    return ShortOptions, LongOptions


####################################################################################################################################

def getopt_parser (ShortOptions, LongOptions):
    """ Parse command line string using getopt.  Return list of files and a dictionary of options! """
    args = sys.argv[1:]
    if len(args)==1 and 'help=' in LongOptions and '--help' in args:
        options = {'h': None}  # print standard help message
        files   = []
    elif len(args)>0:
        try:
            if LongOptions:
                OptionsList, files = getopt.getopt(args, ShortOptions, LongOptions)
            else:
                OptionsList, files = getopt.getopt(args, ShortOptions)
        except getopt.error as errMsg:
            print("\ncheck your options list!!!")
            print(errMsg)
            print("valid options are ", end=' ')
            #for i in range(len(ShortOptions)): if not ShortOptions[i]==':':  print('-' + ShortOptions[i], end=' ')
            for i,so in enumerate(ShortOptions):
                if not so==':':  print('-' + so, end=' ')
            if LongOptions:
                for option in LongOptions: print('--' + option.replace('=',''), end=' ')
            raise SystemExit ("ERROR --- getopt_parser:  parsing input arguments failed!")
            # return options as a dictionary (getopt returns a double list!)
        options = {}
        #for i in range(len(OptionsList)):
        for i,optn in enumerate(OptionsList):
            key          = optn[0].replace('-','')  # remove leading dash(es)
            options[key] = optn[1]
    else:
        files   = []
        options = {}
    return files, options


####################################################################################################################################

def check_constraint (value, name, constraint):
    """ Perform various checks on name and value of a given option. """
    if name in constraint:
        if keyword.iskeyword(name):
            raise SystemExit ('name conflict: ' + name + ' is a reserved word in PYTHON!')
        if type(value) in (int,float):
            statement = name + '=' + repr(value)
            exec(statement)
            if  not eval(constraint):
                raise SystemExit (statement + '\nconstraint ' + repr(constraint) + ' violated!')
        elif type(value) is np.ndarray:
            statement = name + ' = np.' + repr(value)
            exec(statement)
            if not np.alltrue(eval(constraint)):
                raise SystemExit (statement + '\nconstraint ' + repr(constraint) + ' violated (comparison elementwise)!')
        elif type(value) is str:
            statement = name + '=' + repr(value)
            exec(statement)
            if  not eval(constraint):
                raise SystemExit (statement + '\nconstraint ' + repr(constraint) + ' violated!')
        elif isinstance(value,(Interval,PairOfInts,PairOfFloats)):
            statement = name + '=' + repr(value)
            exec(statement)
            if  not eval(constraint):
                raise SystemExit (statement + '\nconstraint ' + repr(constraint) + ' violated!')
        elif type(value) is bool:
            statement = name + '=' + value
            exec  (statement)
            if  not eval(constraint):
                raise SystemExit(  statement + '\nconstraint ' + repr(constraint) + ' violated!')
        else:
            raise SystemExit ('unknown/unsupported type ' + repr(type(value)) + ' for check_constraint')
    else:
        raise SystemExit ('Variable name ' + repr(name) + ' not used in constraint expression ' + repr(constraint))

####################################################################################################################################

def check_type (id, name, given, oType):
    try:
        if   oType==str:             typeChecked = given.strip()
        elif oType==float:           typeChecked = float(given)
        elif oType==int:             typeChecked = int(given)
        elif oType in (list,    tuple,    np.ndarray, Interval, PairOfInts, PairOfFloats,ListOfInts):
        #lif oType in (ListType,TupleType,np.ndarray, Interval, PairOfInts, PairOfFloats,ListOfInts):
            typeChecked = re.split('[,;\s]',given)
            if   oType==np.ndarray:
                typeChecked=np.array(list(map(float,typeChecked)))
            elif oType==Interval:
                low, hi = list(map(float,typeChecked))
                typeChecked=Interval(low,hi)
            elif oType==PairOfInts:
                left, right = list(map(int,typeChecked))
                typeChecked=PairOfInts(left,right)
            elif oType==PairOfFloats:
                left, right = list(map(float,typeChecked))
                typeChecked=PairOfFloats(left,right)
            elif oType==ListOfInts:
                typeChecked=ListOfInts(list(map(int,typeChecked)))
            #else: print 'unrecognized type ', id, name, given, typeChecked, oType, oType==np.ndarray
        else:
            raise SystemExit (6*'%s ' % ('ERROR --- check_type:', oType,
                              'for option', repr(id), repr(name), 'not yet supported, sorry'))
    except ValueError as errMsg:
        raise SystemExit ('ERROR ---check_type:  option ' + id + ' = ' + name + '   ' + str(errMsg))
    except Exception as errMsg:
        raise SystemExit ('ERROR ---check_type:  type checking of options failed!\n' + str(errMsg))
    return typeChecked

####################################################################################################################################

def check_options (optionsGiven, knownOptions, verbose=0):
    """ Check the options specified on the command line wrt type and constraits,
        add unspecified options with defaults if available."""
    for option in knownOptions:
        id = option.get('ID',option.get('id'))
        name = option.get('name',id)
        if id in optionsGiven:
            given = optionsGiven.pop(id)
            if 'type' in option:
                typeChecked = check_type (id, name, given, option['type'])
            else:
                typeChecked = given
            if 'constraint' in option:
                check_constraint (typeChecked, option['name'], option['constraint'])
            optionsGiven[name]=typeChecked
        else:
            if 'default' in option:
                optionsGiven[name]=option['default']
                if verbose and not id=='c':  print(id, name, 'set to default:', optionsGiven[name])
    return optionsGiven

####################################################################################################################################




def parse_command (knownOptions, numFiles=None, env4defaults='', verbose=0):
    """ Parse command line arguments or interactively ask directly for files,  return options as a dictionary. """
    print(knownOptions)
    import sys

    # translate options specs into string (for short options) and optionally list (for long options) appropriate for standard getopt
    ShortOptions, LongOptions = prepare4getopt (knownOptions)
    print("shirt=",ShortOptions)
    print("long=",LongOptions)
    # parse the command using getopt, but return a dictionary instead of getopt's list of two-element tuples
    # (only with those options specified in the command line!)
    files, optionsGiven = getopt_parser (ShortOptions, LongOptions)
    print("files->",files)
#    print("->",optionsGiven)

    # check options for type etc, return dictionary, now with defaults added!
    optionsGiven = check_options (optionsGiven, knownOptions, verbose)
    print("optionsGiven->",optionsGiven)


    
    # check number of (input) files
    if not ('h' in optionsGiven or 'help' in optionsGiven or 'about' in optionsGiven):
        commandName = os.path.basename(sys.argv[0])
        print(commandName,numFiles)

    # extract common options
    commentChar = optionsGiven.pop('c','#')
    outFile     = optionsGiven.pop('o',None)
    print(commentChar)
    print(outFile)

    return files, optionsGiven, commentChar, outFile

####################################################################################################################################

def multiple_outFiles (inFiles, outFile):
    """ Given a list of input file names and a 'template' (e.g. extension) for the output files,
        return a list of output file names. """
    if outFile:
        if len(inFiles)>1:
            if not outFile.startswith('.'): outFile='.'+outFile
            # -o option specifies extension of output files
            commonExt = commonExtension(inFiles)
            print('commonExt:', commonExt)
            # replace extension only when all input files have the same extension
            if commonExt:  outFiles = [os.path.splitext(iFile)[0]+outFile for iFile in inFiles]
            else:          outFiles = [iFile+outFile for iFile in inFiles]
        else:
            if outFile.startswith('.'): return [os.path.splitext(inFiles[0])[0]+outFile]  # do not write to a hidden file!
            else:                       return [outFile]
    else:
        outFiles = [None for iFile in inFiles]
    return outFiles

####################################################################################################################################

def change_defaults (opts, userDefaults):
    """ Replace standard options by user specified options. """
    newDefaults = dict([kv.strip().split('=') for kv in userDefaults.split(';')])
    for opt in opts:
        #if opt.has_key('default') and newDefaults.has_key(opt['ID']):
        if opt['ID'] in newDefaults:
            if 'default' in opt:
                if isinstance(opt['default'],int):      opt['default'] = int(newDefaults[opt['ID']])
                elif isinstance(opt['default'],float):  opt['default'] = float(newDefaults[opt['ID']])
                else:                                   opt['default'] = newDefaults[opt['ID']]
            elif 'type' in opt:
                if   opt['type']==str:           opt['default']= newDefaults[opt['ID']]
                elif opt['type']==float:            opt['default']= float(newDefaults[opt['ID']])
                elif opt['type']==int:              opt['default']= int(newDefaults[opt['ID']])
    print('\n---> new defaults <---')
    for opt in opts:
        if opt['ID'] in newDefaults:  print(opt['ID'], opt.get('default'))
    print()
    return opts

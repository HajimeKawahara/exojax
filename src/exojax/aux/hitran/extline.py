#!/usr/bin/env python3
"""
  higstract

  higstract (extract/grep/select) line parameters from a spectroscopic data base file

  usage:
  higstract  [options]  line_parameter_database

  command line options:
    -h            help
   --help         help extended
    -c   char(s)  comment character used in output file (default #)
    -o   file     output file (default: standard output, see last note of extended help)

    -f   string   output format:  original or simple lists (position vs strengths etc)
                                  default "vSEan" (see notes of extended help)
    -i   integer  isotope name (e.g. 4 or 162 for heavy water HDO, the fourth most abundant isotope)
    -m   string   molecule name  (one molecule only!)
                  "main" to save only lines of the main molecules (ignore trace gases)
    -S   float    minimum line strength to accept
    -x   interval lower and upper end of spectral (default: wavenumber) range (comma separated pair without blanks)
    -X   string   unit used for setting spectral range with -x option (does not affect output listing!)
                  (default: "cm-1",  other choices: "Hz", "MHz", "GHz", "THz", "mue", "nm")

  NOTE:  to avoid name clashes with numpy's extract function,
         this module and its the 'main' function have been renamed to 'higstract'
     (short for HItran-GeiSa-exTRACT)

  For more information use
  higstract --help
"""
more_help = \
"""

  OUTPUT FORMAT:
    *  "simple lists" --- for each molecule a (tabular ascii) file with columns for position etc is generated
                          (where the filename is set automatically by molecule name with the format indicated by the extension)

       The actual format is defined by a combination of single letters
       "v" --- wavenumber/frequency position   (Hint: the letter "v" looks like the greek nu)
       "S" --- Strengths
       "E" --- Energy (lower state)
       "a" --- air broadening half width (pressure, collision broadening)
       "s" --- self broadening half width
       "n" --- temperature exponent n of pressure broadening
       "i" --- isotope number
       "b" --- all broadening parameters, equivalent to "asni"

       use "vSEan" or "vSEasni"="vSEb" to produce a line list acceptable by lbl2xs and lbl2od
       use "o" or "h" or "g" to save the extract in the original format

       Valid formats are:   "o", "g", "h",  "vS","vSa","vSE","vSEa","vSEb","vSEan","vSEasn","vSEasni"


  NOTES:
    *  at least wavenumber range OR molecule has to be selected!
    *  molecule names are case sensitive!
    *  extracting lines of some selected molecules simultaneously is (currently) not supported:
       either specify one molecule OR main OR none
    *  selecting an isotope by its abundance number might not work perfectly
       (CO2 has 10 isotopes, but H2 and HBr have "11" as isotope-ID)
    *  the database filename must include either the string "hit" or "geisa" or "sao" (case insensitive)
       in order to give higstract a chance to read with the proper format!
    *  currently only HITRAN, GEISA or SAO are supported
    *  format conversion HITRAN <---> GEISA not implemented
    *  if lines for all gases in a spectral range are to be selected and if the output is to be written
       to simple line lists (format 'vS' etc), separate files are produced for each molecule individually
       (the code definig the final output file names is not yet perfect)
    *  main molecules only:  H2O, CO2, O3, N2O, CH4, CO, O2,  but no trace gases
"""

_LICENSE_ = """\n
This file is part of the Py4CAtS package.

Authors:
Franz Schreier
DLR-IMF Oberpfaffenhofen
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

####################################################################################################################################
#####  ToDo:                                                                                                                   #####
#####                                                                                                                          #####
#####  allow isotope aliases like HDO                                                                                          #####
#####                                                                                                                          #####
####################################################################################################################################

import os
import os.path
import numpy as np

#from ir import c, h, k
from exojax.aux.hitran.aeiou import open_outFile, join_words
from exojax.aux.hitran.cgsUnits import change_frequency_units
from exojax.aux.hitran.molecules import molecules, mainMolecules, get_mol_id_nr, isotope_id
from exojax.aux.hitran.lines import lineArray
from exojax.aux.hitran.pairTypes import Interval


####################################################################################################################################

def save_lines_orig (lineList, lineFile, outFile=None, format='', mainOnly=False, commentChar='#'):
    """ Save Hitran|Geisa lines extracted in original format. """
        
    if len(lineList)==0:  raise SystemExit('WARNING --- higstract.save_lines_orig:  no lines')

    if   'hit' in lineFile.lower():
        # open output file and write some info to header
        out = open_outFile (outFile, '000')
        if len(lineList[0].rstrip())==100:  # old hitran versions <=2000
            out.write ('%3s%12s%10s%10s%5s%5s%10s%4s%8s%3s%3s%9s%9s%3s%6s\n' %
                  ('000','wavenumber','S','A','air','self','Energy','n','pShift','uV','lV','uL','lL','er','ref'))
        else:  # new hitran versions >=2004
            out.write ('%3s%12s%10s%10s%5s%5s%10s%4s%8s%15s%15s%15s%15s%6s%12s %s7%7s\n' %
                  ('000','wavenumber','S','A','air','self','Energy','n','pShift','upVib','loVib','upLocal','loLocal','err','ref','usw','lsw'))
        if mainOnly:
            for line in lineList:
                if int(line[:2])<=len(mainMolecules):    out.write ('%s\n' % line.rstrip().decode())
        else:
            for line in lineList:  out.write ('%s\n' % line.rstrip().decode())
    elif 'geisa' in lineFile.lower():
        out = open_outFile (outFile, 5*commentChar)
        if mainOnly:
            from geisa import get_geisa_fields
            mField = get_geisa_fields (lineFile, 'M')
            for line in lineList:
                if int(line[mField.left:mField.right])<=len(mainMolecules):  out.write ('%s\n' % line.strip().decode())
        else:
            for line in lineList:  out.write ('%s\n' % line.strip().decode())
    else:
        raise SystemExit ('\nERROR --- higstract.save_lines_orig:  unknown lineFile')

    if outFile: out.close()

####################################################################################################################################

def save_lines_core (dictOfLineLists, outFile='', format='vSEan', mainOnly=False, commentChar='#'):
    """ Save a dictionary of line arrays (structured numpy arrays with attributes) to files, molecule by molecule. """

    # replace format shortcut with 'full' name
    if format.lower()=="vseb":  format='vSEasni'

    # check if output file extension indicates the format (os.path.extsep = os.extsep normally returns period ".")
    if outFile:
        if os.path.extsep in outFile:     outFileRoot, outFileExt = os.path.splitext(outFile)
        elif outFile.startswith('v'):     outFileRoot, outFileExt = '', os.path.extsep+outFile
        else:                             outFileRoot, outFileExt =  outFile, ''
        if format=='vSEan' and outFileExt.lower().startswith('.vs'):
            format = outFileExt[1:]
            print('format automatically determined from output file extension ', format)
    else:
        outFileRoot, outFileExt =  '', os.path.extsep+format

    if isinstance(dictOfLineLists,dict):
        for molec,lines in list(dictOfLineLists.items()):
            if mainOnly and molec not in mainMolecules: continue
            # set file name by name of molecule and column identifiers
            if molec in outFileRoot:  outFileName =         outFileRoot + os.path.extsep + format
            else:                     outFileName = molec + outFileRoot + os.path.extsep + format
            # loop over all lines (one entry=string in list per line), cut requested columns and write to file
            write_lines_xy (lines, outFileName, format, commentChar)
    elif isinstance(dictOfLineLists,lineArray):
        write_lines_xy (dictOfLineLists, outFile, format, commentChar)
    else:
        raise SystemExit ("ERROR --- save_lines_core:  unknown data type, expected a dictionary of lineArray's or an lineArray")


def write_lines_xy (data, outFile, job=None, commentChar='#'):
    """ Print 'core' line parameters, i.e., positions vs strengths, and optionally energies, airWidths, tempExponents. """

    # open an output file if explicitely specified (otherwise use standard output)
    out = open_outFile (outFile, commentChar)

    if hasattr(data,'molec'):
        out.write ('%s %s %s\n' % (commentChar, "molecule:", data.molec))
    else:
        raise SystemExit ('ERROR --- write_lines_xy:  inconsistent/nonexisting molecule info')

    if not job:
        oFileExt = os.path.splitext(outFile)[1]
        if oFileExt.startswith('.vS'):  job=oFileExt[1:]
        else:                           job='vSEan'

    # reference pressure and temperature of database
    if hasattr(data,'t'): out.write ('%s %s %8.2f %s\n' % (commentChar, "temperature:", data.t, "K"))
    if hasattr(data,'p'): out.write ('%s %s %8.2f %s\n' % (commentChar, "pressure:   ", data.p, "dyn/cm**2"))

    # some statistical information
    nLines = len(data)
    out.write ('%s %-30s %12i\n' % (commentChar, "number of lines:     ", nLines))
    out.write ('%s %-30s %12.3g%13.3g\n' % (commentChar, "min, max line strength:     ", min(data['S']),  max(data['S'])))
    if 'E' in job:
        out.write ('%s %-30s %12.3f%13.3f\n' % (commentChar, "min, max lower state energy:", min(data['E']),  max(data['E'])))
    if 'a' in job:
        out.write ('%s %-30s %12.3f%13.3f\n' % (commentChar, "min, max airbroad.  widths: ", min(data['a']),  max(data['a'])))
    if 's' in job:
        out.write ('%s %-30s %12.3f%13.3f\n' % (commentChar, "min, max selfbroad. widths: ", min(data['s']), max(data['s'])))
    if 'n' in job:
        out.write ('%s %-30s %12.3f%13.3f\n' % (commentChar, "min, max temp. exponent:    ", min(data['n']),    max(data['n'])))
    if 'd' in job:
        out.write ('%s %-30s %12.3f%13.3f\n' % (commentChar, "min, max air press shift:   ", min(data['d']),    max(data['d'])))
    out.write ('%s %s %s\n' % (commentChar, "format:", job))
    if outFile:
        print('%-8s %8i %s %10.3f %s %-10.3f %s %11.3g %s %-11.3g %s %s' % \
              (data.molec, nLines, ' lines in', data['v'][0],  '< v <', data['v'][-1],
               ' with', min(data['S']),  '< S <', max(data['S']), ' writing to file', outFile))

    if job=="vSEasndi":
        format = '%12f %11.3e %11.5f %8.5f %8.5f %7.4f %9.6f %3i\n'
        out.write ('%1s %10s %11s %11s %8s %8s %7s %9s %4s\n' % (commentChar,'position','strength', 'energy', 'airWidth', 'selfWidth', 'Tdep', 'shift','iso'))
        out.write ('%1s %10s %11s %11s %8s %8s %7s %9s\n' % (commentChar,'cm-1','cm-1/cm-2', 'cm-1', 'cm-1',  'cm-1', '', 'cm-1'))
        for line in data:  out.write ( format % (line['v'],line['S'],line['E'],line['a'],line['s'],line['n'],line['d'],line['i']))
    elif job=="vSEasnd":
        format = '%12f %11.3e %11.5f %8.5f %8.5f %7.4f %9.6f\n'
        out.write ('%1s %10s %11s %11s %8s %8s %7s %9s\n' % (commentChar,'position','strength', 'energy', 'airWidth', 'selfWidth', 'Tdep', 'shift'))
        out.write ('%1s %10s %11s %11s %8s %8s %7s %9s\n' % (commentChar,'cm-1','cm-1/cm-2', 'cm-1', 'cm-1',  'cm-1', '', 'cm-1'))
        for line in data:  out.write ( format % (line['v'],line['S'],line['E'],line['a'],line['s'],line['n'],line['d']))
    elif job=="vSEasni":
        format = '%12f %11.3e %11.5f %8.5f %8.5f %7.4f %4i\n'
        out.write ('%1s %10s %11s %11s %8s %8s %7s %4s\n' % (commentChar,'position','strength', 'energy', 'airWidth', 'selfWidth', 'Tdep', 'iso'))
        out.write ('%1s %10s %11s %11s %8s %8s %7s\n' % (commentChar,'cm-1','cm-1/cm-2', 'cm-1', 'cm-1',  'cm-1', ''))
        for line in data:  out.write ( format % (line['v'],line['S'],line['E'],line['a'],line['s'],line['n'],line['i']))
    elif job=="vSEani":
        format = '%12f %11.3e %11.5f %8.5f %7.4f %3i\n'
        out.write ('%1s %10s %11s %11s %8s %7s %3s\n' % (commentChar,'position','strength', 'energy', 'airWidth', 'Tdep', 'iso'))
        out.write ('%1s %10s %11s %11s %8s %7s\n' % (commentChar,'cm-1','cm-1/cm-2', 'cm-1', 'cm-1', ''))
        for line in data:  out.write ( format % (line['v'],line['S'],line['E'],line['a'],line['n'],line['i']))
    elif job=="vSEan":
        format = '%12f %11.3e %11.5f %8.5f %7.4f\n'
        out.write ('%1s %10s %11s %11s %8s %7s\n' % (commentChar,'position','strength', 'energy', 'airWidth', 'Tdep'))
        out.write ('%1s %10s %11s %11s %8s %7s\n' % (commentChar,'cm-1','cm-1/cm-2', 'cm-1', 'cm-1', ''))
        for line in data:  out.write ( format % (line['v'],line['S'],line['E'],line['a'],line['n']))
    elif job=="vSEasn":
        format = '%12f %11.3e %11.5f %8.5f %8.5f %7.4f\n'
        out.write ('%1s %10s %11s %11s %8s %8s %7s\n' % (commentChar,'position','strength', 'energy', 'airWidth', 'selfWidth', 'Tdep'))
        out.write ('%1s %10s %11s %11s %8s %8s %7s\n' % (commentChar,'cm-1','cm-1/cm-2', 'cm-1', 'cm-1',  'cm-1', ''))
        for line in data:  out.write ( format % (line['v'],line['S'],line['E'],line['a'],line['s'],line['n']))
    elif job=="vSEa":
        format = '%12f %11.3e %11.5f %8.5f\n'
        for line in data:  out.write ( format % (line['v'],line['S'],line['E'],line['a']))
    elif job=="vSE":
        format = '%12f %11.3e %11.5f\n'
        for line in data:  out.write ( format % (line['v'],line['S'],line['E']))
    elif job=="vSa":
        format = '%12f %11.3e %8.5f\n'
        for line in data:  out.write ( format % (line['v'],line['S'],line['a']))
    else:
        format = '%12f %11.3e\n'
        for line in data:  out.write ( format % (line['v'],line['S']))

    # close the output file (if its not stdout)
    if outFile: out.close()
    return


####################################################################################################################################

def check_database_file (lineFile):
    """ Check command line argument supplied to higstract. """
    if not os.path.isfile(lineFile):
        return 'ERROR: ' + repr(lineFile) + ' --- line parameter file invalid, nonexisting, ... ?'

    # parse line database filename to identify type
    count_hit   = int('HIT'   in lineFile.upper())
    count_geisa = int('GEISA' in lineFile.upper())
    count_sao   = int('SAO'   in lineFile.upper())
    count_jpl   = int('JPL'   in lineFile.upper())
    countAll    = count_hit+count_geisa+count_sao+count_jpl

    if   countAll>1:
        raise SystemExit ('ERROR: line parameter database filename is ambiguous!!!' + \
                        '\n       (filename should include either "HIT" or "GEISA" (case insensitive))')
    elif countAll<1:
        raise SystemExit ('ERROR: type of line parameter database invalid or unknown!!!' +
                        '\n       (filename should contain either "HIT" or "GEISA" (case insensitive))')
    else:
        return

####################################################################################################################################

def core_parameters (lines, dataFile):
    """ Given a list of data base records (one entry per transition)
        return numpy structured array with the most important spectrocopic line parameters. """

    # column start/stop for all types of parameters
    if 'hit' in dataFile.lower() or 'sao' in dataFile.lower():  # vLine stren  energ  airWi  selfW   tExp  iso
        iw,lw, iS,lS, iE, lE, iA,lA, isw, lsw, iT,lT, iI,lI = 3,15, 15,25, 45,55, 35,40, 40,45, 55,59, 2,3
        iD,lD = 59,67  # press induced line shift
    else:
        from geisa import get_geisa_fields
        fields = get_geisa_fields (dataFile)
        #for key,val in list(fields.items()):  exec(key + '=' + repr(val))
        iw,  lw  = fields['iw'],  fields['lw']
        iS,  lS  = fields['iS'],  fields['lS']
        iE,  lE  = fields['iE'],  fields['lE']
        iA,  lA  = fields['iA'],  fields['lA']
        iT,  lT  = fields['iT'],  fields['lT']
        iI,  lI  = fields['iI'],  fields['lI']
        isw, lsw = fields['isw'], fields['lsw']

    # allocate structured array
    if 'seom' in dataFile.lower():
        data = np.empty(len(lines), dtype={'names': 'v S E a n s i d N m sd'.split(), 'formats': 11*[np.float]})
        print ('SEOM-IAS database: ', len(data), data.dtype.names)
    elif 'hit' in dataFile.lower():
        data = np.empty(len(lines), dtype={'names': 'v S E a n s i d'.split(), 'formats': 8*[np.float]})
    elif 'geisa' in dataFile.lower():
        data = np.empty(len(lines), dtype={'names': 'v S E a n s i'.split(), 'formats': 7*[np.float]})
    else:
        data = np.empty(len(lines), dtype={'names': 'v S E a n s'.split(), 'formats': 6*[np.float]})

    # now extract columns, convert to appropriate type (int/float)
    data['v'] = np.array([float(line[iw:lw]) for line in lines])
    data['S'] = np.array([float(line[iS:lS].replace('D','e')) for line in lines])
    data['E'] = np.array([float(line[iE:lE]) for line in lines])
    data['a'] = np.array([float(line[iA:lA]) for line in lines])
    data['n'] = np.array([float(line[iT:lT]) for line in lines])
    data['i'] = np.array([int(line[iI:lI]) for line in lines])

    if 'hit' in dataFile.lower() or 'geisa' in dataFile.lower():
        data['s'] = np.array([float(line[isw:lsw]) for line in lines])
    if 'hit' in dataFile.lower():
        data['d'] = np.array([float(line[iD:lD]) for line in lines])

    if 'seom' in dataFile.lower():  # SEOM-Improved Atmospheric Spectroscopy Databases  https://www.wdc.dlr.de/seom-ias/
        data['sd'] = np.array([float(line[181:191]) for line in lines])  # speed-dependence air-broadening
        data['N'] = np.array([float(line[219:229]) for line in lines])  # Dicke narrowing
        data['m'] = np.array([float(line[238:248]) for line in lines])  # line mixing

    return data


####################################################################################################################################

def split_molecules (lineList, dataFile):
    """ Given the list of database records extracted from Hitran/Geisa, distribute the entries in separate lists for each molecule. """

    # define position of molecular ID number in Hitran or Geisa record
    if 'hit' in dataFile.lower():
        mol_id = get_mol_id_nr (molecules, 'hitran')  # translation dictionary: hitran molecular ID numbers --> names
        im, lm =  0, 2                               # set indices for molecular id
    elif 'geisa' in dataFile.lower():
        from geisa import get_geisa_fields
        mField = get_geisa_fields (dataFile, 'M')
        im, lm = mField.left,mField.right
        mol_id = get_mol_id_nr (molecules, 'geisa')  # translation dictionary: geisa molecular ID numbers --> names

    # initialize dictionary
    dictOfLineLists = {}
    # distribute lines of individual molecules into separate lists
    for line in lineList:
        molNr = int(line[im:lm])
        molec = mol_id.get(molNr)
        if molec in dictOfLineLists:            dictOfLineLists[molec].append(line.decode())
        elif molec.isalnum() or molec=='NO+':   dictOfLineLists[molec]  = [line.decode()]
        else:
            print (line)
            raise SystemExit ('ERROR --- higstract:  unknown molecule "' + molec + '" with ID number ' + repr(molNr))

    return dictOfLineLists


####################################################################################################################################

def higstract (lineFile, xLimits=None, molecule=None, isotope=None, strMin=0.0, xUnit='cm-1', format='vSEasni', verbose=False):
    """ HItran GeiSa exTRACT line data (position, strength, width, ...) from spectroscopic line parameter data file.
        RETURN  a list of extracted lines in the original format (one list entry for each data record)
            OR      a dictionary of numpy structured arrays with the core parameters (by molecule).

        Parameters:
        -----------
        lineFile    string      a name uniquely identifying the database
        xLimits     Interval    pair of wavenumbers/frequencies/wavelengths
                                (default al wavenumbers, also see xUnit)
        molecule    string      species to be selected (default: 'all')
                                'main' returns the first 7 Hitran/Geisa molecules H2O, CO2, O3, N2O, CH4, CO, O2
        isotope     string      the isotope ID as used by hitran, e.g. '162' for HDO
                                or
                    int         1=most abundant (e.g. 4 for HDO)
                        default:  all isotopes
            strMin      float       strength of weakest line to be accepted (default: 0.0)
        xUnit       string      "cm-1" (default) | mue | nm | Hz | kHz | MHz| GHz | THz
                                    (only relevant for xLimits, does not change units of returned line positions)
            format      string      a combination of letters indicating the core line parameters

        RETURNS:
        --------
        EITHER      a dictionary of 'lineArray', a subclassed structured numpy array of core line parameters,
        OR          a single 'lineArray, if only a single molecule is to be extracted,
        OR          a list of all lines accepted, i.e. a string (record) for each line in the original format.

        At least a spectral range (xLimits) OR a molecule (name) has to be specified.
        Currently Hitran, Geisa, and SAO formats are supported.

        NOTE:  main function called 'higstract' (short for HItranGeiSaexTRACT)
               to avoid a name clash with numpy's extract function
    """

    if isinstance(xLimits,str):
        raise SystemExit ('ERROR --- higstract:  got a string as second argument instead of a wavenumber interval')

    check_database_file (lineFile)

    mainOnly = molecule=='main'

    #print 'xLimits', type(xLimits), xLimits, '   molecule', type(molecule), molecule
    if not (molecule or xLimits):
        raise SystemExit ('ERROR --- higstract:  neither molecule nor wavenumber range specified!')
    else:
        if mainOnly:  molecule=None
        if isotope and not molecule:
            # NOTE: recent CO2 with 10 isotopes, and H2 and HBr have '11' as isotope-ID
            if isotope<10:
                raise SystemExit ('%s %i%s' %
                                  ('selecting', isotope,'. abundant isotope of all molecules not implemented!'))
            else:
                raise SystemExit ('ERROR --- higstract: searching isotopes requires specification of molecule!')

    if xLimits:
        if isinstance(xLimits,(tuple,list)):  xLimits=Interval(*xLimits)
        if xUnit!='cm-1':
            print(xLimits, xUnit, '-->', end=' ')
            xLimits = change_frequency_units (xLimits, xUnit, 'cm-1')
            print(xLimits)
    else:
        xLimits = Interval(0.0,9999999.9)

    # read data file and return a single list with accepted lines (essentially the data file records)
    if   lineFile.upper().count('HIT'):
        if molecule:
            try:             MolNr = molecules[molecule]['hitran']
            except KeyError: raise SystemExit('ERROR --- higstract: invalid/unknown molecule ' + repr(molecule))
            IsoNr = isotope_id (molecule, isotope, 'hitran')
        else:   MolNr = 0; IsoNr=0
        from exojax.aux.hitran.hitran import extract_hitran
        lineList = extract_hitran (lineFile, xLimits, MolNr, IsoNr, strMin)
        moreInfo = {'T': 296.0, 'p': 1013.25e3, 'file': lineFile, 'x': 'cm-1'}
    elif lineFile.upper().count('GEISA'):
        if '2015' in lineFile:
            print('INFO --- higstract: `molecule` HDO adjusted for Geisa2015, new id=51, maybe more changes required!?!')
            print('old HDO: ', molecules['HDO'])
            molecules['HDO']['geisa']=51
            molecules['H2O']['isotopes'] = ['161', '181', '171', '262']
            print('new H2O: ', molecules['H2O'])
            print('    HDO: ', molecules['HDO'])

        if molecule:
            try:             MolNr = molecules[molecule]['geisa']
            except KeyError: raise SystemExit('ERROR --- higstract: invalid/unknown molecule ' + repr(molecule))
            if isotope:
                if isotope<10:
                    print(str(isotope) + '. most abundant isotope', end=' ')
                    try:               isotope=int(molecules[molecule]['isotopes'][isotope-1])
                    except IndexError: raise SystemExit('ERROR: invalid/unknown isotope ' + repr(isotope))
                    else:              print('-->', isotope)
        else:
            MolNr = 0; IsoNr=0

        from geisa import extract_geisa
        lineList = extract_geisa (lineFile, xLimits, MolNr, isotope, strMin)
        moreInfo = {'T': 296.0, 'p': 1013.25e3, 'file': lineFile, 'x': 'cm-1'}
    elif   lineFile.upper().count('JPL'):
        moreInfo = {'T': 300.0, 'p': 1013.0e3, 'file': lineFile, 'x': 'MHz'}
        raise SystemExit ('ERROR --- higstract:  sorry, JPL not yet implemented!')
    else:
        raise SystemExit ('ERROR --- higstract:  sorry, currently only HITRAN, GEISA, or SAO line database!')


    # EITHER return the "core parameters" molecule-by-molecule  OR  the linelist as extracted in the original format
    if format.lower().startswith('v'):
        dictOfLineLists = split_molecules (lineList, lineFile)  # returns a dictionary of line lists!

        numLines = len(lineList)
        if mainOnly:
            nAll = len(dictOfLineLists)
            for mol in list(dictOfLineLists.keys()):
                if mol not in mainMolecules:  del dictOfLineLists[mol]
            print(' main gases only:  deleted ', nAll-len(dictOfLineLists), ' of ',
                  nAll, ' molecules in dict of linelists with originally', numLines, 'lines')
            numLines = sum([len(dictOfLineLists[mol]) for mol in dictOfLineLists.keys()])

        for mol,lines in list(dictOfLineLists.items()):
                        # return structured array of most important numeric line parameters
                    # and add attributes to the numpy array by subclassing
            dictOfLineLists[mol] = lineArray (core_parameters (lines, lineFile),
                                                      p=moreInfo['p'], t=moreInfo['T'], molec=mol)

        print('\n', numLines, 'lines of ', len(dictOfLineLists), ' molecule(s) extracted from ', lineFile)

        if verbose:
            for lla in list(dictOfLineLists.values()):  lla.info()

        if isinstance(molecule,str) and len(dictOfLineLists)>1:
            print('WARNING --- higstract:  strange, a single molecule requested, but dictOfLineLists has several items!?!')

        if molecule:
            print("returning a lineArray for ",  molecule)
            return dictOfLineLists.get(molecule)
        else:
            print(" returning a dictionary of lineArray's for ",  join_words(list(dictOfLineLists.keys())))
            return dictOfLineLists
    else:
        print(len(lineList), ' lines extracted from ', lineFile)
        return lineList

####################################################################################################################################

def get_standard_options():
    from command_parser import parse_command, standardOptions
    opts = standardOptions + [  # h=help, c=commentChar, o=outFile
           {'ID': 'help'},
           {'ID': 'about'},
           {'ID': 'f', 'name': 'format', 'type': str, 'default': 'vSEan',
                       'constraint': 'format.lower() in ["o", "g", "h", "vs","vsa","vse","vsea","vseb","vsean","vseasn","vseasni","vseasnd","vseasnid","vseasndi"]'},
               {'ID': 'i', 'name': 'isotope', 'type': int, 'constraint': 'isotope>=0', 'default': 0},
               {'ID': 'm', 'name': 'molecule', 'type': str, 'constraint': 'len(re.split("[,;\s]",molecule))==1'},
               {'ID': 'S', 'name': 'strMin', 'type': float, 'constraint': 'strMin>=0.0', 'default': 0.0},
           {'ID': 'x', 'name': 'xLimits', 'type': Interval, 'constraint': 'xLimits.lower>=0.0'},
           {'ID': 'X', 'name': 'xUnit', 'type': str, 'default': 'cm-1',
                       'constraint': "xUnit in ['cm-1', 'mue', 'nm', 'Hz', 'kHz', 'MHz', 'GHz', 'THz']"},
               {'ID': 'v',  'name': 'verbose'}]
    
    return opts

if __name__ == "__main__":
    from command_parser import parse_command
    import re
    opts=get_standard_options()    
    lineFiles, options, commentChar, outFile = parse_command (opts, 1)

    options['verbose'] = 'verbose' in options
    mainOnly = options.get('molecule')=='main'
    print("--YOUR options---")
    print(options)
    print("------------")
#    options={'format': 'vSEan', 'isotope': 0, 'molecule': 'CO', 'strMin': 0.0, 'xLimits': Interval(2290.0,2350.0), 'xUnit': 'nm', 'verbose': False}
    print("---YOUR lineFile---")
    print(lineFiles[0]) #.par file
#    lineFiles[0]="../../../../data/hitemp/CO/05_HITEMP2019.par"
    print("------------")

    dictOfLineLists = higstract (lineFiles[0], **options)
    print("\(*_*)")
 #   print(dictOfLineLists)

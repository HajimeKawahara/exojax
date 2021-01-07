"""  Frequently used mathematical and physical constants. """

##############################################################################################################
#####  LICENSE issues:                                                                                   #####
#####                 This file is part of the Py4CAtS package.                                          #####
#####                 Copyright 2002 - 2019; Franz Schreier;  DLR-IMF Oberpfaffenhofen                   #####
#####                 Py4CAtS is distributed under the terms of the GNU General Public License;          #####
#####                 see the file ../license.txt in the parent directory.                               #####
##############################################################################################################

from math import pi, sqrt, log

ln2 = log(2.)
sqrtLn2 = sqrt(ln2)
recPi   = 1.0/pi
sqrtPi  = sqrt(pi)
recSqrtPi  = 1./sqrtPi

# Some fundamental physical constants, see  https://physics.nist.gov/cuu/Constants/

amu = 1.660538782e-24   # atomic mass unit [g]
c   = 2.99792458e+10    # speed of light       [cm/s]
h   = 6.62606896e-27    # Planck's constant    [erg.s]
k   = 1.3806504e-16     # Boltzmann's constant [erg/K]
C1  = 2.*pi * h * c**2  # 3.74177138E-05 erg cm**2 / s    first radiation constant
C2  = h * c / k         # 1.4387752 cm*K                  second radiation constant

# Constants found at NIST, October 2016
# k = 1.380 648 52 e-23    J/K
# amu = 1.660 539 040 e-27 kg
# h = 6.626 070 040 e-34   J*s
# c = 299 792 458          m/s

# pc = 3.08567758e18  # cm parsec
# AU = 14959787070000 # cm astronomical unit
# rSun = 6.96e10      # cm sun equatorial radius

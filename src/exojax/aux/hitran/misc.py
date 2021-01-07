""" A 'random' collection of some useful(?) functions. """

##############################################################################################################
#####  LICENSE issues:                                                                                   #####
#####                 This file is part of the Py4CAtS package.                                          #####
#####                 Copyright 2002 - 2019; Franz Schreier;  DLR-IMF Oberpfaffenhofen                   #####
#####                 Py4CAtS is distributed under the terms of the GNU General Public License;          #####
#####                 see the file ../license.txt in the parent directory.                               #####
##############################################################################################################

try:                        import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numeric python failed!')

pi = np.pi

try:
	from scipy.interpolate import interp1d
except ImportError as msg:
	print(str(msg) + '\nWARNING -- misc: "from scipy.interpolate import interp1d failed", use default numpy.interp')
else:
	pass # print(' from scipy.interpolate import pchip_interpolate')

try:
	from matplotlib.pyplot import xlim, ylim, text
except ImportError as msg:
	print (str(msg) + '\nWARNING --- misc:  matplotlib not available, no quicklook!')
	#raise SystemExit ('ERROR --- lines.lines_atlas:  matplotlib not available, no quicklook!')
else:
	pass # print 'from matplotlib import figure, plot, ...'

from exojax.aux.hitran.pairTypes import Interval
from exojax.aux.hitran.cgsUnits import cgs, wavelengthUnits
from exojax.aux.hitran.lagrange_interpolation import lagrange2_regularGrid, lagrange3_regularGrid, lagrange4_regularGrid


####################################################################################################################################

### this "generic" regrid function defined here is slightly slower (percent) than the regrid methods
### defined for the subclassed arrays (xsArray, acArray, odArray, wfArray, riArray).
### However, the times reported for the old (individual regrid's) and this new regrid also vary by some percent from job to job.
### And the new default np.interp is slightly faster (percent) than the old default method lagrange2_regularGrid.
### Finally, the new generic function is easier for code maintainance, esp. for a consistent selection of the interpolation method.

def regrid (yValues, newLen, method='L'):
	""" Regrid function values given on an equidistant/uniform grid to a new (usually denser) grid in the same interval.

	    yValues:  the function values to be interpolated with len(yValues)=oldLen
	    newLen:   the number of new function values after interpolation
	    method:   the interpolation method
	              integers 2, 3, 4  ===> the self-made linear, quadratic or cubic Lagrange
		      "linear", "quadratic", "cubic", etc. ===> scipy.interp1d
		      otherwise ===> numpy.interp

	    RETURNS:  yData   ---   the function values interpolated with len(yData)=newLen
	"""
	if len(yValues)==newLen:  return yValues

	if   method==2:
		# our self-made linear, quadratic or cubic Lagrange (see lagrange_interpolation module for timing tests)
		yData = lagrange2_regularGrid (yValues,newLen)
	elif method==3:
		yData = lagrange3_regularGrid (yValues,newLen)
	elif method==4:
		yData = lagrange4_regularGrid (yValues,newLen)
	elif method in 'linear nearest zero slinear quadratic cubic previous next'.split():
		# WARNING: scipy's "advanced" interpolations are much slower!
		oldGrid = Interval(0.0,1.0).grid(len(yValues))
		newGrid = Interval(0.0,1.0).grid(newLen)
		if len(yValues)>1e5:
			print("WARNING --- regrid scipy.interp1d ", method, len(yValues), ' ---> ', newLen, ' may take some time!')
		yData = interp1d(oldGrid, yValues, method)(newGrid)
	else:
		# numpy one-dimensional linear interpolation
		oldGrid = Interval(0.0,1.0).grid(len(yValues))
		newGrid = Interval(0.0,1.0).grid(newLen)
		yData = np.interp(newGrid, oldGrid, yValues)
	return  yData


####################################################################################################################################

def trapez (x, Y, xLow=None, xHigh=None):
	""" Integral_x[0]^x[-1] y(x) dx  with tabulated x,y values (2 arrays) using trapezoid quadrature.

	    ARGUMENTS:
	    ----------
	    x         a rank 1 array of grid points
	    Y         a rank 1 or rank 2 array
	              Y can be single or multi-dimensional, i.e. Y.shape=len(x) or Y.shape=[len(x),*]
	    xLow      lower integration limit; default None: start at x[0]
	    xHigh     upper integration limit; default None: start at x[-1]
	              if xLow or xHigh are not grid points, Lagrange two-point interpolation is used for the first/last interval.

	    RETURNS:
	    --------
	    the integral approximated by 0.5*sum (y[i]+y[i-1]) * (x[i]-x[i-1])

	    NOTE:
	    -----
	    An alternative implementation is given by numpy's `trapz` function;
	    however, this does not allow to set the limits.
	"""
	if isinstance(xLow,(int,float)) or isinstance(xHigh,(int,float)):
		if not xLow:   xLow = x[0]
		if not xHigh:  xHigh= x[-1]
		if xLow>=x[0] and xHigh<=x[-1]:
			iFirst, iLast = x.searchsorted([xLow,xHigh])
			xDelta = x[iFirst+1:iLast] - x[iFirst:iLast-1]  # grid point intervals
			if len(Y.shape)==1:
				ySum  = Y[iFirst+1:iLast] + Y[iFirst:iLast-1]
				yLow  = Y[iFirst-1] + (xLow-x[iFirst-1])*(Y[iFirst]-Y[iFirst-1])/(x[iFirst]-x[iFirst-1])
				# interpolate function values at end point
				yHigh = Y[iLast-1]  + (xHigh-x[iLast-1])*(Y[iLast]-Y[iLast-1])/(x[iLast]-x[iLast-1])
				integral = 0.5*(np.dot(xDelta,ySum) +
				                (x[iFirst]-xLow)*(Y[iFirst]+yLow) + (xHigh-x[iLast-1])*(yHigh+Y[iLast-1]))
			else:
				ySum  = Y[iFirst+1:iLast,:] + Y[iFirst:iLast-1,:]
				integral = 0.5*np.dot(xDelta,ySum)
				for i,y in enumerate(Y.T):
					yLow  = y[iFirst-1] + (xLow-x[iFirst-1])*(y[iFirst]-y[iFirst-1])/(x[iFirst]-x[iFirst-1])
					yHigh = y[iLast-1]  + (xHigh-x[iLast-1])*(y[iLast]-y[iLast-1])/(x[iLast]-x[iLast-1])
					integral[i] += 0.5*((x[iFirst]-xLow)*(y[iFirst]+yLow) + (xHigh-x[iLast-1])*(yHigh+y[iLast-1]))
		else:
			print('xGrid', x[0], '...', x[-1], '<---> integral limits', xLow, '...', xHigh)
			raise SystemExit ('ERROR --- trapez:  integral limits outside xGrid bounds')
	else:
		xDelta = x[1:] - x[:-1]  # grid point intervals
		if len(Y.shape)==1: integral = 0.5*np.dot(xDelta,Y[1:] + Y[:-1])
		else:               integral = 0.5*np.dot(xDelta,Y[1:,:] + Y[:-1,:])
	return integral


####################################################################################################################################

def runningAverage (xy, n=2, splitXY=False):
	""" Compute running average, i.e. sum up `n` consecutive array elements and divide by `n`.

	    ARGUMENTS:
	    ----------
	    xy        a rank 1 or rank 2 array
	    n         the number of elements to combine (default 2)
	    splitXY   flag, default False
	              if True and xy is a rank 2 array, separately return first column (xGrid) and further columns

	    RETURNS:
	    --------
	    a rank 1 or rank 2 array XY  (or a rank 1 array xGrid and rank 2 array yValues if splitXY)
	    the length of the returned array(s) is roughly xy.shape[0]/n
	"""
	mx = int(xy.shape[0]/n)  # number of data rows outgoing
	if len(xy.shape)==1:
		XY= np.array([sum(xy[i:i+n])/n for i in range(0,n*mx,n)])
	else:
		XY= np.array([sum(xy[i:i+n,j])/n for i in range(0,n*mx,n) for j in range(xy.shape[1])]).reshape(-1,xy.shape[1])
	# optionally return the xGrid separately
	if splitXY and len(xy.shape)==2:  return XY[:,0], XY[:,1:]
	else:                             return XY


####################################################################################################################################

def approx (this, other, eps=0.001):
	""" Return True if the two numbers (usually floats) are equal within the tolerance eps (default 0.001). """
	return abs(this-other) < eps*abs(this)


####################################################################################################################################

def float_in_list (value, floatList, eps=0.001):
        """ Return True if value is contained in a list of float values within the tolerance eps (default 0.001). """
        checkList = [approx(value,other,eps) for other in floatList]
        return any(checkList)


####################################################################################################################################

def monotone (data):
	""" Check data for monotonicity and return
	    +1   increasing monotonically
	    -1   decreasing monotonically
	     0   otherwise
	"""
	diff = np.ediff1d(data)
	if   all(diff>0):  return +1
	elif all(diff<0):  return -1
	else:              return 0


####################################################################################################################################

def xTruncate (xGrid, yValues, xLimits):
	""" Given an array of x grid points and a 'matrix' of y values (with len(xGrid) rows)
	    delete data outside the Interval defined by xLimits
	    and return the truncated grid and the corresponding 'matrix'. """
	if   isinstance(xLimits,(tuple,list)) and len(xLimits)==2:  xLimits=Interval(*xLimits)
	elif isinstance(xLimits,Interval):                          pass
	else:  raise SystemExit ("ERROR --- xTruncate:  xLimits is not an Interval (or pair of floats)")

	mask    = np.logical_and(np.greater_equal(xGrid,xLimits.lower), np.less_equal(xGrid,xLimits.upper))
	xGrid   = np.compress(mask, xGrid)
	yValues = np.compress(mask, yValues, 0)
	return xGrid, yValues


####################################################################################################################################

def wien (**kwArgs):
	""" Wien's displacement law:
	    for given temperature [K] return wavenumber (or wavelength) of maximum blackbody emission
	    or
	    for given wavenumber (or wavelength) return corresponding temperature.

	    KEYWORD ARGUMENTS:
	    x:      wavenumber or wavelength
	    T:      temperature [K]
	    xUnit:  cm-1 (default) or a wavelength unit
	"""

	args  = kwArgs.keys()
	if not args:  raise SystemExit ("ERROR --- wien:  neither x (wavenumber/length) nor T (temperature) specified!")

	for key in args:
		if key not in 'x T xUnit'.split():
			raise SystemExit ("ERROR --- wien:  invalid function keyword!")

	xUnit = kwArgs.get('xUnit','cm-1')
	if xUnit not in ['cm-1'] + list(wavelengthUnits.keys()):
			raise SystemExit ("ERROR --- wien:  invalid wavenumber / wavelength unit!")

	if 'x' in args and 'T' in args:
		raise SystemExit ("ERROR --- wien:  either specify x (wavenumber/length) or T (temperature)!")
	elif 'x' in args:
		# return temperature
		x = kwArgs.get('x')
		if xUnit=='cm-1':
			return x * 0.50995
		else:
			return 0.289777/cgs(xUnit,x)  # first convert wavelength to cm
	elif 'T' in args:
		# return wavenumber or wavelength
		temp = kwArgs.get('T')
		if xUnit=='cm-1':
			return temp / 0.50995   # cm-1
		else:
			lambdaMax=0.289777/temp  # cm
			return lambdaMax/wavelengthUnits[xUnit]
	else:
		raise SystemExit ("ERROR --- wien:  neither x (wavenumber/length) nor T (temperature) specified!")


####################################################################################################################################

def effHgt2transitDepth (effHgt, radiusPlanet=6371.23, radiusStar=696342.0):
	""" Return the additional transit depth from a given effective height.

	    effHgt:        effective height spectrum
	    radiusPlanet:  default Earth 6371.23km
	    radiusStar:    default Sun 696342km (https://de.wikipedia.org/wiki/Sonnenradius)

	    NOTE:  according to https://en.wikipedia.org/wiki/Solar_radius 695700km
	"""

	return ((radiusPlanet+effHgt)**2 - radiusPlanet**2) / radiusStar**2


####################################################################################################################################

def zenithAngle_boa2toa (beta, zToA=120., radiusEarth=6371.23, degree=False):
	""" Return the zenith angle at ToA (or nadir viewing observer) from given angle at BoA.
	    (beta=0 for vertical uplooking, alpha=pi=90dg for horizontal view) """

	if degree:  return 180.0 - np.arcsin((radiusEarth/(radiusEarth+zToA)) * np.cos((90-beta)*pi/180.)) * (180./pi)
	else:       return    pi - np.arcsin((radiusEarth/(radiusEarth+zToA)) * np.cos(pi/2 - beta))


def zenithAngle_toa2boa (alpha, zToA=120., radiusEarth=6371.23, degree=False):
	""" Return the viewing angle at BoA (or nadir viewing observer) from given zenith angle at ToA.
	    (alpha=0 for vertical uplooking, alpha=pi=180dg for downlooking observer) """

	if degree:  return 90.0 - np.arccos((radiusEarth+zToA)/radiusEarth * np.sin(alpha*pi/180.)) * (180./pi)
	else:       return   pi - np.arccos((radiusEarth+zToA)/radiusEarth * np.sin(alpha))


####################################################################################################################################

def show_lambda (wavenumbers=[100,200,500,800,1000,2000,2500,3333,4000,5000,8000,10000,12500]):
	""" Write wavelengths [mue] corresponding to x-axis wavenumbers on the top-axis of plot. """
	xMin,xMax=xlim()
	yMin,yMax=ylim()
	for nu in wavenumbers:
		if xMin<=nu<=xMax:
			text(nu,1.01*yMax,'%.1f$\mu$m' % (10000.0/nu), horizontalalignment='center')

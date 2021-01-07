"""
Lagrange 2, 3, and 4 point interpolation y(x) ---> Y(X)
for uniform, equidistant old and new x-grids
where the X interval is identical to the x-interval
(i.e. min(x)=min(X) and max(x)=max(X))

Y = lagrange2_regularGrid (y, n)  --- linear interpolation with len(Y) > len(y)
Y = lagrange3_regularGrid (y, n)  --- quadratic interpolation
Y = lagrange4_regularGrid (y, n)  --- cubic interpolation

Y = lagrange2_interpolate2 (y)    --- interpolation for len(Y)-1=2*(len(y)-1)
Y = lagrange3_interpolate2 (y)
Y = lagrange4_interpolate2 (y)

Y = lagrange2_interpolate4 (y)    --- interpolation for len(Y)-1=4*(len(y)-1)
Y = lagrange3_interpolate4 (y)
Y = lagrange4_interpolate4 (y)

Y = lagrange2_interpolate8 (y)    --- interpolation for len(Y)-1=8*(len(y)-1)
Y = lagrange3_interpolate8 (y)
Y = lagrange4_interpolate8 (y)
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

r6 = 1./6.

## Some tests on execution time:
#from scipy.interpolate import interp1d, pchip_interpolate, krogh_interpolate, barycentric_interpolate
#x = linspace(0.,2.*pi,23); y=np.sqrt(x)*sin(x)
#X = linspace(0.,2.*pi,200)
#%timeit Y = interp(X,x,y)                      #  2.35 microseconds per loop
#%timeit Y = lagrange2_regularGrid (y,len(X))   #  8.16 microseconds per loop
#%timeit Y = lagrange3_regularGrid (y,len(X))   #  15.1 microseconds per loop
#%timeit Y = lagrange4_regularGrid (y,len(X))   #  23.1 microseconds per loop
#print 'scipy.interp1d'
#%timeit Y = interp1d(x,y)(X)                   #  37.9 microseconds per loop
#%timeit Y = interp1d(x,y,'linear')(X)          #  38.1 microseconds per loop
#%timeit Y = interp1d(x,y,'quadratic')(X)       #  407 microseconds per loop
#%timeit Y = interp1d(x,y,'cubic')(X)           #  437 microseconds per loop
#print 'scipy.pchip, krogh, bary'
#%timeit Y = pchip_interpolate(x,y,X)           #  672 microseconds per loop
#%timeit Y = krogh_interpolate(x,y,X)           #  802 microseconds per loop
#%timeit Y = barycentric_interpolat(x,y,X)      #  153 microseconds per loop

#x = linspace(0.,2.*pi,2222); y=np.sqrt(x)*sin(x)
#X = linspace(0.,2.*pi,22222)
#%timeit Y = interp(X,x,y)                      #  409 microseconds per loop
#%timeit Y = lagrange2_regularGrid (y,len(X))   #  238 microseconds per loop
#%timeit Y = lagrange3_regularGrid (y,len(X))   #  395 microseconds per loop
#%timeit Y = lagrange4_regularGrid (y,len(X))   #  598 microseconds per loop
#%timeit Y = barycentric_interpolat(x,y,X)      #  382 milliseconds per loop

#plot (x,y,'b+', X,Y,'r')


####################################################################################################################################

def lagrange2_regularGrid (y, n):
	""" Lagrange 2 point interpolation for regular equidistant grid x to finer regular grid X of length n.
	    (assumes end points are identical:  x[0]=X[0] and x[-1]=X[-1]) """

	m=len(y)  # number of old grid points
	if n>m>1:
		ratio = float(m-1)/float(n-1)
		new   = ratio*np.arange(n)
		next  = np.floor(new).astype(int)
		next[-1]=next[-1]-1
		p     = new-next
		z     = (1-p)*y[next] + p*y[next+1]
	elif n<m:
		raise SystemExit ('ERROR -- lagrange2_regularGrid interpolation: output array y should be larger than input array!')
	elif m<2:
		raise SystemExit ('ERROR -- lagrange2_regularGrid interpolation: need at least 2 points on input!')
	else:
		z = y
	return z

####################################################################################################################################

def lagrange3_regularGrid (y, n):
	""" Lagrange 3 point interpolation for regular equidistant grid x to finer regular grid X of length n.
	    (assumes end points are identical:  x[0]=X[0] and x[-1]=X[-1]) """

	m=len(y)  # number of old grid points
	if n>m>1:
		ratio = float(m-1)/float(n-1)
		iRatio = int(1./ratio)
		new  = ratio*np.arange(n)
		next = np.floor(new).astype(int)
		next[:iRatio+1]=1; next[-1]=next[-1]-1
		p    = new-next
		z      = 0.5*p*(p-1.0)*y[next-1] + (1.0-p)*(1.0+p)*y[next] + 0.5*p*(p+1.0)*y[next+1]
	elif n<m:
		raise SystemExit ('ERROR -- lagrange2_regularGrid interpolation: output array y should be larger than input array!')
	elif m<3:
		print('WARNING --- lagrange3_regularGrid interpolation: need at least 4 points on input (using linear lagrange)!')
		z = lagrange2_regularGrid(y,n)
	else:
		z = y
	return z

####################################################################################################################################

def lagrange4_regularGrid (y, n):
	""" Lagrange 4 point interpolation for regular equidistant grid x to finer regular grid X of length n.
	    (assumes end points are identical:  x[0]=X[0] and x[-1]=X[-1]) """

	m=len(y)
	if n>m>3:
		ratio = float(m-1)/float(n-1)
		iRatio = int(1./ratio)
		new   = ratio*np.arange(n)
		next  = np.floor(new).astype(int)
		next[:iRatio+1]=1;  next[-iRatio-1:]=next[-iRatio-2]
		p    = new-next
		z    = 0.5*(p-1.0)*(p+1.0)*(p-2.0)*y[next] \
		        - r6*p*(p-1.0)*(p-2.0)*y[next-1] - 0.5*p*(p+1.0)*(p-2.0)*y[next+1] + r6*p*(p-1.0)*(p+1.0)*y[next+2]
	elif n<m:
		raise SystemExit ('ERROR -- lagrange4_regularGrid interpolation: output array y should be larger than input array!')
	elif m<4:
		print('WARNING --- lagrange4_regularGrid interpolation: need at least 4 points on input (using lower order lagrange)!')
		z = lagrange3_regularGrid(y,n)
	else:
		z = y
	return z


####################################################################################################################################
##########         Lagrange linear interpolation with number of grid intervals increased by factor 2**n                   ##########
####################################################################################################################################

def lagrange2_interpolate2 (y):
	""" Lagrange 2-point interpolation with doubled number of intervals. """

	ny = len(y)
	if ny<2:  raise SystemExit (' ERROR  lagrange2_interpolate2: ny<2\n' + np.array2string(y))

	# initialize and copy old data to every second new data element
	Y = np.empty(2*ny-1)
	Y[::2]  = y                   # for i in range(ny):   Y[2*i]   = y[i]
	Y[1::2] = 0.5*(y[:-1]+y[1:])  # for i in range(ny-1): Y[2*i+1] = 0.5*(y[i]+y[i+1])
	return Y

####################################################################################################################################

def lagrange2_interpolate4 (y):
	""" Lagrange 2-point interpolation with quadrupled number of intervals. """

	ny = len(y)
	if ny<2:  raise SystemExit(' ERROR  lagrange2_interpolate4: ny<2\n' + np.array2string(y))

	# initialize and copy old data to every fourth new data element
	Y = np.empty(4*ny-3)
	Y[ ::4] = y
	Y[1::4] = 0.75*y[:-1]+0.25*y[1:]
	Y[2::4] = 0.5*(y[:-1]+y[1:])
	Y[3::4] = 0.25*y[:-1]+0.75*y[1:]
	return Y

####################################################################################################################################

def lagrange2_interpolate8 (y):
	""" Lagrange 2-point interpolation with eightfold number of intervals. """

	ny = len(y)
	if ny<2:  raise SystemExit(' ERROR --- lagrange2_interpolate8: ny<2\n' + np.array2string(y))

	# initialize and copy old data to every eighth new data element
	Y = np.empty(8*ny-7)
	Y[ ::8] = y
	Y[1::8] = 0.875*y[:-1] + 0.125*y[1:]
	Y[2::8] = 0.750*y[:-1] + 0.250*y[1:]
	Y[3::8] = 0.625*y[:-1] + 0.375*y[1:]
	Y[4::8] = 0.500*(y[:-1]+y[1:])
	Y[5::8] = 0.375*y[:-1] + 0.625*y[1:]
	Y[6::8] = 0.250*y[:-1] + 0.750*y[1:]
	Y[7::8] = 0.125*y[:-1] + 0.875*y[1:]
	return Y

####################################################################################################################################
##########         Lagrange quadratic interpolation with number of grid intervals increased by factor 2**n                ##########
##########                                                                                                                ##########
##########    yNew = 0.5*delta*(delta-1.)*yLeft + (1.-delta)*(1.+delta)*yMid + 0.5*delta*(delta+1.)*yRight                ##########
##########    delta = xNew-xMid                                                                                           ##########
####################################################################################################################################

def lagrange3_interpolate2 (y):
	""" Lagrange 3-point interpolation with doubled number of intervals. """

	ny = len(y)
	if ny<3:
		print(' WARNING --- lagrange3_interpolate2: ny<4   ' + np.array2string(y) + '\ntrying lagrange2_interpolate2')
		return lagrange2_interpolate2(y)

	# initialize and copy old data to every second new data element
	Y = np.empty(2*ny-1)
	Y[::2]  = y
	# special treatment for first interval i=0
	Y[1] = 0.375*y[0] + 0.750*y[1] - 0.125*y[2]  # delta=-0.5
	# loop over inner intervals
	Y[3:-2:2] = -0.0625*y[:-3] + 0.5625*y[1:-2] + 0.5625*y[2:-1] - 0.0625*y[3:]
	# and again a special treatment for last interval i=len(y)-1
	Y[-2] = -0.125*y[-3] + 0.750*y[-2] + 0.375*y[-1]   # delta=0.5
	return Y

####################################################################################################################################

def lagrange3_interpolate4 (y):
	""" Lagrange 3-point interpolation with quadrupled number of intervals. """

	ny = len(y)
	if ny<3:
		print(' WARNING --- lagrange3_interpolate4: ny<4   ' + np.array2string(y) + '\ntrying lagrange2_interpolate4')
		return lagrange2_interpolate4(y)

	# initialize and copy old data to every fourth new data element
	Y = np.empty(4*ny-3)
	Y[::4]  = y
	# special treatment for first interval i=0
	Y[1] = 0.65625*y[0] + 0.4375*y[1] - 0.09375*y[2]  # delta=-0.75
	Y[2] = 0.37500*y[0] + 0.7500*y[1] - 0.12500*y[2]  # delta=-0.5
	Y[3] = 0.15625*y[0] + 0.9375*y[1] - 0.09375*y[2]  # delta=-0.25
	# loop over inner intervals
	Y[5:-6:4] = -0.09375*y[:-3] + 0.93750*y[1:-2] + 0.15625*y[2:-1]
	Y[6:-5:4] = -0.06250*y[:-3] + 0.56250*y[1:-2] + 0.56250*y[2:-1] - 0.06250*y[3:]
	Y[7:-4:4] =                   0.15625*y[1:-2] + 0.93750*y[2:-1] - 0.09375*y[3:]
	# and again a special treatment for last interval i=len(y)-1
	Y[-4] = -0.09375*y[-3] + 0.9375*y[-2] + 0.15625*y[-1]  # delta=0.25
	Y[-3] = -0.12500*y[-3] + 0.7500*y[-2] + 0.3750*y[-1]   # delta=0.5
	Y[-2] = -0.09375*y[-3] + 0.4375*y[-2] + 0.65625*y[-1]  # delta=0.75
	return Y

####################################################################################################################################

def lagrange3_interpolate8 (y):
	""" Lagrange 3-point interpolation with eightfold number of intervals. """

	ny = len(y)
	if ny<3:
		print(' WARNING --- lagrange3_interpolate8: ny<4   ' + np.array2string(y) + '\ntrying lagrange2_interpolate8')
		return lagrange2_interpolate8(y)

	# initialize and copy old data to every eighth new data element
	Y = np.empty(8*ny-7)
	Y[::8] = y
	# special treatment for first interval i=0
	Y[1] = 0.8203125*y[0] + 0.234375*y[1] - 0.0546875*y[2]  # delta=-0.875
	Y[2] = 0.6562500*y[0] + 0.437500*y[1] - 0.0937500*y[2]  # delta=-0.75
	Y[3] = 0.5078125*y[0] + 0.609375*y[1] - 0.1171875*y[2]  # delta=-0.625
	Y[4] = 0.3750000*y[0] + 0.750000*y[1] - 0.1250000*y[2]  # delta=-0.5
	Y[5] = 0.2578125*y[0] + 0.859375*y[1] - 0.1171875*y[2]  # delta=-0.375
	Y[6] = 0.1562500*y[0] + 0.937500*y[1] - 0.0937500*y[2]  # delta=-0.25
	Y[7] = 0.0703125*y[0] + 0.984375*y[1] - 0.0546875*y[2]  # delta=-0.125
	# loop over inner intervals
	Y[ 9:-15:8] = -0.0546875*y[:-3] + 0.9843750*y[1:-2] + 0.0703125*y[2:-1]                    # delta=0.125
	Y[10:-14:8] = -0.0937500*y[:-3] + 0.9375000*y[1:-2] + 0.1562500*y[2:-1]                    # delta=0.25
	Y[11:-13:8] = -0.1171875*y[:-3] + 0.8593750*y[1:-2] + 0.2578125*y[2:-1]                    # delta=0.375
	Y[12:-12:8] = -0.0625000*y[:-3] + 0.5625000*y[1:-2] + 0.5625000*y[2:-1] - 0.0625000*y[3:]  # delta=0.5
	Y[13:-11:8] =                     0.2578125*y[1:-2] + 0.8593750*y[2:-1] - 0.1171875*y[3:]  # delta=-0.375
	Y[14:-10:8] =                     0.1562500*y[1:-2] + 0.9375000*y[2:-1] - 0.0937500*y[3:]  # delta=-0.25
	Y[15: -9:8] =                     0.0703125*y[1:-2] + 0.9843750*y[2:-1] - 0.0546875*y[3:]  # delta=-0.125
	# and again a special treatment for last interval i=len(y)-1
	Y[-8] = -0.0546875*y[-3] + 0.984375*y[-2] + 0.0703125*y[-1]  # delta=0.125
	Y[-7] = -0.09375*y[-3] + 0.9375*y[-2] + 0.15625*y[-1]        # delta=0.25
	Y[-6] = -0.1171875*y[-3] + 0.859375*y[-2] + 0.2578125*y[-1]  # delta=0.375
	Y[-5] = -0.12500*y[-3] + 0.7500*y[-2] + 0.3750*y[-1]         # delta=0.5
	Y[-4] = -0.1171875*y[-3] + 0.609375*y[-2] + 0.5078125*y[-1]  # delta=0.625
	Y[-3] = -0.09375*y[-3] + 0.4375*y[-2] + 0.65625*y[-1]        # delta=0.75
	Y[-2] = -0.0546875*y[-3] + 0.234375*y[-2] + 0.8203125*y[-1]  # delta=0.875

	# if min(Y)<0.0:  Y = np.where(Y>0.0, Y, lagrange2_interpolate8(y))
	return Y

####################################################################################################################################
##########         Lagrange cubic interpolation with number of grid intervals increased by factor 2**n                    ##########
##########                                                                                                                ##########
##########      yNew = 1/6*p*(p-1)*(p-2)*yL + (p-1)*(p+1)*(p-2)*yM/2 + p*(p+1)*(p-2)*yR/2 + 1/6*p*(p-1)*(p+1)*yRR         ##########
##########      p    = xNew-xMid                                                                                          ##########
####################################################################################################################################

def lagrange4_interpolate2 (y):
	""" Lagrange 4-point interpolation with doubled number of intervals. """

	ny = len(y)
	if ny<4:
		print(' WARNING --- lagrange4_interpolate2: ny<4   ' + np.array2string(y) + '\ntrying lagrange3_interpolate2')
		return lagrange3_interpolate2(y)

	# initialize and copy old data to every second new data element
	Y = np.empty(2*ny-1)
	Y[::2]  = y
	# special treatment for first interval i=0
	Y[1] = +0.3125000*y[0] + 0.9375000*y[1] - 0.3125000*y[2] + 0.0625000*y[3]  # delta=-0.5
	# loop over inner intervals
	Y[3:-2:2] = -0.0625*y[:-3] + 0.5625*y[1:-2] + 0.5625*y[2:-1] - 0.0625*y[3:]
	# and again a special treatment for last interval i=len(y)-1
	Y[-2] = 0.0625000*y[-4] - 0.3125000*y[-3] + 0.9375000*y[-2] + 0.3125000*y[-1]  # delta=1.5
	return Y

####################################################################################################################################

def lagrange4_interpolate4 (y):
	""" Lagrange 4-point interpolation with quadrupled number of intervals. """

	ny = len(y)
	if ny<4:
		print(' WARNING --- lagrange4_interpolate4: ny<4   ' + np.array2string(y) + '\ntrying lagrange3_interpolate4')
		return lagrange3_interpolate4(y)

	# initialize and copy old data to every fourth new data element
	Y = np.empty(4*ny-3)
	Y[::4]  = y
	# special treatment for first interval i=0
	Y[1] = 0.6015625*y[0] + 0.6015625*y[1] - 0.2578125*y[2] + 0.0546875*y[3]  # delta=-0.75
	Y[2] = 0.3125000*y[0] + 0.9375000*y[1] - 0.3125000*y[2] + 0.0625000*y[3]  # delta=-0.5
	Y[3] = 0.1171875*y[0] + 1.0546875*y[1] - 0.2109375*y[2] + 0.0390625*y[3]  # delta=-0.25
	# loop over inner intervals
	Y[5:-7:4] = -0.0546875*y[:-3] + 0.8203125*y[1:-2] + 0.2734375*y[2:-1] - 0.0390625*y[3:]
	Y[6:-6:4] = -0.0625000*y[:-3] + 0.5625000*y[1:-2] + 0.5625000*y[2:-1] - 0.0625000*y[3:]
	Y[7:-5:4] = -0.0390625*y[:-3] + 0.2734375*y[1:-2] + 0.8203125*y[2:-1] - 0.0546875*y[3:]
	# and again a special treatment for last interval i=len(y)-1
	Y[-4] = 0.0390625*y[-4] - 0.2109375*y[-3] + 1.0546875*y[-2] + 0.1171875*y[-1]  # delta=1.25
	Y[-3] = 0.0625000*y[-4] - 0.3125000*y[-3] + 0.9375000*y[-2] + 0.3125000*y[-1]  # delta=1.5
	Y[-2] = 0.0546875*y[-4] - 0.2578125*y[-3] + 0.6015625*y[-2] + 0.6015625*y[-1]  # delta=1.75
	return Y

####################################################################################################################################

def lagrange4_interpolate8 (y):
	""" Lagrange 4-point interpolation with eightfold number of intervals. """

	ny = len(y)
	if ny<4:
		print(' WARNING --- lagrange4_interpolate8: ny<4   ' + np.array2string(y) + '\ntrying lagrange3_interpolate8')
		return lagrange3_interpolate8(y)

	# initialize and copy old data to every eighth new data element
	Y = np.empty(8*ny-7)
	Y[::8] = y
	# special treatment for first interval i=0
	Y[1] = 0.7861328125*y[0] + 0.3369140625*y[1] - 0.1572265625*y[2] + 0.0341796875*y[3]  # p=-7/8=-0.875
	Y[2] = 0.6015625000*y[0] + 0.6015625000*y[1] - 0.2578125000*y[2] + 0.0546875000*y[3]  # p=-6/8=-0.75
	Y[3] = 0.4443359375*y[0] + 0.7998046875*y[1] - 0.3076171875*y[2] + 0.0634765625*y[3]  # p=-5/8=-0.625
	Y[4] = 0.3125000000*y[0] + 0.9375000000*y[1] - 0.3125000000*y[2] + 0.0625000000*y[3]  # p=-4/8=-0.5
	Y[5] = 0.2041015625*y[0] + 1.0205078125*y[1] - 0.2783203125*y[2] + 0.0537109375*y[3]  # p=-3/8=-0.375
	Y[6] = 0.1171875000*y[0] + 1.0546875000*y[1] - 0.2109375000*y[2] + 0.0390625000*y[3]  # p=-2/8=-0.25
	Y[7] = 0.0498046875*y[0] + 1.0458984375*y[1] - 0.1162109375*y[2] + 0.0205078125*y[3]  # p=-1/8=-0.125
	# loop over inner intervals
	Y[ 9:-15:8] = -0.0341796875*y[:-3] + 0.9228515625*y[1:-2] + 0.1318359375*y[2:-1] - 0.0205078125*y[3:]  # p=0.125
	Y[10:-14:8] = -0.0546875000*y[:-3] + 0.8203125000*y[1:-2] + 0.2734375000*y[2:-1] - 0.0390625000*y[3:]  # p=0.25
	Y[11:-13:8] = -0.0634765625*y[:-3] + 0.6982421875*y[1:-2] + 0.4189453125*y[2:-1] - 0.0537109375*y[3:]  # p=0.375
	Y[12:-12:8] = -0.0625000000*y[:-3] + 0.5625000000*y[1:-2] + 0.5625000000*y[2:-1] - 0.0625000000*y[3:]  # delta=0.5
	Y[13:-11:8] = -0.0537109375*y[:-3] + 0.4189453125*y[1:-2] + 0.6982421875*y[2:-1] - 0.0634765625*y[3:]  # p=0.625
	Y[14:-10:8] = -0.0390625000*y[:-3] + 0.2734375000*y[1:-2] + 0.8203125000*y[2:-1] - 0.0546875000*y[3:]  # p=0.75
	Y[15: -9:8] = -0.0205078125*y[:-3] + 0.1318359375*y[1:-2] + 0.9228515625*y[2:-1] - 0.0341796875*y[3:]  # p=0.875
	# and again a special treatment for last interval i=len(y)-1
	Y[-8] = 0.0205078125*y[-4] - 0.1162109375*y[-3] + 1.0458984375*y[-2] + 0.0498046875*y[-1]  # p=1.125
	Y[-7] = 0.0390625000*y[-4] - 0.2109375000*y[-3] + 1.0546887500*y[-2] + 0.1171875000*y[-1]  # p=1.25
	Y[-6] = 0.0537109375*y[-4] - 0.2783203125*y[-3] + 1.0205078125*y[-2] + 0.2041015625*y[-1]  # p=1.375
	Y[-5] = 0.0625000000*y[-4] - 0.3125000000*y[-3] + 0.9375000000*y[-2] + 0.3125000000*y[-1]  # p=1.5
	Y[-4] = 0.0634765625*y[-4] - 0.3076171875*y[-3] + 0.7998046875*y[-2] + 0.4443359375*y[-1]  # p=1.625
	Y[-3] = 0.0546875000*y[-4] - 0.2578125000*y[-3] + 0.6015625000*y[-2] + 0.6015625000*y[-1]  # p=1.75
	Y[-2] = 0.0341796875*y[-4] - 0.1572265625*y[-3] + 0.3369140625*y[-2] + 0.7861328125*y[-1]  # p=1.875
	return Y

####################################################################################################################################

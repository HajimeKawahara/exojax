""" A collection of some special data types. """

##############################################################################################################
#####  LICENSE issues:                                                                                   #####
#####                 This file is part of the Py4CAtS package.                                          #####
#####                 Copyright 2002 - 2019; Franz Schreier;  DLR-IMF Oberpfaffenhofen                   #####
#####                 Py4CAtS is distributed under the terms of the GNU General Public License;          #####
#####                 see the file ../license.txt in the parent directory.                               #####
##############################################################################################################

try:                      import numpy as np
except ImportError as msg:  raise SystemExit (str(msg) + '\nimport numpy (numeric python) failed!')

####################################################################################################################################

class PairOfInts:
	""" A kind of sequence / set with just two integer elements. """
	def __init__ (self, *pair):
		try:
			left, right = pair
			if isinstance(left,(float,int)) and isinstance(right,(float,int)):
				self.left  = left
				self.right = right
			else:
				print("ERROR:  PairOfInts must be floats or integers!")
		except ValueError:
			raise SystemExit ('ERROR:  PairOfInts initialization requires a pair of floats (or integers)!')
		except:
			raise SystemExit ("ERROR:  PairOfInts initialization failed!")
	def __str__ (self):
		return  'PairOfInts [%g,%g]' % (self.left,self.right)
	def __repr__ (self):
		return 'PairOfInts(%s,%s)' % (self.left,self.right)
	def __add__ (self,other):
		if isinstance(other,PairOfInts):
			return PairOfInts(self.left+other.left, self.right+other.right)
		elif isinstance(other,(float,int)):
			return PairOfInts(self.left+other,self.right+other)
		else:
			print('ERROR: other is not a PairOfInts (instance) or an integer or float!')
	def __radd__ (self, other):
		return PairOfInts(self.left+other, self.right+other)
	def __sub__ (self,other):
		if isinstance(other,PairOfInts):
			return PairOfInts(self.left-other.left, self.right-other.right)
		elif isinstance(other,(float,int)):
			return PairOfInts(self.left-other,self.right-other)
		else:
			print('ERROR: other is not a PairOfInts (instance) or an integer or float!')
	def __rsub__ (self, other):
		return PairOfInts(other-self.left, other-self.right)
	def __mul__ (self, other):
		if isinstance(other,PairOfInts):
			return PairOfInts(self.left*other.left, self.right*other.right)
		elif isinstance(self,PairOfInts) and isinstance(other,(float,int)):
			return PairOfInts(other*self.left, other*self.right)
		else:
			print('ERROR: other is not a PairOfInts (instance) or an integer or float!')
	def __rmul__ (self, other):
		return PairOfInts(other*self.left, other*self.right)
	def __div__ (self, other):
		if isinstance(other,PairOfInts):
			return PairOfInts(self.left/other.left, self.right/other.right)
		elif isinstance(self,PairOfInts) and isinstance(other,(float,int)):
			return PairOfInts(self.left/other, self.right/other)
		else:
			print('ERROR: other is not a PairOfInts (instance) or an integer or float!')
	def __rdiv__ (self, other):
		return PairOfInts(float(other)/self.left, float(other)/self.right)
	def __eq__ (self,other):
		if isinstance(other,PairOfInts):
			return self.left==other.left and self.right==other.right
		else:
			print('ERROR: other is not a PairOfInts (instance)!')
	def distinct (self):
		return self.left!=self.right
	def min (self):
		return min(self.left,self.right)
	def max (self):
		return max(self.left,self.right)
	def list (self):
		return [self.left,self.right]


####################################################################################################################################

class PairOfFloats:
	""" A kind of sequence / set with just two float elements. """
	def __init__ (self, *pair):
		try:
			left, right = pair
			if isinstance(left,(float,int)) and isinstance(right,(float,int)):
				self.left  = left
				self.right = right
			else:
				print("ERROR:  PairOfFloats must be floats or integers!")
		except ValueError:
			raise SystemExit ('ERROR:  PairOfFloats initialization requires a pair of floats (or integers)!')
		except:
			raise SystemExit ("ERROR:  PairOfFloats initialization failed!")
	def __str__ (self):
		return  'PairOfFloats [%g,%g]' % (self.left,self.right)
	def __repr__ (self):
		return 'PairOfFloats(%s,%s)' % (self.left,self.right)
	def __add__ (self,other):
		if isinstance(other,PairOfFloats):
			return PairOfFloats(self.left+other.left, self.right+other.right)
		elif isinstance(other,(float,int)):
			return PairOfFloats(self.left+other,self.right+other)
		else:
			print('ERROR: other is not a PairOfFloats (instance) or an integer or float!')
	def __radd__ (self, other):
		return PairOfFloats(self.left+other, self.right+other)
	def __sub__ (self,other):
		if isinstance(other,PairOfFloats):
			return PairOfFloats(self.left-other.left, self.right-other.right)
		elif isinstance(other,(float,int)):
			return PairOfFloats(self.left-other,self.right-other)
		else:
			print('ERROR: other is not a PairOfFloats (instance) or an integer or float!')
	def __rsub__ (self, other):
		return PairOfFloats(other-self.left, other-self.right)
	def __mul__ (self, other):
		if isinstance(other,PairOfFloats):
			return PairOfFloats(self.left*other.left, self.right*other.right)
		elif isinstance(self,PairOfFloats) and isinstance(other,(float,int)):
			return PairOfFloats(other*self.left, other*self.right)
		else:
			print('ERROR: other is not a PairOfFloats (instance) or an integer or float!')
	def __rmul__ (self, other):
		return PairOfFloats(other*self.left, other*self.right)
	def __div__ (self, other):
		if isinstance(other,PairOfFloats):
			return PairOfFloats(self.left/other.left, self.right/other.right)
		elif isinstance(self,PairOfFloats) and isinstance(other,(float,int)):
			return PairOfFloats(self.left/other, self.right/other)
		else:
			print('ERROR: other is not a PairOfFloats (instance) or an integer or float!')
	def __rdiv__ (self, other):
		return PairOfFloats(float(other)/self.left, float(other)/self.right)
	def __eq__ (self,other):
		if isinstance(other,PairOfFloats):
			return self.left==other.left and self.right==other.right
		else:
			print('ERROR: other is not a PairOfFloats (instance)!')
	def approx (self,other,eps=0.001):
		if isinstance(other,PairOfFloats):
			return abs(self.left-other.left)<eps*self.left and abs(self.right-other.right)<eps*self.right
		else:
			print('ERROR: other is not a PairOfFloats (instance)!')
	def list (self):
		return [self.left,self.right]
	def min (self):
		return min(self.left,self.right)
	def max (self):
		return max(self.left,self.right)
	def swap (self):
		return PairOfFloats(self.right, self.left)


####################################################################################################################################

class Interval:
	""" A kind of sequence / set with just two float elements definind an interval/range. """
	def __init__ (self, *limits):
		try:
			lower, upper = limits
			if isinstance(lower,(np.float32,np.float64)):      # ? np.float128 ?
				if lower.ndim==0:  lower=float(lower)
			if isinstance(upper,(np.float32,np.float64)):      # ? np.float128 ?
				if upper.ndim==0:  upper=float(upper)
			if isinstance(lower,(float,int)) and isinstance(upper,(float,int)):
				self.lower = min(lower,upper)
				self.upper = max(lower,upper)
			else:
				raise SystemExit ("ERROR:  Interval bounds must be floats or integers!")
		except ValueError:
			raise SystemExit ('ERROR:  Interval initialization requires a pair of floats (or integers)!')
		except Exception as errMsg:
			raise SystemExit ("ERROR:  Interval initialization failed!\n"+str(errMsg))
	def __str__ (self):
		return  'Interval [%g,%g]' % (self.lower,self.upper)
	def __repr__ (self):
		return 'Interval(%s,%s)' % (self.lower,self.upper)
	def limits (self):
		return self.lower, self.upper
	def __contains__ (self, other):
		if   isinstance(other,Interval):     return self.lower <= other.lower <= other.upper <= self.upper
		elif isinstance(other,(int,float)):  return self.lower <= other <= self.upper
	def member (self, value):
		""" Test if other is inside the interval, i.e., lower<=value<=upper. """
		if isinstance(value,(int,float)):    return self.lower <= value <= self.upper
		else:                                print("ERROR:  value must be an integer or float!")
	def part (self, other):
		""" Test if interval is inside other . """
		if isinstance(other,Interval):  return other.lower <= self.lower <= self.upper <= other.upper
		else:                           raise SystemExit ('ERROR --- Interval:  not an Interval')
	def inside (self, value):
		""" Test if other is inside the interval, i.e., lower<value<upper. """
		return self.lower < value < self.upper
	def intersect (self, other):
		""" Return the common intersect of two intervals. """
		if Interval.overlap(self,other): return Interval(max(self.lower,other.lower),
		                                                 min(self.upper,other.upper))
	def overlap (self, other):
		""" Test if the two intervals have a common intersect. """
		if isinstance(other,Interval):  return self.upper>other.lower and other.upper>self.lower
		else:                           raise SystemExit ('ERROR --- Interval:  not an Interval')
	def grid (self, nPoints=11, logGrid=False):
		""" Generate an equidistant, uniform grid with nPoints-1 intervals. """
		if isinstance(nPoints,int) and nPoints>0:
			if logGrid:  return np.logspace(self.lower,self.upper,nPoints)
			else:        return np.linspace(self.lower,self.upper,nPoints)
		else:
			raise SystemExit ('ERROR --- Interval:  number of grid points NOT a positive integer')
	def __add__ (self,other):
		if isinstance(other,Interval):
			if min(self.upper,other.upper)>max(self.lower,other.lower):
				return Interval(min(self.lower,other.lower),
				                max(self.upper,other.upper))
			else:
				return
		elif isinstance(other,(float,int)):
			if other>=0.0:
				return Interval(self.lower-other,self.upper+other)
			else:
				raise SystemExit ('ERROR --- Interval:  positive float or integer required!')
		else:
			raise SystemExit ('ERROR --- Interval: other is not an Interval (instance)!')
	def __radd__ (self, other):
		return Interval(self.lower-other, self.upper+other)
	def shift (self, other):
		""" shift the interval limits by a constant. """
		if isinstance(other,(float,int)):
			return Interval(self.lower+other, self.upper+other)
		else:
			raise SystemExit ('ERROR: argument of Interval shift must be float or integrer!')
	def __sub__ (self,other):
		if isinstance(other,Interval):
			if min(self.upper,other.upper)>max(self.lower,other.lower):
				return Interval(max(self.lower,other.lower),
				                min(self.upper,other.upper))
			else:
				return
		elif isinstance(other,(float,int)):
			if self.upper-self.lower>2.*other: return Interval(self.lower+other,self.upper-other)
		else:
			raise SystemExit ('ERROR --- Interval: other is not an Interval (instance)!')
	def __eq__ (self,other):
		""" Interval equality test with == """
		if isinstance(other,Interval):
			return self.lower==other.lower and self.upper==other.upper
		else:
			raise SystemExit ('ERROR: other is not an Interval (instance)!')
	def __cmp__(self, other):
		""" Interval comparison using cmp(Interval(,),Interval(,)),  returns -1, 0, or +1. """
		#return cmp(self.upper-self.lower,other.upper-other.lower)   # python2 builtin function obsolete!
		if   self.upper-self.lower<other.upper-other.lower:  return -1
		elif self.upper-self.lower>other.upper-other.lower:  return +1
		else:                                                return  0
	def approx (self,other,eps=0.001):
		if isinstance(other,Interval):
			return abs(self.lower-other.lower)<eps*self.lower and abs(self.upper-other.upper)<eps*self.upper
		else:
			raise SystemExit ('ERROR --- Interval: other is not an Interval (instance)!')
	def __len__ (self):
		return int(self.upper-self.lower)
	def __bool__ (self):
		return bool(self.upper-self.lower)
	def size (self):
		return float(self.upper-self.lower)
	def __mul__ (self, other):
		if isinstance(self,Interval) and isinstance(other,(float,int)):
			return Interval(other*self.lower, other*self.upper)
		else:
			raise SystemExit ('ERROR --- Interval: no multiplication of Intervals!')
	def __rmul__ (self, other):
		return Interval(other*self.lower, other*self.upper)
	def __div__ (self, other):
		if isinstance(self,Interval) and isinstance(other,(float,int)):
			return Interval(self.lower/other, self.upper/other)
		else:
			raise SystemExit ('ERROR --- Interval: no division of Intervals!')
	def __rdiv__ (self, other):
		return Interval(other/self.upper, other/self.lower)


####################################################################################################################################
###### not really a pair, but also useful for option parsing                                                                  ######
####################################################################################################################################

# see Langtangen py4cs p. 398

class ListOfInts (list):
	""" A special list with only integers allowed. """
	def __init__ (self, *someList):
		if all([isinstance(l,int) for l in someList]):
			list.__init__ (self,someList)
		else:
			raise SystemExit ('ERROR ListOfInts:  some of the list elements are non-integers!')
	def __add__ (self,other):
		return ListOfInts(list.__add__(self,other))
	def __iadd__ (self,other):
		return ListOfInts(list.__iadd__(self,other))
	def __setitem__ (self, i, item):
		if isinstance(item,int):  list.__setitem__(self, i, item)
		else:                     raise SystemExit ('ERROR --- ListOfInts:  list item to set in not an integer!')
	def __setslice (self, slice, otherList):
		if isinstance(otherList,int):  list.__setslice__(self, slice, otherList)
		else:                          raise SystemExit ('ERROR --- ListOfInts:  RHS is not a ListOfInts (list contains some non-integers)!')
	def append (self, item):
		if isinstance(item, int):  list.append(self, item)
		else:                      raise SystemExit ('ERROR --- ListOfInts:  new list element is not an integer!')
	def extend (self, otherList):
		if all([isinstance(l, int) for l in otherList]): list.extend(self, otherList)
		else:                                            raise TypeError ('ERROR --- ListOfInts:  list to be extended with non-integer(s)!')
	def insert (self, index, item):
		if isinstance(item, int):  list.insert(self, index, item)
		else:                      raise SystemExit ('ERROR --- ListOfInts:  list element to be replaced is not an integer!')

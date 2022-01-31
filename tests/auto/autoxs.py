import matplotlib.pyplot as plt
import numpy
from exojax.spec import AutoXS
import pytest
import jax.numpy as jnp
logplot = True
nus = numpy.linspace(1900.0, 2300.0, 40000, dtype=numpy.float64)
nuslog = numpy.logspace(numpy.log10(1900.0), numpy.log10(
    2300.0), 40000, dtype=numpy.float64)


# using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.
autoxs = AutoXS(nuslog, 'ExoMol', 'CO', xsmode='MODIT')
xsv0 = autoxs.xsection(1000.0, 1.0)  # cross section for 1000K, 1bar (cm2)
print(xsv0)
plt.plot(nuslog, xsv0, label='MODIT')
plt.legend()
plt.show()

# using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.
autoxs = AutoXS(nuslog, 'ExoMol', 'CO', xsmode='LPF')
xsv1 = autoxs.xsection(1000.0, 1.0)  # cross section for 1000K, 1bar (cm2)
dif = (numpy.sum((xsv0-xsv1)**2))

print('difference')
print('MODIT-LPF:', dif)

plt.plot(nus, xsv0, label='MODIT')
plt.plot(nus, xsv1, '.', label='LPF', alpha=0.1)
if logplot:
    plt.yscale('log')
plt.legend()
plt.show()


# using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.
autoxs = AutoXS(nus, 'ExoMol', 'CO', xsmode='DIT')
xsv0 = autoxs.xsection(1000.0, 1.0)  # cross section for 1000K, 1bar (cm2)

# using ExoMol CO (12C-16O). HITRAN and HITEMP are also supported.
autoxs = AutoXS(nus, 'ExoMol', 'CO', xsmode='LPF')
xsv1 = autoxs.xsection(1000.0, 1.0)  # cross section for 1000K, 1bar (cm2)
dif = (numpy.sum((xsv0-xsv1)**2))

plt.plot(nus, xsv0, label='DIT')
plt.plot(nus, xsv1, '.', label='LPF', alpha=0.1)
if logplot:
    plt.yscale('log')
plt.legend()
plt.show()

print('difference')
print('DIT-LPF:', dif)

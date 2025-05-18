Extra Databases (``xdb``)
==============================

The extra database (xdb) is a general term for classes that handle databases not conforming to the formats of other databases 
such as ``mdb``, ``adb``, ``cdb``, or ``pdb``. Accordingly, unlike the other database classes, each xdb class has its own distinct attributes 
and methods and is used for specific purposes.


ExomolHR  (``xdbExomolHR``)
----------------------------

``spec.exomolhr.xdbExomolHR`` is a class that retrieves line information from 
`ExoMolHR <https://www.exomol.com/exomolhr/>`_ 
for a given wavelength range and temperature (
`Zhang et al. (2025) <https://arxiv.org/abs/2504.08731>`_
). Since the temperature is fixed, it does not include methods that treat temperature as a variable, unlike those provided by ``mdb``. 
However, because the data is not as massive as that of ExoMol or HITEMP, it is useful for specific applications such as line identification 
at a fixed temperature. See below for an example of its use in line identification.

:doc:`../tutorials/exomolhr`

(Available from version 2.1 onward)
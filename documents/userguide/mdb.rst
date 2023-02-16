Molecular and Atomic Databases (mdb)
======================================

Multiple molecular and atomic databases are available in ExoJAX. 
These molecular database can be controlled using the molecular/atomic database class 
(mdb) for each database.


+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+
|**database**           |**mdb**  |**API**                                                                          | **notes**                          |
+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+
|ExoMol                 |MdbExomol|`spec.api.MdbExomol <../exojax/exojax.spec.html#exojax.spec.api.MdbExomol>`_.    | auto download.                     |
+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+
|HITEMP                 |MdbHitemp|`sepc.api.MdbHitemp <../exojax/exojax.spec.html#exojax.spec.api.MdbHitemp>`_.    | auto download or .par              |
+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+
|HITRAN                 |MdbHitran|`spec.api.MdbHitran <../exojax/exojax.spec.html#exojax.spec.api.MdbHitran>`_.    | auto download                      |
+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+
|Vald                   |AdbVald  |`sepc.moldb.AdbVald <../exojax/exojax.spec.html#exojax.spec.moldb.AdbVald>`_.    | manual download                    |
+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+
|Kurucz                 |AdbKurucz|`spec.moldb.AdbKurucz <../exojax/exojax.spec.html#exojax.spec.moldb.AdbKurucz>`_.| auto download                      |
+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+

See :doc:`api` and :doc:`atomll` for the details.
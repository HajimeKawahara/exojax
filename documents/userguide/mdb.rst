Molecular and Atomic Databases (``mdb`` / ``adb``)
====================================================

Multiple molecular and atomic databases are available in ExoJAX. 
These molecular database can be controlled using the molecular/atomic database class 
(``mdb`` / ``adb``) for each database.


+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+
|**database**           |mdb/adb  |**API**                                                                          | **notes**                          |
+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+
|ExoMol                 |MdbExomol|`spec.exomol.api.MdbExomol <../exojax/exojax.spec.html#exojax.spec.exomol.api.MdbExomol>`_.    | auto download.                     |
+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+
|HITEMP                 |MdbHitemp|`sepc.MdbHitemp <../exojax/exojax.spec.html#exojax.spec.MdbHitemp>`_.    | auto download or .par              |
+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+
|HITRAN                 |MdbHitran|`spec.MdbHitran <../exojax/exojax.spec.html#exojax.spec.MdbHitran>`_.    | auto download                      |
+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+
|Vald                   |AdbVald  |`sepc.moldb.AdbVald <../exojax/exojax.spec.html#exojax.spec.moldb.AdbVald>`_.    | manual download                    |
+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+
|Kurucz                 |AdbKurucz|`spec.moldb.AdbKurucz <../exojax/exojax.spec.html#exojax.spec.moldb.AdbKurucz>`_.| auto download                      |
+-----------------------+---------+---------------------------------------------------------------------------------+------------------------------------+

See :doc:`api` and :doc:`atomll` for the details.

For VALD3, you need to request the database one by one from the VALD3 website.

Molecular and Atomic Databases (``mdb`` / ``adb``)
====================================================

Multiple molecular and atomic databases are available in ExoJAX. 
These molecular database can be controlled using the molecular/atomic database class 
(``mdb`` / ``adb``) for each database.


.. list-table::
   :header-rows: 1

   * - Database
     - mdb/adb
     - API
     - Notes
   * - ExoMol
     - MdbExomol
     - ``exojax.database.MdbExomol``
     - Automatic download
   * - HITEMP
     - MdbHitemp
     - ``exojax.database.MdbHitemp``
     - Automatic download or local ``.par`` file
   * - HITRAN
     - MdbHitran
     - ``exojax.database.MdbHitran``
     - Automatic download
   * - VALD
     - AdbVald
     - ``exojax.database.AdbVald``
     - Manual download
   * - Kurucz
     - AdbKurucz
     - ``exojax.database.AdbKurucz``
     - Automatic download

See :doc:`api` and :doc:`atomll` for the details.

For VALD3, you need to request the database one by one from the VALD3 website.

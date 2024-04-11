Particulates Database (pdb)
==================================

The ``pdb`` is the database for particulates in ``ExoJAX``, namely the microphysics of clouds.
Currently, the refractive index is downloaded from ``VIRGA`` and made available within ``pdb``.


.. code:: ipython3
    
    from exojax.spec.pardb import PdbCloud
    miedir = "/home/kawahara/exojax/documents/tutorials/.database/particulates/virga"
    pdb = PdbCloud("MgSiO3", path=miedir)


For specific examples, please refer to 
:doc:`comp_pymiescatt`
for example.


Available information 
------------------------


+-----------------------+----------------------------------+
|quantity               |instance/method                   |
+-----------------------+----------------------------------+
|substance density      |pdb.condensate_substance_density  |
+-----------------------+----------------------------------+
|saturation pressure    |pdb.saturation_pressure           |
+-----------------------+----------------------------------+
|refraction index (RI)  |pdb.refraction_index              |
+-----------------------+----------------------------------+
|RI wavenumber          |pdb.refraction_index_wavenumber   |
+-----------------------+----------------------------------+
|RI wavelenght (nm)     |pdb.refraction_index_wavelength_nm|
+-----------------------+----------------------------------+
|Miegrid                |pdb.miegrid                       |
+-----------------------+----------------------------------+

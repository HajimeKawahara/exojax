Databases
===================

ExoJAX uses ExoMol, HITRAN/HITEMP, and VALD3 as molecular/atomic databases.

- :doc:`exomol`
- :doc:`hitran`
- :doc:`atomll`

For ExoMol/HITRAN/HITEMP, ExoJAX automatically downloads the database from their website and those are usually saved in .database directory under the directory where you run the code. There is no official common database directory in ExoJAX. So, if you want to reuse the database in other directory, you may use the soft link to the master .database directory. 

.. code:: sh

   > ls ~/database/.database  #master .database
   
   ~/database/.database 
   
   ~/database/.database/CO:
   12C-16O
   
   > cd /your-other-workspace/
   > ln -s ~/database/.database ./

For VALD3, you need to request the database one by one from the VALD3 website.

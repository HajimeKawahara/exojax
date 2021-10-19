Databases
===================

ExoJAX uses ExoMol, HITRAN/HITEMP, and VALD3 (planned) as molecular/atomic databases.

- ":doc:`exomol`"
- ":doc:`hitran`"
- VALD3 (prepared)

They are automatically downloaded and are usually saved in .database directory under the directory where you run the code. There is no official common database directory in ExoJAX. So, if you want to reuse the database in other directory, you may use the soft link to the master .database directory.

``` sh
> ls /home/kawahara/database/.database  #master .database

/home/kawahara/database/.database 

/home/kawahara/database/.database/CO:
12C-16O

> ls /your-other-workspace/
> ln -s /home/kawahara/database/.database ./

```

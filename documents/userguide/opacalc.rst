Opacity Calculator Classes (``opa``)
======================================

Several opacity calculation methods are available in ExoJAX. 
Opacity calculations can be controlled using the opacity calculation class 
(``opa``) for each method.


+--------------------------+-------------+-----------+------------------------------------+
|**calculator**            |**opa**      |**nu_grid**| **features**                       |
+--------------------------+-------------+-----------+------------------------------------+
|:doc:`premodit` (default) |OpaPremodit  |ESLOG      | memory saving, need to set T range |
+--------------------------+-------------+-----------+------------------------------------+
|:doc:`diffgrid`           |OpaDiffgrid  |ESLOG      | fixed pressure, repeated T profiles|
+--------------------------+-------------+-----------+------------------------------------+
|CKD                       |OpaCKD       |arbitrary  |Correlated k-Distribution           |
+--------------------------+-------------+-----------+------------------------------------+
|Direct LPF                |OpaDirect    |arbitrary  | line by line, arbitrary T          |
+--------------------------+-------------+-----------+------------------------------------+
|:doc:`modit`              |OpaMODIT     |ESLOG      | arbitrary T                        |
+--------------------------+-------------+-----------+------------------------------------+
|DIT                       |N/A          |ESLIN      | linear grid                        |
+--------------------------+-------------+-----------+------------------------------------+

Links to API
----------------

- `OpaPremodit <../exojax/exojax.spec.html#exojax.opacity.api.OpaPremodit>`_
- :class:`exojax.opacity.diffgrid.api.OpaDiffgrid`

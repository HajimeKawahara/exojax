Opacity Calculator Classes (opa)
======================================

Several opacity calculation methods are available in ExoJAX. 
Opacity calculations can be controlled using the opacity calculation class 
(opa) for each method.


+-----------------------+-------------+-----------+------------------------------------+
|**method**             |**opa**      |**nu_grid**| **features**                       |
+-----------------------+-------------+-----------+------------------------------------+
|PreMODIT   (default)   |OpaPremodit  |ESLOG      | memory saving, need to set T range |
+-----------------------+-------------+-----------+------------------------------------+
|Direct LPF             |OpaDirect    |arbitrary  | line by line, arbitrary T          |
+-----------------------+-------------+-----------+------------------------------------+
|MODIT                  |OpaMODIT     |ESLOG      | arbitrary T                        |
+-----------------------+-------------+-----------+------------------------------------+
|DIT                    |N/A          |ESLIN      | linear grid                        |
+-----------------------+-------------+-----------+------------------------------------+


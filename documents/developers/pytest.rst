Test codes for developers
==============================

ExoJAX has many test codes in the ``tests`` directory. The ``test`` directory contains several types of the collection of ``pytest`` code.

- ``tests/unittests``: the collection of the unit tests. The GitHub action runs the test code in this directory.
- ``tests/integration``: the collection of the test codes that need longer time to run than the code in ``unittest``.

test/unittests
---------------------

We recommend to write the unit test code in ``tests/unittests`` directory before pull-request and to perform the unit tests before your submission of the pull-request:

.. code:: sh

   cd exojax/test/unittests
   pytest 


test/integration/unittest_long
----------------------------------

In essence, these are the unit tests that need longer time than the code in ``unittest``, sometimes including downloading the data.  

test/integration/comparison
---------------------------

The code for the comparison with external data, packages, etc 

- ``transmission/comparison_with_kawashima_transmission.py``: comparison with Yui Kawashima's computation of the transmission spectrum
- ``twostream/comparison_petitRADTRANS_*.py``: comparison with pRT
- ``nonair/nonair_co_hitran_comp.py``: non-air broadening comparison with ``radis``



Others
--------------

VALD data 
^^^^^^^^^^^^^^^^

You can download them from `here <http://secondearths.sakura.ne.jp/exojax/data/>`_, but see the following warning.

.. warning::
   
   Note that if you use Windows or Mac, .gz might be unziped when downloading despite no renaming. I mean, the same name with .gz, but unziped!  In this case, download ``extradata.tar`` and untar it.


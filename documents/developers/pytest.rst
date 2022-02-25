Test codes for developers
==============================

ExoJAX has many test codes in 'tests' directory.
We recommend to write the unit test code in 'tests' directory before pull-request and to perform the unit tests before your submission of the pull-request:

.. code:: sh

   cd exojax
   pytest tests

You might need some files. 

- VALD data 

You can download them from `here <http://secondearths.sakura.ne.jp/exojax/data/>`_, but see the following warning.

.. warning::
   
   Note that if you use Windows or Mac, .gz might be unziped when downloading despite no renaming. I mean, the same name with .gz, but unziped!  In this case, download ``extradata.tar`` and untar it.

   
Unit test using pytest
----------------------------

Before starting the unit test, install pytest:

.. code:: sh

   pip install pytest


To test all of the unit tests, perform

.. code:: sh

   cd exojax
   pytest tests

or you can test one by one 

.. code:: sh

   cd tests/auto
   pytest autoxs_test.py


Tests for the reverse modeling (retrieval)
-----------------------------------------------

The unit test is not appropriate for the reverse modeling because it takes a lot of time.
For the tests of the reverse modeling, use code in 'tests/reverse' directory.

- reverse_lpf.py simple test for HMC-NUTS using LPF
- reverse_methane.py simple test for HMC-NUTS using MODIT

 .. code:: sh

   cd tests/reverse
   python reverse_lpf.py
   python reverse_methane.py

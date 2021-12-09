Unit test using pytest
========================

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

We recommend to write the unit test code in 'tests' directory before submitting pull-request.


You might need some files. 

- VALD data 

You can download them from `here <http://secondearths.sakura.ne.jp/exojax/data/>`_, but see the following warning.

.. warning::
   
   Note that if you use Windows or Mac, .gz might be unziped when downloading despite no renaming. I mean, the same name with .gz, but unziped!  In this case, download ``extradata.tar`` and untar it.

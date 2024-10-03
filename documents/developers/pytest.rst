Test codes for developers
==============================

ExoJAX has many test codes in the ``tests`` directory. ExoJAX has three test categories. 

Unit Tests
-----------------
``tests/unittests``: Tests in this category are automatically executed by GitHub Actions 
when a pull request is made to the develop or master branch. 
Therefore, items that need to be downloaded from external sites or take more than 10 seconds to run should not be included in this category. 
Tests that take a long time but are considered unit tests should be placed in ``integrations/unittests_long``.

Integration Tests
-----------------
``tests/integration``:　This category is for testing the behavior of multiple integrated functions. Tests that have a long execution time, 
involve external downloads, or depend on the status of external servers should be included here if they are to be part of automated testing.
ntegration tests also include comparisons with other codes or outputs, ensuring higher reliability. 
However, since changes in the counterpart code can occur, the tests do not always succeed. 

- ``tests/integration/comparison/transmission`` : An example of a transmission comparison with calculations done by Y. Kawashima using a different method.
- ``tests/integration/comparison/twostream``: A comparison code with the radiative spectrum calculations performed by petitRADTRANS.
- ``tests/integration/comparison/clouds``: A comparison with cloud models from VIRGA.

End-to-end Tests
-----------------
``tests/endtoend``: In ExoJAX, codes like HMC-NUTS that require long execution times are often used in the final application. 
Therefore, such tests belong to the end-to-end category. However, due to the long execution times, these tests are not run frequently.




Others
--------------

VALD data 
^^^^^^^^^^^^^^^^

You can download them from `here <http://secondearths.sakura.ne.jp/exojax/data/>`_, but see the following warning.

.. warning::
   
   Note that if you use Windows or Mac, .gz might be unziped when downloading despite no renaming. I mean, the same name with .gz, but unziped!  In this case, download ``extradata.tar`` and untar it.


Test codes for developers
==============================

ExoJAX tests are organized by the current package modules and the data and
computation needed to run them. See ``tests/README.md`` for the test conventions.

Unit Tests
-----------------
``tests/unittests`` contains small numerical, API, and regression tests.
These run automatically for pull requests to develop and master. Tests use
synthetic inputs or small bundled samples and must not download data.

Run the unit suite with ``JAX_PLATFORMS=cpu python -m pytest``. Each test uses a
temporary working directory and starts with JAX x64 enabled. Precision-specific
tests select their precision before creating arrays; the fixture restores the
previous setting after each test. Test modules must not change global precision
or device settings during import.

Integration Tests
-----------------
``tests/integration/offline`` contains spectrum comparisons and CLI workflows
that use bundled or synthetic data. CI runs these alongside unit tests:

.. code-block:: sh

   JAX_PLATFORMS=cpu python -m pytest tests/unittests tests/integration/offline

Other integration directories may require external databases, downloads, or
manual setup. Run them explicitly after preparing their inputs. Comparisons
with external codes can also depend on the version of the reference code.
Measure runtime with ``--durations=20`` before moving expensive tests, and
preserve their numerical checks and CI execution when reorganizing them.

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

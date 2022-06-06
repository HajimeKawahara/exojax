# Tests

## unittests
Unit tests, but parts of the integration tests that do not take long computation time are included.
The real molecular database should not be used in the code. 
We frequently test the codes in this category when writing code. 

## integration
Integration test. Longer computation time and/or use real molecular database are allowed.

### integration/comparison
Internal comparison of opacity/radiative transfer codes 
We recommend to test the code in this category before the Pull Request.

# endtoend
Endtoend test. It takes much longer time than integration tests/unit tests.

To test opacity calculators, use auto/*.py

###  endtoend/manual_check
 not test, but check various things
 To check opacity calculators manually, use auto/*.py



"""Use unit-test isolation for bundled-data integration workflows."""

import pytest


@pytest.fixture(autouse=True)
def offline_test_environment(isolated_test_environment):
    yield

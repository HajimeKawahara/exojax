"""Run every unit test with isolated files and precision settings."""

import pytest


@pytest.fixture(autouse=True)
def unit_test_environment(isolated_test_environment):
    yield

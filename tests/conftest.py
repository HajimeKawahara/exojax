"""Shared isolation for unit tests and offline integration tests."""

import jax
import pytest


@pytest.fixture
def isolated_test_environment(tmp_path, monkeypatch):
    """Keep generated files local and restore precision after each test."""
    previous_x64 = jax.config.jax_enable_x64
    monkeypatch.chdir(tmp_path)
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)

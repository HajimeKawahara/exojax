# Tests

## Unit tests

`tests/unittests` contains small numerical, API, and regression tests. Small
bundled database samples are allowed when the database adapter itself is under
test. Use synthetic arrays, snapshots, or simple stubs when only their values
are needed. Tests must not download data.

Directories follow the current `src/exojax` modules. Name files
`test_<module>.py`, or `test_<module>_<behavior>.py` for larger modules, and name
functions `test_<behavior>`. Pytest uses importlib mode so separate directories
can contain files such as `test_api.py`.

```sh
JAX_PLATFORMS=cpu python -m pytest
```

Each test runs in its own temporary working directory with JAX x64 enabled.
Tests of float32 behavior must select that precision before constructing arrays.
The fixture restores the previous precision after every test. Do not change
precision, devices, or the working directory during module import. Use
`tmp_path` for generated data and `monkeypatch` for environment changes.

## Integration tests

`tests/integration/offline` contains spectrum comparisons and CLI workflows
using bundled or synthetic data. These tests run in CI alongside the unit
tests, with the same file and precision isolation:

```sh
JAX_PLATFORMS=cpu python -m pytest tests/unittests tests/integration/offline
```

Other `tests/integration` directories may need external databases, network
access, or manual setup. Run those tests explicitly after preparing their data.

Use `--durations=20` to find expensive tests. Reduce arrays in shape and input
validation tests before moving them. Preserve numerical comparisons, gradient
checks, and regression conditions when combining tests; a filename or old API
name alone is not a reason to delete a test. When moving a test, retain its CI
execution unless external data makes that impossible.

## Other tests

- `tests/endtoend`: long application workflows, including inference.
- `tests/figures`: manual plotting scripts, outside automated test collection.
- `tests/benchmark`: performance measurements.

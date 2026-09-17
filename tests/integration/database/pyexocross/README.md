# PyExoCross loader validation (#804)

The local PyExoCross -> NumPy arrays -> `MDBSnapshot` path reproduces the
existing RADIS-backed ExoMol results for the bundled CO and H2O samples, after
matching the reference line-strength constant convention. The stage-1
experiment uses `snapshot.py`; `--public-api` validates the stage-2
`MdbExomol(..., backend="pyexocross")` implementation.

`snapshot.py` is a deliberately limited experimental adapter. `compare_loaders.py`
loads both backends independently, checks their selected transitions and physical
inputs, and compares existing LPF and PreMODIT calculations. The PyExoCross path
does not use RADIS line arrays, partition functions, or broadening coefficients.

## Reproduce

From the repository root, with ExoJAX's dependencies already installed, create
an overlay environment to keep the existing environment intact:

```bash
python3.12 -m venv --system-site-packages /tmp/exojax-pyexocross-validation-venv
/tmp/exojax-pyexocross-validation-venv/bin/python -m pip install \
    'pyexocross==1.1.16' 'radis==0.16.3' 'numpy==1.26.4' \
    'pyarrow==23.0.1' 'zarr==2.18.7' 'matplotlib==3.10.0'

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
NUMBA_CACHE_DIR=/tmp/exojax-px-numba MPLCONFIGDIR=/tmp/exojax-px-mpl \
PYTHONPATH=src \
/tmp/exojax-pyexocross-validation-venv/bin/python \
    tests/integration/database/pyexocross/compare_loaders.py \
    --output /tmp/exojax-px-validation.json
```

Use `--case CO-H2` to run one case. The script exits unsuccessfully if a
comparison fails. The JSON report records versions, source-file SHA256 hashes,
line counts, comparison settings, and numerical errors.

To validate the public API, add `--public-api` to the command above. This
constructs both LPF and PreMODIT directly from the new `MdbExomol` instance,
checks its snapshot, and blocks network access. The production reader uses
PyExoCross's low-level readers without calling `px.load` or changing global
configuration. It preserves supplied line positions without a Parquet cache.

The public API run passed the same 126 comparisons with the numerical results
in the table below. The corresponding unit checks cover deferred activation,
optional quantum columns, device arrays and masks, missing dependencies,
JSON definitions, missing line positions across chunks, and configuration
isolation. Download checks use mocked responses, including dataset selection,
overlapping segments, local file reuse, and interrupted-transfer cleanup.
These small fixtures do not establish memory use or speed for large line lists.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
NUMBA_CACHE_DIR=/tmp/exojax-px-numba MPLCONFIGDIR=/tmp/exojax-px-mpl \
PYTHONPATH=src /tmp/exojax-pyexocross-validation-venv/bin/python -m pytest -q \
    tests/unittests/database/exomol \
    tests/unittests/database/_common/test_radis_adapter.py \
    tests/unittests/database/test_optional_imports.py
```

This targeted suite passed 70 tests in the supported optional environment.

The validated environment used Python 3.12.7, PyExoCross 1.1.16, RADIS 0.16.3,
JAX/JAXlib 0.6.2, NumPy 1.26.4, pandas 2.2.3, PyArrow 23.0.1, SciPy 1.14.1,
and Zarr 2.18.7. The overlay inherits other dependencies from the existing
environment. Zarr 2 avoids the installed Zarr 3 release's NumPy >=2 requirement.
On the validation host, `JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1` was also set to skip
an installed CUDA plugin's device check while explicitly using the CPU backend.

RADIS 0.17.1 declares `numpy~=2.0`, while PyExoCross 1.1.16 declares
`numpy<2.0`. Their current requirements do not overlap. RADIS 0.16.3 permits
NumPy 1.26.4 and satisfies ExoJAX's `radis>=0.16.1` requirement. The final
run used this compatible combination; the numerical results also matched an
earlier diagnostic run with RADIS 0.17.1 under NumPy 1.26.4. That earlier
run was outside RADIS 0.17.1's declared support range and is not evidence
that its dependency constraint can safely be removed.

## Stage-1 experiment coverage

- Fresh temporary copies of bundled `.def`, `.pf`, `.broad`, `.states.bz2`,
  and `.trans.bz2` files; existing binary caches are excluded.
- RADIS uses PyTables with downloads disabled and registration lookup isolated.
  Its download method raises if an unexpected file is requested.
- PyExoCross uses `cache='none'`, with 37-row chunks and sparse state-ID joins.
- Transition identities, supplied wavenumbers, Einstein A, lower energies,
  upper weights, lower/upper J, mass, partition tables and interpolation,
  and a0/a1 broadening are compared. Array comparisons use `atol=0`.
- H2 and He broadening, definition defaults, and nonzero intensity/energy
  cutoffs are exercised. Filters use `Ttyp=1000 K`.
- LPF uses the resulting snapshot plus its independently extracted A array.
  PreMODIT uses `OpaPremodit.from_snapshot()`; its RADIS reference uses the
  ordinary MDB constructor. No opacity implementation is changed.
- Wavenumber grid: 4330-4360 cm^-1, 2048 logarithmically spaced points.
  Temperatures: 500, 1000, 1500 K. Pressures: 0.1, 1, 10 bar.
  PreMODIT: diffmode 2, `(dE, Tref, Twt) = (100, 1000, 1200)`.

## Recorded result (2026-09-16)

All seven cases passed: 7 cases x 9 temperature/pressure pairs x 2 opacity
calculators = 126 spectrum comparisons. The source checkout was ExoJAX
`799ec414`.

| Case | Selected lines | LPF peak-relative error | PreMODIT peak-relative error |
|---|---:|---:|---:|
| CO, H2 | 220 | 0 | 6.36e-16 |
| CO, He | 220 | 0 | 3.80e-15 |
| CO, definition defaults | 220 | 0 | 1.03e-15 |
| CO, H2, filtered | 9 | 0 | 1.74e-15 |
| H2O, H2 | 175 | 6.95e-16 | 1.88e-15 |
| H2O, He | 175 | 7.16e-16 | 2.82e-15 |
| H2O, H2, filtered | 19 | 0 | 3.27e-16 |

Filtered cases use `crit=1e-25` and `elower_max=2000 cm^-1`. Raw transition
counts are 259 for CO and 197 for H2O. All line fields except reference
strengths agree exactly; reference strengths agree within 5.69e-16 relative.

Peak-relative error means `max(abs(actual-reference)) / max(abs(reference))`,
maximized over the nine conditions. Spectrum assertions use `rtol=1e-9` and
`atol=1e-12 * max(abs(reference))`. Pointwise errors are also recorded, with
a 1e-12 peak floor in the denominator. The largest such pointwise error is
8.70e-9 in weak PreMODIT tails; its peak-relative error remains 2.83e-15.
This checks agreement between loaders within each opacity method, not the
approximation error of PreMODIT relative to LPF.

## Findings that matter for a production adapter

**Reference-strength constants must be specified.** The current ExoJAX
converter uses `hcperk=1.4387773538277202 cm K`; RADIS uses modern SI h*c/k,
approximately `1.4387768775039338 cm K`. Using the ExoJAX converter unchanged
gives maximum per-line relative S(296 K) differences of 1.37457e-4 for CO and
7.60863e-6 for H2O. The large CO relative difference occurs in extremely weak
high-energy lines; its peak-relative strength error is 1.70474e-6.

The comparison explicitly supplies `reference_c2=scipy.constants.h*c/k*100`
to the experimental adapter before applying the intensity cutoff. Temperature
scaling retains ExoJAX's existing convention, matching the current MDB path.
The unadjusted ExoJAX strengths are retained separately in the diagnostics.
The adapter's default `reference_c2=None` is therefore not the strict-parity
configuration. No global ExoJAX constants were changed.

**Preserve the fourth transition column.** CO's supplied line centers differ
from the state energy differences by up to 9.6e-5 cm^-1. Replacing those
centers would change the spectra. If the fourth column is absent or contains
NaN, the adapter uses the energy difference for the whole source, matching
RADIS. Separate checks of three-column CO data and a NaN on the last raw row,
outside the selected interval, both matched RADIS for all 220 selected lines.

**Preserve the current metadata and width defaults.** RADIS ultimately uses
the `.def` mass (28.0101 for this CO fixture), even though the initial ExoJAX
isotopic mass lookup differs. Broadening applies a1 pair matches over a0
lower-J matches, then the `.def` defaults. PyExoCross's generic defaults are
not substituted.

The experiment is restricted to local text definitions, three/four-column
transitions, a0/a1 recipes, and PyExoCross 1.1.16. It does not establish
large-dataset memory/performance, download compatibility, optional quantum
state filtering, other broadening recipes, or ExoAtom support. PyExoCross's
changes to `__main__` are restored, but its internal globals remain mutable:
run the experiment serially in a dedicated process. Parquet range caching is
intentionally excluded because it filters by energy differences before an
adapter can select the supplied line centers.

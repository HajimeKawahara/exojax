# ExoJAX model records

These optional MCEG records describe ExoJAX's scientific choices, quantity
conventions and the scope of their evidence. The [native inventory](models.yaml)
surveys the packaged source tree, including data. Six initial contracts follow
the main path from a local line opacity to an emission or transmission spectrum
and its observation operator. Known uncovered families remain visible.

| Contract | Initial scope | Important convention |
| --- | --- | --- |
| [Planck](contracts/planck.yaml) | Ordinary-frequency and wavenumber Planck intensity and hemispheric flux | `piB` and `piBarr` return flux per inverse centimetre; use the kernel's actual effective constants. |
| [LTE line opacity](contracts/lte_line_opacity.yaml) | LTE line-strength scaling, line widths, normalized profiles and a line contribution | Per-absorber cross sections differ from volume extinction. LPF, MODIT and PreMODIT have distinct numerical approximations and evidence scopes. |
| [Atmospheric columns](contracts/atmospheric_columns.yaml) | Cross section times absorber column, pressure columns, CIA and hydrostatic conventions | Mass and volume mixing ratios have different denominators. The CIA pair convention introduces no extra factor of one half. |
| [Thermal emission](contracts/thermal_emission.yaml) | Plane-parallel pure absorption, a supplied source and boundary conditions | When `ArtEmisPure` supplies `piBarr`, internal intensity is pi-scaled and angular flux integration uses a factor of two. |
| [Transmission](contracts/transmission.yaml) | Spherical chords, blocked area and radius normalization | `ArtTransPure.run` returns `(R_transit / R_bottom)**2`; stellar normalization for transit depth is a separate step. |
| [Spectral observation](contracts/spectral_observation.yaml) | Gaussian response, rotation, limb darkening and RV sampling | `response.sampling` uses `wavenumber * (1 + RV/c)`, with velocities in km/s; finite grids and edge rules belong to the implementation. |

The inventory also records continuum prescriptions, atomic and molecular data
policies, scattering/reflection closures, correlated-k methods, RCE and cloud
microphysics. An empty contract link or pending boundary review is a coverage
gap, not an absence of the numerical implementation. Priors, likelihoods and
study-specific model choices belong to their applications. A model inventory
does not certify that every scientific assumption has been discovered.

The initial native baseline lists 65 model variants and 284 packaged files.
Twenty-three variants link to the six partial contracts; 42 have no contract.
Sixty-three files have an API/boundary inspection hash and 221 remain unreviewed.
All 65 model-boundary reviews remain pending. These counts describe coverage,
not scientific support.

## Separable inspection

The dependency runs from this companion to the provider. Installing, importing,
building and running ordinary ExoJAX tests or documentation requires neither
MCEG nor Planetlib. No runtime registration, decorator or package import is
added for these records. The existing Python protocols in
`src/exojax/opacity/contracts.py` retain their runtime responsibility.

The dedicated workflow checks three independent environments:

1. MCEG and SymPy read contracts, inspect source ASTs and compare hashes without
   installing or importing ExoJAX or JAX.
2. The provider environment runs explicitly selected correspondence tests.
3. A provider-only environment builds and exercises ExoJAX with the companion
   removed and without MCEG or Planetlib installed.

Metadata-only checks live in `mceg/tests/`, outside `tests/conftest.py`, which
imports JAX. Provider comparisons live in `tests/model_contracts/`; the ordinary
pytest target remains `tests/unittests`. Standard Sphinx and wheel builds do
not generate or validate MCEG records.

## Definition and implementation ownership

Shared equations are maintained only in Planetlib's catalog `Equation.sympy`
fields. Import locks fix both catalog-file bytes and native model digests.
ExoJAX authors local choices, source mappings, Claims and Assessments in
`templates/`. `contracts/` contains generated standalone native contracts;
do not independently edit their copied mathematical definitions. Reading those
contracts needs no Planetlib checkout. Rebuilding requires the fixed checkout
selected by the dedicated workflow.

The current build uses Planetlib
[`20c0a3936cc6c1ad790945b936c1ef3ad74af31e`](https://github.com/HajimeKawahara/planetlib/tree/20c0a3936cc6c1ad790945b936c1ef3ad74af31e),
including the shared definitions proposed in
[Planetlib PR #2](https://github.com/HajimeKawahara/planetlib/pull/2).
The five new catalog families contain 29 equations. Locks select definitions;
they do not import the library's assessments into ExoJAX.

The original Planck selection at Planetlib commit
`a83b3fb255c7a383c6a8ef502328266f4dd16bff` remains identifiable in its existing
Context and Evidence. Its historical lock is retained at
`planetlib/imports.lock.json`. Moving the current materialized contract does
not renew that review or change its equation/model revisions.

Bindings declare directly representable arguments and scalar results. Array
components, internal pi scaling, tuple fields, numerical differentiation and
implicit return conversions need explicit descriptions and scoped comparisons.
Native AST inspection does not establish mathematical equivalence. Source
manifests identify selected files rather than an exhaustive transitive lock.

## Native formats and checks

The contracts use MCEG schema `0.6`; the inventory independently uses
`inventory_version: '0.1'`. The optional requirements pin MCEG
`4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88`. At that revision, `mceg init`
produces an older schema-0.2 draft; these contracts are authored and checked
with native `ContractV06` instead of treating that example as a schema upgrade.

The native specifications define [symbolic source precedence](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/symbolic.md),
[model connections](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/mathematical_models.md),
[bindings and review pins](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/equations.md),
[coverage](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/coverage.md)
and [assessment history](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/assessment_lifecycle.md).
MCEG supplies `load_contract`, `to_sympy`, native record serialization,
`mathematical_model_digest`, `check_contract`, `Inventory` and `check_coverage`.
Planetlib's existing build command supplies pinned external-definition selection.
There is no local formula registry, parser or replacement scientific schema.

MCEG's coverage exit code 1 includes both known gaps and new breakages. The
local policy test compares native issue identities with a reviewed baseline;
it does not accept every exit code 1. Changes to reviewed files, missing
symbols, broken contracts and additional unreviewed files require attention.
The baseline comparison is local policy because MCEG does not provide it.
Inspect inventory scope reductions and removed variants explicitly: a file
hash cannot discover a scientific option omitted from an authored inventory.

From the repository root, in the appropriate optional environment:

```sh
python -m pip install -r mceg/requirements.txt
python -m pytest -q mceg/tests
python -m mceg coverage . --json
```

Inspect one complete record directly:

```sh
python -m mceg mceg/contracts/planck.yaml
python -m mceg explain mceg/contracts/planck.yaml \
  --assessment exojax.planetlib.planck.pending
```

In a provider environment with the optional requirements installed, run the
additional numerical comparisons explicitly:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  python -m pytest -q tests/model_contracts
```

## Evidence and maintenance

The [initial execution record](evidence/initial_review/README.md) retains the selected command, test identities,
source and fixture hashes, definitions, Python/dependency versions, backend,
precision, result log and JUnit output. This is a provider-owned execution
artifact, not another MCEG schema or an automatic scientific assessor.
Future CI checks do not overwrite that receipt or its recorded judgments.

For decisive execution assessments, the local guard requires native selected-file
and artifact-hash checks to pass. A missing hash remains unknown in MCEG and is
insufficient for this provider policy. A recorded contradiction is allowed only
as the native `recorded_support` outcome; structural, binding, source or artifact
failures still fail CI.

Source correspondence, sampled numerical comparisons, derivative checks and
physical applicability have distinct Claims and scopes. A passed test is not
an empirical validation or an all-input correctness proof. Shared tests are
not counted as independent scientific evidence. Unknown Assessments have
concrete FollowUps; earlier judgments remain bound to their original Subjects
and Contexts.

Preserve published Quantity, Equation and MathematicalModel IDs/revisions.
Add a new revision or explicit variant when meanings or connections change.
Retain original Assessments, source snapshots and execution artifacts. A new
source hash, import lock or successful structural check does not renew a
judgment. Review the affected model choices and record a new execution before
claiming that previous numerical evidence applies to a changed implementation.

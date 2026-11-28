# Shared Planck definitions

This is ExoJAX's first MCEG adoption record. It imports fixed definitions from
Planetlib [`a83b3fb255c7a383c6a8ef502328266f4dd16bff`](https://github.com/HajimeKawahara/planetlib/tree/a83b3fb255c7a383c6a8ef502328266f4dd16bff)
and composes a local `exojax.planetlib.planck.flux@1` model for
`exojax.rt.planck.piB` and `piBarr`. The [standalone contract](contract.yaml)
is readable using native MCEG `0.6`; numerical code and runtime dependencies
are unchanged.

| Selected relation | Convention |
| --- | --- |
| `planetlib.radiation.lte.frequency_coordinate@1` | Ordinary frequency is `c * wavenumber`; `nu_grid` is in inverse centimetres. |
| `planetlib.radiation.lte.planck_frequency@1` | Blackbody intensity per Hz and steradian at positive temperature. |
| `planetlib.radiation.lte.intensity_conversion@1` | Spectral intensity per inverse centimetre acquires the Jacobian `c`. |
| `planetlib.radiation.lte.hemisphere_planck@1` | Integration of isotropic intensity over an outgoing hemisphere contributes `pi`. |

The local model selects only these four relations. Planetlib's import resolver
copies complete selected model dependency closures, so the native snapshot
also contains unmodified `lte.populations@1` and `lte.coordinates@1` records.
Their Boltzmann, profile, emission and extinction relations are not selected
by the local model and have no ExoJAX implementation claim here. No line
strength, molecular partition function, population solver, or atmospheric
LTE applicability is adopted.

## Actual kernel parameters

The provider takes temperature in Kelvin and wavenumber in `cm^-1`, returning
hemispheric flux in `erg s^-1 cm^-2 / (cm^-1)`. `piB` returns `(Nnu,)` and
`piBarr` returns `(Nlayer, Nnu)`. A result/location EquationBinding identifies
the final `piB` quantity; the internally composed intensity and array axes are
explicit context and test conventions, not inferred by native AST inspection.

The adopted constants come from the kernel's actual `fac_planck` and `hcperk`:

- `c = constants.ccgs = 29979245800 cm/s`;
- `h = fac_planck / (2*pi*c**2) = 6.62607004e-27 erg s`;
- `k = h*c/hcperk = 1.38064852e-16 erg/K`.

The exported `constants.hcgs` is `6.62607015e-27 erg s`, about `1.66e-8`
larger relative to the effective Planck-kernel value. The Planck functions do
not use that exported constant. The adoption records this existing numerical
choice explicitly; it does not update constants or claim exact agreement with
a different constant set. Planetlib supplies these quantities as parameters.

## One definition source and native checks

Edit local mappings, composition, claims and assessments in
[adoption.yaml](adoption.yaml). Shared formulas are maintained only in
Planetlib's catalog `Equation.sympy` fields. [contract.yaml](contract.yaml) is
a generated distribution snapshot, not a second authoring location.
[imports.lock.json](imports.lock.json) pins the catalog bytes and native model
digests. Updating a lock does not supply a new review.

MCEG schema requirements cover native records, references and review pins.
Its optional symbolic workflow does not require a Python formula registry.
The upstream catalog-only editing policy and the local template/snapshot
layout are project choices. Planetlib's existing `tools/planetlib.py build`
resolves the fixed selection; ExoJAX adds no parser, resolver or formula
registry. Tests use native `load_contract`, `to_sympy`,
`mathematical_model_digest`, `equation_checks`, and `file_integrity`. Temporary
symbolic substitution and differentiation use the imported expressions.

The [development dependency](requirements.txt) pins MCEG
[`4034c925`](https://github.com/HajimeKawahara/mceg/tree/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88), whose
[symbolic source](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/symbolic.md),
[model](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/mathematical_models.md),
[binding](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/equations.md), and
[history](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/assessment_lifecycle.md)
specifications define the native behavior used here.

With those development dependencies, JAX and NumPy installed, and a separate
Planetlib checkout at the fixed commit, run from the ExoJAX root:

```bash
python "$PLANETLIB_CHECKOUT/tools/planetlib.py" build \
  mceg/planetlib/adoption.yaml --lock mceg/planetlib/imports.lock.json \
  --output /tmp/exojax-planetlib-contract.yaml
diff -u mceg/planetlib/contract.yaml /tmp/exojax-planetlib-contract.yaml
python -m mceg mceg/planetlib/contract.yaml
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q \
  tests/model_contracts/test_planetlib.py tests/unittests/rt/test_planck.py
```

The separate [adoption workflow](../../.github/workflows/planetlib.yml) prepares
pinned MCEG and Planetlib, then reproduces the snapshot and checks it offline.
Tests compare provider values, temperature/wavenumber derivatives and array
axes at selected finite positive inputs. Existing float32 regression tests
remain unchanged. The new correspondence assessment remains `unknown` pending
an archived scoped review; sampled checks do not prove general floating-point
stability or physical applicability and no Planetlib support is imported.

For subsequent changes, preserve published shared and local model revisions,
subjects and assessments. Add a new revision or adoption snapshot and review
`exojax.planetlib.planck.correspondence`, including any changed constants,
units or array conventions. Do not refresh pins to imply a new execution or
transfer unrelated molecular/radiative-transfer evidence into this model.

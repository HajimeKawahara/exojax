# Real CO/H2O CKD comparison

This offline example uses the bundled **real but truncated** ExoMol CO
(Li2015) and H2O (POKAZATEL) SAMPLE line lists. It compares algorithms on the
same retained lines; it cannot establish absolute molecular opacity accuracy.
The atmosphere has four pure-emission layers, H2 pressure broadening, natural
widths, and no continuum. Eight top-hat observation bins cover 4330–4362 cm^-1.
Temperature scale and both VMRs vary; VMRs also change mean molecular weight.
The three parameters are natural log VMR(CO), natural log VMR(H2O), and
natural log temperature scale, in that fixed order.

From the repository root, with dependencies installed:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=True PYTHONPATH=src python examples/ckd_mixture_validation.py prepare --output-dir /tmp/co-h2o-case
JAX_PLATFORMS=cpu JAX_ENABLE_X64=True PYTHONPATH=src python examples/ckd_mixture_validation.py validate --output-dir /tmp/co-h2o-case --validation-id cpu --methods lbl rorr
```

Remove `JAX_PLATFORMS=cpu` only when deliberately selecting another device.
Preparation copies bundled files into a temporary directory before the database
adapter reads them; no external line database or download is needed. Saved
cases load through a data-only adapter without reopening the database.

| Method | Online opacity and transfer |
| --- | --- |
| `lbl` | Direct Voigt cross sections, summed at each wavenumber, full-resolution emission, then bin averaging |
| `premixed` | The same summed LBL optical depths, sorted/compressed per layer and band, then CKD transfer |
| `rorr` | Independent saved molecular CKD tables, temperature interpolation, VMR conversion, random overlap and rebinning, then CKD transfer |
| `same_g` | The same saved individual tables, added at matching g ordinates, then CKD transfer |

The premixed comparison includes compression, vertical rank correlation and
band-center Planck errors. RORR additionally assumes random overlap and
interpolates individual tables. Its difference from premixed is **not a pure
measurement of overlap error**. The g-point sweep holds all other settings
fixed. Both methods' sorting operations and CKD temperature interpolation are
piecewise differentiable.

Preparation freezes the truth, prior, noise realization, validation points,
budgets, band definition, line arrays (including partition functions and both
broadening contributions), and individual tables. `manifest.json` binds the
case metadata, array archive, and numerical source hashes. Loading rejects
changed files or numerical source; prepare a new output directory after a
model edit. Precompute time includes compilation and is kept separate from
online evaluation and retrieval time. Normalization is the fixed 1000 K Planck
flux at the band centers; observational sigma is 0.01 in these units.

Validation saves every method's result even on failure. It checks spectra
against a reference with twice as many LBL spectral samples, and checks coarse
versus refined LBL against one tenth of the candidate error budget. The frozen
candidate thresholds are maximum absolute residual/noise <= 0.01 and summed
squared residual/noise <= 0.1. Local AD/finite-difference checks use parameter
coordinates scaled by prior widths and require two adjacent successful step
sizes at tolerance 1e-3. Both whitened spectra and the same-data log likelihood
are checked. Differences between candidate and LBL Jacobians are recorded
separately, without treating AD correctness as physical accuracy.

The default five points include the truth and four reproducible interior
prior points. Finite checks do not certify the entire prior or differentiability
at all sorting/interpolation boundaries. The spectral refinement check does
not test missing lines, the vertical discretization or broadening assumptions.

`validate` evaluates all four methods and returns a nonzero exit status if a
requested method fails. Each method's `passed` includes the reference check;
an unrequested diagnostic failure cannot turn a selected method into a pass
or invalidate an otherwise passing selected method. Results remain in
`validations/<validation-id>/validation.json` and `spectra.npz`. The latter
contains all spectra, parameters and noise-scaled Jacobians for reanalysis.
A validation ID or case directory cannot be overwritten. These are strict
scientific gates, not an assertion that the default approximate models pass.
Do not relax the saved thresholds after inspecting a result.

For a small implementation smoke check, prepare a separate case using
`--samples-per-band 16 --ng 4 --temperature-nodes 3 --validation-points 1`.
Such a coarse case is expected to fail scientific precision requirements.

## Initial CPU diagnostic

On 2026-09-14, the defaults (1024 spectral samples per band, Ng=16, 21
temperature nodes and five validation points) completed on CPU with JAX/JAXLIB
0.6.2, NumPy 2.1.0 and x64. Reference refinement passed. All four methods passed
the 15 local directional checks of both spectra and log likelihood. Maximum
spectral errors divided by noise were:

| LBL | Premixed | RORR | Same-g |
| --- | --- | --- | --- |
| 0.000162903 | 0.306598 | 5.799967 | 15.905373 |

Thus LBL passed and all three CKD approximations failed the frozen 0.01 budget.
This is a saved failure of the approximate forward models for this case, not
a gradient implementation failure. A scientific retrieval comparison must
respect that result; a diagnostic retrieval must be explicitly identified as
such. The CPU preparation reported 8.27 s for table generation including
compilation; validation took 278.40 s. These are single diagnostic timings,
not GPU performance results or a formal speed comparison.

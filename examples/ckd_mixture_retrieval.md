# Matched CO/H2O mock retrievals

This example uses the saved case from `ckd_mixture_validation.py`. Read
`ckd_mixture_validation.md` for its physical assumptions, frozen error budgets,
and the limits of the bundled, truncated molecular line lists. This is a small
mixture comparison, not a validation of a broad-band planetary retrieval.

All four opacity methods (`lbl`, `premixed`, `rorr`, `same_g`) use the same
three parameters: natural-log CO VMR, natural-log H2O VMR, and natural-log
temperature scale. Priors are independent uniforms in these saved coordinates.
All methods see the same LBL-generated mock observation and Gaussian noise.
The truth is never used to initialize NUTS. Each seed gives prior-drawn starting
positions shared between methods, and distinct keys for warmup and sampling.

Run from the repository root in an environment containing CUDA-enabled JAX,
NumPyro, ArviZ and the ordinary ExoJAX dependencies:

```csh
csh -f examples/run_ckd_mixture_gpu.csh
```

Optionally supply a **new** output directory and Python executable as the first
and second arguments. The script records these paths on startup. It sets CUDA,
x64, disables the persistent compilation cache and GPU preallocation, then
prepares and validates a fresh case. It stops if either LBL or RORR fails its
predefined validation gates. Defaults are not a claim that RORR passes those
gates. Inspect `validations/gpu/validation.json` and `spectra.npz` on failure;
do not change a frozen threshold merely to start sampling. Increase numerical
resolution only in a new case and validate it again; overlap or vertical-rank
errors may remain even at high resolution.

The default case's recorded CPU validation passes reference refinement and
local gradients, but RORR's maximum error is about 5.8 times the observational
noise. To investigate the resulting posterior bias, explicitly select a
**diagnostic** run with the third launcher argument:

```csh
csh -f examples/run_ckd_mixture_gpu.csh output/ckd_diagnostic python diagnostic
```

This mode permits a failed spectral-approximation check while still requiring
reference refinement, local gradients, hashes and environment checks to pass.
Every run is marked diagnostic; even converged chains cannot produce a
scientific speedup. The launcher returns zero when this diagnostic workflow
finishes successfully, regardless of final scientific eligibility; inspect
the saved convergence diagnostics. Numerical refinement may reduce compression
or interpolation errors, but cannot be assumed to fix overlap errors.

After validation, the launcher runs two independent seeds (41 and 42), four
sequential chains, 500 warmup steps and 1000 retained draws per chain. Every
method runs in a fresh process; the second pair reverses method order. Logs and
`/usr/bin/time -p` whole-process timings are kept in `<output>.logs`. Existing
cases, validation IDs and method/run IDs cannot be overwritten. A failed run
retains its last stage and completed timing measurements in `result.json`.

The equivalent individual commands, for an already validated case, are:

```sh
python examples/ckd_mixture_retrieval.py run --output-dir CASE --validation-id gpu --run-id seed41 --method lbl --seed 41
python examples/ckd_mixture_retrieval.py run --output-dir CASE --validation-id gpu --run-id seed41 --method rorr --seed 41
python examples/ckd_mixture_retrieval.py run --output-dir CASE --validation-id gpu --run-id seed42 --method rorr --seed 42
python examples/ckd_mixture_retrieval.py run --output-dir CASE --validation-id gpu --run-id seed42 --method lbl --seed 42
python examples/ckd_mixture_retrieval.py summarize --output-dir CASE --run-id seed41 --repeat-run-id seed42 --methods lbl rorr
```

Keep the same x64/backend environment for validation and sampling. The run
checks case hashes, current numerical source, method-specific validation,
validation environment and residual-archive hash before constructing NUTS.
To compare other methods, validate them first and explicitly select the same
methods in `summarize`. Run IDs identify matched sets; methods within a set
must use identical controls and initial states. Repeated sets need independent
seeds and keys. Individual diagnostic runs require `--diagnostic` on every
matched method; default runs require all validation gates to pass.

Each `runs/<run-id>/<method>/` directory contains:

- `result.json`: settings, provenance, validation binding, timings, device
  memory (or explicit unavailability), and convergence diagnostics.
- `samples.npz`: unchanged primary chains, divergence flags, NUTS step counts
  and acceptance probabilities, with shapes, dtypes and SHA256 recorded.
- `posterior_predictive.npz`: noise-free spectra at evenly spaced retained
  draws from each chain, draw indices, observation, sigma and band centers.

Summarization only reads saved evidence. It verifies hashes and matched
conditions, recomputes diagnostics from the raw chains, and records posterior
differences with Monte Carlo uncertainties. Because methods share random seeds,
combined MCSEs are descriptive scales, not a calibrated test of their difference.
Scientific timing summaries require at least two pairs, four chains, at least
500 warmup and 1000 retained draws, rank R-hat < 1.01, bulk/tail ESS >= 400 for
every parameter, and zero divergences. Failed comparisons keep all diagnostics
and raw timing ratios but have no scientific median speedup. The csh launcher
returns exit code 3 if these final quality gates fail.

Timings distinguish synchronized compilation plus warmup from **cold sampling
compilation plus execution**; neither is a warmed steady-state throughput
measurement. Per-parameter bulk/tail ESS per cold sampling second is recorded.
Shared preparation/validation time is excluded from these ratios and remains
in separate artifacts. GPU memory values are first-device high-water marks
including earlier phases, not isolated incremental allocations. These results
do not establish coverage, posterior equivalence or speed for other cases.

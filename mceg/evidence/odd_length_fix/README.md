# Odd FFT-length fix execution

The saved run contains **five passing cases** on CPU with float64 enabled:

- 40- and 41-point ramps convolved with a normalized 13-point Gaussian kernel,
  checking shape and values against NumPy direct convolution.
- The original 41-point `ipgauss` shape counterexample, with its strict xfail removed.
- Existing Gaussian/rotation and zero-rotation comparisons.

`receipt.json` records the exact command, outcomes, implementation commit,
source/test hashes, package versions and execution environment. `pytest.log`
and `junit.xml` are the unchanged execution output. This is bounded numerical
evidence, not physical validation or a guarantee for every input shape.

The new spectral-observation Subject and Assessment apply only to these cases.
The initial contradicted Assessment and execution receipts are retained. The two
changed historical files are preserved under
`../initial_review/source_snapshot/`, and their historical file references now
resolve there with the original hashes. Other historical judgments are not
renewed.

PR #705 corrected the same missing inverse FFT length in `olaconv`; this fix
adds the length to `convolve_same`. The existing broader FollowUps remain open
for even-length and singleton kernels, whose slicing is outside this change.

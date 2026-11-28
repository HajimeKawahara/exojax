# Initial scoped execution review

The saved run contains **43 passed cases and one strict expected failure** on CPU.
The companion tests decode the imported native `Equation.sympy` records. The
selected ordinary tests supplement those comparisons with existing provider
checks. No spectral database or network access is required by the selected cases.

`receipt.json` records the exact command, every collected node ID and outcome,
package versions, the installed MCEG Git revision, backend/precision, bounded
source/test/config hashes and native equation/model/binding digests. `pytest.log`
and `junit.xml` are the unchanged execution outputs. The manifest hashes source
and tests, not the enclosing contract, avoiding a receipt/assessment hash cycle.

The six new `initial_samples.reviewed` assessments support only their listed
passing samples. Existing broader assessments remain unchanged. In particular,
this run does not establish primary-source correspondence, full numerical or
derivative domains, MODIT/PreMODIT accuracy, or physical applicability.

The odd-length convolution case is a recorded counterexample: `ipgauss` returns
40 values for the selected 41-value spectrum and 13-value kernel.
`odd_length_receipt.json`, `odd_length_failure.log` and `odd_length_failure.xml`
record a separate invocation with `--runxfail`, exposing the failed shape
assertion without changing the test. Its own narrow claim is `contradicted`.
The successful process exit of the main run is not evidence that this claim
holds. The implementation defect remains open and has explicit FollowUps.

The recorded interpreter and PYTHONPATH describe the local capture environment;
the companion workflow creates its own isolated environment. Future CI runs do
not rewrite these artifacts or automatically renew any assessment. Append a new
run and scoped assessment after a reviewed change; preserve this record.

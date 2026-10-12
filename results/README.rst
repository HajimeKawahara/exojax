Selected P7 execution and acceptance records
===========================================

This directory contains selected small records and execution helpers from
the WASP-18b P7 study. The complete local work directory also contains large
outputs, historical checkpoints, and job logs that are not versioned.
``examples/rce_wasp18b_p7.rst`` describes the model and acceptance scope.

``rce_wasp18b_p7_acceptance.json`` is the immutable acceptance record from
2026-10-11. It includes comparison results, source and asset hashes,
regression-log reviews, recorded approximations, and the P8 handoff. Its
absolute paths identify the original local evidence archive; a checkout
alone does not contain every referenced file or the external database.
``rce_wasp18b_final8_prior_analysis.json`` and ``.csv`` retain the 27 per-case
results, including seven unconverged RCE attempts.

The following helpers are kept at their original paths so their source
hashes, recorded commands, and regression references remain valid:

* ``rce_wasp18b_explicit_ckd.py`` and
  ``run_rce_wasp18b_explicit_ckd.py`` pass large CKD tables as runtime operands
  to the frozen forward/validation callers. They require JAX 0.6.2 and verify
  the listed source-file hashes. The context is single-threaded and restores
  all patched module bindings on exit. It is not an implicit-AD API.
* ``rce_wasp18b_final8_prior.py`` performs the fixed 27-point finite-state
  diagnostic. Its ``--plan`` input pins the run configuration, baseline,
  source metadata, and external assets. The plan and domain inventory are
  local production inputs, not portable example defaults.
* ``run_rce_wasp18b_h2o_sampling_backlog.py`` is the frozen H2O preparation
  recovery worker. Its publication, failure, and resource-guard behavior is
  exercised by the offline integration tests. Production use requires the
  original input fingerprints and coordinated GPU scheduling.
* ``rce_wasp18b_validation_stellar_audit_previous/rce_wasp18b_forward.py`` is
  a historical test reference used to verify that extracting the shared
  stellar audit preserved the original forward report. It is not a current
  runner.

An explicit-CKD forward invocation with independently supplied local inputs
has the following form (GPU work; use new output paths)::

    JAX_PLATFORMS=cuda JAX_ENABLE_X64=True XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python results/run_rce_wasp18b_explicit_ckd.py \
        --driver forward --evidence /path/to/new_adapter_report.json -- \
        --manifest /path/to/manifest.json \
        --observation /path/to/emission_candidate.ecsv \
        --stream-resolved --mask-continuation --response-resolution 700 \
        --output /path/to/new_forward_report.json

The numerical evidence predates PR packaging and subsequent changes to
``develop``. Its source hashes remain historical identifiers; no GPU
calculations are rerun merely to update a Git revision. Fresh CPU checks of
the submitted files are recorded separately in the PR description.

# Initial review source snapshots

These two files retain the exact bytes used by the initial execution records,
before the odd FFT-length fix. They were recovered from ExoJAX commit
`b3e1dc82` and match the SHA-256 values already recorded in the unchanged
initial receipts. Their directory layout preserves the original source paths.

Historical Subject and Evidence file references now resolve to these snapshots;
the recorded hashes, IDs, outcomes and judgments are unchanged. This preserves
native integrity checks without reusing historical results for the corrected
implementation. The new run is recorded in `../../odd_length_fix/`.

| Original path | SHA-256 |
| --- | --- |
| `src/exojax/signal/convolve.py` | `d217c75401a2bf53a6bfa0e5a3ee6c2efe2d2b59addc50da3c49de5efb8c6a10` |
| `tests/model_contracts/test_core_models.py` | `219266d8de527ccde71554b36e3f3404dda72a415c07e7fedc7c780d0a4dd216` |

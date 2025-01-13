# Comparison Opart with Opa and Art

## Why we have these comparison/benchmark files? 


In the implementation of opart, there was initially an overhead associated with XLA compilation. This is clearly visible when using a profiler and can also be observed by varying the number of layers. The computation time does not scale with the number of layers; when the number of layers is small, the apparent computation time per layer becomes larger due to the XLA overhead. Conversely, when the number of layers is large, the time spent on XLA compilation becomes relatively negligible.

Try modifying the number of layers in the *_opart.py files included in this directory and measure the execution time. With a correct implementation, the computation time should scale with the number of layers. This issue is particularly dependent on factors such as where `jit` is applied or whether `scan` is included within a class. Therefore, whenever modifying the opart source code, it is necessary not only to perform unit tests but also to ensure there is no execution overhead. This is a crucial step in the workflow.

Additionally, to confirm that Out of Memory (OoM) errors occur with opa+art when the number of layers is large, or that opart and opa+art exhibit comparable execution times when there is no overhead, we have provided an equivalent calculation for opa+art for reference.

See #542 #547 for the details.

HajimeKawahara 1/13 (2025)
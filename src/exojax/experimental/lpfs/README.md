# lpfs

JAX/numpyro JVP/VJP is still unstable.
In this directory, we have three different types of lpf.

- lpf: no custom derivative
- clpf: JVP custom derivative
- rlpf: VJP custom derivative

The current fiducial lpf in exojax.spec is clpf.py.
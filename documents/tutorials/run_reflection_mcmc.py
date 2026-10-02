"""Run the reflection tutorial's NUTS retrieval without the optional optimizer."""

import argparse
import json
import os
from pathlib import Path


def load_model():
    """Reuse the notebook setup and model definition, stopping before sampling."""
    notebook = Path(__file__).with_name("get_started_reflection.ipynb")
    cells = json.loads(notebook.read_text())["cells"]
    namespace = {"__name__": "__main__"}
    section = "setup"
    for index, cell in enumerate(cells):
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            if source.startswith("## Optimization"):
                section = "skip"
            elif source.startswith("## HMC-NUTS"):
                section = "mcmc"
        elif cell["cell_type"] == "code" and section != "skip":
            print(f"Preparing notebook cell {index}", flush=True)
            exec(compile(source, f"{notebook}:cell-{index}", "exec"), namespace)
            if section == "mcmc" and "model_c" in namespace:
                return namespace
    raise RuntimeError("The notebook's HMC-NUTS model definition was not found.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("output/reflection_mcmc/samples.npz"))
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("--warmup must be nonnegative and --samples must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MIEPYTHON_USE_JIT", "1")
    os.environ["MPLBACKEND"] = "Agg"
    import jax
    import matplotlib.pyplot as plt
    import numpy as np
    from numpyro.infer import MCMC, NUTS

    print("JAX devices:", jax.devices(), flush=True)
    namespace = load_model()
    plt.close("all")
    mcmc = MCMC(NUTS(namespace["model_c"]), num_warmup=args.warmup,
                num_samples=args.samples, num_chains=1)
    key = jax.random.split(jax.random.PRNGKey(args.seed))[1]
    mcmc.run(key, y1=namespace["flux"], y1err=namespace["err_flux"])
    samples = {name: np.asarray(value)
               for name, value in mcmc.get_samples(group_by_chain=True).items()}
    np.savez_compressed(
        args.output, **samples,
        diverging=np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]),
        nu_obs=np.asarray(namespace["nu_obs"]),
        flux=np.asarray(namespace["flux"]),
        err_flux=np.asarray(namespace["err_flux"]),
        seed=args.seed, num_warmup=args.warmup,
    )
    print(f"Saved posterior samples to {args.output}", flush=True)
    if args.samples >= 4:
        mcmc.print_summary()


if __name__ == "__main__":
    main()

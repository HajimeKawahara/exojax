from jax.config import config

def device_memory_use(opa, art=None, nfree=None):
    """device memory use given opa and art (optional), n free parameters (optional)

    Args:
        opa (opa): opa instance
        art (art, optional): art instance. Defaults to None.
        nfree (int, optional): the number of free parameters. Defaults to None.

    Raises:
        ValueError: method not implemented yet 

    Returns:
        float: estimated device memory use
    """
    if config.values["jax_enable_x64"]:
        precision = "FP64"
    else:
        precision = "FP32"

    if art is not None:
        nlayer = art.nlayer
    else:
        nlayer = None
    ngrid_nu_grid = len(opa.nu_grid)

    if opa.method == "premodit":
        ngrid_broadpar = opa.ngrid_broadpar
        devmemuse = premodit_devmemory_use(ngrid_nu_grid,
                          ngrid_broadpar,
                          nlayer=nlayer,
                          nfree=nfree,
                          precision=precision)
    else:
        raise ValueError("unknown method.")

    return devmemuse


def premodit_devmemory_use(ngrid_nu_grid,
                          ngrid_broadpar,
                          nlayer=None,
                          nfree=None,
                          precision="FP64"):
    """compute approximate required device memory for PreMODIT algorithm

    Args:
        ngrid_nu_grid (int): the number of the wavenumber grid
        ngrid_broadpar (int): the number of the broadening parameter grid
        nlayer (int, optional): If not None (when computing spectrum), the number of the atmospheric layers. Defaults to None.
        nfree (_type_, optional): If not None (when computing an HMC or optimization), the number of free parameters. Defaults to None.
        precision (str, optional): precision of JAX mode FP32/FP64. Defaults to "FP64".

    Raises:
        ValueError: _description_

    Returns:
        float: predicted required device memory (byte)
    """
    mode = "opacity"
    n = 1
    n *= ngrid_nu_grid
    n *= ngrid_broadpar
    if nlayer is not None:
        n *= nlayer
        mode = "spectrum"
    if nlayer is not None:
        n *= nfree
        mode += "/inference"

    if precision == "FP64":
        n *= 8
    elif precision == "FP32":
        n *= 4
    else:
        raise ValueError("choose FP64 or FP32")

    mode += "("+precision+")"
    print(mode + " : required device memory = ", n / (1024.)**3, "GB")

    return n


if __name__ == "__main__":
    n_nu_grid = 700000.0 * 0.1
    n_broadpar = 8
    nlayer = 200
    nfree = 10
    premodit_devmemory_use(n_nu_grid, n_broadpar, nlayer=nlayer, nfree=nfree)

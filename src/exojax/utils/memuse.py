from jax.config import config


def device_memory_use(opa, art=None, nfree=None, print_summary=True):
    """device memory use given opa and art (optional), n free parameters (optional)

    Args:
        opa (opa): opa instance
        art (art, optional): art instance. Defaults to None.
        nfree (int, optional): the number of free parameters. Defaults to None.
        print_summary (bool): printing summary Defaults to True.
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
        ngrid_elower = opa.ngrid_elower
        devmemuse, memdict = premodit_devmemory_use(ngrid_nu_grid,
                                                    ngrid_broadpar,
                                                    ngrid_elower,
                                                    nlayer=nlayer,
                                                    nfree=nfree,
                                                    precision=precision)
        memcase, info = memdict
        _print_summary_premodit(opa, nfree, print_summary, nlayer,
                                ngrid_nu_grid, ngrid_broadpar, ngrid_elower,
                                devmemuse, memcase, info)

    else:
        raise ValueError("unknown method.")

    return devmemuse


def _print_summary_premodit(opa, nfree, print_summary, nlayer, ngrid_nu_grid,
                            ngrid_broadpar, ngrid_elower, devmemuse, memcase,
                            info):
    explanation = ["(less important)", ""]
    if print_summary:
        print("Device memory use prediction:", opa.method)
        print("# of the wavenumber grid:", ngrid_nu_grid)
        print("# of the broadening par grids:", ngrid_broadpar)
        print("# of the elower grids:", ngrid_elower, explanation[memcase])
        print("# of the layers:", nlayer, explanation[1 - memcase])
        print("# of the free parameters:", nfree, explanation[1 - memcase])
        print(info + " : required device memory = ", devmemuse / (1024.)**3,
              "GB")


def premodit_devmemory_use(ngrid_nu_grid,
                           ngrid_broadpar,
                           ngrid_elower,
                           nlayer=None,
                           nfree=None,
                           precision="FP64"):
    """compute approximate required device memory for PreMODIT algorithm

    Notes:
        This method estimates the major device moemory use for PreMODIT. In Case 0 (memcase=0), the use is limited by FFT/IFFT by modit_scanfft 
        while in Case1 (memcase) by LBD. 
        

    Args:
        ngrid_nu_grid (int): the number of the wavenumber grid
        ngrid_broadpar (int): the number of the broadening parameter grid
        ngrid_elower: (int): the number of the lower energy grid
        nlayer (int, optional): If not None (when computing spectrum), the number of the atmospheric layers. Defaults to None.
        nfree (_type_, optional): If not None (when computing an HMC or optimization), the number of free parameters. Defaults to None.
        precision (str, optional): precision of JAX mode FP32/FP64. Defaults to "FP64".

    Raises:
        ValueError: _description_

    Returns:
        float: predicted required device memory (byte)
        (str, str): memory computation case (memcase), info 
    """
    info = "opacity "
    factor_case0 = 4  # FFT and InvFFT w/ the same size of buffer
    factor_case1 = 2
    memuse_case0 = ngrid_nu_grid * ngrid_broadpar * factor_case0
    memuse_case1 = ngrid_nu_grid * ngrid_elower * ngrid_broadpar * factor_case1

    if nfree is not None:
        memuse_case0 *= nfree
        info += "+ inference "

    if nlayer is not None:
        memuse_case0 *= nlayer
        info += "+ spectrum nlayer*"

    if precision == "FP64":
        memuse_case0 *= 8
        memuse_case1 *= 8
    elif precision == "FP32":
        memuse_case0 *= 4
        memuse_case1 *= 8
    else:
        raise ValueError("choose FP64 or FP32")

    if memuse_case0 > memuse_case1:
        info += "broadening "
        memuse = memuse_case0
        memcase = 0
    else:
        info += "elower*broadening "
        memuse = memuse_case1
        memcase = 1
    info += "(" + precision + ")"

    return memuse, (memcase, info)


if __name__ == "__main__":
    n_nu_grid = 700000.0 * 0.1
    n_broadpar = 8
    n_elower = 20
    nlayer = None
    nfree = 10
    memuse, case = premodit_devmemory_use(n_nu_grid,
                                          n_broadpar,
                                          n_elower,
                                          nlayer=nlayer,
                                          nfree=nfree)
    print(case)
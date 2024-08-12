import numpy as np
import matplotlib.pyplot as plt

def generate_miegrid_new(Tarr, Parr, mu, gravity, pdb_nh3, amp_nh3, molmass_nh3, sigmag, MMRbase_nh3, fsed_range, Kzz_range):
    fsed_grid = np.logspace(np.log10(fsed_range[0]), np.log10(fsed_range[1]), 3)
    Kzz_grid = np.logspace(np.log10(Kzz_range[0]), np.log10(Kzz_range[1]), 5)

    
    rg_val = []
    for fsed in fsed_grid:
        for Kzz in Kzz_grid:
            rg_layer, MMRc = amp_nh3.calc_ammodel(
                Parr, Tarr, mu, molmass_nh3, gravity, fsed, sigmag, Kzz, MMRbase_nh3
            )
            rg_val.append(np.nanmean(rg_layer))
            plt.plot(fsed, np.nanmean(rg_layer), ".", color="black")
            plt.text(fsed, np.nanmean(rg_layer), f"{Kzz:.1e}")
    rg_val = np.array(rg_val)
    plt.yscale("log")
    plt.show()

    rg_range = [np.min(rg_val), np.max(rg_val)]
    N_rg = 10
    rg_grid = np.logspace(np.log10(rg_range[0]), np.log10(rg_range[1]), N_rg)

    # make miegrid
    pdb_nh3.generate_miegrid(
        sigmagmin=sigmag,
        sigmagmax=sigmag,
        Nsigmag=1,
        log_rg_min=np.log10(rg_range[0]),
        log_rg_max=np.log10(rg_range[1]),
        Nrg=N_rg,
    )
    print("Please rerun after setting miegird_generate = True")
    import sys

    sys.exit()
    return Kzz,rg_layer,MMRc

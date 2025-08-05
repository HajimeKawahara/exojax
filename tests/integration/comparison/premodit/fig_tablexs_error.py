"""
This code is a snippet to check the tabulated cross section error.
See Issue #491 for more details.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py


def read_petitRadtrans_highres_xs_data(
    h5file="12C-16O__HITEMP.R1e6_0.3-28mu.xsec.petitRADTRANS.h5",
):
    """
    Read the high resolution cross section data from petitRadtrans.
    This code is a snippet to check the temperature grid of the pRT hihg-res cross section data.


    Last check in June 10th in 2024 (HK).
    12C-16O__HITEMP.R1e6_0.3-28mu.xsec.petitRADTRANS.h5 is the file used. 4.8GB in size (0.3-28micron).
    Temeprature grid = [  81.14113605  109.60677358  148.0586223   200.          270.16327371
    364.9409723   492.96823893  665.90956631  899.52154213 1215.08842295
    1641.36133093 2217.17775249 2995.        ] indicating delta log10t = 0.13059631
    Pressure Grid = [1.e-06 1.e-05 1.e-04 1.e-03 1.e-02 1.e-01 1.e+00 1.e+01 1.e+02 1.e+03]

    """
    with h5py.File(h5file, "r") as f:
        print(f.keys())
        p = f["p"][:]
        t = f["t"][:]
        print(t)
        print(p)
        log10t = np.log10(t)
        print(log10t[1:] - log10t[:-1])


def make_fig_tabulate_crosssection_error(
    p0=1.0e0, p1=1.0e1, t0=899.52154213, t1=1215.08842295, output="fig_tablexs_error.png"
):
    from exojax.utils.grids import wavenumber_grid

    nu, wav, res = wavenumber_grid(22910, 22960, 2000, xsmode="lpf", unit="AA")
    
    #tc = 10 ** ((np.log10(t1) + np.log10(t0)) / 2.0) #log interpolation
    tc = (t1 + t0) / 2.0 #linear interpolation
    pc = 10 ** ((np.log10(p1) + np.log10(p0)) / 2.0)

    print(p0,p1,pc)
    print(t0,t1,tc)

    from exojax.database.hitemp.api import MdbHitemp
    from exojax.opacity import OpaDirect

    mdb_co = MdbHitemp("CO", nurange=[nu[0], nu[-1]])
    mdb_h2o = MdbHitemp("H2O", nurange=[nu[0], nu[-1]])

    f, (ax, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
    cp = ["C0", "C1"]
    mol = ["H2O", "CO"]
    lwi = [1.0, 0.5]
    for i, mdb in enumerate([mdb_h2o, mdb_co]):
        opa = OpaDirect(mdb, nu)
        xs = opa.xsvector(tc, pc)
        xs00 = opa.xsvector(t0, p0)
        xs11 = opa.xsvector(t1, p1)
        xs01 = opa.xsvector(t0, p1)
        xs10 = opa.xsvector(t1, p0)

        #avexs = (xs00 + xs11) / 2.0
        avexs = (xs00 + xs10 + xs01 + xs11) / 4.0
        ax.plot(nu, xs, color=cp[i], label="direct (" + mol[i] + ")", lw=lwi[i])
        ax.plot(
            nu,
            avexs,
            color=cp[i],
            ls="dashed",
            label="grid interpolation " + "(" + mol[i] + ")",
            lw=lwi[i],
        )
        ax2.plot(nu, avexs / xs - 1.0, color=cp[i], label=mol[i], lw=lwi[i])
    ax.legend()
    ax2.set_ylim(-0.7,0.7)
    ax2.legend()
    ax2.axhline(0.0, color="k")
    ax.set_ylabel("cross section [cm2]")
    ax2.set_xlabel("wavenumber [cm-1]")
    ax2.set_ylabel("relative error")
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight", pad_inches=0.1)
    #plt.show()



if __name__ == "__main__":
    # read_petitRadtrans_highres_xs_data()
    make_fig_tabulate_crosssection_error(p0=1.0e0, p1=1.0e1, t0=899.52154213, t1=899.52154213, output="fig_tablexs_error_difp.png")
    make_fig_tabulate_crosssection_error(p0=1.0e0, p1=1.0e0, t0=899.52154213, t1=1215.08842295, output="fig_tablexs_error_dift.png")
    make_fig_tabulate_crosssection_error(p0=1.0e0, p1=1.0e1, t0=899.52154213, t1=1215.08842295, output="fig_tablexs_error.png")
    
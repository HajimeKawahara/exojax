from exojax.opacity.premodit.lbderror import single_tilde_line_strength_zeroth
import matplotlib.pyplot as plt
import numpy as np


def make_fig():
    """
    Figure for  the error of the line strength due to the grid of the lower energy level.
    """
    ttyp, tref, tlist, x = compute_line_strength_err(dE=300.0)
    ttyp, tref, tlist, x1 = compute_line_strength_err(dE=100.0)

    fig = plt.figure(figsize=(7, 3))
    plt.plot(1.0 / tlist, x, color="k", label="$\Delta E=300 \mathrm{cm}^{-1}$")
    plt.plot(
        1.0 / tlist,
        x1,
        color="red",
        ls="dashed",
        label="$\Delta E=100 \mathrm{cm}^{-1}$",
    )
    plt.ylim(-0.02, 0.03)
    plt.axhline(0.0, color="k")
    plt.axhline(0.01, color="gray", alpha=0.5)
    plt.axhline(-0.01, color="gray", alpha=0.5)
    plt.plot(1 / tref, 0.0, "o", color="green", alpha=0.5)
    plt.plot(1 / ttyp, 0.0, "o", color="red", alpha=0.5)
    plt.text(
        1 / tref, 0.0015, "$T_\mathrm{ref}$", color="green", alpha=1.0, fontsize=14
    )
    plt.text(1 / ttyp, 0.0015, "$T_\mathrm{wp}$", color="red", alpha=1.0, fontsize=14)
    plt.xlabel("Temperature [K]", fontsize=14)
    plt.ylabel("Line strength error", fontsize=14)
    plt.legend()
    plt.savefig("fig_elower_grid_error.png", bbox_inches="tight", pad_inches=0.1)
    plt.show()


def compute_line_strength_err(dE=300.0):
    """
    Compute the error of the line strength due to the grid of the lower energy level.

    Args:
        dE (float, optional): The interval of the Egrid. Defaults to 300.0 cm-1.

    Returns:
        _type_: _description_
    """
    p = 1.0 / 2.0
    ttyp = 1.0 / 1200.0
    tref = 1.0 / 500.0
    # tref = 1.0 / 1200.
    # ttyp = 1.0 / 500.0

    tlist = 1.0 / np.linspace(100, 2000, 100)
    x = single_tilde_line_strength_zeroth(tlist, ttyp, tref, dE, p)
    print(np.array([1 / tlist, x]).T)

    return ttyp, tref, tlist, x


if __name__ == "__main__":
    make_fig()

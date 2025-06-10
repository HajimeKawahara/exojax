"""This module tests the correlated k distribution implementation in Exojax."""

import numpy as np
import jax.numpy as jnp
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid
import matplotlib.pyplot as plt
from exojax.spec.opacalc import OpaPremodit
from jax import config

config.update("jax_enable_x64", True)  # use double precision

#from exojax.utils.grids import wavenumber_grid
#from exojax.spec.api import MdbExomol
# N = 70000
# nus, wav, res = wavenumber_grid(6400.0, 6800.0, N, unit="cm-1", xsmode = "premodit")
# mdb = MdbExomol(".databases/H2O/1H2-16O/POKAZATEL/",nus)
# print("resolution = ", res)

nus, wav, res = mock_wavenumber_grid(lambda0=22930.0, lambda1=22940.0, Nx=2000)
mdb = mock_mdbExomol("H2O")

opa = OpaPremodit(mdb, nus, auto_trange=[500.0, 1500.0])

T = 1000.0
P = 1.0e-2
xsv = opa.xsvector(T, P)


def computeg(nus, xsv, Ng=10):
    idx = jnp.argsort(xsv)
    k_g = xsv[idx]
    g = jnp.arange(xsv.size, dtype=xsv.dtype) / xsv.size

    # segments
    edges = jnp.linspace(0.0, 1.0, Ng + 1)  # 0,1/Ng,…,1
    cut_idx = jnp.searchsorted(g, edges)  # shape (Ng+1,)

    nus_segments = [nus[idx[cut_idx[i] : cut_idx[i + 1]]] for i in range(Ng)]
    xsv_segments = [xsv[idx[cut_idx[i] : cut_idx[i + 1]]] for i in range(Ng)]
    cut_idx = np.linspace(0, len(k_g), Ng + 1, dtype=int)

    j = 6
    k_low = k_g[cut_idx[j]]
    k_high = k_g[cut_idx[j + 1] - 1]
    k_med = k_low + (k_high - k_low) * 0.5
    mask = (xsv >= k_low) & (xsv < k_high)
    y_base = xsv.min() * 0.5 * jnp.ones_like(k_low)
    y_top = xsv.max() * 2.0 * jnp.ones_like(k_low)
    y_base_ = xsv.min() * 0.5 * jnp.ones(2)
    return (
        k_g,
        g,
        edges,
        cut_idx,
        nus_segments,
        xsv_segments,
        j,
        k_med,
        mask,
        y_base,
        y_base_,
        y_top,
    )


def plotxsv_1(nus, xsv, nus_segments, xsv_segments, j, k_med, mask, y_base, text, ax):
    plt.plot(nus, xsv, label="xsv", alpha=0.3)
    plt.fill_between(
        # nus, k_low, k_high,
        nus,
        y_base,
        y_top,
        where=mask,
        step="mid",
        color="tab:orange",
        alpha=0.35,
    )
    plt.plot(nus_segments[j], xsv_segments[j], ".", lw=0.5)
    plt.axhline(xsv_segments[j].max(), alpha=0.3, color="gray")
    plt.axhline(xsv_segments[j].min(), alpha=0.3, color="gray")
    ax.set_title(text, fontsize=8, loc="left", pad=0.5)
    plt.ylim(xsv.min() * 0.5, xsv.max() * 2)
    plt.yscale("log")
    plt.ylabel("$\\sigma(\\nu) (\\mathrm{cm}^{2})$")


fig = plt.figure(figsize=(13, 3.5))

temperatures = [700.0, 1000.0, 1000.0, 1000.0, 1300.0, 1000.0]
pressures = [0.1, 0.01, 0.1, 0.1, 0.1, 1.0]
titles = [
    "T=700 K, P=0.1 bar",
    "T=1000 K, P=0.01 bar",
    "T=1000 K, P=0.1 bar",
    "T=1000 K, P=0.1 bar",
    "T=1300 K, P=0.1 bar",
    "T=1000 K, P=1.0 bar",
]

for i, (T, P, title) in enumerate(zip(temperatures, pressures, titles)):
    ax = plt.subplot(3, 2, i + 1)
    xsv = opa.xsvector(T, P)
    (
        k_g,
        g,
        edges,
        cut_idx,
        nus_segments,
        xsv_segments,
        j,
        k_med,
        mask,
        y_base,
        y_base_,
        y_top,
    ) = computeg(nus, xsv)
    plotxsv_1(nus, xsv, nus_segments, xsv_segments, j, k_med, mask, y_base, title, ax)
    if i < 4:
        ax.xaxis.set_ticklabels([])
        ax.axes.get_xaxis().set_ticks([])
    if i == 4 or i == 5:
        ax.set_xlabel("$\\nu$ (cm-1)")
plt.savefig("corrk_corr.png", dpi=300, bbox_inches="tight")
plt.show()

T = 1000.0
P = 1.0e-2
xsv = opa.xsvector(T, P)
(
    k_g,
    g,
    edges,
    cut_idx,
    nus_segments,
    xsv_segments,
    j,
    k_med,
    mask,
    y_base,
    y_base_,
    y_top,
) = computeg(nus, xsv)


def plotxsv(nus, xsv, nus_segments, xsv_segments, j, k_med, mask, y_base):
    plt.plot(nus, xsv, label="xsv")
    plt.fill_between(
        # nus, k_low, k_high,
        nus,
        y_base,
        k_med,
        where=mask,
        step="mid",
        color="tab:orange",
        alpha=0.35,
    )
    plt.plot(nus_segments[j], xsv_segments[j], ".", lw=0.5)
    plt.axhline(xsv_segments[j].max(), alpha=0.3, color="gray")
    plt.axhline(xsv_segments[j].min(), alpha=0.3, color="gray")

    plt.ylim(xsv.min() * 0.5, xsv.max() * 2)
    plt.yscale("log")
    plt.ylabel("$\\sigma(\\nu) (\\mathrm{cm}^{2})$")


fig = plt.figure(figsize=(13, 3.5))
plt.subplot(1, 2, 1)
plotxsv(nus, xsv, nus_segments, xsv_segments, j, k_med, mask, y_base)
plt.xlabel("$\\nu$ (cm-1)")

plt.subplot(1, 2, 2)
plt.plot(g, k_g, label="xsv sorted")
plt.plot(g[cut_idx[j] : cut_idx[j + 1]], xsv_segments[j], ".", lw=0.5)

plt.ylim(xsv.min() * 0.5, xsv.max() * 2)
plt.axhline(xsv_segments[j].max(), alpha=0.3, color="gray")
plt.axhline(xsv_segments[j].min(), alpha=0.3, color="gray")
plt.axvline(edges[j], alpha=0.3, color="gray")
plt.axvline(edges[j + 1], alpha=0.3, color="gray")

plt.fill_between(
    # nus, k_low, k_high,
    [edges[j], edges[j + 1]],
    y_base,
    k_med,
    step="mid",
    color="tab:orange",
    alpha=0.35,
)

plt.text(
    (edges[j] + edges[j + 1]) / 2,
    y_base,
    "$\\Delta g_j$",
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=12,
    color="k",
)
plt.plot(edges[j : j + 2], y_base_, color="k", lw=3)
plt.yscale("log")
plt.xlabel("g")
plt.ylabel("$\\sigma(g) (\\mathrm{cm}^{2})$")
plt.savefig("corrk_test.png", dpi=300, bbox_inches="tight")
plt.show()

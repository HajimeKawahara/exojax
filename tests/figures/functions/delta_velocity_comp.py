import jax.numpy as jnp
import numpy as np
from exojax.utils.constants import c
import matplotlib.pyplot as plt
from jax import config
#config.update("jax_enable_x64", True)


def dv_comparison():

    N = 60
    Rarray = np.logspace(2, 7, N)
    dv_np = np.log(1.0 / Rarray + 1.0)
    dv_jnp = jnp.log(1.0 / Rarray + 1.0)
    dv_jnp_log1p = jnp.log1p(1.0 / Rarray)
    plt.plot(Rarray,
             np.abs(dv_jnp_log1p / dv_np - 1.0),
             ".",
             label="jnp.log1p(1/R) (FP32)")
    plt.plot(Rarray,
             np.abs(dv_jnp / dv_np - 1.0),
             ".",
             label="jnp.log(1/R + 1.0) (FP32)")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("R")
    plt.ylabel("difference from np.log(1.0/R + 1.0)")
    plt.legend()
    plt.savefig("delta_velocity_comp.png")
    plt.show()


def dv_each_comparison():
    nus = 4010.5
    N = 80
    Rinv = np.logspace(-9, -2, N)
    R = 1.0 / Rinv
    nusd = nus * (1.0 + Rinv)
    dv_np = np.array(c * np.log(nusd / nus))
    dv_jnp = jnp.array(c * jnp.log(nusd / nus))
    dv_jnp_ = jnp.array(c * (jnp.log(nusd) - jnp.log(nus)))
    dv_jnp_1p = jnp.array(c * (jnp.log1p(1.0 - nus / nusd)))
    plt.plot(R, np.abs(1.0 - dv_jnp / dv_np), label="c*log(nusd/nus)")
    plt.plot(R,
             np.abs(1.0 - dv_jnp_ / dv_np),
             label="c*(log(nusd)-log(nus))",
             ls="dotted")
    plt.plot(R,
             np.abs(1.0 - dv_jnp_1p / dv_np),
             label="c*log1p(1-nus/nusd)",
             ls="dashed")

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("R")
    plt.ylabel("difference from np")
    plt.legend()
    plt.savefig("delta_velocity_comp_each.png")
    plt.show()


if __name__ == "__main__":
    dv_comparison()
    dv_each_comparison()
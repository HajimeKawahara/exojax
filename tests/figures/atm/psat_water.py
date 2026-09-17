"""Compare water vapor-pressure formulas."""

import matplotlib.pyplot as plt
import numpy as np

from exojax.atm.psat import psat_water_AM01, psat_water_Magnus


if __name__ == "__main__":
    temperature = np.logspace(2, 3, 300)
    plt.plot(temperature, psat_water_Magnus(temperature), label="Magnus")
    plt.plot(temperature, psat_water_AM01(temperature), label="AM01 (Buck 81,96)")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("vapor pressure (bar)")
    plt.xlabel("temperature (K)")
    plt.legend()
    plt.show()

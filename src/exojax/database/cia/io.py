import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

HITRAN_DEFCIA = {
    "H2-CH4 (equilibrium)": "H2-CH4_eq_2011.cia",
    "H2-CH4 (normal)": "H2-CH4_norm_2011.cia",
    "H2-H2": "H2-H2_2011.cia",
    "H2-H": "H2-H_2011.cia",
    "H2-He": "H2-He_2011.cia",
    "He-H": "He-H_2011.cia",
    "N2-H2": "N2-H2_2011.cia",
    "N2-He": "N2-He_2018.cia",
    "N2-N2": "N2-N2_2018.cia",
    "N2-air": "N2-air_2018.cia",
    "N2-H2O": "N2-H2O_2018.cia",
    "O2-CO2": "O2-CO2_2011.cia",
    "O2-N2": "O2-N2_2018.cia",
    "O2-O2": "O2-O2_2018b.cia",
    "O2-air": "O2-Air_2018.cia",
    "CO2-CO2": "CO2-CO2_2018.cia",
    "CO2-H2": "CO2-H2_2018.cia",
    "CO2-He": "CO2-He_2018.cia",
    "CO2-CH4": "CO2-CH4_2018.cia",
    "CH4-He": "CH4-He_2018.cia",
}


def read_cia(filename, nus, nue):
    """READ hitrancia data.

    Args:
        filename: hitrancia file name (_2011.cia)
        nus: wavenumber min (cm-1)
        nue: wavenumber max (cm-1)

    Returns:
        nucia: wavenumber (cm-1)
        tcia: temperature (K)
        ac: cia coefficient
    """
    # read first line
    com = filename.split("/")[-1].split("_")[0]
    print(com)
    f = open(filename)
    header = f.readline()
    info = header.strip().split()
    nnu = int(info[3])
    nu = []
    for i in range(0, nnu):
        column = f.readline().strip().split()
        # print(column)
        nu.append(float(column[0]))
    f.close()
    f = open(filename)
    tcia = []
    for line in f:
        line = line.strip()
        column = line.split()
        if column[0] == com:
            tcia.append(float(column[4]))
    f.close()
    tcia = np.array(tcia)
    nu = np.array(nu)
    ijnu = np.digitize([nus, nue], nu)
    nucia = np.array(nu[ijnu[0] : ijnu[1] + 1])
    # read data
    data = np.loadtxt(filename, comments=com)
    nt = data.shape[0] / nnu
    data = data.reshape((int(nt), int(nnu), 2))
    ac = data[:, ijnu[0] : ijnu[1] + 1, 1]
    return nucia, tcia, ac



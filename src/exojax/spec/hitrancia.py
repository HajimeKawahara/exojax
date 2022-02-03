import numpy as np
import jax.numpy as jnp
from jax import jit, vmap


def read_cia(filename, nus, nue):
    """READ HITRAN CIA data.

    Args:
       filename: HITRAN CIA file name (_2011.cia)
       nus: wavenumber min (cm-1)
       nue: wavenumber max (cm-1)

    Returns:
       nucia: wavenumber (cm-1)
       tcia: temperature (K)
       ac: cia coefficient
    """
    # read first line
    com = filename.split('/')[-1].replace('_2011.cia', '')
    print(com)
    f = open(filename, 'r')
    header = f.readline()
    info = header.strip().split()
    nnu = int(info[3])
    nu = []
    for i in range(0, nnu):
        column = f.readline().strip().split()
        nu.append(float(column[0]))
    f.close()
    f = open(filename, 'r')
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
    nucia = np.array(nu[ijnu[0]:ijnu[1]+1])
    # read data
    data = np.loadtxt(filename, comments=com)
    nt = data.shape[0]/nnu
    data = data.reshape((int(nt), int(nnu), 2))
    ac = data[:, ijnu[0]:ijnu[1]+1, 1]
    return nucia, tcia, ac


@jit
def logacia(Tarr, nus, nucia, tcia, logac):
    """interpolated function of log10(alpha_CIA)

    Args:
       Tarr: temperature array
       nus: wavenumber array
       nucia: CIA wavenumber (cm-1)
       tcia: CIA temperature (K)
       logac: log10 cia coefficient

    Returns:
       logac(Tarr, nus)

    Example:
       >>> nucia,tcia,ac=read_cia("../../data/CIA/H2-H2_2011.cia",nus[0]-1.0,nus[-1]+1.0)
       >>> logac=jnp.array(np.log10(ac))
       >>> logacia(Tarr,nus,nucia,tcia,logac)
    """
    def fcia(x, i): return jnp.interp(x, tcia, logac[:, i])
    vfcia = vmap(fcia, (None, 0), 0)
    mfcia = vmap(vfcia, (0, None), 0)
    inus = jnp.digitize(nus, nucia)
    return mfcia(Tarr, inus)


if __name__ == '__main__':
    nucia, tcia, ac = read_cia(
        '../../data/CIA/H2-H2_2011.cia', nus[0]-1.0, nus[-1]+1.0)
    logac = jnp.array(np.log10(ac))
    logacia(Tarr, nus, nucia, tcia, logac)

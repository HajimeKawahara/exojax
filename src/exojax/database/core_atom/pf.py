import jax.numpy as jnp
import numpy as np
def interp_QT_284(T, T_gQT, gQT_284species):
    """interpolated partition function of all 284 species.

    Args:
        T: temperature
        T_gQT: temperature in the grid obtained from the adb instance [N_grid(42)]
        gQT_284species: partition function in the grid from the adb instance [N_species(284) x N_grid(42)]

    Returns:
        QT_284: interpolated partition function at T Q(T) for all 284 Atomic Species [284]
    """
    list_gQT_eachspecies = gQT_284species.tolist()
    listofDA_gQT_eachspecies = list(map(lambda x: jnp.array(x), list_gQT_eachspecies))
    listofQT = list(map(lambda x: jnp.interp(T, T_gQT, x), listofDA_gQT_eachspecies))
    QT_284 = jnp.array(listofQT)
    return QT_284

def partfn_Fe(T):
    """Partition function of Fe I from Irwin_1981.

    Args:
        T: temperature

    Returns:
        partition function Q
    """
    # Irwin_1981
    a = np.zeros(6)
    a[0] = -1.15609527e3
    a[1] = 7.46597652e2
    a[2] = -1.92865672e2
    a[3] = 2.49658410e1
    a[4] = -1.61934455e0
    a[5] = 4.21182087e-2

    Qln = 0.0
    for i, a in enumerate(a):
        Qln = Qln + a * np.log(T) ** i
    Q = np.exp(Qln)
    return Q

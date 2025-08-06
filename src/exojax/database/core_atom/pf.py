import jax.numpy as jnp
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

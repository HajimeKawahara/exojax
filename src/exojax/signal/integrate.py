import jax.numpy as jnp


def simpson(f, f_lower, f_top, h):
    """Simpson integral for (Nlayer, Nnus) form

    Args:
        f (2D array): f at the midpoint (Nlayer, Nnus)
        f_lower (2D array): f at the lower bondary (Nlayer, Nnus)
        f_top (1D array): f at the ToA  (Nnus)
        h (1D array): interval (Nlayer)

    Returns:
        1D array: simson integral value (Nnus)
    """
    N = len(f)
    hh = jnp.roll(h, -1) + h  # h_{n+1} + h_n
    fac = hh[: N - 1, None] * f_lower[: N - 1, :]
    return (
        2.0 / 3.0 * jnp.sum(h[:, None] * f, axis=0)
        + h[0] * f_top / 6.0
        + h[-1] * f_lower[-1, :] / 6.0
        + jnp.sum(fac, axis=0) / 6.0
    )

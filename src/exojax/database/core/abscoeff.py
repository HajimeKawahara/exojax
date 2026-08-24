"""Interpolate collision-induced absorption coefficients."""

import jax.numpy as jnp
from jax import custom_jvp, jit, vmap


@custom_jvp
def _mix_logacia(lower_logac, upper_logac, weight):
    """Mix two log10 coefficients with a linear coefficient weight."""

    offset = jnp.maximum(lower_logac, upper_logac)
    offset = jnp.where(jnp.isfinite(offset), offset, 0.0)
    is_interior = (weight > 0.0) & (weight < 1.0)
    safe_weight = jnp.where(is_interior, weight, 0.5)
    interpolated = jnp.log10(
        (1.0 - safe_weight) * 10.0 ** (lower_logac - offset)
        + safe_weight * 10.0 ** (upper_logac - offset)
    ) + offset
    result = jnp.where(
        weight <= 0.0,
        lower_logac,
        jnp.where(weight >= 1.0, upper_logac, interpolated),
    )
    return jnp.where(jnp.isnan(weight), jnp.nan, result)


@_mix_logacia.defjvp
def _mix_logacia_jvp(primals, tangents):
    lower_logac, upper_logac, weight = primals
    lower_dot, upper_dot, weight_dot = tangents
    result = _mix_logacia(lower_logac, upper_logac, weight)

    lower_weight = 1.0 - weight
    safe_lower_weight = jnp.where(lower_weight > 0.0, lower_weight, 1.0)
    safe_upper_weight = jnp.where(weight > 0.0, weight, 1.0)
    lower_fraction = jnp.where(
        lower_weight > 0.0,
        10.0
        ** jnp.minimum(
            jnp.log10(safe_lower_weight) + lower_logac - result, 0.0
        ),
        0.0,
    )
    upper_fraction = jnp.where(
        weight > 0.0,
        10.0
        ** jnp.minimum(
            jnp.log10(safe_upper_weight) + upper_logac - result, 0.0
        ),
        0.0,
    )
    # Leave headroom when an endpoint derivative exceeds the dtype range.
    exponent_limit = jnp.log10(
        jnp.asarray(jnp.finfo(result.dtype).max, dtype=result.dtype)
    ) - 2.0
    upper_ratio = 10.0 ** jnp.minimum(
        upper_logac - result, exponent_limit
    )
    lower_ratio = 10.0 ** jnp.minimum(
        lower_logac - result, exponent_limit
    )
    weight_gradient = (upper_ratio - lower_ratio) / jnp.log(
        jnp.asarray(10.0, dtype=result.dtype)
    )
    tangent = (
        lower_fraction * lower_dot
        + upper_fraction * upper_dot
        + weight_gradient * weight_dot
    )
    return result, tangent


def _interp_logacia_in_coefficient_space(x, xp, logac):
    """Linearly interpolate coefficients stored as log10 values."""

    if xp.shape[0] == 1:
        return jnp.broadcast_to(logac[0], jnp.shape(x))

    upper = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, xp.size - 1)
    lower = upper - 1
    raw_weight = (x - xp[lower]) / (xp[upper] - xp[lower])
    weight = jnp.where(
        x < xp[0], 0.0, jnp.where(x > xp[-1], 1.0, raw_weight)
    )
    return _mix_logacia(logac[lower], logac[upper], weight)


@jit
def _interp_logacia_matrix(Tarr, nu_grid, nucia, tcia, logac):
    """Interpolate linear CIA coefficients and return their logarithms."""

    def interpolate_temperature(temperature):
        return vmap(
            _interp_logacia_in_coefficient_space,
            in_axes=(None, None, 1),
        )(temperature, tcia, logac)

    logac_at_temperature = vmap(interpolate_temperature)(Tarr)
    return vmap(
        _interp_logacia_in_coefficient_space,
        in_axes=(None, None, 0),
    )(nu_grid, nucia, logac_at_temperature)


@jit
def _digitize_logacia_matrix(Tarr, nu_grid, nucia, tcia, logac):
    """Apply the legacy log-temperature and wavenumber-bin interpolation."""

    def fcia(x, i):
        return jnp.interp(x, tcia, logac[:, i])

    vfcia = vmap(fcia, (None, 0), 0)
    mfcia = vmap(vfcia, (0, None), 0)
    inus = jnp.digitize(nu_grid, nucia)
    return mfcia(Tarr, inus)


def interp_logacia_matrix(
    Tarr,
    nu_grid,
    nucia,
    tcia,
    logac,
    wavenumber_interpolation="interp",
):
    """Interpolate log10 CIA coefficients for an atmospheric profile.

    Args:
        Tarr (1D array): temperature array (K) [Nlayer]
        nu_grid (1D array): wavenumber array (cm-1) [Nnus]
        nucia: Native CIA wavenumber grid (cm-1).
        tcia: Native CIA temperature grid (K).
        logac: Native log10 CIA coefficients in cm5.
        wavenumber_interpolation: ``"interp"`` interpolates linear
            coefficients in temperature and wavenumber. ``"digitize"``
            reproduces the legacy log-temperature interpolation and
            right-bin wavenumber selection.

    Returns:
        Log10 absorption coefficient [Nlayer, Nnus] in units of cm5.

    Example:
        nucia,tcia,ac=read_cia(
            "../../data/CIA/H2-H2_2011.cia", nus[0]-1.0, nus[-1]+1.0
        )
        logac=jnp.array(np.log10(ac))
        interp_logacia_matrix(Tarr,nus,nucia,tcia,logac)
    """

    if wavenumber_interpolation == "interp":
        return _interp_logacia_matrix(Tarr, nu_grid, nucia, tcia, logac)
    if wavenumber_interpolation == "digitize":
        return _digitize_logacia_matrix(Tarr, nu_grid, nucia, tcia, logac)
    raise ValueError("wavenumber_interpolation must be 'interp' or 'digitize'.")


@jit
def _interp_logacia_vector(T, nu_grid, nucia, tcia, logac):
    """Interpolate one linear CIA coefficient vector in log representation."""

    return _interp_logacia_matrix(
        jnp.atleast_1d(T), nu_grid, nucia, tcia, logac
    )[0]


@jit
def _digitize_logacia_vector(T, nu_grid, nucia, tcia, logac):
    """Apply the legacy interpolation to one CIA coefficient vector."""

    def fcia(x, i):
        return jnp.interp(x, tcia, logac[:, i])

    vfcia = vmap(fcia, (None, 0), 0)
    inus = jnp.digitize(nu_grid, nucia)
    return vfcia(T, inus)


def interp_logacia_vector(
    T,
    nu_grid,
    nucia,
    tcia,
    logac,
    wavenumber_interpolation="interp",
):
    """Interpolate a log10 CIA coefficient vector at one temperature.

    Args:
        T (float): Temperature in Kelvin.
        nu_grid: Wavenumber array (cm-1).
        nucia: Native CIA wavenumber grid (cm-1).
        tcia: Native CIA temperature grid (K).
        logac: Native log10 CIA coefficients in cm5.
        wavenumber_interpolation: ``"interp"`` interpolates linear
            coefficients in temperature and wavenumber. ``"digitize"``
            reproduces the legacy log-temperature interpolation and
            right-bin wavenumber selection.

    Returns:
        Log10 absorption coefficient [Nnus] at T in units of cm5.

    """

    if wavenumber_interpolation == "interp":
        return _interp_logacia_vector(T, nu_grid, nucia, tcia, logac)
    if wavenumber_interpolation == "digitize":
        return _digitize_logacia_vector(T, nu_grid, nucia, tcia, logac)
    raise ValueError("wavenumber_interpolation must be 'interp' or 'digitize'.")

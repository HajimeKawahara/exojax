""" compute opacity difference in atmospheric layers

"""

import jax.numpy as jnp
from exojax.atm.idealgas import number_density
from exojax.database.core.abscoeff import interp_logacia_matrix
from exojax.database.hminus  import log_hminus_continuum, log_hminus_continuum_single
from exojax.utils.constants import bar_cgs, logkB, logm_ucgs, opacity_factor


def _layer_opacity_array(coefficient, number_of_layers, name):
    """Add a layer axis to a common spectrum or validate a layered array."""
    coefficient = jnp.asarray(coefficient)
    if coefficient.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension.")
    if coefficient.ndim == 1:
        return coefficient[None, :]
    if coefficient.shape[0] not in (1, number_of_layers):
        raise ValueError(
            f"The leading axis of {name} must have length 1 or "
            f"{number_of_layers}, but has length {coefficient.shape[0]}."
        )
    return coefficient


def _layer_factor(value, number_of_layers, target_shape, name):
    """Reshape a layer-dependent factor for opacity-array broadcasting."""
    value = jnp.asarray(value)
    if value.ndim == 0:
        return value
    if value.shape[0] not in (1, number_of_layers):
        raise ValueError(
            f"The leading axis of {name} must have length 1 or "
            f"{number_of_layers}, but has length {value.shape[0]}."
        )
    if value.ndim > len(target_shape):
        raise ValueError(
            f"{name} has {value.ndim} dimensions, but the opacity array has "
            f"only {len(target_shape)}."
        )

    value_shape = value.shape + (1,) * (len(target_shape) - value.ndim)
    for value_size, target_size in zip(value_shape[1:], target_shape[1:]):
        if value_size not in (1, target_size):
            raise ValueError(
                f"{name} with shape {value.shape} cannot be broadcast to an "
                f"opacity array with shape {target_shape}."
            )
    if value.shape == value_shape:
        return value
    return value.reshape(value_shape)


def _layer_opacity_inputs(coefficient, layer_factor, coefficient_name, factor_name):
    """Validate and broadcast an opacity coefficient and a layer factor."""
    coefficient = jnp.asarray(coefficient)
    if coefficient.ndim == 0:
        raise ValueError(f"{coefficient_name} must have at least one dimension.")

    layer_factor = jnp.asarray(layer_factor)
    if layer_factor.ndim == 0:
        number_of_layers = coefficient.shape[0] if coefficient.ndim > 1 else 1
    else:
        number_of_layers = layer_factor.shape[0]
        if number_of_layers == 0:
            raise ValueError(f"{factor_name} must contain at least one layer.")

    coefficient = _layer_opacity_array(
        coefficient, number_of_layers, coefficient_name
    )
    layer_factor = _layer_factor(
        layer_factor, number_of_layers, coefficient.shape, factor_name
    )
    return coefficient, layer_factor


def _layer_optical_depth_from_cross_section(
    cross_section, absorber_number_column
):
    """Multiply prepared cross sections and absorber number columns."""
    return cross_section * absorber_number_column


def _pressure_number_of_layers(cross_section, *layer_factors):
    """Infer the broadcast layer size of pressure-based opacity inputs."""
    layer_sizes = []
    if cross_section.ndim > 1:
        layer_sizes.append(cross_section.shape[0])
    for factor in layer_factors:
        if factor.ndim > 0:
            layer_sizes.append(factor.shape[0])
    non_singleton_sizes = [size for size in layer_sizes if size != 1]
    return max(non_singleton_sizes, default=1)


def _layer_optical_depth_from_pressure(
    dpressure,
    cross_section,
    mixing_ratio,
    mass,
    gravity,
    cross_section_name,
):
    """Build a pressure-based absorber column and apply its cross section."""
    cross_section = jnp.asarray(cross_section)
    dpressure = jnp.asarray(dpressure)
    mixing_ratio = jnp.asarray(mixing_ratio)
    mass = jnp.asarray(mass)
    gravity = jnp.asarray(gravity)
    number_of_layers = _pressure_number_of_layers(
        cross_section, dpressure, mixing_ratio, mass, gravity
    )
    cross_section = _layer_opacity_array(
        cross_section, number_of_layers, cross_section_name
    )
    dpressure = _layer_factor(
        dpressure, number_of_layers, cross_section.shape, "dParr"
    )
    mixing_ratio = _layer_factor(
        mixing_ratio, number_of_layers, cross_section.shape, "mixing_ratio"
    )
    mass = _layer_factor(mass, number_of_layers, cross_section.shape, "mass")
    gravity = _layer_factor(
        gravity, number_of_layers, cross_section.shape, "gravity"
    )
    absorber_number_column = (
        opacity_factor * dpressure * mixing_ratio / (mass * gravity)
    )
    return _layer_optical_depth_from_cross_section(
        cross_section, absorber_number_column
    )


def layer_optical_depth_from_cross_section(cross_section, absorber_number_column):
    """Compute layer optical depth from an absorption cross section.

    This function evaluates ``dtau = sigma * N_absorber`` in cgs units.
    The absorber column may be obtained from either a pressure interval or a
    geometrical path length. This coordinate-independent product is shared by
    the pressure-based layer optical-depth functions.

    A one-dimensional cross section is interpreted as a common spectral vector
    and is broadcast over all layers. Arrays with two or more dimensions must
    have the layer axis first, for example ``(Nlayer, Nnu)`` for line-by-line
    opacity or ``(Nlayer, Ng, Nband)`` for correlated-k opacity.

    Args:
        cross_section: Absorption cross section in cm2. Its shape is ``(Nnu,)``
            or ``(Nlayer, ...)``.
        absorber_number_column: Absorber column number density in cm-2,
            provided as a scalar or an array with leading layer axis.

    Returns:
        Dimensionless layer optical depth with shape ``(Nlayer, ...)``.
    """
    cross_section, absorber_number_column = _layer_opacity_inputs(
        cross_section,
        absorber_number_column,
        "cross_section",
        "absorber_number_column",
    )
    return _layer_optical_depth_from_cross_section(
        cross_section, absorber_number_column
    )


def layer_optical_depth_from_log_cia(
    log_cia_coefficient, number_density_1, number_density_2, path_length
):
    """Compute geometric CIA optical depth without linearizing the coefficient.

    The calculation is performed as
    ``log10(dtau) = log10(k_cia) + log10(n1) + log10(n2) + log10(dz)``.
    Keeping the CIA coefficient in logarithmic form avoids an intermediate
    underflow in single precision. No symmetry factor is applied when the two
    collision partners are identical.

    Args:
        log_cia_coefficient: Base-10 logarithm of the CIA coefficient in cm5.
            Its shape is ``(Nnu,)`` or ``(Nlayer, ...)``.
        number_density_1: Number density of the first collision partner in
            cm-3, provided as a scalar or an array with shape ``(Nlayer,)``.
        number_density_2: Number density of the second collision partner in
            cm-3, provided as a scalar or an array with shape ``(Nlayer,)``.
        path_length: Geometric path length in cm with shape ``(Nlayer,)``.

    Returns:
        Dimensionless layer optical depth with shape ``(Nlayer, ...)``. A zero
        number density or zero path length produces exactly zero.
    """
    path_length = jnp.asarray(path_length)
    if path_length.ndim != 1:
        raise ValueError("path_length must be a one-dimensional array.")
    log_cia_coefficient, path_length = _layer_opacity_inputs(
        log_cia_coefficient,
        path_length,
        "log_cia_coefficient",
        "path_length",
    )
    number_of_layers = path_length.shape[0]
    number_density_1 = _layer_factor(
        number_density_1,
        number_of_layers,
        log_cia_coefficient.shape,
        "number_density_1",
    )
    number_density_2 = _layer_factor(
        number_density_2,
        number_of_layers,
        log_cia_coefficient.shape,
        "number_density_2",
    )

    zero_optical_depth = (
        (number_density_1 == 0)
        | (number_density_2 == 0)
        | (path_length == 0)
    )
    safe_number_density_1 = jnp.where(
        number_density_1 == 0, jnp.ones_like(number_density_1), number_density_1
    )
    safe_number_density_2 = jnp.where(
        number_density_2 == 0, jnp.ones_like(number_density_2), number_density_2
    )
    safe_path_length = jnp.where(
        path_length == 0, jnp.ones_like(path_length), path_length
    )
    log_optical_depth = (
        log_cia_coefficient
        + jnp.log10(safe_number_density_1)
        + jnp.log10(safe_number_density_2)
        + jnp.log10(safe_path_length)
    )
    log_optical_depth = jnp.where(
        zero_optical_depth,
        jnp.full_like(log_optical_depth, -jnp.inf),
        log_optical_depth,
    )
    return jnp.power(
        jnp.asarray(10.0, dtype=log_optical_depth.dtype), log_optical_depth
    )


def layer_optical_depth_from_extinction(extinction_coefficient, path_length):
    """Compute geometric layer optical depth from an extinction coefficient.

    This function evaluates ``dtau = alpha * dz`` with the extinction
    coefficient in cm-1 and the layer thickness in cm. A one-dimensional
    coefficient is interpreted as a common spectral vector; layered arrays
    must have the layer axis first.

    Args:
        extinction_coefficient: Extinction coefficient in cm-1. Its shape is
            ``(Nnu,)`` or ``(Nlayer, ...)``.
        path_length: Geometric path length in cm with shape ``(Nlayer,)``.

    Returns:
        Dimensionless layer optical depth with shape ``(Nlayer, ...)``.
    """
    path_length = jnp.asarray(path_length)
    if path_length.ndim != 1:
        raise ValueError("path_length must be a one-dimensional array.")
    extinction_coefficient, path_length = _layer_opacity_inputs(
        extinction_coefficient,
        path_length,
        "extinction_coefficient",
        "path_length",
    )
    return extinction_coefficient * path_length


def single_layer_optical_depth(dpressure, xsv, mixing_ratio, mass, gravity):
    """opacity for a single layer (delta tau) from cross section vector, molecular line/Rayleigh scattering (for opart)

    Args:
        dpressure (float): pressure difference (dP) of the layer in bar
        xsv (array): cross section vector i.e. xsvector (N_wavenumber)
        mixing_ratio (float): mass mixing ratio, (or volume mixing ratio profile)
        mass (float): molecular mass (or mean molecular weight)
        gravity (float): constant or 1d profile of gravity in cgs

    Returns:
        dtau (array): opacity whose element is optical depth in a single layer [N_wavenumber].
    """
    return opacity_factor * xsv * dpressure * mixing_ratio / (mass * gravity)


def layer_optical_depth(dParr, xsmatrix, mixing_ratio, mass, gravity):
    """dtau matrix from the cross section matrix/vector.

    Note:
        opfac=bar_cgs/(m_u (g)). m_u: atomic mass unit. It can be obtained by fac=1.e3/m_u, where m_u = scipy.constants.m_u.

    Args:
        dParr (array): delta pressure profile (bar) [N_layer]
        xsmatrix (2D or 1D array): cross section matrix (cm2) [N_layer, N_nus] or cross section vector (cm2) [N_nus]
        mixing_ratio (array): volume mixing ratio (VMR) or mass mixing ratio (MMR) [N_layer]
        mass: mean molecular weight for VMR or molecular mass for MMR
        gravity: gravity (cm/s2)

    Returns:
        2D array: optical depth matrix, dtau  [N_layer, N_nus]
    """

    dParr = jnp.asarray(dParr)
    if dParr.ndim != 1:
        raise ValueError("dParr must be a one-dimensional array.")
    return _layer_optical_depth_from_pressure(
        dParr,
        xsmatrix,
        mixing_ratio,
        mass,
        gravity,
        "xsmatrix",
    )


def layer_optical_depth_ckd(dParr, xstensor_ckd, mixing_ratio, mass, gravity):
    """dtau tensor from the CKD cross section tensor for correlated-k distribution.

    Args:
        dParr (array): delta pressure profile (bar) [N_layer]
        xstensor_ckd (3D array): CKD cross section tensor (cm2) [N_layer, N_g, N_bands]
        mixing_ratio (array): volume mixing ratio (VMR) or mass mixing ratio (MMR) [N_layer]
        mass: mean molecular weight for VMR or molecular mass for MMR
        gravity: gravity (cm/s2), scalar or 1D profile [N_layer]

    Returns:
        3D array: optical depth tensor, dtau_ckd [N_layer, N_g, N_bands]
    """

    xstensor_ckd = jnp.asarray(xstensor_ckd)
    if xstensor_ckd.ndim != 3:
        raise ValueError("xstensor_ckd must be a three-dimensional array.")
    return _layer_optical_depth_from_pressure(
        dParr,
        xstensor_ckd,
        mixing_ratio,
        mass,
        gravity,
        "xstensor_ckd",
    )


def single_layer_optical_depth_CIA(
    temperature, pressure, dpressure, vmr1, vmr2, mmw, g, logacia_vector
):
    """dtau of the CIA continuum for a single layer (for opart).

    Args:
        temperature (float): layer temperature (K)
        pressure (float): layer pressure (bar)
        dpressure (float) : delta temperature (bar)
        vmr1 (float): volume mixing ratio (VMR) for molecules 1
        vmr2 (float): volume mixing ratio (VMR) for molecules 2
        mmw: mean molecular weight of atmosphere
        g: gravity (cm2/s)
        logacia_vector: log CIA coefficient vector [N_nus], usually obtained by opacont.OpaCIA.logacia_vector(temperature)

    Returns:
        1D array: optical depth matrix, dtau  [N_nus]
    """
    n = number_density(pressure, temperature)
    logn1 = jnp.log10(vmr1 * n)  # log number density
    logn2 = jnp.log10(vmr2 * n)  # log number density
    logg = jnp.log10(g)
    ddpressure = dpressure / pressure
    dtauc = (
        10 ** (logacia_vector + logn1 + logn2 + logkB - logg - logm_ucgs)
        * temperature
        / mmw
        * ddpressure
    )

    return dtauc


def layer_optical_depth_CIA(
    nu_grid,
    temperature,
    pressure,
    dParr,
    vmr1arr,
    vmr2arr,
    mmw,
    g,
    nucia,
    tcia,
    logac,
    wavenumber_interpolation="interp",
):
    """dtau of the CIA continuum.

    Warnings:
        Not used in art.

    Args:
        nu_grid (array): wavenumber matrix (cm-1)
        temperature (array): temperature array (K)
        pressure (array): pressure array (bar)
        dParr (array): delta temperature array (bar)
        vmr1arr (array): volume mixing ratio (VMR) for molecules 1 [N_layer]
        vmr2arr  (array): volume mixing ratio (VMR) for molecules 2 [N_layer]
        mmw: mean molecular weight of atmosphere
        g: gravity (cm2/s)
        nucia (array): wavenumber array for CIA
        tcia (array): temperature array for CIA
        logac: log10(absorption coefficient of CIA)
        wavenumber_interpolation: CIA interpolation method, ``"interp"`` or
            ``"digitize"``.

    Returns:
        2D array: optical depth matrix, dtau  [N_layer, N_nus]
    """
    narr = number_density(pressure, temperature)
    lognarr1 = jnp.log10(vmr1arr * narr)  # log number density
    lognarr2 = jnp.log10(vmr2arr * narr)  # log number density
    logg = jnp.log10(g)
    ddParr = dParr / pressure
    dtauc = (
        10
        ** (
            interp_logacia_matrix(
                temperature,
                nu_grid,
                nucia,
                tcia,
                logac,
                wavenumber_interpolation,
            )
            + lognarr1[:, None]
            + lognarr2[:, None]
            + logkB
            - logg
            - logm_ucgs
        )
        * temperature[:, None]
        / mmw
        * ddParr[:, None]
    )

    return dtauc


def single_layer_optical_depth_Hminus(
    nu_grid, temperature, pressure, dpressure, vmre, vmrh, mmw, g
):
    """dtau of the H- continuum for a single layer (e.g. for opart).

    Args:
        nu_grid (array): wavenumber matrix (cm-1) [N_nus]
        temperature (float): temperature (K)
        pressure (float): pressure (bar)
        dpressure (float): delta pressure (bar)
        vmre: volume mixing ratio (VMR) for e- [N_layer]
        vmrH: volume mixing ratio (VMR) for H atoms [N_layer]
        mmw: mean molecular weight of atmosphere
        g: gravity (cm2/s)

    Returns:
        optical depth matrix  [N_layer, N_nus]
    """
    n = number_density(pressure, temperature)
    number_density_e = vmre * n  # number density for e- [N_layer]
    number_density_h = vmrh * n  # number density for H atoms [N_layer]
    logg = jnp.log10(g)
    ddParr = dpressure / pressure
    logabc = log_hminus_continuum_single(
        nu_grid, temperature, number_density_e, number_density_h
    )
    dtauh = 10 ** (logabc + logkB - logg - logm_ucgs) * temperature / mmw * ddParr

    return dtauh


def layer_optical_depth_Hminus(
    nu_grid, temperature, pressure, dParr, vmre, vmrh, mmw, gravity
):
    """dtau of the H- continuum.

    Args:
        nu_grid (array): wavenumber matrix (cm-1) [N_nus]
        temperature (array): temperature array (K) [N_layer]
        pressure (array): pressure array (bar) [N_layer]
        dParr (array): delta temperature array (bar) [N_layer]
        vmre (array): volume mixing ratio (VMR) for e- [N_layer]
        vmrH: volume mixing ratio (VMR) for H atoms [N_layer]
        mmw: mean molecular weight of atmosphere
        gravity: gravity (cm2/s)

    Returns:
        optical depth matrix  [N_layer, N_nus]
    """
    narr = number_density(pressure, temperature)
    number_density_e = vmre * narr  # number density for e- [N_layer]
    number_density_h = vmrh * narr  # number density for H atoms [N_layer]
    logg = jnp.log10(gravity)
    ddParr = dParr / pressure
    logabc = log_hminus_continuum(
        nu_grid, temperature, number_density_e, number_density_h
    )
    dtauh = (
        10 ** (logabc + logkB - logg - logm_ucgs)
        * temperature[:, None]
        / mmw
        * ddParr[:, None]
    )

    return dtauh


def layer_optical_depth_cloudgeo(
    dParr, condensate_substance_density, mmr_condensate, rg, sigmag, gravity
):
    """the optical depth using a geometric cross-section approximation, based
    on (16) in AM01.

    Args:
        dParr: delta pressure profile (bar)
        condensate_substance_density: condensate substance density (g/cm3)
        mmr_condensate: Mass mixing ratio (array) of condensate [Nlayer]
        rg: rg parameter in the lognormal distribution of condensate size, defined by (9) in AM01
        sigmag:sigmag parameter (geometric standard deviation) in the lognormal distribution of condensate size, defined by (9) in AM01, must be sigmag > 1
        gravity: gravity (cm/s2)

    """

    fac = jnp.exp(-2.5 * jnp.log(sigmag) ** 2)
    dtau = (
        1.5
        * mmr_condensate
        * fac
        / (rg * condensate_substance_density * gravity)
        * dParr
        * bar_cgs
    )
    return dtau


def single_layer_optical_depth_clouds_lognormal(
    dpressure,
    extinction_coefficient,
    condensate_substance_density,
    mmr_condensate,
    rg,
    sigmag,
    gravity,
    N0=1.0,
):
    """dtau matrix from the cross section matrix/vector for the lognormal particulate distribution, for a single layer.


    Args:
        dpressure (float): delta pressure (bar)
        extinction coefficient (array): extinction coefficient  in cgs (cm-1) [N_nus]
        condensate_substance_density (float): condensate substance density (g/cm3)
        mmr_condensate (float): Mass mixing ratio of condensate
        rg (float): rg parameter in the lognormal distribution of condensate size, defined by (9) in AM01
        sigmag (float):sigmag parameter (geometric standard deviation) in the lognormal distribution of condensate size, defined by (9) in AM01, must be sigmag > 1
        gravity (float): gravity (cm/s2)
        N0 (float, optional): the normalization of the lognormal distribution ($N_0$). Defaults to 1.0.

    Returns:
        1D array: optical depth matrix, dtau  [N_nus]
    """
    expfac = bar_cgs * sigmag ** (
        jnp.log(sigmag**-4.5)
    )  # bar_cgs * exp(-9/2 * (log sigmag)**2), see tests/manual_check/f32/lnmoment_amcloud.py
    fac = 0.75 / jnp.pi / rg**3 / condensate_substance_density
    em = extinction_coefficient * mmr_condensate / N0
    return expfac * fac * em * dpressure / gravity


def layer_optical_depth_clouds_lognormal(
    dParr,
    extinction_coefficient,
    condensate_substance_density,
    mmr_condensate,
    rg,
    sigmag,
    gravity,
    N0=1.0,
):
    """dtau matrix from the cross section matrix/vector for the lognormal particulate distribution.


    Args:
        dParr (array): delta pressure profile (bar) [N_layer]
        extinction coefficient (array): extinction coefficient  in cgs (cm-1) [N_layer, N_nus]
        condensate_substance_density (float): condensate substance density (g/cm3)
        mmr_condensate (array): Mass mixing ratio (array) of condensate [Nlayer]
        rg (float or array): rg parameter in the lognormal distribution of condensate size, defined by (9) in AM01 [N_layer]
        sigmag (float or array): sigmag parameter (geometric standard deviation) in the lognormal distribution of condensate size, defined by (9) in AM01, must be sigmag > 1 [N_layer]
        gravity (float or array): gravity (cm/s2) [N_layer]
        N0 (float, optional): the normalization of the lognormal distribution ($N_0$). Defaults to 1.0.

    Returns:
        2D array: optical depth matrix, dtau  [N_layer, N_nus]
    """
    if jnp.ndim(rg) == 1:
        rg = rg[:, None]
    if jnp.ndim(sigmag) == 1:
        sigmag = sigmag[:, None]
    if jnp.ndim(gravity) == 1:
        gravity = gravity[:, None]

    expfac = bar_cgs * sigmag ** (
        jnp.log(sigmag**-4.5)
    )  # bar_cgs * exp(-9/2 * (log sigmag)**2), see tests/manual_check/f32/lnmoment_amcloud.py
    fac = 0.75 / jnp.pi / rg**3 / condensate_substance_density
    em = extinction_coefficient * mmr_condensate[:, None] / N0
    return expfac * fac * em * dParr[:, None] / gravity

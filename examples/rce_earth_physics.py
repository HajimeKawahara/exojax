"""Liquid-water thermodynamics for the N2--H2O forward RCE example.

Pressure arguments are in bar; all thermodynamic constants use SI units.
The ocean supplies saturated vapor, condensate rains out immediately, and
the cold trap limits water above it. Ice and cloud opacity are omitted.
The saturation law is extrapolated over supercooled liquid below 273.16 K.
"""

import jax
import jax.numpy as jnp


MOLAR_MASS_N2 = 0.0280134  # kg/mol
MOLAR_MASS_H2O = 0.01801528  # kg/mol
R_DRY = 8.314462618 / MOLAR_MASS_N2
R_VAPOR = 8.314462618 / MOLAR_MASS_H2O
EPSILON = MOLAR_MASS_H2O / MOLAR_MASS_N2
CP_DRY = 3.5 * R_DRY
CP_VAPOR = 1850.0  # J/kg/K
CP_LIQUID = 4180.0  # J/kg/K
REFERENCE_TEMPERATURE = 273.16  # K
REFERENCE_VAPOR_PRESSURE = 611.657  # Pa
REFERENCE_LATENT_HEAT = 2.501e6  # J/kg


def latent_heat(temperature):
    """Return the liquid-to-vapor latent heat in J/kg."""
    return REFERENCE_LATENT_HEAT + (CP_VAPOR - CP_LIQUID) * (
        temperature - REFERENCE_TEMPERATURE
    )


def saturation_vapor_pressure(temperature):
    """Return saturation pressure over liquid water in Pa.

    This integrates Clausius--Clapeyron with constant heat capacities and
    temperature-dependent latent heat (Ambaum 2020, doi:10.1002/qj.3899).
    """
    temperature = jnp.asarray(temperature)
    exponent = REFERENCE_LATENT_HEAT / (R_VAPOR * REFERENCE_TEMPERATURE)
    exponent -= latent_heat(temperature) / (R_VAPOR * temperature)
    return REFERENCE_VAPOR_PRESSURE * (
        REFERENCE_TEMPERATURE / temperature
    ) ** ((CP_LIQUID - CP_VAPOR) / R_VAPOR) * jnp.exp(exponent)


def water_vmr(pressure_bar, temperature, bottom_temperature, bottom_pressure_bar):
    """Return water mole fractions at layer centers and the ocean boundary.

    Nodes run from top to bottom. Ascending vapor retains the smallest
    saturation mole fraction encountered since leaving the ocean. The
    resulting upper atmosphere is dry above its cold trap. The caller must
    restrict the state to a liquid ocean below its boiling temperature.
    """
    node_temperature = jnp.append(temperature, bottom_temperature)
    node_pressure = jnp.append(pressure_bar, bottom_pressure_bar)
    saturated_vmr = saturation_vapor_pressure(node_temperature) / (
        1.0e5 * node_pressure
    )
    return jax.lax.associative_scan(jnp.minimum, saturated_vmr, reverse=True)


def pseudoadiabatic_gradient(temperature, water_mass_mixing_ratio):
    """Return saturated d ln(T)/d ln(P) for immediate condensate rainout.

    The mixing ratio is vapor mass divided by dry N2 mass, not total mass.
    This retains nondilute terms from Ding and Pierrehumbert (2016), Eq. 12,
    doi:10.3847/0004-637X/822/1/24. The pressure grid used by the example
    still prescribes total surface pressure rather than N2 column mass.
    """
    ratio = water_mass_mixing_ratio
    heat = latent_heat(temperature)
    numerator = (R_DRY + ratio * R_VAPOR) * (
        1.0 + heat * ratio / (R_DRY * temperature)
    )
    denominator = (
        CP_DRY
        + ratio * CP_VAPOR
        + heat**2 * ratio * (1.0 + ratio / EPSILON)
        / (R_VAPOR * temperature**2)
    )
    return numerator / denominator


def convective_gradient(
    pressure_bar, temperature, bottom_temperature, bottom_pressure_bar
):
    """Return the neutral gradient for each center-to-center/bottom connection.

    Evaluate parcel thermodynamics at geometric pressure and temperature
    midpoints. A parcel can condense only if vapor available from the deeper
    node reaches midpoint saturation. Otherwise use its ideal-gas mixture
    dry adiabat, including in the cold-trapped upper atmosphere.
    """
    node_temperature = jnp.append(temperature, bottom_temperature)
    node_pressure = jnp.append(pressure_bar, bottom_pressure_bar)
    midpoint_temperature = jnp.sqrt(node_temperature[:-1] * node_temperature[1:])
    midpoint_pressure = jnp.sqrt(node_pressure[:-1] * node_pressure[1:])
    saturated_vmr = saturation_vapor_pressure(midpoint_temperature) / (
        1.0e5 * midpoint_pressure
    )
    available_vmr = water_vmr(
        pressure_bar, temperature, bottom_temperature, bottom_pressure_bar
    )[1:]
    # Limit vapor before converting to a mass ratio: saturation pressure can
    # exceed total pressure in the dry upper atmosphere.
    midpoint_vmr = jnp.minimum(saturated_vmr, available_vmr)
    ratio = EPSILON * midpoint_vmr / (1.0 - midpoint_vmr)
    dry_gradient = (R_DRY + ratio * R_VAPOR) / (CP_DRY + ratio * CP_VAPOR)
    moist_gradient = pseudoadiabatic_gradient(midpoint_temperature, ratio)
    return jnp.where(saturated_vmr <= available_vmr, moist_gradient, dry_gradient)

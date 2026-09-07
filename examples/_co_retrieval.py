"""Shared physical model and probability densities for the CO tutorials.

Importing this module does not select JAX precision or import either sampler.
The six spectral parameters and inferred exponential noise prior retain the
original independent-noise tutorial model. The GP extension is a separate case.
"""

from dataclasses import dataclass

import numpy as np


PRIOR_SPECS = {
    "logg": {"distribution": "uniform", "low": 4.0, "high": 5.0},
    "RV": {"distribution": "uniform", "low": 35.0, "high": 45.0},
    "MMR": {"distribution": "uniform", "low": 0.0, "high": 0.015},
    "T0": {"distribution": "uniform", "low": 1000.0, "high": 1500.0},
    "alpha": {"distribution": "uniform", "low": 0.05, "high": 0.2},
    "vsini": {"distribution": "uniform", "low": 5.0, "high": 15.0},
    "sigmain": {"distribution": "exponential", "rate": 1.0e-3},
}
UNITS = {
    "nu": "cm-1",
    "flux": "erg/s/cm2/cm-1",
    "pressure": "bar",
    "temperature": "K",
    "gravity": "cm/s2",
    "RV": "km/s",
    "vsini": "km/s",
    "MMR": "mass fraction",
    "sigmain": "erg/s/cm2/cm-1",
}


@dataclass(frozen=True)
class CaseConfig:
    wavelength_min: float = 22920.0
    wavelength_max: float = 23000.0
    number_of_wavenumbers: int = 3500
    number_of_layers: int = 100
    pressure_top: float = 1.0e-5
    pressure_bottom: float = 10.0
    temperature_min: float = 500.0
    temperature_max: float = 1500.0
    observation_stride: int = 5
    observation_trim: int = 50
    instrument_resolution: float = 70000.0
    maximum_vsini: float = 100.0
    maximum_instrument_velocity: float = 1000.0
    hydrogen_volume_mixing_ratio: float = 0.855
    mean_molecular_weight: float = 2.33
    noise_sigma: float = 500.0
    premodit_diffmode: int = 0
    broadening_resolution: float = 1.0


def make_forward(context):
    """Return the original fspec(T0, alpha, MMR, g, RV, vsini) operator."""
    art, opa, opacia = (context[name] for name in ("art", "opa", "opacia"))

    def fspec(T0, alpha, MMR, g, RV, vsini):
        temperature = art.powerlaw_temperature(T0, alpha)
        cross_sections = opa.xsmatrix(temperature, art.pressure)
        dtau = art.opacity_profile_xs(
            cross_sections, art.constant_mmr_profile(MMR), context["molmass"], g
        )
        logacia = opacia.logacia_matrix(temperature)
        dtau += art.opacity_profile_cia(
            logacia, temperature, context["vmrH2"], context["vmrH2"], context["mmw"], g
        )
        flux = art.run(dtau, temperature)
        rotated = context["sop_rot"].rigid_rotation(flux, vsini, 0.0, 0.0)
        blurred = context["sop_inst"].ipgauss(rotated, context["beta_inst"])
        return context["sop_inst"].sampling(blurred, RV, context["nu_obs"])

    return fspec


def predict(fspec, parameters):
    return fspec(
        parameters["T0"],
        parameters["alpha"],
        parameters["MMR"],
        10.0 ** parameters["logg"],
        parameters["RV"],
        parameters["vsini"],
    )


def physical_log_likelihood(fspec, observation, parameters):
    """Normalized independent Gaussian likelihood in physical coordinates."""
    import jax.numpy as jnp
    from jax.scipy.stats import norm

    sigma = parameters["sigmain"]
    value = norm.logpdf(observation, predict(fspec, parameters), sigma).sum()
    return jnp.where(sigma > 0, value, -jnp.inf)


def sample_priors(prior_specs=PRIOR_SPECS):
    """Shared NumPyro prior sites, also usable by the separate GP illustration."""
    import numpyro
    import numpyro.distributions as dist

    return {
        name: numpyro.sample(
            name,
            dist.Uniform(spec["low"], spec["high"])
            if spec["distribution"] == "uniform"
            else dist.Exponential(spec["rate"]),
        )
        for name, spec in prior_specs.items()
    }


def make_numpyro_model(fspec, prior_specs=PRIOR_SPECS):
    import numpyro
    import numpyro.distributions as dist

    def model(spectrum=None):
        parameters = sample_priors(prior_specs)
        return numpyro.sample(
            "spectrum",
            dist.Normal(predict(fspec, parameters), parameters["sigmain"]),
            obs=spectrum,
        )

    return model


def unit_to_physical(unit, prior_specs=PRIOR_SPECS):
    """Inverse CDFs map a unit cube onto exactly the normalized tutorial prior."""
    import jax.numpy as jnp

    return {
        name: spec["low"] + unit[index] * (spec["high"] - spec["low"])
        if spec["distribution"] == "uniform"
        else -jnp.log1p(-unit[index]) / spec["rate"]
        for index, (name, spec) in enumerate(prior_specs.items())
    }


def physical_to_unit(parameters, prior_specs=PRIOR_SPECS):
    import jax.numpy as jnp

    return jnp.stack(
        [
            (parameters[name] - spec["low"]) / (spec["high"] - spec["low"])
            if spec["distribution"] == "uniform"
            else -jnp.expm1(-spec["rate"] * parameters[name])
            for name, spec in prior_specs.items()
        ]
    )


def normalized_log_prior(parameters, prior_specs=PRIOR_SPECS):
    import jax.numpy as jnp

    result = jnp.asarray(0.0)
    for name, spec in prior_specs.items():
        value = parameters[name]
        if spec["distribution"] == "uniform":
            density = -jnp.log(spec["high"] - spec["low"])
            valid = (value >= spec["low"]) & (value <= spec["high"])
        else:
            density = jnp.log(spec["rate"]) - spec["rate"] * value
            valid = value >= 0
        result += jnp.where(valid, density, -jnp.inf)
    return result


def unit_log_jacobian(unit, prior_specs=PRIOR_SPECS):
    import jax.numpy as jnp

    return sum(
        jnp.log(spec["high"] - spec["low"])
        if spec["distribution"] == "uniform"
        else -jnp.log(spec["rate"]) - jnp.log1p(-unit[index])
        for index, spec in enumerate(prior_specs.values())
    )


def mock_truth(noise_sigma=500.0):
    from exojax.utils.astrofunc import gravity_jupiter

    return {
        "logg": float(np.log10(gravity_jupiter(1.0, 10.0))),
        "RV": 40.0,
        "MMR": 0.01,
        "T0": 1200.0,
        "alpha": 0.1,
        "vsini": 10.0,
        "sigmain": noise_sigma,
    }

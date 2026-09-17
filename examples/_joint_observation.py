"""Two fixed instrument responses of the shared CO emission model.

Bins average flux density in ascending wavenumber (cm-1), after rotation,
instrumental broadening, and the shared radial-velocity shift. Instrument A
anchors the additive flux offset at zero; each instrument scales its own known
error bars. This module does not select JAX precision or import a sampler.
"""

import numpy as np

from _co_retrieval import PRIOR_SPECS, make_rotated_flux, sample_priors
from _co_retrieval import mock_truth as _co_truth


JOINT_PRIORS = {
    name: dict(spec) for name, spec in PRIOR_SPECS.items() if name != "sigmain"
}
JOINT_PRIORS.update(
    offset_b={"distribution": "uniform", "low": -1000.0, "high": 1000.0},
    scale_a={"distribution": "uniform", "low": 0.5, "high": 2.0},
    scale_b={"distribution": "uniform", "low": 0.5, "high": 2.0},
)
INSTRUMENTS = {
    "a": {
        "resolution": 70000.0,
        "num_bins": 64,
        "sigma": 500.0,
        "kernel_velocity": 100.0,
    },
    "b": {
        "resolution": 20000.0,
        "num_bins": 32,
        "sigma": 800.0,
        "kernel_velocity": 100.0,
    },
}


def mock_truth():
    parameters = _co_truth()
    parameters.pop("sigmain")
    parameters.update(offset_b=250.0, scale_a=1.0, scale_b=1.0)
    return parameters


def _response_setup(context, instruments):
    """Construct fixed LSFs and retain their complete finite-kernel coverage."""
    from exojax.postproc.specop import SopInstProfile
    from exojax.utils.instfunc import resolution_to_gaussian_std

    if set(instruments) != {"a", "b"}:
        raise ValueError("Exactly instruments a and b are required.")
    nu = np.asarray(context["nu_grid"], dtype=float)
    if (
        nu.ndim != 1
        or nu.size < 3
        or not np.all(np.isfinite(nu))
        or np.any(nu <= 0)
        or np.any(np.diff(nu) <= 0)
    ):
        raise ValueError("The wavenumber grid must be positive and increasing.")
    log_spacing = np.diff(np.log(nu))
    if not np.allclose(log_spacing, log_spacing[0], rtol=1e-8, atol=0.0):
        raise ValueError("Instrument convolution requires an evenly log-spaced grid.")
    rotation = context["sop_rot"]
    if not np.array_equal(np.asarray(rotation.nu_grid), nu):
        raise ValueError("Rotation and instrument wavenumber grids must agree.")
    velocity = np.asarray(rotation.vrarray)
    maximum_rotation = JOINT_PRIORS["vsini"]["high"]
    if velocity.min() > -maximum_rotation or velocity.max() < maximum_rotation:
        raise ValueError("The rotation kernel does not cover the shared vsini prior.")
    center = len(velocity) // 2
    active_rotation = np.flatnonzero(np.abs(velocity) <= maximum_rotation)
    rotation_radius = int(np.max(np.abs(active_rotation - center)))
    responses = {}
    padding = rotation_radius
    for name, specification in instruments.items():
        if set(specification) != {"resolution", "num_bins", "sigma", "kernel_velocity"}:
            raise ValueError(
                "Every instrument needs resolution, bins, sigma, and kernel velocity."
            )
        count = specification["num_bins"]
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count <= 0
        ):
            raise ValueError("Instrument num_bins must be a positive integer.")
        values = [
            specification[key] for key in ("resolution", "sigma", "kernel_velocity")
        ]
        if not np.all(np.isfinite(values)) or np.any(np.asarray(values) <= 0):
            raise ValueError(
                "Instrument resolution, sigma, and kernel velocity must be positive."
            )
        beta = resolution_to_gaussian_std(specification["resolution"])
        if specification["kernel_velocity"] < 5.0 * beta:
            raise ValueError(
                "The fixed Gaussian kernel must cover at least five standard deviations."
            )
        profile = SopInstProfile(nu, vrmax=specification["kernel_velocity"])
        responses[name] = (profile, beta)
        padding = max(padding, len(profile.vrarray) // 2 + rotation_radius)
    if instruments["a"]["num_bins"] % instruments["b"]["num_bins"]:
        raise ValueError("Instrument B bins must exactly coarsen instrument A bins.")
    if 2 * padding >= len(nu) - 1:
        raise ValueError("The model grid has insufficient full-kernel coverage.")
    return nu, responses, (nu[padding], nu[-padding - 1])


def _check_sample_coverage(sample_grid, usable):
    from exojax.utils.constants import c

    grid = np.asarray(sample_grid, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.all(np.isfinite(grid))
        or np.any(grid <= 0)
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError(
            "The observation sampling grid must be positive and increasing."
        )
    shifted_lower = grid[0] * (1.0 + JOINT_PRIORS["RV"]["low"] / c)
    shifted_upper = grid[-1] * (1.0 + JOINT_PRIORS["RV"]["high"] / c)
    if shifted_lower < usable[0] or shifted_upper > usable[1]:
        raise ValueError("Observation sampling lacks full LSF/rotation/RV coverage.")


def make_geometry(context, instruments=INSTRUMENTS):
    """Build fixed covered sampling knots, contiguous bins, and known errors."""
    from exojax.utils.constants import c

    nu, _, usable = _response_setup(context, instruments)
    grid = nu[
        (nu * (1.0 + JOINT_PRIORS["RV"]["low"] / c) >= usable[0])
        & (nu * (1.0 + JOINT_PRIORS["RV"]["high"] / c) <= usable[1])
    ]
    _check_sample_coverage(grid, usable)
    edges_a = np.linspace(grid[0], grid[-1], instruments["a"]["num_bins"] + 1)
    group_size = instruments["a"]["num_bins"] // instruments["b"]["num_bins"]
    edges_b = edges_a[::group_size]
    return {
        "sample_grid": grid,
        "bins_a": np.column_stack((edges_a[:-1], edges_a[1:])),
        "bins_b": np.column_stack((edges_b[:-1], edges_b[1:])),
        **{
            f"error_{name}": np.full(spec["num_bins"], spec["sigma"])
            for name, spec in instruments.items()
        },
    }


def make_forward(context, arrays, instruments=INSTRUMENTS):
    """Build the two observation paths; evaluate the shared atmosphere once."""
    from exojax.postproc.binning import (
        apply_bin_operator,
        band_mean_bin_operator,
        piecewise_linear_bin_operator,
    )

    _, responses, usable = _response_setup(context, instruments)
    sample_grid = np.asarray(arrays["sample_grid"])
    _check_sample_coverage(sample_grid, usable)
    expected = make_geometry(context, instruments)
    for name in ("sample_grid", "bins_a", "bins_b", "error_a", "error_b"):
        if not np.array_equal(np.asarray(arrays[name]), expected[name]):
            raise ValueError(
                f"Saved {name} does not match the fixed observation geometry."
            )
    base_bins = piecewise_linear_bin_operator(sample_grid, arrays["bins_a"])
    coarse_bins = band_mean_bin_operator(arrays["bins_a"], arrays["bins_b"])
    rotated_flux = make_rotated_flux(context)

    def forward(parameters):
        spectrum = rotated_flux(
            parameters["T0"],
            parameters["alpha"],
            parameters["MMR"],
            10.0 ** parameters["logg"],
            parameters["vsini"],
        )
        predictions = {}
        for name, (profile, beta) in responses.items():
            blurred = profile.ipgauss(spectrum, beta)
            shifted = profile.sampling(blurred, parameters["RV"], sample_grid)
            values = apply_bin_operator(base_bins, shifted)
            if name == "b":
                values = (
                    apply_bin_operator(coarse_bins, values) + parameters["offset_b"]
                )
            predictions[name] = values
        return predictions

    return forward


def error_scales(arrays, parameters):
    import jax.numpy as jnp

    return {
        name: jnp.asarray(arrays[f"error_{name}"]) * parameters[f"scale_{name}"]
        for name in ("a", "b")
    }


def log_likelihood(forward, arrays, parameters):
    """Normalized Gaussian likelihood with separately scaled saved errors."""
    import jax.numpy as jnp
    from jax.scipy.stats import norm

    prediction = forward(parameters)
    errors = error_scales(arrays, parameters)
    value = sum(
        norm.logpdf(arrays[f"observed_{name}"], prediction[name], errors[name]).sum()
        for name in ("a", "b")
    )
    valid = (parameters["scale_a"] > 0) & (parameters["scale_b"] > 0)
    return jnp.where(valid, value, -jnp.inf)


def make_numpyro_model(forward, arrays, prior_specs=JOINT_PRIORS):
    """Use one concatenated observation site with the shared sampler contract."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model(spectrum=None):
        parameters = sample_priors(prior_specs)
        prediction = forward(parameters)
        errors = error_scales(arrays, parameters)
        return numpyro.sample(
            "spectrum",
            dist.Normal(
                jnp.concatenate((prediction["a"], prediction["b"])),
                jnp.concatenate((errors["a"], errors["b"])),
            ),
            obs=spectrum,
        )

    return model

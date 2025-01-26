"""Kernels used in Gaussian process."""

from jax import random
from jax import jit
import jax.numpy as jnp
import tqdm
import numpyro.distributions as dist
import numpyro


def sampling_prediction(
    x,
    data,
    scale_sampling,
    amplitude_sampling,
    err_sampling,
    prediction_spectrum,
    key,
):
    """computes GP predictions
    
    Args:
        x (array): variable vector x [N]
        data (array): data vector [N]
        scale_sampling (array): sampling of scale [num_samples]
        amplitude_sampling (array): sampling of amplitude [num_samples]
        err_sampling (array): sampling of error array [num_samples, N]
        prediction_sampling (array): sampling of prediction [num_samples, N]
        key: random key, given by numpyro PRNGKey, e.g. key = random.PRNGKey(20)

    Returns:
        2D array: predictions with a GP model [num_samples, N]
    
    Note:
        The reason why we do not use jax.vmap for average_covariance_gpmodel is that it requires extremely large device memory.
    
    """
    num_samples = len(scale_sampling)
    gp_predictions = []
    for i in tqdm.tqdm(range(0, num_samples)):
        ave, cov = average_covariance_gpmodel(
            x,
            data,
            prediction_spectrum[i],
            scale_sampling[i],
            amplitude_sampling[i],
            err_sampling[i],
        )
        mn = dist.MultivariateNormal(loc=ave, covariance_matrix=cov)
        key, _ = random.split(key)
        mk = numpyro.sample("gp_prediction", mn, rng_key=key)
        gp_predictions.append(mk)
    return jnp.array(gp_predictions)


def gpkernel_RBF(x, scale, amplitude, err):
    """RBF kernel with diagnoal error.

    Args:
        x (array): variable vector (N)
        scale (float): scale parameter
        amplitude (float) : amplitude (scalar)
        err (1D array): diagnonal error vector (N)

    Returns:
        kernel
    """

    diff = x - jnp.array([x]).T
    return amplitude * jnp.exp(-((diff) ** 2) / 2 / (scale**2)) + jnp.diag(err**2)


@jit
def average_covariance_gpmodel(x, data, model, scale, amplitude, err):
    """computes average and covariance of GP model
    
    Args:
        x (array): variable vector (N)
        data (array): data vector (N)
        scale (float): scale parameter
        amplitude (float) : amplitude (scalar)
        err (1D array): diagnonal error vector (N)

    Returns:
        _type_: average, covariance
    """
    cov = gpkernel_RBF(x, scale, amplitude, err)
    covx = gpkernel_RBF(x, scale, amplitude, jnp.zeros_like(x))
    A = jnp.linalg.solve(cov, data - model)
    IKw = jnp.linalg.inv(cov)
    return model + covx @ A, cov - covx @ IKw @ covx.T


def gpkernel_RBF_cross(x, y, scale, amplitude):
    """cross RBF kernel (no diagonal term)

    Args:
        x (array): first variable vector (Nx)
        y (array): second variable vector (Ny)
        scale (float): scale parameter
        amplitude (float) : amplitude (scalar)

    Returns:
        kernel
    """

    diff = x - jnp.array([y]).T
    return amplitude * jnp.exp(-((diff) ** 2) / 2 / (scale**2))


@jit
def average_covariance_gpmodel_cross(x, y, data, model, scale, amplitude, sigma):
    err = sigma * jnp.ones_like(x)
    cov = gpkernel_RBF(x, scale, amplitude, err)
    covx = gpkernel_RBF_cross(x, y, scale, amplitude)
    covxx = gpkernel_RBF(y, scale, amplitude, err)
    A = jnp.linalg.solve(cov, data - model)
    IKw = jnp.linalg.inv(cov)
    return model + covx @ A, covxx - covx @ IKw @ covx.T

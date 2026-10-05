"""Local linear-Gaussian diagnostics for differentiable spectral models."""

from jax.core import Tracer
import jax.numpy as jnp
import numpy as np


def linear_gaussian_diagnostics(jacobian, noise_std, prior_std):
    """Forecast parameter constraints from an observation-space Jacobian.

    The observation covariance is fixed and diagonal, and the prior is an
    independent Gaussian in the same parameter coordinates as the Jacobian.
    Compute the Jacobian of the complete forward model, including instrumental
    response and binning, at the reference state before calling this function.
    This is a local forecast, not a posterior sampler or a Gaussian fit to an
    existing retrieval.

    Args:
        jacobian (array): Finite real derivatives with shape ``(nobs, npar)``.
        noise_std (array): Positive finite observational standard deviations,
            shape ``(nobs,)``, in the units of the forward model's output.
        prior_std (array): Positive finite Gaussian prior standard deviations,
            shape ``(npar,)``. For log10 parameters these are in dex; a uniform
            prior's width is not a Gaussian standard deviation.

    Returns:
        dict: JAX arrays with the following keys:

        * ``fisher``: Data-only Fisher matrix, shape ``(npar, npar)``.
        * ``posterior_cov``: Local posterior covariance in input parameter units.
        * ``posterior_std``: Square root of its diagonal, shape ``(npar,)``.
        * ``averaging_kernel``: Posterior covariance times the data Fisher
          matrix. Its diagonal measures data influence; the matrix need not
          be symmetric in the input parameter coordinates.
        * ``degrees_of_freedom``: Trace of the averaging kernel.
        * ``information_bits``: Gaussian prior-to-posterior entropy reduction.
        * ``singular_values``: Descending singular values of the noise-whitened,
          prior-scaled Jacobian, padded with zeros if ``nobs < npar``.
        * ``parameter_modes``: Orthonormal right singular vectors as rows,
          shape ``(npar, npar)``, in coordinates ``delta_x / prior_std``.
          Small singular values identify weakly constrained combinations.
          Signs and bases within degenerate subspaces are not unique.

    Raises:
        ValueError: Inputs have invalid shapes, complex values, nonfinite
            concrete values, or nonpositive concrete standard deviations.

    Notes:
        Uses an SVD to retain prior uncertainty in unconstrained directions
        without inverting a potentially singular Fisher matrix. No observation
        covariance matrix is materialized. JIT compilation is supported; value
        checks are skipped for traced inputs, which must satisfy the same
        finite/positive preconditions. Prefer JAX 64-bit precision for forecasts
        with very different sensitivity scales. Differentiation through these
        diagnostics is not guaranteed: full-matrix SVD and degenerate singular
        modes impose additional restrictions.

        These diagnostics assume local linearity and Gaussian distributions.
        They do not capture multimodal posteriors or upper limits. Parameters
        controlling the noise covariance require additional Fisher terms and
        are outside this function's scope.

    References:
        Batalha & Wogan (2026), https://arxiv.org/abs/2609.22634, equations
        (7)--(9). Information is returned in bits, including division by ln(2).
    """
    jacobian = jnp.asarray(jacobian)
    noise_std = jnp.asarray(noise_std)
    prior_std = jnp.asarray(prior_std)
    if jacobian.ndim != 2 or min(jacobian.shape) == 0:
        raise ValueError("jacobian must have nonempty shape (nobs, npar).")
    nobs, npar = jacobian.shape
    if noise_std.shape != (nobs,):
        raise ValueError("noise_std must have shape (nobs,).")
    if prior_std.shape != (npar,):
        raise ValueError("prior_std must have shape (npar,).")
    for name, value in (
        ("jacobian", jacobian), ("noise_std", noise_std), ("prior_std", prior_std)
    ):
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise ValueError(f"{name} must be real.")
        if not isinstance(value, Tracer):
            concrete = np.asarray(value)
            if not np.all(np.isfinite(concrete)):
                raise ValueError(f"{name} must be finite.")
            if name != "jacobian" and np.any(concrete <= 0):
                raise ValueError(f"{name} must be strictly positive.")

    whitened = jacobian / noise_std[:, None]
    scaled = whitened * prior_std[None, :]
    # Full parameter modes for a wide Jacobian, without a large nobs-by-nobs U.
    _, singular_values, modes = jnp.linalg.svd(scaled, full_matrices=nobs < npar)
    if nobs < npar:
        singular_values = jnp.pad(singular_values, (0, npar - nobs))

    # log(1 + s**2) without overflowing s**2 or losing weak-mode information.
    log_precision = jnp.logaddexp(0.0, 2.0 * jnp.log(singular_values))
    mode_variance = jnp.exp(-log_precision)
    mode_gain = -jnp.expm1(-log_precision)
    scaled_cov = (modes.T * mode_variance) @ modes
    scaled_kernel = (modes.T * mode_gain) @ modes
    posterior_cov = prior_std[:, None] * scaled_cov * prior_std[None, :]
    averaging_kernel = prior_std[:, None] * scaled_kernel / prior_std[None, :]
    return {
        "fisher": whitened.T @ whitened,
        "posterior_cov": posterior_cov,
        "posterior_std": jnp.sqrt(jnp.diag(posterior_cov)),
        "averaging_kernel": averaging_kernel,
        "degrees_of_freedom": jnp.sum(mode_gain),
        "information_bits": jnp.sum(log_precision) / (2.0 * jnp.log(2.0)),
        "singular_values": singular_values,
        "parameter_modes": modes,
    }

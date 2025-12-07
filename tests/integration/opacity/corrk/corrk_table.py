"""This module tests the correlated k distribution implementation in ExoJAX.

Demonstrates CKD method using the implemented core functions:
- compute_g_ordinates: sorts cross-sections and computes g-ordinates
- safe_log_k: computes safe logarithm avoiding log(0)
- gauss_legendre_grid: generates quadrature points and weights
- interpolate_log_k_to_g_grid: interpolates log(k) onto g-grid
"""

import numpy as np
import jax.numpy as jnp
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.opacity.opacalc import OpaPremodit
from exojax.opacity.ckd.core import (
    compute_g_ordinates, 
    gauss_legendre_grid,
    safe_log_k,
    interpolate_log_k_to_g_grid
)
from jax import config

config.update("jax_enable_x64", True)  # use double precision

# Setup wavenumber grid and molecular database
nus, wav, res = mock_wavenumber_grid(lambda0=22930.0, lambda1=22940.0, Nx=20000)
mdb = mock_mdbExomol("H2O")

# Initialize opacity calculator
opa = OpaPremodit(mdb, nus, auto_trange=[500.0, 1500.0])

# Set temperature and pressure conditions
T = 1000.0
P = 1.0e-2

# Compute cross-section vector
print("Computing cross-section vector...")
xsv = opa.xsvector(T, P)

# Generate CKD g-ordinates using core function
print("Computing g-ordinates...")
idx, k_g, g = compute_g_ordinates(xsv)

# Compute safe logarithm using core function
print("Computing safe logarithm...")
log_k_g = safe_log_k(k_g)

# Generate Gauss-Legendre quadrature grid using core function  
Ng = 32
print(f"Generating Gauss-Legendre grid with {Ng} points...")
ggrid, weights = gauss_legendre_grid(Ng)

# Interpolate log(k) values onto g-grid using core function
print("Interpolating onto g-grid...")
log_kggrid = interpolate_log_k_to_g_grid(g, log_k_g, ggrid)

# Validation: Compare direct integration vs CKD quadrature
print("\nValidating CKD approximation...")

# Direct spectral integration
dnus_ = nus[-1] - nus[-2]  # wavenumber spacing
L = 1.e22  # optical depth parameter
direct_sum = jnp.sum(jnp.exp(-xsv * L) * dnus_)

# CKD quadrature integration
dnus_whole = nus[-1] - nus[0]  # total spectral range
ckd_sum = jnp.sum(weights * jnp.exp(-jnp.exp(log_kggrid) * L)) * dnus_whole

# Results
print(f"Direct integration result: {direct_sum:.6e}")
print(f"CKD quadrature result:    {ckd_sum:.6e}")
relative_error = abs(direct_sum - ckd_sum) / direct_sum
print(f"Relative error:           {relative_error:.6e}")

# Validation check
tolerance = 0.01
assert relative_error < tolerance, f"CKD error {relative_error} exceeds tolerance {tolerance}"

print(f"\n✓ CKD validation successful! Relative error < {tolerance}")
print(f"✓ Grid resolution: {res:.1f}")
print(f"✓ Number of spectral points: {len(nus)}")
print(f"✓ Number of g-points: {Ng}")

print("\nCKD table generation completed successfully!")

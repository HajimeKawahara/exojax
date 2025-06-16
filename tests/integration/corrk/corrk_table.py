"""This module tests the correlated k distribution implementation in ExoJAX.

Demonstrates CKD method using the new compute_g_ordinates and gauss_legendre_grid functions
from the ExoJAX opacity.ckd.core module.
"""

import numpy as np
import jax.numpy as jnp
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_wavenumber_grid
from exojax.opacity.opacalc import OpaPremodit
from exojax.opacity.ckd.core import compute_g_ordinates, gauss_legendre_grid
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

# Generate CKD g-ordinates using new function
print("Computing g-ordinates...")
idx, k_g, g = compute_g_ordinates(xsv)
log_k_g = jnp.log(jnp.maximum(k_g, 1e-30))  # Avoid log(0)

# Generate Gauss-Legendre quadrature grid using new function  
Ng = 32
print(f"Generating Gauss-Legendre grid with {Ng} points...")
ggrid, weights = gauss_legendre_grid(Ng)

# Interpolate log(k) values onto g-grid
print("Interpolating onto g-grid...")
log_kggrid = jnp.interp(ggrid, g, log_k_g)

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

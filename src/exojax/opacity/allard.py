"""Alkali resonance doublets from the CDS Allard density expansions."""

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from exojax.atm.idealgas import number_density
from exojax.database.core.broadening import doppler_sigma
from exojax.opacity.base import OpaCalc
from exojax.opacity.lpf.lpf import voigt


@jit
def density_expansion(coefficients, density_ratio, volume, normalization):
    """Evaluate a CDS wing in cm2, retaining signed expansion coefficients.

    The first column multiplies q, not 1. This follows ``lect_sig.f`` and
    the CDS examples, rather than the off-by-one powers in its Python port.
    Nonpositive results are left intact for the caller to handle.
    """
    # Horner evaluation avoids a separate array of powers and works at q=0.
    value = jnp.zeros(coefficients.shape[:-1])
    for index in range(coefficients.shape[-1] - 1, -1, -1):
        value = (value + coefficients[..., index]) * density_ratio
    return normalization * jnp.exp(-volume * density_ratio) * value


def _wing(profile, offsets, density):
    """Interpolate a nonnegative wing; never extend an endpoint plateau."""
    q = density / profile.density
    values = jnp.maximum(density_expansion(
        jnp.asarray(profile.coefficients), q, profile.volume,
        profile.normalization,
    ), 0.0)
    grid = jnp.asarray(profile.offsets)
    if profile.red_offsets is not None:
        # Na D1's separately tabulated far red wing scales linearly in n.
        # Keep the near-wing value at the common endpoint, without summing
        # two descriptions of the same absorption.
        mask = profile.red_offsets < profile.offsets[0]
        grid = jnp.concatenate((jnp.asarray(profile.red_offsets[mask]), grid))
        values = jnp.concatenate((
            jnp.asarray(profile.red_cross_sections[mask]) * q, values,
        ))
    return jnp.interp(offsets, grid, values, left=0.0, right=0.0)


class OpaAlkaliTable(OpaCalc):
    """Allard wings joined to a thermal and natural broadened impact core.

    This is an explicit numerical hybrid: the supplied CDS code does not
    specify a unique core/wing join. A cubic smoothstep connects a Voigt
    core to the tabulated wing over ``core_transition`` in absolute
    detuning (cm-1). The default is (20, 30), not a fitted physical
    parameter or an exact unified-core calculation. Core asymmetry is
    approximated by the tabulated impact shift. Doppler and natural
    broadening are included in the core only.

    Cross sections are cm2 per ground-state neutral atom for D1+D2 only.
    The table already contains oscillator strengths. No partition-function,
    stimulated-emission, ionization, or abundance correction is applied.
    Do not add these doublets again from an atomic line list, or add two
    complete single-perturber profiles to model a gas mixture.

    Temperature interpolation is linear in cross section, after evaluating
    both bounding tables at the requested perturber density. Negative wing
    values from the truncated expansion are set to zero before linear
    wavenumber interpolation. This fixed-grid convention differs from the
    reference reader's omission of nonpositive points. Outside each table's
    spectral support the wing is zero; no wing extrapolation is attempted.

    T or density outside the supported range produces NaNs, including under
    jit. The density ceiling is 1e21 cm-3 (the Na table's stated limit).
    Interpolation and clipping give piecewise differentiable T/P dependence.
    """

    def __init__(
        self, nu_grid, data_path, *, model="allard2019_na_h2",
        vmr_perturber=1.0, core_transition=(20.0, 30.0),
    ):
        """Load local CDS data once, before entering JAX transformations.

        Args:
            nu_grid: Increasing vacuum wavenumbers (cm-1).
            data_path: Local CDS archive or extracted data directory.
            model: ``allard2019_na_h2`` for Na broadened by H2.
            vmr_perturber: Perturber number fraction of the total gas. P in
                xsvector/xsmatrix is total pressure in bar. Other collision
                partners are not included by setting a fraction below one.
            core_transition: Inner/outer absolute detuning in cm-1 for the
                smooth core/wing join. Both must be positive and increasing.
        """
        from exojax.database.alkali import load_allard2019

        if model != "allard2019_na_h2":
            raise ValueError("model must be 'allard2019_na_h2'.")
        self.profiles = load_allard2019(data_path)
        self.species, self.perturber, self.mass = "Na", "H2", 22.98976928

        grid = np.asarray(nu_grid, dtype=float)
        if (grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid))
                or np.any(grid <= 0) or np.any(np.diff(grid) <= 0)):
            raise ValueError("nu_grid must contain increasing positive wavenumbers.")
        if not np.isfinite(vmr_perturber) or not 0 <= vmr_perturber <= 1:
            raise ValueError("vmr_perturber must be between zero and one.")
        transition = np.asarray(core_transition, dtype=float)
        if (transition.shape != (2,) or not np.all(np.isfinite(transition))
                or not 0 < transition[0] < transition[1]):
            raise ValueError("core_transition must be two increasing positive detunings.")
        super().__init__(grid)
        self.model = model
        self.method = "allard"
        self.vmr_perturber = float(vmr_perturber)
        self.core_transition = tuple(transition)
        self.temperatures = np.array([p.temperature for p in self.profiles["D1"]])
        self.density_max = 1.0e21
        self.ready = True

    def _at_temperature(self, line, profile, T, density):
        center = 1.0e8 / profile.wavelength
        offsets = jnp.asarray(self.nu_grid) - center
        q = density / profile.density
        # A/(4 pi c) = 2 (pi r_e f) (g_lower/g_upper) nu_0**2.
        # D1: g_lower/g_upper=1; D2: 1/2. Ground-state lifetime is infinite.
        natural = (2.0 if line == "D1" else 1.0) * profile.normalization * center**2
        core = profile.normalization * voigt(
            offsets - profile.shift * q, doppler_sigma(center, T, self.mass),
            profile.width * q + natural,
        )
        wing = _wing(profile, offsets, density)
        inner, outer = self.core_transition
        fraction = jnp.clip((jnp.abs(offsets) - inner) / (outer - inner), 0.0, 1.0)
        weight = fraction**2 * (3.0 - 2.0 * fraction)
        return (1.0 - weight) * core + weight * wing

    def xsvector(self, T, P):
        """D1+D2 cross sections at scalar T (K), total P (bar).

        Nonfinite inputs, negative pressure, out-of-range temperature, or
        perturber densities above ``density_max`` return an all-NaN vector.
        """
        T, P = jnp.asarray(T), jnp.asarray(P)
        if T.ndim != 0 or P.ndim != 0:
            raise ValueError("xsvector requires scalar T and P; use xsmatrix for layers.")
        temperatures = jnp.asarray(self.temperatures)
        # Keep inactive branches finite so invalid inputs cannot overflow a
        # density polynomial or contaminate gradients through jnp.where.
        safe_T = jnp.clip(T, temperatures[0], temperatures[-1])
        density = number_density(P, safe_T) * self.vmr_perturber
        valid = (jnp.isfinite(T) & jnp.isfinite(P) & (P >= 0)
                 & (T >= temperatures[0]) & (T <= temperatures[-1])
                 & (density <= self.density_max))
        safe_density = jnp.clip(density, 0.0, self.density_max)
        lower = jnp.clip(jnp.searchsorted(temperatures, safe_T, side="right") - 1,
                         0, len(self.temperatures) - 2)
        fraction = ((safe_T - temperatures[lower])
                    / (temperatures[lower + 1] - temperatures[lower]))
        result = jnp.zeros_like(jnp.asarray(self.nu_grid))
        for line in ("D1", "D2"):
            spectra = jnp.stack([
                self._at_temperature(line, profile, safe_T, safe_density)
                for profile in self.profiles[line]
            ])
            result = result + (1.0 - fraction) * spectra[lower] + fraction * spectra[lower + 1]
        return jnp.where(valid, result, jnp.nan)

    def xsmatrix(self, Tarr, Parr):
        """Layer cross sections of shape (Nlayer, Nnu), using paired T/P."""
        Tarr, Parr = jnp.asarray(Tarr), jnp.asarray(Parr)
        if Tarr.ndim != 1 or Tarr.shape != Parr.shape:
            raise ValueError("Tarr and Parr must be one-dimensional arrays of equal shape.")
        return vmap(self.xsvector)(Tarr, Parr)

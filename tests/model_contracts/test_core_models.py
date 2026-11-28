"""Bounded provider comparisons; mathematical references come from native records.

Synthetic inputs require no spectral database or network. These points test
quantity interfaces and selected derivatives, not every numerical algorithm or
an atmospheric applicability claim. Tests never promote an MCEG assessment.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

pytest.importorskip("mceg.schema_v06")
sp = pytest.importorskip("sympy")
from mceg import load_contract, to_sympy

from exojax.atm.atmprof import (
    hydrostatic_radius_profile, hydrostatic_radius_profile_ideal_gas,
)
from exojax.database.core.broadening import doppler_sigma
from exojax.database.core.line_strength import line_strength
from exojax.opacity.lpf.lpf import voigtone, xsvector
from exojax.postproc.response import ipgauss, sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation, rotkernel
from exojax.rt.chord import (
    chord_geometric_matrix, chord_geometric_matrix_lower, chord_optical_depth,
)
from exojax.rt.layeropacity import (
    layer_optical_depth, layer_optical_depth_from_cross_section,
    layer_optical_depth_from_extinction, layer_optical_depth_from_log_cia,
)
from exojax.rt.planck import piB
from exojax.rt.rtransfer import (
    rtrun_emis_pureabs_ibased_intensity,
    rtrun_emis_pureabs_ibased_intensity_surface,
    rtrun_emis_pureabs_ibased_flux_from_intensity,
    rtrun_trans_pureabs_simpson, rtrun_trans_pureabs_trapezoid,
)
from exojax.rt.trans import ArtTransPure
from exojax.utils.constants import G, bar_cgs, c, ccgs, hcperk, kB, m_u, opacity_factor

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = {name: load_contract(ROOT / "mceg/contracts" / f"{name}.yaml") for name in (
    "lte_line_opacity", "atmospheric_columns", "thermal_emission", "transmission", "spectral_observation",
)}
EXPRESSIONS = {name: {e.id.rsplit(".", 1)[-1]: to_sympy(e.sympy)
                      for e in contract.equations} for name, contract in CONTRACTS.items()}


@pytest.fixture(autouse=True)
def precision(isolated_test_environment):
    yield


def rhs(family, equation):
    return EXPRESSIONS[family][equation].rhs


def symbols(expression):
    return {s.name: s for s in expression.free_symbols}


def evaluate(expression, **values):
    """Evaluate a decoded native expression; no alternate formula is authored."""
    args = sorted(expression.free_symbols, key=lambda s: s.name)
    return sp.lambdify(args, expression, "numpy")(*(values[s.name] for s in args))


def expand_functions(expression, functions):
    return expression.replace(
        lambda node: node.is_Function and node.func.__name__ in functions,
        lambda node: functions[node.func.__name__](*node.args),
    )


def voigt_integral(sigma, gamma):
    """Substitute the shared component profiles into the shared convolution."""
    gaussian = rhs("lte_line_opacity", "gaussian_profile")
    lorentz = rhs("lte_line_opacity", "lorentz_profile")
    gs, ls = symbols(gaussian), symbols(lorentz)
    return expand_functions(rhs("lte_line_opacity", "voigt_profile"), {
        "G": lambda x: gaussian.subs({gs["x"]: x, gs["sigma_D"]: sigma}),
        "L": lambda x: lorentz.subs({ls["x"]: x, ls["gamma_L"]: gamma}),
    })


def reference_voigt(offset, sigma, gamma):
    integral = voigt_integral(sigma, gamma)
    integrand = integral.function
    return quad(lambda y: evaluate(integrand, x=offset, y=y), 0., np.inf,
                epsabs=2e-12, epsrel=2e-12)[0]


@pytest.mark.parametrize("temperature", [296., 900., 1800.])
def test_lte_strength_value_and_partial_temperature_derivative(temperature):
    expr = rhs("lte_line_opacity", "strength_temperature")
    args = dict(T=temperature, T_ref=296., c2=hcperk, w0=2300., E_lower=800.,
                S_ref=2e-21, Q_ratio=1.7)
    # Q is held fixed for this partial derivative, not a partition-function model.
    kernel = lambda t: line_strength(t, np.log(args["S_ref"]), args["w0"],
                                     args["E_lower"], args["Q_ratio"], args["T_ref"])
    np.testing.assert_allclose(kernel(temperature), evaluate(expr, **args), rtol=8e-15)
    derivative = sp.diff(expr, symbols(expr)["T"])
    np.testing.assert_allclose(jax.grad(kernel)(temperature), evaluate(derivative, **args), rtol=2e-14)
    args.update(T=296., Q_ratio=1.)
    np.testing.assert_allclose(evaluate(expr, **args), args["S_ref"], rtol=2e-15)


def test_doppler_width_records_rounded_prefactor():
    expected = evaluate(rhs("lte_line_opacity", "doppler_width"),
                        w0=np.array([1000., 4000.]), T=1000., k_B=kB, m_abs=18.*m_u, c=ccgs)
    # This is the tolerance of the rounded implementation constant, not exact
    # agreement with the currently exported atomic-mass unit.
    np.testing.assert_allclose(doppler_sigma(jnp.array([1000., 4000.]), 1000., 18.),
                               expected, rtol=3e-8)


def test_lpf_voigt_convolution_and_finite_line_sum():
    offsets = np.array([-1., -.2, 0., .35, 2.])
    expected = np.array([reference_voigt(x, .3, .15) for x in offsets])
    np.testing.assert_allclose(jax.vmap(lambda x: voigtone(x, .3, .15))(offsets),
                               expected, rtol=1e-6, atol=0.)
    centers = np.array([2000., 2000.5]); strengths = np.array([2e-21, 4e-21])
    grid = np.array([1999.5, 2000., 2000.4, 2001.])
    line_sum = rhs("lte_line_opacity", "line_sum")
    line_sum = line_sum.subs(symbols(line_sum)["N_lines"], 2).doit()
    # A reference profile is independent quadrature of the catalog convolution.
    expected = []
    for w in grid:
        resolved = line_sum.subs(symbols(line_sum)["w"], w)
        resolved = expand_functions(resolved, {
            "S_line": lambda i: strengths[int(i)-1],
            "w_line": lambda i: centers[int(i)-1],
        })
        resolved = expand_functions(resolved, {
            "phi_line": lambda i, x: reference_voigt(float(x), .3, .15),
        })
        expected.append(float(resolved))
    actual = xsvector(jnp.asarray(grid[None, :] - centers[:, None]),
                      jnp.array([.3, .3]), jnp.array([.15, .15]), jnp.asarray(strengths))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_columns_pressure_bases_and_cia_identical_partners():
    sigma = jnp.array([[1e-23, 2e-23], [2e-23, 3e-23]])
    density, dz = np.array([1e17, 2e17]), np.array([1e5, 2e5])
    column = evaluate(rhs("atmospheric_columns", "geometric_column"), n_abs=density, dz=dz)
    expected = evaluate(rhs("atmospheric_columns", "cross_section_depth"),
                        N_abs=column[:, None], sigma_abs=sigma)
    np.testing.assert_allclose(layer_optical_depth_from_cross_section(sigma, column), expected)
    np.testing.assert_allclose(layer_optical_depth_from_extinction(sigma*density[:, None], dz), expected)
    dp, vmr, absorber_mass, mean_mass, gravity = np.array([.1, .2]), .01, 18., 2.3, 1000.
    mass_unit = bar_cgs / opacity_factor
    mmr = evaluate(rhs("atmospheric_columns", "mixing_conversion"),
                   x_abs=vmr, m_abs=absorber_mass, m_mean=mean_mass)
    nv = evaluate(rhs("atmospheric_columns", "pressure_column_volume"),
                  dP=dp*bar_cgs, x_abs=vmr, m_mean=mean_mass*mass_unit, g=gravity)
    nm = evaluate(rhs("atmospheric_columns", "pressure_column_mass"),
                  dP=dp*bar_cgs, q=mmr, m_abs=absorber_mass*mass_unit, g=gravity)
    np.testing.assert_allclose(nv, nm, rtol=2e-15)
    expected = evaluate(rhs("atmospheric_columns", "cross_section_depth"),
                        N_abs=nv[:, None], sigma_abs=sigma)
    np.testing.assert_allclose(layer_optical_depth(dp, sigma, vmr, mean_mass, gravity), expected)
    np.testing.assert_allclose(layer_optical_depth(dp, sigma, mmr, absorber_mass, gravity), expected)
    wrong_basis = layer_optical_depth(dp, sigma, vmr, absorber_mass, gravity)
    assert not np.allclose(wrong_basis, expected, rtol=.01, atol=0.)
    cia = evaluate(rhs("atmospheric_columns", "cia_depth"),
                   k_cia=1e-45, n1=density, n2=density, dz=dz)
    actual = layer_optical_depth_from_log_cia(jnp.full((2, 1), -45.), density, density, dz)
    np.testing.assert_allclose(actual[:, 0], cia, rtol=2e-14)
    assert not np.allclose(actual[:, 0], cia/2, rtol=.01, atol=0.)
    np.testing.assert_array_equal(layer_optical_depth_from_log_cia(
        jnp.full((2, 1), -45.), jnp.zeros(2), density, dz), 0.)


def test_inverse_square_hydrostatic_constant_density_integral():
    pressures = jnp.array([.1, .2, .4, .8]); rho, radius, gravity = 1e-4, 7e9, 1000.
    radii, gravities = hydrostatic_radius_profile(pressures, jnp.full(3, rho),
                                                 gravity*radius**2/G, radius)
    expr = rhs("atmospheric_columns", "hydrostatic")
    expr = expand_functions(expr, {"rho": lambda r: sp.Float(rho)})
    r = symbols(expr)["r"]
    primitive = sp.integrate(expr, r)
    for index in range(3):
        dp = evaluate(primitive, r=float(radii[index+1]), g0=gravity, r0=radius) - evaluate(
            primitive, r=float(radii[index]), g0=gravity, r0=radius)
        np.testing.assert_allclose(dp, float(pressures[index+1]-pressures[index])*bar_cgs,
                                   rtol=4e-10)
    np.testing.assert_allclose(gravities*radii**2, gravity*radius**2, rtol=2e-15)


def test_ideal_gas_isothermal_hydrostatic_inverse_square_integral():
    pressures = jnp.array([.02, .07, .25, 1.])
    temperature, mean_mass, radius, gravity = 900., 2.3, 7e9, 1000.
    radii, gravities = hydrostatic_radius_profile_ideal_gas(
        pressures, jnp.full(3, temperature), mean_mass, radius, gravity,
        hydrostatic_scheme="variable_gravity",
    )
    density = rhs("atmospheric_columns", "ideal_gas_density")
    density_radius = symbols(density)["r"]
    gradient = expand_functions(rhs("atmospheric_columns", "hydrostatic"), {
        "rho": lambda r: density.subs(density_radius, r),
    })
    gradient = expand_functions(gradient, {
        "T": lambda r: sp.Float(temperature),
        "m_mean_r": lambda r: sp.Float(mean_mass*m_u),
    })
    pressure = next(call for call in gradient.atoms(sp.Function) if call.func.__name__ == "P")
    # Integrate d(log P)/dr from the lower boundary, deriving the analytic
    # isothermal consequence from the two catalog relations.
    logarithmic_gradient = sp.cancel(gradient / pressure)
    primitive = sp.integrate(logarithmic_gradient, symbols(logarithmic_gradient)["r"])
    parameters = dict(g0=gravity, r0=radius, k_B=kB)
    predicted_log_ratio = evaluate(primitive, r=np.asarray(radii), **parameters) - evaluate(
        primitive, r=radius, **parameters)
    np.testing.assert_allclose(predicted_log_ratio, np.log(np.asarray(pressures)/float(pressures[-1])),
                               rtol=2e-13, atol=2e-13)
    acceleration = expand_functions(-rhs("atmospheric_columns", "hydrostatic"), {"rho": lambda r: 1})
    np.testing.assert_allclose(gravities, evaluate(acceleration, r=np.asarray(radii), g0=gravity, r0=radius),
                               rtol=2e-15)
    assert float(radii[-1]) == radius


def test_homogeneous_emission_pi_convention_surface_and_derivative():
    tau = jnp.array([1e-10, .4, 5.]); mu = jnp.array([.2, .7, 1.]); source = 3.; bottom = 5.
    expr = rhs("thermal_emission", "homogeneous_layer")
    args = dict(I_in=bottom, S=source, tau=np.asarray(tau)[None, :], mu=np.asarray(mu)[:, None])
    expected = evaluate(expr, **args)
    actual = rtrun_emis_pureabs_ibased_intensity_surface(
        tau[None, :], jnp.full((1, 3), np.pi*source), jnp.full(3, np.pi*bottom), mu)/np.pi
    np.testing.assert_allclose(actual, expected, rtol=3e-15)
    no_surface = rtrun_emis_pureabs_ibased_intensity(tau[None, :], jnp.full((1, 3), np.pi*source), mu)/np.pi
    np.testing.assert_allclose(no_surface, evaluate(expr, **(args | {"I_in": 0.})),
                               rtol=3e-7, atol=2e-15)
    kernel = lambda t: rtrun_emis_pureabs_ibased_intensity_surface(
        jnp.array([[t]]), jnp.array([[np.pi*source]]), jnp.array([np.pi*bottom]), jnp.array([.7]))[0, 0]/np.pi
    np.testing.assert_allclose(jax.grad(kernel)(.4), evaluate(sp.diff(expr, symbols(expr)["tau"]),
                               I_in=bottom, S=source, tau=.4, mu=.7), rtol=2e-15)
    # Isothermal source and equally bright lower surface make I independent of
    # angle. This checks the 2*pi integral without quadrature approximation.
    flux_expr = expand_functions(rhs("thermal_emission", "flux"), {"I_mu": lambda mu: sp.Float(source)}).doit()
    nodes, weights = np.polynomial.legendre.leggauss(4); mus=(nodes+1)/2; weights=weights/2
    intensity = rtrun_emis_pureabs_ibased_intensity_surface(jnp.array([[.4]]),
        jnp.array([[np.pi*source]]), jnp.array([np.pi*source]), jnp.asarray(mus))
    flux = rtrun_emis_pureabs_ibased_flux_from_intensity(intensity, mus, weights)
    np.testing.assert_allclose(flux, float(flux_expr), rtol=2e-15)
    assert not np.isclose(float(flux[0])/np.pi, float(flux_expr))


def test_two_unequal_emitting_layers_preserve_top_to_bottom_order():
    optical_depth = np.array([[.25, .7], [1.1, .35]])
    source = np.array([[2., 4.], [6., 1.]])
    bottom = np.array([3., 5.]); mus = np.array([.3, .8])
    expression = rhs("thermal_emission", "homogeneous_layer")

    def compose(order):
        intensity = bottom[None, :]
        for layer in order:
            intensity = evaluate(expression, I_in=intensity, S=source[layer][None, :],
                                 tau=optical_depth[layer][None, :], mu=mus[:, None])
        return intensity

    # Arrays are top to bottom; an outgoing ray crosses the lower layer first.
    expected = compose((1, 0))
    actual = rtrun_emis_pureabs_ibased_intensity_surface(
        jnp.asarray(optical_depth), jnp.asarray(np.pi*source),
        jnp.asarray(np.pi*bottom), jnp.asarray(mus),
    )/np.pi
    np.testing.assert_allclose(actual, expected, rtol=3e-15)
    assert not np.allclose(expected, compose((0, 1)), rtol=.01, atol=0.)


def constant_extinction_chord(alpha, top):
    integral = expand_functions(rhs("transmission", "chord_depth"), {"alpha": lambda r: sp.Float(alpha)})
    integral = integral.subs(symbols(integral)["R_top"], top)
    return integral.doit()


def test_chords_and_area_converge_to_shared_continuous_relation():
    bottom, top, alpha = 2., 2.4, .8
    chord_expr = constant_extinction_chord(alpha, top)
    area = expand_functions(rhs("transmission", "blocked_area"), {
        "tau_chord": lambda b: chord_expr.subs(symbols(chord_expr)["b"], b),
    })
    integral = next(iter(area.atoms(sp.Integral)))
    expected = bottom**2 + 2*quad(lambda b: evaluate(integral.function, b=b), bottom, top,
                                 epsabs=1e-11, epsrel=1e-11)[0]
    errors = []
    for size in [16, 128]:
        boundaries = np.linspace(top, bottom, size+1)
        lower, heights = jnp.asarray(boundaries[1:]), jnp.asarray(-np.diff(boundaries))
        radial = (alpha*heights)[:, None]
        low = chord_optical_depth(chord_geometric_matrix_lower(heights, lower), radial)
        mid = chord_optical_depth(chord_geometric_matrix(heights, lower), radial)
        np.testing.assert_allclose(low[:, 0], evaluate(chord_expr, b=np.asarray(lower)), rtol=2e-13)
        np.testing.assert_allclose(mid[:, 0], evaluate(chord_expr, b=np.asarray(lower+heights/2)), rtol=2e-13)
        simpson = rtrun_trans_pureabs_simpson(mid, low, lower, heights)[0]
        trapezoid = rtrun_trans_pureabs_trapezoid(low, lower, top)[0]
        errors.append([abs(float(simpson)-expected), abs(float(trapezoid)-expected)])
    assert np.all(np.array(errors[1]) < np.array(errors[0]))
    assert np.all(np.array(errors[1])/expected < np.array([2e-5, 1.2e-4]))


@pytest.mark.parametrize("integration", ["simpson", "trapezoid"])
def test_transparent_art_output_is_bottom_normalized_area(integration):
    art = ArtTransPure(pressure_top=1e-5, pressure_btm=1., nlayer=8,
                       integration=integration, nu_grid=jnp.array([1000., 2000.]))
    result = art.run(jnp.zeros((8, 2)), jnp.full(8, 800.), jnp.full(8, 2.3), 7e9, 1000.)
    expected = evaluate(rhs("transmission", "normalized_radius"), R_tr_sq=(7e9)**2, R0=7e9)
    np.testing.assert_allclose(result, expected, rtol=0., atol=2e-15)
    depth = evaluate(rhs("transmission", "transit_depth"), R_tr_sq=(7e9)**2, R_star=7e10)
    assert depth == pytest.approx(.01)
    assert not np.allclose(result, depth)


def test_gaussian_and_rotation_discrete_kernels_and_zero_rotation():
    velocities = jnp.arange(-6., 7.); spectrum = jnp.zeros(40).at[20].set(1.)
    kernel = evaluate(rhs("spectral_observation", "gaussian_kernel"), v=np.asarray(velocities), beta=1.3)
    expected = np.zeros(40); expected[14:27] = kernel/kernel.sum()
    np.testing.assert_allclose(ipgauss(spectrum, velocities, 1.3), expected, atol=1e-16, rtol=2e-14)
    x = np.array([-.7, -.2, .1, .8, 1.2]); expr = rhs("spectral_observation", "rotation_raw")
    expected_raw = np.array([evaluate(expr, x=xi, u1=.3, u2=.1) for xi in x])
    np.testing.assert_allclose(rotkernel(jnp.asarray(x), .3, .1), expected_raw, atol=2e-15)
    # Compare a derivative away from compact-support and interpolation boundaries.
    grad = sp.diff(expr, symbols(expr)["x"])
    np.testing.assert_allclose(jax.grad(lambda x: rotkernel(x, .3, .1))(.2),
                               evaluate(grad, x=.2, u1=.3, u2=.1), rtol=3e-15)
    np.testing.assert_allclose(convolve_rigid_rotation(spectrum, velocities, 0., .3, .1), spectrum, atol=2e-16)
    broadened = convolve_rigid_rotation(spectrum, velocities, 3., .3, .1)
    assert float(broadened.sum()) == pytest.approx(1., abs=1e-14)


def test_sampling_velocity_sign_units_and_derivative():
    grid = jnp.linspace(1990., 2010., 101); observed = jnp.array([1995., 2000., 2005.]); rv=12.
    expr = rhs("spectral_observation", "doppler_sampling")
    shifted = evaluate(expr, w_obs=observed, RV=rv, c=c)
    actual = sampling(observed, grid, grid, rv)
    np.testing.assert_allclose(actual, shifted, rtol=2e-15)
    assert np.all(np.asarray(actual) > np.asarray(observed))
    gradient = jax.jacrev(lambda velocity: sampling(observed, grid, grid, velocity))(rv)
    np.testing.assert_allclose(gradient, evaluate(sp.diff(expr, symbols(expr)["RV"]),
                                               w_obs=observed, RV=rv, c=c), rtol=2e-15)


def test_single_line_column_emission_sampling_interfaces():
    """One deterministic path checks the quantity connections between families."""
    grid = jnp.linspace(1998., 2002., 101); temperature = 900.; reference_strength = 2e-21
    strength = line_strength(temperature, np.log(reference_strength), 2000., 800., 1.7, 296.)
    cross_section = xsvector(grid[None, :]-2000., jnp.array([.15]), jnp.array([.08]), jnp.array([strength]))
    optical_depth = layer_optical_depth_from_cross_section(cross_section[None, :], jnp.array([2e20]))
    source = piB(temperature, grid)
    mu = jnp.array([.5]); intensity = rtrun_emis_pureabs_ibased_intensity(optical_depth, source[None, :], mu)[0]
    observed = grid[jnp.array([35, 50, 65])]; rv=0.
    actual = sampling(observed, grid, intensity, rv)
    strength_ref = evaluate(rhs("lte_line_opacity", "strength_temperature"),
        T=temperature, T_ref=296., c2=hcperk, w0=2000., E_lower=800., S_ref=reference_strength, Q_ratio=1.7)
    expected=[]
    for wn, blackbody in zip(np.asarray(observed), np.asarray(source)[[35, 50, 65]]):
        sigma = evaluate(rhs("lte_line_opacity", "line_cross_section").replace(
            lambda n: n.is_Function and n.func.__name__ == 'phi',
            lambda n: sp.Float(reference_voigt(float(wn-2000.), .15, .08))), S=strength_ref)
        tau = evaluate(rhs("atmospheric_columns", "cross_section_depth"), sigma_abs=sigma, N_abs=2e20)
        physical = evaluate(rhs("thermal_emission", "homogeneous_layer"),
                            I_in=0., S=blackbody/np.pi, tau=tau, mu=.5)
        expected.append(np.pi*physical)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.xfail(strict=True, reason="convolve_same omits irfft length for odd total FFT length; tracked in the spectral observation FollowUp")
def test_odd_length_convolution_preserves_shape():
    spectrum = jnp.zeros(41).at[20].set(1.)
    actual = ipgauss(spectrum, jnp.arange(-6., 7.), 1.3)
    assert actual.shape == spectrum.shape

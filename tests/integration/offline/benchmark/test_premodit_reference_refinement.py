"""Selected-fixture convergence of a numerical PreMODIT emission reference.

The five synthetic lines exercise the real opacity and radiative-transfer
calculations. This narrow, smooth slab does not establish production CH4
accuracy, errors in the fixed single-broadening model, or absolute convergence.
"""

import importlib
import json
from pathlib import Path

import numpy as np

from exojax.database.contracts import Lines, MDBMeta, MDBSnapshot
from exojax.opacity import OpaPremodit
from exojax.postproc.binning import apply_bin_operator, piecewise_linear_bin_operator
from exojax.rt import ArtEmisPure
from exojax.utils.grids import wavenumber_grid


def _small_snapshot():
    return MDBSnapshot(
        meta=MDBMeta(
            dbtype="exomol",
            molmass=18.0,
            T_gQT=np.asarray([300.0, 600.0, 1000.0, 1500.0, 2200.0]),
            gQT=np.asarray([1.0, 1.35, 1.9, 2.7, 3.8]),
        ),
        lines=Lines(
            nu_lines=np.asarray([997.0, 999.0, 1000.0, 1002.0, 1004.0]),
            elower=np.asarray([20.0, 350.0, 900.0, 1800.0, 3200.0]),
            line_strength_ref_original=np.asarray(
                [2.0e-23, 4.0e-23, 3.0e-23, 5.0e-23, 2.5e-23]
            ),
        ),
        n_Texp=np.asarray([0.45, 0.55, 0.5, 0.65, 0.4]),
        alpha_ref=np.asarray([0.06, 0.07, 0.05, 0.08, 0.06]),
    )


def test_small_emission_reference_refinements_stay_within_budget(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "benchmark"))
    metrics = importlib.import_module("benchmark_metrics")
    snapshot = _small_snapshot()
    bin_edges = np.asarray([[995.0, 999.0], [999.0, 1003.0], [1003.0, 1007.0]])
    noise_sigma = 0.05
    maximum_error = 0.01 / 10.0
    maximum_q = 0.1 / 100.0
    configurations = {
        "base": (512, 32, 1, 100.0),
        "wavenumber": (1024, 32, 1, 100.0),
        "layers": (512, 64, 1, 100.0),
        "approximation": (512, 32, 2, 50.0),
        "combined": (1024, 64, 2, 50.0),
        "further": (2048, 128, 2, 25.0),
    }
    results = {}
    for label, (
        number_of_wavenumbers,
        number_of_layers,
        diffmode,
        d_e,
    ) in configurations.items():
        nu_grid, _, _ = wavenumber_grid(
            990.0, 1010.0, number_of_wavenumbers, unit="cm-1", xsmode="premodit"
        )
        opacity = OpaPremodit.from_snapshot(
            snapshot,
            nu_grid,
            diffmode=diffmode,
            broadening_resolution={"mode": "single", "value": None},
        )
        opacity.manual_setting(dE=d_e, Tref=1000.0, Twt=1200.0, Tmin=500.0, Tmax=2000.0)
        # Representative layer endpoints would change the atmospheric column
        # when nlayer changes; fix the physical boundaries instead.
        art = ArtEmisPure.from_pressure_boundaries(
            1.0, 2.0, nlayer=number_of_layers, nu_grid=nu_grid
        )
        temperature = art.powerlaw_temperature(1000.0, 0.005)
        cross_sections = opacity.xsmatrix(temperature, art.pressure)
        optical_depth = art.opacity_profile_xs(
            cross_sections,
            art.constant_mmr_profile(0.01),
            opacity.molmass,
            2478.57,
        )
        flux = art.run(optical_depth, temperature) / 20000.0
        operator = piecewise_linear_bin_operator(nu_grid, bin_edges)
        prediction = np.asarray(apply_bin_operator(operator, flux))
        results[label] = {
            "number_of_wavenumbers": number_of_wavenumbers,
            "number_of_layers": number_of_layers,
            "diffmode": diffmode,
            "dE": d_e,
            "prediction": prediction.tolist(),
            "pressure_boundaries_bar": np.asarray(art.pressure_boundary)[
                [0, -1]
            ].tolist(),
        }

    comparisons = {}
    for label in configurations:
        if label == "base":
            continue
        reference = "combined" if label == "further" else "base"
        comparisons[label] = {
            "reference": reference,
            **metrics.observation_error(
                results[label]["prediction"],
                results[reference]["prediction"],
                noise_sigma,
                max_error=maximum_error,
                max_q=maximum_q,
            ),
        }
    report = {
        "scope": "Numerical refinement of a selected synthetic five-line slab only",
        "production_reference_convergence": "not_established",
        "fixed_model": {
            "opacity": "PreMODIT with fixed single broadening parameters",
            "solver": "ArtEmisPure ibased, 8 streams",
            "temperature_profile_kelvin": "1000 * pressure_bar**0.005",
            "mass_mixing_ratio": 0.01,
            "gravity_cm_s2": 2478.57,
            "flux_scale": 20000.0,
            "bin_edges_cm_inverse": bin_edges.tolist(),
            "bin_measure": "wavenumber",
            "noise_sigma": noise_sigma,
        },
        "reference_budget": {
            "max_error_in_noise": maximum_error,
            "max_q": maximum_q,
        },
        "calculations": results,
        "comparisons": comparisons,
    }
    (tmp_path / "premodit_reference_refinement.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for result in results.values():
        np.testing.assert_array_equal(result["pressure_boundaries_bar"], [1.0, 2.0])
    assert all(comparison["passed"] for comparison in comparisons.values())
    assert (
        0.0
        < comparisons["further"]["max_error_in_noise"]
        < comparisons["combined"]["max_error_in_noise"]
    )

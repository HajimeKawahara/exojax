WASP-18b broadband RCE forward model (P7)
=======================================

P7 connects the device RCE solver and equilibrium chemistry to a broadband
WASP-18b column, then computes a resolved emission spectrum and the observed
planet/star flux ratio. The accepted scope is the eight-absorber model below,
finite numerical comparisons at the nominal point, and finite returned-state
diagnostics at 27 representative prior points. Retrieval validation follows
in P8.

Entry points
------------

.. list-table:: Implementation guide
   :header-rows: 1
   :widths: 35 65

   * - File(s) in ``examples/``
     - Purpose
   * - ``rce_wasp18b_forward.py``
     - Validate inputs, solve the column, stream resolved opacity chunks,
       integrate the observation, and write provenance and comparisons.
   * - ``rce_wasp18b_validation.py``
     - Generate fixed prior, initial-state, vertical-grid, opacity-node,
       bottom-depth, and quadrature suites; preserve failed states.
   * - ``_rce_wasp18b*.py``, ``_rce_broadband.py``
     - Prepare chemistry/entropy, absorption and stellar scattering, shared
       flux diagnostics, bounded mask continuation, and streamed spectra.
   * - ``prepare_rce_wasp18b_*.py``, ``assemble_rce_wasp18b_opacity.py``
     - Prepare and verify molecular/atomic opacity, high-temperature CIA,
       and stellar spectra; assemble manifests with hashes and domain checks.
   * - ``run_rce_wasp18b_*.csh``
     - Submit preparation and sensitivity jobs with resumable chunk records.
       Read each script's usage; preparation baselines are not the final
       eight-species acceptance configuration.
   * - ``rce_wasp18b_*audit.py``
     - Compare fluxes, spectral support, scattering treatments, broadening,
       and molecular omissions separately from numerical refinement.

For a manifest whose external assets are available, inspect it before a run::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=True python examples/rce_wasp18b_forward.py \
        --manifest /path/to/manifest.json \
        --observation /path/to/emission_candidate.ecsv \
        --check-inputs --output /path/to/new_input_report.json

The A100 calculations used the explicitly bound CKD adapter in
``results/run_rce_wasp18b_explicit_ckd.py`` to avoid embedding the large tables
as compiler constants. See ``results/README.rst`` for its narrower version
and source requirements. Database locations can be supplied with
``--database-root``; preparation scripts also document ``RCE_DATABASE_ROOT``
and ``RCE_INVENTORY``. Keep raw and prepared assets outside the source tree.

Runtime and external inputs
---------------------------

The numerical record uses Python 3.11, JAX/JAXLIB 0.6.2 with x64, and an
NVIDIA A100 40 GB. The production data are the WASP-18b SOSS emission candidate
in ExoSpecDB; 121 observation bins are used. Large opacity assets, PHOENIX
inputs, and the observational product are external inputs, not bundled data.
The accepted manifest and the local input locations are recorded in
``results/rce_wasp18b_p7_acceptance.json``.

ExoGibbs is an optional, separately installed source dependency. The accepted
configuration enables ``EquilibriumOptions.conservation_rtol`` and
``EquilibriumOptions.equilibrate_constraints`` and uses local ExoGibbs changes
in the gas solver and standard thermodynamics. Those changes are outside this
ExoJAX PR. The exact ExoGibbs source digest is
``48c08a79d971c47aea36559d8fdf28865c5e4d9e6dc2b5b9390739a9e00f42a2``;
an arbitrary published ExoGibbs installation does not reproduce this record.

Accepted configuration and evidence
-----------------------------------

.. list-table:: Adopted model
   :header-rows: 1
   :widths: 30 70

   * - Quantity
     - Setting
   * - Line absorbers
     - H2O, CO, CO2, Fe, TiO, SiO, Na, K
   * - Continuum / scattering
     - H-minus, H2-H2 and H2-He CIA, stellar scattering with the recorded
       finite spectral support
   * - Atmosphere
     - 32 layers; pressure boundaries from 1e-5 to 100 bar
   * - Broadband opacity
     - 0.2--50 microns; 5,522 bands; 16 g points; 512 native samples per band;
       16 temperature nodes and 32 pressure nodes
   * - Angular / Planck quadrature
     - Order 8 / 8
   * - Observation
     - 84 resolved chunks, Gaussian response R=700, 121 bins
   * - Fixed column quantities
     - Gravity 17,990 cm/s2, internal temperature 200 K, mu0=0.5
   * - Parameters
     - log10 metal/H, C/O, and log irradiation scaling

All 12 required nominal numerical comparisons meet the existing budget:
maximum absolute spectral change divided by the observed uncertainty below
0.1, and the Euclidean norm of those standardized changes below 0.3.
These are finite comparisons, not an error bound over the continuous prior.

The 27-point prior diagnostic records 27 finite, domain-valid returned states
and valid chemistry at all 891 layer/bottom nodes. RCE converges at 20 points;
four stop at the iteration budget and three at a domain step failure. The
seven states remain in the record with their failure flags. The diagnostic
does not compute resolved spectra at these 27 points or certify them all as
RCE solutions. Per-case results are in
``results/rce_wasp18b_final8_prior_analysis.csv`` and its JSON companion.

The acceptance audit reviewed 42 pytest logs, archived four historical logs,
and checked 176 evidence-file hashes. Counts from overlapping runs must not be
added as unique tests. The primitive derivative repair has 91 passing CPU
regressions; its nominal spectrum reproduces the previous spectrum exactly,
with a maximum temperature difference of 7.28e-12 K. The local environment
did not have Zarr for four historical serialization cases; P7 uses NPZ.

Recorded approximations and P8 handoff
--------------------------------------

The R=700 response is retained after the recorded R=400--1600 sensitivity
(maximum 0.1284 observed sigma). The eight absorbers are retained after
fixed-column OH/FeH/VO/HCN-HNC omission checks; the combined addition changes
the spectrum by up to 0.483 observed sigma. These are accepted physical
approximations, not numerical convergence tests or bounds after RCE
re-equilibration.

CO2's declared line-list completeness temperature is 4,000 K, exceeded in
18 prior cases. Numerical table validity and line-list completeness are
reported separately; missing hot lines are not bounded. The native
broadening mixtures, isotope convention, line wings, finite CIA/scattering
support, and stellar-tail prescription also remain recorded assumptions.
Stellar-tail RCE sensitivities were measured on an earlier Fe model, not
across the final eight-species prior.

Halving the bands changes the top-layer temperature by 61.834 K despite
passing the observation-space budget. The coarse 16-versus-5 temperature-node
comparison has norm 0.3733; the required 16-versus-9 comparison passes and
16 nodes are retained. No sub-kelvin temperature-accuracy claim is made.

P8 must investigate the seven unconverged points without trimming the prior,
validate implicit derivatives and convective switches on this real model,
and perform prior-predictive checks and synthetic recovery before conditional
retrieval. Posterior sensitivity to the accepted physical and observational
assumptions remains a later task.

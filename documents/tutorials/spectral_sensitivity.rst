Spectral sensitivity, information, and degeneracy
=================================================

A CO emission spectrum responds to molecular abundance, temperature, and
the temperature gradient. Those responses can resemble one another, so a
large derivative does not by itself guarantee a tight abundance constraint.
A Jacobian shows the individual responses; a covariance and its parameter
modes show which combinations are measured. This tutorial uses
ExoJAX automatic differentiation and ``linear_gaussian_diagnostics`` to
forecast local uncertainties and compare spectral binning, following the
information-based approach discussed by Batalha & Wogan (2026),
`Exoplanet Atmosphere Spectral Sensitivity Analysis with PICASO: Jacobians,
Linear-Gaussian Approximations, and Information Content
<https://arxiv.org/abs/2609.22634>`_.

The example uses a small set of real CO transitions, H2-H2 collision-induced
absorption (CIA), and ``ArtEmisPure`` radiative transfer. All data are bundled
with the repository, so no downloads or retrieval run are needed. The
atmosphere and observational errors are illustrative; this is not a forecast
for a particular object or instrument.

Run the example
---------------

From a checkout with ExoJAX and Matplotlib installed:

.. code-block:: console

   JAX_PLATFORMS=cpu MPLBACKEND=Agg python examples/spectral_sensitivity.py

The script enables JAX 64-bit arithmetic and writes the two figures below
to ``documents/tutorials/spectral_sensitivity_files``. Use
``--output-dir /path/to/figures`` to choose another directory.
The :download:`complete example <../../examples/spectral_sensitivity.py>`
contains the imports, model setup, and plotting code.
Run it from the checkout so that the accompanying
``examples/spectral_sensitivity_data`` directory is available.
The :download:`CIA table <../../examples/spectral_sensitivity_data/H2-H2_2011_4320-4370.cia>`
is about 135 KiB.
On the CPU used for this example, data setup and diagnostics took about
7.4 seconds including the first JAX compilation, excluding imports and
plotting. The script prints this timing for your machine.

The CO database is the repository's 259-transition subset of the ExoMol
Li2015 line list for :math:`{}^{12}\mathrm{C}{}^{16}\mathrm{O}`, including
its partition function and H2 pressure-broadening data. ``OpaDirect``
evaluates the Voigt profiles directly. The CIA file is a small, unmodified
spectral selection from HITRAN's ``H2-H2_2011.cia``: 4320--4370
:math:`\mathrm{cm^{-1}}`, with all 113 temperatures from 200 to 3000 K.
See the :download:`data provenance <../../examples/spectral_sensitivity_data/README.md>`
for its source and extraction details. This temperature coverage matters:
the full fiducial temperature profile lies inside the tabulated range, so
the CIA coefficients can respond to changes in temperature.

Define the observation and parameter coordinates
------------------------------------------------

We model a 24-layer atmosphere from :math:`10^{-4}` to 100 bar, using
``ArtEmisPure`` with four streams and

.. math::

   T(P)=T_0(P/1\,\mathrm{bar})^\alpha.

The fixed gravity is :math:`10^{4.4}\,\mathrm{cm\,s^{-2}}`, the mean
molecular weight is 2.33 atomic mass units, and the H2 volume mixing ratio
is 0.855. CO has a constant mass mixing ratio :math:`q_{\rm CO}`.
The example holds the background composition fixed as the trace CO
abundance varies. Molecular line absorption and CIA are the two opacity
sources.

The parameters and independent Gaussian prior standard deviations are

.. math::

   \boldsymbol{\theta}=(T_0,\log_{10}q_{\rm CO},\alpha),\qquad
   \boldsymbol{\theta}_0=(1200\,\mathrm{K},-2.3,0.1),\qquad
   \boldsymbol{\sigma}_a=(100\,\mathrm{K},0.5\,\mathrm{dex},0.03).

The Gaussian prior is defined in these coordinates; the CO abundance prior
is therefore lognormal in linear mixing ratio. A uniform-prior interval is not
a Gaussian standard deviation.

The data and opacity calculators are initialized once, outside the
differentiated function. Temporary copies keep the database reader's caches
away from the installed CO data:

.. literalinclude:: ../../examples/spectral_sensitivity.py
   :language: python
   :start-after: # BEGIN MODEL SETUP
   :end-before: # END MODEL SETUP
   :dedent: 4

The model first calculates emission at 4096 uniformly spaced cell midpoints
over 4330--4362 :math:`\mathrm{cm^{-1}}`, near the 2.3-micron CO band head.
It then averages groups of 16 cells into 256 nonoverlapping top-hat
observation bins. The output
is :math:`y_i=\langle F_{\tilde\nu}\rangle_i/F_*`, with the fixed scale
:math:`F_*=10^4\,\mathrm{erg\,s^{-1}\,cm^{-2}/(cm^{-1})}`. This is a
fixed unit conversion, not a fitted normalization. The figure displays bin
centers in wavelength, but the averages remain in wavenumber space.

Inside the forward model, CO line strengths, Doppler widths, pressure
widths, CIA coefficients, and the thermal source function all respond to
the temperature profile. The opacity matrices are recomputed for every
parameter vector, so their derivatives are included in the Jacobian.

.. literalinclude:: ../../examples/spectral_sensitivity.py
   :language: python
   :start-after: # BEGIN OBSERVED SPECTRUM
   :end-before: # END OBSERVED SPECTRUM
   :dedent: 4

The binning is inside ``observed_spectrum``, so ``jax.jacfwd`` differentiates
the complete observable. For your own model, include its convolution,
radial velocity, throughput, and sampling operations before taking the
Jacobian. In this example the only instrumental operation is the top-hat
average.

We assume independent, fixed Gaussian measurement errors
:math:`\sigma_{e,i}=0.03` on the scaled flux. This is an absolute error,
not a constant fractional error or a noise model differentiated with the
spectrum. No noisy realization is needed for this local forecast.

Compute the diagnostics
-----------------------

For the observation-space Jacobian
:math:`K_{ij}=\partial y_i/\partial\theta_j`, define diagonal error and
prior covariances :math:`S_e=\mathrm{diag}(\sigma_e^2)` and
:math:`S_a=\mathrm{diag}(\sigma_a^2)`. Linearizing the spectrum at
:math:`\boldsymbol{\theta}_0` gives

.. math::

   F=K^\mathsf{T}S_e^{-1}K,\qquad
   S_{\rm post}=(F+S_a^{-1})^{-1},\qquad
   A=S_{\rm post}F.

Here :math:`F` is the data Fisher matrix, :math:`S_{\rm post}` is the
local posterior covariance, and :math:`A` is the averaging kernel. The
diagonal of :math:`S_{\rm post}` gives the marginal variances after allowing
all three parameters to vary. The effective degrees of freedom and the
Gaussian entropy reduction in bits are

.. math::

   d=\mathrm{tr}(A),\qquad
   H=\frac{\ln\det S_a-\ln\det S_{\rm post}}{2\ln2}.

The factor :math:`\ln2` converts natural-log entropy to bits. This is a
covariance-based entropy reduction, not the realized posterior-to-prior
KL divergence, which also depends on the posterior mean displacement.

.. code-block:: python

   from exojax.utils.information import linear_gaussian_diagnostics

.. literalinclude:: ../../examples/spectral_sensitivity.py
   :language: python
   :start-after: # BEGIN DIAGNOSTICS
   :end-before: # END DIAGNOSTICS
   :dedent: 4

The function accepts a two-dimensional Jacobian, one noise standard
deviation per observation, and one prior standard deviation per parameter.
Both standard-deviation arrays must be finite and strictly positive.
It returns a dictionary containing ``fisher``, ``posterior_cov``,
``posterior_std``, ``averaging_kernel``, ``degrees_of_freedom``,
``information_bits``, ``singular_values``, and ``parameter_modes``.
Covariances and the averaging kernel use the original parameter coordinates.
Correlated errors or correlated priors are outside this function's scope.
The function can be JIT-compiled. Numerical input checks run on concrete
arrays; callers must ensure finite inputs and positive standard deviations
when supplying traced arrays. Differentiating the diagnostic function itself
has additional SVD restrictions, distinct from differentiating the forward
model to obtain its Jacobian.

Interpret sensitivities and weak modes
--------------------------------------

Temperature, log abundance, and the gradient exponent have different
scales. To compare modes we use the dimensionless Jacobian and parameter
perturbations

.. math::

   B_{ij}=\frac{K_{ij}\sigma_{a,j}}{\sigma_{e,i}},\qquad
   z_j=\frac{\theta_j-\theta_{0,j}}{\sigma_{a,j}},\qquad
   B=U\,\mathrm{diag}(s_k)V^\mathsf{T}.

Each row of ``parameter_modes`` is a row of :math:`V^\mathsf{T}`, ordered
by descending ``singular_values``. A mode coordinate is
:math:`a_k=\boldsymbol{v}_k^\mathsf{T}\boldsymbol{z}`. Its prior variance
is one and its posterior variance is :math:`1/(1+s_k^2)`. Thus
:math:`s_k\ll1` identifies a combination dominated by its prior, while
:math:`s_k\gg1` indicates substantial information from the observations.
Mode signs are arbitrary. In terms of these singular values,

.. math::

   d=\sum_k\frac{s_k^2}{1+s_k^2},\qquad
   H=\frac12\sum_k\log_2(1+s_k^2).

These expressions also remain well defined for a rank-deficient Jacobian
because the proper Gaussian prior supplies finite variance in its null
space. The implementation uses the scaled decomposition to avoid directly
inverting the Fisher matrix or forming determinants.

.. figure:: spectral_sensitivity_files/spectrum_and_jacobian.png
   :alt: CO emission spectrum with H2-H2 CIA and the scaled Jacobian with respect to temperature, CO mass mixing ratio, and temperature gradient.

   The lower panel shows :math:`B`, the flux response to a one-prior-standard-
   deviation parameter change in units of the noise standard deviation.
   CO abundance changes line depths, while temperature and its gradient
   affect both the line and continuum emission. Their partially similar
   responses create parameter correlations. Derivative curves describe
   local slopes, not finite excursions over the full prior.

Compare spectral binning with propagated noise
----------------------------------------------

The fine observation has 256 bins, each :math:`0.125\,\mathrm{cm^{-1}}`
wide. The coarse observation averages every eight adjacent fine bins,
giving 32 bins of width :math:`1\,\mathrm{cm^{-1}}`. Both cover the
same wavelength interval. The average is a fixed linear operator, so the
coarse Jacobian follows directly from the fine one:

.. math::

   \bar y_b=\frac18\sum_{i\in b}y_i,\qquad
   \bar K_{bj}=\frac18\sum_{i\in b}K_{ij},\qquad
   \bar\sigma_{e,b}=\frac{\sigma_e}{\sqrt8}.

The last relation propagates the independent equal errors of the fine bins.
It gives the coarse observation the gain in per-bin precision associated
with averaging the same measured data, with no additional exposure time.
This compares how much information is retained after binning under the
stated noise model; no instrument-specific photon-counting model is
assumed. The top-hat bins do not describe a specific spectrograph's
line-spread function.

The script gives the following rounded results:

.. list-table::
   :header-rows: 1

   * - Quantity
     - Fine bins
     - Coarse bins
   * - :math:`\sigma(T_0)` [K]
     - 1.427
     - 2.464
   * - :math:`\sigma(\log_{10}q_{\rm CO})` [dex]
     - 0.0201
     - 0.0337
   * - :math:`\sigma(\alpha)`
     - 0.00203
     - 0.00420
   * - Information :math:`H` [bits]
     - 18.376
     - 16.391
   * - Effective degrees of freedom :math:`d`
     - 2.994
     - 2.975
   * - Smallest singular value
     - 12.71
     - 6.40

.. figure:: spectral_sensitivity_files/binning_and_degeneracy.png
   :alt: Marginal uncertainties for fine and coarse spectral bins, the weakest coarse-bin parameter mode, and the coarse-bin posterior correlation matrix.

   Averaging spectral structure can weaken joint constraints even though
   each coarse bin has a smaller error. The weakest mode is expressed in
   prior-scaled parameter coordinates. Correlation entries are rounded to
   two decimals.

The coarse observation retains information on all three modes: even its
smallest singular value is greater than one. Its weakest mode combines a
higher temperature and CO abundance with a shallower temperature gradient,
with coefficients approximately :math:`(0.154,0.402,-0.902)` in the
prior-scaled coordinates. The CO abundance and gradient are therefore
strongly anticorrelated, but do not have an exact local degeneracy.

Binning increases the CO abundance uncertainty by a factor of about 1.7
and the gradient uncertainty by about 2.1, while losing approximately
2 bits of information. The degrees of freedom remain close to three
because all three modes remain data-informed. This illustrates why the
mode strengths and marginal errors give useful detail beyond a count of
constrained parameters. Coarse binning loses the within-bin differences
between the parameter responses; those differences cannot be recovered
by increasing the precision of the bin average.

Numerical checks
----------------

At the fiducial state, doubling the internal quadrature grid from 4096 to
8192 cells changes the binned flux by less than :math:`0.02\sigma_e` and
each Jacobian column by less than 0.1% of its maximum absolute amplitude.
The observation-space Jacobian also agrees with centered finite differences
to within :math:`3\times10^{-7}` on the same per-column scale, using steps
of :math:`10^{-3}` K, :math:`10^{-5}` dex, and :math:`10^{-6}` for
:math:`T_0`, log CO abundance, and :math:`\alpha`, respectively.
These checks concern the local numerical calculation; the finite line list
and chosen atmospheric model still set the physical scope of the example.

Limits and applying this to a retrieval
---------------------------------------

Extend the CO model with your usual ExoJAX opacity sources and choose a
fiducial state and Gaussian prior in the exact coordinates passed to
``jax.jacfwd``. Include nuisance parameters such as
calibration or radius when they affect the intended observation; fixing
them can make a forecast too optimistic. If you change spectral binning,
propagate the measurement covariance with the same binning operator.
Averaging independent equal-error samples reduces their standard deviation
by the square root of the number averaged; overlapping kernels may produce
correlations that this diagonal-error interface cannot represent.

This is a local linear Gaussian forecast. Automatic differentiation computes
local derivatives accurately, but does not remove nonlinearity, multimodality,
physical boundaries, upper limits, or forward-model error. In particular,
parameter correlations need not remain linear over the full prior.
The selected CO transitions and H2-H2 CIA do not constitute a complete
opacity model for an observed atmosphere. Compare with posterior sampling
when those effects matter, and recompute at other plausible fiducial states
before drawing observing-strategy conclusions.

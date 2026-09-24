Comparing Na and K resonance profiles
====================================

``OpaAlkaliTable`` combines tabulated Allard wings with an approximate Voigt
core. This tutorial compares it with the existing sub-Voigt prescription and
Voigt profiles using the same D1 and D2 line strengths, collision widths,
pressure shifts, temperature, and perturber density. No Kurucz or VALD
line-list download is required.

The two table models have different collision partners:

.. list-table::
   :header-rows: 1

   * - Model
     - Absorber / perturber
     - Source
   * - ``allard2019_na_h2``
     - Na I / H2
     - `Allard et al. (2019) <https://doi.org/10.1051/0004-6361/201935593>`_
   * - ``allard2024_k_he``
     - K I / He
     - `Allard et al. (2024) <https://doi.org/10.1051/0004-6361/202348711>`_

Obtain the data
--------------

For Na, download ``opacity.tar.gz`` from the
`CDS Na-H2 catalogue <https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/628/A120>`_.
The reader accepts the original archive, including its nested tables.

For K, download the ``D1`` and ``D2`` directories from the
`CDS K-He catalogue <https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/683/A188>`_
into one local directory. Include the corrected
``D2/tableD2_KHe_800_1e21_2025.omg`` file rather than its superseded 800 K version.
Keep the CDS ``ReadMe`` and ``README.pdf`` alongside the tables for their
format descriptions and provenance. The example reads local data and does
not download files.

Run the comparison
------------------

Run these commands from the repository root, adjusting the data paths:

.. code-block:: console

   python examples/compare_alkali_models.py /path/to/opacity.tar.gz \
       --species Na --output alkali_models_na.png
   python examples/compare_alkali_models.py /path/to/KHe \
       --species K --output alkali_models_k.png

The defaults are 1000 K and a perturber density of 10^19 cm-3. Each comparison
uses a single perturber with ``vmr_perturber=1``. The corresponding pressure
is computed as ``P_bar = n * kB * T / 1e6``. Use ``--temperature`` to select
another exact tabulated temperature and ``--density`` to change the density
within the model's supported range.

The left panels show both wings and the summed resonance doublet. The right
panels magnify the cores and the grey bands marking the 20--30 cm-1 core/wing
transitions on either side of each unshifted line center. The ratio below
each spectrum makes departures from the sub-Voigt prescription visible.

What is being compared
----------------------

For every curve, the example reads the reference wavelength, integrated
strength, collisional HWHM, and pressure shift from the same CDS table. It
scales the last two by ``n / n_reference`` and uses the same thermal Doppler
width for the Voigt and sub-Voigt references. All cores also include the
natural width inferred from the table's oscillator strength. Only D1 and D2
contribute; other transitions, ionization, and absorber abundance are excluded.
Cross sections are expressed per ground-state Na or K atom, with the table's
oscillator-strength normalization, without an LTE ground-state population
or stimulated-emission correction. No curve is renormalized to its peak or
to the area of the displayed wavelength interval. The existing sub-Voigt
factor ``1 / 0.998`` is retained.

The table curve is a numerical hybrid: it uses a Voigt core and a smooth
transition to the density-expanded asymmetric wings. This core is an
ExoJAX approximation, not a reconstruction of the authors' exact unified
core. The shaded transition is configurable through ``core_transition``;
it is not a new broadening law. Temperature interpolation is avoided by
selecting an exact table temperature.

Each component's tabulated wing is set to zero outside its spectral support;
the example keeps this behavior instead of extrapolating. In particular,
the Na D1 density expansion at 1000 K ends at +86 cm-1, while D2 extends much
farther into the blue wing. The Na D1 far red wing is supplied separately
and scales linearly with density. Negative density-expansion results are
clipped to zero by the calculator; zeros are omitted from the logarithmic
plots. These finite-table conventions also affect the comparison.

The example deliberately matches widths and strengths to isolate profile
shape. It does not reproduce the default VALD/Kurucz broadening of
``OpaAlkali`` or ``OpaDirect``. Na-H2 and K-He also do not constitute a full
H2/He mixture model; each result represents its named absorber/perturber pair.

.. literalinclude:: ../../examples/compare_alkali_models.py
   :language: python
   :start-at: def compare(
   :end-before: def main():

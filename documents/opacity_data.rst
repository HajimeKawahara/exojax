Starter Opacity Data
====================

Start computing spectra with small, precomputed opacity tables. These
datasets load into the existing ``OpaCKD`` and ``OpaDiffgrid`` calculators
without downloading a molecular line list or building an opacity table.
Follow :doc:`tutorials/starter_opacity` to calculate a water transmission
spectrum and a carbon-monoxide emission spectrum with a temperature
derivative.

.. note::

   This distribution is being prepared. The server download URLs below
   become usable after the generated data directories are published.
   Maintainers can build and test the same examples locally before
   publication.

Datasets
--------

Each dataset has its own versioned directory. Download only the dataset
needed for your example. Its ``manifest.json`` records the file sizes,
SHA256 digests, source data, generation settings, software versions, and
the checksum of the accompanying ``validation.json`` report.

.. list-table::
   :header-rows: 1
   :widths: 20 25 30 25

   * - Dataset
     - Example
     - Source
     - Manifest
   * - ``h2o-ckd-v1``
     - Broad-band water transmission
     - A compact subset of an ExoMolOP water k table
     - `H2O manifest <https://secondearths.sakura.ne.jp/exojax/data/opacity/h2o-ckd-v1/manifest.json>`_
   * - ``co-diffgrid-v1``
     - CO emission near 2.3 micrometers and its temperature derivative
     - ExoMol Li2015, calculated with PreMODIT
     - `CO manifest <https://secondearths.sakura.ne.jp/exojax/data/opacity/co-diffgrid-v1/manifest.json>`_

Direct file links after publication:

* H2O, approximately 37.53 MB: `h2o_ckd.h5 <https://secondearths.sakura.ne.jp/exojax/data/opacity/h2o-ckd-v1/h2o_ckd.h5>`_.
* CO, approximately 13.79 MB: `co_diffgrid.npz <https://secondearths.sakura.ne.jp/exojax/data/opacity/co-diffgrid-v1/co_diffgrid.npz>`_
  and its required `co_diffgrid_metadata.json <https://secondearths.sakura.ne.jp/exojax/data/opacity/co-diffgrid-v1/co_diffgrid_metadata.json>`_.

The two prepared datasets together use approximately 51.3 MB. Sizes are
decimal megabytes; exact file sizes are recorded in their manifests.

Use the downloader below to keep the manifest, license, and validation
report together with the opacity files and verify their contents.

The tables contain one molecular absorber each. The examples illustrate
radiative transfer with that absorber alone; realistic atmosphere models
may also require CIA, scattering, other gases, and instrumental effects.

Download and reuse
------------------

``fetch_starter_opacity`` downloads a manifest and its files, verifies
their sizes and SHA256 digests, and returns the local dataset directory.
Subsequent calls reuse verified cached files.

.. code-block:: python

    from exojax.provider.starter import fetch_starter_opacity

    directory = fetch_starter_opacity("h2o-ckd-v1")
    print(directory)

Use ``cache_dir="./opacity_cache"`` to choose another cache location.
The ``base_url`` argument selects a mirror with the same directory
structure. Keep the manifest with the data when sharing a local copy.

Choosing a table
----------------

CKD represents absorption distributions within spectral bands. It is
suited to the broad-band example on this page. The supplied table retains
its original band and quadrature definitions; it does not resolve
individual spectral lines. Use temperatures and pressures within its
recorded grid. The current CKD interpolator uses edge values outside that
grid, so a calculation outside the range does not demonstrate validity
there. Combining several molecular k tables also requires an appropriate
mixing method and validation.

DiffGrid samples high-resolution opacity and its inverse-temperature
derivative. Its pressure layers are fixed: changing their number, bounds,
or placement requires another table. Call ``opa.check_pressure_grid``
before using a downloaded table with an atmospheric model. Keep every
temperature in the model or inference prior within the supplied range,
including when running a compiled JAX function.

The DiffGrid loader checks the ExoJAX version and precision by default.
Use the version recorded in the manifest and enable JAX 64-bit precision
before loading. After a software upgrade, obtain or rebuild a compatible
table. ``strict=False`` is intended for compatibility checks; it is not
the default for this tutorial.

Provenance and reuse
--------------------

Precomputation preserves a route back to the source line data. Cite the
ExoJAX method, the source line list, and ExoMolOP when applicable, as
recorded in the dataset manifest. The software's MIT license does not
replace the source data license. Follow the dataset's recorded license
and attribution requirements when redistributing modified tables.
These two ExoMol-derived datasets use
`CC BY-SA 4.0 <https://creativecommons.org/licenses/by-sa/4.0/>`_;
see the `ExoMol data license <https://exomol.com/data/licence/>`_.

These small tables are starting points. See
:doc:`tutorials/transmission_ckd_exomolop` for loading ExoMolOP tables
directly, :doc:`userguide/diffgrid` for constructing DiffGrid, and
:doc:`userguide/save_diffgrid` for saving your own tables. Retain the
generation settings and validate the opacity, spectrum, and derivatives
for the temperatures and application you intend to use.

Build and publish a release
---------------------------

The maintainer build needs the upstream ExoMolOP HDF5 table for water
and the ExoMol Li2015 database directory for CO. The beginner's example
does not need either source dataset. The build script checks the water
source against a pinned SHA256 digest.

Prepare public datasets from a clean checkout with the intended ExoJAX
version freshly installed. Confirm that ``exojax.__version__`` identifies
that installation: a stale generated version file in a development
checkout must not determine a published DiffGrid archive's version.
Rebuild locally staged tables in this release environment before
publication. From the repository root, run:

.. code-block:: console

    python tools/build_starter_opacity.py \
        --dataset h2o-ckd-v1 \
        --h2o-source /path/to/full-water-ExoMolOP-table.h5 \
        --output-dir documents/_build/opacity-data
    python tools/build_starter_opacity.py \
        --dataset co-diffgrid-v1 \
        --co-source /path/to/CO/12C-16O/Li2015 \
        --output-dir documents/_build/opacity-data
    python examples/starter_opacity.py \
        --data-root documents/_build/opacity-data \
        --output documents/_build/starter_opacity.png

The build script writes the table files and their manifests under the
chosen output directory. Source data and generated binaries remain
outside the versioned documentation sources. Inspect the recorded
validation results and the example figure before publication.

Publish the generated ``h2o-ckd-v1/`` and ``co-diffgrid-v1/`` directories
under ``https://secondearths.sakura.ne.jp/exojax/data/opacity/``. The
Sphinx HTML landing page and the data directories can share the same
server, but the data directories are deployed separately from the
documentation build. Confirm both hosted downloads with an empty cache
and run the tutorial before removing the preparation notice above.

Treat a published dataset ID as immutable. A changed table or manifest
gets a new ID so cached examples remain reproducible. Keep previous
datasets available when publishing a replacement, including when an
ExoJAX upgrade requires a new DiffGrid archive.

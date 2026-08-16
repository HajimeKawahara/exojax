Patch-based CKD table precomputation
====================================

Hajime Kawahara with codex June 21st (2026)

Large self-generated CKD tables can require too much memory when the
full wavenumber range is precomputed at once. This tutorial shows how to
split the range into smaller patches and save one CKD table per patch.

The example below uses a small toy opacity calculator so that the
notebook runs quickly. The same helper can be used with an ExoMol or
HITRAN based opacity calculator by changing ``make_base_opa`` and
``make_nu_grid``.

Imports
-------

.. code:: ipython3

    import json
    #import os
    import shutil
    from pathlib import Path
    
    #os.environ["JAX_PLATFORMS"] = "cpu"
    
    import numpy as np
    
    from exojax.opacity import OpaCKD
    from exojax.opacity.ckd.precompute import precompute_ckd_tables_by_patches

Define a base opacity factory
-----------------------------

``precompute_ckd_tables_by_patches`` receives two callables.
``make_nu_grid`` creates the wavenumber grid for one patch.
``make_base_opa`` creates the opacity calculator used by ``OpaCKD`` for
that patch.

.. code:: ipython3

    class ToyBaseOpacity:
        def __init__(self, nu_grid):
            self.nu_grid = np.asarray(nu_grid)
    
        def xsmatrix(self, T_grid, P_grid):
            T_grid = np.asarray(T_grid)
            P_grid = np.asarray(P_grid)
            nu_scale = self.nu_grid / self.nu_grid[0]
            return (
                1.0e-24
                * T_grid[:, None]
                / T_grid[0]
                * P_grid[:, None]
                / P_grid[0]
                * nu_scale[None, :]
            )
    
    
    def make_nu_grid(nu_min, nu_max, n_grid):
        return np.linspace(nu_min, nu_max, n_grid)
    
    
    def make_base_opa(nu_grid, nu_min, nu_max):
        return ToyBaseOpacity(nu_grid)

Build patch tables
------------------

The manifest is updated after each patch. If a long run stops midway,
completed patch tables and the manifest remain on disk.

.. code:: ipython3

    out_dir = Path("ckd_patch_demo_output")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    
    T_grid = np.array([500.0, 1000.0])
    P_grid = np.array([1.0e-3, 1.0e-1])
    
    manifest = precompute_ckd_tables_by_patches(
        make_base_opa,
        make_nu_grid,
        nu_min=1000.0,
        nu_max=1040.0,
        patch_width=10.0,
        T_grid=T_grid,
        P_grid=P_grid,
        out_dir=out_dir,
        Ng=4,
        ckd_resolution=100.0,
        nu_grid_points_per_patch=32,
        overwrite=True,
        table_prefix="toy_ckd",
    )
    
    summary = {
        "schema_version": manifest["schema_version"],
        "n_tables": len(manifest["tables"]),
        "nu_range_cm-1": [manifest["nu_min_cm-1"], manifest["nu_max_cm-1"]],
        "patch_width_cm-1": manifest["patch_width_cm-1"],
    }
    print(json.dumps(summary, indent=2))


.. parsed-literal::

    Patch 1/4: 1000.000-1010.000 cm-1, nu_grid_points=32
    Generated g-grid: 4 points, range [0.0694, 0.9306]
    Processing 1 spectral bands...
      Band 1: [1000.0, 1010.0] cm⁻¹, 32 frequencies
    Creating CKD table info...
    CKD precomputation complete! Ready for interpolation.
    Table dimensions: T=2, P=2, g=4, bands=1
    Saved CKD table to: ckd_patch_demo_output/toy_ckd_R100_01000_01010.npz
    Patch 2/4: 1010.000-1020.000 cm-1, nu_grid_points=32
    Generated g-grid: 4 points, range [0.0694, 0.9306]
    Processing 1 spectral bands...
      Band 1: [1010.0, 1020.0] cm⁻¹, 32 frequencies
    Creating CKD table info...
    CKD precomputation complete! Ready for interpolation.
    Table dimensions: T=2, P=2, g=4, bands=1
    Saved CKD table to: ckd_patch_demo_output/toy_ckd_R100_01010_01020.npz
    Patch 3/4: 1020.000-1030.000 cm-1, nu_grid_points=32
    Generated g-grid: 4 points, range [0.0694, 0.9306]
    Processing 1 spectral bands...
      Band 1: [1020.0, 1030.0] cm⁻¹, 32 frequencies
    Creating CKD table info...
    CKD precomputation complete! Ready for interpolation.
    Table dimensions: T=2, P=2, g=4, bands=1
    Saved CKD table to: ckd_patch_demo_output/toy_ckd_R100_01020_01030.npz
    Patch 4/4: 1030.000-1040.000 cm-1, nu_grid_points=32
    Generated g-grid: 4 points, range [0.0694, 0.9306]
    Processing 1 spectral bands...
      Band 1: [1030.0, 1040.0] cm⁻¹, 32 frequencies
    Creating CKD table info...
    CKD precomputation complete! Ready for interpolation.
    Table dimensions: T=2, P=2, g=4, bands=1
    Saved CKD table to: ckd_patch_demo_output/toy_ckd_R100_01030_01040.npz
    {
      "schema_version": "ckd_patch_manifest.v1",
      "n_tables": 4,
      "nu_range_cm-1": [
        1000.0,
        1040.0
      ],
      "patch_width_cm-1": 10.0
    }


Inspect the manifest and load a table
-------------------------------------

Each table is a normal ``OpaCKD`` table. It can be loaded with
``OpaCKD.load_only().load_tables(...)``.

.. code:: ipython3

    manifest_path = out_dir / "ckd_patch_manifest.json"
    manifest_on_disk = json.loads(manifest_path.read_text())
    
    for table in manifest_on_disk["tables"]:
        print(table["index"], table["path"], table["n_bands"])
    
    first_table = manifest_on_disk["tables"][0]
    opa_ckd = OpaCKD.load_only().load_tables(first_table["path"])
    
    print("ready:", opa_ckd.ready)
    print("log_kggrid shape:", opa_ckd.ckd_info.log_kggrid.shape)
    print("nu_bands:", opa_ckd.nu_bands[:3])


.. parsed-literal::

    1 ckd_patch_demo_output/toy_ckd_R100_01000_01010.npz 1
    2 ckd_patch_demo_output/toy_ckd_R100_01010_01020.npz 1
    3 ckd_patch_demo_output/toy_ckd_R100_01020_01030.npz 1
    4 ckd_patch_demo_output/toy_ckd_R100_01030_01040.npz 1
    ready: True
    log_kggrid shape: (2, 2, 4, 1)
    nu_bands: [1004.98755]


Using an ExoMol or PreMODIT base opacity
----------------------------------------

For real self CKD generation, replace the toy opacity factory with the
base opacity used for your line database. A typical pattern is:

.. code:: python

   from exojax.database.exomol.api import MdbExomol
   from exojax.opacity import OpaPremodit
   from exojax.utils.grids import wavenumber_grid

   def make_nu_grid(patch_min, patch_max, n_grid):
       nu_grid, _, _ = wavenumber_grid(
           patch_min,
           patch_max,
           n_grid,
           unit="cm-1",
           xsmode="premodit",
       )
       return nu_grid

   def make_base_opa(nu_grid, patch_min, patch_max):
       mdb = MdbExomol(
           ".database/H2O/1H2-16O/POKAZATEL",
           nurange=nu_grid,
           gpu_transfer=False,
           local_databases=".",
           broadf_download=False,
       )
       return OpaPremodit(
           mdb,
           nu_grid,
           auto_trange=[500.0, 2000.0],
           allow_32bit=True,
       )

   manifest = precompute_ckd_tables_by_patches(
       make_base_opa,
       make_nu_grid,
       nu_min=200.0,
       nu_max=33293.0,
       patch_width=100.0,
       T_grid=T_grid,
       P_grid=P_grid,
       out_dir="ckd_h2o_R1000_patches",
       Ng=16,
       ckd_resolution=1000.0,
       nu_grid_points_per_patch=8000,
       overwrite=True,
   )

Keep ``patch_width`` and ``nu_grid_points_per_patch`` small enough for
the available host and device memory. Larger temperature and pressure
grids increase memory use inside each patch.

.. code:: ipython3

    shutil.rmtree(out_dir)
    print(f"removed {out_dir}")

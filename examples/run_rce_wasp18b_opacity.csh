#!/bin/csh -f
# Download line lists and prepare matched CKD/resolved opacities on one GPU.
# Run from the repository root. Re-running resumes verified completed chunks.
# Usage: csh -f examples/run_rce_wasp18b_opacity.csh [output-dir] [python] [dry-run]
if ($#argv > 3) then
    echo "Usage: csh -f examples/run_rce_wasp18b_opacity.csh [output-dir] [python] [dry-run]"
    exit 2
endif
if (! -f examples/prepare_rce_wasp18b_opacity.py) then
    echo "Run this script from the ExoJAX repository root."
    exit 2
endif

set database_root = "$HOME/.database"
if ($?RCE_DATABASE_ROOT) then
    set database_root = "$RCE_DATABASE_ROOT"
endif
set inventory = "$database_root/rce_wasp18b/local_inventory.json"
if ($?RCE_INVENTORY) then
    set inventory = "$RCE_INVENTORY"
endif
set output_dir = "$database_root/rce_wasp18b/prepared"
if ($#argv >= 1) then
    set output_dir = "$argv[1]"
endif
set python_executable = "python"
if ($#argv >= 2) then
    set python_executable = "$argv[2]"
endif
set dry_run = 0
if ($#argv == 3) then
    if ("$argv[3]" != "dry-run") then
        echo "The third argument must be dry-run."
        exit 2
    endif
    set dry_run = 1
endif
if (! -f "$inventory") then
    echo "Missing input inventory: $inventory"
    echo "Set RCE_INVENTORY to an existing WASP-18b asset manifest."
    exit 2
endif

# NH3/CH4 lists in the original inventory lack high-temperature completeness.
# These three species are a preparation baseline; additional absorbers require
# their own data and validation. This job does not certify the physical model.
set species = (H2O1 C1O1 C1O2)
if ($?PYTHONPATH) then
    setenv PYTHONPATH "$cwd/src:$PYTHONPATH"
else
    setenv PYTHONPATH "$cwd/src"
endif
setenv JAX_ENABLE_X64 True
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv NUMBA_DISABLE_JIT 1
setenv MPLBACKEND Agg
"$python_executable" -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 is required"'
if ($status != 0) exit 2

if ($dry_run == 1) then
    foreach molecule ($species)
        "$python_executable" examples/prepare_rce_wasp18b_opacity.py \
            --database-root "$database_root" --output-dir "$output_dir" \
            --inventory "$inventory" --species "$molecule" --dry-run
        if ($status != 0) exit 1
        if ("$molecule" == "C1O2") then
            "$python_executable" examples/prepare_rce_wasp18b_line_cache.py \
                --database-root "$database_root" --species "$molecule" \
                --rows-per-chunk 1000000 --dry-run
            if ($status != 0) exit 1
        endif
    end
    "$python_executable" examples/prepare_rce_wasp18b_cia.py \
        --database-root "$database_root" --output-dir "$output_dir/high_temperature" --dry-run
    exit $status
endif

setenv JAX_PLATFORMS cuda
"$python_executable" -c 'import jax; print(jax.__version__, jax.devices()); assert jax.default_backend() == "gpu", "A CUDA JAX backend is required"'
if ($status != 0) exit 2
set run_stamp = `date +%Y%m%d_%H%M%S`
set log_dir = "$output_dir/logs/${run_stamp}_$$"
mkdir -p "$log_dir"
if ($status != 0) exit 2
echo "Raw data: $database_root"
echo "Prepared tables: $output_dir"
echo "Logs: $log_dir"

foreach molecule ($species)
    if ("$molecule" == "C1O2") then
        echo "Preparing missing CO2 line caches in bounded row batches."
        /usr/bin/time -p -o "$log_dir/${molecule}_line_cache.time" \
            /usr/bin/env JAX_PLATFORMS=cpu "$python_executable" -u \
            examples/prepare_rce_wasp18b_line_cache.py \
            --database-root "$database_root" --species "$molecule" \
            --rows-per-chunk 1000000 >& "$log_dir/${molecule}_line_cache.log"
        if ($status != 0) then
            tail -40 "$log_dir/${molecule}_line_cache.log"
            echo "Line-cache preparation failed; completed caches are retained for a rerun."
            exit 1
        endif
    endif
    echo "Preparing $molecule (downloads missing raw files; reuses verified chunks)."
    /usr/bin/time -p -o "$log_dir/${molecule}.time" \
        "$python_executable" -u examples/prepare_rce_wasp18b_opacity.py \
        --database-root "$database_root" --output-dir "$output_dir" \
        --inventory "$inventory" --species "$molecule" >& "$log_dir/${molecule}.log"
    if ($status != 0) then
        tail -40 "$log_dir/${molecule}.log"
        echo "Preparation failed; inspect the log and rerun the same command to resume."
        echo "An interrupted raw download/parser cache may require repairing that incomplete file first."
        exit 1
    endif
end

"$python_executable" -u examples/prepare_rce_wasp18b_opacity.py \
    --database-root "$database_root" --output-dir "$output_dir" \
    --inventory "$inventory" --assemble --assemble-species $species >& "$log_dir/assemble.log"
if ($status != 0) then
    tail -40 "$log_dir/assemble.log"
    exit 1
endif
"$python_executable" -u examples/prepare_rce_wasp18b_cia.py \
    --database-root "$database_root" --output-dir "$output_dir/high_temperature" \
    --manifest "$output_dir/prepared_manifest.json" >& "$log_dir/cia.log"
if ($status != 0) then
    tail -40 "$log_dir/cia.log"
    exit 1
endif
echo "Prepared opacity manifest: $output_dir/prepared_manifest.json"
echo "Stellar coverage and opacity/grid validation remain separate from table preparation."

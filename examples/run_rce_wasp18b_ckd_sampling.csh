#!/bin/csh -f
# Share native teacher evaluations with the frozen CKD refinement driver.
# Usage: csh -f examples/run_rce_wasp18b_ckd_sampling.csh [output-dir] [python] [dry-run]
if ($#argv > 3 || ! -f examples/prepare_rce_wasp18b_ckd_sampling.py) then
    echo "Run from the repository root: csh -f examples/run_rce_wasp18b_ckd_sampling.csh [output-dir] [python] [dry-run]"
    exit 2
endif
set database_root = "$HOME/.database"
if ($?RCE_DATABASE_ROOT) then
    set database_root = "$RCE_DATABASE_ROOT"
endif
set base_preparation = "$database_root/rce_wasp18b/prepared/H2O1/complete.json"
if ($?RCE_REFINEMENT_BASE_PREPARATION) then
    set base_preparation = "$RCE_REFINEMENT_BASE_PREPARATION"
endif
set refinement_dir = "$database_root/rce_wasp18b/prepared/ckd_refinement"
if ($?RCE_REFINEMENT_DIR) then
    set refinement_dir = "$RCE_REFINEMENT_DIR"
endif
set output_dir = "$database_root/rce_wasp18b/prepared/ckd_sampling"
if ($#argv >= 1) then
    set output_dir = "$argv[1]"
endif
set python_executable = "python"
if ($#argv >= 2) then
    set python_executable = "$argv[2]"
endif
set mode = ()
if ($#argv == 3) then
    if ("$argv[3]" != "dry-run") then
        echo "The third argument must be dry-run."
        exit 2
    endif
    set mode = (--dry-run)
endif
set species = (H2O1 C1O1 C1O2)
if ($?RCE_REFINEMENT_SPECIES) then
    set species = ($RCE_REFINEMENT_SPECIES)
endif
if ($#species == 0) exit 2
setenv JAX_ENABLE_X64 True
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv NUMBA_DISABLE_JIT 1
setenv MPLBACKEND Agg
if ($?PYTHONPATH) then
    setenv PYTHONPATH "$cwd/src:$PYTHONPATH"
else
    setenv PYTHONPATH "$cwd/src"
endif
if ($#mode != 0) then
    foreach absorber ($species)
        "$python_executable" examples/prepare_rce_wasp18b_ckd_sampling.py \
            --base-preparation "$base_preparation" --refinement-dir "$refinement_dir" \
            --output-dir "$output_dir" --species "$absorber" $mode
        if ($status != 0) exit 1
    end
    exit 0
endif
setenv JAX_PLATFORMS cuda
"$python_executable" -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
if ($status != 0) exit 2
set run_stamp = `date +%Y%m%d_%H%M%S`
set log_dir = "$output_dir/logs/${run_stamp}_sampling_$$"
mkdir -p "$log_dir"
if ($status != 0) exit 2
echo "Logs: $log_dir"
foreach absorber ($species)
    set available_kib = `df -Pk "$output_dir" | awk 'NR==2 {print $4}'`
    if ($available_kib < 14680064) then
        echo "Less than 14 GiB free before $absorber; completed files retained."
        exit 1
    endif
    echo "Preparing ${absorber}: missing refinements first, then remaining sampling chunks."
    /usr/bin/time -v -o "$log_dir/${absorber}.time" \
        "$python_executable" -u examples/prepare_rce_wasp18b_ckd_sampling.py \
        --base-preparation "$base_preparation" --refinement-dir "$refinement_dir" \
        --output-dir "$output_dir" --species "$absorber" >& "$log_dir/${absorber}.log"
    if ($status != 0) then
        tail -30 "$log_dir/${absorber}.log"
        echo "Preparation stopped; rerun the same command to resume."
        exit 1
    endif
end
echo "Sampling tables complete. Assemble both variants with the unchanged baseline LBL."

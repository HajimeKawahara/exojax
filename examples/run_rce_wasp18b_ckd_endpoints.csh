#!/bin/csh -f
# Extend only the CKD spectral endpoints, resuming verified chunks.
# Usage: csh -f examples/run_rce_wasp18b_ckd_endpoints.csh [output-dir] [python] [dry-run]
if ($#argv > 3 || ! -f examples/prepare_rce_wasp18b_ckd_endpoints.py) exit 2
set database_root = "$HOME/.database"
if ($?RCE_DATABASE_ROOT) then
    set database_root = "$RCE_DATABASE_ROOT"
endif
set prepared = "$database_root/rce_wasp18b/prepared"
set output_dir = "$prepared/ckd_endpoints"
if ($#argv >= 1) then
    set output_dir = "$argv[1]"
endif
set python_executable = "python"
if ($#argv >= 2) then
    set python_executable = "$argv[2]"
endif
set dry_run = 0
if ($#argv == 3) then
    if ("$argv[3]" != "dry-run") exit 2
    set dry_run = 1
endif
set species = (H2O1 C1O1 C1O2)
if ($?RCE_ENDPOINT_SPECIES) then
    set species = ($RCE_ENDPOINT_SPECIES)
endif
if ($#species == 0) exit 2
set end_args = ()
if ($?RCE_ENDPOINT_END) then
    if ("$RCE_ENDPOINT_END" != "lower" && "$RCE_ENDPOINT_END" != "upper") exit 2
    set end_args = (--end "$RCE_ENDPOINT_END")
endif
setenv JAX_ENABLE_X64 True
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv NUMBA_DISABLE_JIT 1
setenv MPLBACKEND Agg
if (! $?JAX_PLATFORMS) setenv JAX_PLATFORMS cuda
if ($dry_run == 0) then
    set run_stamp = `date +%Y%m%d_%H%M%S`
    set log_dir = "$output_dir/logs/${run_stamp}_endpoints_$$"
    mkdir -p "$log_dir"
    if ($status != 0) exit 2
    echo "Logs: $log_dir"
endif
foreach absorber ($species)
    if ($dry_run == 1) then
        "$python_executable" examples/prepare_rce_wasp18b_ckd_endpoints.py \
            --base-preparation "$prepared/H2O1/complete.json" --species "$absorber" \
            --output-dir "$output_dir" $end_args --dry-run
        if ($status != 0) exit 1
    else
        set available_kib = `df -Pk "$output_dir" | awk 'NR==2 {print $4}'`
        if ($available_kib < 14680064) then
            echo "Less than 14 GiB free before $absorber; completed files retained."
            exit 1
        endif
        echo "Extending $absorber."
        /usr/bin/time -v -o "$log_dir/${absorber}.time" \
            "$python_executable" -u examples/prepare_rce_wasp18b_ckd_endpoints.py \
            --base-preparation "$prepared/H2O1/complete.json" --species "$absorber" \
            --output-dir "$output_dir" $end_args >& "$log_dir/${absorber}.log"
        if ($status != 0) then
            tail -30 "$log_dir/${absorber}.log"
            echo "Endpoint preparation stopped; rerun the same command to resume."
            exit 1
        endif
    endif
end
echo "Endpoint preparation finished; manifest assembly and RCE comparison remain."

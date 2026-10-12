#!/bin/csh -f
# Prepare the additional heating absorbers on the pinned P7 grids.
# Run from the repository root; each species resumes verified chunks.
# Usage: csh -f examples/run_rce_wasp18b_additional_opacity.csh [prepared-dir] [python] [dry-run]
if ($#argv > 3) then
    echo "Usage: csh -f examples/run_rce_wasp18b_additional_opacity.csh [prepared-dir] [python] [dry-run]"
    exit 2
endif
if (! -f examples/prepare_rce_wasp18b_atomic.py) then
    echo "Run from the ExoJAX repository root."
    exit 2
endif
set database_root = "$HOME/.database"
if ($?RCE_DATABASE_ROOT) then
    set database_root = "$RCE_DATABASE_ROOT"
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
set species = (Fe1 O1Ti1 O1Si1 Na1 K1)
if ($?RCE_ADDITIONAL_SPECIES) then
    set species = ($RCE_ADDITIONAL_SPECIES)
endif
setenv JAX_ENABLE_X64 True
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv NUMBA_DISABLE_JIT 1
setenv MPLBACKEND Agg
if ($?PYTHONPATH) then
    setenv PYTHONPATH "$cwd/src:$PYTHONPATH"
else
    setenv PYTHONPATH "$cwd/src"
endif
if ($dry_run == 0) then
    setenv JAX_PLATFORMS cuda
    "$python_executable" -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
    if ($status != 0) exit 2
    set run_stamp = `date +%Y%m%d_%H%M%S`
    set log_dir = "$output_dir/logs/${run_stamp}_additional_$$"
    mkdir -p "$log_dir"
    if ($status != 0) exit 2
    echo "Logs: $log_dir"
endif
foreach absorber ($species)
    switch ($absorber)
    case Fe1:
    case Na1:
    case K1:
        set driver = examples/prepare_rce_wasp18b_atomic.py
        set destination = "$output_dir/atomic_jit"
        set molecular = 0
        breaksw
    case H1O1:
    case Fe1H1:
    case O1Ti1:
    case O1Si1:
    case O1V1:
        set driver = examples/prepare_rce_wasp18b_supplemental.py
        set destination = "$output_dir/supplemental"
        set molecular = 1
        breaksw
    default:
        echo "Unsupported additional absorber: $absorber"
        exit 2
    endsw
    if ($dry_run == 1) then
        "$python_executable" "$driver" --database-root "$database_root" \
            --output-dir "$destination" --species "$absorber" --dry-run
        if ($status != 0) exit 1
    else
        # A full resolved table plus its retained chunks needs about 12 GiB.
        set available_kib = `df -Pk "$output_dir" | awk 'NR==2 {print $4}'`
        if ($available_kib < 14680064) then
            echo "Less than 14 GiB free before $absorber; completed files retained."
            exit 1
        endif
        if ($molecular == 1) then
            echo "Preparing searchable line caches for $absorber."
            /usr/bin/env JAX_PLATFORMS=cpu "$python_executable" -u \
                examples/prepare_rce_wasp18b_line_cache.py \
                --database-root "$database_root" --species "$absorber" \
                --nu-min 175 --nu-max 50025 --rows-per-chunk 1000000 \
                >& "$log_dir/${absorber}_line_cache.log"
            if ($status != 0) exit 1
        endif
        echo "Preparing $absorber."
        /usr/bin/time -v -o "$log_dir/${absorber}.time" \
            "$python_executable" -u "$driver" --database-root "$database_root" \
            --output-dir "$destination" --species "$absorber" \
            >& "$log_dir/${absorber}.log"
        if ($status != 0) then
            tail -30 "$log_dir/${absorber}.log"
            echo "Preparation stopped; rerun the same job to resume."
            exit 1
        endif
    endif
end
if ($dry_run == 1) then
    echo "Dry-run plans checked; no tables written."
else
    echo "Additional tables prepared; combined-manifest and physical convergence checks remain."
endif

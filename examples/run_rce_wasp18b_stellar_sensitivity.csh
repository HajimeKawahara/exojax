#!/bin/csh -f
# Run two independently reconverged stellar-tail variants against native32/R700.
# Usage: csh -f examples/run_rce_wasp18b_stellar_sensitivity.csh [output-dir] [python] [dry-run]
if ($#argv > 3) then
    echo "Usage: run_rce_wasp18b_stellar_sensitivity.csh [output-dir] [python] [dry-run]"
    exit 2
endif
set input_dir = "$cwd/results/rce_wasp18b_stellar_sensitivity"
if ($?RCE_STELLAR_SENSITIVITY_INPUT_DIR) then
    set input_dir = "$RCE_STELLAR_SENSITIVITY_INPUT_DIR"
endif
set forward_runner = "examples/rce_wasp18b_forward.py"
if (-f "$input_dir/forward_runner.py") then
    set forward_runner = "$input_dir/forward_runner.py"
endif
set output_dir = "$input_dir/runs"
set python_executable = "python"
if ($#argv >= 1) then
    set output_dir = "$argv[1]"
endif
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
if (! -f "$input_dir/preparation.json" || ! -f examples/rce_wasp18b_forward.py) then
    echo "Run from the repository root after preparing the fixed-seed stellar manifests."
    exit 2
endif
set baseline = "$input_dir/baseline_native32.json"
set observation = "/home/kawahara/ExoSpecDB/production/wasp18b-soss-emission-spatial-gls-0.1.0-r1/emission_candidate.ecsv"
# Verify the small immutable run inputs; the forward runner verifies all assets.
"$python_executable" -c 'import hashlib,json,sys; from pathlib import Path; p=json.loads(Path(sys.argv[1]).read_text()); entries=[p["baseline"]]+[x["manifest"] for x in p["cases"]]; assert all(hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"] for x in entries); b=json.loads(Path(p["baseline"]["path"]).read_text()); assert b["success"] and b["state"]["converged"] and b["response_resolution"]==700.' "$input_dir/preparation.json"
if ($status != 0) exit 2
if ($dry_run == 0) then
    setenv JAX_ENABLE_X64 True
    setenv JAX_PLATFORMS cuda
    setenv XLA_PYTHON_CLIENT_PREALLOCATE false
    setenv NUMBA_DISABLE_JIT 1
    setenv MPLBACKEND Agg
    if ($?PYTHONPATH) then
        setenv PYTHONPATH "$cwd/src:$PYTHONPATH"
    else
        setenv PYTHONPATH "$cwd/src"
    endif
    "$python_executable" -c 'import jax; print(jax.devices()); assert jax.default_backend()=="gpu"'
    if ($status != 0) exit 2
    mkdir -p "$output_dir"
    if ($status != 0) exit 2
endif
foreach variant (stellar_window_4_5 stellar_temperature_6300)
    set manifest = "$input_dir/${variant}_manifest.json"
    set output = "$output_dir/${variant}_a100.json"
    if ($dry_run == 1) then
        echo "$python_executable -u $forward_runner --manifest $manifest --observation $observation --output $output --stream-resolved --mask-continuation --response-resolution 700 --compare $baseline --comparison-kind physics"
    else
        if (-e "$output") then
            echo "Existing result $output; select a new output directory."
            exit 2
        endif
        /usr/bin/time -v -o "$output_dir/${variant}.time" \
            "$python_executable" -u "$forward_runner" \
            --manifest "$manifest" --observation "$observation" --output "$output" \
            --stream-resolved --mask-continuation --response-resolution 700 \
            --compare "$baseline" --comparison-kind physics >& "$output_dir/${variant}.log"
        if ($status != 0) then
            tail -20 "$output_dir/${variant}.log"
            exit 1
        endif
        echo "Saved comparison: $output"
    endif
end

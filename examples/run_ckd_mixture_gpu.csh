#!/bin/csh -f
# Run from an environment with CUDA JAX, NumPyro, ArviZ and ExoJAX dependencies.
# Usage: csh -f examples/run_ckd_mixture_gpu.csh [new-output-directory] [python] [diagnostic]
if ($#argv > 3) then
    echo "Usage: csh -f examples/run_ckd_mixture_gpu.csh [new-output-directory] [python] [diagnostic]"
    exit 2
endif
set output = "$cwd/output/ckd_mixture_`date +%Y%m%d_%H%M%S`"
if ($#argv >= 1) then
    set output = "$argv[1]"
endif
set python = "python"
if ($#argv >= 2) then
    set python = "$argv[2]"
endif
set diagnostic = 0
set run_options = ()
if ($#argv >= 3) then
    if ("$argv[3]" != "diagnostic") then
        echo "The optional third argument must be diagnostic."
        exit 2
    endif
    set diagnostic = 1
    set run_options = (--diagnostic)
endif
if (! -f examples/ckd_mixture_retrieval.py) then
    echo "Run this script from the ExoJAX repository root."
    exit 2
endif
set logs = "${output}.logs"
if (-e "$output" || -e "$logs") then
    echo "Choose a new output directory; existing case/run evidence is preserved: $output"
    exit 2
endif
mkdir -p "$logs"
if ($status != 0) exit 2
setenv PYTHONPATH "$cwd/src"
setenv JAX_PLATFORMS cuda
setenv JAX_ENABLE_X64 True
setenv JAX_ENABLE_COMPILATION_CACHE false
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv NUMBA_DISABLE_JIT 1
setenv MPLBACKEND Agg
echo "Saving case: $output"
echo "Saving logs and whole-process timings: $logs"
/usr/bin/time -p -o "$logs/prepare.time" "$python" -u examples/ckd_mixture_validation.py prepare --output-dir "$output" >& "$logs/prepare.log"
if ($status != 0) then
    tail -40 "$logs/prepare.log"
    exit 1
endif
/usr/bin/time -p -o "$logs/validate.time" "$python" -u examples/ckd_mixture_validation.py validate --output-dir "$output" --validation-id gpu --methods lbl rorr >& "$logs/validate.log"
if ($status != 0) then
    tail -40 "$logs/validate.log"
    if ($diagnostic == 0) then
        echo "Validation failed; inspect the saved report before starting retrieval."
        exit 1
    endif
    "$python" -c 'import json,sys; v=json.load(open(sys.argv[1])); ok=v["status"]=="completed" and v["reference"]["passed"] and all(v["methods"][m]["gradient_passed"] for m in ("lbl","rorr")); sys.exit(0 if ok else 1)' "$output/validations/gpu/validation.json"
    if ($status != 0) then
        echo "Diagnostic retrieval also requires successful reference and gradient validation."
        exit 1
    endif
    echo "DIAGNOSTIC ONLY: spectral accuracy failed; these runs cannot establish scientific speedup."
endif
foreach seed (41 42)
    set methods = (lbl rorr)
    if ($seed == 42) set methods = (rorr lbl)
    foreach method ($methods)
        set label = "seed${seed}_${method}"
        echo "Running $label (four sequential chains, 500 warmup, 1000 retained draws)."
        /usr/bin/time -p -o "$logs/${label}.time" "$python" -u examples/ckd_mixture_retrieval.py run --output-dir "$output" --validation-id gpu --run-id "seed${seed}" --method "$method" --seed "$seed" --num-chains 4 --num-warmup 500 --num-samples 1000 $run_options >& "$logs/${label}.log"
        if ($status != 0) then
            tail -40 "$logs/${label}.log"
            exit 1
        endif
    end
end
"$python" -u examples/ckd_mixture_retrieval.py summarize --output-dir "$output" --run-id seed41 --repeat-run-id seed42 --methods lbl rorr >& "$logs/summary.log"
if ($status != 0) then
    tail -40 "$logs/summary.log"
    exit 1
endif
cat "$logs/summary.log"
if ($diagnostic == 1) then
    echo "Diagnostic retrieval completed: $output/runs/seed41/comparison.json"
    echo "Review posterior convergence and bias; this is excluded from scientific performance comparisons."
    exit 0
endif
"$python" -c 'import json,sys; result=json.load(open(sys.argv[1])); sys.exit(0 if result["quality"]["eligible"] else 3)' "$output/runs/seed41/comparison.json"
if ($status != 0) then
    echo "Sampling finished, but this comparison did not pass the scientific quality gates."
    exit 3
endif
echo "Completed: $output/runs/seed41/comparison.json"

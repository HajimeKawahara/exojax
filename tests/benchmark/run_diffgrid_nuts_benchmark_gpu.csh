#!/bin/tcsh

# Run this script from the ExoJAX repository root.  Preparation artifacts are
# reused when present; choose a new output directory to rebuild them.  Use a
# new EXOJAX_BENCHMARK_RUN_PREFIX to retain results from another pair of runs.

set SCRIPT = "tests/benchmark/diffgrid_nuts_benchmark.py"
set OUTPUT_DIR = "tests/benchmark/output_diffgrid_nuts"
set MDB_PATH = ".database/CH4/12C-1H4/YT10to10"
set CIA_PATH = ".database/H2-H2_2011.cia"
set NUM_WARMUP = 500
set NUM_SAMPLES = 1000
set RUN_PREFIX = "baseline"
set SEED_0 = 0
set SEED_1 = 1
set PYTHON = "python"

if ( $#argv >= 1 ) then
  set OUTPUT_DIR = "$argv[1]"
endif
if ( $?EXOJAX_CH4_MDB_PATH ) then
  set MDB_PATH = "$EXOJAX_CH4_MDB_PATH"
endif
if ( $?EXOJAX_H2H2_CIA_PATH ) then
  set CIA_PATH = "$EXOJAX_H2H2_CIA_PATH"
endif
if ( $?EXOJAX_BENCHMARK_PYTHON ) then
  set PYTHON = "$EXOJAX_BENCHMARK_PYTHON"
endif
if ( $?EXOJAX_BENCHMARK_RUN_PREFIX ) then
  set RUN_PREFIX = "$EXOJAX_BENCHMARK_RUN_PREFIX"
endif
if ( $?EXOJAX_BENCHMARK_SEED_0 ) then
  set SEED_0 = "$EXOJAX_BENCHMARK_SEED_0"
endif
if ( $?EXOJAX_BENCHMARK_SEED_1 ) then
  set SEED_1 = "$EXOJAX_BENCHMARK_SEED_1"
endif
if ( "$SEED_0" == "$SEED_1" ) then
  echo "The two independent runs require different seeds."
  exit 2
endif
set VALIDATION_ID = "${RUN_PREFIX}-accuracy"
set RUN_IDS = ("${RUN_PREFIX}-0" "${RUN_PREFIX}-1")
set RUN_SEEDS = ("$SEED_0" "$SEED_1")
set PROCESS_TIMES_DIR = "$OUTPUT_DIR/process_times"

if ( ! -e "$SCRIPT" ) then
  echo "Run this launcher from the ExoJAX repository root."
  exit 2
endif
if ( -e "$OUTPUT_DIR/validations/$VALIDATION_ID" ) then
  echo "Validation already exists; choose a new EXOJAX_BENCHMARK_RUN_PREFIX."
  exit 2
endif
foreach RUN_ID ($RUN_IDS)
  if ( -e "$OUTPUT_DIR/runs/$RUN_ID" ) then
    echo "Run $RUN_ID already exists; choose a new EXOJAX_BENCHMARK_RUN_PREFIX."
    exit 2
  endif
end
foreach TIME_ID ("${RUN_PREFIX}-prepare" "${RUN_PREFIX}-validation" "${RUN_IDS[1]}-premodit" "${RUN_IDS[1]}-diffgrid" "${RUN_IDS[2]}-premodit" "${RUN_IDS[2]}-diffgrid")
  if ( -e "$PROCESS_TIMES_DIR/$TIME_ID.txt" ) then
    echo "Process timing already exists; choose a new EXOJAX_BENCHMARK_RUN_PREFIX."
    exit 2
  endif
end

setenv JAX_PLATFORMS cuda
setenv JAX_PLATFORM_NAME cuda
setenv JAX_ENABLE_X64 True
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv MPLCONFIGDIR /tmp/exojax_diffgrid_nuts_mpl
setenv NUMBA_DISABLE_JIT 1
setenv PYTHONUNBUFFERED 1
if ( $?JAX_COMPILATION_CACHE_DIR ) then
  unsetenv JAX_COMPILATION_CACHE_DIR
endif
mkdir -p "$MPLCONFIGDIR" "$PROCESS_TIMES_DIR"
if ( $status != 0 ) then
  echo "Cannot create benchmark output directories."
  exit 1
endif

set PREPARED = 0
if ( -e "$OUTPUT_DIR/prepare.json" ) then
  if ( -e "$OUTPUT_DIR/case.npz" ) then
    if ( -e "$OUTPUT_DIR/premodit.npz" && -e "$OUTPUT_DIR/diffgrid.npz" ) then
      if ( -e "$OUTPUT_DIR/premodit_metadata.json" && -e "$OUTPUT_DIR/diffgrid_metadata.json" ) then
        set PREPARED = 1
      endif
    endif
  endif
endif

if ( $PREPARED == 0 ) then
  echo "Preparing shared opacity artifacts in $OUTPUT_DIR"
  /usr/bin/time -p -o "$PROCESS_TIMES_DIR/${RUN_PREFIX}-prepare.txt" "$PYTHON" "$SCRIPT" prepare \
    --output-dir "$OUTPUT_DIR" \
    --mdb-path "$MDB_PATH" \
    --cia-path "$CIA_PATH" \
    --overwrite
  if ( $status != 0 ) then
    echo "Benchmark preparation failed."
    exit 1
  endif
else
  echo "Reusing preparation artifacts in $OUTPUT_DIR"
endif

echo "Validating observation-space accuracy and gradients"
/usr/bin/time -p -o "$PROCESS_TIMES_DIR/${RUN_PREFIX}-validation.txt" "$PYTHON" "$SCRIPT" validate \
  --output-dir "$OUTPUT_DIR" \
  --validation-id "$VALIDATION_ID"
if ( $status != 0 ) then
  echo "Benchmark validation failed; no sampling runs were started."
  exit 1
endif

foreach REPEAT (1 2)
  set RUN_ID = "$RUN_IDS[$REPEAT]"
  foreach METHOD (premodit diffgrid)
    echo "Running $METHOD, $RUN_IDS[$REPEAT], in a fresh Python process"
    /usr/bin/time -p -o "$PROCESS_TIMES_DIR/${RUN_ID}-${METHOD}.txt" "$PYTHON" "$SCRIPT" run \
      --method "$METHOD" \
      --output-dir "$OUTPUT_DIR" \
      --run-id "$RUN_IDS[$REPEAT]" \
      --validation-id "$VALIDATION_ID" \
      --num-chains 4 \
      --chain-method sequential \
      --initialization prior \
      --initialization-seed "$RUN_SEEDS[$REPEAT]" \
      --seed "$RUN_SEEDS[$REPEAT]" \
      --measure-steady-sampling \
      --num-warmup $NUM_WARMUP \
      --num-samples $NUM_SAMPLES
    if ( $status != 0 ) then
      echo "$METHOD benchmark failed for $RUN_IDS[$REPEAT]."
      exit 1
    endif
  end
end

"$PYTHON" "$SCRIPT" summarize \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_IDS[1]" \
  --repeat-run-id "$RUN_IDS[2]" \
  --validation-id "$VALIDATION_ID"
if ( $status != 0 ) then
  echo "Benchmark summary failed."
  exit 1
endif

echo "Benchmark complete: $OUTPUT_DIR/runs/$RUN_IDS[1]"

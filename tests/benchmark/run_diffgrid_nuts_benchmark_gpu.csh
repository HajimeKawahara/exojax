#!/bin/tcsh

# Run this script from the ExoJAX repository root.  Preparation artifacts are
# reused when present; choose a new output directory to rebuild them.

set SCRIPT = "tests/benchmark/diffgrid_nuts_benchmark.py"
set OUTPUT_DIR = "tests/benchmark/output_diffgrid_nuts"
set MDB_PATH = ".database/CH4/12C-1H4/YT10to10"
set CIA_PATH = ".database/H2-H2_2011.cia"
set NUM_WARMUP = 500
set NUM_SAMPLES = 1000
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

if ( ! -e "$SCRIPT" ) then
  echo "Run this launcher from the ExoJAX repository root."
  exit 2
endif

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
mkdir -p "$MPLCONFIGDIR"

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
  /usr/bin/time -p $PYTHON $SCRIPT prepare \
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

foreach METHOD (premodit diffgrid)
  echo "Running $METHOD in a fresh Python process"
  /usr/bin/time -p $PYTHON $SCRIPT run \
    --method "$METHOD" \
    --output-dir "$OUTPUT_DIR" \
    --num-warmup $NUM_WARMUP \
    --num-samples $NUM_SAMPLES
  if ( $status != 0 ) then
    echo "$METHOD benchmark failed."
    exit 1
  endif
end

$PYTHON $SCRIPT summarize --output-dir "$OUTPUT_DIR"
if ( $status != 0 ) then
  echo "Benchmark summary failed."
  exit 1
endif

echo "Benchmark complete: $OUTPUT_DIR/comparison.png"

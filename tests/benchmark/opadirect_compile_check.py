"""Manual OpaDirect compile-stability check.

This script is for observation only. It does not change ExoJAX behavior or
apply any optimization. It exercises the real ``OpaDirect.xsmatrix`` path and
reports timings that make JAX compilation and recompilation easier to inspect.

Useful JAX settings when running this script manually:

- ``JAX_LOG_COMPILES=1``: print when JAX/XLA compilation happens
- ``JAX_EXPLAIN_CACHE_MISSES=1``: explain why JIT cache misses occur
- ``JAX_TRACEBACK_FILTERING=off``: keep full stack traces in compile logs

Signs that recompilation or retracing is happening:

- ``JAX_LOG_COMPILES=1`` prints a new compile on the first call
- repeated same-shape calls should usually avoid new compile messages
- changing the layer count often triggers a fresh compile for shape-specialized
  code paths
- changing dtype may trigger a fresh compile if x64 is enabled and the input
  dtype actually changes
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

from jax import config
import jax

if "--enable-x64" in sys.argv:
    config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from exojax.opacity import OpaDirect
from exojax.test.emulate_mdb import mock_mdbExomol
from exojax.test.emulate_mdb import mock_mdbHitemp
from exojax.test.emulate_mdb import mock_wavenumber_grid


COLD_WARM_RATIO_THRESHOLD = 1.5
COLD_WARM_ABS_THRESHOLD_S = 0.05
SAME_SHAPE_MEAN_RATIO_THRESHOLD = 1.25
SAME_SHAPE_MAX_RATIO_THRESHOLD = 1.5
SAME_SHAPE_ABS_THRESHOLD_S = 0.02
RECOMPILE_RATIO_THRESHOLD = 1.5
RECOMPILE_ABS_THRESHOLD_S = 0.05
JSON_BEGIN_MARKER = "EXOJAX_JSON_RESULT_BEGIN"
JSON_END_MARKER = "EXOJAX_JSON_RESULT_END"
PHASE_BEGIN_MARKER = "EXOJAX_PHASE_BEGIN:"
HELPER_NAMES = [
    "gamma_exomol",
    "gamma_hitran",
    "line_strength",
    "doppler_sigma",
    "gamma_natural",
]
STRONG_SUSPICIOUS_PATTERNS = {
    "being_redefined_repeatedly": "being re-defined repeatedly",
    "same_line_redefinition": "seen another function defined on the same line",
    "preventing_caching": "preventing caching",
}


@dataclass
class TimingResult:
    database: str
    jax_enable_x64: bool
    nlayer_base: int
    nlayer_changed: int
    input_dtype: str
    changed_dtype: str | None
    output_shape_base: tuple[int, int]
    output_shape_changed: tuple[int, int]
    cold_call_s: float
    warm_call_s: float
    repeated_same_shape_s: list[float]
    changed_shape_call_s: float
    dtype_changed_call_s: float | None


@dataclass
class DerivedMetrics:
    repeated_same_shape_mean_s: float | None
    repeated_same_shape_max_s: float | None
    cold_to_warm_ratio: float | None
    repeated_mean_to_warm_ratio: float | None
    repeated_max_to_warm_ratio: float | None
    changed_shape_to_warm_ratio: float | None
    dtype_changed_to_warm_ratio: float | None


@dataclass
class HelperSignalSummary:
    helper_log_capture_enabled: bool
    helper_log_capture_succeeded: bool
    same_shape_repeat_phase_count: int
    captured_log_line_count: int
    relevant_log_line_count: int
    suspicious_helper_signal_detected: str
    suspicious_helper_signal_count: int
    suspicious_helper_names: list[str]
    suspicious_log_patterns_found: list[str]
    same_shape_inner_helper_stability: str
    helper_signal_category: str
    helper_patterns_used: list[str]
    log_capture_note: str | None


@dataclass
class Interpretation:
    cold_to_warm_drop_detected: str
    same_shape_top_level_timing_appears_stable: str
    shape_change_recompilation_likely: str
    dtype_change_recompilation_likely: str
    same_shape_top_level_category: str
    shape_change_category: str
    dtype_change_category: str
    thresholds: dict[str, float]
    log_hint: str | None
    conclusion: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual compile-stability check for ExoJAX OpaDirect.xsmatrix."
    )
    parser.add_argument(
        "--database",
        choices=("exomol", "hitemp"),
        default="exomol",
        help="Mock database backend to use for the real OpaDirect path.",
    )
    parser.add_argument(
        "--nlayer",
        type=int,
        default=64,
        help="Base number of layers for same-shape timing calls.",
    )
    parser.add_argument(
        "--changed-nlayer",
        type=int,
        default=None,
        help="Layer count for the changed-shape call. Default: nlayer + 1.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of repeated same-shape warm calls to time.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Base dtype for Tarr/Parr inputs.",
    )
    parser.add_argument(
        "--dtype-change",
        choices=("none", "float32", "float64"),
        default="float64",
        help="Extra dtype case to probe after same-shape calls.",
    )
    parser.add_argument(
        "--nu-grid-size",
        type=int,
        default=20000,
        help="Wavenumber grid size used to build OpaDirect.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the normal human-readable summary and append JSON.",
    )
    parser.add_argument(
        "--quiet-json",
        action="store_true",
        help="Print only structured JSON output.",
    )
    parser.add_argument(
        "--enable-x64",
        action="store_true",
        help="Enable JAX float64 so dtype-change checks can use real fp64 inputs.",
    )
    parser.add_argument(
        "--capture-helper-signals",
        action="store_true",
        help="Run a child copy of this script with JAX compile/cache-miss logging and summarize suspicious helper-level same-shape signals.",
    )
    parser.add_argument(
        "--internal-child-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def build_mdb(database: str):
    if database == "exomol":
        return mock_mdbExomol()
    if database == "hitemp":
        return mock_mdbHitemp()
    raise ValueError(f"Unsupported database: {database}")


def make_profiles(nlayer: int, dtype_name: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    dtype = getattr(jnp, dtype_name)
    temperatures = jnp.asarray(np.linspace(700.0, 1400.0, nlayer), dtype=dtype)
    pressures = jnp.asarray(np.geomspace(1.0e-5, 1.0, nlayer), dtype=dtype)
    return temperatures, pressures


def phase_marker(phase_name: str) -> None:
    print(f"{PHASE_BEGIN_MARKER}{phase_name}", file=sys.stderr, flush=True)


def timed_xsmatrix(
    opa: OpaDirect,
    Tarr: jnp.ndarray,
    Parr: jnp.ndarray,
    phase_name: str | None = None,
) -> tuple[float, tuple[int, int]]:
    if phase_name is not None:
        phase_marker(phase_name)
    start = time.perf_counter()
    xsm = opa.xsmatrix(Tarr, Parr)
    xsm.block_until_ready()
    elapsed = time.perf_counter() - start
    return elapsed, tuple(int(v) for v in xsm.shape)


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def classify_cold_to_warm(cold_call_s: float, warm_call_s: float) -> str:
    ratio = safe_ratio(cold_call_s, warm_call_s)
    if ratio is None:
        return "unclear"
    if (
        ratio >= COLD_WARM_RATIO_THRESHOLD
        and (cold_call_s - warm_call_s) >= COLD_WARM_ABS_THRESHOLD_S
    ):
        return "yes"
    if ratio < 1.2:
        return "no"
    return "unclear"


def classify_same_shape_top_level(
    warm_call_s: float, repeated_same_shape_s: list[float]
) -> tuple[str, str]:
    if not repeated_same_shape_s:
        return "unclear", "Not enough evidence"
    repeated_mean = float(np.mean(repeated_same_shape_s))
    repeated_max = float(np.max(repeated_same_shape_s))
    mean_ratio = safe_ratio(repeated_mean, warm_call_s)
    max_ratio = safe_ratio(repeated_max, warm_call_s)
    if mean_ratio is None or max_ratio is None:
        return "unclear", "Not enough evidence"
    mean_limit = max(
        SAME_SHAPE_MEAN_RATIO_THRESHOLD,
        1.0 + SAME_SHAPE_ABS_THRESHOLD_S / max(warm_call_s, 1.0e-12),
    )
    max_limit = max(
        SAME_SHAPE_MAX_RATIO_THRESHOLD,
        1.0 + RECOMPILE_ABS_THRESHOLD_S / max(warm_call_s, 1.0e-12),
    )
    if mean_ratio <= mean_limit and max_ratio <= max_limit:
        return "yes", "Expected / normal"
    if mean_ratio >= SAME_SHAPE_MAX_RATIO_THRESHOLD or max_ratio >= (
        SAME_SHAPE_MAX_RATIO_THRESHOLD + 0.25
    ):
        return "no", "Suspicious / worth investigating"
    return "unclear", "Not enough evidence"


def classify_recompile_probe(probe_time_s: float | None, warm_call_s: float) -> str:
    if probe_time_s is None:
        return "unclear"
    ratio = safe_ratio(probe_time_s, warm_call_s)
    if ratio is None:
        return "unclear"
    if (
        ratio >= RECOMPILE_RATIO_THRESHOLD
        and (probe_time_s - warm_call_s) >= RECOMPILE_ABS_THRESHOLD_S
    ):
        return "yes"
    if ratio < 1.2:
        return "no"
    return "unclear"


def derive_metrics(result: TimingResult) -> DerivedMetrics:
    repeated_mean = None
    repeated_max = None
    if result.repeated_same_shape_s:
        repeated_mean = float(np.mean(result.repeated_same_shape_s))
        repeated_max = float(np.max(result.repeated_same_shape_s))
    return DerivedMetrics(
        repeated_same_shape_mean_s=repeated_mean,
        repeated_same_shape_max_s=repeated_max,
        cold_to_warm_ratio=safe_ratio(result.cold_call_s, result.warm_call_s),
        repeated_mean_to_warm_ratio=safe_ratio(repeated_mean, result.warm_call_s),
        repeated_max_to_warm_ratio=safe_ratio(repeated_max, result.warm_call_s),
        changed_shape_to_warm_ratio=safe_ratio(
            result.changed_shape_call_s, result.warm_call_s
        ),
        dtype_changed_to_warm_ratio=safe_ratio(
            result.dtype_changed_call_s, result.warm_call_s
        ),
    )


def build_log_hint(helper_summary: HelperSignalSummary | None) -> str | None:
    if helper_summary is not None and helper_summary.helper_log_capture_enabled:
        return (
            "Helper-log capture ran in a child process with JAX compile/cache-miss "
            "logging enabled. Suspicious same-shape helper signals are searched only "
            "inside repeated same-shape phases, not the initial cold call."
        )
    if os.environ.get("JAX_LOG_COMPILES") == "1":
        return (
            "JAX compile logging is enabled. For same-shape stability, scan for new "
            "compile lines after the first warm call. For changed-shape or changed-dtype "
            "probes, one new compile line is often expected."
        )
    return None


def build_interpretation(
    result: TimingResult,
    metrics: DerivedMetrics,
    helper_summary: HelperSignalSummary | None,
) -> Interpretation:
    cold_to_warm_drop_detected = classify_cold_to_warm(
        result.cold_call_s, result.warm_call_s
    )
    same_shape_top_level_timing_appears_stable, same_shape_top_level_category = (
        classify_same_shape_top_level(result.warm_call_s, result.repeated_same_shape_s)
    )
    shape_change_recompilation_likely = classify_recompile_probe(
        result.changed_shape_call_s, result.warm_call_s
    )
    dtype_change_recompilation_likely = classify_recompile_probe(
        result.dtype_changed_call_s, result.warm_call_s
    )

    if shape_change_recompilation_likely == "yes":
        shape_change_category = "Expected / normal"
    elif shape_change_recompilation_likely == "no":
        shape_change_category = "Expected / normal"
    else:
        shape_change_category = "Not enough evidence"

    if result.dtype_changed_call_s is None:
        dtype_change_category = "Not enough evidence"
    elif dtype_change_recompilation_likely == "yes":
        dtype_change_category = "Expected / normal"
    elif dtype_change_recompilation_likely == "no":
        dtype_change_category = "Expected / normal"
    else:
        dtype_change_category = "Not enough evidence"

    conclusion = []
    if cold_to_warm_drop_detected == "yes":
        conclusion.append(
            "The base-shape first call is much slower than the warm call, consistent with initial compilation."
        )
    elif cold_to_warm_drop_detected == "no":
        conclusion.append(
            "The base-shape first call is close to the warm call, so compilation cost is not strongly visible in this run."
        )
    else:
        conclusion.append(
            "Cold-versus-warm timing is ambiguous in this run, so initial compilation cost is not cleanly separated."
        )

    if same_shape_top_level_timing_appears_stable == "yes":
        conclusion.append(
            "Top-level repeated same-shape timings remain close to warm timing, which suggests stable top-level caching for the tested shape."
        )
    elif same_shape_top_level_timing_appears_stable == "no":
        conclusion.append(
            "Top-level repeated same-shape timings are noticeably slower than the warm call, which is suspicious and worth investigating."
        )
    else:
        conclusion.append(
            "Top-level same-shape timing stability is unclear from timing alone; use more repeats or helper-log capture for confirmation."
        )

    if helper_summary is None:
        conclusion.append(
            "Inner-helper stability was not measured in this run. Use --capture-helper-signals to inspect helper-level cache-miss or recompilation signals."
        )
    elif helper_summary.same_shape_inner_helper_stability == "stable":
        conclusion.append(
            "No suspicious helper-level same-shape signals were observed in the captured logs for this run."
        )
    elif helper_summary.same_shape_inner_helper_stability == "suspicious":
        conclusion.append(
            "Top-level timing may look stable, but suspicious helper-level same-shape signals were still observed in the captured logs."
        )
    else:
        conclusion.append(
            "Helper-level same-shape stability could not be determined confidently from the captured logs in this run."
        )

    if shape_change_recompilation_likely == "yes":
        conclusion.append(
            "The changed-shape call is much slower than the warm call, which is consistent with expected shape-triggered recompilation."
        )
    elif shape_change_recompilation_likely == "no":
        conclusion.append(
            "The changed-shape call is close to warm timing, so this run does not show a strong shape-change recompilation signal."
        )
    else:
        conclusion.append("Shape-change recompilation is unclear from timing alone.")

    if result.dtype_changed_call_s is None:
        conclusion.append(
            "The dtype-change probe was skipped or unavailable, so no dtype conclusion can be drawn from this run."
        )
    elif dtype_change_recompilation_likely == "yes":
        conclusion.append(
            "The changed-dtype call is much slower than the warm call, which is consistent with expected dtype-triggered recompilation."
        )
    elif dtype_change_recompilation_likely == "no":
        conclusion.append(
            "The changed-dtype call is close to warm timing, so this run does not show a strong dtype-change recompilation signal."
        )
    else:
        conclusion.append("Dtype-change recompilation is unclear from timing alone.")

    return Interpretation(
        cold_to_warm_drop_detected=cold_to_warm_drop_detected,
        same_shape_top_level_timing_appears_stable=same_shape_top_level_timing_appears_stable,
        shape_change_recompilation_likely=shape_change_recompilation_likely,
        dtype_change_recompilation_likely=dtype_change_recompilation_likely,
        same_shape_top_level_category=same_shape_top_level_category,
        shape_change_category=shape_change_category,
        dtype_change_category=dtype_change_category,
        thresholds={
            "cold_warm_ratio_threshold": COLD_WARM_RATIO_THRESHOLD,
            "cold_warm_abs_threshold_s": COLD_WARM_ABS_THRESHOLD_S,
            "same_shape_mean_ratio_threshold": SAME_SHAPE_MEAN_RATIO_THRESHOLD,
            "same_shape_max_ratio_threshold": SAME_SHAPE_MAX_RATIO_THRESHOLD,
            "same_shape_abs_threshold_s": SAME_SHAPE_ABS_THRESHOLD_S,
            "recompile_ratio_threshold": RECOMPILE_RATIO_THRESHOLD,
            "recompile_abs_threshold_s": RECOMPILE_ABS_THRESHOLD_S,
        },
        log_hint=build_log_hint(helper_summary),
        conclusion=conclusion,
    )


def serialize_payload(
    result: TimingResult,
    metrics: DerivedMetrics,
    interpretation: Interpretation,
    helper_summary: HelperSignalSummary | None,
) -> dict[str, object]:
    return {
        "timing": asdict(result),
        "derived_metrics": asdict(metrics),
        "helper_signal_summary": None if helper_summary is None else asdict(helper_summary),
        "interpretation": asdict(interpretation),
    }


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "skipped"
    return f"{value:.6f}"


def format_optional_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def print_human_summary(
    result: TimingResult,
    metrics: DerivedMetrics,
    interpretation: Interpretation,
    helper_summary: HelperSignalSummary | None,
) -> None:
    print("opadirect_compile_check")
    print("run_config")
    print(f"database: {result.database}")
    print(f"jax_enable_x64: {result.jax_enable_x64}")
    print(f"nlayer_base: {result.nlayer_base}")
    print(f"nlayer_changed: {result.nlayer_changed}")
    print(f"input_dtype: {result.input_dtype}")
    print(f"changed_dtype: {result.changed_dtype}")
    print(f"output_shape_base: {result.output_shape_base}")
    print(f"output_shape_changed: {result.output_shape_changed}")
    print()
    print("timing_summary")
    print(f"cold_call_s: {result.cold_call_s:.6f}")
    print(f"warm_call_s: {result.warm_call_s:.6f}")
    print(
        "repeated_same_shape_values_s: "
        + ", ".join(f"{value:.6f}" for value in result.repeated_same_shape_s)
    )
    print(
        "repeated_same_shape_mean_s: "
        f"{format_optional_float(metrics.repeated_same_shape_mean_s)}"
    )
    print(
        "repeated_same_shape_max_s: "
        f"{format_optional_float(metrics.repeated_same_shape_max_s)}"
    )
    print(f"changed_shape_call_s: {result.changed_shape_call_s:.6f}")
    print(f"dtype_changed_call_s: {format_optional_float(result.dtype_changed_call_s)}")
    print()
    print("derived_interpretation")
    print(f"cold_to_warm_drop_detected: {interpretation.cold_to_warm_drop_detected}")
    print(
        "same_shape_top_level_timing_appears_stable: "
        f"{interpretation.same_shape_top_level_timing_appears_stable}"
    )
    print(
        f"shape_change_recompilation_likely: {interpretation.shape_change_recompilation_likely}"
    )
    print(
        f"dtype_change_recompilation_likely: {interpretation.dtype_change_recompilation_likely}"
    )
    print(f"cold_to_warm_ratio: {format_optional_ratio(metrics.cold_to_warm_ratio)}")
    print(
        "repeated_same_shape_mean_to_warm_ratio: "
        f"{format_optional_ratio(metrics.repeated_mean_to_warm_ratio)}"
    )
    print(
        "repeated_same_shape_max_to_warm_ratio: "
        f"{format_optional_ratio(metrics.repeated_max_to_warm_ratio)}"
    )
    print(
        "changed_shape_to_warm_ratio: "
        f"{format_optional_ratio(metrics.changed_shape_to_warm_ratio)}"
    )
    print(
        "dtype_changed_to_warm_ratio: "
        f"{format_optional_ratio(metrics.dtype_changed_to_warm_ratio)}"
    )
    print(
        f"same_shape_top_level_category: {interpretation.same_shape_top_level_category}"
    )
    print(f"shape_change_category: {interpretation.shape_change_category}")
    print(f"dtype_change_category: {interpretation.dtype_change_category}")
    if helper_summary is not None:
        print()
        print("helper_signal_summary")
        print(f"helper_log_capture_enabled: {helper_summary.helper_log_capture_enabled}")
        print(
            f"helper_log_capture_succeeded: {helper_summary.helper_log_capture_succeeded}"
        )
        print(
            f"same_shape_repeat_phase_count: {helper_summary.same_shape_repeat_phase_count}"
        )
        print(f"captured_log_line_count: {helper_summary.captured_log_line_count}")
        print(f"relevant_log_line_count: {helper_summary.relevant_log_line_count}")
        print(
            "suspicious_helper_signal_detected: "
            f"{helper_summary.suspicious_helper_signal_detected}"
        )
        print(
            "suspicious_helper_signal_count: "
            f"{helper_summary.suspicious_helper_signal_count}"
        )
        print(
            "suspicious_helper_names: "
            + ", ".join(helper_summary.suspicious_helper_names)
            if helper_summary.suspicious_helper_names
            else "suspicious_helper_names: none"
        )
        print(
            "suspicious_log_patterns_found: "
            + ", ".join(helper_summary.suspicious_log_patterns_found)
            if helper_summary.suspicious_log_patterns_found
            else "suspicious_log_patterns_found: none"
        )
        print(
            "same_shape_inner_helper_stability: "
            f"{helper_summary.same_shape_inner_helper_stability}"
        )
        print(f"helper_signal_category: {helper_summary.helper_signal_category}")
        print(
            "helper_patterns_used: "
            + ", ".join(helper_summary.helper_patterns_used)
        )
        if helper_summary.log_capture_note is not None:
            print(f"log_capture_note: {helper_summary.log_capture_note}")
    print()
    print("judgment_thresholds")
    for key, value in interpretation.thresholds.items():
        print(f"{key}: {value}")
    print()
    print("conclusion")
    for line in interpretation.conclusion:
        print(f"- {line}")
    if interpretation.log_hint is not None:
        print()
        print("jax_log_hint")
        print(f"- {interpretation.log_hint}")


def build_empty_helper_summary(note: str) -> HelperSignalSummary:
    return HelperSignalSummary(
        helper_log_capture_enabled=True,
        helper_log_capture_succeeded=False,
        same_shape_repeat_phase_count=0,
        captured_log_line_count=0,
        relevant_log_line_count=0,
        suspicious_helper_signal_detected="unclear",
        suspicious_helper_signal_count=0,
        suspicious_helper_names=[],
        suspicious_log_patterns_found=[],
        same_shape_inner_helper_stability="unclear",
        helper_signal_category="Not enough evidence",
        helper_patterns_used=[
            "being re-defined repeatedly",
            "seen another function defined on the same line",
            "preventing caching",
            "cache miss with helper name during same-shape repeat",
            "compile/tracing line with helper name during same-shape repeat",
        ],
        log_capture_note=note,
    )


def build_child_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        __file__,
        "--internal-child-run",
        "--quiet-json",
        "--database",
        args.database,
        "--nlayer",
        str(args.nlayer),
        "--repeat",
        str(args.repeat),
        "--dtype",
        args.dtype,
        "--dtype-change",
        args.dtype_change,
        "--nu-grid-size",
        str(args.nu_grid_size),
    ]
    if args.changed_nlayer is not None:
        cmd.extend(["--changed-nlayer", str(args.changed_nlayer)])
    if args.enable_x64:
        cmd.append("--enable-x64")
    return cmd


def extract_json_payload(stdout_text: str) -> dict[str, object] | None:
    start = stdout_text.find(JSON_BEGIN_MARKER)
    end = stdout_text.find(JSON_END_MARKER)
    if start == -1 or end == -1 or end <= start:
        return None
    json_text = stdout_text[start + len(JSON_BEGIN_MARKER) : end].strip()
    if not json_text:
        return None
    return json.loads(json_text)


def summarize_helper_logs(log_text: str) -> HelperSignalSummary:
    phase_logs: dict[str | None, list[str]] = defaultdict(list)
    current_phase: str | None = None
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(PHASE_BEGIN_MARKER):
            current_phase = line[len(PHASE_BEGIN_MARKER) :]
            continue
        phase_logs[current_phase].append(line)

    same_shape_phase_names = sorted(
        phase for phase in phase_logs if phase is not None and phase.startswith("same_shape_repeat_")
    )
    same_shape_lines = []
    for phase in same_shape_phase_names:
        same_shape_lines.extend(phase_logs[phase])

    suspicious_names: set[str] = set()
    suspicious_patterns: set[str] = set()
    suspicious_count = 0
    relevant_log_line_count = 0

    for line in same_shape_lines:
        lowered = line.lower()
        helper_hits = [name for name in HELPER_NAMES if name in lowered]
        is_relevant = False
        line_patterns = []

        for label, pattern in STRONG_SUSPICIOUS_PATTERNS.items():
            if pattern in lowered:
                line_patterns.append(label)
                is_relevant = True

        if helper_hits and (
            "cache miss" in lowered
            or "compil" in lowered
            or "tracing" in lowered
            or "re-tracing" in lowered
        ):
            if "cache miss" in lowered:
                line_patterns.append("cache_miss_with_helper")
            if "compil" in lowered:
                line_patterns.append("compile_with_helper")
            if "tracing" in lowered or "re-tracing" in lowered:
                line_patterns.append("tracing_with_helper")
            is_relevant = True

        if helper_hits:
            is_relevant = True

        if is_relevant:
            relevant_log_line_count += 1

        if line_patterns:
            suspicious_count += 1
            suspicious_names.update(helper_hits)
            suspicious_patterns.update(line_patterns)

    captured_log_line_count = len([line for line in log_text.splitlines() if line.strip()])

    if not same_shape_phase_names:
        return build_empty_helper_summary(
            "Helper log capture succeeded, but no repeated same-shape phases were identified in the captured output."
        )

    if captured_log_line_count == 0:
        return build_empty_helper_summary(
            "No logs were captured from the child run, so helper-level stability could not be determined."
        )

    if relevant_log_line_count == 0:
        return HelperSignalSummary(
            helper_log_capture_enabled=True,
            helper_log_capture_succeeded=True,
            same_shape_repeat_phase_count=len(same_shape_phase_names),
            captured_log_line_count=captured_log_line_count,
            relevant_log_line_count=0,
            suspicious_helper_signal_detected="unclear",
            suspicious_helper_signal_count=0,
            suspicious_helper_names=[],
            suspicious_log_patterns_found=[],
            same_shape_inner_helper_stability="unclear",
            helper_signal_category="Not enough evidence",
            helper_patterns_used=[
                "being re-defined repeatedly",
                "seen another function defined on the same line",
                "preventing caching",
                "cache miss with helper name during same-shape repeat",
                "compile/tracing line with helper name during same-shape repeat",
            ],
            log_capture_note="Logs were captured, but no helper-relevant compile or cache-miss lines were found during repeated same-shape phases.",
        )

    if suspicious_count > 0:
        return HelperSignalSummary(
            helper_log_capture_enabled=True,
            helper_log_capture_succeeded=True,
            same_shape_repeat_phase_count=len(same_shape_phase_names),
            captured_log_line_count=captured_log_line_count,
            relevant_log_line_count=relevant_log_line_count,
            suspicious_helper_signal_detected="yes",
            suspicious_helper_signal_count=suspicious_count,
            suspicious_helper_names=sorted(suspicious_names),
            suspicious_log_patterns_found=sorted(suspicious_patterns),
            same_shape_inner_helper_stability="suspicious",
            helper_signal_category="Suspicious / worth investigating",
            helper_patterns_used=[
                "being re-defined repeatedly",
                "seen another function defined on the same line",
                "preventing caching",
                "cache miss with helper name during same-shape repeat",
                "compile/tracing line with helper name during same-shape repeat",
            ],
            log_capture_note="Suspicious helper-level signals were observed during repeated same-shape phases after the initial warm call.",
        )

    return HelperSignalSummary(
        helper_log_capture_enabled=True,
        helper_log_capture_succeeded=True,
        same_shape_repeat_phase_count=len(same_shape_phase_names),
        captured_log_line_count=captured_log_line_count,
        relevant_log_line_count=relevant_log_line_count,
        suspicious_helper_signal_detected="no",
        suspicious_helper_signal_count=0,
        suspicious_helper_names=[],
        suspicious_log_patterns_found=[],
        same_shape_inner_helper_stability="stable",
        helper_signal_category="Expected / normal",
        helper_patterns_used=[
            "being re-defined repeatedly",
            "seen another function defined on the same line",
            "preventing caching",
            "cache miss with helper name during same-shape repeat",
            "compile/tracing line with helper name during same-shape repeat",
        ],
        log_capture_note="No suspicious helper-level signals were found in repeated same-shape phases, but this remains dependent on the emitted JAX log format.",
    )


def run_capture_mode(args: argparse.Namespace) -> dict[str, object]:
    child_cmd = build_child_command(args)
    child_env = os.environ.copy()
    child_env.setdefault("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", "cpu"))
    child_env["JAX_LOG_COMPILES"] = child_env.get("JAX_LOG_COMPILES", "1")
    child_env["JAX_EXPLAIN_CACHE_MISSES"] = child_env.get(
        "JAX_EXPLAIN_CACHE_MISSES", "1"
    )
    completed = subprocess.run(
        child_cmd,
        cwd=os.getcwd(),
        env=child_env,
        text=True,
        capture_output=True,
    )

    payload = extract_json_payload(completed.stdout)
    if payload is None:
        helper_summary = build_empty_helper_summary(
            "The child run finished without a parseable JSON payload, so helper-level analysis could not be completed."
        )
        return {
            "timing": None,
            "derived_metrics": None,
            "helper_signal_summary": asdict(helper_summary),
            "interpretation": {
                "error": "Failed to parse child payload.",
                "child_return_code": completed.returncode,
            },
            "child_stdout_tail": completed.stdout[-2000:],
            "child_stderr_tail": completed.stderr[-2000:],
        }

    combined_logs = completed.stdout.replace(JSON_BEGIN_MARKER, "").replace(
        JSON_END_MARKER, ""
    ) + "\n" + completed.stderr
    helper_summary = summarize_helper_logs(combined_logs)
    payload["helper_signal_summary"] = asdict(helper_summary)
    payload["interpretation"] = build_interpretation(
        TimingResult(**payload["timing"]),
        DerivedMetrics(**payload["derived_metrics"]),
        helper_summary,
    )
    payload["interpretation"] = asdict(payload["interpretation"])
    return payload


def execute_measurement(args: argparse.Namespace) -> tuple[TimingResult, DerivedMetrics]:
    changed_nlayer = args.changed_nlayer or (args.nlayer + 1)

    nu_grid, _, _ = mock_wavenumber_grid(Nx=args.nu_grid_size)
    mdb = build_mdb(args.database)
    opa = OpaDirect(mdb=mdb, nu_grid=nu_grid)

    phase_name = "cold_call" if args.internal_child_run else None
    Tarr_base, Parr_base = make_profiles(args.nlayer, args.dtype)
    cold_call_s, output_shape_base = timed_xsmatrix(
        opa, Tarr_base, Parr_base, phase_name=phase_name
    )
    phase_name = "warm_call" if args.internal_child_run else None
    warm_call_s, _ = timed_xsmatrix(opa, Tarr_base, Parr_base, phase_name=phase_name)

    repeated_same_shape_s = []
    for index in range(args.repeat):
        phase_name = f"same_shape_repeat_{index + 1}" if args.internal_child_run else None
        elapsed, _ = timed_xsmatrix(opa, Tarr_base, Parr_base, phase_name=phase_name)
        repeated_same_shape_s.append(elapsed)

    Tarr_changed, Parr_changed = make_profiles(changed_nlayer, args.dtype)
    phase_name = "changed_shape_call" if args.internal_child_run else None
    changed_shape_call_s, output_shape_changed = timed_xsmatrix(
        opa, Tarr_changed, Parr_changed, phase_name=phase_name
    )

    dtype_changed_call_s = None
    changed_dtype = None
    if args.dtype_change != "none" and args.dtype_change != args.dtype:
        changed_dtype = args.dtype_change
        if args.dtype_change == "float64" and not config.read("jax_enable_x64"):
            changed_dtype = "float64_requested_but_x64_disabled"
        else:
            Tarr_dtype_changed, Parr_dtype_changed = make_profiles(
                args.nlayer, args.dtype_change
            )
            phase_name = "changed_dtype_call" if args.internal_child_run else None
            dtype_changed_call_s, _ = timed_xsmatrix(
                opa,
                Tarr_dtype_changed,
                Parr_dtype_changed,
                phase_name=phase_name,
            )

    result = TimingResult(
        database=args.database,
        jax_enable_x64=bool(config.read("jax_enable_x64")),
        nlayer_base=args.nlayer,
        nlayer_changed=changed_nlayer,
        input_dtype=args.dtype,
        changed_dtype=changed_dtype,
        output_shape_base=output_shape_base,
        output_shape_changed=output_shape_changed,
        cold_call_s=cold_call_s,
        warm_call_s=warm_call_s,
        repeated_same_shape_s=repeated_same_shape_s,
        changed_shape_call_s=changed_shape_call_s,
        dtype_changed_call_s=dtype_changed_call_s,
    )
    metrics = derive_metrics(result)
    return result, metrics


def emit_json_payload(payload: dict[str, object]) -> None:
    print(JSON_BEGIN_MARKER)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(JSON_END_MARKER)


def main() -> None:
    args = parse_args()

    if args.capture_helper_signals and not args.internal_child_run:
        payload = run_capture_mode(args)
        if args.quiet_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return
        if payload.get("timing") is None:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return
        result = TimingResult(**payload["timing"])
        metrics = DerivedMetrics(**payload["derived_metrics"])
        helper_summary = HelperSignalSummary(**payload["helper_signal_summary"])
        interpretation = Interpretation(**payload["interpretation"])
        print_human_summary(result, metrics, interpretation, helper_summary)
        if args.json:
            print()
            print("json_result")
            print(json.dumps(payload, indent=2, sort_keys=True))
        return

    result, metrics = execute_measurement(args)
    helper_summary = None
    interpretation = build_interpretation(result, metrics, helper_summary)
    payload = serialize_payload(result, metrics, interpretation, helper_summary)

    if args.quiet_json:
        if args.internal_child_run:
            emit_json_payload(payload)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print_human_summary(result, metrics, interpretation, helper_summary)
    if args.json:
        print()
        print("json_result")
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

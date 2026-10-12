"""Run only the old H2O sampling backlog in an isolated, permanent output root.

Dry-run is read-only. Execution requires a separately coordinated GPU slot.
Use --execute --start 0 --stop 1 for the first memory pilot. The main worker
must not enter its 0..38 backlog concurrently; this worker detects active
overlap but cannot lock a dispatcher that does not implement that protocol.
Keep both worker-B directories permanently: published records use their paths.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import prepare_rce_wasp18b_ckd_sampling as sampling

PREPARED = Path("/home/kawahara/.database/rce_wasp18b/prepared")
EXPECTED_SAMPLING = "1303050c6a301c0d59657d6cdd61095ffbf914f8cbb5cef5bbd78bbbfe62ade4"
EXPECTED_REFINEMENT = "50ef57232dbd89a66afa8568f98fa8d5c459531bc72f818bbaac8f0e5fb45689"
FROZEN = {
    "prepare_rce_wasp18b_opacity.py": "1e4e8e452aae42c29d3d0c3562424a010cbc357c275b9a3fc1dc0e740dd1abb8",
    "prepare_rce_wasp18b_supplemental.py": "c3389a8ecae79649513b7ae66c2765bc174739e6b57f0e7b57e642b9d338f24f",
    "prepare_rce_wasp18b_atomic.py": "8b3f54b76c1e5035404cdbadac0008ac19739d408d2a46520cb037aa34a2a1fc",
    "prepare_rce_wasp18b_ckd_refinement.py": "38487f790b1ec513cafc0310ff9d70c104640facc01f3f1415857d64faa4ba0b",
    "prepare_rce_wasp18b_ckd_sampling.py": "f8d8c939bdda5b667542299bf6b6846da7047cee4aac97c6d5ef232fd6f76f3f",
    "prefill_rce_wasp18b_ckd_refinement.py": "d30276e91c02435a9a0f131ad78b0629a3fe710e293f114d7a3994aaaf641125",
}


def pin(path):
    return sampling.base._file_entry(Path(path), "Backlog worker input or artifact")


def verify_frozen():
    for name, expected in FROZEN.items():
        if sampling.base._sha256(ROOT / "examples" / name) != expected:
            raise ValueError(f"Frozen preparation changed: {name}")


def make_plans(args):
    verify_frozen()
    common = dict(base_preparation=PREPARED / "H2O1/complete.json",
                  source_preparation=None, species="H2O1")
    a = sampling.make_plan(argparse.Namespace(**common,
        refinement_dir=PREPARED / "ckd_refinement", output_dir=PREPARED / "ckd_sampling"))
    b = sampling.make_plan(argparse.Namespace(**common,
        refinement_dir=args.refinement_dir, output_dir=args.output_dir))
    if (a["identity"] != b["identity"] or a["fingerprint"] != EXPECTED_SAMPLING
            or a["refinement_plan"]["identity"] != b["refinement_plan"]["identity"]
            or a["refinement_plan"]["fingerprint"] != EXPECTED_REFINEMENT):
        raise ValueError("Worker plans differ from the pinned H2O teacher identities")
    roots = [Path(p) for p in (a["output_dir"], a["refinement_plan"]["output_dir"],
                              b["output_dir"], b["refinement_plan"]["output_dir"])]
    if any(x.is_relative_to(y) or y.is_relative_to(x)
           for i, x in enumerate(roots) for y in roots[i + 1:]):
        raise ValueError("All four worker output roots must be disjoint")
    return a, b


def checked_refinement(plan, index):
    record = sampling._existing_refinement(plan, index)
    if record is None:
        raise ValueError(f"The old refinement chunk {index} is missing")
    path, _ = sampling.refinement._source_chunk(plan["refinement_plan"], index)
    source = record["source_chunk"]
    if (Path(source["path"]).resolve() != path.resolve()
            or source["sha256"] != sampling.base._sha256(path)):
        raise ValueError(f"Refinement completion references another chunk: {index}")
    return record


def no_replace_link(source, destination):
    """Publish one complete inode; an existing different record is never replaced."""
    source, destination = Path(source), Path(destination)
    expected = sampling.base._sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        if destination.is_symlink() or sampling.base._sha256(destination) != expected:
            raise ValueError(f"Refusing to replace existing completion: {destination}")
        return False
    if sampling.base._sha256(destination) != expected:
        raise ValueError(f"Completion changed while publishing: {destination}")
    return True


def available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise ValueError("MemAvailable is unavailable")


def active_main_backlog(plan):
    """Read process arguments only; never signal or stop the existing jobs."""
    active = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = path.read_bytes().decode().split("\0")
            if not any(Path(x).name == Path(sampling.__file__).name for x in argv):
                continue
            value = lambda name: argv[argv.index(name) + 1]
            if (value("--species") == "H2O1"
                    and Path(value("--output-dir")).resolve() == Path(plan["output_dir"])
                    and 0 <= int(value("--chunk-index")) < 39):
                active.append(dict(pid=int(path.parent.name), chunk_index=int(value("--chunk-index"))))
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return active


def require_slot(plan, minimum_gib):
    active = active_main_backlog(plan)
    if active:
        raise ValueError(f"Main worker has entered the backlog; coordinate before resuming B: {active}")
    available = available_gib()
    if available < minimum_gib:
        raise ValueError(f"Only {available:.3f} GiB available; require {minimum_gib:.3f} GiB")
    return available


def child_environment():
    env = dict(os.environ)
    env.update(JAX_ENABLE_X64="True", JAX_PLATFORMS="cuda", JAX_PLATFORM_NAME="gpu",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", NUMBA_DISABLE_JIT="1",
               MPLBACKEND="Agg", LC_ALL="C")
    env["PYTHONPATH"] = str(ROOT / "src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def runtime_record(env):
    packages = {}
    for name in ("exojax", "jax", "jaxlib", "numpy", "scipy", "pandas", "tables", "radis", "numba"):
        packages[name] = importlib.metadata.version(name)
    sources = [Path(__file__), ROOT / "examples/_rce_wasp18b_assets.py"]
    sources += [ROOT / "examples" / name for name in FROZEN]
    sources += sorted((ROOT / "src/exojax").rglob("*.py"))
    radis = importlib.metadata.distribution("radis")
    sources += [Path(radis.locate_file("radis/" + name)) for name in
                ("api/exomolapi.py", "api/dbmanager.py", "api/hdf5.py", "lbl/base.py")]
    return dict(python=pin(sys.executable), python_version=sys.version, platform=platform.platform(),
                package_versions=packages, code_sources={str(p.resolve()): pin(p) for p in sources},
                environment={key: env[key] for key in ("PYTHONPATH", "JAX_ENABLE_X64", "JAX_PLATFORMS",
                    "JAX_PLATFORM_NAME", "XLA_PYTHON_CLIENT_PREALLOCATE", "NUMBA_DISABLE_JIT", "LC_ALL",
                    "CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "LD_LIBRARY_PATH") if key in env})


def command(plan, index):
    native = plan["refinement_plan"]
    return [sys.executable, "-u", str(Path(sampling.__file__).resolve()),
            "--base-preparation", native["base_preparation"]["path"],
            "--refinement-dir", native["output_dir"], "--output-dir", plan["output_dir"],
            "--species", "H2O1", "--chunk-index", str(index)]


def transfer_chunk(a, b, index, directory, env, minimum_gib, runtime):
    record = checked_refinement(a, index)
    original = sampling.refinement._chunk_path(a["refinement_plan"], index)
    protected = {"completion": pin(original), **record["files"]}
    if sampling.completed_chunk(a, index) is not None:
        return dict(chunk_index=index, status="verified_existing_main", main=pin(sampling._chunk_path(a, index)))
    before_available = require_slot(a, minimum_gib)
    no_replace_link(original, sampling.refinement._chunk_path(b["refinement_plan"], index))
    checked_refinement(b, index)
    child = None
    try:
        if sampling.completed_chunk(b, index) is None:
            sampling.base._check_files(runtime["code_sources"])
            log, timing = directory / f"chunk_{index:05d}.log", directory / f"chunk_{index:05d}.time"
            argv = command(b, index)
            with log.open("x") as output:
                result = subprocess.run(["/usr/bin/time", "-v", "-o", str(timing), *argv],
                                        cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            child = dict(command=argv, returncode=result.returncode, log=pin(log), timing=pin(timing))
            for line in timing.read_text().splitlines():
                if "Maximum resident set size (kbytes):" in line:
                    child["max_rss_kib"] = int(line.split(":")[-1].strip())
            if result.returncode:
                raise ValueError(f"Chunk {index} failed; completed files retained; see {log}")
        sampling.base._check_files(runtime["code_sources"])
        if sampling.completed_chunk(b, index) is None:
            raise ValueError(f"Child did not complete sampling chunk {index}")
        # Never publish into an index the main dispatcher has begun processing.
        active = active_main_backlog(a)
        if active:
            raise ValueError(f"Main dispatcher reached backlog before publication: {active}")
        if sampling.completed_chunk(a, index) is None:
            no_replace_link(sampling._chunk_path(b, index), sampling._chunk_path(a, index))
        if sampling.completed_chunk(a, index) is None:
            raise ValueError("Published completion failed validation")
        return dict(chunk_index=index, status="verified_main_sampling", child=child,
                    available_gib_before=before_available, main=pin(sampling._chunk_path(a, index)),
                    worker_b=pin(sampling._chunk_path(b, index)))
    finally:
        sampling.base._check_files(protected)
        if pin(sampling.refinement._chunk_path(b["refinement_plan"], index))["sha256"] != protected["completion"]["sha256"]:
            raise ValueError("The protected worker-B refinement completion changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=39, help="Exclusive stop; pilot uses --stop 1")
    parser.add_argument("--minimum-available-gib", type=float, default=32.)
    parser.add_argument("--refinement-dir", type=Path, default=PREPARED / "ckd_refinement_h2o_backlog")
    parser.add_argument("--output-dir", type=Path, default=PREPARED / "ckd_sampling_h2o_backlog")
    args = parser.parse_args(argv)
    try:
        if not 0 <= args.start < args.stop <= 39 or not 32 <= args.minimum_available_gib < float("inf"):
            raise ValueError("Require 0 <= start < stop <= 39 and a finite RAM guard >=32 GiB")
        a, b = make_plans(args)
        targets, protected = [], {}
        for index in range(args.start, args.stop):
            record = checked_refinement(a, index)
            original = sampling.refinement._chunk_path(a["refinement_plan"], index)
            protected[str(index)] = pin(original)
            protected.update({f"{index}/{name}": item for name, item in record["files"].items()})
            targets.append(dict(chunk_index=index, refinement=pin(original),
                                sampling_complete=sampling.completed_chunk(a, index) is not None))
        preview = dict(sampling_fingerprint=a["fingerprint"],
            refinement_fingerprint=a["refinement_plan"]["fingerprint"], targets=targets,
            worker_b_refinement_root=b["refinement_plan"]["output_dir"], worker_b_sampling_root=b["output_dir"],
            minimum_available_gib=args.minimum_available_gib, available_gib=available_gib(),
            active_main_backlog=active_main_backlog(a), frozen_script_sha256=FROZEN,
            retention="Keep worker-B coefficient files permanently; main records reference their absolute paths.",
            scheduling="Coordinator must keep the main dispatcher out of indices 0..38 while B is active.",
            first_command=command(b, args.start))
        if args.dry_run:
            print(json.dumps(preview, indent=2))
            return 0
        require_slot(a, args.minimum_available_gib)
        env = child_environment()
        runtime = runtime_record(env)
        root = Path(b["output_dir"])
        root.mkdir(parents=True, exist_ok=True)
        with (root / "backlog_worker.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"_{os.getpid()}"
            directory = root / "backlog_runs" / run_id
            directory.mkdir(parents=True)
            identity = dict(runtime=runtime, sampling_identity=b["identity"],
                            refinement_identity=b["refinement_plan"]["identity"],
                            refinement_root=b["refinement_plan"]["output_dir"], sampling_root=b["output_dir"])
            sampling.base._write_json(directory / "identity.json", identity)
            no_replace_link(directory / "identity.json", root / "backlog_identity.json")
            report_path = directory / "run.json"
            report = dict(**preview, runtime=runtime, protected_refinement_files=protected,
                          status="running", chunks=[])
            sampling.base._write_json(report_path, report)
            print(f"Run report: {report_path}", flush=True)
            try:
                for index in range(args.start, args.stop):
                    row = transfer_chunk(a, b, index, directory, env, args.minimum_available_gib, runtime)
                    report["chunks"].append(row)
                    sampling.base._write_json(report_path, report)
                    print(json.dumps(row), flush=True)
                report["status"] = "complete_selected_range"
            except Exception as error:
                report.update(status="failed", error=str(error))
                raise
            finally:
                try:
                    verify_frozen()
                    sampling.base._check_files(protected)
                    report["protected_refinements_unchanged"] = True
                except Exception as error:
                    report.update(status="failed", protection_error=str(error), protected_refinements_unchanged=False)
                    raise
                finally:
                    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
                    sampling.base._write_json(report_path, report)
        return 0
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

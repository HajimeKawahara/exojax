#!/usr/bin/env python3
"""Run a frozen P7 caller with separate CKD execution-adapter evidence."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

from rce_wasp18b_explicit_ckd import ExplicitCKDContext, ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, choices=("forward", "validation"))
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    allowed = {"--manifest", "--observation", "--output", "--response-resolution",
               "--stream-resolved", "--mask-continuation", "--check-inputs",
               "--fixed-temperature", "--resolved-spectrum-output", "--compare",
               "--comparison-kind", "--suite", "--initial-temperatures", "--help"}
    if any(value.startswith("--") and value.split("=", 1)[0] not in allowed
           for value in forwarded):
        parser.error("Use complete frozen option names; abbreviations are not accepted.")
    outputs = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    outputs.add_argument("--output", required=True, type=Path)
    outputs.add_argument("--resolved-spectrum-output", type=Path)
    paths, _ = outputs.parse_known_args(forwarded)
    destinations = [path for path in (args.evidence, paths.output,
                    paths.resolved_spectrum_output) if path is not None]
    destinations += [path.with_name(path.name + ".tmp") for path in destinations]
    if (len({path.resolve() for path in destinations}) != len(destinations)
            or any(os.path.lexists(path) for path in destinations)):
        parser.error("Every output and evidence destination must be new and distinct.")
    sys.path.insert(0, str(ROOT / "examples"))
    context = ExplicitCKDContext()
    report = {"schema_version": "exojax.rce-explicit-ckd-execution/1", "status": "starting",
              "wrapper_path": str(Path(__file__).resolve()),
              "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "driver": args.driver, "arguments": forwarded, "execution_adapter": context.evidence()}
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        with context:
            code = importlib.import_module("rce_wasp18b_" + args.driver).main(forwarded)
        report.update(status="completed" if code == 0 else "failed", exit_status=int(code))
        return code
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["execution_adapter"] = context.evidence()
        temporary = args.evidence.with_name(args.evidence.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.evidence)


if __name__ == "__main__":
    raise SystemExit(main())

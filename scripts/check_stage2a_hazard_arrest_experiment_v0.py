from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.stage2a_hazard_arrest_runner import (
    EXPERIMENT_OUTPUT_PATH,
    QUALIFICATION_OUTPUT_PATH,
    load_experiment_payloads,
    load_qualification_payloads,
    validate_static_sources,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Stage 2A experiment checker.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-static", action="store_true")
    modes.add_argument("--validate-qualification", action="store_true")
    modes.add_argument("--validate-published", action="store_true")
    modes.add_argument("--print-selection", action="store_true")
    modes.add_argument("--print-result", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any(
        (
            args.validate_static,
            args.validate_qualification,
            args.validate_published,
            args.print_selection,
            args.print_result,
        )
    ):
        parser.print_help()
        return 0
    try:
        if args.validate_static:
            report = validate_static_sources(ROOT)
            print(
                "STAGE2A_STATIC: passed; "
                f"traces={report['source_trace_count']}; physical_executions=0"
            )
        elif args.validate_qualification:
            payloads = load_qualification_payloads(ROOT)
            manifest = json.loads(payloads["qualification_manifest.json"])
            print(
                "STAGE2A_QUALIFICATION: passed; "
                f"eligible={manifest['eligible_boundary_count']}; physical_executions=0"
            )
        elif args.validate_published:
            payloads = load_experiment_payloads(ROOT)
            manifest = json.loads(payloads["experiment_manifest.json"])
            print(
                "STAGE2A_EXPERIMENT: passed; bounded_runs=2; "
                f"manifest_hash={manifest['canonical_manifest_hash']}"
            )
        elif args.print_selection:
            selected = json.loads(
                (ROOT / QUALIFICATION_OUTPUT_PATH / "selected_experiment.json").read_text(
                    "utf-8"
                )
            )
            for key in sorted(selected):
                print(f"{key}={json.dumps(selected[key], sort_keys=True, separators=(',', ':'))}")
        else:
            for name in (
                "baseline_summary.json",
                "active_summary.json",
                "release_report.json",
            ):
                value = json.loads((ROOT / EXPERIMENT_OUTPUT_PATH / name).read_text("utf-8"))
                print(f"[{name}]")
                for key in sorted(value):
                    print(f"{key}={json.dumps(value[key], sort_keys=True, separators=(',', ':'))}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

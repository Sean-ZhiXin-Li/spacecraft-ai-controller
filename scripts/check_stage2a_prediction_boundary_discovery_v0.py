from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.stage2a_prediction_boundary_discovery import (  # noqa: E402
    load_published_payloads,
    validate_static_sources,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Stage 2A prediction-boundary discovery checker."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-static", action="store_true")
    modes.add_argument("--validate-published", action="store_true")
    modes.add_argument("--print-summary", action="store_true")
    modes.add_argument("--print-candidates", action="store_true")
    return parser


def _print_mapping(value: dict[str, object]) -> None:
    for key in sorted(value):
        print(
            f"{key}="
            + json.dumps(value[key], ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any(
        (
            args.validate_static,
            args.validate_published,
            args.print_summary,
            args.print_candidates,
        )
    ):
        parser.print_help()
        return 0
    try:
        if args.validate_static:
            report = validate_static_sources(ROOT)
            print(
                "STAGE2A_PREDICTION_BOUNDARY_DISCOVERY_STATIC: passed; "
                f"cases={report['source_case_count']}; "
                f"trajectories={report['planned_trajectory_count']}; "
                "simulation_executed=false; write_performed=false"
            )
            return 0
        payloads = load_published_payloads(ROOT)
        manifest = json.loads(payloads["discovery_manifest.json"])
        if args.validate_published:
            print(
                "STAGE2A_PREDICTION_BOUNDARY_DISCOVERY_PUBLISHED: passed; "
                f"trajectories={manifest['completed_trajectory_count']}; "
                f"candidates={manifest['candidate_boundary_count']}; "
                f"manifest_hash={manifest['canonical_manifest_hash']}"
            )
        elif args.print_summary:
            _print_mapping(json.loads(payloads["coverage_summary.json"]))
        else:
            document = json.loads(payloads["candidate_boundaries.json"])
            print(f"candidate_boundary_count={document['candidate_boundary_count']}")
            for candidate in document["candidate_boundaries"]:
                print(
                    f"{candidate['candidate_id']}="
                    + json.dumps(candidate, sort_keys=True, separators=(",", ":"))
                )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

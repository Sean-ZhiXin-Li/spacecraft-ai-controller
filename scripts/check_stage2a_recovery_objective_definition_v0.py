from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_stage2a_recovery_objective_definition_v0 import (  # noqa: E402
    load_published_payloads,
    validate_sources,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Stage 2A recovery-objective audit checker."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-static", action="store_true")
    modes.add_argument("--validate-published", action="store_true")
    modes.add_argument("--print-objectives", action="store_true")
    modes.add_argument("--print-conflicts", action="store_true")
    modes.add_argument("--print-unknowns", action="store_true")
    return parser


def _print(document: dict[str, object]) -> None:
    for key in sorted(document):
        print(
            f"{key}="
            + json.dumps(document[key], ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any(
        (
            args.validate_static,
            args.validate_published,
            args.print_objectives,
            args.print_conflicts,
            args.print_unknowns,
        )
    ):
        parser.print_help()
        return 0
    try:
        if args.validate_static:
            validate_sources(ROOT)
            print(
                "STAGE2A_RECOVERY_OBJECTIVE_DEFINITION_STATIC: passed; "
                "physical_executions=0; controller_executions=0; authority_granted=false"
            )
            return 0
        payloads = load_published_payloads(ROOT)
        inventory = json.loads(payloads["objective_inventory.json"])
        if args.validate_published:
            print(
                "STAGE2A_RECOVERY_OBJECTIVE_DEFINITION_PUBLISHED: passed; "
                f"metrics={inventory['metric_count']}; conflicts={inventory['conflict_count']}; "
                f"inventory_hash={inventory['canonical_payload_hash']}"
            )
        elif args.print_objectives:
            _print(inventory)
        elif args.print_conflicts:
            _print(json.loads(payloads["conflict_analysis.json"]))
        else:
            _print(json.loads(payloads["unknown_evidence.json"]))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

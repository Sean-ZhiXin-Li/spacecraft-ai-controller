from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_stage2a_hazard_trigger_relevance_v0 import (  # noqa: E402
    load_published_payloads,
    validate_sources,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only checker for the Stage 2A hazard-trigger relevance audit."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-static", action="store_true")
    modes.add_argument("--validate-published", action="store_true")
    modes.add_argument("--print-trigger-comparison", action="store_true")
    modes.add_argument("--print-final-veto-role", action="store_true")
    return parser


def _print(value: dict[str, object]) -> None:
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
            args.print_trigger_comparison,
            args.print_final_veto_role,
        )
    ):
        parser.print_help()
        return 0
    try:
        if args.validate_static:
            validate_sources(ROOT)
            print(
                "STAGE2A_HAZARD_TRIGGER_RELEVANCE_STATIC: passed; "
                "physical_executions=0; write_performed=false; authority_granted=false"
            )
            return 0
        payloads = load_published_payloads(ROOT)
        manifest = json.loads(payloads["audit_manifest.json"])
        if args.validate_published:
            print(
                "STAGE2A_HAZARD_TRIGGER_RELEVANCE_PUBLISHED: passed; "
                f"trigger_a={manifest['trigger_a_observation_count']}; "
                f"trigger_b={manifest['trigger_b_observation_count']}; "
                f"manifest_hash={manifest['canonical_manifest_hash']}"
            )
        elif args.print_trigger_comparison:
            _print(json.loads(payloads["trigger_a_report.json"]))
            _print(json.loads(payloads["trigger_b_report.json"]))
        else:
            _print(json.loads(payloads["final_veto_role_report.json"]))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

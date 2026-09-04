from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_stage2a_post_veto_alternative_audit_v0 import (  # noqa: E402
    load_published_payloads,
    validate_sources,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only Stage 2A post-veto alternative audit checker."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-static", action="store_true")
    modes.add_argument("--validate-published", action="store_true")
    modes.add_argument("--print-alternatives", action="store_true")
    modes.add_argument("--print-interpretation", action="store_true")
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
            args.print_alternatives,
            args.print_interpretation,
        )
    ):
        parser.print_help()
        return 0
    try:
        if args.validate_static:
            validate_sources(ROOT)
            print(
                "STAGE2A_POST_VETO_ALTERNATIVE_STATIC: passed; "
                "physical_executions=0; write_performed=false; authority_granted=false"
            )
            return 0
        payloads = load_published_payloads(ROOT)
        manifest = json.loads(payloads["audit_manifest.json"])
        if args.validate_published:
            print(
                "STAGE2A_POST_VETO_ALTERNATIVE_PUBLISHED: passed; "
                f"veto_events={manifest['duplicate_aware_veto_event_count']}; "
                f"safe_alternatives={manifest['veto_events_with_safe_alternative']}; "
                f"manifest_hash={manifest['canonical_manifest_hash']}"
            )
        elif args.print_alternatives:
            _print(json.loads(payloads["alternative_coverage.json"]))
        else:
            _print(json.loads(payloads["final_veto_interpretation.json"]))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

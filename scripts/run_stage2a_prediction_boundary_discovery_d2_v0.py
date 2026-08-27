from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_assurance.stage2a_prediction_boundary_discovery_d2 import (  # noqa: E402
    ANGLE_GRID,
    MAXIMUM_RECOVERY_TRANSITIONS,
    OUTPUT_PATH,
    RECOVERY_BRANCH_ID,
    D2DiscoveryError,
    build_d2_payloads,
    execute_frozen_discovery,
    load_d2_plan,
    protected_evidence_hashes,
    validate_d2_payloads,
    validate_static_sources,
)
from runtime_assurance.staged_recovery_shadow_calibration import (  # noqa: E402
    atomic_publish_new_directory,
)


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _require_clean_pushed_head() -> str:
    for command, label in (
        (("diff", "--quiet"), "tracked changes"),
        (("diff", "--cached", "--quiet"), "staged changes"),
    ):
        if subprocess.run(["git", *command], cwd=ROOT, check=False).returncode:
            raise D2DiscoveryError(f"frozen D2 discovery blocked by {label}")
    head = _git_output("rev-parse", "HEAD")
    origin = _git_output("rev-parse", "origin/main")
    if not head or head != origin:
        raise D2DiscoveryError(
            "frozen D2 discovery requires the implementation HEAD pushed to origin/main"
        )
    return head


def _plan() -> None:
    plan = load_d2_plan(ROOT)
    print(f"discovery_id={plan['discovery_id']}")
    print("angles=" + json.dumps(list(ANGLE_GRID), separators=(",", ":")))
    print(f"source_case_count={len(ANGLE_GRID)}")
    print(f"recovery_branch={RECOVERY_BRANCH_ID}")
    print(f"maximum_recovery_physical_transitions={MAXIMUM_RECOVERY_TRANSITIONS}")
    print(f"canonical_plan_hash={plan['canonical_plan_hash']}")
    print(f"output={OUTPUT_PATH.as_posix()}")
    print("active_authority_granted=false")
    print("hazard_arrest_interventions=0")
    print("execution=false")


def _validate_only() -> None:
    report = validate_static_sources(ROOT, require_output_absent=True)
    print(
        "STAGE2A_TARGETED_PREDICTION_BOUNDARY_DISCOVERY_D2_STATIC: passed; "
        f"source_cases={report['source_case_count']}; "
        f"maximum_recovery_transitions={report['maximum_recovery_physical_transitions']}; "
        "simulation_executed=false; write_performed=false"
    )


def _execute() -> None:
    validate_static_sources(ROOT, require_output_absent=True)
    implementation_commit = _require_clean_pushed_head()
    protected_before = protected_evidence_hashes(ROOT)
    boundaries, recoveries = execute_frozen_discovery(
        ROOT, implementation_commit=implementation_commit
    )
    protected_after = protected_evidence_hashes(ROOT)
    payloads = build_d2_payloads(
        ROOT,
        boundaries,
        recoveries,
        implementation_commit=implementation_commit,
        protected_before=protected_before,
        protected_after=protected_after,
    )
    plan = load_d2_plan(ROOT)
    target = atomic_publish_new_directory(
        ROOT,
        OUTPUT_PATH,
        payloads,
        lambda values: validate_d2_payloads(values, source_plan=plan),
    )
    manifest = json.loads(payloads["discovery_manifest.json"])
    coverage = json.loads(payloads["coverage_summary.json"])
    diagnostics = json.loads(payloads["near_boundary_diagnostics.json"])
    print(f"published={target.relative_to(ROOT).as_posix()}")
    print("formal_invocation_count=1")
    print("automatic_retry_count=0")
    for key in (
        "upstream_source_execution_count",
        "upstream_prefix_physical_transition_count",
        "source_boundary_Final_Veto_rejection_count",
        "valid_source_boundary_count",
        "unavailable_source_boundary_count",
        "recovery_trajectory_count",
        "recovery_physical_transition_count",
        "states_evaluated",
        "candidate_boundary_count",
        "candidate_Final_Veto_rejection_count",
        "fallback_execution_count",
        "hazard_arrest_interventions",
        "total_physical_transition_count",
    ):
        print(f"{key}={manifest[key]}")
    for key in (
        "maximum_zero_action_predicted_speed_ratio",
        "closest_headroom",
        "closest_angle",
        "closest_event_index",
    ):
        print(f"{key}={json.dumps(diagnostics[key], separators=(',', ':'))}")
    print(
        "coverage="
        + json.dumps(coverage, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    )
    print(f"discovery_manifest_hash={manifest['canonical_manifest_hash']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen Stage 2A-D2 targeted prediction-boundary discovery. Default "
            "invocation executes no transition and writes nothing."
        )
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute-frozen-discovery", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.plan, args.validate_only, args.execute_frozen_discovery)):
        parser.print_help()
        return 0
    try:
        if args.plan:
            _plan()
        elif args.validate_only:
            _validate_only()
        else:
            _execute()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

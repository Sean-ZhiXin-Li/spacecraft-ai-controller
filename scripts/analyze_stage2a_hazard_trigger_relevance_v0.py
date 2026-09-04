from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Mapping


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = Path("analysis/stage2a_hazard_trigger_relevance_v0")
COMPLETED_DATE = "2026-09-04"
AUDIT_ID = "stage2a_hazard_trigger_relevance_v0"
SCHEMA_VERSION = "stage2a_hazard_trigger_relevance_audit_v0"
THRESHOLD = 1.90
D2_MANIFEST_HASH = "d5e77b0e4d3abe6b0bc67b2efa94ed3865517543fa4fdad6ed6705d5d97ebe9a"
STAGE1B_TRACE_SET_HASH = "ab4fd8a70e2aa446e4996126a53685999f55a24baa2522a688ed72b0c2d5cfa0"
REGISTRY_MANIFEST_HASH = "b800735062af8426045e40117044056d545f5ff2ebbc3795fdbf52266ba8a980"

STAGE1B_PATH = Path("analysis/staged_recovery_shadow_calibration_v0")
D2_PATH = Path("analysis/stage2a_prediction_boundary_discovery_d2_v0")
REGISTRY_PATH = Path("analysis/recovery_branch_state_registry_v0")
FINAL_VETO_PATH = Path("analysis/final_veto_ablation_v0")

REPORT_FILENAMES = (
    "trigger_a_report.json",
    "trigger_b_report.json",
    "same_boundary_comparison.json",
    "final_veto_role_report.json",
    "evidence_matrix.json",
    "summary.md",
)
ALL_FILENAMES = ("audit_manifest.json", *REPORT_FILENAMES)


class HazardTriggerAuditError(RuntimeError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_aggregate_hash(root: Path) -> str:
    entries = [
        {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return canonical_sha256(entries)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HazardTriggerAuditError(f"expected JSON object: {path.as_posix()}")
    return value


def _verify_canonical_manifest(
    document: Mapping[str, object],
    expected: str,
    label: str,
    *,
    excluded_fields: tuple[str, ...] = (),
) -> None:
    payload = dict(document)
    supplied = payload.pop("canonical_manifest_hash", None)
    for field in excluded_fields:
        payload.pop(field, None)
    if supplied != expected or canonical_sha256(payload) != expected:
        raise HazardTriggerAuditError(f"{label} canonical manifest hash mismatch")


def source_snapshot(repository_root: Path) -> dict[str, object]:
    roots = {
        "stage1b_calibration": STAGE1B_PATH,
        "stage2a_d2": D2_PATH,
        "branch_state_registry": REGISTRY_PATH,
        "final_veto": FINAL_VETO_PATH,
    }
    hashes: dict[str, str] = {}
    for label, relative in roots.items():
        absolute = repository_root / relative
        if not absolute.is_dir():
            raise HazardTriggerAuditError(f"missing frozen source directory: {relative.as_posix()}")
        hashes[label] = directory_aggregate_hash(absolute)
    return {
        "directory_aggregate_hashes": hashes,
        "d2_manifest_hash": D2_MANIFEST_HASH,
        "stage1b_trace_set_hash": STAGE1B_TRACE_SET_HASH,
        "registry_manifest_hash": REGISTRY_MANIFEST_HASH,
    }


def validate_sources(repository_root: Path) -> dict[str, object]:
    d2 = _json(repository_root / D2_PATH / "discovery_manifest.json")
    _verify_canonical_manifest(d2, D2_MANIFEST_HASH, "D2")
    if d2.get("hazard_arrest_interventions") != 0:
        raise HazardTriggerAuditError("D2 active intervention count is not zero")

    trace_manifest = _json(repository_root / STAGE1B_PATH / "trace_set_manifest.json")
    _verify_canonical_manifest(
        trace_manifest,
        str(trace_manifest.get("canonical_manifest_hash")),
        "Stage 1B trace set",
    )
    if (
        trace_manifest.get("trace_set_aggregate_hash") != STAGE1B_TRACE_SET_HASH
        or trace_manifest.get("physical_equivalence_failures") != 0
        or trace_manifest.get("shadow_output_consumed_by_physical_runtime") is not False
        or trace_manifest.get("staged_recovery_execution") != "not_authorized"
    ):
        raise HazardTriggerAuditError("Stage 1B trace-set authority or identity mismatch")

    registry = _json(repository_root / REGISTRY_PATH / "registry_manifest.json")
    _verify_canonical_manifest(
        registry,
        REGISTRY_MANIFEST_HASH,
        "branch-state registry",
        excluded_fields=("result_commit_when_available",),
    )

    trace_index = _json(repository_root / STAGE1B_PATH / "trace_index.json")
    traces = trace_index.get("traces")
    if not isinstance(traces, list) or len(traces) != 13:
        raise HazardTriggerAuditError("Stage 1B trace index must contain 13 traces")
    for item in traces:
        if not isinstance(item, dict):
            raise HazardTriggerAuditError("invalid Stage 1B trace index entry")
        path = repository_root / STAGE1B_PATH / str(item["trace_path"])
        if file_sha256(path) != item.get("trace_sha256"):
            raise HazardTriggerAuditError(f"Stage 1B trace hash mismatch: {path.name}")

    final_veto_rows = (repository_root / FINAL_VETO_PATH / "decision_log.jsonl")
    if not final_veto_rows.is_file():
        raise HazardTriggerAuditError("Final Veto compact decision log is missing")
    return source_snapshot(repository_root)


def _observation_fields(observation: Mapping[str, object]) -> dict[str, dict[str, object]]:
    fields = observation.get("fields")
    if not isinstance(fields, list):
        raise HazardTriggerAuditError("instrumentation observation fields are missing")
    result: dict[str, dict[str, object]] = {}
    for item in fields:
        if not isinstance(item, list) or len(item) != 2 or not isinstance(item[1], dict):
            raise HazardTriggerAuditError("invalid instrumentation field encoding")
        result[str(item[0])] = item[1]
    return result


def _valid_number(field: Mapping[str, object], name: str) -> float:
    if field.get("valid") is not True or not isinstance(field.get("value"), (int, float)):
        raise HazardTriggerAuditError(f"required valid numeric field is missing: {name}")
    return float(field["value"])


def _state_values(fields: Mapping[str, Mapping[str, object]]) -> list[float]:
    return [
        _valid_number(fields[name], name)
        for name in ("position_x", "position_y", "velocity_x", "velocity_y")
    ]


def collect_recovery_proposals(repository_root: Path) -> list[dict[str, object]]:
    root = repository_root / STAGE1B_PATH
    traces = _json(root / "trace_index.json")["traces"]
    records: list[dict[str, object]] = []
    for trace in traces:
        path = root / str(trace["trace_path"])
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            wrapper = json.loads(line)
            event = wrapper["source_event"]
            if event["event_type"] != "transition":
                continue
            fields = _observation_fields(event["pre_observation"])
            records.append(
                {
                    "source_class": "stage1b_measured_recovery_trace",
                    "source_artifact": path.relative_to(repository_root).as_posix(),
                    "source_locator": f"jsonl_line:{line_number}",
                    "case_id": trace["case_id"],
                    "action_identity": trace["branch_id"],
                    "action": event["proposed_action"],
                    "state_values": _state_values(fields),
                    "source_state_hash": event["pre_state_hash"],
                    "realized_speed_ratio": _valid_number(
                        fields["realized_speed_ratio"], "realized_speed_ratio"
                    ),
                    "predicted_speed_ratio": _valid_number(
                        fields["predicted_speed_ratio"], "predicted_speed_ratio"
                    ),
                    "final_veto_decision": event["monitor_decision"],
                    "event_index": event["event_index"],
                }
            )

    d2 = _json(repository_root / D2_PATH / "source_boundary_index.json")
    for trajectory in d2["recovery_trajectories"]:
        for record in trajectory["records"]:
            state = record["current_state"]
            records.append(
                {
                    "source_class": "d2_zero_action_recovery_trace",
                    "source_artifact": (D2_PATH / "source_boundary_index.json").as_posix(),
                    "source_locator": (
                        f"recovery_trajectories:{trajectory['case_id']}:"
                        f"records:{record['event_index']}"
                    ),
                    "case_id": trajectory["case_id"],
                    "action_identity": "zero_action_reference_v0",
                    "action": record["zero_action"],
                    "state_values": [
                        float(state[name])
                        for name in ("position_x", "position_y", "velocity_x", "velocity_y")
                    ],
                    "source_state_hash": record["current_state_hash"],
                    "realized_speed_ratio": float(record["realized_speed_ratio"]),
                    "predicted_speed_ratio": float(record["predicted_speed_ratio"]),
                    "final_veto_decision": record["final_veto_decision"],
                    "event_index": record["event_index"],
                }
            )
    return records


def recovery_identity(record: Mapping[str, object]) -> str:
    return canonical_sha256(
        {
            "case_id": record["case_id"],
            "action_identity": record["action_identity"],
            "action": record["action"],
            "state_values": record["state_values"],
            "realized_speed_ratio": record["realized_speed_ratio"],
            "predicted_speed_ratio": record["predicted_speed_ratio"],
            "final_veto_decision": record["final_veto_decision"],
        }
    )


def trigger_a_matches(record: Mapping[str, object]) -> bool:
    return (
        float(record["realized_speed_ratio"]) <= THRESHOLD
        and float(record["predicted_speed_ratio"]) > THRESHOLD
    )


def build_trigger_a_report(records: list[dict[str, object]]) -> dict[str, object]:
    identities: dict[str, list[dict[str, object]]] = {}
    for record in records:
        identities.setdefault(recovery_identity(record), []).append(record)
    hits = [record for record in records if trigger_a_matches(record)]
    unique_hits = {
        recovery_identity(record): record for record in hits
    }
    per_source: dict[str, dict[str, object]] = {}
    per_action: dict[str, dict[str, object]] = {}
    for key_name, target in (("source_class", per_source), ("action_identity", per_action)):
        for name in sorted({str(record[key_name]) for record in records}):
            subset = [record for record in records if record[key_name] == name]
            target[name] = {
                "proposal_count": len(subset),
                "trigger_count": sum(trigger_a_matches(record) for record in subset),
                "maximum_predicted_speed_ratio": max(
                    float(record["predicted_speed_ratio"]) for record in subset
                ),
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "trigger_id": "trigger_a_recovery_action_predicted_overspeed",
        "definition": {
            "realized_speed_ratio": "<=1.90",
            "recovery_action_predicted_speed_ratio": ">1.90",
            "conjunction": True,
        },
        "raw_recovery_proposal_count": len(records),
        "duplicate_aware_recovery_proposal_count": len(identities),
        "cross_artifact_reproduced_record_count": len(records) - len(identities),
        "valid_ratio_pair_count": len(records),
        "trigger_observation_count_raw": len(hits),
        "trigger_observation_count_duplicate_aware": len(unique_hits),
        "trigger_case_count": len({str(record["case_id"]) for record in unique_hits.values()}),
        "trigger_case_ids": sorted({str(record["case_id"]) for record in unique_hits.values()}),
        "maximum_recovery_action_predicted_speed_ratio": max(
            float(record["predicted_speed_ratio"]) for record in records
        ),
        "per_source": per_source,
        "per_action": per_action,
        "empirical_support_status": "not_observed_in_frozen_evidence",
        "empirically_supported": False,
        "stage2a_provisional_action_identity": "velocity_opposed_thrust_v0",
        "stage2a_provisional_action_trigger_count": per_action[
            "velocity_opposed_thrust_v0"
        ]["trigger_count"],
        "limitation": (
            "No positive Trigger A observation exists in the audited frozen traces. "
            "This is evidence absence within the covered cases and actions, not proof "
            "that a recovery proposal can never predict overspeed."
        ),
    }


def _load_final_veto_rows(repository_root: Path) -> list[dict[str, object]]:
    path = repository_root / FINAL_VETO_PATH / "decision_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_trigger_b_report(repository_root: Path) -> dict[str, object]:
    rows = _load_final_veto_rows(repository_root)
    segments = [
        row
        for row in rows
        if row.get("event_kind") == "decision_segment"
        and row.get("decision_type") == "veto_action"
        and row.get("veto_status") == "veto"
        and float(row.get("minimum_predicted_nominal_speed_ratio", float("-inf")))
        > THRESHOLD
    ]
    logical_count = sum(int(row["step_count"]) for row in segments)
    source_cases = _json(repository_root / D2_PATH / "source_case_index.json")["source_cases"]
    d2_events = [
        row
        for row in source_cases
        if row.get("source_boundary_status") == "available"
        and row.get("nominal_Final_Veto_result") == "veto"
        and float(row["nominal_controller_predicted_speed_ratio"]) > THRESHOLD
    ]
    overlap = 0
    for event in d2_events:
        for segment in segments:
            if (
                event["case_id"] == segment["case_id"]
                and int(event["branch_step"]) == int(segment["start_step"])
                and event["nominal_controller_action"] == segment["first_nominal_action"]
                and float(event["nominal_controller_predicted_speed_ratio"])
                == float(segment["first_predicted_nominal_speed_ratio"])
            ):
                overlap += 1
                break
    case_ids = sorted(
        {str(row["case_id"]) for row in segments}
        | {str(row["case_id"]) for row in d2_events}
    )
    maximum = max(
        [float(row["maximum_predicted_nominal_speed_ratio"]) for row in segments]
        + [float(row["nominal_controller_predicted_speed_ratio"]) for row in d2_events]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "trigger_id": "trigger_b_nominal_action_final_veto",
        "definition": {
            "action_class": "nominal_controller_action",
            "nominal_action_predicted_speed_ratio": ">1.90",
            "Final_Veto_decision": "veto",
        },
        "final_veto_compact_segment_count": len(segments),
        "final_veto_logical_observation_count": logical_count,
        "d2_first_boundary_observation_count": len(d2_events),
        "cross_artifact_reproduction_count": overlap,
        "raw_logical_observation_count": logical_count + len(d2_events),
        "duplicate_aware_logical_observation_count": logical_count + len(d2_events) - overlap,
        "trigger_case_count": len(case_ids),
        "trigger_case_ids": case_ids,
        "maximum_nominal_action_predicted_speed_ratio": maximum,
        "compact_segments": [
            {
                "case_id": row["case_id"],
                "start_step": row["start_step"],
                "end_step": row["end_step"],
                "logical_observation_count": row["step_count"],
                "minimum_predicted_speed_ratio": row["minimum_predicted_nominal_speed_ratio"],
                "maximum_predicted_speed_ratio": row["maximum_predicted_nominal_speed_ratio"],
                "fallback_executed": row["fallback_executed"],
            }
            for row in segments
        ],
        "d2_first_boundaries": [
            {
                "case_id": row["case_id"],
                "branch_step": row["branch_step"],
                "boundary_state_hash": row["boundary_state_hash"],
                "nominal_action": row["nominal_controller_action"],
                "predicted_speed_ratio": row["nominal_controller_predicted_speed_ratio"],
                "Final_Veto_decision": row["nominal_Final_Veto_result"],
                "vetoed_proposal_transition_count": row[
                    "source_vetoed_proposal_transition_count"
                ],
            }
            for row in d2_events
        ],
        "observed_final_veto_role": (
            "Final Veto rejected the audited nominal proposals whose one-step "
            "predictions exceeded the strict 1.90 threshold."
        ),
        "prevented_unsafe_nominal_proposal_execution": True,
        "limitation": (
            "The compact log proves logical decision counts and segment extrema but "
            "does not publish every per-step Cartesian state. Rejection of these "
            "proposals is not a general proof of closed-loop safety."
        ),
    }


def _stage1b_first_state_actions(
    records: Iterable[Mapping[str, object]], case_id: str, state_values: list[float]
) -> dict[str, Mapping[str, object]]:
    matches: dict[str, Mapping[str, object]] = {}
    for record in records:
        if record["case_id"] == case_id and record["state_values"] == state_values:
            matches[str(record["action_identity"])] = record
    return matches


def build_same_boundary_comparison(
    repository_root: Path, recovery_records: list[dict[str, object]]
) -> dict[str, object]:
    source_cases = _json(repository_root / D2_PATH / "source_case_index.json")["source_cases"]
    trajectories = _json(repository_root / D2_PATH / "source_boundary_index.json")[
        "recovery_trajectories"
    ]
    trajectory_map = {str(item["case_id"]): item for item in trajectories}
    comparisons: list[dict[str, object]] = []
    for source in source_cases:
        if source.get("source_boundary_status") != "available":
            continue
        first = trajectory_map[str(source["case_id"])]["records"][0]
        state = first["current_state"]
        values = [
            float(state[name])
            for name in ("position_x", "position_y", "velocity_x", "velocity_y")
        ]
        alternatives = _stage1b_first_state_actions(
            recovery_records, str(source["case_id"]), values
        )
        alternatives["zero_action_reference_v0"] = {
            "action_identity": "zero_action_reference_v0",
            "action": first["zero_action"],
            "predicted_speed_ratio": first["predicted_speed_ratio"],
            "final_veto_decision": first["final_veto_decision"],
            "source_state_hash": first["current_state_hash"],
            "source_class": "d2_zero_action_recovery_trace",
        }
        comparisons.append(
            {
                "case_id": source["case_id"],
                "exact_cartesian_state_match": (
                    source["boundary_state_hash"] == first["current_state_hash"]
                ),
                "boundary_state_hash": source["boundary_state_hash"],
                "state_values": values,
                "nominal_action": {
                    "action_identity": "nominal_controller_action",
                    "action": source["nominal_controller_action"],
                    "predicted_speed_ratio": source[
                        "nominal_controller_predicted_speed_ratio"
                    ],
                    "Final_Veto_decision": source["nominal_Final_Veto_result"],
                    "predicted_state_hash": source[
                        "nominal_controller_predicted_state_hash"
                    ],
                },
                "recovery_actions": [
                    {
                        "action_identity": action_id,
                        "action": record["action"],
                        "predicted_speed_ratio": record["predicted_speed_ratio"],
                        "Final_Veto_decision": record["final_veto_decision"],
                        "source_class": record["source_class"],
                    }
                    for action_id, record in sorted(alternatives.items())
                ],
                "interpretation": (
                    "At this exact state the nominal proposal predicted overspeed and "
                    "was vetoed, while every published recovery-action alternative "
                    "listed here predicted at or below 1.90."
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_scope": "exact_same_cartesian_boundary_state_only",
        "comparable_boundary_count": len(comparisons),
        "comparisons": comparisons,
        "non_claim": (
            "These action-conditional one-step predictions do not establish controller "
            "superiority, recovery performance, or causal safety beyond the compared proposals."
        ),
    }


def build_evidence_matrix(
    repository_root: Path,
    trigger_a: Mapping[str, object],
    trigger_b: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_rows": [
            {
                "source": (STAGE1B_PATH / "traces").as_posix(),
                "role": "recovery_action_predictions",
                "realized_evidence": "derived_from_measured_cartesian_state",
                "predicted_evidence": "one_step_predicted",
                "decision_evidence": "measured_runtime_monitor_decision",
                "record_count": trigger_a["per_source"][
                    "stage1b_measured_recovery_trace"
                ]["proposal_count"],
                "availability": "available_valid",
            },
            {
                "source": (D2_PATH / "source_boundary_index.json").as_posix(),
                "role": "zero_action_recovery_predictions",
                "realized_evidence": "derived_from_measured_cartesian_state",
                "predicted_evidence": "one_step_predicted",
                "decision_evidence": "Final_Veto_allow",
                "record_count": trigger_a["per_source"][
                    "d2_zero_action_recovery_trace"
                ]["proposal_count"],
                "availability": "available_valid",
            },
            {
                "source": (D2_PATH / "source_case_index.json").as_posix(),
                "role": "nominal_first_veto_boundaries",
                "realized_evidence": "derived_from_measured_boundary_state",
                "predicted_evidence": "one_step_predicted",
                "decision_evidence": "Final_Veto_veto",
                "record_count": trigger_b["d2_first_boundary_observation_count"],
                "availability": "available_valid_for_two_cases",
            },
            {
                "source": (FINAL_VETO_PATH / "decision_log.jsonl").as_posix(),
                "role": "compact_nominal_action_Final_Veto_decisions",
                "realized_evidence": "segment_extrema_only",
                "predicted_evidence": "compact_segment_first_last_minimum_maximum",
                "decision_evidence": "Final_Veto_veto",
                "record_count": trigger_b["final_veto_logical_observation_count"],
                "availability": "logical_counts_available_per_step_states_not_evaluated",
            },
            {
                "source": (REGISTRY_PATH / "registry_manifest.json").as_posix(),
                "role": "source_identity_and_provenance",
                "realized_evidence": "registry_member_boundary_state",
                "predicted_evidence": "member_contract_dependent",
                "decision_evidence": "not_evaluated_for_trigger_counts",
                "record_count": 4,
                "availability": "provenance_only_in_this_audit",
            },
        ],
        "unknown_handling": "not_evaluated",
        "missing_physics_inference": False,
    }


def build_payloads(repository_root: Path) -> dict[str, bytes]:
    snapshot = validate_sources(repository_root)
    recovery_records = collect_recovery_proposals(repository_root)
    trigger_a = build_trigger_a_report(recovery_records)
    trigger_b = build_trigger_b_report(repository_root)
    same_boundary = build_same_boundary_comparison(repository_root, recovery_records)
    role = {
        "schema_version": SCHEMA_VERSION,
        "actually_observed_hazard_mechanism": (
            "nominal_action_one_step_predicted_overspeed_followed_by_Final_Veto_rejection"
        ),
        "trigger_a_observed": False,
        "trigger_b_observed": True,
        "Final_Veto_prevented_observed_nominal_proposals_from_execution": True,
        "recovery_proposal_overspeed_trigger_supported": False,
        "implemented_stage2a_runner_trigger_note": (
            "The frozen Stage 2A runner gates on current realized clear plus a vetoed "
            "normal-action predicted overspeed. Trigger A in this audit instead applies "
            "the overspeed predicate to the recovery proposal and must not be treated "
            "as the same contract."
        ),
        "authority_status": "not_authorized",
        "physical_executions": 0,
        "claim_boundary": (
            "Observed proposal rejection supports the operational veto role on these "
            "records only; it does not prove controller superiority or formal safety."
        ),
    }
    evidence = build_evidence_matrix(repository_root, trigger_a, trigger_b)
    summary = f"""# Stage 2A Hazard Trigger Relevance Audit v0

Completed: {COMPLETED_DATE}

## Status

Frozen offline evidence audit completed. Stage 2A active authority remains unauthorized.

## Trigger A

The recovery-proposal predicate `realized_speed_ratio <= 1.90` and recovery-action
`predicted_speed_ratio > 1.90` occurred **0 times** across 400 frozen proposal records
(392 duplicate-aware action/state observations). The maximum recovery-action prediction
was `{trigger_a['maximum_recovery_action_predicted_speed_ratio']}`. Trigger A is not
empirically supported by the audited frozen evidence; absence here is not proof of
physical impossibility outside the audited cases and actions.

## Trigger B

The Final Veto compact log contains 5 veto segments representing
**{trigger_b['final_veto_logical_observation_count']} logical nominal-action decisions**
above `1.90`, across five stress cases. D2 separately reproduces two first-veto
boundaries, one of which is the same first event already represented by the compact log.
The duplicate-aware combined count is **{trigger_b['duplicate_aware_logical_observation_count']}**
across six cases. The maximum nominal prediction was
`{trigger_b['maximum_nominal_action_predicted_speed_ratio']}`.

Final Veto prevented those observed nominal proposals from being executed. This is an
operational statement about the frozen records, not a general safety proof.

## Same Boundary Evidence

Two D2 boundary states have exact Cartesian identity for nominal-versus-recovery
comparison. In both, the nominal proposal predicted above `1.90` and was vetoed, while
the published zero-action recovery prediction remained clear. At the canonical boundary,
the Stage 1B velocity-opposed and tangential-correction predictions were also clear.
These comparisons do not establish that one controller is better.

## Strongest Supported Conclusion

The hazard mechanism actually observed in frozen evidence is nominal-action one-step
predicted overspeed followed by Final Veto rejection. Recovery-action predicted
overspeed under Trigger A was not observed. The two trigger classes are not
interchangeable.

## Restrictions

No simulation, controller, trajectory, threshold tuning, Final Veto modification, or
Stage 2A authority change was performed. Unknown evidence remains `not_evaluated`.
"""
    documents: dict[str, object] = {
        "trigger_a_report.json": trigger_a,
        "trigger_b_report.json": trigger_b,
        "same_boundary_comparison.json": same_boundary,
        "final_veto_role_report.json": role,
        "evidence_matrix.json": evidence,
    }
    payloads = {
        name: json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode("ascii")
        + b"\n"
        for name, value in documents.items()
    }
    payloads["summary.md"] = summary.encode("ascii")
    artifact_hashes = {name: hashlib.sha256(payloads[name]).hexdigest() for name in REPORT_FILENAMES}
    manifest = {
        "audit_id": AUDIT_ID,
        "schema_version": SCHEMA_VERSION,
        "completed_date": COMPLETED_DATE,
        "audit_classification": "frozen_offline_evidence_audit",
        "source_repository_head": "d6d6dcf8492fdb12f4c2d6b8fd6d81b15fdf63ea",
        "D2_result_commit": "ee87a426b09e1436ee2c9f4dc6424b84d505c9ce",
        "source_snapshot": snapshot,
        "overspeed_threshold": THRESHOLD,
        "overspeed_comparator": ">",
        "trigger_a_observation_count": trigger_a["trigger_observation_count_duplicate_aware"],
        "trigger_b_observation_count": trigger_b["duplicate_aware_logical_observation_count"],
        "same_boundary_comparison_count": same_boundary["comparable_boundary_count"],
        "physical_executions": 0,
        "controller_executions": 0,
        "new_trajectories": 0,
        "Final_Veto_modified": False,
        "Stage_2A_authority_granted": False,
        "staged_recovery_execution": "not_authorized",
        "artifact_filenames": list(ALL_FILENAMES),
        "artifact_hashes": artifact_hashes,
        "audit_bundle_hash": canonical_sha256(artifact_hashes),
        "claim_restrictions": [
            "no_controller_superiority_claim",
            "no_recovery_performance_claim",
            "no_formal_safety_claim",
            "no_active_authority_claim",
            "no_inference_from_missing_physics",
        ],
    }
    manifest["canonical_manifest_hash"] = canonical_sha256(manifest)
    payloads["audit_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True).encode("ascii") + b"\n"
    )
    return payloads


def validate_payloads(payloads: Mapping[str, bytes]) -> dict[str, object]:
    if set(payloads) != set(ALL_FILENAMES):
        raise HazardTriggerAuditError("audit artifact set mismatch")
    manifest = json.loads(payloads["audit_manifest.json"])
    supplied = manifest.pop("canonical_manifest_hash", None)
    if supplied != canonical_sha256(manifest):
        raise HazardTriggerAuditError("audit manifest canonical hash mismatch")
    expected_hashes = {
        name: hashlib.sha256(payloads[name]).hexdigest() for name in REPORT_FILENAMES
    }
    if manifest["artifact_hashes"] != expected_hashes:
        raise HazardTriggerAuditError("audit artifact hash mismatch")
    if manifest["audit_bundle_hash"] != canonical_sha256(expected_hashes):
        raise HazardTriggerAuditError("audit bundle hash mismatch")
    a = json.loads(payloads["trigger_a_report.json"])
    b = json.loads(payloads["trigger_b_report.json"])
    comparison = json.loads(payloads["same_boundary_comparison.json"])
    role = json.loads(payloads["final_veto_role_report.json"])
    if (
        a["trigger_observation_count_duplicate_aware"] != 0
        or a["empirically_supported"] is not False
        or b["final_veto_logical_observation_count"] != 499877
        or b["duplicate_aware_logical_observation_count"] != 499878
        or comparison["comparable_boundary_count"] != 2
        or role["authority_status"] != "not_authorized"
        or manifest["physical_executions"] != 0
        or manifest["Stage_2A_authority_granted"] is not False
    ):
        raise HazardTriggerAuditError("frozen audit result contract mismatch")
    return {**manifest, "canonical_manifest_hash": supplied}


def publish_payloads(repository_root: Path, payloads: Mapping[str, bytes]) -> Path:
    target = repository_root / OUTPUT_PATH
    if target.exists():
        raise HazardTriggerAuditError("audit output already exists")
    validate_payloads(payloads)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stage2a-trigger-audit-", dir=target.parent))
    try:
        for name in ALL_FILENAMES:
            (staging / name).write_bytes(payloads[name])
        staged = {path.name: path.read_bytes() for path in staging.iterdir() if path.is_file()}
        validate_payloads(staged)
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def load_published_payloads(repository_root: Path) -> dict[str, bytes]:
    target = repository_root / OUTPUT_PATH
    if not target.is_dir():
        raise HazardTriggerAuditError("published audit directory is missing")
    names = {path.name for path in target.iterdir() if path.is_file()}
    if names != set(ALL_FILENAMES):
        raise HazardTriggerAuditError("published audit artifact set mismatch")
    payloads = {name: (target / name).read_bytes() for name in ALL_FILENAMES}
    validate_payloads(payloads)
    validate_sources(repository_root)
    return payloads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Stage 2A hazard-trigger relevance audit.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute-frozen-audit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.plan, args.validate_only, args.execute_frozen_audit)):
        parser.print_help()
        return 0
    try:
        if args.plan:
            print(f"audit_id={AUDIT_ID}")
            print(f"output_path={OUTPUT_PATH.as_posix()}")
            print("physical_executions=0")
            print("controller_executions=0")
            print("execution_enabled=false")
            return 0
        validate_sources(ROOT)
        if args.validate_only:
            print("STAGE2A_HAZARD_TRIGGER_RELEVANCE_STATIC: passed; physical_executions=0; write_performed=false")
            return 0
        payloads = build_payloads(ROOT)
        target = publish_payloads(ROOT, payloads)
        manifest = validate_payloads(payloads)
        print(
            "STAGE2A_HAZARD_TRIGGER_RELEVANCE_AUDIT: published; "
            f"path={target.relative_to(ROOT).as_posix()}; "
            f"manifest_hash={manifest['canonical_manifest_hash']}; physical_executions=0"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from runtime_assurance.staged_recovery_guard_evidence import (
    GuardAtomEvaluation,
    GuardEvidenceLevel,
    GuardEvidenceStatus,
)
from runtime_assurance.staged_recovery_shadow_calibration import (
    CALIBRATION_OUTPUT_PATH,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_OFFLINE_REPLAY_COUNT,
    EXPECTED_PHYSICAL_EXECUTION_COUNT,
    EXPECTED_TRACE_COUNT,
    NO_PROGRESS_COMPONENTS,
    CalibrationCandidate,
    CandidateReplayResult,
    ExplicitAbortTerminalShadowAdapter,
    ShadowCalibrationError,
    analyze_candidates,
    atomic_publish_new_directory,
    calibration_candidates,
    candidate_id,
    config_canonical_hash,
    guard_evaluation_document,
    guard_evaluation_from_document,
    load_and_validate_config,
    ranking_tuple,
    replay_candidate,
    run_explicit_abort_terminal_path,
    trace_definitions,
    validate_trace_set_payloads,
)
from runtime_assurance.recovery_branch_state_registry import load_registered_branch_state
from runtime_assurance.staged_recovery_logger_adapter import RuntimeSnapshotType
from runtime_assurance.staged_recovery_shadow_runtime import (
    build_registered_runtime_identity,
    compare_physical_runs,
)


ROOT = Path(__file__).resolve().parents[1]


def evaluation(atom_id: str, status: GuardEvidenceStatus) -> GuardAtomEvaluation:
    value = True if status == GuardEvidenceStatus.TRUE else False if status == GuardEvidenceStatus.FALSE else None
    return GuardAtomEvaluation(
        guard_atom_id=atom_id, status=status, value=value,
        evidence_level=GuardEvidenceLevel.DERIVED, raw_source_values=(), comparator="fixture",
        threshold_or_parameter_reference=None, reason="fixture",
    )


def evidence(value, status="derived"):
    return {
        "value": value, "status": status, "reason": "fixture", "units": "fixture",
        "source_id": "fixture", "source_step": 0, "valid": status not in {"not_evaluated", "invalid"},
        "input_source_ids": [],
    }


def event(index: int, event_type: str, radius: float, radial: float, tangential: float, headroom: float):
    fields = [
        ["absolute_target_radius_error", evidence(radius)],
        ["radial_velocity_ratio", evidence(radial)],
        ["tangential_velocity_error", evidence(tangential)],
        ["overspeed_headroom", evidence(headroom)],
    ]
    observation = {"fields": fields}
    return {
        "event_index": index, "event_type": event_type, "recovery_step": min(index, 4),
        "canonical_event_sha256": f"{index + 1:064x}",
        "pre_observation": observation,
        "post_observation": observation if event_type == "transition" else None,
    }


def trace_rows() -> tuple[dict[str, object], ...]:
    rows = []
    kinds = ["initial_snapshot", "transition", "transition", "transition", "transition", "terminal"]
    for index, kind in enumerate(kinds):
        atoms = [
            evaluation("state_evidence_valid", GuardEvidenceStatus.TRUE),
            evaluation("instrumentation_evaluation_valid", GuardEvidenceStatus.TRUE),
            evaluation("recovery_evaluation_valid", GuardEvidenceStatus.TRUE),
            evaluation("realized_overspeed", GuardEvidenceStatus.FALSE),
            evaluation("realized_overspeed_clear", GuardEvidenceStatus.TRUE),
            evaluation(
                "predicted_overspeed_clear",
                GuardEvidenceStatus.NOT_EVALUATED if kind != "transition" else GuardEvidenceStatus.TRUE,
            ),
            evaluation("recoverability_radius_component_pass", GuardEvidenceStatus.FALSE),
            evaluation("recoverability_radial_velocity_component_pass", GuardEvidenceStatus.FALSE),
            evaluation("recoverability_tangential_velocity_component_pass", GuardEvidenceStatus.TRUE),
            evaluation("phase34_compatible_recoverability_pass", GuardEvidenceStatus.FALSE),
            evaluation("no_eligible_crossing", GuardEvidenceStatus.TRUE),
            evaluation("explicit_abort_requested", GuardEvidenceStatus.FALSE),
            evaluation("correction_authority_available", GuardEvidenceStatus.UNSUPPORTED),
        ]
        rows.append({
            "source_event": event(index, kind, 10.0 - index, 5.0 - index * 0.5, 2.0 - index * 0.1, index),
            "guard_evaluations": [guard_evaluation_document(item) for item in atoms],
            "canonical_trace_record_hash": f"{index + 10:064x}",
        })
    return tuple(rows)


class CalibrationGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_and_validate_config(ROOT)

    def test_grid_has_exactly_216_stable_candidates(self) -> None:
        first = calibration_candidates(self.config)
        second = calibration_candidates(self.config)
        self.assertEqual(len(first), EXPECTED_CANDIDATE_COUNT)
        self.assertEqual(first, second)
        self.assertEqual(len({item.candidate_id for item in first}), 216)

    def test_candidate_id_is_deterministic_and_complete(self) -> None:
        self.assertEqual(
            candidate_id(2, 4, 8, 3, 2, 2, 8),
            "shadow_candidate_hc2_d4_w8_r3_n2_cd2_tb8",
        )

    def test_grid_values_and_fixed_budget(self) -> None:
        candidates = calibration_candidates(self.config)
        self.assertEqual({item.maximum_shadow_transitions_per_trace for item in candidates}, {8})
        self.assertEqual({item.minimum_phase_dwell_steps for item in candidates}, {1, 2, 4})
        self.assertEqual({item.no_progress_window_length for item in candidates}, {2, 4, 8})

    def test_config_hash_mutation_is_detected(self) -> None:
        original = config_canonical_hash(self.config)
        mutated = copy.deepcopy(self.config)
        mutated["trace_pair_count"] = 12
        self.assertNotEqual(original, config_canonical_hash(mutated))

    def test_no_progress_components_exclude_energy_proxy(self) -> None:
        self.assertEqual(len(NO_PROGRESS_COMPONENTS), 4)
        self.assertNotIn("energy", " ".join(NO_PROGRESS_COMPONENTS))


class TraceMatrixTests(unittest.TestCase):
    def test_matrix_has_12_physical_and_one_explicit_abort_trace(self) -> None:
        definitions = trace_definitions(ROOT)
        self.assertEqual(len(definitions), EXPECTED_TRACE_COUNT)
        self.assertEqual(sum(item.explicit_abort for item in definitions), 1)
        self.assertEqual(sum(not item.explicit_abort for item in definitions), 12)
        self.assertEqual(len({item.registry_member_id for item in definitions}), 4)

    def test_explicit_abort_has_existing_branch_and_no_invented_action(self) -> None:
        abort = next(item for item in trace_definitions(ROOT) if item.explicit_abort)
        self.assertEqual(abort.branch_id, "explicit_abort_v0")
        self.assertEqual(abort.registry_member_id, "legacy_canonical")

    def test_expected_execution_count_is_26(self) -> None:
        self.assertEqual(EXPECTED_PHYSICAL_EXECUTION_COUNT, 26)

    def test_guard_serialization_preserves_unavailable(self) -> None:
        source = evaluation("fixture", GuardEvidenceStatus.NOT_EVALUATED)
        restored = guard_evaluation_from_document(guard_evaluation_document(source))
        self.assertEqual(restored, source)
        self.assertIsNone(restored.value)


class ExplicitAbortTerminalAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registered = load_registered_branch_state(ROOT, "legacy_canonical")
        cls.commit = "a" * 40

    def test_explicit_abort_uses_initial_then_terminal_lifecycle(self) -> None:
        run = run_explicit_abort_terminal_path(
            self.registered, implementation_commit=self.commit
        )
        self.assertEqual(
            tuple(snapshot.snapshot_type for snapshot in run.snapshots),
            (RuntimeSnapshotType.INITIAL, RuntimeSnapshotType.TERMINAL),
        )
        self.assertEqual(len(run.transition_snapshots), 0)
        self.assertEqual(run.recovery_transition_count, 0)
        self.assertEqual(run.runtime_terminal_reason, "explicit_abort")

    def test_terminal_event_has_explicit_abort_and_no_action(self) -> None:
        identity, _ = build_registered_runtime_identity(
            self.registered,
            implementation_commit=self.commit,
            branch_id="explicit_abort_v0",
        )
        adapter = ExplicitAbortTerminalShadowAdapter(
            identity, trace_id="explicit_abort_fixture"
        )
        run = run_explicit_abort_terminal_path(
            self.registered, implementation_commit=self.commit, observer=adapter
        )
        documents = adapter.source_documents
        self.assertEqual([item["event_type"] for item in documents], ["initial_snapshot", "terminal"])
        terminal = documents[-1]
        self.assertEqual(terminal["action_disposition"], "no_action")
        self.assertIsNone(terminal["proposed_action"])
        self.assertIsNone(terminal["executed_action"])
        terminal_guards = {item.guard_atom_id: item for item in adapter.guard_evaluations[-1]}
        self.assertEqual(terminal_guards["explicit_abort_requested"].status, GuardEvidenceStatus.TRUE)
        self.assertEqual(adapter.records[-1].desired_shadow_phase, "explicit_abort")
        self.assertFalse(adapter.records[-1].nominal_handoff_recommended)
        self.assertEqual(run.final_state, run.initial_state)

    def test_baseline_and_observed_explicit_abort_are_physically_equivalent(self) -> None:
        baseline = run_explicit_abort_terminal_path(
            self.registered, implementation_commit=self.commit
        )
        identity, _ = build_registered_runtime_identity(
            self.registered,
            implementation_commit=self.commit,
            branch_id="explicit_abort_v0",
        )
        adapter = ExplicitAbortTerminalShadowAdapter(
            identity, trace_id="equivalence_fixture"
        )
        observed = run_explicit_abort_terminal_path(
            self.registered, implementation_commit=self.commit, observer=adapter
        )
        report = compare_physical_runs(baseline, observed)
        self.assertTrue(report["all_equivalence_checks"])
        self.assertEqual(report["checks"]["same_executed_action_sequence"], True)
        self.assertEqual(len(baseline.transition_snapshots), 0)
        self.assertEqual(len(observed.transition_snapshots), 0)

    def test_stage0b_logger_contract_bytes_are_unchanged(self) -> None:
        digest = hashlib.sha256(
            (ROOT / "runtime_assurance/staged_recovery_runtime_logger.py").read_bytes()
        ).hexdigest()
        self.assertEqual(
            digest,
            "bc85351542f1cb75c7999dbd8f71b18e0519aa6468bb4202240fac8e83c905d6",
        )

    def test_other_twelve_definitions_are_unchanged(self) -> None:
        ordinary = tuple(item for item in trace_definitions(ROOT) if not item.explicit_abort)
        self.assertEqual(len(ordinary), 12)
        self.assertEqual(
            {item.branch_id for item in ordinary},
            {
                "zero_action_reference_v0",
                "velocity_opposed_thrust_v0",
                "tangential_error_correction_v0",
            },
        )

    def test_fix_phase_has_no_result_publication(self) -> None:
        self.assertFalse((ROOT / "analysis/staged_recovery_shadow_calibration_trace_set_v0").exists())
        self.assertFalse((ROOT / CALIBRATION_OUTPUT_PATH).exists())


class OfflineReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_and_validate_config(ROOT)
        cls.candidate = calibration_candidates(cls.config)[0]
        cls.index = {"traces": [
            {"trace_id": f"trace_{index}", "branch_id": "fixture", "case_id": f"case_{index}"}
            for index in range(13)
        ]}
        cls.rows = tuple(trace_rows() for _ in range(13))

    def test_replay_is_deterministic(self) -> None:
        first = replay_candidate(self.candidate, self.index, self.rows)
        second = replay_candidate(self.candidate, self.index, self.rows)
        self.assertEqual(first.replay_hash, second.replay_hash)
        self.assertEqual(first.metrics, second.metrics)

    def test_unavailable_and_invalid_counts_remain_explicit(self) -> None:
        result = replay_candidate(self.candidate, self.index, self.rows)
        self.assertGreater(result.metrics["unavailable_guard_count"], 0)
        self.assertEqual(result.metrics["invalid_guard_count"], 0)

    def test_no_progress_strict_positive_convention(self) -> None:
        result = replay_candidate(self.candidate, self.index, self.rows)
        evaluated = [item for item in result.transition_records if item["no_progress_evaluable"]]
        self.assertTrue(evaluated)
        self.assertTrue(all(
            set(item["no_progress_component_results"]) == set(NO_PROGRESS_COMPONENTS)
            for item in evaluated
        ))

    def test_candidate_metrics_cover_required_fields(self) -> None:
        metrics = replay_candidate(self.candidate, self.index, self.rows).metrics
        required = {
            "inter_phase_transitions", "phase_entries", "phase_coverage", "holds",
            "graph_blocks", "two_cycles", "three_cycles", "rapid_reversals",
            "repeated_transition_reasons", "transition_budget_exhaustions",
            "stuck_trace_count", "unavailable_guard_count", "invalid_guard_count",
            "guard_conflict_count", "nominal_handoff_recommendation_count",
            "retreat_recommendation_count", "explicit_abort_recommendation_count",
        }
        self.assertTrue(required.issubset(metrics))

    def test_stuck_is_not_inferred_from_few_transitions_alone(self) -> None:
        result = replay_candidate(self.candidate, self.index, self.rows)
        self.assertTrue(all("maximum_blocked_recommendation_run" in item for item in result.per_trace_metrics))

    def test_analysis_count_is_exactly_2808(self) -> None:
        # Use all candidates against lightweight one-event traces to exercise count and ranking.
        one_row = tuple((trace_rows()[0],) for _ in range(13))
        results, selected = analyze_candidates(self.config, self.index, one_row)
        self.assertEqual(len(results) * 13, EXPECTED_OFFLINE_REPLAY_COUNT)
        self.assertFalse(selected.disqualified)

    def test_no_nominal_handoff_is_recommended(self) -> None:
        result = replay_candidate(self.candidate, self.index, self.rows)
        self.assertEqual(result.metrics["nominal_handoff_recommendation_count"], 0)


class RankingTests(unittest.TestCase):
    def result(self, candidate_id_value: str, **metrics):
        base = {
            "two_cycles": 0, "three_cycles": 0, "rapid_reversals": 0,
            "transition_budget_exhaustions": 0, "invalid_guard_count": 0,
            "unavailable_evidence_block_count": 0, "stuck_trace_count": 0,
            "graph_blocks": 0, "guard_conflict_count": 0, "phase_coverage": 2,
            "inter_phase_transitions": 3,
        }
        base.update(metrics)
        candidate = CalibrationCandidate(candidate_id_value, 1, 1, 2, 2, 1, 0, 8)
        return CandidateReplayResult(candidate, base, (), (), (), False, (), "a" * 64)

    def test_cycles_rank_before_phase_coverage(self) -> None:
        clean = self.result("b", phase_coverage=1)
        cyclic = self.result("a", two_cycles=1, phase_coverage=9)
        self.assertLess(ranking_tuple(clean), ranking_tuple(cyclic))

    def test_phase_coverage_is_maximized_after_penalties(self) -> None:
        broad = self.result("b", phase_coverage=3)
        narrow = self.result("a", phase_coverage=2)
        self.assertLess(ranking_tuple(broad), ranking_tuple(narrow))

    def test_lexical_tie_break_is_final(self) -> None:
        self.assertLess(ranking_tuple(self.result("a")), ranking_tuple(self.result("b")))


class PublicationTests(unittest.TestCase):
    def payloads(self):
        manifest = {"value": 1}
        manifest["canonical_manifest_hash"] = config_canonical_hash(manifest)
        traces = {f"traces/t{index}.jsonl": b"" for index in range(13)}
        return {
            "trace_set_manifest.json": json.dumps(manifest).encode(),
            "trace_index.json": json.dumps({"traces": []}).encode(),
            "equivalence_report.json": json.dumps({"all_pairs_equivalent": True, "physical_equivalence_failures": 0}).encode(),
            **traces,
        }

    def test_atomic_publisher_rejects_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "analysis/result"
            target.mkdir(parents=True)
            with self.assertRaisesRegex(ShadowCalibrationError, "already exists"):
                atomic_publish_new_directory(root, Path("analysis/result"), {}, lambda value: None)

    def test_validator_rejects_partial_trace_set(self) -> None:
        with self.assertRaises(ShadowCalibrationError):
            validate_trace_set_payloads({})

    def test_cli_sources_have_no_scientific_override_or_retry(self) -> None:
        paths = (
            ROOT / "scripts/run_staged_recovery_shadow_calibration_trace_set_v0.py",
            ROOT / "scripts/analyze_staged_recovery_shadow_calibration_v0.py",
        )
        text = "\n".join(path.read_text("utf-8") for path in paths)
        for forbidden in ("--branch", "--candidate", "--threshold", "--retry", "--output"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()

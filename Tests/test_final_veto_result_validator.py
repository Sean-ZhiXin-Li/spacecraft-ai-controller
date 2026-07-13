from __future__ import annotations

import copy
import unittest

from scripts.final_veto_artifacts import ARM_FIELDNAMES, DECISION_FIELDNAMES
from scripts.run_final_veto_ablation import (
    build_pair_record,
    build_planned_jobs,
    load_frozen_manifest,
)
from scripts.check_final_veto_results import (
    PRESERVATION_SUBSET_ID,
    PROJECT_ROOT,
    STRESS_SUBSET_ID,
    validate_decision_events,
    validate_result_records,
)


_RUN_HASH_BY_CASE_ARM = {
    (planned.case_id, planned.arm_id): planned.run_config_hash
    for planned in build_planned_jobs(load_frozen_manifest())
}


def make_arm(
    job,
    arm_id: str,
    *,
    formal: bool = False,
    overspeed: bool = False,
    crossing: bool = True,
    recoverable: bool = True,
    success: bool = True,
    **overrides,
):
    monitor_on = arm_id == "monitor_on"
    row = {field: "" for field in ARM_FIELDNAMES}
    row.update(
        {
            "schema_version": "result_schema_v1",
            "benchmark_id": "recoverability_benchmark",
            "benchmark_version": "v1",
            "experiment_id": job.experiment_id,
            "experiment_status": (
                "formal_executed_pending_validation" if formal else "nonformal_smoke"
            ),
            "implementation_commit": "9" * 40,
            "run_id": f"{job.paired_run_id}__{arm_id}",
            "paired_run_id": job.paired_run_id,
            "case_config_hash": job.case_config_hash,
            "run_config_hash": _RUN_HASH_BY_CASE_ARM[(job.case_id, arm_id)],
            "subset_id": job.subset_id,
            "case_id": job.case_id,
            "arm_id": arm_id,
            "seed": job.seed,
            "controller_id": job.controller_id,
            "controller_family": job.controller_family,
            "artifact_path": "" if not formal else "analysis/final_veto_ablation_v0/results.csv",
            "source_script": "scripts/run_final_veto_ablation.py",
            "monitor_enabled": monitor_on,
            "monitor_id": "one_step_overspeed_veto_v0" if monitor_on else "",
            "hazard_target": "overspeed",
            "hazard_threshold": 1.90,
            "hazard_comparator": ">",
            "r0_over_target": job.r0_over_target,
            "initial_velocity_angle_deg": job.initial_velocity_angle_deg,
            "thrust_scale": job.thrust_scale,
            "crossed_target_radius": crossing,
            "first_crossing_step": 100 if crossing else None,
            "recoverable_crossing": recoverable,
            "final_simulator_success": success,
            "overspeed": overspeed,
            "max_speed_ratio": 1.91 if overspeed else 1.5,
            "instability": False,
            "unsafe_state": False,
            "invalid_simulation": False,
            "terminal_label": "overspeed" if overspeed else ("success" if success else "no_crossing"),
            "precursor_labels": ["target_radius_crossing"] if crossing else [],
            "diagnostic_labels": [],
            "manual_audit_note": "synthetic validator fixture",
            "label_taxonomy_version": "failure_label_taxonomy_v0",
            "is_full_benchmark": False,
            "subset_claim_scope": (
                "protected_preservation_set"
                if job.subset_id == PRESERVATION_SUBSET_ID
                else "diagnostic_stress_set"
            ),
            "regression_set_membership": (
                ["known_phase34_recoverable_preservation"]
                if job.subset_id == PRESERVATION_SUBSET_ID
                else ["diagnostic_stress"]
            ),
            "known_phase34_recoverable_case": job.subset_id == PRESERVATION_SUBSET_ID,
            "monitor_evaluation_count": 10 if monitor_on else 0,
            "allow_count": 9 if monitor_on else 0,
            "veto_count": 1 if monitor_on else 0,
            "fallback_count": 1 if monitor_on else 0,
            "false_negative_count": 0,
            "fallback_failure_count": 0,
            "invalid_monitor_evaluation_count": 0,
            "nominal_actions_unchanged_count": 9 if monitor_on else 10,
            "steps": 100,
            "accepted_as_progress": False,
            "acceptance_reason": "synthetic fixture; aggregate validation pending",
            "is_formal_experiment": formal,
        }
    )
    row.update(overrides)
    return row


class FinalVetoResultValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_frozen_manifest()
        cls.jobs = build_planned_jobs(cls.manifest)
        cls.job = cls.jobs[0]

    def fixture(self):
        off = make_arm(self.job, "monitor_off")
        on = make_arm(self.job, "monitor_on")
        return [off, on], [build_pair_record([off, on])]

    def report(self, arms=None, pairs=None, manifest=None, **kwargs):
        if arms is None or pairs is None:
            arms, pairs = self.fixture()
        return validate_result_records(
            arms,
            pairs,
            manifest or self.manifest,
            **kwargs,
        )

    def test_valid_nonformal_pair_is_structurally_valid_but_not_claim_eligible(self) -> None:
        report = self.report()
        self.assertTrue(report.structural_valid)
        self.assertTrue(report.pair_complete)
        self.assertEqual(report.preservation_acceptance, "not_evaluated_nonformal")
        self.assertFalse(report.positive_claim_eligible)

    def test_missing_arm_is_rejected(self) -> None:
        arms, _ = self.fixture()
        report = self.report(arms=arms[:1], pairs=[])
        self.assertFalse(report.structural_valid)
        self.assertFalse(report.pair_complete)

    def test_duplicate_arm_is_rejected(self) -> None:
        arms, pairs = self.fixture()
        report = self.report(arms=[arms[0], copy.deepcopy(arms[0]), arms[1]], pairs=pairs)
        self.assertFalse(report.structural_valid)
        self.assertTrue(any("duplicate pair/arm" in error for error in report.errors))

    def test_mismatched_case_hash_is_rejected(self) -> None:
        arms, pairs = self.fixture()
        arms[1]["case_config_hash"] = "drift"
        report = self.report(arms=arms, pairs=pairs)
        self.assertFalse(report.structural_valid)
        self.assertTrue(any("mismatched case hashes" in error for error in report.errors))

    def test_threshold_drift_to_1_91_is_rejected(self) -> None:
        arms, pairs = self.fixture()
        arms[1]["hazard_threshold"] = 1.91
        report = self.report(arms=arms, pairs=pairs)
        self.assertFalse(report.structural_valid)
        self.assertTrue(any("strict > 1.90" in error for error in report.errors))

    def test_comparator_drift_is_rejected(self) -> None:
        arms, pairs = self.fixture()
        arms[1]["hazard_comparator"] = ">="
        report = self.report(arms=arms, pairs=pairs)
        self.assertFalse(report.structural_valid)

    def test_formal_result_marked_from_smoke_mode_is_rejected(self) -> None:
        arms, _ = self.fixture()
        for row in arms:
            row["is_formal_experiment"] = True
        pairs = [build_pair_record(arms)]
        report = self.report(arms=arms, pairs=pairs)
        self.assertFalse(report.structural_valid)
        self.assertTrue(any("smoke status as formal" in error for error in report.errors))

    def test_protected_output_directory_is_rejected(self) -> None:
        report = self.report(
            output_directory=PROJECT_ROOT / "analysis" / "phase34_post_cross_sync"
        )
        self.assertFalse(report.structural_valid)
        self.assertTrue(any("protected path" in error for error in report.errors))

    def test_monitor_off_veto_activity_is_rejected(self) -> None:
        arms, pairs = self.fixture()
        arms[0]["veto_count"] = 1
        report = self.report(arms=arms, pairs=pairs)
        self.assertFalse(report.structural_valid)
        self.assertTrue(any("monitor activity in monitor_off" in error for error in report.errors))

    def test_negative_monitor_count_is_rejected(self) -> None:
        arms, pairs = self.fixture()
        arms[1]["veto_count"] = -1
        report = self.report(arms=arms, pairs=pairs)
        self.assertFalse(report.structural_valid)
        self.assertTrue(any("negative monitor count" in error for error in report.errors))

    def test_allow_plus_veto_must_match_evaluations(self) -> None:
        arms, pairs = self.fixture()
        arms[1]["allow_count"] = 8
        report = self.report(arms=arms, pairs=pairs)
        self.assertFalse(report.structural_valid)
        self.assertTrue(any("does not equal evaluations" in error for error in report.errors))

    def test_pair_cannot_claim_avoided_failure_without_off_hazard(self) -> None:
        arms, pairs = self.fixture()
        pairs[0]["avoided_failure"] = True
        report = self.report(arms=arms, pairs=pairs)
        self.assertFalse(report.structural_valid)
        self.assertTrue(any("without a valid hazard counterfactual" in error for error in report.errors))

    def test_pair_cannot_claim_blocked_success_without_lost_success(self) -> None:
        arms, pairs = self.fixture()
        pairs[0]["blocked_success"] = True
        report = self.report(arms=arms, pairs=pairs)
        self.assertFalse(report.structural_valid)
        self.assertTrue(any("without lost off-arm success" in error for error in report.errors))

    def test_formal_safety_claim_is_rejected(self) -> None:
        arms, pairs = self.fixture()
        arms[1]["formal_safety_claim"] = True
        report = self.report(arms=arms, pairs=pairs)
        self.assertFalse(report.structural_valid)
        self.assertTrue(any("formal-safety claim" in error for error in report.errors))

    def test_manifest_with_measured_results_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["measured_results"] = {"avoided_failure_count": 1}
        report = self.report(manifest=manifest)
        self.assertFalse(report.structural_valid)
        self.assertTrue(any("frozen manifest" in error for error in report.errors))

    def test_decision_event_cannot_embed_pair_counterfactual_claim(self) -> None:
        event = {field: None for field in DECISION_FIELDNAMES}
        event.update(
            {
                "decision_schema_version": "decision_log_schema_v0",
                "decision_id": "decision-1",
                "arm_id": "monitor_on",
                "hazard_threshold": 1.90,
                "hazard_comparator": ">",
                "avoided_failure": True,
            }
        )
        errors = validate_decision_events([event])
        self.assertTrue(any("pair-level counterfactual" in error for error in errors))

    def formal_fixture(self, *, exercise_stress: bool = True):
        arms = []
        for job in self.jobs:
            preservation = job.subset_id == PRESERVATION_SUBSET_ID
            off_hazard = (not preservation) and exercise_stress and job.arm_id == "monitor_off"
            row = make_arm(
                job,
                job.arm_id,
                formal=True,
                overspeed=off_hazard,
                crossing=preservation,
                recoverable=preservation,
                success=preservation,
            )
            if job.arm_id == "monitor_on":
                row["monitor_evaluation_count"] = 10
                row["allow_count"] = 9
                row["veto_count"] = 1
                row["fallback_count"] = 1
            arms.append(row)
        grouped = {}
        for row in arms:
            grouped.setdefault(row["paired_run_id"], []).append(row)
        pairs = [build_pair_record(grouped[pair_id]) for pair_id in sorted(grouped)]
        return arms, pairs

    def test_synthetic_complete_formal_fixture_exercises_all_acceptance_checks(self) -> None:
        arms, pairs = self.formal_fixture()
        report = self.report(arms=arms, pairs=pairs)
        self.assertTrue(report.structural_valid, report.errors)
        self.assertTrue(report.pair_complete)
        self.assertEqual(report.preservation_acceptance, "pass")
        self.assertEqual(report.stress_hazard_exercised, "pass")
        self.assertTrue(report.positive_claim_eligible)
        self.assertEqual(report.metrics["arm_rows"], 26)
        self.assertEqual(report.metrics["pair_rows"], 13)

    def test_no_fresh_monitor_off_hazard_means_monitor_not_exercised(self) -> None:
        arms, pairs = self.formal_fixture(exercise_stress=False)
        report = self.report(arms=arms, pairs=pairs)
        self.assertTrue(report.structural_valid, report.errors)
        self.assertEqual(report.stress_hazard_exercised, "monitor_not_exercised")
        self.assertFalse(report.positive_claim_eligible)


if __name__ == "__main__":
    unittest.main()

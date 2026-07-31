from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from functools import lru_cache
from pathlib import Path
from unittest import mock

import runtime_assurance.staged_recovery_guard_evidence as guard


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts/analyze_staged_recovery_guard_evidence_v0.py"
CHECKER = ROOT / "scripts/check_staged_recovery_guard_evidence.py"
SOURCE_DIRECTORY = ROOT / guard.SOURCE_STAGE0C_DIRECTORY


@lru_cache(maxsize=1)
def source_bundle():
    return guard.load_validated_source_bundle(ROOT)


@lru_cache(maxsize=1)
def source_events():
    return source_bundle().event_documents()


@lru_cache(maxsize=1)
def raw_source_events():
    return tuple(
        json.loads(line)
        for line in (SOURCE_DIRECTORY / "staged_recovery_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )


@lru_cache(maxsize=1)
def definitions():
    return guard.guard_atom_definitions()


@lru_cache(maxsize=1)
def guard_trace():
    return guard.build_guard_evaluation_trace(source_events(), definitions())


@lru_cache(maxsize=1)
def analysis_payloads():
    return guard.build_analysis_payloads(ROOT, implementation_commit="a" * 40)


def atom_map(event: dict[str, object], previous: dict[str, object] | None = None):
    return {
        item.guard_atom_id: item
        for item in guard.evaluate_guard_atoms_for_event(event, previous, definitions())
    }


def selected_observation(event: dict[str, object]) -> dict[str, object]:
    key = "post_observation" if event.get("event_type") == "transition" else "pre_observation"
    value = event.get(key)
    if not isinstance(value, dict):
        raise AssertionError(f"event has no selected observation: {key}")
    return value


def set_evidence(
    event: dict[str, object],
    field_id: str,
    value: object,
    *,
    status: str = "derived",
    valid: bool = True,
    context: str = "selected",
) -> None:
    if context == "selected":
        observation = selected_observation(event)
        pairs = observation.get("fields")
    else:
        container = event.get(context)
        if context == "predicted_observation":
            if not isinstance(container, dict):
                template = copy.deepcopy(source_events()[1]["predicted_observation"])
                event[context] = template
                container = template
            pairs = container.get("fields") if isinstance(container, dict) else None
        else:
            pairs = container
    if isinstance(pairs, dict):
        evidence = pairs.get(field_id)
        if isinstance(evidence, dict):
            evidence.update(value=value, status=status, valid=valid)
        else:
            pairs[field_id] = {
                "value": value,
                "status": status,
                "reason": "synthetic_test_input",
                "units": "dimensionless",
                "source_id": "test",
                "source_step": event.get("recovery_step"),
                "valid": valid,
                "input_source_ids": [],
            }
        return
    if not isinstance(pairs, list):
        raise AssertionError(f"event context has no field list: {context}")
    for pair in pairs:
        if isinstance(pair, list) and len(pair) == 2 and pair[0] == field_id:
            if not isinstance(pair[1], dict):
                raise AssertionError(f"malformed evidence for {field_id}")
            pair[1].update(value=value, status=status, valid=valid)
            return
    pairs.append(
        [
            field_id,
            {
                "value": value,
                "status": status,
                "reason": "synthetic_test_input",
                "units": "dimensionless",
                "source_id": "test",
                "source_step": event.get("recovery_step"),
                "valid": valid,
                "input_source_ids": [],
            },
        ]
    )


def status(atoms, atom_id: str) -> str:
    return atoms[atom_id].status.value


def copy_stage0c(root: Path) -> Path:
    target = root / guard.SOURCE_STAGE0C_DIRECTORY
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_DIRECTORY, target)
    return target


def rewrite_document(path: Path, mutation) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutation(document)
    document = guard.with_document_hash(document)
    path.write_bytes(guard.pretty_json_bytes(document))


def load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load script {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourceValidationTests(unittest.TestCase):
    def test_valid_stage0c_source_is_accepted(self) -> None:
        report = guard.validate_source_bundle(ROOT)
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.source_event_count, 10)
        self.assertEqual(report.source_transition_count, 8)

    def test_trace_manifest_and_aggregate_hashes_recompute(self) -> None:
        bundle = source_bundle()
        trace_manifest = bundle.document("trace_manifest")
        events = raw_source_events()
        self.assertTrue(guard.document_hash_recomputes(trace_manifest))
        self.assertEqual(
            guard.aggregate_source_event_hashes(events),
            guard.SOURCE_TRACE_AGGREGATE_HASH,
        )
        self.assertTrue(all(guard.source_event_hash_recomputes(item) for item in events))

    def test_event_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = copy_stage0c(Path(directory))
            trace = source / "staged_recovery_trace.jsonl"
            lines = trace.read_text(encoding="utf-8").splitlines()
            trace.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            report = guard.validate_source_bundle(Path(directory))
            self.assertFalse(report.valid)
            self.assertIn("source_event_count_mismatch", report.errors)

    def test_event_order_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = copy_stage0c(Path(directory))
            trace = source / "staged_recovery_trace.jsonl"
            lines = trace.read_text(encoding="utf-8").splitlines()
            lines[1], lines[2] = lines[2], lines[1]
            trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = guard.validate_source_bundle(Path(directory))
            self.assertFalse(report.valid)
            self.assertIn("source_event_order_mismatch", report.errors)

    def test_source_event_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = copy_stage0c(Path(directory))
            trace = source / "staged_recovery_trace.jsonl"
            events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            events[1]["recovery_step"] = 999
            trace.write_text(
                "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in events),
                encoding="utf-8",
            )
            report = guard.validate_source_bundle(Path(directory))
            self.assertFalse(report.valid)
            self.assertIn("source_event_hash_mismatch", report.errors)

    def test_wrong_or_synthetic_trace_classification_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = copy_stage0c(Path(directory))
            rewrite_document(
                source / "trace_manifest.json",
                lambda value: value.__setitem__("trace_classification", "synthetic"),
            )
            report = guard.validate_source_bundle(Path(directory))
            self.assertFalse(report.valid)
            self.assertIn("source_trace_is_not_measured_validation", report.errors)

    def test_missing_equivalence_success_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = copy_stage0c(Path(directory))
            rewrite_document(
                source / "equivalence_report.json",
                lambda value: value.__setitem__("all_equivalence_checks", False),
            )
            report = guard.validate_source_bundle(Path(directory))
            self.assertFalse(report.valid)
            self.assertIn("source_equivalence_not_complete", report.errors)


class EvidenceStatusTests(unittest.TestCase):
    def test_true_and_false_atoms_preserve_values(self) -> None:
        atoms = atom_map(copy.deepcopy(source_events()[0]))
        self.assertEqual(status(atoms, "realized_overspeed"), "false")
        self.assertIs(atoms["realized_overspeed"].value, False)
        self.assertEqual(status(atoms, "realized_overspeed_clear"), "true")
        self.assertIs(atoms["realized_overspeed_clear"].value, True)

    def test_missing_invalid_unsupported_and_policy_unresolved_stay_distinct(self) -> None:
        event = copy.deepcopy(raw_source_events()[0])
        set_evidence(event, "realized_speed_ratio", None, status="not_evaluated", valid=False)
        set_evidence(event, "radius_error_ratio", None, status="invalid", valid=False)
        atoms = atom_map(event)
        self.assertEqual(status(atoms, "realized_overspeed"), "not_evaluated")
        self.assertIsNone(atoms["realized_overspeed"].value)
        self.assertEqual(status(atoms, "recoverability_radius_component_pass"), "invalid")
        self.assertEqual(status(atoms, "correction_authority_available"), "unsupported")
        self.assertEqual(status(atoms, "no_progress_policy_evaluable"), "policy_unresolved")
        self.assertIsNone(atoms["correction_authority_available"].value)
        self.assertIsNone(atoms["no_progress_policy_evaluable"].value)

    def test_true_atom_does_not_authorize_phase_transition(self) -> None:
        atoms = atom_map(copy.deepcopy(source_events()[0]))
        self.assertEqual(atoms["realized_overspeed_clear"].policy_authorization_status, "not_authorized")

    def test_immutable_definition_rejects_mutation(self) -> None:
        definition = definitions()[0]
        with self.assertRaises(FrozenInstanceError):
            definition.guard_atom_id = "changed"  # type: ignore[misc]

    def test_invalid_guard_constructor_rejects_favorable_value(self) -> None:
        with self.assertRaises(guard.GuardEvidenceError):
            guard.GuardAtomEvaluation(
                guard_atom_id="test",
                status=guard.GuardEvidenceStatus.NOT_EVALUATED,
                value=False,
                evidence_level=guard.GuardEvidenceLevel.NOT_EVALUATED,
                raw_source_values=(),
                comparator="none",
                threshold_or_parameter_reference=None,
                reason="test",
            )


class SignalProfileTests(unittest.TestCase):
    def test_numeric_profile_statistics_are_exact(self) -> None:
        events = (
            {"event_index": 0, "event_type": "initial_snapshot", "sample": -1.0},
            {"event_index": 1, "event_type": "transition", "sample": 1.0},
            {"event_index": 2, "event_type": "transition", "sample": 3.0},
        )
        profile = {item.field_id: item for item in guard.build_measured_signal_profiles(events)}["event.sample"]
        stats = dict(profile.statistics)
        self.assertEqual(stats["valid_count"], 3)
        self.assertEqual(stats["first_valid_value"], -1.0)
        self.assertEqual(stats["last_valid_value"], 3.0)
        self.assertEqual(stats["minimum"], -1.0)
        self.assertEqual(stats["maximum"], 3.0)
        self.assertEqual(stats["arithmetic_change_first_to_last"], 4.0)
        self.assertEqual(stats["positive_delta_count"], 2)
        self.assertEqual(stats["sign_change_count"], 1)
        self.assertTrue(stats["monotonic_nondecreasing"])
        self.assertFalse(stats["constant"])

    def test_decreasing_and_constant_profiles_are_detected(self) -> None:
        decreasing = tuple(
            {"event_index": index, "event_type": "transition", "sample": value}
            for index, value in enumerate((3.0, 2.0, 1.0))
        )
        constant = tuple(
            {"event_index": index, "event_type": "transition", "sample": 2.0}
            for index in range(3)
        )
        d = dict({item.field_id: item for item in guard.build_measured_signal_profiles(decreasing)}["event.sample"].statistics)
        c = dict({item.field_id: item for item in guard.build_measured_signal_profiles(constant)}["event.sample"].statistics)
        self.assertTrue(d["monotonic_nonincreasing"])
        self.assertEqual(d["negative_delta_count"], 2)
        self.assertTrue(c["constant"])
        self.assertEqual(c["zero_delta_count"], 2)

    def test_missing_values_are_not_zero_filled_and_invalid_values_are_counted(self) -> None:
        events = (
            {"event_index": 0, "event_type": "initial_snapshot", "sample": 2.0},
            {"event_index": 1, "event_type": "transition"},
            {"event_index": 2, "event_type": "terminal", "sample": object()},
        )
        profile = {item.field_id: item for item in guard.build_measured_signal_profiles(events)}["event.sample"]
        stats = dict(profile.statistics)
        self.assertEqual(stats["valid_count"], 1)
        self.assertEqual(stats["not_evaluated_count"], 1)
        self.assertEqual(stats["invalid_count"], 1)
        self.assertEqual(stats["first_valid_value"], 2.0)

    def test_boolean_and_categorical_profiles_have_no_numeric_mean(self) -> None:
        events = (
            {"event_index": 0, "event_type": "initial_snapshot", "flag": False, "label": "a"},
            {"event_index": 1, "event_type": "transition", "flag": True, "label": "b"},
        )
        profiles = {item.field_id: item for item in guard.build_measured_signal_profiles(events)}
        flag = dict(profiles["event.flag"].statistics)
        label = dict(profiles["event.label"].statistics)
        self.assertEqual(flag["true_count"], 1)
        self.assertEqual(flag["false_count"], 1)
        self.assertEqual(flag["value_change_count"], 1)
        self.assertNotIn("mean", label)
        self.assertEqual(label["value_change_count"], 1)

    def test_vector_components_and_magnitude_are_profiled_deterministically(self) -> None:
        events = (
            {"event_index": 0, "event_type": "initial_snapshot", "vector": [3.0, 4.0]},
            {"event_index": 1, "event_type": "transition", "vector": [0.0, 5.0]},
        )
        profiles = {item.field_id: item for item in guard.build_measured_signal_profiles(events)}
        self.assertIn("event.vector.component_0", profiles)
        self.assertIn("event.vector.component_1", profiles)
        magnitude = dict(profiles["event.vector.magnitude"].statistics)
        self.assertEqual(magnitude["first_valid_value"], 5.0)
        self.assertEqual(magnitude["last_valid_value"], 5.0)

    def test_observed_delta_is_labeled_as_resolution_not_noise(self) -> None:
        profiles = guard.build_measured_signal_profiles(source_events())
        numeric = next(item for item in profiles if item.profile_kind == "numeric")
        self.assertIn("trace resolution statistic", numeric.scientific_limitation)
        self.assertNotIn("noise floor", numeric.scientific_limitation)


class ExactGuardAtomTests(unittest.TestCase):
    def test_realized_overspeed_strict_boundary(self) -> None:
        event = copy.deepcopy(source_events()[0])
        for ratio, expected in ((1.89, False), (1.90, False), (1.9000001, True)):
            set_evidence(event, "realized_speed_ratio", ratio)
            atoms = atom_map(event)
            self.assertIs(atoms["realized_overspeed"].value, expected)
            self.assertIs(atoms["realized_overspeed_clear"].value, not expected)

    def test_predicted_and_realized_hazard_atoms_remain_separate(self) -> None:
        event = copy.deepcopy(source_events()[1])
        set_evidence(event, "realized_speed_ratio", 1.0)
        set_evidence(event, "predicted_speed_ratio", 2.0, status="one_step_predicted", context="predicted_observation")
        atoms = atom_map(event, copy.deepcopy(source_events()[0]))
        self.assertEqual(status(atoms, "realized_overspeed"), "false")
        self.assertEqual(status(atoms, "predicted_overspeed"), "true")

    def test_missing_and_invalid_predicted_ratio_stay_explicit(self) -> None:
        initial = copy.deepcopy(source_events()[0])
        self.assertEqual(status(atom_map(initial), "predicted_overspeed"), "not_evaluated")
        event = copy.deepcopy(source_events()[1])
        set_evidence(event, "predicted_speed_ratio", None, status="invalid", valid=False, context="predicted_observation")
        self.assertEqual(status(atom_map(event, initial), "predicted_overspeed"), "invalid")

    def test_phase34_exact_boundaries_pass(self) -> None:
        event = copy.deepcopy(source_events()[0])
        set_evidence(event, "radius_error_ratio", 0.0025)
        set_evidence(event, "radial_velocity_ratio", -0.02)
        set_evidence(event, "tangential_velocity_error_ratio", 0.25)
        atoms = atom_map(event)
        self.assertEqual(status(atoms, "recoverability_radius_component_pass"), "true")
        self.assertEqual(status(atoms, "recoverability_radial_velocity_component_pass"), "true")
        self.assertEqual(status(atoms, "recoverability_tangential_velocity_component_pass"), "true")
        self.assertEqual(status(atoms, "phase34_compatible_recoverability_pass"), "true")

    def test_each_just_above_phase34_boundary_fails(self) -> None:
        cases = (
            ("radius_error_ratio", 0.002500001, "recoverability_radius_component_pass"),
            ("radial_velocity_ratio", 0.020000001, "recoverability_radial_velocity_component_pass"),
            ("tangential_velocity_error_ratio", 0.250000001, "recoverability_tangential_velocity_component_pass"),
        )
        for field_id, value, atom_id in cases:
            event = copy.deepcopy(source_events()[0])
            set_evidence(event, "radius_error_ratio", 0.0)
            set_evidence(event, "radial_velocity_ratio", 0.0)
            set_evidence(event, "tangential_velocity_error_ratio", 0.0)
            set_evidence(event, field_id, value)
            atoms = atom_map(event)
            self.assertEqual(status(atoms, atom_id), "false")
            self.assertEqual(status(atoms, "phase34_compatible_recoverability_pass"), "false")

    def test_missing_component_prevents_combined_true(self) -> None:
        event = copy.deepcopy(source_events()[0])
        set_evidence(event, "radius_error_ratio", 0.0)
        set_evidence(event, "radial_velocity_ratio", None, status="not_evaluated", valid=False)
        set_evidence(event, "tangential_velocity_error_ratio", 0.0)
        atoms = atom_map(event)
        self.assertEqual(status(atoms, "phase34_compatible_recoverability_pass"), "not_evaluated")


class DirectionalAndCrossingTests(unittest.TestCase):
    def directional_atoms(self, previous_values: dict[str, float], current_values: dict[str, float]):
        previous = copy.deepcopy(source_events()[0])
        current = copy.deepcopy(source_events()[1])
        for key, value in previous_values.items():
            set_evidence(previous, key, value)
        for key, value in current_values.items():
            set_evidence(current, key, value)
        return atom_map(current, previous)

    def test_radius_gap_improving_unchanged_and_worsening(self) -> None:
        for current, atom_id in (
            (9.0, "absolute_radius_gap_improving"),
            (10.0, "absolute_radius_gap_unchanged"),
            (11.0, "absolute_radius_gap_worsening"),
        ):
            atoms = self.directional_atoms(
                {"absolute_target_radius_error": 10.0},
                {"absolute_target_radius_error": current},
            )
            self.assertEqual(status(atoms, atom_id), "true")

    def test_radial_direction_all_sign_cases_and_zero(self) -> None:
        cases = (
            (-1.0, 1.0, "radial_velocity_toward_target"),
            (-1.0, -1.0, "radial_velocity_away_from_target"),
            (1.0, -1.0, "radial_velocity_toward_target"),
            (1.0, 1.0, "radial_velocity_away_from_target"),
            (0.0, 1.0, "radial_velocity_no_directional_commitment"),
        )
        for gap, velocity, atom_id in cases:
            event = copy.deepcopy(source_events()[0])
            set_evidence(event, "signed_target_radius_error", gap)
            set_evidence(event, "radial_velocity", velocity)
            self.assertEqual(status(atom_map(event), atom_id), "true")

    def test_tangential_error_and_headroom_direction(self) -> None:
        improving = self.directional_atoms(
            {"tangential_velocity_error": -5.0, "overspeed_headroom": 0.1},
            {"tangential_velocity_error": -4.0, "overspeed_headroom": 0.2},
        )
        worsening = self.directional_atoms(
            {"tangential_velocity_error": -5.0, "overspeed_headroom": 0.2},
            {"tangential_velocity_error": -6.0, "overspeed_headroom": 0.1},
        )
        self.assertEqual(status(improving, "absolute_tangential_error_improving"), "true")
        self.assertEqual(status(improving, "overspeed_headroom_improving"), "true")
        self.assertEqual(status(worsening, "absolute_tangential_error_worsening"), "true")
        self.assertEqual(status(worsening, "overspeed_headroom_worsening"), "true")

    def test_energy_proxy_remains_diagnostic(self) -> None:
        atoms = self.directional_atoms(
            {"specific_energy_error": 5.0}, {"specific_energy_error": 4.0}
        )
        item = atoms["absolute_energy_proxy_error_improving"]
        self.assertEqual(item.status.value, "true")
        self.assertEqual(item.evidence_level.value, "diagnostic_proxy")

    def test_crossing_eligible_prebranch_and_none_are_distinct(self) -> None:
        for crossing, eligible, expected in (
            (True, True, "eligible_target_radius_crossing"),
            (True, False, "pre_branch_crossing_only"),
            (False, False, "no_eligible_crossing"),
        ):
            event = copy.deepcopy(source_events()[1])
            set_evidence(event, "target_radius_crossing", crossing)
            set_evidence(event, "crossing_recovery_eligible", eligible)
            atoms = atom_map(event, copy.deepcopy(source_events()[0]))
            self.assertEqual(status(atoms, expected), "true")

    def test_initial_and_terminal_do_not_fabricate_crossing(self) -> None:
        initial = atom_map(copy.deepcopy(source_events()[0]))
        terminal = atom_map(copy.deepcopy(source_events()[-1]), copy.deepcopy(source_events()[-2]))
        self.assertEqual(status(initial, "eligible_target_radius_crossing"), "not_evaluated")
        self.assertEqual(status(terminal, "eligible_target_radius_crossing"), "not_evaluated")

    def test_crossing_atoms_create_no_interpolation(self) -> None:
        definition_fields = {item.guard_atom_id: item.required_fields for item in definitions()}
        self.assertNotIn("crossing_interpolation_fraction", definition_fields["eligible_target_radius_crossing"])

    def test_explicit_abort_uses_supplied_abort_signal(self) -> None:
        event = copy.deepcopy(source_events()[0])
        set_evidence(event, "explicit_abort_requested", True, status="measured")
        atoms = atom_map(event)
        self.assertEqual(status(atoms, "explicit_abort_requested"), "true")
        self.assertEqual(atoms["explicit_abort_requested"].policy_authorization_status, "not_authorized")


class GuardTraceAndWindowTests(unittest.TestCase):
    def test_one_guard_record_per_source_event_and_source_hashes_preserved(self) -> None:
        trace = guard_trace()
        events = source_events()
        self.assertEqual(len(trace), len(events))
        self.assertEqual([item["event_index"] for item in trace], list(range(10)))
        self.assertEqual(
            [item["source_event_scientific_hash"] for item in trace],
            [item["canonical_event_sha256"] for item in events],
        )

    def test_initial_transition_and_terminal_semantics_are_preserved(self) -> None:
        trace = guard_trace()
        initial = {item["guard_atom_id"]: item for item in trace[0]["guard_atoms"]}
        transition = {item["guard_atom_id"]: item for item in trace[1]["guard_atoms"]}
        self.assertEqual(initial["absolute_radius_gap_improving"]["status"], "not_evaluated")
        self.assertIn(transition["absolute_radius_gap_improving"]["status"], {"true", "false"})
        self.assertEqual(trace[-1]["recovery_step"], trace[-2]["recovery_step"])
        self.assertFalse(trace[-1]["action_generated"])
        self.assertFalse(trace[-1]["phase_selected"])
        self.assertFalse(trace[-1]["stop_condition_selected"])

    def test_guard_trace_hashes_and_aggregate_are_deterministic(self) -> None:
        first = guard.build_guard_evaluation_trace(source_events(), definitions())
        second = guard.build_guard_evaluation_trace(source_events(), definitions())
        self.assertEqual(first, second)
        self.assertTrue(all(guard.guard_evaluation_event_hash_recomputes(item) for item in first))
        self.assertEqual(
            guard.aggregate_guard_evaluation_hashes(first),
            guard.aggregate_guard_evaluation_hashes(second),
        )

    def test_source_mutation_changes_derived_guard_trace(self) -> None:
        events = list(copy.deepcopy(source_events()))
        set_evidence(events[1], "realized_speed_ratio", 2.0)
        changed = guard.build_guard_evaluation_trace(events, definitions())
        self.assertNotEqual(changed[1]["canonical_guard_evaluation_event_hash"], guard_trace()[1]["canonical_guard_evaluation_event_hash"])

    def test_window_lengths_and_counts_cover_all_structural_windows(self) -> None:
        records = guard.build_windowed_progress_records(source_events())
        counts = {
            length: sum(item.window_length == length for item in records)
            for length in range(1, 9)
        }
        self.assertEqual(counts, {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1})

    def test_window_component_changes_are_raw_and_separate(self) -> None:
        record = next(item for item in guard.build_windowed_progress_records(source_events()) if item.window_length == 8)
        changes = dict(record.component_changes)
        labels = dict(record.component_labels)
        self.assertLess(changes["absolute_radius_gap_change"], 0.0)
        self.assertLess(changes["absolute_tangential_error_change"], 0.0)
        self.assertGreater(changes["overspeed_headroom_change"], 0.0)
        self.assertLess(changes["absolute_energy_proxy_error_change"], 0.0)
        self.assertIn("recoverability_radius_component_change", changes)
        self.assertIn("recoverability_radial_component_change", changes)
        self.assertIn("recoverability_tangential_component_change", changes)
        self.assertEqual(labels["absolute_radius_gap_change"], "component_improved")

    def test_windows_have_no_score_stall_classification_or_preferred_length(self) -> None:
        payload = json.loads(analysis_payloads()[guard.NO_PROGRESS_FILENAME])
        self.assertIsNone(payload["combined_progress_score"])
        self.assertIsNone(payload["stall_progress_regression_policy_classification"])
        self.assertIsNone(payload["preferred_window_length"])
        self.assertIsNone(payload["minimum_improvement_threshold"])


class ParameterAndPhaseTests(unittest.TestCase):
    def test_all_required_no_progress_parameters_remain_unresolved(self) -> None:
        items = guard.unresolved_parameter_inventory()
        self.assertEqual(tuple(item["parameter_id"] for item in items), guard.UNRESOLVED_PARAMETER_IDS)
        self.assertTrue(all(item["current_status"] == "unresolved" for item in items))
        self.assertTrue(all(item["selected_value"] is None for item in items))
        self.assertTrue(all(item["authorization_status"] == "not_authorized" for item in items))
        self.assertTrue(all(item["next_experiment_needed"] for item in items))

    def test_phase_matrix_contains_all_nine_phases_and_evidence_sections(self) -> None:
        document = json.loads(analysis_payloads()[guard.PHASE_MATRIX_FILENAME])
        self.assertEqual([item["phase_id"] for item in document["phases"]], list(guard.PHASE_IDS))
        for item in document["phases"]:
            self.assertIn("possible_entry_evidence", item)
            self.assertIn("possible_stay_evidence", item)
            self.assertIn("possible_exit_evidence", item)
            self.assertEqual(item["policy_authorization"], "not_authorized")

    def test_phase_observability_preserves_future_evaluator_and_unsupported_dependencies(self) -> None:
        document = json.loads(analysis_payloads()[guard.PHASE_MATRIX_FILENAME])
        phases = {item["phase_id"]: item for item in document["phases"]}
        self.assertEqual(phases["hazard_arrest"]["current_observability_status"], "partially_observable")
        self.assertEqual(phases["nominal_handoff"]["current_observability_status"], "future_evaluator_required")
        self.assertIn("handoff_readiness", phases["nominal_handoff"]["future_evaluator_dependencies"])
        self.assertIn("available_correction_authority", phases["retreat"]["unsupported_dependencies"])

    def test_recoverability_verification_uses_existing_component_atoms(self) -> None:
        document = json.loads(analysis_payloads()[guard.PHASE_MATRIX_FILENAME])
        phase = next(item for item in document["phases"] if item["phase_id"] == "recoverability_verification")
        self.assertIn("phase34_compatible_recoverability_pass", phase["available_guard_atoms"])

    def test_no_phase_action_or_authorized_guard_exists(self) -> None:
        document = json.loads(analysis_payloads()[guard.PHASE_MATRIX_FILENAME])
        self.assertEqual(document["totals"]["phases_with_implemented_action_laws"], 0)
        self.assertEqual(document["totals"]["phases_with_authorized_executable_guards"], 0)
        self.assertTrue(all(not item["action_law_complete"] for item in document["phases"]))

    def test_no_default_phase_or_branch_action_relabeling(self) -> None:
        document = json.loads(analysis_payloads()[guard.PHASE_MATRIX_FILENAME])
        serialized = json.dumps(document, sort_keys=True)
        self.assertNotIn("velocity_opposed_thrust_v0", serialized)
        self.assertNotIn('"default_phase"', serialized)


class CanonicalArtifactTests(unittest.TestCase):
    def test_payloads_are_repeatedly_byte_identical_and_validate(self) -> None:
        first = guard.build_analysis_payloads(ROOT, implementation_commit="a" * 40)
        second = guard.build_analysis_payloads(ROOT, implementation_commit="a" * 40)
        self.assertEqual(first, second)
        manifest_hash, trace_hash = guard.validate_analysis_payloads(first)
        self.assertEqual(len(manifest_hash), 64)
        self.assertEqual(len(trace_hash), 64)

    def test_manifest_hash_recomputes_and_mutation_invalidates_it(self) -> None:
        manifest = json.loads(analysis_payloads()[guard.ANALYSIS_MANIFEST_FILENAME])
        self.assertTrue(guard.document_hash_recomputes(manifest))
        manifest["source_seed"] = 999
        self.assertFalse(guard.document_hash_recomputes(manifest))

    def test_volatile_timestamp_is_excluded_from_canonical_hash_helper(self) -> None:
        event = copy.deepcopy(raw_source_events()[0])
        original_hash = event["canonical_event_sha256"]
        event["volatile_timestamp"] = "different"
        self.assertTrue(guard.source_event_hash_recomputes(event))
        self.assertEqual(event["canonical_event_sha256"], original_hash)

    def test_guard_event_or_order_mutation_invalidates_trace_validation(self) -> None:
        payloads = dict(analysis_payloads())
        lines = payloads[guard.GUARD_TRACE_FILENAME].decode("utf-8").splitlines()
        event = json.loads(lines[1])
        event["recovery_step"] = 999
        lines[1] = json.dumps(event, sort_keys=True, separators=(",", ":"))
        payloads[guard.GUARD_TRACE_FILENAME] = ("\n".join(lines) + "\n").encode("utf-8")
        with self.assertRaises(guard.GuardEvidenceError):
            guard.validate_analysis_payloads(payloads)

    def test_manifest_classification_and_authorization_are_scoped(self) -> None:
        manifest = json.loads(analysis_payloads()[guard.ANALYSIS_MANIFEST_FILENAME])
        self.assertEqual(manifest["analysis_classification"], "offline_guard_evidence_analysis")
        self.assertFalse(manifest["new_runtime_execution"])
        self.assertFalse(manifest["new_measured_trace"])
        self.assertEqual(manifest["phase_guard_policy"], "not_frozen")
        self.assertEqual(manifest["phase_actions"], "not_implemented")
        self.assertEqual(manifest["staged_recovery_execution"], "not_authorized")

    def test_source_trace_identity_is_preserved(self) -> None:
        manifest = json.loads(analysis_payloads()[guard.ANALYSIS_MANIFEST_FILENAME])
        self.assertEqual(manifest["stage0c_result_commit"], guard.SOURCE_STAGE0C_RESULT_COMMIT)
        self.assertEqual(manifest["source_trace_manifest_canonical_hash"], guard.SOURCE_TRACE_MANIFEST_CANONICAL_HASH)
        self.assertEqual(manifest["source_trace_aggregate_hash"], guard.SOURCE_TRACE_AGGREGATE_HASH)


class AtomicPublicationTests(unittest.TestCase):
    def test_complete_analysis_bundle_publishes_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "analysis").mkdir()
            result = guard.publish_analysis_payloads(root, analysis_payloads())
            self.assertTrue(result.published)
            self.assertEqual(len(result.artifact_paths), 8)
            self.assertEqual(
                sorted(path.name for path in (root / guard.OUTPUT_RELATIVE_PATH).iterdir()),
                sorted(guard.PUBLISHED_FILENAMES),
            )

    def test_existing_target_is_rejected_and_never_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / guard.OUTPUT_RELATIVE_PATH
            target.mkdir(parents=True)
            sentinel = target / "user.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(guard.GuardEvidenceError):
                guard.publish_analysis_payloads(root, analysis_payloads())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_writer_failure_publishes_nothing_and_cleans_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "analysis").mkdir()

            def fail_writer(path: Path, payload: bytes) -> None:
                raise OSError("injected writer failure")

            with self.assertRaises(OSError):
                guard.publish_analysis_payloads(root, analysis_payloads(), writer=fail_writer)
            self.assertFalse((root / guard.OUTPUT_RELATIVE_PATH).exists())
            self.assertEqual(list((root / "analysis").glob(".*.staging-*")), [])

    def test_validation_failure_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "analysis").mkdir()

            def fail_validation(payloads):
                raise guard.GuardEvidenceError("injected validation failure")

            with self.assertRaises(guard.GuardEvidenceError):
                guard.publish_analysis_payloads(root, analysis_payloads(), validator=fail_validation)
            self.assertFalse((root / guard.OUTPUT_RELATIVE_PATH).exists())

    def test_protected_source_directory_is_rejected_as_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "analysis").mkdir()
            target = root / guard.SOURCE_STAGE0C_DIRECTORY
            with self.assertRaises(guard.GuardEvidenceError):
                guard.publish_analysis_payloads(root, analysis_payloads(), target_directory=target)


class CliAndRepositorySafetyTests(unittest.TestCase):
    def test_default_analyzer_and_checker_write_nothing(self) -> None:
        before = {path: path.read_bytes() for path in SOURCE_DIRECTORY.iterdir() if path.is_file()}
        analyzer = subprocess.run([sys.executable, str(ANALYZER)], cwd=ROOT, capture_output=True, text=True)
        checker = subprocess.run([sys.executable, str(CHECKER)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(analyzer.returncode, 2)
        self.assertEqual(checker.returncode, 2)
        self.assertIn("usage:", analyzer.stdout.lower())
        self.assertIn("usage:", checker.stdout.lower())
        after = {path: path.read_bytes() for path in SOURCE_DIRECTORY.iterdir() if path.is_file()}
        self.assertEqual(before, after)

    def test_plan_and_checker_static_modes_are_read_only(self) -> None:
        before = {path: path.read_bytes() for path in SOURCE_DIRECTORY.iterdir() if path.is_file()}
        plan = subprocess.run([sys.executable, str(ANALYZER), "--plan"], cwd=ROOT, capture_output=True, text=True)
        static = subprocess.run([sys.executable, str(CHECKER), "--validate-static"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(plan.returncode, 0, plan.stderr)
        self.assertEqual(static.returncode, 0, static.stderr)
        self.assertIn("\"new_runtime_execution\": false", plan.stdout)
        after = {path: path.read_bytes() for path in SOURCE_DIRECTORY.iterdir() if path.is_file()}
        self.assertEqual(before, after)

    def test_validate_only_can_be_exercised_without_execution(self) -> None:
        module = load_script(ANALYZER, "stage1a_analyzer_test")
        report = guard.GuardEvidenceValidationReport(
            valid=True,
            errors=(),
            source_event_count=10,
            source_transition_count=8,
            source_trace_manifest_hash=guard.SOURCE_TRACE_MANIFEST_CANONICAL_HASH,
            source_trace_aggregate_hash=guard.SOURCE_TRACE_AGGREGATE_HASH,
        )
        with mock.patch.object(module, "validate_static_contract", return_value=report), mock.patch.object(
            module, "build_analysis_payloads"
        ) as build:
            self.assertEqual(module.main(["--validate-only"]), 0)
            build.assert_not_called()

    def test_frozen_analysis_mode_calls_only_offline_build_and_publication(self) -> None:
        module = load_script(ANALYZER, "stage1a_execute_test")
        report = guard.GuardEvidenceValidationReport(
            valid=True,
            errors=(),
            source_event_count=10,
            source_transition_count=8,
            source_trace_manifest_hash=guard.SOURCE_TRACE_MANIFEST_CANONICAL_HASH,
            source_trace_aggregate_hash=guard.SOURCE_TRACE_AGGREGATE_HASH,
        )
        publication = guard.AnalysisPublicationResult(
            published=True,
            target_directory="synthetic",
            artifact_paths=tuple(str(index) for index in range(8)),
            artifact_hashes=(),
            analysis_manifest_hash="a" * 64,
            guard_trace_aggregate_hash="b" * 64,
        )
        with mock.patch.object(module, "require_clean_committed_repository", return_value="c" * 40), mock.patch.object(
            module, "validate_static_contract", return_value=report
        ), mock.patch.object(module, "build_analysis_payloads", return_value={"offline": b"only"}) as build, mock.patch.object(
            module, "publish_analysis_payloads", return_value=publication
        ) as publish:
            self.assertEqual(module.main(["--execute-frozen-analysis"]), 0)
            build.assert_called_once()
            publish.assert_called_once()

    def test_no_override_or_execution_options_exist(self) -> None:
        analyzer_text = ANALYZER.read_text(encoding="utf-8")
        checker_text = CHECKER.read_text(encoding="utf-8")
        for forbidden in ("--trace", "--branch", "--seed", "--window", "--threshold", "--phase", "--output-override", "--retry", "--simulate", "--run-controller"):
            self.assertNotIn(f'add_argument("{forbidden}"', analyzer_text)
        self.assertNotIn("--execute", checker_text)
        self.assertNotIn("--tune", checker_text)

    def test_analyzer_import_boundary_contains_no_runtime_execution_dependency(self) -> None:
        source = (ROOT / "runtime_assurance/staged_recovery_guard_evidence.py").read_text(encoding="utf-8")
        for forbidden in (
            "recovery_branch_runner",
            "recovery_branch_executor",
            "staged_recovery_logger_adapter",
            "run_bounded_recovery_validation_path",
            "transition_fn",
            "controller",
        ):
            self.assertNotIn(f"import {forbidden}", source)
            self.assertNotIn(f"from runtime_assurance.{forbidden}", source)

    def test_protected_stage0c_files_remain_byte_identical(self) -> None:
        before = {
            path.name: path.read_bytes()
            for path in SOURCE_DIRECTORY.iterdir()
            if path.is_file()
        }
        guard.build_analysis_payloads(ROOT, implementation_commit="a" * 40)
        after = {
            path.name: path.read_bytes()
            for path in SOURCE_DIRECTORY.iterdir()
            if path.is_file()
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

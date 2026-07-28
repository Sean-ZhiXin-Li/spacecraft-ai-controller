from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

from runtime_assurance.recovery_evaluators import (
    EVALUATION_CLEAR,
    EVALUATION_TRIGGERED,
    PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
    PHASE34_RECOVERABLE_VR_RATIO_MAX,
    PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
    evaluate_phase34_compatible_recoverability,
)
from runtime_assurance.staged_recovery_contract import SIGNAL_IDS
from runtime_assurance.staged_recovery_instrumentation import (
    ARCHITECTURE_SIGNAL_IDS,
    CANONICAL_FIELD_ORDER,
    COMPLETED_DATE,
    COVERAGE_DIRECT_INPUT,
    COVERAGE_FUTURE_EVALUATOR,
    COVERAGE_NOT_SUPPORTED,
    COVERAGE_PREDICTED_STATE,
    COVERAGE_PREVIOUS_STATE,
    COVERAGE_PURE_DERIVATION,
    COVERAGE_RUNTIME_INTEGRATION,
    CURRENT_GRAVITY_MODEL_ID,
    EXECUTION_NOT_AUTHORIZED_REASON,
    INSTRUMENTATION_SCHEMA_VERSION,
    POSITION_NORM_ZERO_TOLERANCE,
    SOURCE_ARCHITECTURE_ARTIFACT_HASHES,
    SOURCE_ARCHITECTURE_CANONICAL_HASH,
    STAGED_EXECUTION_NOT_AUTHORIZED,
    ActionGeometry,
    CartesianState2D,
    InstrumentationContractError,
    InstrumentationEvidenceStatus,
    InstrumentedValue,
    OrbitalConfiguration,
    StagedRecoveryInstrumentationRecord,
    architecture_signal_coverage,
    build_instrumentation_record,
    canonical_document_hash_is_valid,
    canonical_instrumentation_manifest_sha256,
    canonical_json_bytes,
    canonical_record_sha256,
    canonical_sha256,
    coverage_counts,
    derive_action_geometry,
    derive_crossing_event,
    derive_orbital_basis,
    derive_orbital_state,
    derive_phase34_recoverability,
    derive_predicted_hazard_state,
    derive_progress_sample,
    derivation_traceability_document,
    derived_value,
    field_catalog,
    field_catalog_document,
    instrumentation_manifest_document,
    invalid_value,
    measured_value,
    not_evaluated_value,
    predicted_value,
    validate_instrumentation_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = OrbitalConfiguration(mu=100.0, target_radius=25.0)
VCIRC = math.sqrt(CONFIG.mu / CONFIG.target_radius)
INSTRUMENTATION_DIR = ROOT / "analysis/staged_recovery_instrumentation_v0"
FROZEN_RECOVERY_DIR = ROOT / "analysis/recovery_action_branching_nonformal_v0"
DIAGNOSIS_DIR = ROOT / "analysis/recovery_branch_mechanism_diagnosis_v0"


def orbital(state: CartesianState2D, *, step: int = 1):
    return derive_orbital_state(state, CONFIG, source_step=step)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_staged_recovery_instrumentation.py"),
            *arguments,
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        env=environment,
    )


class InstrumentedValueTests(unittest.TestCase):
    def test_measured_finite_value_is_accepted(self) -> None:
        value = measured_value(2.5, units="m", source_id="sensor", source_step=3)
        self.assertEqual(value.status, InstrumentationEvidenceStatus.MEASURED)
        self.assertEqual(value.value, 2.5)
        self.assertTrue(value.valid)

    def test_derived_finite_value_is_accepted(self) -> None:
        value = derived_value(
            2.5,
            units="m",
            source_id="derivation",
            input_source_ids=("b", "a", "a"),
        )
        self.assertEqual(value.status, InstrumentationEvidenceStatus.DERIVED)
        self.assertEqual(value.input_source_ids, ("a", "b"))

    def test_missing_numeric_remains_null_not_evaluated(self) -> None:
        value = not_evaluated_value(reason="missing", units="m", source_id="test")
        self.assertIsNone(value.value)
        self.assertFalse(value.valid)
        self.assertEqual(value.status, InstrumentationEvidenceStatus.NOT_EVALUATED)

    def test_missing_boolean_remains_null_not_evaluated(self) -> None:
        value = not_evaluated_value(
            reason="missing", units="boolean", source_id="test"
        )
        self.assertIsNone(value.value)
        self.assertIsNot(value.value, False)

    def test_nonfinite_numeric_becomes_invalid(self) -> None:
        value = measured_value(math.nan, units="m", source_id="test")
        self.assertEqual(value.status, InstrumentationEvidenceStatus.INVALID)
        self.assertIsNone(value.value)

    def test_boolean_is_rejected_as_numeric_state(self) -> None:
        basis = derive_orbital_basis(CartesianState2D(True, 0.0, 0.0, 0.0))
        self.assertEqual(basis.radius.status, InstrumentationEvidenceStatus.INVALID)

    def test_invalid_status_cannot_be_valid(self) -> None:
        with self.assertRaises(InstrumentationContractError):
            InstrumentedValue(
                value=None,
                status=InstrumentationEvidenceStatus.INVALID,
                reason="bad",
                units="m",
                source_id="test",
                source_step=None,
                valid=True,
            )

    def test_not_evaluated_cannot_carry_favorable_value(self) -> None:
        with self.assertRaises(InstrumentationContractError):
            InstrumentedValue(
                value=False,
                status=InstrumentationEvidenceStatus.NOT_EVALUATED,
                reason="missing",
                units="boolean",
                source_id="test",
                source_step=None,
                valid=False,
            )

    def test_predicted_and_measured_statuses_remain_distinct(self) -> None:
        measured = measured_value(1.0, units="ratio", source_id="measured")
        predicted = predicted_value(
            1.0, units="ratio", source_id="predicted", horizon_steps=1
        )
        self.assertNotEqual(measured.status, predicted.status)

    def test_multi_step_prediction_is_labeled(self) -> None:
        value = predicted_value(
            1.0, units="ratio", source_id="predicted", horizon_steps=3
        )
        self.assertEqual(
            value.status, InstrumentationEvidenceStatus.MULTI_STEP_PREDICTED
        )

    def test_source_provenance_is_preserved(self) -> None:
        value = measured_value(
            2.0, units="m", source_id="source.path", source_step=9
        )
        self.assertEqual((value.source_id, value.source_step), ("source.path", 9))

    def test_nested_mapping_is_canonicalized_immutably(self) -> None:
        original = {"b": [2, 3], "a": 1}
        value = measured_value(original, units="structured", source_id="test")
        original["a"] = 9
        self.assertEqual(value.value, (("a", 1), ("b", (2, 3))))

    def test_instrumented_value_is_immutable(self) -> None:
        value = measured_value(1.0, units="m", source_id="test")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            value.value = 2.0  # type: ignore[misc]

    def test_negative_source_step_is_rejected(self) -> None:
        with self.assertRaises(InstrumentationContractError):
            measured_value(1.0, units="m", source_id="test", source_step=-1)

    def test_prediction_horizon_must_be_positive(self) -> None:
        with self.assertRaises(InstrumentationContractError):
            predicted_value(
                1.0, units="ratio", source_id="test", horizon_steps=0
            )


class CartesianAndBasisTests(unittest.TestCase):
    def test_valid_cartesian_state_is_accepted(self) -> None:
        basis = derive_orbital_basis(CartesianState2D(3.0, 4.0, 0.0, 0.0))
        self.assertTrue(basis.radius.valid)

    def test_missing_state_is_not_evaluated(self) -> None:
        basis = derive_orbital_basis(None)
        self.assertEqual(
            basis.radius.status, InstrumentationEvidenceStatus.NOT_EVALUATED
        )

    def test_nonfinite_state_is_invalid(self) -> None:
        basis = derive_orbital_basis(CartesianState2D(math.inf, 0.0, 0.0, 0.0))
        self.assertEqual(basis.radius.status, InstrumentationEvidenceStatus.INVALID)

    def test_zero_position_norm_is_invalid(self) -> None:
        self.assertEqual(POSITION_NORM_ZERO_TOLERANCE, 0.0)
        basis = derive_orbital_basis(CartesianState2D(0.0, 0.0, 1.0, 1.0))
        self.assertEqual(basis.radius.status, InstrumentationEvidenceStatus.INVALID)

    def test_radius_derivation_is_exact(self) -> None:
        basis = derive_orbital_basis(CartesianState2D(3.0, 4.0, 0.0, 0.0))
        self.assertEqual(basis.radius.value, 5.0)

    def test_speed_magnitude_is_exact(self) -> None:
        basis = derive_orbital_basis(CartesianState2D(3.0, 4.0, 5.0, 12.0))
        self.assertEqual(basis.speed_magnitude.value, 13.0)

    def test_radial_unit_vector_is_normalized(self) -> None:
        vector = derive_orbital_basis(
            CartesianState2D(3.0, 4.0, 0.0, 0.0)
        ).radial_unit_vector.value
        self.assertAlmostEqual(math.hypot(*vector), 1.0)

    def test_tangential_unit_vector_is_normalized(self) -> None:
        vector = derive_orbital_basis(
            CartesianState2D(3.0, 4.0, 0.0, 0.0)
        ).tangential_unit_vector.value
        self.assertAlmostEqual(math.hypot(*vector), 1.0)

    def test_basis_vectors_are_orthogonal(self) -> None:
        basis = derive_orbital_basis(CartesianState2D(3.0, 4.0, 0.0, 0.0))
        er = basis.radial_unit_vector.value
        et = basis.tangential_unit_vector.value
        self.assertAlmostEqual(er[0] * et[0] + er[1] * et[1], 0.0)

    def test_tangential_orientation_is_counterclockwise(self) -> None:
        basis = derive_orbital_basis(CartesianState2D(5.0, 0.0, 0.0, 0.0))
        self.assertEqual(basis.tangential_unit_vector.value, (0.0, 1.0))

    def test_all_position_quadrants_preserve_orientation(self) -> None:
        for x, y in ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)):
            with self.subTest(x=x, y=y):
                basis = derive_orbital_basis(CartesianState2D(x, y, 0.0, 0.0))
                er = basis.radial_unit_vector.value
                et = basis.tangential_unit_vector.value
                self.assertAlmostEqual(et[0], -er[1])
                self.assertAlmostEqual(et[1], er[0])


class VelocityAndTargetDerivationTests(unittest.TestCase):
    def test_purely_radial_outward_velocity(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, 3.0, 0.0))
        self.assertEqual(value.field("radial_velocity").value, 3.0)
        self.assertEqual(value.field("tangential_velocity").value, 0.0)

    def test_purely_radial_inward_velocity(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, -3.0, 0.0))
        self.assertEqual(value.field("radial_velocity").value, -3.0)

    def test_purely_positive_tangential_velocity(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, 0.0, 3.0))
        self.assertEqual(value.field("tangential_velocity").value, 3.0)

    def test_purely_negative_tangential_velocity(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, 0.0, -3.0))
        self.assertEqual(value.field("tangential_velocity").value, -3.0)

    def test_mixed_velocity_decomposition(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, 3.0, 4.0))
        self.assertEqual(value.field("radial_velocity").value, 3.0)
        self.assertEqual(value.field("tangential_velocity").value, 4.0)

    def test_recomposed_velocity_matches_original(self) -> None:
        state = CartesianState2D(3.0, 4.0, -2.0, 7.0)
        basis = derive_orbital_basis(state)
        value = orbital(state)
        vr = value.field("radial_velocity").value
        vt = value.field("tangential_velocity").value
        er = basis.radial_unit_vector.value
        et = basis.tangential_unit_vector.value
        self.assertAlmostEqual(vr * er[0] + vt * et[0], state.vx)
        self.assertAlmostEqual(vr * er[1] + vt * et[1], state.vy)

    def test_invalid_basis_propagates_invalid(self) -> None:
        value = orbital(CartesianState2D(0.0, 0.0, 1.0, 1.0))
        self.assertEqual(
            value.field("radial_velocity").status,
            InstrumentationEvidenceStatus.INVALID,
        )

    def test_signed_radius_error_is_preserved(self) -> None:
        value = orbital(CartesianState2D(24.0, 0.0, 0.0, VCIRC))
        self.assertEqual(value.field("signed_target_radius_error").value, -1.0)

    def test_absolute_radius_error_is_separate(self) -> None:
        value = orbital(CartesianState2D(24.0, 0.0, 0.0, VCIRC))
        self.assertEqual(value.field("absolute_target_radius_error").value, 1.0)
        self.assertNotEqual(
            value.field("absolute_target_radius_error").value,
            value.field("signed_target_radius_error").value,
        )

    def test_radius_error_ratio_matches_repository_semantics(self) -> None:
        value = orbital(CartesianState2D(27.0, 0.0, 0.0, VCIRC))
        self.assertAlmostEqual(value.field("radius_error_ratio").value, 2.0 / 25.0)

    def test_radial_velocity_ratio_matches_repository_semantics(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, 0.4, VCIRC))
        self.assertAlmostEqual(
            value.field("radial_velocity_ratio").value, 0.4 / (VCIRC + 1.0e-12)
        )

    def test_tangential_error_ratio_matches_repository_semantics(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, 0.0, 2.5))
        self.assertAlmostEqual(
            value.field("tangential_velocity_error_ratio").value,
            0.5 / (VCIRC + 1.0e-12),
        )

    def test_target_circular_speed_matches_repository_formula(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, 0.0, 0.0))
        self.assertEqual(value.field("target_circular_speed").value, 2.0)

    def test_invalid_target_denominator_is_explicit(self) -> None:
        config = OrbitalConfiguration(mu=100.0, target_radius=0.0)
        value = derive_orbital_state(
            CartesianState2D(25.0, 0.0, 0.0, 2.0), config
        )
        self.assertEqual(
            value.field("radius_error_ratio").status,
            InstrumentationEvidenceStatus.INVALID,
        )

    def test_missing_configuration_uses_no_guessed_defaults(self) -> None:
        value = derive_orbital_state(
            CartesianState2D(25.0, 0.0, 0.0, 2.0), None
        )
        self.assertEqual(
            value.field("target_circular_speed").status,
            InstrumentationEvidenceStatus.NOT_EVALUATED,
        )


class HazardAndEnergyTests(unittest.TestCase):
    def test_realized_speed_ratio_matches_existing_semantics(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, 0.0, 3.0))
        self.assertAlmostEqual(
            value.field("realized_speed_ratio").value, 3.0 / (VCIRC + 1.0e-12)
        )

    def test_predicted_ratio_remains_separate(self) -> None:
        record = build_instrumentation_record(
            state=CartesianState2D(25.0, 0.0, 0.0, 2.0),
            predicted_state=CartesianState2D(25.0, 0.0, 0.0, 4.0),
            configuration=CONFIG,
        )
        self.assertEqual(
            record.field("realized_speed_ratio").status,
            InstrumentationEvidenceStatus.DERIVED,
        )
        self.assertEqual(
            record.field("predicted_speed_ratio").status,
            InstrumentationEvidenceStatus.ONE_STEP_PREDICTED,
        )
        self.assertNotEqual(
            record.field("realized_speed_ratio").value,
            record.field("predicted_speed_ratio").value,
        )

    def test_headroom_is_positive_below_threshold(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, 0.0, 2.0))
        self.assertGreater(value.field("overspeed_headroom").value, 0.0)

    def test_headroom_is_zero_at_exact_threshold(self) -> None:
        speed = 1.90 * (VCIRC + 1.0e-12)
        value = orbital(CartesianState2D(25.0, 0.0, 0.0, speed))
        self.assertAlmostEqual(value.field("overspeed_headroom").value, 0.0)

    def test_exact_threshold_is_not_overspeed(self) -> None:
        speed = 1.90 * (VCIRC + 1.0e-12)
        value = orbital(CartesianState2D(25.0, 0.0, 0.0, speed))
        self.assertFalse(value.field("overspeed_status").value)

    def test_ratio_above_threshold_is_overspeed(self) -> None:
        speed = 1.900001 * (VCIRC + 1.0e-12)
        value = orbital(CartesianState2D(25.0, 0.0, 0.0, speed))
        self.assertTrue(value.field("overspeed_status").value)
        self.assertLess(value.field("overspeed_headroom").value, 0.0)

    def test_predicted_overspeed_does_not_become_realized(self) -> None:
        predicted = dict(
            derive_predicted_hazard_state(
                CartesianState2D(25.0, 0.0, 0.0, 4.0), CONFIG
            )
        )
        realized = orbital(CartesianState2D(25.0, 0.0, 0.0, 2.0))
        self.assertTrue(predicted["predicted_overspeed_status"].value)
        self.assertFalse(realized.field("overspeed_status").value)

    def test_specific_energy_proxy_matches_declared_formula(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, 0.0, 2.0))
        expected = 0.5 * 2.0**2 - 100.0 / (25.0 + 1.0e-12)
        self.assertEqual(value.field("specific_orbital_energy").value, expected)

    def test_target_circular_energy_matches_declared_formula(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, 0.0, 2.0))
        self.assertEqual(value.field("target_circular_specific_energy").value, -2.0)

    def test_invalid_radius_rejects_energy(self) -> None:
        value = orbital(CartesianState2D(0.0, 0.0, 0.0, 2.0))
        self.assertEqual(
            value.field("specific_orbital_energy").status,
            InstrumentationEvidenceStatus.INVALID,
        )

    def test_invalid_gravitational_parameter_rejects_energy(self) -> None:
        config = OrbitalConfiguration(mu=math.nan, target_radius=25.0)
        value = derive_orbital_state(
            CartesianState2D(25.0, 0.0, 0.0, 2.0), config
        )
        self.assertEqual(
            value.field("specific_orbital_energy").status,
            InstrumentationEvidenceStatus.INVALID,
        )

    def test_energy_units_and_proxy_label_are_preserved(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, 0.0, 2.0))
        self.assertEqual(value.field("specific_orbital_energy").units, "J/kg")
        self.assertEqual(value.field("energy_model_status").value, "declared_diagnostic_proxy")

    def test_specific_energy_is_not_mass_scaled(self) -> None:
        value = orbital(CartesianState2D(25.0, 0.0, 0.0, 2.0))
        self.assertAlmostEqual(value.field("specific_orbital_energy").value, -2.0)
        self.assertNotIn("mass", value.field("specific_orbital_energy").input_source_ids)

    def test_unsupported_gravity_model_is_not_evaluated(self) -> None:
        config = OrbitalConfiguration(
            mu=100.0, target_radius=25.0, gravity_model_id="unsupported_model"
        )
        value = derive_orbital_state(
            CartesianState2D(25.0, 0.0, 0.0, 2.0), config
        )
        self.assertEqual(
            value.field("specific_orbital_energy").status,
            InstrumentationEvidenceStatus.NOT_EVALUATED,
        )


class RecoverabilityTests(unittest.TestCase):
    def test_exact_recoverability_boundaries_pass(self) -> None:
        result = derive_phase34_recoverability(
            PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
            PHASE34_RECOVERABLE_VR_RATIO_MAX,
            PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
        )
        self.assertTrue(result.field("phase34_compatible_recoverability").value)

    def test_just_below_recoverability_boundaries_pass(self) -> None:
        result = derive_phase34_recoverability(
            math.nextafter(PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX, 0.0),
            math.nextafter(PHASE34_RECOVERABLE_VR_RATIO_MAX, 0.0),
            math.nextafter(PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX, 0.0),
        )
        self.assertTrue(result.field("phase34_compatible_recoverability").value)

    def test_just_above_radius_threshold_fails(self) -> None:
        result = derive_phase34_recoverability(
            math.nextafter(PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX, math.inf), 0.0, 0.0
        )
        self.assertFalse(result.field("radius_component_pass").value)
        self.assertFalse(result.field("phase34_compatible_recoverability").value)

    def test_just_above_radial_threshold_fails(self) -> None:
        result = derive_phase34_recoverability(
            0.0,
            math.nextafter(PHASE34_RECOVERABLE_VR_RATIO_MAX, math.inf),
            0.0,
        )
        self.assertFalse(result.field("radial_velocity_component_pass").value)

    def test_just_above_tangential_threshold_fails(self) -> None:
        result = derive_phase34_recoverability(
            0.0,
            0.0,
            math.nextafter(PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX, math.inf),
        )
        self.assertFalse(result.field("tangential_velocity_component_pass").value)

    def test_missing_component_yields_not_evaluated(self) -> None:
        result = derive_phase34_recoverability(0.0, None, 0.0)
        self.assertEqual(
            result.field("phase34_compatible_recoverability").status,
            InstrumentationEvidenceStatus.NOT_EVALUATED,
        )

    def test_invalid_component_yields_invalid(self) -> None:
        result = derive_phase34_recoverability(0.0, math.inf, 0.0)
        self.assertEqual(
            result.field("phase34_compatible_recoverability").status,
            InstrumentationEvidenceStatus.INVALID,
        )

    def test_one_component_pass_does_not_imply_recoverability(self) -> None:
        result = derive_phase34_recoverability(0.0, 1.0, 1.0)
        self.assertTrue(result.field("radius_component_pass").value)
        self.assertFalse(result.field("phase34_compatible_recoverability").value)

    def test_wrapper_matches_existing_evaluator_positive(self) -> None:
        existing = evaluate_phase34_compatible_recoverability(
            r_error_ratio=0.001,
            vr_ratio=0.01,
            vt_error_ratio=0.2,
        )
        wrapped = derive_phase34_recoverability(0.001, 0.01, 0.2)
        self.assertEqual(existing.status, EVALUATION_TRIGGERED)
        self.assertTrue(wrapped.field("phase34_compatible_recoverability").value)

    def test_wrapper_matches_existing_evaluator_negative(self) -> None:
        existing = evaluate_phase34_compatible_recoverability(
            r_error_ratio=0.01,
            vr_ratio=0.01,
            vt_error_ratio=0.2,
        )
        wrapped = derive_phase34_recoverability(0.01, 0.01, 0.2)
        self.assertEqual(existing.status, EVALUATION_CLEAR)
        self.assertFalse(wrapped.field("phase34_compatible_recoverability").value)


class CrossingTests(unittest.TestCase):
    def crossing(
        self,
        previous_radius: float | None,
        current_radius: float | None,
        *,
        current_step: int = 5,
        branch_step: int = 5,
    ):
        previous = (
            None
            if previous_radius is None
            else CartesianState2D(previous_radius, 0.0, 0.0, VCIRC)
        )
        current = (
            None
            if current_radius is None
            else CartesianState2D(current_radius, 0.0, 0.0, VCIRC)
        )
        return derive_crossing_event(
            previous,
            current,
            CONFIG,
            previous_step=current_step - 1,
            current_step=current_step,
            branch_step=branch_step,
        )

    def test_below_to_above_crossing_is_detected(self) -> None:
        result = self.crossing(24.0, 26.0)
        self.assertTrue(result.field("target_radius_crossing").value)
        self.assertEqual(result.field("crossing_direction").value, "below_to_above")

    def test_above_to_below_crossing_is_detected(self) -> None:
        result = self.crossing(26.0, 24.0)
        self.assertTrue(result.field("target_radius_crossing").value)
        self.assertEqual(result.field("crossing_direction").value, "above_to_below")

    def test_no_crossing_is_detected(self) -> None:
        result = self.crossing(24.0, 23.0)
        self.assertFalse(result.field("target_radius_crossing").value)

    def test_previous_exact_target_does_not_create_new_crossing(self) -> None:
        result = self.crossing(25.0, 26.0)
        self.assertFalse(result.field("target_radius_crossing").value)

    def test_current_exact_target_completes_crossing(self) -> None:
        result = self.crossing(24.0, 25.0)
        self.assertTrue(result.field("target_radius_crossing").value)

    def test_crossing_before_branch_is_not_recovery_eligible(self) -> None:
        result = self.crossing(24.0, 26.0, current_step=4, branch_step=5)
        self.assertTrue(result.field("target_radius_crossing").value)
        self.assertFalse(result.field("crossing_recovery_eligible").value)

    def test_crossing_at_branch_step_is_recovery_eligible(self) -> None:
        result = self.crossing(24.0, 26.0, current_step=5, branch_step=5)
        self.assertTrue(result.field("crossing_recovery_eligible").value)

    def test_missing_previous_state_is_not_evaluated(self) -> None:
        result = self.crossing(None, 26.0)
        self.assertEqual(
            result.field("target_radius_crossing").status,
            InstrumentationEvidenceStatus.NOT_EVALUATED,
        )

    def test_malformed_crossing_step_is_invalid(self) -> None:
        result = derive_crossing_event(
            CartesianState2D(24.0, 0.0, 0.0, VCIRC),
            CartesianState2D(26.0, 0.0, 0.0, VCIRC),
            CONFIG,
            previous_step=0,
            current_step=-1,
            branch_step=0,
        )
        self.assertEqual(
            result.field("target_radius_crossing").status,
            InstrumentationEvidenceStatus.INVALID,
        )

    def test_no_crossing_interpolation_occurs(self) -> None:
        result = self.crossing(24.0, 26.0)
        self.assertEqual(
            result.field("crossing_interpolation_fraction").status,
            InstrumentationEvidenceStatus.NOT_EVALUATED,
        )


class ProgressTests(unittest.TestCase):
    def test_raw_signed_deltas_are_computed(self) -> None:
        previous = orbital(CartesianState2D(24.0, 0.0, 0.0, 2.0), step=1)
        current = orbital(CartesianState2D(24.5, 0.0, 0.2, 2.1), step=2)
        result = derive_progress_sample(previous, current, source_step=2)
        self.assertAlmostEqual(result.field("delta_signed_target_radius_error").value, 0.5)
        self.assertAlmostEqual(result.field("delta_radial_velocity").value, 0.2)

    def test_transition_count_delta_is_computed(self) -> None:
        result = derive_progress_sample(
            None,
            None,
            previous_transition_count=10,
            current_transition_count=12,
        )
        self.assertEqual(result.field("transition_count_delta").value, 2)

    def test_elapsed_time_delta_is_computed(self) -> None:
        result = derive_progress_sample(
            None, None, previous_time=1.0, current_time=2.5
        )
        self.assertEqual(result.field("elapsed_time_delta").value, 1.5)

    def test_no_threshold_based_classification_occurs(self) -> None:
        result = derive_progress_sample(None, None)
        self.assertEqual(
            result.field("progress_classification").status,
            InstrumentationEvidenceStatus.NOT_EVALUATED,
        )

    def test_missing_previous_sample_is_not_evaluated(self) -> None:
        current = orbital(CartesianState2D(24.5, 0.0, 0.0, 2.0))
        result = derive_progress_sample(None, current)
        self.assertEqual(
            result.field("delta_radial_velocity").status,
            InstrumentationEvidenceStatus.NOT_EVALUATED,
        )

    def test_invalid_sample_propagates_invalid(self) -> None:
        previous = orbital(CartesianState2D(24.0, 0.0, 0.0, 2.0))
        current = orbital(CartesianState2D(math.nan, 0.0, 0.0, 2.0))
        result = derive_progress_sample(previous, current)
        self.assertEqual(
            result.field("delta_radial_velocity").status,
            InstrumentationEvidenceStatus.INVALID,
        )

    def test_timeout_alone_does_not_become_stalled(self) -> None:
        result = derive_progress_sample(
            None,
            None,
            previous_transition_count=0,
            current_transition_count=10000,
        )
        self.assertEqual(
            result.field("progress_classification").status,
            InstrumentationEvidenceStatus.NOT_EVALUATED,
        )

    def test_progress_components_are_not_combined(self) -> None:
        result = derive_progress_sample(None, None)
        keys = {key for key, _ in result.values}
        self.assertNotIn("combined_progress_score", keys)
        self.assertNotIn("recovery_score", keys)

    def test_invalid_time_delta_is_explicit(self) -> None:
        result = derive_progress_sample(
            None, None, previous_time=True, current_time=1.0
        )
        self.assertEqual(
            result.field("elapsed_time_delta").status,
            InstrumentationEvidenceStatus.INVALID,
        )


class ActionGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.basis = derive_orbital_basis(CartesianState2D(25.0, 0.0, 0.0, 2.0))

    def geometry(self) -> ActionGeometry:
        return derive_action_geometry((0.3, 0.4), (0.3, 0.4), self.basis)

    def test_action_magnitude_is_computed(self) -> None:
        self.assertAlmostEqual(self.geometry().field("proposed_action_magnitude").value, 0.5)

    def test_radial_action_component_is_computed(self) -> None:
        self.assertEqual(self.geometry().field("proposed_action_radial_component").value, 0.3)

    def test_tangential_action_component_is_computed(self) -> None:
        self.assertEqual(
            self.geometry().field("proposed_action_tangential_component").value, 0.4
        )

    def test_action_saturation_margin_uses_component_limit(self) -> None:
        self.assertEqual(self.geometry().field("action_saturation_margin").value, 0.6)

    def test_proposed_and_executed_equality_is_recorded(self) -> None:
        self.assertTrue(self.geometry().field("proposed_equals_executed").value)

    def test_explicit_abort_has_no_action(self) -> None:
        result = derive_action_geometry(
            None, None, self.basis, explicit_abort=True
        )
        self.assertIsNone(result.field("proposed_action").value)
        self.assertIsNone(result.field("executed_action").value)
        self.assertEqual(
            result.field("proposed_action").status,
            InstrumentationEvidenceStatus.NOT_EVALUATED,
        )

    def test_rejected_action_is_distinct_from_zero_action(self) -> None:
        result = derive_action_geometry(
            (0.0, 0.0), None, self.basis, action_rejected=True
        )
        self.assertEqual(result.field("proposed_action").value, (0.0, 0.0))
        self.assertIsNone(result.field("executed_action").value)
        self.assertEqual(result.field("action_geometry_status").value, "rejected")

    def test_missing_basis_blocks_component_decomposition(self) -> None:
        result = derive_action_geometry((0.3, 0.4), (0.3, 0.4), None)
        self.assertEqual(
            result.field("proposed_action_radial_component").status,
            InstrumentationEvidenceStatus.NOT_EVALUATED,
        )

    def test_normalized_action_is_not_labeled_delta_v(self) -> None:
        result = self.geometry()
        self.assertEqual(result.field("proposed_action_magnitude").units, "normalized_action")
        self.assertNotEqual(result.field("proposed_action_magnitude").units, "m/s")

    def test_malformed_action_is_invalid(self) -> None:
        result = derive_action_geometry((math.nan, 0.0), (0.0, 0.0), self.basis)
        self.assertEqual(
            result.field("proposed_action").status,
            InstrumentationEvidenceStatus.INVALID,
        )


class RecordAndManifestTests(unittest.TestCase):
    def record(self, *, timestamp: str | None = None):
        return build_instrumentation_record(
            state=CartesianState2D(25.0, 0.0, 0.0, 2.0),
            configuration=CONFIG,
            case_id="synthetic_case",
            seed=0,
            implementation_commit="a" * 40,
            branch_state_hash="b" * 64,
            simulator_configuration_hash="c" * 64,
            constants_hash="d" * 64,
            recovery_step=1,
            total_transition_count=2,
            simulation_time=0.5,
            volatile_provenance_timestamp=timestamp,
        )

    def test_record_preserves_provenance(self) -> None:
        record = self.record()
        self.assertEqual(record.field("case_id").value, "synthetic_case")
        self.assertEqual(record.field("seed").value, 0)
        self.assertEqual(record.field("recovery_step").value, 1)

    def test_phase_fields_remain_not_evaluated_when_unsupplied(self) -> None:
        record = self.record()
        self.assertEqual(
            record.field("current_phase").status,
            InstrumentationEvidenceStatus.NOT_EVALUATED,
        )

    def test_external_phase_fields_are_preserved_without_decision_logic(self) -> None:
        record = build_instrumentation_record(
            state=CartesianState2D(25.0, 0.0, 0.0, 2.0),
            configuration=CONFIG,
            external_fields={
                "current_phase": measured_value(
                    "hazard_arrest", units="categorical", source_id="external.phase"
                )
            },
        )
        self.assertEqual(record.field("current_phase").value, "hazard_arrest")

    def test_unknown_external_field_is_rejected(self) -> None:
        with self.assertRaises(InstrumentationContractError):
            build_instrumentation_record(
                state=CartesianState2D(25.0, 0.0, 0.0, 2.0),
                configuration=CONFIG,
                external_fields={
                    "unknown": measured_value(1, units="count", source_id="test")
                },
            )

    def test_record_uses_canonical_field_order(self) -> None:
        record = self.record()
        self.assertEqual(tuple(key for key, _ in record.fields), CANONICAL_FIELD_ORDER)

    def test_record_hash_recomputes(self) -> None:
        record = self.record()
        self.assertEqual(record.canonical_record_hash, canonical_record_sha256(record))

    def test_payload_mutation_invalidates_record_hash(self) -> None:
        record = self.record()
        fields = list(record.fields)
        index = next(index for index, (key, _) in enumerate(fields) if key == "case_id")
        fields[index] = (
            "case_id",
            measured_value("changed", units="categorical", source_id="test"),
        )
        mutated = dataclasses.replace(record, fields=tuple(fields))
        self.assertNotEqual(mutated.canonical_record_hash, canonical_record_sha256(mutated))

    def test_volatile_timestamp_does_not_change_scientific_hash(self) -> None:
        first = self.record(timestamp="2026-07-28T00:00:00Z")
        second = self.record(timestamp="2026-07-28T12:00:00Z")
        self.assertEqual(first.canonical_record_hash, second.canonical_record_hash)

    def test_record_is_immutable(self) -> None:
        record = self.record()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            record.schema_version = "changed"  # type: ignore[misc]

    def test_field_ids_are_unique(self) -> None:
        self.assertEqual(len(CANONICAL_FIELD_ORDER), len(set(CANONICAL_FIELD_ORDER)))

    def test_canonical_field_order_is_stable(self) -> None:
        self.assertEqual(
            tuple(definition.field_id for definition in field_catalog()),
            CANONICAL_FIELD_ORDER,
        )

    def test_every_architecture_signal_has_coverage(self) -> None:
        coverage = dict(architecture_signal_coverage())
        self.assertEqual(set(coverage), set(SIGNAL_IDS))
        self.assertEqual(tuple(ARCHITECTURE_SIGNAL_IDS), tuple(SIGNAL_IDS))

    def test_runtime_fields_are_not_claimed_as_pure(self) -> None:
        coverage = dict(architecture_signal_coverage())
        self.assertEqual(
            coverage["phase_transition_reason"], COVERAGE_RUNTIME_INTEGRATION
        )
        self.assertEqual(coverage["no_progress_status"], COVERAGE_RUNTIME_INTEGRATION)

    def test_handoff_readiness_requires_future_evaluator(self) -> None:
        self.assertEqual(
            dict(architecture_signal_coverage())["handoff_readiness"],
            COVERAGE_FUTURE_EVALUATOR,
        )

    def test_available_correction_authority_remains_unsupported(self) -> None:
        self.assertEqual(
            dict(architecture_signal_coverage())["available_correction_authority"],
            COVERAGE_NOT_SUPPORTED,
        )

    def test_coverage_totals_are_explicit(self) -> None:
        self.assertEqual(
            dict(coverage_counts()),
            {
                COVERAGE_DIRECT_INPUT: 16,
                COVERAGE_PURE_DERIVATION: 14,
                COVERAGE_PREVIOUS_STATE: 8,
                COVERAGE_PREDICTED_STATE: 3,
                COVERAGE_RUNTIME_INTEGRATION: 9,
                COVERAGE_FUTURE_EVALUATOR: 1,
                COVERAGE_NOT_SUPPORTED: 1,
            },
        )

    def test_manifest_canonical_hash_recomputes(self) -> None:
        document = instrumentation_manifest_document()
        self.assertTrue(canonical_document_hash_is_valid(document))
        self.assertEqual(
            document["canonical_payload_hash"],
            canonical_instrumentation_manifest_sha256(document),
        )

    def test_manifest_payload_mutation_invalidates_hash(self) -> None:
        document = copy.deepcopy(instrumentation_manifest_document())
        document["staged_recovery_execution"] = "authorized"
        self.assertFalse(canonical_document_hash_is_valid(document))

    def test_source_architecture_hash_matches(self) -> None:
        architecture = json.loads(
            (ROOT / "analysis/staged_recovery_architecture_v0/architecture_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            architecture["canonical_payload_hash"], SOURCE_ARCHITECTURE_CANONICAL_HASH
        )

    def test_source_architecture_artifact_hashes_match(self) -> None:
        for relative_path, expected in SOURCE_ARCHITECTURE_ARTIFACT_HASHES:
            with self.subTest(path=relative_path):
                self.assertEqual(sha256(ROOT / relative_path), expected)

    def test_execution_remains_not_authorized(self) -> None:
        manifest = instrumentation_manifest_document()
        self.assertEqual(
            manifest["staged_recovery_execution"], STAGED_EXECUTION_NOT_AUTHORIZED
        )
        self.assertEqual(
            manifest["execution_authorization_reason"],
            EXECUTION_NOT_AUTHORIZED_REASON,
        )

    def test_contract_validation_passes(self) -> None:
        report = validate_instrumentation_contract()
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.architecture_signal_count, 52)

    def test_catalog_mutation_is_rejected(self) -> None:
        catalog = copy.deepcopy(field_catalog_document())
        catalog["fields"][0]["field_id"] = "changed"
        report = validate_instrumentation_contract(catalog=catalog)
        self.assertFalse(report.valid)
        self.assertIn("field_catalog_canonical_hash_mismatch", report.errors)

    def test_traceability_is_complete_and_hashed(self) -> None:
        document = derivation_traceability_document()
        self.assertTrue(canonical_document_hash_is_valid(document))
        self.assertGreaterEqual(len(document["derivations"]), 10)

    def test_completed_date_is_exact(self) -> None:
        self.assertEqual(COMPLETED_DATE, "2026-07-28")


class CliAndRepositorySafetyTests(unittest.TestCase):
    def protected_hashes(self) -> dict[str, str]:
        paths = {
            "manifest": FROZEN_RECOVERY_DIR / "manifest.json",
            "branch_state": FROZEN_RECOVERY_DIR / "branch_state.json",
            "results": FROZEN_RECOVERY_DIR / "results.csv",
            "decision_log": FROZEN_RECOVERY_DIR / "decision_log.jsonl",
            "summary": FROZEN_RECOVERY_DIR / "summary.md",
            "comparison": FROZEN_RECOVERY_DIR / "comparison.png",
            "diagnosis": DIAGNOSIS_DIR / "summary.md",
        }
        return {name: sha256(path) for name, path in paths.items()}

    def test_default_cli_performs_no_transition(self) -> None:
        result = run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stdout.lower())

    def test_validate_only_passes_without_transition(self) -> None:
        result = run_cli("--validate-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("VALIDATION PASS", result.stdout)

    def test_print_coverage_passes_without_transition(self) -> None:
        result = run_cli("--print-coverage")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ARCHITECTURE_DECLARED_SIGNALS 52", result.stdout)
        self.assertIn("DIRECT_INPUT_SUPPORTED 16", result.stdout)

    def test_cli_exposes_no_execution_mode(self) -> None:
        result = run_cli("--execute")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrecognized arguments", result.stderr)

    def test_instrumentation_source_imports_no_rollout_module(self) -> None:
        source = (
            ROOT / "runtime_assurance/staged_recovery_instrumentation.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "recovery_experiment_runner",
            "recovery_branch_runner",
            "phase34_35_transition",
            "explicit_controller_phase34",
            "explicit_controller_phase35",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(f"import {forbidden}", source)

    def test_focused_import_loads_no_simulator_or_controller(self) -> None:
        code = (
            "import sys; import runtime_assurance.staged_recovery_instrumentation; "
            "print(any(name.startswith(('simulator.', 'controller.')) for name in sys.modules))"
        )
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
            env=environment,
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_cli_modifies_no_frozen_or_measured_artifact(self) -> None:
        before = self.protected_hashes()
        result = run_cli("--validate-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, self.protected_hashes())

    def test_instrumentation_artifacts_are_outside_frozen_directories(self) -> None:
        self.assertFalse(INSTRUMENTATION_DIR.is_relative_to(FROZEN_RECOVERY_DIR))
        self.assertFalse(INSTRUMENTATION_DIR.is_relative_to(DIAGNOSIS_DIR))

    def test_generated_json_is_deterministic(self) -> None:
        first = canonical_json_bytes(instrumentation_manifest_document())
        second = canonical_json_bytes(instrumentation_manifest_document())
        self.assertEqual(first, second)

    def test_generated_catalog_matches_checked_in_bytes(self) -> None:
        expected = json.dumps(
            field_catalog_document(),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8") + b"\n"
        self.assertEqual(
            (INSTRUMENTATION_DIR / "field_catalog.json").read_bytes(), expected
        )

    def test_generated_manifest_matches_checked_in_bytes(self) -> None:
        expected = json.dumps(
            instrumentation_manifest_document(),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8") + b"\n"
        self.assertEqual(
            (INSTRUMENTATION_DIR / "instrumentation_manifest.json").read_bytes(),
            expected,
        )

    def test_generated_traceability_matches_checked_in_bytes(self) -> None:
        expected = json.dumps(
            derivation_traceability_document(),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8") + b"\n"
        self.assertEqual(
            (INSTRUMENTATION_DIR / "derivation_traceability.json").read_bytes(),
            expected,
        )

    def test_summary_markdown_is_stable_on_repeated_reads(self) -> None:
        path = INSTRUMENTATION_DIR / "summary.md"
        self.assertEqual(path.read_bytes(), path.read_bytes())

    def test_no_real_trace_artifact_exists(self) -> None:
        forbidden = {
            "results.csv",
            "decision_log.jsonl",
            "comparison.png",
            "trajectory.jsonl",
        }
        self.assertTrue(forbidden.isdisjoint({path.name for path in INSTRUMENTATION_DIR.iterdir()}))

    def test_no_controller_or_physics_edit_is_in_task_diff(self) -> None:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={ROOT.as_posix()}",
                "-C",
                str(ROOT),
                "diff",
                "--name-only",
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        changed = set(result.stdout.splitlines())
        self.assertFalse(
            any(
                path.startswith(("controller/", "simulator/"))
                for path in changed
            )
        )


if __name__ == "__main__":
    unittest.main()

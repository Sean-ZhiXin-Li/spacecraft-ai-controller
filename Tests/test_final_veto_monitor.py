from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from runtime_assurance.final_veto_monitor import (
    FALLBACK_ACTION,
    FALLBACK_PROVEN_SAFE,
    MONITOR_ID,
    OVERSPEED_COMPARATOR,
    OVERSPEED_THRESHOLD,
    PREDICTION_HORIZON_STEPS,
    VALID_DECISIONS,
    FinalVetoDecision,
    MonitorEvaluationError,
    OneStepPrediction,
    evaluate_overspeed_veto,
)
from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435DynamicsContext,
    step_phase34_35_transition,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MONITOR_PATH = REPOSITORY_ROOT / "runtime_assurance" / "final_veto_monitor.py"
MANIFEST_PATH = REPOSITORY_ROOT / "analysis" / "final_veto_ablation_v0" / "manifest.json"
FROZEN_MANIFEST_CANONICAL_SHA256 = (
    "e8ef7954a370ca8d93aae9cb798d674970b82885c5a6d5d39f5bee6b54904c79"
)


class SpyPredictor:
    def __init__(self, ratios: dict[tuple[float, float], float]) -> None:
        self.ratios = ratios
        self.calls: list[tuple[object, tuple[float, float]]] = []

    def __call__(
        self,
        state: object,
        action: tuple[float, float],
    ) -> OneStepPrediction[object]:
        self.calls.append((state, action))
        return OneStepPrediction(
            next_state=("predicted", state, action),
            speed_ratio=self.ratios[action],
        )


class FinalVetoMonitorDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = object()
        self.nominal_action = (0.25, -0.50)

    def decision_for(
        self,
        nominal_ratio: float,
        fallback_ratio: float = 1.0,
    ) -> tuple[FinalVetoDecision, SpyPredictor]:
        predictor = SpyPredictor(
            {
                self.nominal_action: nominal_ratio,
                FALLBACK_ACTION: fallback_ratio,
            }
        )
        decision = evaluate_overspeed_veto(
            self.state,
            self.nominal_action,
            predictor,
        )
        return decision, predictor

    def test_predicted_ratio_below_threshold_allows(self) -> None:
        decision, _ = self.decision_for(1.89)
        self.assertEqual(decision.decision, "allow")
        self.assertFalse(decision.veto_applied)

    def test_predicted_ratio_exactly_at_threshold_allows(self) -> None:
        decision, _ = self.decision_for(1.90)
        self.assertEqual(decision.decision, "allow")
        self.assertFalse(decision.veto_applied)

    def test_predicted_ratio_just_above_threshold_vetoes(self) -> None:
        decision, _ = self.decision_for(math.nextafter(1.90, math.inf))
        self.assertEqual(decision.decision, "veto")
        self.assertTrue(decision.veto_applied)

    def test_veto_substitutes_exact_zero_action(self) -> None:
        decision, _ = self.decision_for(1.91)
        self.assertEqual(decision.executed_action, (0.0, 0.0))
        self.assertEqual(decision.fallback_action, (0.0, 0.0))

    def test_allow_preserves_exact_nominal_action(self) -> None:
        action = (2.5, -3.5)
        predictor = SpyPredictor({action: 1.0})
        decision = evaluate_overspeed_veto(self.state, action, predictor)
        self.assertIs(decision.nominal_action, action)
        self.assertIs(decision.executed_action, action)

    def test_predictor_receives_original_nominal_action_unchanged(self) -> None:
        action = (2.5, -3.5)
        predictor = SpyPredictor({action: 1.0})
        evaluate_overspeed_veto(self.state, action, predictor)
        self.assertIs(predictor.calls[0][1], action)
        self.assertEqual(predictor.calls[0][1], (2.5, -3.5))

    def test_predictor_receives_fallback_after_veto(self) -> None:
        _, predictor = self.decision_for(1.91, fallback_ratio=1.2)
        self.assertEqual(len(predictor.calls), 2)
        self.assertIs(predictor.calls[1][1], FALLBACK_ACTION)

    def test_nominal_and_fallback_use_same_predictor_callback(self) -> None:
        _, predictor = self.decision_for(1.91, fallback_ratio=1.2)
        self.assertEqual(
            [action for _, action in predictor.calls],
            [self.nominal_action, FALLBACK_ACTION],
        )

    def test_fallback_above_threshold_is_recorded(self) -> None:
        decision, _ = self.decision_for(1.91, fallback_ratio=1.95)
        self.assertEqual(decision.predicted_fallback_speed_ratio, 1.95)
        self.assertIs(decision.fallback_predicted_to_exceed_threshold, True)

    def test_fallback_above_threshold_does_not_trigger_another_fallback(self) -> None:
        decision, predictor = self.decision_for(1.91, fallback_ratio=1.95)
        self.assertEqual(decision.decision, "veto")
        self.assertEqual(len(predictor.calls), 2)

    def test_fallback_is_not_evaluated_for_allow(self) -> None:
        decision, predictor = self.decision_for(1.20)
        self.assertEqual(len(predictor.calls), 1)
        self.assertIsNone(decision.predicted_fallback_speed_ratio)
        self.assertIsNone(decision.fallback_predicted_to_exceed_threshold)

    def test_invalid_predicted_ratio_raises_explicit_error(self) -> None:
        predictor = SpyPredictor({self.nominal_action: float("nan")})
        with self.assertRaises(MonitorEvaluationError):
            evaluate_overspeed_veto(self.state, self.nominal_action, predictor)

    def test_invalid_fallback_ratio_raises_explicit_error(self) -> None:
        predictor = SpyPredictor(
            {
                self.nominal_action: 1.91,
                FALLBACK_ACTION: float("nan"),
            }
        )
        with self.assertRaises(MonitorEvaluationError):
            evaluate_overspeed_veto(self.state, self.nominal_action, predictor)

    def test_non_finite_threshold_raises_explicit_error(self) -> None:
        predictor = SpyPredictor({self.nominal_action: 1.0})
        with self.assertRaises(MonitorEvaluationError):
            evaluate_overspeed_veto(
                self.state,
                self.nominal_action,
                predictor,
                threshold=float("inf"),
            )

    def test_malformed_state_raises_explicit_error(self) -> None:
        predictor = SpyPredictor({self.nominal_action: 1.0})
        with self.assertRaises(MonitorEvaluationError):
            evaluate_overspeed_veto(None, self.nominal_action, predictor)

    def test_malformed_action_raises_explicit_error(self) -> None:
        predictor = SpyPredictor({self.nominal_action: 1.0})
        with self.assertRaises(MonitorEvaluationError):
            evaluate_overspeed_veto(
                self.state,
                [0.25, -0.50],  # type: ignore[arg-type]
                predictor,
            )

    def test_invalid_predictor_result_raises_explicit_error(self) -> None:
        def invalid_predictor(state: object, action: tuple[float, float]) -> object:
            return (state, action)

        with self.assertRaises(MonitorEvaluationError):
            evaluate_overspeed_veto(
                self.state,
                self.nominal_action,
                invalid_predictor,  # type: ignore[arg-type]
            )

    def test_predictor_exception_is_wrapped_as_explicit_error(self) -> None:
        def failing_predictor(
            state: object,
            action: tuple[float, float],
        ) -> OneStepPrediction[object]:
            raise TypeError(f"bad input: {state!r}, {action!r}")

        with self.assertRaises(MonitorEvaluationError) as caught:
            evaluate_overspeed_veto(
                self.state,
                self.nominal_action,
                failing_predictor,
            )
        self.assertIsInstance(caught.exception.__cause__, TypeError)

    def test_monitor_does_not_alter_current_state(self) -> None:
        _, predictor = self.decision_for(1.91, fallback_ratio=1.0)
        self.assertTrue(all(state is self.state for state, _ in predictor.calls))

    def test_decision_record_is_immutable(self) -> None:
        decision, _ = self.decision_for(1.0)
        with self.assertRaises(FrozenInstanceError):
            decision.decision = "veto"  # type: ignore[misc]

    def test_valid_decision_outputs_are_only_allow_and_veto(self) -> None:
        allow, _ = self.decision_for(1.0)
        veto, _ = self.decision_for(2.0)
        self.assertEqual(VALID_DECISIONS, frozenset({"allow", "veto"}))
        self.assertEqual({allow.decision, veto.decision}, VALID_DECISIONS)


class FinalVetoMonitorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_monitor_id_matches_frozen_manifest(self) -> None:
        self.assertEqual(MONITOR_ID, "one_step_overspeed_veto_v0")
        self.assertEqual(MONITOR_ID, self.manifest["monitor"]["monitor_id"])

    def test_threshold_comparator_horizon_and_fallback_match_manifest(self) -> None:
        trigger = self.manifest["monitor"]["veto_trigger"]
        fallback = self.manifest["fallback"]
        self.assertEqual(OVERSPEED_THRESHOLD, 1.90)
        self.assertEqual(OVERSPEED_THRESHOLD, trigger["threshold"])
        self.assertEqual(OVERSPEED_COMPARATOR, ">")
        self.assertEqual(OVERSPEED_COMPARATOR, trigger["comparator"])
        self.assertEqual(PREDICTION_HORIZON_STEPS, 1)
        self.assertEqual(
            PREDICTION_HORIZON_STEPS,
            self.manifest["monitor"]["prediction_horizon_steps"],
        )
        self.assertEqual(list(FALLBACK_ACTION), fallback["action"])
        self.assertIs(FALLBACK_PROVEN_SAFE, False)
        self.assertIs(fallback["proven_safe"], False)

    def test_frozen_manifest_content_is_unchanged(self) -> None:
        canonical = json.dumps(
            self.manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            FROZEN_MANIFEST_CANONICAL_SHA256,
        )
        self.assertEqual(
            self.manifest["experiment_status"],
            "design_frozen_not_run",
        )
        self.assertIs(self.manifest["monitor"]["implemented"], False)

    def test_monitor_source_has_no_rollout_implementation_dependencies(self) -> None:
        source = MONITOR_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_roots: set[str] = set()
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
            elif isinstance(node, ast.Name):
                names.add(node.id)

        self.assertLessEqual(
            imported_roots,
            {"__future__", "math", "dataclasses", "numbers", "typing"},
        )
        self.assertTrue(
            names.isdisjoint(
                {
                    "MU",
                    "DT",
                    "MASS",
                    "thrust_scale",
                    "gravity",
                    "gravitational_acceleration",
                    "thrust_acceleration",
                }
            )
        )
        lowered = source.lower()
        for forbidden in (
            "numpy",
            "matplotlib",
            "gravity",
            "gravitational",
            "thrust",
            "thrust_scale",
            "simulator",
            "controller.",
            "envs.",
            "explicit_controller",
            "orbit_env",
            "formal safety",
            "formal-safety",
            "formally safe",
            "guaranteed safe",
            "avoided_failure",
            "blocked_success",
            "results.csv",
        ):
            self.assertNotIn(forbidden, lowered)

    def run_import_probe(self) -> tuple[dict[str, object], list[Path]]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (str(REPOSITORY_ROOT), existing_pythonpath)
            if part
        )
        code = (
            "import json,sys; "
            "import runtime_assurance.final_veto_monitor; "
            "print(json.dumps({'phase_modules': sorted(name for name in sys.modules "
            "if name.startswith('scripts.explicit_controller_'))}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=temp_dir.name,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return json.loads(completed.stdout), list(Path(temp_dir.name).iterdir())

    def test_monitor_imports_without_phase_scripts(self) -> None:
        payload, _ = self.run_import_probe()
        self.assertEqual(payload["phase_modules"], [])

    def test_monitor_import_creates_no_files_or_directories(self) -> None:
        _, entries = self.run_import_probe()
        self.assertEqual(entries, [])

    def test_near_threshold_physical_prediction_uses_injected_transition(self) -> None:
        target_radius = 1.0e6
        context = Phase3435DynamicsContext(
            mu=1.0e10,
            dt=1.0,
            mass=1.0,
            thrust_scale=1.0,
        )
        circular_speed = math.sqrt(context.mu / target_radius)
        state = CartesianState2D(
            x=target_radius,
            y=0.0,
            vx=0.0,
            vy=(1.90 - 1.0e-4) * circular_speed,
        )

        def predictor(
            current_state: CartesianState2D,
            action: tuple[float, float],
        ) -> OneStepPrediction[CartesianState2D]:
            transition = step_phase34_35_transition(
                current_state,
                NormalizedAction2D(*action),
                context,
            )
            next_state = transition.next_state
            speed_ratio = math.sqrt(
                next_state.vx * next_state.vx + next_state.vy * next_state.vy
            ) / circular_speed
            return OneStepPrediction(next_state=next_state, speed_ratio=speed_ratio)

        allow = evaluate_overspeed_veto(state, (0.0, 0.0), predictor)
        veto = evaluate_overspeed_veto(state, (0.0, 0.02), predictor)
        self.assertLess(allow.predicted_nominal_speed_ratio, 1.90)
        self.assertGreater(veto.predicted_nominal_speed_ratio, 1.90)
        self.assertLess(abs(allow.predicted_nominal_speed_ratio - 1.90), 2.0e-4)
        self.assertLess(abs(veto.predicted_nominal_speed_ratio - 1.90), 2.0e-4)
        self.assertEqual(allow.decision, "allow")
        self.assertEqual(veto.decision, "veto")
        self.assertLess(veto.predicted_fallback_speed_ratio, 1.90)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import explicit_controller_phase34_post_cross_sync as phase34
from scripts import explicit_controller_phase35_crossing_basin_expansion as phase35
from simulator.phase34_35_transition import (
    GRAVITY_DENOMINATOR_EPSILON,
    CartesianState2D,
    NormalizedAction2D,
    Phase3435DynamicsContext,
    step_phase34_35_transition,
)


PHASE34_REFERENCE_JSON = r'''{"actual_post_cross_steps":81,"best_distance_to_recoverable":0.9967521592988361,"best_post_cross_r_error_ratio":-2.624540755208333e-09,"best_post_cross_step":1797,"best_post_cross_sync":0.9967521495896127,"best_post_cross_vr_ratio":2.78239297486729e-06,"best_post_cross_vt_error_ratio":-0.24918803739740317,"capture_entered":true,"control_smoothness":3.835300568355358e-06,"controller_name":"phase34_post_cross_sync","crossing_distance_to_recoverable":3.078230071647359,"crossing_occurs":true,"crossing_step":1716,"crossing_sync":3.000745803957997,"crossing_vr_ratio":-0.013726252132647544,"crossing_vt_error_ratio":-0.7501864509894992,"dominant_stage":"phase76_handoff","initial_velocity_angle_deg":150.0,"lock_entered":true,"max_post_cross_control_norm":0.025999999999999995,"max_speed_ratio":1.000000000000007,"mean_post_cross_control_norm":0.02389290676123946,"minimum_abs_radius_error":1161.72265625,"near_miss":false,"overspeed":false,"post_cross_duration":360,"post_cross_mode":"radius_priority","r0_over_target":1.0,"radius_crossings_total":1,"recoverable_crossing":true,"recoverable_state":true,"stage_counts_json":"{\"capture\": 11, \"lock\": 205, \"phase76_handoff\": 1716, \"post_cross_radius_priority\": 81}","steps":2013,"success":true,"tail_mean_abs_vr":9.970352874020607,"target_radius_scale":1.0,"terminated":true,"termination_reason":"success","thrust_scale":8000.0,"truncated":false}'''

PHASE35_REFERENCE_JSON = r'''{"best_crossing_potential":0.774804774612656,"best_post_cross_distance":75.20059002954395,"best_post_cross_step":28,"best_post_cross_sync":74.76956729981178,"capture_entered":false,"closest_angular_momentum_error_ratio":0.1776422744967391,"closest_approach_step":28,"closest_energy_error_ratio":2.5335644280960055,"closest_vr_ratio":1.4595743750990422,"closest_vt_error_ratio":0.20167393676697382,"controller_name":"phase35_crossing_basin_expansion","crossing_occurs":false,"crossing_potential_at_closest":0.5086759829208283,"crossing_step":"","deadness":"near_crossing","dominant_stage":"radial_energy_push","failure_label":"near_crossing","final_angular_momentum_error_ratio":0.17764227449673886,"final_crossing_potential":0.507831750981778,"final_energy_error_ratio":2.6394022768109267,"final_radius_error":-149988033771.83496,"final_radius_error_ratio":-0.019998404502911328,"final_vr_ratio":1.4953913459962358,"final_vt_error_ratio":0.20167383390779112,"initial_velocity_angle_deg":150.0,"instability":false,"lock_entered":false,"max_speed_ratio":1.9183887199363643,"min_abs_radius_error":149988662907.7422,"min_abs_radius_error_ratio":0.019998488387698958,"non_crossing":true,"overspeed":true,"post_cross_mode":"radius_priority","post_cross_steps":0,"r0_over_target":0.98,"radial_velocity_trend":"same_direction_weakening","radius_crossings_total":0,"recoverable_crossing":false,"recoverable_state":false,"stage_counts_json":"{\"radial_energy_push\": 28}","steps":28,"success":false,"target_radius_scale":1.0,"terminated":true,"termination_reason":"overspeed","thrust_scale":8000.0,"truncated":false,"upstream_variant":"radial_energy_push"}'''

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def legacy_phase34_35_step(
    state: CartesianState2D,
    action: NormalizedAction2D,
    context: Phase3435DynamicsContext,
) -> tuple[CartesianState2D, NormalizedAction2D]:
    action_x = max(-1.0, min(1.0, action.action_x))
    action_y = max(-1.0, min(1.0, action.action_y))
    radius = math.sqrt(state.x * state.x + state.y * state.y)
    denominator = radius**3 + 1.0e-12
    acceleration_x = -context.mu * state.x / denominator + context.thrust_scale * action_x / context.mass
    acceleration_y = -context.mu * state.y / denominator + context.thrust_scale * action_y / context.mass
    next_vx = state.vx + acceleration_x * context.dt
    next_vy = state.vy + acceleration_y * context.dt
    next_x = state.x + next_vx * context.dt
    next_y = state.y + next_vy * context.dt
    return (
        CartesianState2D(next_x, next_y, next_vx, next_vy),
        NormalizedAction2D(action_x, action_y),
    )


class Phase3435TransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = CartesianState2D(
            x=7.5e12,
            y=-2.0e11,
            vx=1234.5,
            vy=4567.8,
        )
        self.context = Phase3435DynamicsContext(
            mu=phase34.MU,
            dt=phase34.DT,
            mass=phase34.MASS,
            thrust_scale=8000.0,
        )

    def assert_matches_legacy(self, action: NormalizedAction2D) -> None:
        expected_state, expected_action = legacy_phase34_35_step(
            self.state,
            action,
            self.context,
        )
        actual = step_phase34_35_transition(self.state, action, self.context)
        self.assertEqual(actual.next_state, expected_state)
        self.assertEqual(actual.executed_action, expected_action)

    def test_zero_action_one_step_transition(self) -> None:
        self.assert_matches_legacy(NormalizedAction2D(0.0, 0.0))

    def test_nominal_in_range_action(self) -> None:
        self.assert_matches_legacy(NormalizedAction2D(0.25, -0.75))

    def test_positive_component_clamping(self) -> None:
        result = step_phase34_35_transition(
            self.state,
            NormalizedAction2D(1.25, 2.0),
            self.context,
        )
        self.assertEqual(result.executed_action, NormalizedAction2D(1.0, 1.0))
        self.assert_matches_legacy(NormalizedAction2D(1.25, 2.0))

    def test_negative_component_clamping(self) -> None:
        result = step_phase34_35_transition(
            self.state,
            NormalizedAction2D(-1.25, -2.0),
            self.context,
        )
        self.assertEqual(result.executed_action, NormalizedAction2D(-1.0, -1.0))
        self.assert_matches_legacy(NormalizedAction2D(-1.25, -2.0))

    def test_mixed_component_clamping(self) -> None:
        result = step_phase34_35_transition(
            self.state,
            NormalizedAction2D(1.5, -1.5),
            self.context,
        )
        self.assertEqual(result.executed_action, NormalizedAction2D(1.0, -1.0))
        self.assert_matches_legacy(NormalizedAction2D(1.5, -1.5))

    def test_velocity_is_updated_before_position(self) -> None:
        state = CartesianState2D(x=10.0, y=20.0, vx=3.0, vy=4.0)
        context = Phase3435DynamicsContext(mu=0.0, dt=2.0, mass=1.0, thrust_scale=1.0)
        result = step_phase34_35_transition(
            state,
            NormalizedAction2D(0.5, -0.5),
            context,
        )
        self.assertEqual(result.next_state.vx, 4.0)
        self.assertEqual(result.next_state.vy, 3.0)
        self.assertEqual(result.next_state.x, 18.0)
        self.assertEqual(result.next_state.y, 26.0)
        self.assertNotEqual(result.next_state.x, state.x + state.vx * context.dt)

    def test_exact_gravitational_denominator_epsilon(self) -> None:
        state = CartesianState2D(x=3.0e-5, y=4.0e-5, vx=0.0, vy=0.0)
        context = Phase3435DynamicsContext(mu=2.0, dt=0.5, mass=1.0, thrust_scale=0.0)
        result = step_phase34_35_transition(
            state,
            NormalizedAction2D(0.0, 0.0),
            context,
        )
        radius = math.sqrt(state.x * state.x + state.y * state.y)
        denominator = radius**3 + GRAVITY_DENOMINATOR_EPSILON
        expected_vx = (-context.mu * state.x / denominator) * context.dt
        expected_vy = (-context.mu * state.y / denominator) * context.dt
        self.assertEqual(result.next_state.vx, expected_vx)
        self.assertEqual(result.next_state.vy, expected_vy)

    def test_transition_output_is_finite(self) -> None:
        result = step_phase34_35_transition(
            self.state,
            NormalizedAction2D(0.1, -0.2),
            self.context,
        )
        values = (
            result.next_state.x,
            result.next_state.y,
            result.next_state.vx,
            result.next_state.vy,
            result.executed_action.action_x,
            result.executed_action.action_y,
        )
        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_transition_is_deterministic(self) -> None:
        action = NormalizedAction2D(0.1, -0.2)
        first = step_phase34_35_transition(self.state, action, self.context)
        second = step_phase34_35_transition(self.state, action, self.context)
        self.assertEqual(first, second)

    def test_phase34_context_equality(self) -> None:
        context = Phase3435DynamicsContext(
            mu=phase34.MU,
            dt=phase34.DT,
            mass=phase34.MASS,
            thrust_scale=8000.0,
        )
        self.assertEqual(context, self.context)

    def test_phase35_context_equality(self) -> None:
        context = Phase3435DynamicsContext(
            mu=phase35.MU,
            dt=phase35.DT,
            mass=phase35.MASS,
            thrust_scale=8000.0,
        )
        self.assertEqual(context, self.context)

    def test_extracted_helper_matches_pre_change_reference_formula(self) -> None:
        cases = (
            NormalizedAction2D(0.0, 0.0),
            NormalizedAction2D(0.4, -0.6),
            NormalizedAction2D(2.0, -3.0),
        )
        for action in cases:
            with self.subTest(action=action):
                self.assert_matches_legacy(action)

    def test_transition_module_import_is_pure(self) -> None:
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
            "import simulator.phase34_35_transition; "
            "print(json.dumps({'phase_modules': sorted(name for name in sys.modules "
            "if name.startswith('scripts.explicit_controller_')), "
            "'matplotlib_loaded': 'matplotlib' in sys.modules}))"
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
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["phase_modules"], [])
        self.assertIs(payload["matplotlib_loaded"], False)
        self.assertEqual(list(Path(temp_dir.name).iterdir()), [])


class PhaseRolloutReferenceTests(unittest.TestCase):
    def assert_reference_row(self, actual: dict[str, object], reference_json: str) -> None:
        expected = json.loads(reference_json)
        self.assertEqual(set(actual), set(expected))
        for field, expected_value in expected.items():
            self.assertEqual(actual[field], expected_value, field)

    def test_phase34_pre_post_rollout_reference_output_equality(self) -> None:
        mode = next(mode for mode in phase34.MODES if mode.name == "radius_priority")
        actual = phase34.rollout_phase34_case(
            mode,
            1.00,
            150.0,
            8000.0,
            record_trajectory=False,
        )
        self.assert_reference_row(actual, PHASE34_REFERENCE_JSON)

    def test_phase35_pre_post_rollout_reference_output_equality(self) -> None:
        variant = next(
            variant
            for variant in phase35.VARIANTS
            if variant.name == "radial_energy_push"
        )
        actual = phase35.rollout_phase35_case(
            variant,
            phase35.PHASE34_TERMINAL_MODE,
            0.98,
            150.0,
            8000.0,
            record_trajectory=False,
        )
        self.assert_reference_row(actual, PHASE35_REFERENCE_JSON)


if __name__ == "__main__":
    unittest.main()

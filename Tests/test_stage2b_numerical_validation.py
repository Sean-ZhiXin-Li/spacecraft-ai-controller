import math
import unittest

from runtime_assurance.dop853_validation import (
    step_phase34_35_transition_dop853,
)
from runtime_assurance.recovery_branch_executor import (
    generate_tangential_correction_action,
    generate_velocity_opposed_action,
)
from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435DynamicsContext,
    step_phase34_35_transition,
)


class DOP853ValidationTests(unittest.TestCase):

    def test_constant_velocity_without_force(self) -> None:
        state = CartesianState2D(
            x=10.0,
            y=20.0,
            vx=3.0,
            vy=-4.0,
        )

        action = NormalizedAction2D(
            action_x=0.0,
            action_y=0.0,
        )

        context = Phase3435DynamicsContext(
            mu=0.0,
            dt=5.0,
            mass=1.0,
            thrust_scale=0.0,
        )

        result = step_phase34_35_transition_dop853(
            state,
            action,
            context,
        )

        self.assertAlmostEqual(
            result.next_state.x,
            25.0,
            places=9,
        )

        self.assertAlmostEqual(
            result.next_state.y,
            0.0,
            places=9,
        )

        self.assertAlmostEqual(
            result.next_state.vx,
            3.0,
            places=9,
        )

        self.assertAlmostEqual(
            result.next_state.vy,
            -4.0,
            places=9,
        )

    def test_constant_thrust_matches_analytic_solution(self) -> None:
        state = CartesianState2D(
            x=0.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
        )

        action = NormalizedAction2D(
            action_x=0.5,
            action_y=0.0,
        )

        context = Phase3435DynamicsContext(
            mu=0.0,
            dt=10.0,
            mass=2.0,
            thrust_scale=4.0,
        )

        result = step_phase34_35_transition_dop853(
            state,
            action,
            context,
        )

        self.assertAlmostEqual(
            result.next_state.vx,
            10.0,
            places=9,
        )

        self.assertAlmostEqual(
            result.next_state.x,
            50.0,
            places=9,
        )

        self.assertAlmostEqual(
            result.next_state.vy,
            0.0,
            places=9,
        )

        self.assertAlmostEqual(
            result.next_state.y,
            0.0,
            places=9,
        )

    def test_legacy_phase35_boundary_euler_vs_dop853(self) -> None:
        state = CartesianState2D(
            x=-12481438.746593231,
            y=7350011337081.66,
            vx=-5055.651850370408,
            vy=6140.662726914771,
        )

        action = NormalizedAction2D(
            action_x=-2.3094152700156713e-07,
            action_y=0.13599576748536102,
        )

        context = Phase3435DynamicsContext(
            mu=1.3275182699999999e20,
            dt=100.0,
            mass=722.0,
            thrust_scale=8000.0,
        )

        euler_result = step_phase34_35_transition(
            state,
            action,
            context,
        )

        dop853_result = step_phase34_35_transition_dop853(
            state,
            action,
            context,
        )

        target_speed = 4207.165744298648
        threshold = 1.90

        euler_speed = math.hypot(
            euler_result.next_state.vx,
            euler_result.next_state.vy,
        )

        dop853_speed = math.hypot(
            dop853_result.next_state.vx,
            dop853_result.next_state.vy,
        )

        euler_speed_ratio = euler_speed / target_speed
        dop853_speed_ratio = dop853_speed / target_speed

        euler_veto = euler_speed_ratio > threshold
        dop853_veto = dop853_speed_ratio > threshold

        print()
        print("=== Legacy Phase35 Boundary ===")
        print("Euler next state:")
        print(euler_result.next_state)
        print("DOP853 next state:")
        print(dop853_result.next_state)
        print("Euler speed ratio:", euler_speed_ratio)
        print("DOP853 speed ratio:", dop853_speed_ratio)
        print("Euler veto:", euler_veto)
        print("DOP853 veto:", dop853_veto)

        print(
            "Delta x:",
            dop853_result.next_state.x
            - euler_result.next_state.x,
        )

        print(
            "Delta y:",
            dop853_result.next_state.y
            - euler_result.next_state.y,
        )

        print(
            "Delta vx:",
            dop853_result.next_state.vx
            - euler_result.next_state.vx,
        )

        print(
            "Delta vy:",
            dop853_result.next_state.vy
            - euler_result.next_state.vy,
        )

        self.assertAlmostEqual(
            euler_result.next_state.x,
            -12987003.95721932,
            places=6,
        )

        self.assertAlmostEqual(
            euler_result.next_state.y,
            7350011966216.691,
            places=3,
        )

        self.assertAlmostEqual(
            euler_result.next_state.vx,
            -5055.65210626088,
            places=9,
        )

        self.assertAlmostEqual(
            euler_result.next_state.vy,
            6291.350312189912,
            places=9,
        )

        self.assertAlmostEqual(
            euler_speed_ratio,
            1.9183887199363643,
            places=12,
        )

        self.assertTrue(
            math.isfinite(dop853_speed_ratio)
        )

        self.assertTrue(euler_veto)
        self.assertTrue(dop853_veto)

        self.assertEqual(
            euler_veto,
            dop853_veto,
        )

    def test_legacy_phase35_zero_action_classification_consistency(
        self,
    ) -> None:
        state = CartesianState2D(
            x=-12481438.746593231,
            y=7350011337081.66,
            vx=-5055.651850370408,
            vy=6140.662726914771,
        )

        action = NormalizedAction2D(
            action_x=0.0,
            action_y=0.0,
        )

        context = Phase3435DynamicsContext(
            mu=1.3275182699999999e20,
            dt=100.0,
            mass=722.0,
            thrust_scale=8000.0,
        )

        euler_result = step_phase34_35_transition(
            state,
            action,
            context,
        )

        dop853_result = step_phase34_35_transition_dop853(
            state,
            action,
            context,
        )

        target_speed = 4207.165744298648
        threshold = 1.90

        euler_speed_ratio = (
            math.hypot(
                euler_result.next_state.vx,
                euler_result.next_state.vy,
            )
            / target_speed
        )

        dop853_speed_ratio = (
            math.hypot(
                dop853_result.next_state.vx,
                dop853_result.next_state.vy,
            )
            / target_speed
        )

        euler_veto = euler_speed_ratio > threshold
        dop853_veto = dop853_speed_ratio > threshold

        print()
        print("=== Legacy Phase35 Zero Action ===")
        print("Euler speed ratio:", euler_speed_ratio)
        print("DOP853 speed ratio:", dop853_speed_ratio)

        print(
            "Speed-ratio difference:",
            dop853_speed_ratio - euler_speed_ratio,
        )

        print(
            "Euler headroom:",
            threshold - euler_speed_ratio,
        )

        print(
            "DOP853 headroom:",
            threshold - dop853_speed_ratio,
        )

        print("Euler veto:", euler_veto)
        print("DOP853 veto:", dop853_veto)

        self.assertAlmostEqual(
            euler_speed_ratio,
            1.8906024003603095,
            places=12,
        )

        self.assertTrue(
            math.isfinite(dop853_speed_ratio)
        )

        self.assertFalse(euler_veto)
        self.assertFalse(dop853_veto)

        self.assertEqual(
            euler_veto,
            dop853_veto,
        )

    def test_legacy_phase35_active_alternatives_classification_consistency(
        self,
    ) -> None:
        state = CartesianState2D(
            x=-12481438.746593231,
            y=7350011337081.66,
            vx=-5055.651850370408,
            vy=6140.662726914771,
        )

        context = Phase3435DynamicsContext(
            mu=1.3275182699999999e20,
            dt=100.0,
            mass=722.0,
            thrust_scale=8000.0,
        )

        target_speed = 4207.165744298648
        threshold = 1.90

        velocity_opposed_tuple = generate_velocity_opposed_action(
            state,
        )

        tangential_correction_tuple = (
            generate_tangential_correction_action(
                state,
                target_speed,
            )
        )

        actions = {
            "velocity_opposed": NormalizedAction2D(
                action_x=velocity_opposed_tuple[0],
                action_y=velocity_opposed_tuple[1],
            ),
            "tangential_correction": NormalizedAction2D(
                action_x=tangential_correction_tuple[0],
                action_y=tangential_correction_tuple[1],
            ),
        }

        for action_name, action in actions.items():
            with self.subTest(action=action_name):
                euler_result = step_phase34_35_transition(
                    state,
                    action,
                    context,
                )

                dop853_result = (
                    step_phase34_35_transition_dop853(
                        state,
                        action,
                        context,
                    )
                )

                euler_speed_ratio = (
                    math.hypot(
                        euler_result.next_state.vx,
                        euler_result.next_state.vy,
                    )
                    / target_speed
                )

                dop853_speed_ratio = (
                    math.hypot(
                        dop853_result.next_state.vx,
                        dop853_result.next_state.vy,
                    )
                    / target_speed
                )

                euler_veto = (
                    euler_speed_ratio > threshold
                )

                dop853_veto = (
                    dop853_speed_ratio > threshold
                )

                print()
                print(
                    f"=== {action_name} ==="
                )

                print(
                    "Action:",
                    action,
                )

                print(
                    "Euler speed ratio:",
                    euler_speed_ratio,
                )

                print(
                    "DOP853 speed ratio:",
                    dop853_speed_ratio,
                )

                print(
                    "Difference:",
                    dop853_speed_ratio
                    - euler_speed_ratio,
                )

                print(
                    "Euler headroom:",
                    threshold - euler_speed_ratio,
                )

                print(
                    "DOP853 headroom:",
                    threshold - dop853_speed_ratio,
                )

                print(
                    "Euler veto:",
                    euler_veto,
                )

                print(
                    "DOP853 veto:",
                    dop853_veto,
                )

                self.assertTrue(
                    math.isfinite(
                        dop853_speed_ratio
                    )
                )

                self.assertFalse(
                    euler_veto
                )

                self.assertFalse(
                    dop853_veto
                )

                self.assertEqual(
                    euler_veto,
                    dop853_veto,
                )


if __name__ == "__main__":
    unittest.main()
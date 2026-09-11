from __future__ import annotations

import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from runtime_assurance.dop853_validation import (
    step_phase34_35_transition_dop853,
)
from simulator.phase34_35_transition import (
    CartesianState2D,
    NormalizedAction2D,
    Phase3435DynamicsContext,
    step_phase34_35_transition,
)


INPUT_PATH = Path(
    "analysis/stage2a_post_veto_alternative_audit_v0"
    "/exact_state_comparisons.json"
)

OUTPUT_PATH = Path(
    "analysis/stage2b_numerical_validation_v0"
    "/results.json"
)

MU = 1.3275182699999999e20
DT = 100.0
MASS = 722.0
TARGET_SPEED = 4207.165744298648
OVERSPEED_THRESHOLD = 1.90


def thrust_scale_from_case_id(
    case_id: str,
) -> float:
    marker = "__thrust_"

    if marker not in case_id:
        raise ValueError(
            f"case_id does not contain thrust scale: {case_id}"
        )

    thrust_text = case_id.rsplit(
        marker,
        maxsplit=1,
    )[1]

    return float(thrust_text)


def speed_ratio(
    state: CartesianState2D,
) -> float:
    speed = math.hypot(
        state.vx,
        state.vy,
    )

    return speed / TARGET_SPEED


def main() -> None:
    document = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8",
        )
    )

    comparisons = document["comparisons"]

    validation_records = []

    print(
        "schema_version:",
        document["schema_version"],
    )

    print(
        "exact_state_comparison_count:",
        len(comparisons),
    )

    print()

    # ------------------------------------------------------------
    # Part 1:
    # Validate frozen nominal proposals with Euler vs DOP853.
    # ------------------------------------------------------------

    nominal_classification_match_count = 0

    print("=== Nominal Proposals ===")
    print()

    for comparison in comparisons:
        case_id = comparison["case_id"]

        state_values = comparison[
            "state_values"
        ]

        nominal = comparison[
            "nominal_proposal"
        ]

        nominal_action_values = nominal[
            "action"
        ]

        frozen_ratio = nominal[
            "predicted_speed_ratio"
        ]

        state = CartesianState2D(
            x=float(state_values[0]),
            y=float(state_values[1]),
            vx=float(state_values[2]),
            vy=float(state_values[3]),
        )

        action = NormalizedAction2D(
            action_x=float(
                nominal_action_values[0]
            ),
            action_y=float(
                nominal_action_values[1]
            ),
        )

        thrust_scale = (
            thrust_scale_from_case_id(
                case_id
            )
        )

        context = Phase3435DynamicsContext(
            mu=MU,
            dt=DT,
            mass=MASS,
            thrust_scale=thrust_scale,
        )

        euler_result = (
            step_phase34_35_transition(
                state,
                action,
                context,
            )
        )

        dop853_result = (
            step_phase34_35_transition_dop853(
                state,
                action,
                context,
            )
        )

        euler_ratio = speed_ratio(
            euler_result.next_state
        )

        dop853_ratio = speed_ratio(
            dop853_result.next_state
        )

        euler_veto = (
            euler_ratio
            > OVERSPEED_THRESHOLD
        )

        dop853_veto = (
            dop853_ratio
            > OVERSPEED_THRESHOLD
        )

        classification_match = (
            euler_veto == dop853_veto
        )

        validation_records.append(
            {
                "case_id": case_id,
                "proposal_type": "nominal",
                "action_identity": "nominal_proposal",
                "action": [
                    action.action_x,
                    action.action_y,
                ],
                "thrust_scale": thrust_scale,
                "frozen_euler_speed_ratio": float(
                    frozen_ratio
                ),
                "recomputed_euler_speed_ratio": (
                    euler_ratio
                ),
                "dop853_speed_ratio": (
                    dop853_ratio
                ),
                "dop853_minus_euler": (
                    dop853_ratio
                    - euler_ratio
                ),
                "euler_veto": euler_veto,
                "dop853_veto": dop853_veto,
                "classification_match": (
                    classification_match
                ),
            }
        )

        if classification_match:
            nominal_classification_match_count += 1

        print(
            "case_id:",
            case_id,
        )

        print(
            "thrust_scale:",
            thrust_scale,
        )

        print(
            "frozen Euler ratio:",
            frozen_ratio,
        )

        print(
            "recomputed Euler ratio:",
            euler_ratio,
        )

        print(
            "DOP853 ratio:",
            dop853_ratio,
        )

        print(
            "DOP853 - Euler:",
            dop853_ratio
            - euler_ratio,
        )

        print(
            "Euler veto:",
            euler_veto,
        )

        print(
            "DOP853 veto:",
            dop853_veto,
        )

        print(
            "classification_match:",
            classification_match,
        )

        print()

        if not math.isclose(
            euler_ratio,
            float(frozen_ratio),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise AssertionError(
                "Euler reproduction failed "
                f"for {case_id}: "
                f"{euler_ratio} != "
                f"{frozen_ratio}"
            )

    print(
        "nominal_case_count:",
        len(comparisons),
    )

    print(
        "nominal_classification_match_count:",
        nominal_classification_match_count,
    )

    print(
        "all_nominal_classifications_match:",
        nominal_classification_match_count
        == len(comparisons),
    )

    # ------------------------------------------------------------
    # Part 2:
    # Validate all frozen exact physical alternatives.
    #
    # Important:
    # We use the exact action stored in the frozen artifact.
    # We do NOT regenerate actions here, because Stage 2B is
    # isolating propagation-method dependence.
    # ------------------------------------------------------------

    alternative_case_count = 0
    alternative_classification_match_count = 0

    print()
    print("=== Exact Physical Alternatives ===")
    print()

    for comparison in comparisons:
        case_id = comparison["case_id"]

        state_values = comparison[
            "state_values"
        ]

        state = CartesianState2D(
            x=float(state_values[0]),
            y=float(state_values[1]),
            vx=float(state_values[2]),
            vy=float(state_values[3]),
        )

        thrust_scale = (
            thrust_scale_from_case_id(
                case_id
            )
        )

        context = Phase3435DynamicsContext(
            mu=MU,
            dt=DT,
            mass=MASS,
            thrust_scale=thrust_scale,
        )

        alternatives = comparison[
            "alternatives"
        ]

        for alternative in alternatives:
            action_values = alternative.get(
                "action"
            )

            frozen_ratio = alternative.get(
                "predicted_speed_ratio"
            )

            # explicit_abort_v0 and unavailable exact-state
            # alternatives have no physical action to propagate.
            if (
                action_values is None
                or frozen_ratio is None
            ):
                continue

            action_identity = alternative[
                "action_identity"
            ]

            action = NormalizedAction2D(
                action_x=float(
                    action_values[0]
                ),
                action_y=float(
                    action_values[1]
                ),
            )

            euler_result = (
                step_phase34_35_transition(
                    state,
                    action,
                    context,
                )
            )

            dop853_result = (
                step_phase34_35_transition_dop853(
                    state,
                    action,
                    context,
                )
            )

            euler_ratio = speed_ratio(
                euler_result.next_state
            )

            dop853_ratio = speed_ratio(
                dop853_result.next_state
            )

            euler_veto = (
                euler_ratio
                > OVERSPEED_THRESHOLD
            )

            dop853_veto = (
                dop853_ratio
                > OVERSPEED_THRESHOLD
            )

            classification_match = (
                euler_veto
                == dop853_veto
            )

            validation_records.append(
                {
                    "case_id": case_id,
                    "proposal_type": (
                        "physical_alternative"
                    ),
                    "action_identity": (
                        action_identity
                    ),
                    "action": [
                        action.action_x,
                        action.action_y,
                    ],
                    "thrust_scale": (
                        thrust_scale
                    ),
                    "frozen_euler_speed_ratio": float(
                        frozen_ratio
                    ),
                    "recomputed_euler_speed_ratio": (
                        euler_ratio
                    ),
                    "dop853_speed_ratio": (
                        dop853_ratio
                    ),
                    "dop853_minus_euler": (
                        dop853_ratio
                        - euler_ratio
                    ),
                    "euler_veto": (
                        euler_veto
                    ),
                    "dop853_veto": (
                        dop853_veto
                    ),
                    "classification_match": (
                        classification_match
                    ),
                }
            )

            alternative_case_count += 1

            if classification_match:
                alternative_classification_match_count += 1

            print(
                "case_id:",
                case_id,
            )

            print(
                "action_identity:",
                action_identity,
            )

            print(
                "thrust_scale:",
                thrust_scale,
            )

            print(
                "frozen Euler ratio:",
                frozen_ratio,
            )

            print(
                "recomputed Euler ratio:",
                euler_ratio,
            )

            print(
                "DOP853 ratio:",
                dop853_ratio,
            )

            print(
                "DOP853 - Euler:",
                dop853_ratio
                - euler_ratio,
            )

            print(
                "Euler veto:",
                euler_veto,
            )

            print(
                "DOP853 veto:",
                dop853_veto,
            )

            print(
                "classification_match:",
                classification_match,
            )

            print()

            if not math.isclose(
                euler_ratio,
                float(frozen_ratio),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise AssertionError(
                    "Alternative Euler reproduction "
                    f"failed for {case_id} / "
                    f"{action_identity}: "
                    f"{euler_ratio} != "
                    f"{frozen_ratio}"
                )

    print(
        "alternative_case_count:",
        alternative_case_count,
    )

    print(
        "alternative_classification_match_count:",
        alternative_classification_match_count,
    )

    print(
        "all_alternative_classifications_match:",
        alternative_classification_match_count
        == alternative_case_count,
    )

    # ------------------------------------------------------------
    # Overall Stage 2B one-step exact-state summary.
    # ------------------------------------------------------------

    total_evaluated_count = (
        len(comparisons)
        + alternative_case_count
    )

    total_match_count = (
        nominal_classification_match_count
        + alternative_classification_match_count
    )

    print()
    print(
        "=== Stage 2B One-Step Exact-State Summary ==="
    )

    print(
        "total_evaluated_proposals:",
        total_evaluated_count,
    )

    print(
        "total_classification_matches:",
        total_match_count,
    )

    print(
        "all_classifications_match:",
        total_match_count
        == total_evaluated_count,
    )

    maximum_absolute_speed_ratio_difference = max(
        abs(record["dop853_minus_euler"])
        for record in validation_records
    )

    output_document = {
        "schema_version": (
            "stage2b_numerical_validation_v0"
        ),
        "validation_scope": (
            "one_step_exact_state"
        ),
        "baseline_propagator": (
            "phase34_35_semi_implicit_euler"
        ),
        "independent_propagator": (
            "scipy_solve_ivp_DOP853"
        ),
        "overspeed_threshold": (
            OVERSPEED_THRESHOLD
        ),
        "exact_state_count": (
            len(comparisons)
        ),
        "nominal_proposal_count": (
            len(comparisons)
        ),
        "physical_alternative_count": (
            alternative_case_count
        ),
        "total_evaluated_proposal_count": (
            total_evaluated_count
        ),
        "classification_match_count": (
            total_match_count
        ),
        "all_classifications_match": (
            total_match_count
            == total_evaluated_count
        ),
        "maximum_absolute_speed_ratio_difference": (
            maximum_absolute_speed_ratio_difference
        ),
        "records": validation_records,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output_document,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "wrote:",
        OUTPUT_PATH,
    )

    print(
        "maximum_absolute_speed_ratio_difference:",
        maximum_absolute_speed_ratio_difference,
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Mapping, Sequence

from runtime_assurance.final_veto_monitor import (
    OVERSPEED_COMPARATOR,
    OVERSPEED_THRESHOLD,
)
from runtime_assurance.recovery_evaluators import (
    EVALUATION_CLEAR,
    EVALUATION_INVALID,
    EVALUATION_NOT_EVALUATED,
    EVALUATION_TRIGGERED,
    PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
    PHASE34_RECOVERABLE_VR_RATIO_MAX,
    PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
    evaluate_phase34_compatible_recoverability,
)
from runtime_assurance.staged_recovery_contract import (
    ARCHITECTURE_ID,
    FROZEN_ADVERSE_STOP_PRIORITY,
    RecoveryPhase,
    RecoverySignalStatus,
    SIGNAL_IDS as ARCHITECTURE_SIGNAL_IDS,
    default_phase_contracts,
)


INSTRUMENTATION_ID = "staged_recovery_instrumentation_v0"
INSTRUMENTATION_SCHEMA_VERSION = "staged_recovery_instrumentation_record_v0"
INSTRUMENTATION_MANIFEST_SCHEMA_VERSION = (
    "staged_recovery_instrumentation_manifest_v0"
)
FIELD_CATALOG_SCHEMA_VERSION = "staged_recovery_instrumentation_field_catalog_v0"
DERIVATION_TRACEABILITY_SCHEMA_VERSION = (
    "staged_recovery_instrumentation_derivation_traceability_v0"
)
COMPLETED_DATE = "2026-07-28"

SOURCE_ARCHITECTURE_COMMIT = "0d416603027e8a27991baf4f89445f6f466b86e6"
SOURCE_ARCHITECTURE_CANONICAL_HASH = (
    "22fa7e0f01c7836ecb1f10838ef00c4cafa937d212bba579fffb25e2c8f11971"
)

POSITION_NORM_ZERO_TOLERANCE = 0.0
VECTOR_ZERO_TOLERANCE = 1.0e-12
RATIO_DENOMINATOR_EPSILON = 1.0e-12
SPEED_RATIO_DENOMINATOR_EPSILON = 1.0e-12
SPECIFIC_ENERGY_RADIUS_EPSILON = 1.0e-12
NORMALIZED_ACTION_COMPONENT_LIMIT = 1.0

CURRENT_GRAVITY_MODEL_ID = "phase34_35_newtonian_2d_softened_denominator_v0"
SPECIFIC_ENERGY_MODEL_STATUS = "declared_diagnostic_proxy"

PURE_DERIVATION_IMPLEMENTED = "implemented"
RUNTIME_LOGGER_NOT_IMPLEMENTED = "not_implemented"
STAGED_EXECUTION_NOT_AUTHORIZED = "not_authorized"
EXECUTION_NOT_AUTHORIZED_REASON = (
    "the observation schema and pure derivations are implemented, but no runtime "
    "logger integration, phase action law, numerical guard, no-progress threshold, "
    "hysteresis parameter, or staged recovery execution path has been frozen"
)

COVERAGE_DIRECT_INPUT = "direct_input_supported"
COVERAGE_PURE_DERIVATION = "pure_derivation_supported"
COVERAGE_PREVIOUS_STATE = "requires_previous_state"
COVERAGE_PREDICTED_STATE = "requires_predicted_state"
COVERAGE_RUNTIME_INTEGRATION = "requires_runtime_phase_integration"
COVERAGE_FUTURE_EVALUATOR = "requires_future_evaluator"
COVERAGE_NOT_SUPPORTED = "not_yet_supported"
COVERAGE_CLASSIFICATIONS = (
    COVERAGE_DIRECT_INPUT,
    COVERAGE_PURE_DERIVATION,
    COVERAGE_PREVIOUS_STATE,
    COVERAGE_PREDICTED_STATE,
    COVERAGE_RUNTIME_INTEGRATION,
    COVERAGE_FUTURE_EVALUATOR,
    COVERAGE_NOT_SUPPORTED,
)

SOURCE_ARCHITECTURE_ARTIFACT_HASHES = (
    (
        "runtime_assurance/staged_recovery_contract.py",
        "ae32a961b30e5f1c4fcbb59fe9c0902f6d82bb2deafb8576ad36307b90e35b53",
    ),
    (
        "Tests/test_staged_recovery_contract.py",
        "b4888330860992595603329e6ea4584f4f41715c2da843826e7f30d184504c40",
    ),
    (
        "docs/architecture/staged_recovery_architecture_v0.md",
        "728f534dab6e68b0913568aa395e21b948670c65eeacd741204ff365c9da1c70",
    ),
    (
        "docs/experiments/staged_recovery_minimal_experiment_plan_v0.md",
        "3106b5852211729d345341e2ef3f044cea09cc0b42444d0a176195c6811e2b8b",
    ),
    (
        "analysis/staged_recovery_architecture_v0/architecture_manifest.json",
        "fbad0f6cbfba884a244bdccb9e12693bf3d8b5f8eefbf5a682065f740a7beb7c",
    ),
    (
        "analysis/staged_recovery_architecture_v0/evidence_traceability.json",
        "e848f9724aea1e114cc81fe65dc198069fc44f2d0ee78f15dcec1b6b958cf499",
    ),
    (
        "analysis/staged_recovery_architecture_v0/summary.md",
        "809bbb0f8bf9c4492c169bbe0f17bb749fd3b859a286d306cd0093fad2d39746",
    ),
)


class InstrumentationEvidenceStatus(str, Enum):
    MEASURED = "measured"
    DERIVED = "derived"
    ONE_STEP_PREDICTED = "one_step_predicted"
    MULTI_STEP_PREDICTED = "multi_step_predicted"
    HEURISTIC = "heuristic"
    NOT_EVALUATED = "not_evaluated"
    INVALID = "invalid"


EVIDENCE_STATUS_TO_ARCHITECTURE_STATUS = tuple(
    (status.value, RecoverySignalStatus(status.value).value)
    for status in InstrumentationEvidenceStatus
)


class InstrumentationContractError(ValueError):
    pass


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in value.items()
        )
    return False


def _freeze_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(
            (key, _freeze_json_value(item))
            for key, item in sorted(value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class InstrumentedValue:
    value: object
    status: InstrumentationEvidenceStatus
    reason: str
    units: str
    source_id: str
    source_step: int | None
    valid: bool
    input_source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze_json_value(self.value))
        object.__setattr__(
            self,
            "input_source_ids",
            tuple(self.input_source_ids),
        )
        if not isinstance(self.status, InstrumentationEvidenceStatus):
            raise InstrumentationContractError("unsupported evidence status")
        if not self.reason or not self.units or not self.source_id:
            raise InstrumentationContractError(
                "reason, units, and source_id must be nonempty"
            )
        if self.source_step is not None and (
            isinstance(self.source_step, bool)
            or not isinstance(self.source_step, int)
            or self.source_step < 0
        ):
            raise InstrumentationContractError(
                "source_step must be a nonnegative integer or None"
            )
        if self.input_source_ids != tuple(sorted(set(self.input_source_ids))):
            raise InstrumentationContractError(
                "input_source_ids must be unique and sorted"
            )
        unavailable = self.status in {
            InstrumentationEvidenceStatus.NOT_EVALUATED,
            InstrumentationEvidenceStatus.INVALID,
        }
        if unavailable:
            if self.value is not None or self.valid:
                raise InstrumentationContractError(
                    "not_evaluated and invalid values must carry null and valid=false"
                )
        elif self.value is None or not self.valid:
            raise InstrumentationContractError(
                "evaluated evidence must carry a value and valid=true"
            )
        if not _is_json_value(self.value):
            raise InstrumentationContractError(
                "instrumented value must be finite and canonical-JSON compatible"
            )

    @property
    def available(self) -> bool:
        return self.status not in {
            InstrumentationEvidenceStatus.NOT_EVALUATED,
            InstrumentationEvidenceStatus.INVALID,
        }


def measured_value(
    value: object,
    *,
    units: str,
    source_id: str,
    source_step: int | None = None,
    reason: str = "explicit_measured_input",
) -> InstrumentedValue:
    if not _is_json_value(value) or value is None:
        return invalid_value(
            reason="measured_input_is_missing_nonfinite_or_not_serializable",
            units=units,
            source_id=source_id,
            source_step=source_step,
        )
    return InstrumentedValue(
        value=value,
        status=InstrumentationEvidenceStatus.MEASURED,
        reason=reason,
        units=units,
        source_id=source_id,
        source_step=source_step,
        valid=True,
    )


def derived_value(
    value: object,
    *,
    units: str,
    source_id: str,
    source_step: int | None = None,
    reason: str = "pure_derivation",
    input_source_ids: Sequence[str] = (),
) -> InstrumentedValue:
    if not _is_json_value(value) or value is None:
        return invalid_value(
            reason="derived_value_is_missing_nonfinite_or_not_serializable",
            units=units,
            source_id=source_id,
            source_step=source_step,
            input_source_ids=input_source_ids,
        )
    return InstrumentedValue(
        value=value,
        status=InstrumentationEvidenceStatus.DERIVED,
        reason=reason,
        units=units,
        source_id=source_id,
        source_step=source_step,
        valid=True,
        input_source_ids=tuple(sorted(set(input_source_ids))),
    )


def predicted_value(
    value: object,
    *,
    units: str,
    source_id: str,
    source_step: int | None = None,
    horizon_steps: int = 1,
    input_source_ids: Sequence[str] = (),
) -> InstrumentedValue:
    if isinstance(horizon_steps, bool) or not isinstance(horizon_steps, int) or horizon_steps < 1:
        raise InstrumentationContractError("prediction horizon must be a positive integer")
    status = (
        InstrumentationEvidenceStatus.ONE_STEP_PREDICTED
        if horizon_steps == 1
        else InstrumentationEvidenceStatus.MULTI_STEP_PREDICTED
    )
    if not _is_json_value(value) or value is None:
        return invalid_value(
            reason="predicted_value_is_missing_nonfinite_or_not_serializable",
            units=units,
            source_id=source_id,
            source_step=source_step,
            input_source_ids=input_source_ids,
        )
    return InstrumentedValue(
        value=value,
        status=status,
        reason=f"explicit_{horizon_steps}_step_prediction",
        units=units,
        source_id=source_id,
        source_step=source_step,
        valid=True,
        input_source_ids=tuple(sorted(set(input_source_ids))),
    )


def not_evaluated_value(
    *,
    reason: str,
    units: str,
    source_id: str,
    source_step: int | None = None,
    input_source_ids: Sequence[str] = (),
) -> InstrumentedValue:
    return InstrumentedValue(
        value=None,
        status=InstrumentationEvidenceStatus.NOT_EVALUATED,
        reason=reason,
        units=units,
        source_id=source_id,
        source_step=source_step,
        valid=False,
        input_source_ids=tuple(sorted(set(input_source_ids))),
    )


def invalid_value(
    *,
    reason: str,
    units: str,
    source_id: str,
    source_step: int | None = None,
    input_source_ids: Sequence[str] = (),
) -> InstrumentedValue:
    return InstrumentedValue(
        value=None,
        status=InstrumentationEvidenceStatus.INVALID,
        reason=reason,
        units=units,
        source_id=source_id,
        source_step=source_step,
        valid=False,
        input_source_ids=tuple(sorted(set(input_source_ids))),
    )


@dataclass(frozen=True, slots=True)
class CartesianState2D:
    x: float
    y: float
    vx: float
    vy: float


@dataclass(frozen=True, slots=True)
class OrbitalConfiguration:
    mu: float
    target_radius: float
    ratio_denominator_epsilon: float = RATIO_DENOMINATOR_EPSILON
    speed_ratio_denominator_epsilon: float = SPEED_RATIO_DENOMINATOR_EPSILON
    specific_energy_radius_epsilon: float = SPECIFIC_ENERGY_RADIUS_EPSILON
    action_component_limit: float = NORMALIZED_ACTION_COMPONENT_LIMIT
    gravity_model_id: str = CURRENT_GRAVITY_MODEL_ID


@dataclass(frozen=True, slots=True)
class OrbitalBasis2D:
    radius: InstrumentedValue
    radial_unit_vector: InstrumentedValue
    tangential_unit_vector: InstrumentedValue
    speed_magnitude: InstrumentedValue


@dataclass(frozen=True, slots=True)
class OrbitalDerivedState:
    values: tuple[tuple[str, InstrumentedValue], ...]

    def __post_init__(self) -> None:
        _validate_value_pairs(self.values, "orbital derived state")

    def field(self, field_id: str) -> InstrumentedValue:
        return _field_from_pairs(self.values, field_id)


@dataclass(frozen=True, slots=True)
class RecoverabilityComponents:
    values: tuple[tuple[str, InstrumentedValue], ...]

    def __post_init__(self) -> None:
        _validate_value_pairs(self.values, "recoverability components")

    def field(self, field_id: str) -> InstrumentedValue:
        return _field_from_pairs(self.values, field_id)


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    values: tuple[tuple[str, InstrumentedValue], ...]

    def __post_init__(self) -> None:
        _validate_value_pairs(self.values, "crossing event")

    def field(self, field_id: str) -> InstrumentedValue:
        return _field_from_pairs(self.values, field_id)


@dataclass(frozen=True, slots=True)
class RecoveryProgressSample:
    values: tuple[tuple[str, InstrumentedValue], ...]

    def __post_init__(self) -> None:
        _validate_value_pairs(self.values, "progress sample")

    def field(self, field_id: str) -> InstrumentedValue:
        return _field_from_pairs(self.values, field_id)


@dataclass(frozen=True, slots=True)
class ActionGeometry:
    values: tuple[tuple[str, InstrumentedValue], ...]

    def __post_init__(self) -> None:
        _validate_value_pairs(self.values, "action geometry")

    def field(self, field_id: str) -> InstrumentedValue:
        return _field_from_pairs(self.values, field_id)


@dataclass(frozen=True, slots=True)
class StagedRecoveryInstrumentationRecord:
    schema_version: str
    fields: tuple[tuple[str, InstrumentedValue], ...]
    volatile_provenance_timestamp: str | None
    canonical_record_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != INSTRUMENTATION_SCHEMA_VERSION:
            raise InstrumentationContractError("unsupported instrumentation schema")
        _validate_value_pairs(
            self.fields,
            "instrumentation record",
            expected_keys=CANONICAL_FIELD_ORDER,
        )
        if self.canonical_record_hash and (
            len(self.canonical_record_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.canonical_record_hash)
        ):
            raise InstrumentationContractError("canonical_record_hash must be SHA-256")

    def field(self, field_id: str) -> InstrumentedValue:
        return _field_from_pairs(self.fields, field_id)


@dataclass(frozen=True, slots=True)
class InstrumentationFieldDefinition:
    field_id: str
    category: str
    description: str
    data_type: str
    units: str
    signedness: str
    evidence_status_requirements: tuple[str, ...]
    support_classification: str
    direct_or_derived: str
    required_inputs: tuple[str, ...]
    derivation_source: str
    missing_value_rule: str
    invalid_value_rule: str
    architecture_phases_requiring_it: tuple[str, ...]
    scientific_interpretation_limitation: str
    canonical_order_index: int


@dataclass(frozen=True, slots=True)
class InstrumentationValidationReport:
    valid: bool
    errors: tuple[str, ...]
    coverage_counts: tuple[tuple[str, int], ...]
    architecture_signal_count: int
    catalog_field_count: int


def _validate_value_pairs(
    pairs: tuple[tuple[str, InstrumentedValue], ...],
    name: str,
    *,
    expected_keys: tuple[str, ...] | None = None,
) -> None:
    keys = tuple(key for key, _ in pairs)
    required_order = tuple(sorted(keys)) if expected_keys is None else expected_keys
    if keys != required_order or len(keys) != len(set(keys)):
        raise InstrumentationContractError(
            f"{name} fields must use unique keys in the required canonical order"
        )
    if any(not key or not isinstance(value, InstrumentedValue) for key, value in pairs):
        raise InstrumentationContractError(
            f"{name} fields require nonempty IDs and InstrumentedValue values"
        )


def _field_from_pairs(
    pairs: tuple[tuple[str, InstrumentedValue], ...], field_id: str
) -> InstrumentedValue:
    for current_id, value in pairs:
        if current_id == field_id:
            return value
    raise KeyError(field_id)


def _pairs(**values: InstrumentedValue) -> tuple[tuple[str, InstrumentedValue], ...]:
    return tuple(sorted(values.items()))


def _state_problem(state: CartesianState2D | None) -> tuple[str, str] | None:
    if state is None:
        return ("not_evaluated", "cartesian_state_is_missing")
    for field_name in ("x", "y", "vx", "vy"):
        value = getattr(state, field_name)
        if not _is_finite_number(value):
            return ("invalid", f"cartesian_state_{field_name}_is_nonfinite_or_nonnumeric")
    return None


def _configuration_problem(
    configuration: OrbitalConfiguration | None,
) -> tuple[str, str] | None:
    if configuration is None:
        return ("not_evaluated", "orbital_configuration_is_missing")
    numeric = (
        ("mu", configuration.mu, True),
        ("target_radius", configuration.target_radius, True),
        ("ratio_denominator_epsilon", configuration.ratio_denominator_epsilon, False),
        (
            "speed_ratio_denominator_epsilon",
            configuration.speed_ratio_denominator_epsilon,
            False,
        ),
        (
            "specific_energy_radius_epsilon",
            configuration.specific_energy_radius_epsilon,
            False,
        ),
        ("action_component_limit", configuration.action_component_limit, True),
    )
    for name, value, positive in numeric:
        if not _is_finite_number(value):
            return ("invalid", f"configuration_{name}_is_nonfinite_or_nonnumeric")
        if positive and float(value) <= 0.0:
            return ("invalid", f"configuration_{name}_must_be_positive")
        if not positive and float(value) < 0.0:
            return ("invalid", f"configuration_{name}_must_be_nonnegative")
    if not isinstance(configuration.gravity_model_id, str) or not configuration.gravity_model_id:
        return ("invalid", "gravity_model_id_is_missing")
    return None


def _unavailable_basis(status: str, reason: str, source_step: int | None) -> OrbitalBasis2D:
    factory = invalid_value if status == "invalid" else not_evaluated_value
    return OrbitalBasis2D(
        radius=factory(
            reason=reason,
            units="m",
            source_id="derive_orbital_basis",
            source_step=source_step,
        ),
        radial_unit_vector=factory(
            reason=reason,
            units="dimensionless",
            source_id="derive_orbital_basis",
            source_step=source_step,
        ),
        tangential_unit_vector=factory(
            reason=reason,
            units="dimensionless",
            source_id="derive_orbital_basis",
            source_step=source_step,
        ),
        speed_magnitude=factory(
            reason=reason,
            units="m/s",
            source_id="derive_orbital_basis",
            source_step=source_step,
        ),
    )


def derive_orbital_basis(
    state: CartesianState2D | None,
    *,
    source_step: int | None = None,
) -> OrbitalBasis2D:
    problem = _state_problem(state)
    if problem:
        return _unavailable_basis(problem[0], problem[1], source_step)
    assert state is not None
    radius = math.hypot(float(state.x), float(state.y))
    speed = math.hypot(float(state.vx), float(state.vy))
    if not math.isfinite(radius) or radius <= POSITION_NORM_ZERO_TOLERANCE:
        return _unavailable_basis(
            "invalid", "position_norm_is_nonfinite_or_zero", source_step
        )
    radial_x = float(state.x) / radius
    radial_y = float(state.y) / radius
    tangential_x = -radial_y
    tangential_y = radial_x
    source_ids = ("state.position_x", "state.position_y")
    return OrbitalBasis2D(
        radius=derived_value(
            radius,
            units="m",
            source_id="derive_orbital_basis.radius",
            source_step=source_step,
            input_source_ids=source_ids,
        ),
        radial_unit_vector=derived_value(
            (radial_x, radial_y),
            units="dimensionless",
            source_id="derive_orbital_basis.radial_unit_vector",
            source_step=source_step,
            input_source_ids=source_ids,
        ),
        tangential_unit_vector=derived_value(
            (tangential_x, tangential_y),
            units="dimensionless",
            source_id="derive_orbital_basis.tangential_unit_vector",
            source_step=source_step,
            input_source_ids=("orbital_basis.radial_unit_vector",),
        ),
        speed_magnitude=derived_value(
            speed,
            units="m/s",
            source_id="derive_orbital_basis.speed_magnitude",
            source_step=source_step,
            input_source_ids=("state.velocity_x", "state.velocity_y"),
        ),
    )


def _derived_unavailable_fields(
    field_units: Mapping[str, str],
    *,
    status: str,
    reason: str,
    source_id: str,
    source_step: int | None,
) -> OrbitalDerivedState:
    factory = invalid_value if status == "invalid" else not_evaluated_value
    return OrbitalDerivedState(
        tuple(
            sorted(
                (
                    field_id,
                    factory(
                        reason=reason,
                        units=units,
                        source_id=source_id,
                        source_step=source_step,
                    ),
                )
                for field_id, units in field_units.items()
            )
        )
    )


_ORBITAL_DERIVED_UNITS = {
    "radius": "m",
    "radial_unit_vector": "dimensionless",
    "tangential_unit_vector": "dimensionless",
    "speed_magnitude": "m/s",
    "radial_velocity": "m/s",
    "tangential_velocity": "m/s",
    "target_radius": "m",
    "target_radius_error": "m",
    "signed_target_radius_error": "m",
    "absolute_target_radius_error": "m",
    "radius_error_ratio": "dimensionless",
    "target_circular_speed": "m/s",
    "radial_velocity_ratio": "dimensionless",
    "tangential_velocity_error": "m/s",
    "tangential_velocity_error_ratio": "dimensionless",
    "realized_speed_ratio": "dimensionless",
    "overspeed_headroom": "dimensionless",
    "overspeed_status": "boolean",
    "specific_orbital_energy": "J/kg",
    "target_circular_specific_energy": "J/kg",
    "specific_energy_error": "J/kg",
    "orbital_energy_or_proxy": "J/kg",
    "energy_model_status": "categorical",
}


def derive_orbital_state(
    state: CartesianState2D | None,
    configuration: OrbitalConfiguration | None,
    *,
    source_step: int | None = None,
) -> OrbitalDerivedState:
    state_problem = _state_problem(state)
    if state_problem:
        return _derived_unavailable_fields(
            _ORBITAL_DERIVED_UNITS,
            status=state_problem[0],
            reason=state_problem[1],
            source_id="derive_orbital_state",
            source_step=source_step,
        )
    configuration_problem = _configuration_problem(configuration)
    if configuration_problem:
        return _derived_unavailable_fields(
            _ORBITAL_DERIVED_UNITS,
            status=configuration_problem[0],
            reason=configuration_problem[1],
            source_id="derive_orbital_state",
            source_step=source_step,
        )
    assert state is not None and configuration is not None
    basis = derive_orbital_basis(state, source_step=source_step)
    if not basis.radius.available:
        return _derived_unavailable_fields(
            _ORBITAL_DERIVED_UNITS,
            status="invalid",
            reason=basis.radius.reason,
            source_id="derive_orbital_state",
            source_step=source_step,
        )

    radius = float(basis.radius.value)
    speed = float(basis.speed_magnitude.value)
    radial_x, radial_y = basis.radial_unit_vector.value
    tangential_x, tangential_y = basis.tangential_unit_vector.value
    radial_velocity = float(state.vx) * radial_x + float(state.vy) * radial_y
    tangential_velocity = (
        float(state.vx) * tangential_x + float(state.vy) * tangential_y
    )
    target_radius = float(configuration.target_radius)
    mu = float(configuration.mu)
    target_circular_speed = math.sqrt(mu / target_radius)
    ratio_denominator = target_circular_speed + float(
        configuration.ratio_denominator_epsilon
    )
    speed_denominator = target_circular_speed + float(
        configuration.speed_ratio_denominator_epsilon
    )
    if ratio_denominator <= 0.0 or speed_denominator <= 0.0:
        return _derived_unavailable_fields(
            _ORBITAL_DERIVED_UNITS,
            status="invalid",
            reason="target_speed_denominator_is_nonpositive",
            source_id="derive_orbital_state",
            source_step=source_step,
        )
    signed_radius_error = radius - target_radius
    radius_error_ratio = signed_radius_error / target_radius
    tangential_error = tangential_velocity - target_circular_speed
    radial_velocity_ratio = radial_velocity / ratio_denominator
    tangential_error_ratio = tangential_error / ratio_denominator
    speed_ratio = speed / speed_denominator
    overspeed_headroom = OVERSPEED_THRESHOLD - speed_ratio
    overspeed = speed_ratio > OVERSPEED_THRESHOLD

    common_inputs = ("state.position", "state.velocity", "orbital_configuration")
    values: dict[str, InstrumentedValue] = {
        "radius": basis.radius,
        "radial_unit_vector": basis.radial_unit_vector,
        "tangential_unit_vector": basis.tangential_unit_vector,
        "speed_magnitude": basis.speed_magnitude,
        "radial_velocity": derived_value(
            radial_velocity,
            units="m/s",
            source_id="derive_velocity_decomposition.radial_velocity",
            source_step=source_step,
            input_source_ids=("state.velocity", "orbital_basis.radial_unit_vector"),
        ),
        "tangential_velocity": derived_value(
            tangential_velocity,
            units="m/s",
            source_id="derive_velocity_decomposition.tangential_velocity",
            source_step=source_step,
            input_source_ids=(
                "state.velocity",
                "orbital_basis.tangential_unit_vector",
            ),
        ),
        "target_radius": measured_value(
            target_radius,
            units="m",
            source_id="orbital_configuration.target_radius",
            source_step=source_step,
        ),
        "target_radius_error": derived_value(
            signed_radius_error,
            units="m",
            source_id="derive_target_state.signed_target_radius_error",
            source_step=source_step,
            input_source_ids=("orbital.radius", "configuration.target_radius"),
        ),
        "signed_target_radius_error": derived_value(
            signed_radius_error,
            units="m",
            source_id="derive_target_state.signed_target_radius_error",
            source_step=source_step,
            input_source_ids=("orbital.radius", "configuration.target_radius"),
        ),
        "absolute_target_radius_error": derived_value(
            abs(signed_radius_error),
            units="m",
            source_id="derive_target_state.absolute_target_radius_error",
            source_step=source_step,
            input_source_ids=("target.signed_target_radius_error",),
        ),
        "radius_error_ratio": derived_value(
            radius_error_ratio,
            units="dimensionless",
            source_id="derive_target_state.radius_error_ratio",
            source_step=source_step,
            input_source_ids=("target.signed_target_radius_error", "configuration.target_radius"),
        ),
        "target_circular_speed": derived_value(
            target_circular_speed,
            units="m/s",
            source_id="derive_target_state.target_circular_speed",
            source_step=source_step,
            input_source_ids=("configuration.mu", "configuration.target_radius"),
        ),
        "radial_velocity_ratio": derived_value(
            radial_velocity_ratio,
            units="dimensionless",
            source_id="derive_target_state.radial_velocity_ratio",
            source_step=source_step,
            input_source_ids=("orbital.radial_velocity", "target.target_circular_speed"),
        ),
        "tangential_velocity_error": derived_value(
            tangential_error,
            units="m/s",
            source_id="derive_target_state.tangential_velocity_error",
            source_step=source_step,
            input_source_ids=("orbital.tangential_velocity", "target.target_circular_speed"),
        ),
        "tangential_velocity_error_ratio": derived_value(
            tangential_error_ratio,
            units="dimensionless",
            source_id="derive_target_state.tangential_velocity_error_ratio",
            source_step=source_step,
            input_source_ids=("target.tangential_velocity_error", "target.target_circular_speed"),
        ),
        "realized_speed_ratio": derived_value(
            speed_ratio,
            units="dimensionless",
            source_id="derive_speed_ratio.realized",
            source_step=source_step,
            input_source_ids=("orbital.speed_magnitude", "target.target_circular_speed"),
        ),
        "overspeed_headroom": derived_value(
            overspeed_headroom,
            units="dimensionless",
            source_id="derive_overspeed_headroom.realized",
            source_step=source_step,
            input_source_ids=("hazard.realized_speed_ratio",),
        ),
        "overspeed_status": derived_value(
            overspeed,
            units="boolean",
            source_id="derive_overspeed_status.realized",
            source_step=source_step,
            reason="strict_speed_ratio_greater_than_1.90",
            input_source_ids=("hazard.realized_speed_ratio",),
        ),
    }

    if configuration.gravity_model_id == CURRENT_GRAVITY_MODEL_ID:
        energy = 0.5 * speed * speed - mu / (
            radius + float(configuration.specific_energy_radius_epsilon)
        )
        target_energy = -mu / (2.0 * target_radius)
        energy_error = energy - target_energy
        values.update(
            {
                "specific_orbital_energy": derived_value(
                    energy,
                    units="J/kg",
                    source_id="derive_specific_orbital_energy.declared_proxy",
                    source_step=source_step,
                    reason="phase21_declared_specific_energy_diagnostic_proxy",
                    input_source_ids=common_inputs,
                ),
                "target_circular_specific_energy": derived_value(
                    target_energy,
                    units="J/kg",
                    source_id="derive_specific_orbital_energy.target_circular",
                    source_step=source_step,
                    input_source_ids=("configuration.mu", "configuration.target_radius"),
                ),
                "specific_energy_error": derived_value(
                    energy_error,
                    units="J/kg",
                    source_id="derive_specific_orbital_energy.error",
                    source_step=source_step,
                    input_source_ids=(
                        "energy.specific_orbital_energy",
                        "energy.target_circular_specific_energy",
                    ),
                ),
                "orbital_energy_or_proxy": derived_value(
                    energy,
                    units="J/kg",
                    source_id="derive_specific_orbital_energy.declared_proxy",
                    source_step=source_step,
                    reason="declared_proxy_not_exact_softened_model_invariant",
                    input_source_ids=common_inputs,
                ),
                "energy_model_status": derived_value(
                    SPECIFIC_ENERGY_MODEL_STATUS,
                    units="categorical",
                    source_id="derive_specific_orbital_energy.model_status",
                    source_step=source_step,
                    input_source_ids=("configuration.gravity_model_id",),
                ),
            }
        )
    else:
        for field_id, units in (
            ("specific_orbital_energy", "J/kg"),
            ("target_circular_specific_energy", "J/kg"),
            ("specific_energy_error", "J/kg"),
            ("orbital_energy_or_proxy", "J/kg"),
            ("energy_model_status", "categorical"),
        ):
            values[field_id] = not_evaluated_value(
                reason="gravity_model_is_not_supported_by_declared_energy_proxy",
                units=units,
                source_id="derive_specific_orbital_energy",
                source_step=source_step,
                input_source_ids=("configuration.gravity_model_id",),
            )
    return OrbitalDerivedState(tuple(sorted(values.items())))


def derive_predicted_hazard_state(
    predicted_state: CartesianState2D | None,
    configuration: OrbitalConfiguration | None,
    *,
    source_step: int | None = None,
    horizon_steps: int = 1,
) -> tuple[tuple[str, InstrumentedValue], ...]:
    if predicted_state is None:
        return _pairs(
            predicted_speed_ratio=not_evaluated_value(
                reason="predicted_state_is_missing",
                units="dimensionless",
                source_id="derive_predicted_hazard_state",
                source_step=source_step,
            ),
            predicted_overspeed_headroom=not_evaluated_value(
                reason="predicted_state_is_missing",
                units="dimensionless",
                source_id="derive_predicted_hazard_state",
                source_step=source_step,
            ),
            predicted_overspeed_status=not_evaluated_value(
                reason="predicted_state_is_missing",
                units="boolean",
                source_id="derive_predicted_hazard_state",
                source_step=source_step,
            ),
        )
    derived = derive_orbital_state(
        predicted_state, configuration, source_step=source_step
    )
    ratio = derived.field("realized_speed_ratio")
    if not ratio.available:
        factory = invalid_value if ratio.status == InstrumentationEvidenceStatus.INVALID else not_evaluated_value
        return _pairs(
            predicted_speed_ratio=factory(
                reason=ratio.reason,
                units="dimensionless",
                source_id="derive_predicted_hazard_state",
                source_step=source_step,
            ),
            predicted_overspeed_headroom=factory(
                reason=ratio.reason,
                units="dimensionless",
                source_id="derive_predicted_hazard_state",
                source_step=source_step,
            ),
            predicted_overspeed_status=factory(
                reason=ratio.reason,
                units="boolean",
                source_id="derive_predicted_hazard_state",
                source_step=source_step,
            ),
        )
    speed_ratio = float(ratio.value)
    return _pairs(
        predicted_speed_ratio=predicted_value(
            speed_ratio,
            units="dimensionless",
            source_id="derive_predicted_hazard_state.speed_ratio",
            source_step=source_step,
            horizon_steps=horizon_steps,
            input_source_ids=("predicted_state.velocity", "target.target_circular_speed"),
        ),
        predicted_overspeed_headroom=predicted_value(
            OVERSPEED_THRESHOLD - speed_ratio,
            units="dimensionless",
            source_id="derive_predicted_hazard_state.headroom",
            source_step=source_step,
            horizon_steps=horizon_steps,
            input_source_ids=("hazard.predicted_speed_ratio",),
        ),
        predicted_overspeed_status=predicted_value(
            speed_ratio > OVERSPEED_THRESHOLD,
            units="boolean",
            source_id="derive_predicted_hazard_state.overspeed_status",
            source_step=source_step,
            horizon_steps=horizon_steps,
            input_source_ids=("hazard.predicted_speed_ratio",),
        ),
    )


def _coerce_component(
    value: InstrumentedValue | float | int | None,
    *,
    field_id: str,
    source_step: int | None,
) -> InstrumentedValue:
    if isinstance(value, InstrumentedValue):
        return value
    if value is None:
        return not_evaluated_value(
            reason=f"{field_id}_is_missing",
            units="dimensionless",
            source_id="derive_phase34_recoverability",
            source_step=source_step,
        )
    if not _is_finite_number(value):
        return invalid_value(
            reason=f"{field_id}_is_nonfinite_or_nonnumeric",
            units="dimensionless",
            source_id="derive_phase34_recoverability",
            source_step=source_step,
        )
    return measured_value(
        float(value),
        units="dimensionless",
        source_id=f"explicit.{field_id}",
        source_step=source_step,
    )


def derive_phase34_recoverability(
    radius_error_ratio: InstrumentedValue | float | int | None,
    radial_velocity_ratio: InstrumentedValue | float | int | None,
    tangential_velocity_error_ratio: InstrumentedValue | float | int | None,
    *,
    source_step: int | None = None,
) -> RecoverabilityComponents:
    components = {
        "radius_error_ratio": _coerce_component(
            radius_error_ratio,
            field_id="radius_error_ratio",
            source_step=source_step,
        ),
        "radial_velocity_ratio": _coerce_component(
            radial_velocity_ratio,
            field_id="radial_velocity_ratio",
            source_step=source_step,
        ),
        "tangential_velocity_error_ratio": _coerce_component(
            tangential_velocity_error_ratio,
            field_id="tangential_velocity_error_ratio",
            source_step=source_step,
        ),
    }
    output_units = {
        "radius_component_pass": "boolean",
        "radial_velocity_component_pass": "boolean",
        "tangential_velocity_component_pass": "boolean",
        "phase34_compatible_recoverability": "boolean",
        "recoverability_component_vector": "dimensionless",
    }
    invalid_components = [
        field_id
        for field_id, value in components.items()
        if value.status == InstrumentationEvidenceStatus.INVALID
    ]
    missing_components = [
        field_id
        for field_id, value in components.items()
        if value.status == InstrumentationEvidenceStatus.NOT_EVALUATED
    ]
    values = dict(components)
    if invalid_components or missing_components:
        factory = invalid_value if invalid_components else not_evaluated_value
        reason = (
            "recoverability_component_is_invalid:"
            + ",".join(sorted(invalid_components))
            if invalid_components
            else "recoverability_component_is_missing:"
            + ",".join(sorted(missing_components))
        )
        for field_id, units in output_units.items():
            values[field_id] = factory(
                reason=reason,
                units=units,
                source_id="derive_phase34_recoverability",
                source_step=source_step,
                input_source_ids=tuple(sorted(components)),
            )
        return RecoverabilityComponents(tuple(sorted(values.items())))

    r_value = float(components["radius_error_ratio"].value)
    vr_value = float(components["radial_velocity_ratio"].value)
    vt_value = float(components["tangential_velocity_error_ratio"].value)
    radius_pass = abs(r_value) <= PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX
    radial_pass = abs(vr_value) <= PHASE34_RECOVERABLE_VR_RATIO_MAX
    tangential_pass = abs(vt_value) <= PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX
    existing = evaluate_phase34_compatible_recoverability(
        r_error_ratio=r_value,
        vr_ratio=vr_value,
        vt_error_ratio=vt_value,
        evaluated_step=source_step,
        evidence_level="derived",
    )
    combined = existing.status == EVALUATION_TRIGGERED
    if existing.status not in {EVALUATION_TRIGGERED, EVALUATION_CLEAR}:
        factory = invalid_value if existing.status == EVALUATION_INVALID else not_evaluated_value
        for field_id, units in output_units.items():
            values[field_id] = factory(
                reason=f"existing_recovery_evaluator_returned_{existing.status}",
                units=units,
                source_id="derive_phase34_recoverability",
                source_step=source_step,
            )
        return RecoverabilityComponents(tuple(sorted(values.items())))
    for field_id, value in (
        ("radius_component_pass", radius_pass),
        ("radial_velocity_component_pass", radial_pass),
        ("tangential_velocity_component_pass", tangential_pass),
        ("phase34_compatible_recoverability", combined),
    ):
        values[field_id] = derived_value(
            value,
            units="boolean",
            source_id=f"derive_phase34_recoverability.{field_id}",
            source_step=source_step,
            reason="inclusive_absolute_phase34_component_bounds",
            input_source_ids=tuple(sorted(components)),
        )
    values["recoverability_component_vector"] = derived_value(
        (
            ("radius_error_ratio", r_value),
            ("radial_velocity_ratio", vr_value),
            ("tangential_velocity_error_ratio", vt_value),
        ),
        units="dimensionless",
        source_id="derive_phase34_recoverability.component_vector",
        source_step=source_step,
        input_source_ids=tuple(sorted(components)),
    )
    return RecoverabilityComponents(tuple(sorted(values.items())))


def derive_crossing_event(
    previous_state: CartesianState2D | None,
    current_state: CartesianState2D | None,
    configuration: OrbitalConfiguration | None,
    *,
    previous_step: int | None,
    current_step: int | None,
    branch_step: int | None,
) -> CrossingEvent:
    units = {
        "target_radius_crossing": "boolean",
        "crossing_direction": "categorical",
        "crossing_recovery_eligible": "boolean",
        "first_crossing_step": "step",
        "crossing_interpolation_fraction": "dimensionless",
    }
    integer_values = (previous_step, current_step, branch_step)
    if any(
        value is not None
        and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
        for value in integer_values
    ):
        safe_source_step = (
            current_step
            if not isinstance(current_step, bool)
            and isinstance(current_step, int)
            and current_step >= 0
            else None
        )
        return CrossingEvent(
            tuple(
                sorted(
                    (
                        field_id,
                        invalid_value(
                            reason="crossing_step_is_malformed",
                            units=field_units,
                            source_id="derive_crossing_event",
                            source_step=safe_source_step,
                        ),
                    )
                    for field_id, field_units in units.items()
                )
            )
        )
    if previous_state is None or current_state is None or configuration is None:
        return CrossingEvent(
            tuple(
                sorted(
                    (
                        field_id,
                        not_evaluated_value(
                            reason="previous_current_state_or_configuration_is_missing",
                            units=field_units,
                            source_id="derive_crossing_event",
                            source_step=current_step,
                        ),
                    )
                    for field_id, field_units in units.items()
                )
            )
        )
    previous = derive_orbital_state(previous_state, configuration, source_step=previous_step)
    current = derive_orbital_state(current_state, configuration, source_step=current_step)
    previous_error = previous.field("signed_target_radius_error")
    current_error = current.field("signed_target_radius_error")
    if not previous_error.available or not current_error.available:
        invalid = any(
            value.status == InstrumentationEvidenceStatus.INVALID
            for value in (previous_error, current_error)
        )
        factory = invalid_value if invalid else not_evaluated_value
        return CrossingEvent(
            tuple(
                sorted(
                    (
                        field_id,
                        factory(
                            reason="crossing_radius_error_evidence_is_unavailable",
                            units=field_units,
                            source_id="derive_crossing_event",
                            source_step=current_step,
                        ),
                    )
                    for field_id, field_units in units.items()
                )
            )
        )
    previous_value = float(previous_error.value)
    current_value = float(current_error.value)
    outside_to_inside = previous_value > 0.0 and current_value <= 0.0
    inside_to_outside = previous_value < 0.0 and current_value >= 0.0
    crossed = outside_to_inside or inside_to_outside
    direction = (
        "above_to_below"
        if outside_to_inside
        else "below_to_above"
        if inside_to_outside
        else "none"
    )
    recovery_eligible = bool(
        crossed
        and current_step is not None
        and branch_step is not None
        and current_step >= branch_step
    )
    return CrossingEvent(
        _pairs(
            target_radius_crossing=derived_value(
                crossed,
                units="boolean",
                source_id="derive_crossing_event.target_radius_crossing",
                source_step=current_step,
                reason="phase34_signed_radius_error_transition_rule",
                input_source_ids=("previous.signed_radius_error", "current.signed_radius_error"),
            ),
            crossing_direction=derived_value(
                direction,
                units="categorical",
                source_id="derive_crossing_event.crossing_direction",
                source_step=current_step,
                input_source_ids=("previous.signed_radius_error", "current.signed_radius_error"),
            ),
            crossing_recovery_eligible=derived_value(
                recovery_eligible,
                units="boolean",
                source_id="derive_crossing_event.recovery_eligibility",
                source_step=current_step,
                reason="crossing_step_must_be_at_or_after_branch_step",
                input_source_ids=("crossing.target_radius_crossing", "branch_step"),
            ),
            first_crossing_step=(
                derived_value(
                    current_step,
                    units="step",
                    source_id="derive_crossing_event.first_crossing_step",
                    source_step=current_step,
                    input_source_ids=("crossing.target_radius_crossing",),
                )
                if crossed and current_step is not None
                else not_evaluated_value(
                    reason="no_crossing_at_this_state_pair",
                    units="step",
                    source_id="derive_crossing_event.first_crossing_step",
                    source_step=current_step,
                )
            ),
            crossing_interpolation_fraction=not_evaluated_value(
                reason="fractional_crossing_interpolation_is_not_implemented",
                units="dimensionless",
                source_id="derive_crossing_event",
                source_step=current_step,
            ),
        )
    )


_PROGRESS_COMPONENTS = (
    ("delta_signed_target_radius_error", "signed_target_radius_error", "m"),
    ("delta_absolute_target_radius_error", "absolute_target_radius_error", "m"),
    ("delta_radial_velocity", "radial_velocity", "m/s"),
    ("delta_tangential_velocity_error", "tangential_velocity_error", "m/s"),
    ("delta_realized_speed_ratio", "realized_speed_ratio", "dimensionless"),
    ("delta_overspeed_headroom", "overspeed_headroom", "dimensionless"),
    ("delta_specific_orbital_energy", "specific_orbital_energy", "J/kg"),
    ("delta_radius_error_ratio", "radius_error_ratio", "dimensionless"),
    ("delta_radial_velocity_ratio", "radial_velocity_ratio", "dimensionless"),
    (
        "delta_tangential_velocity_error_ratio",
        "tangential_velocity_error_ratio",
        "dimensionless",
    ),
)


def derive_progress_sample(
    previous: OrbitalDerivedState | None,
    current: OrbitalDerivedState | None,
    *,
    previous_transition_count: int | None = None,
    current_transition_count: int | None = None,
    previous_time: float | None = None,
    current_time: float | None = None,
    source_step: int | None = None,
) -> RecoveryProgressSample:
    values: dict[str, InstrumentedValue] = {}
    if previous is None or current is None:
        for output_id, _, units in _PROGRESS_COMPONENTS:
            values[output_id] = not_evaluated_value(
                reason="previous_or_current_orbital_sample_is_missing",
                units=units,
                source_id="derive_progress_sample",
                source_step=source_step,
            )
    else:
        for output_id, input_id, units in _PROGRESS_COMPONENTS:
            previous_value = previous.field(input_id)
            current_value = current.field(input_id)
            if (
                previous_value.status == InstrumentationEvidenceStatus.INVALID
                or current_value.status == InstrumentationEvidenceStatus.INVALID
            ):
                values[output_id] = invalid_value(
                    reason=f"{input_id}_sample_is_invalid",
                    units=units,
                    source_id="derive_progress_sample",
                    source_step=source_step,
                )
            elif not previous_value.available or not current_value.available:
                values[output_id] = not_evaluated_value(
                    reason=f"{input_id}_sample_is_missing",
                    units=units,
                    source_id="derive_progress_sample",
                    source_step=source_step,
                )
            elif not _is_finite_number(previous_value.value) or not _is_finite_number(
                current_value.value
            ):
                values[output_id] = invalid_value(
                    reason=f"{input_id}_sample_is_nonnumeric",
                    units=units,
                    source_id="derive_progress_sample",
                    source_step=source_step,
                )
            else:
                values[output_id] = derived_value(
                    float(current_value.value) - float(previous_value.value),
                    units=units,
                    source_id=f"derive_progress_sample.{output_id}",
                    source_step=source_step,
                    reason="current_minus_previous_raw_delta",
                    input_source_ids=(f"previous.{input_id}", f"current.{input_id}"),
                )

    values["transition_count_delta"] = _derive_scalar_delta(
        previous_transition_count,
        current_transition_count,
        units="transition",
        source_id="derive_progress_sample.transition_count_delta",
        source_step=source_step,
        integer_only=True,
    )
    values["elapsed_time_delta"] = _derive_scalar_delta(
        previous_time,
        current_time,
        units="s",
        source_id="derive_progress_sample.elapsed_time_delta",
        source_step=source_step,
        integer_only=False,
    )
    values["progress_classification"] = not_evaluated_value(
        reason="threshold_free_sample_does_not_classify_progress_stall_or_regression",
        units="categorical",
        source_id="derive_progress_sample",
        source_step=source_step,
    )
    return RecoveryProgressSample(tuple(sorted(values.items())))


def _derive_scalar_delta(
    previous: object,
    current: object,
    *,
    units: str,
    source_id: str,
    source_step: int | None,
    integer_only: bool,
) -> InstrumentedValue:
    if previous is None or current is None:
        return not_evaluated_value(
            reason="previous_or_current_value_is_missing",
            units=units,
            source_id=source_id,
            source_step=source_step,
        )
    valid = (
        not isinstance(previous, bool)
        and not isinstance(current, bool)
        and (
            isinstance(previous, int) and isinstance(current, int)
            if integer_only
            else _is_finite_number(previous) and _is_finite_number(current)
        )
    )
    if not valid:
        return invalid_value(
            reason="delta_input_is_malformed_or_nonfinite",
            units=units,
            source_id=source_id,
            source_step=source_step,
        )
    return derived_value(
        current - previous,
        units=units,
        source_id=source_id,
        source_step=source_step,
        reason="current_minus_previous_raw_delta",
    )


def _action_problem(action: object) -> str | None:
    if not isinstance(action, tuple) or len(action) != 2:
        return "action_must_be_a_two_component_tuple"
    if not all(_is_finite_number(value) for value in action):
        return "action_components_must_be_finite_numbers"
    return None


def derive_action_geometry(
    proposed_action: tuple[float, float] | None,
    executed_action: tuple[float, float] | None,
    basis: OrbitalBasis2D | None,
    *,
    action_component_limit: float = NORMALIZED_ACTION_COMPONENT_LIMIT,
    action_rejected: bool = False,
    explicit_abort: bool = False,
    source_step: int | None = None,
) -> ActionGeometry:
    output_units = {
        "proposed_action": "normalized_action",
        "executed_action": "normalized_action",
        "proposed_action_magnitude": "normalized_action",
        "executed_action_magnitude": "normalized_action",
        "proposed_action_radial_component": "normalized_action",
        "proposed_action_tangential_component": "normalized_action",
        "executed_action_radial_component": "normalized_action",
        "executed_action_tangential_component": "normalized_action",
        "action_saturation_margin": "normalized_action_component",
        "proposed_equals_executed": "boolean",
        "action_suppression_status": "boolean",
        "action_geometry_status": "categorical",
    }
    if type(action_rejected) is not bool or type(explicit_abort) is not bool:
        return ActionGeometry(
            tuple(
                sorted(
                    (
                        field_id,
                        invalid_value(
                            reason="action_rejected_and_explicit_abort_must_be_boolean",
                            units=units,
                            source_id="derive_action_geometry",
                            source_step=source_step,
                        ),
                    )
                    for field_id, units in output_units.items()
                )
            )
        )
    if not _is_finite_number(action_component_limit) or action_component_limit <= 0.0:
        return ActionGeometry(
            tuple(
                sorted(
                    (
                        field_id,
                        invalid_value(
                            reason="action_component_limit_must_be_positive_and_finite",
                            units=units,
                            source_id="derive_action_geometry",
                            source_step=source_step,
                        ),
                    )
                    for field_id, units in output_units.items()
                )
            )
        )
    if explicit_abort:
        return ActionGeometry(
            tuple(
                sorted(
                    (
                        field_id,
                        not_evaluated_value(
                            reason="explicit_abort_has_no_physical_action",
                            units=units,
                            source_id="derive_action_geometry",
                            source_step=source_step,
                        ),
                    )
                    for field_id, units in output_units.items()
                )
            )
        )

    proposed_problem = (
        "proposed_action_is_missing"
        if proposed_action is None
        else _action_problem(proposed_action)
    )
    executed_problem = (
        "executed_action_is_missing"
        if executed_action is None
        else _action_problem(executed_action)
    )
    values: dict[str, InstrumentedValue] = {}
    if proposed_problem:
        factory = not_evaluated_value if proposed_action is None else invalid_value
        for field_id in (
            "proposed_action",
            "proposed_action_magnitude",
            "proposed_action_radial_component",
            "proposed_action_tangential_component",
            "action_saturation_margin",
        ):
            values[field_id] = factory(
                reason=proposed_problem,
                units=output_units[field_id],
                source_id="derive_action_geometry",
                source_step=source_step,
            )
    else:
        assert proposed_action is not None
        proposed = (float(proposed_action[0]), float(proposed_action[1]))
        values["proposed_action"] = measured_value(
            proposed,
            units="normalized_action",
            source_id="explicit.proposed_action",
            source_step=source_step,
        )
        values["proposed_action_magnitude"] = derived_value(
            math.hypot(*proposed),
            units="normalized_action",
            source_id="derive_action_geometry.proposed_magnitude",
            source_step=source_step,
            input_source_ids=("action.proposed",),
        )
        values["action_saturation_margin"] = derived_value(
            float(action_component_limit) - max(abs(proposed[0]), abs(proposed[1])),
            units="normalized_action_component",
            source_id="derive_action_geometry.saturation_margin",
            source_step=source_step,
            reason="component_limit_minus_maximum_absolute_proposed_component",
            input_source_ids=("action.proposed", "configuration.action_component_limit"),
        )
        _add_action_basis_components(
            values,
            prefix="proposed",
            action=proposed,
            basis=basis,
            source_step=source_step,
        )

    if executed_problem:
        factory = not_evaluated_value if executed_action is None else invalid_value
        reason = (
            "rejected_action_has_no_executed_action"
            if action_rejected and executed_action is None
            else executed_problem
        )
        for field_id in (
            "executed_action",
            "executed_action_magnitude",
            "executed_action_radial_component",
            "executed_action_tangential_component",
        ):
            values[field_id] = factory(
                reason=reason,
                units=output_units[field_id],
                source_id="derive_action_geometry",
                source_step=source_step,
            )
    else:
        assert executed_action is not None
        executed = (float(executed_action[0]), float(executed_action[1]))
        values["executed_action"] = measured_value(
            executed,
            units="normalized_action",
            source_id="explicit.executed_action",
            source_step=source_step,
        )
        values["executed_action_magnitude"] = derived_value(
            math.hypot(*executed),
            units="normalized_action",
            source_id="derive_action_geometry.executed_magnitude",
            source_step=source_step,
            input_source_ids=("action.executed",),
        )
        _add_action_basis_components(
            values,
            prefix="executed",
            action=executed,
            basis=basis,
            source_step=source_step,
        )

    if proposed_problem or executed_problem:
        reason = (
            "action_rejected_is_distinct_from_zero_action"
            if action_rejected
            else "proposed_or_executed_action_evidence_is_incomplete"
        )
        values["proposed_equals_executed"] = not_evaluated_value(
            reason=reason,
            units="boolean",
            source_id="derive_action_geometry",
            source_step=source_step,
        )
        values["action_suppression_status"] = not_evaluated_value(
            reason=reason,
            units="boolean",
            source_id="derive_action_geometry",
            source_step=source_step,
        )
        values["action_geometry_status"] = derived_value(
            "rejected" if action_rejected else "incomplete",
            units="categorical",
            source_id="derive_action_geometry.status",
            source_step=source_step,
        )
    else:
        assert proposed_action is not None and executed_action is not None
        equal = tuple(float(value) for value in proposed_action) == tuple(
            float(value) for value in executed_action
        )
        values["proposed_equals_executed"] = derived_value(
            equal,
            units="boolean",
            source_id="derive_action_geometry.equality",
            source_step=source_step,
            input_source_ids=("action.proposed", "action.executed"),
        )
        values["action_suppression_status"] = derived_value(
            not equal,
            units="boolean",
            source_id="derive_action_geometry.suppression",
            source_step=source_step,
            reason="proposed_and_executed_action_vector_inequality",
            input_source_ids=("action.proposed", "action.executed"),
        )
        values["action_geometry_status"] = derived_value(
            "complete",
            units="categorical",
            source_id="derive_action_geometry.status",
            source_step=source_step,
        )
    return ActionGeometry(tuple(sorted(values.items())))


def _add_action_basis_components(
    values: dict[str, InstrumentedValue],
    *,
    prefix: str,
    action: tuple[float, float],
    basis: OrbitalBasis2D | None,
    source_step: int | None,
) -> None:
    radial_id = f"{prefix}_action_radial_component"
    tangential_id = f"{prefix}_action_tangential_component"
    if basis is None or not (
        basis.radial_unit_vector.available and basis.tangential_unit_vector.available
    ):
        values[radial_id] = not_evaluated_value(
            reason="valid_orbital_basis_is_required_for_action_decomposition",
            units="normalized_action",
            source_id="derive_action_geometry",
            source_step=source_step,
        )
        values[tangential_id] = not_evaluated_value(
            reason="valid_orbital_basis_is_required_for_action_decomposition",
            units="normalized_action",
            source_id="derive_action_geometry",
            source_step=source_step,
        )
        return
    radial_x, radial_y = basis.radial_unit_vector.value
    tangential_x, tangential_y = basis.tangential_unit_vector.value
    values[radial_id] = derived_value(
        action[0] * radial_x + action[1] * radial_y,
        units="normalized_action",
        source_id=f"derive_action_geometry.{radial_id}",
        source_step=source_step,
        input_source_ids=(f"action.{prefix}", "orbital_basis.radial_unit_vector"),
    )
    values[tangential_id] = derived_value(
        action[0] * tangential_x + action[1] * tangential_y,
        units="normalized_action",
        source_id=f"derive_action_geometry.{tangential_id}",
        source_step=source_step,
        input_source_ids=(
            f"action.{prefix}",
            "orbital_basis.tangential_unit_vector",
        ),
    )


_PROVENANCE_FIELD_IDS = (
    "case_id",
    "seed",
    "implementation_commit",
    "branch_state_hash",
    "simulator_configuration_hash",
    "constants_hash",
    "recovery_step",
    "total_transition_count",
    "simulation_time",
)

_EXTRA_FIELD_IDS = (
    "radial_unit_vector",
    "tangential_unit_vector",
    "speed_magnitude",
    "signed_target_radius_error",
    "absolute_target_radius_error",
    "radius_error_ratio",
    "tangential_velocity_error",
    "predicted_overspeed_headroom",
    "predicted_overspeed_status",
    "crossing_direction",
    "crossing_recovery_eligible",
    "crossing_interpolation_fraction",
    "radius_component_pass",
    "radial_velocity_component_pass",
    "tangential_velocity_component_pass",
    "specific_orbital_energy",
    "target_circular_specific_energy",
    "specific_energy_error",
    "energy_model_status",
    "delta_signed_target_radius_error",
    "delta_absolute_target_radius_error",
    "delta_radial_velocity",
    "delta_tangential_velocity_error",
    "delta_realized_speed_ratio",
    "delta_overspeed_headroom",
    "delta_specific_orbital_energy",
    "delta_radius_error_ratio",
    "delta_radial_velocity_ratio",
    "delta_tangential_velocity_error_ratio",
    "transition_count_delta",
    "elapsed_time_delta",
    "progress_classification",
    "proposed_action_magnitude",
    "executed_action_magnitude",
    "proposed_action_radial_component",
    "proposed_action_tangential_component",
    "executed_action_radial_component",
    "executed_action_tangential_component",
    "proposed_equals_executed",
    "action_suppression_status",
    "action_geometry_status",
    "retreat_status",
    "current_phase",
    "previous_phase",
)

CANONICAL_FIELD_ORDER = tuple(
    dict.fromkeys(_PROVENANCE_FIELD_IDS + ARCHITECTURE_SIGNAL_IDS + _EXTRA_FIELD_IDS)
)


_PURE_ARCHITECTURE_SIGNALS = frozenset(
    {
        "overspeed_status",
        "radius",
        "target_radius_error",
        "radial_velocity",
        "tangential_velocity",
        "target_circular_speed",
        "radial_velocity_ratio",
        "tangential_velocity_error_ratio",
        "realized_speed_ratio",
        "overspeed_headroom",
        "orbital_energy_or_proxy",
        "phase34_compatible_recoverability",
        "recoverability_component_vector",
        "action_saturation_margin",
    }
)
_PREVIOUS_STATE_SIGNALS = frozenset(
    {
        "target_radius_crossing",
        "first_crossing_step",
        "radial_progress",
        "radial_progress_direction",
        "radial_progress_rate",
        "tangential_progress",
        "crossing_progress",
        "energy_change_direction",
    }
)
_PREDICTED_STATE_SIGNALS = frozenset(
    {"predicted_speed_ratio", "predicted_crossing_direction", "crossing_proximity"}
)
_RUNTIME_SIGNALS = frozenset(
    {
        "recovery_horizon_remaining",
        "recovery_horizon_exhausted",
        "total_horizon_remaining",
        "total_horizon_exhausted",
        "no_progress_status",
        "phase_dwell_count",
        "phase_transition_count",
        "recent_phase_history",
        "phase_transition_reason",
    }
)
_FUTURE_EVALUATOR_SIGNALS = frozenset(
    {
        "handoff_readiness",
    }
)
_NOT_SUPPORTED_SIGNALS = frozenset({"available_correction_authority"})


def architecture_signal_coverage() -> tuple[tuple[str, str], ...]:
    coverage: list[tuple[str, str]] = []
    for signal_id in ARCHITECTURE_SIGNAL_IDS:
        if signal_id in _PURE_ARCHITECTURE_SIGNALS:
            classification = COVERAGE_PURE_DERIVATION
        elif signal_id in _PREVIOUS_STATE_SIGNALS:
            classification = COVERAGE_PREVIOUS_STATE
        elif signal_id in _PREDICTED_STATE_SIGNALS:
            classification = COVERAGE_PREDICTED_STATE
        elif signal_id in _RUNTIME_SIGNALS:
            classification = COVERAGE_RUNTIME_INTEGRATION
        elif signal_id in _FUTURE_EVALUATOR_SIGNALS:
            classification = COVERAGE_FUTURE_EVALUATOR
        elif signal_id in _NOT_SUPPORTED_SIGNALS:
            classification = COVERAGE_NOT_SUPPORTED
        else:
            classification = COVERAGE_DIRECT_INPUT
        coverage.append((signal_id, classification))
    return tuple(coverage)


_UNITS = {
    "position_x": "m",
    "position_y": "m",
    "velocity_x": "m/s",
    "velocity_y": "m/s",
    "radius": "m",
    "target_radius": "m",
    "target_radius_error": "m",
    "radial_velocity": "m/s",
    "tangential_velocity": "m/s",
    "target_circular_speed": "m/s",
    "orbital_energy_or_proxy": "J/kg",
    "simulation_time": "s",
    "seed": "integer",
    "recovery_step": "transition",
    "total_transition_count": "transition",
    "first_crossing_step": "transition",
    "recovery_horizon_remaining": "transition",
    "total_horizon_remaining": "transition",
    "phase_dwell_count": "transition",
    "phase_transition_count": "count",
    "transition_count_delta": "transition",
    "radial_unit_vector": "dimensionless",
    "tangential_unit_vector": "dimensionless",
    "speed_magnitude": "m/s",
    "signed_target_radius_error": "m",
    "absolute_target_radius_error": "m",
    "tangential_velocity_error": "m/s",
    "specific_orbital_energy": "J/kg",
    "target_circular_specific_energy": "J/kg",
    "specific_energy_error": "J/kg",
    "delta_signed_target_radius_error": "m",
    "delta_absolute_target_radius_error": "m",
    "delta_radial_velocity": "m/s",
    "delta_tangential_velocity_error": "m/s",
    "delta_specific_orbital_energy": "J/kg",
    "elapsed_time_delta": "s",
    "proposed_action": "normalized_action",
    "executed_action": "normalized_action",
    "proposed_action_magnitude": "normalized_action",
    "executed_action_magnitude": "normalized_action",
    "proposed_action_radial_component": "normalized_action",
    "proposed_action_tangential_component": "normalized_action",
    "executed_action_radial_component": "normalized_action",
    "executed_action_tangential_component": "normalized_action",
    "action_saturation_margin": "normalized_action_component",
}

_BOOLEAN_FIELDS = frozenset(
    {
        "simulation_validity",
        "recovery_evaluation_validity",
        "overspeed_status",
        "instability_status",
        "unsafe_state_status",
        "action_rejection_status",
        "explicit_abort_requested",
        "target_radius_crossing",
        "phase34_compatible_recoverability",
        "recovery_success_v0",
        "recovery_horizon_exhausted",
        "total_horizon_exhausted",
        "handoff_readiness",
        "simulator_success",
        "predicted_overspeed_status",
        "radius_component_pass",
        "radial_velocity_component_pass",
        "tangential_velocity_component_pass",
        "crossing_recovery_eligible",
        "proposed_equals_executed",
        "action_suppression_status",
        "retreat_status",
    }
)

_INTEGER_FIELDS = frozenset(
    {
        "seed",
        "recovery_step",
        "total_transition_count",
        "first_crossing_step",
        "recovery_horizon_remaining",
        "total_horizon_remaining",
        "phase_dwell_count",
        "phase_transition_count",
        "transition_count_delta",
    }
)

_VECTOR2_FIELDS = frozenset(
    {
        "radial_unit_vector",
        "tangential_unit_vector",
        "proposed_action",
        "executed_action",
    }
)


def _coverage_for_field(field_id: str) -> str:
    coverage = dict(architecture_signal_coverage())
    if field_id in coverage:
        return coverage[field_id]
    if field_id in _PROVENANCE_FIELD_IDS or field_id in {
        "current_phase",
        "previous_phase",
        "retreat_status",
    }:
        return COVERAGE_DIRECT_INPUT
    if field_id.startswith("delta_") or field_id in {
        "transition_count_delta",
        "elapsed_time_delta",
    }:
        return COVERAGE_PREVIOUS_STATE
    if field_id == "crossing_interpolation_fraction":
        return COVERAGE_NOT_SUPPORTED
    if field_id == "progress_classification":
        return COVERAGE_RUNTIME_INTEGRATION
    return COVERAGE_PURE_DERIVATION


def _category_for_field(field_id: str) -> str:
    if field_id in _PROVENANCE_FIELD_IDS:
        return "provenance"
    if field_id in {"position_x", "position_y", "velocity_x", "velocity_y"}:
        return "cartesian_state"
    if field_id in {
        "radius",
        "radial_unit_vector",
        "tangential_unit_vector",
        "speed_magnitude",
        "radial_velocity",
        "tangential_velocity",
    }:
        return "orbital_geometry"
    if field_id in {
        "target_radius",
        "target_radius_error",
        "signed_target_radius_error",
        "absolute_target_radius_error",
        "radius_error_ratio",
        "target_circular_speed",
        "radial_velocity_ratio",
        "tangential_velocity_error",
        "tangential_velocity_error_ratio",
    }:
        return "target_state"
    if "energy" in field_id:
        return "energy"
    if field_id.startswith("delta_") or field_id in {
        "transition_count_delta",
        "elapsed_time_delta",
        "progress_classification",
        "radial_progress",
        "radial_progress_direction",
        "radial_progress_rate",
        "tangential_progress",
        "crossing_progress",
    }:
        return "progress"
    if "action" in field_id or field_id in {
        "final_veto_decision",
        "proposed_equals_executed",
    }:
        return "action"
    if "crossing" in field_id or "recoverability" in field_id or field_id in {
        "recovery_success_v0",
        "simulator_success",
        "radius_component_pass",
        "radial_velocity_component_pass",
        "tangential_velocity_component_pass",
    }:
        return "recoverability"
    if field_id in {
        "realized_speed_ratio",
        "predicted_speed_ratio",
        "overspeed_headroom",
        "predicted_overspeed_headroom",
        "predicted_overspeed_status",
        "overspeed_status",
        "simulation_validity",
        "recovery_evaluation_validity",
        "instability_status",
        "unsafe_state_status",
    }:
        return "hazard_and_validity"
    return "phase_and_runtime"


def _required_inputs_for_field(field_id: str) -> tuple[str, ...]:
    mapping = {
        "radius": ("position_x", "position_y"),
        "radial_unit_vector": ("position_x", "position_y", "radius"),
        "tangential_unit_vector": ("radial_unit_vector",),
        "speed_magnitude": ("velocity_x", "velocity_y"),
        "radial_velocity": ("velocity_x", "velocity_y", "radial_unit_vector"),
        "tangential_velocity": (
            "velocity_x",
            "velocity_y",
            "tangential_unit_vector",
        ),
        "target_radius_error": ("radius", "target_radius"),
        "radius_error_ratio": ("target_radius_error", "target_radius"),
        "target_circular_speed": ("mu", "target_radius"),
        "radial_velocity_ratio": ("radial_velocity", "target_circular_speed"),
        "tangential_velocity_error_ratio": (
            "tangential_velocity",
            "target_circular_speed",
        ),
        "realized_speed_ratio": ("speed_magnitude", "target_circular_speed"),
        "predicted_speed_ratio": ("predicted_state", "target_circular_speed"),
        "overspeed_headroom": ("realized_speed_ratio",),
        "target_radius_crossing": ("previous_state", "current_state", "target_radius"),
        "phase34_compatible_recoverability": (
            "radius_error_ratio",
            "radial_velocity_ratio",
            "tangential_velocity_error_ratio",
        ),
        "orbital_energy_or_proxy": ("position", "velocity", "mu"),
        "action_saturation_margin": ("proposed_action", "action_component_limit"),
    }
    if field_id in mapping:
        return mapping[field_id]
    if field_id.startswith("delta_"):
        source = field_id.removeprefix("delta_")
        return (f"previous_{source}", f"current_{source}")
    return ()


def field_catalog() -> tuple[InstrumentationFieldDefinition, ...]:
    phases_by_signal: dict[str, list[str]] = {
        field_id: [] for field_id in CANONICAL_FIELD_ORDER
    }
    for phase in default_phase_contracts():
        for signal_id in phase.required_signal_ids:
            phases_by_signal.setdefault(signal_id, []).append(phase.phase_id.value)
    definitions = []
    statuses = tuple(status.value for status in InstrumentationEvidenceStatus)
    for index, field_id in enumerate(CANONICAL_FIELD_ORDER):
        coverage = _coverage_for_field(field_id)
        units = _UNITS.get(
            field_id,
            "boolean"
            if field_id in _BOOLEAN_FIELDS
            else "dimensionless"
            if "ratio" in field_id or "headroom" in field_id
            else "categorical",
        )
        data_type = (
            "boolean"
            if field_id in _BOOLEAN_FIELDS
            else "integer"
            if field_id in _INTEGER_FIELDS
            else "vector2"
            if field_id in _VECTOR2_FIELDS
            else "component_vector"
            if field_id == "recoverability_component_vector"
            else "string_list"
            if field_id == "recent_phase_history"
            else "number"
            if units not in {"categorical", "boolean"}
            else "string"
        )
        definitions.append(
            InstrumentationFieldDefinition(
                field_id=field_id,
                category=_category_for_field(field_id),
                description=f"Status-bearing staged-recovery field: {field_id.replace('_', ' ')}.",
                data_type=data_type,
                units=units,
                signedness=(
                    "signed"
                    if field_id.startswith("delta_")
                    or field_id
                    in {
                        "target_radius_error",
                        "signed_target_radius_error",
                        "radial_velocity",
                        "tangential_velocity",
                        "tangential_velocity_error",
                        "radial_velocity_ratio",
                        "tangential_velocity_error_ratio",
                        "overspeed_headroom",
                        "predicted_overspeed_headroom",
                        "specific_energy_error",
                        "orbital_energy_or_proxy",
                    }
                    else "not_applicable_or_nonnegative"
                ),
                evidence_status_requirements=statuses,
                support_classification=coverage,
                direct_or_derived=(
                    "direct"
                    if coverage == COVERAGE_DIRECT_INPUT
                    else "derived"
                    if coverage
                    in {
                        COVERAGE_PURE_DERIVATION,
                        COVERAGE_PREVIOUS_STATE,
                        COVERAGE_PREDICTED_STATE,
                    }
                    else "external_or_unresolved"
                ),
                required_inputs=_required_inputs_for_field(field_id),
                derivation_source=(
                    "runtime_assurance/staged_recovery_instrumentation.py"
                    if coverage
                    in {
                        COVERAGE_PURE_DERIVATION,
                        COVERAGE_PREVIOUS_STATE,
                        COVERAGE_PREDICTED_STATE,
                    }
                    else "explicit_external_input_or_future_contract"
                ),
                missing_value_rule="null with status not_evaluated; never zero or false",
                invalid_value_rule="null with status invalid and a preserved reason",
                architecture_phases_requiring_it=tuple(
                    sorted(set(phases_by_signal.get(field_id, ())))
                ),
                scientific_interpretation_limitation=(
                    "This field is one component only and does not establish safety, recovery, or phase readiness."
                ),
                canonical_order_index=index,
            )
        )
    return tuple(definitions)


def coverage_counts() -> tuple[tuple[str, int], ...]:
    coverage = dict(architecture_signal_coverage())
    return tuple(
        (classification, sum(value == classification for value in coverage.values()))
        for classification in COVERAGE_CLASSIFICATIONS
    )


def _to_json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {
            field.name: _to_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    return value


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _to_json_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_document_hash_is_valid(document: Mapping[str, object]) -> bool:
    payload = dict(document)
    stored_hash = payload.pop("canonical_payload_hash", None)
    return isinstance(stored_hash, str) and stored_hash == canonical_sha256(payload)


def _record_payload(record: StagedRecoveryInstrumentationRecord) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "fields": _to_json_value(record.fields),
    }


def canonical_record_sha256(record: StagedRecoveryInstrumentationRecord) -> str:
    return canonical_sha256(_record_payload(record))


def with_canonical_record_hash(
    record: StagedRecoveryInstrumentationRecord,
) -> StagedRecoveryInstrumentationRecord:
    unhashed = replace(record, canonical_record_hash="")
    return replace(unhashed, canonical_record_hash=canonical_record_sha256(unhashed))


def build_instrumentation_record(
    *,
    state: CartesianState2D | None,
    configuration: OrbitalConfiguration | None,
    case_id: str | None = None,
    seed: int | None = None,
    implementation_commit: str | None = None,
    branch_state_hash: str | None = None,
    simulator_configuration_hash: str | None = None,
    constants_hash: str | None = None,
    recovery_step: int | None = None,
    total_transition_count: int | None = None,
    simulation_time: float | None = None,
    previous_state: CartesianState2D | None = None,
    previous_step: int | None = None,
    predicted_state: CartesianState2D | None = None,
    proposed_action: tuple[float, float] | None = None,
    executed_action: tuple[float, float] | None = None,
    branch_step: int | None = None,
    action_rejected: bool = False,
    explicit_abort: bool = False,
    external_fields: Mapping[str, InstrumentedValue] | None = None,
    volatile_provenance_timestamp: str | None = None,
) -> StagedRecoveryInstrumentationRecord:
    values = {
        field_id: not_evaluated_value(
            reason="field_not_supplied_or_not_derived_in_stage_0a",
            units=next(
                definition.units
                for definition in field_catalog()
                if definition.field_id == field_id
            ),
            source_id="build_instrumentation_record",
            source_step=recovery_step,
        )
        for field_id in CANONICAL_FIELD_ORDER
    }
    direct = {
        "case_id": case_id,
        "seed": seed,
        "implementation_commit": implementation_commit,
        "branch_state_hash": branch_state_hash,
        "simulator_configuration_hash": simulator_configuration_hash,
        "constants_hash": constants_hash,
        "recovery_step": recovery_step,
        "total_transition_count": total_transition_count,
        "simulation_time": simulation_time,
    }
    for field_id, value in direct.items():
        definition = next(item for item in field_catalog() if item.field_id == field_id)
        if value is not None:
            values[field_id] = measured_value(
                value,
                units=definition.units,
                source_id=f"explicit.{field_id}",
                source_step=recovery_step,
            )
    state_problem = _state_problem(state)
    for field_id, attribute in (
        ("position_x", "x"),
        ("position_y", "y"),
        ("velocity_x", "vx"),
        ("velocity_y", "vy"),
    ):
        if state_problem is None and state is not None:
            values[field_id] = measured_value(
                float(getattr(state, attribute)),
                units=_UNITS[field_id],
                source_id=f"explicit.state.{attribute}",
                source_step=recovery_step,
            )
        elif state_problem and state_problem[0] == "invalid":
            values[field_id] = invalid_value(
                reason=state_problem[1],
                units=_UNITS[field_id],
                source_id="explicit.state",
                source_step=recovery_step,
            )
    derived = derive_orbital_state(state, configuration, source_step=recovery_step)
    values.update(dict(derived.values))
    predicted = derive_predicted_hazard_state(
        predicted_state,
        configuration,
        source_step=recovery_step,
    )
    values.update(dict(predicted))
    recoverability = derive_phase34_recoverability(
        derived.field("radius_error_ratio"),
        derived.field("radial_velocity_ratio"),
        derived.field("tangential_velocity_error_ratio"),
        source_step=recovery_step,
    )
    values.update(dict(recoverability.values))
    crossing = derive_crossing_event(
        previous_state,
        state,
        configuration,
        previous_step=previous_step,
        current_step=recovery_step,
        branch_step=branch_step,
    )
    values.update(dict(crossing.values))
    basis = derive_orbital_basis(state, source_step=recovery_step)
    action = derive_action_geometry(
        proposed_action,
        executed_action,
        basis,
        action_component_limit=(
            configuration.action_component_limit
            if configuration is not None
            else NORMALIZED_ACTION_COMPONENT_LIMIT
        ),
        action_rejected=action_rejected,
        explicit_abort=explicit_abort,
        source_step=recovery_step,
    )
    values.update(dict(action.values))
    values["action_rejection_status"] = measured_value(
        action_rejected,
        units="boolean",
        source_id="explicit.action_rejected",
        source_step=recovery_step,
    )
    values["explicit_abort_requested"] = measured_value(
        explicit_abort,
        units="boolean",
        source_id="explicit.explicit_abort",
        source_step=recovery_step,
    )
    if configuration is not None and _configuration_problem(configuration) is None:
        values["target_radius"] = measured_value(
            float(configuration.target_radius),
            units="m",
            source_id="orbital_configuration.target_radius",
            source_step=recovery_step,
        )
    if external_fields:
        unknown = sorted(set(external_fields) - set(CANONICAL_FIELD_ORDER))
        if unknown:
            raise InstrumentationContractError(f"unknown external fields: {unknown}")
        if any(not isinstance(value, InstrumentedValue) for value in external_fields.values()):
            raise InstrumentationContractError(
                "external fields must be InstrumentedValue instances"
            )
        values.update(external_fields)
    ordered = tuple((field_id, values[field_id]) for field_id in CANONICAL_FIELD_ORDER)
    record = StagedRecoveryInstrumentationRecord(
        schema_version=INSTRUMENTATION_SCHEMA_VERSION,
        fields=ordered,
        volatile_provenance_timestamp=volatile_provenance_timestamp,
        canonical_record_hash="",
    )
    return with_canonical_record_hash(record)


def field_catalog_document() -> dict[str, object]:
    payload: dict[str, object] = {
        "field_catalog_schema_version": FIELD_CATALOG_SCHEMA_VERSION,
        "instrumentation_id": INSTRUMENTATION_ID,
        "completed_date": COMPLETED_DATE,
        "fields": [_to_json_value(definition) for definition in field_catalog()],
        "architecture_signal_coverage": [
            list(item) for item in architecture_signal_coverage()
        ],
        "coverage_counts": [list(item) for item in coverage_counts()],
    }
    payload["canonical_payload_hash"] = canonical_sha256(payload)
    return payload


def derivation_traceability() -> tuple[dict[str, object], ...]:
    rows = (
        (
            "orbital_basis_radius",
            "radius",
            "sqrt(x*x + y*y)",
            "analysis/recovery_action_branching_nonformal_v0/manifest.json:coordinate_convention.position_norm",
            ("position_x", "position_y"),
            "m",
            "position norm must be finite and greater than zero",
            "exact nonzero boundary; no state-hash inference",
            ("test_radius_derivation_is_exact", "test_zero_position_norm_is_invalid"),
            "Geometry only; no transition or physical closeness inference.",
        ),
        (
            "orbital_radial_basis",
            "radial_unit_vector",
            "e_r = (x/r, y/r)",
            "runtime_assurance/recovery_branch_executor.py:generate_tangential_correction_action",
            ("position_x", "position_y", "radius"),
            "dimensionless",
            "invalid basis propagates invalid evidence",
            "counterpart tangential basis is a positive 90-degree rotation",
            ("test_radial_unit_vector_is_normalized", "test_all_quadrants"),
            "No normalization is attempted at zero radius.",
        ),
        (
            "orbital_tangential_basis",
            "tangential_unit_vector",
            "e_t = (-e_r_y, e_r_x)",
            "analysis/recovery_action_branching_nonformal_v0/manifest.json:coordinate_convention.positive_tangential_unit_vector",
            ("radial_unit_vector",),
            "dimensionless",
            "invalid radial basis propagates invalid evidence",
            "positive orientation is counterclockwise",
            ("test_tangential_orientation", "test_basis_is_orthogonal"),
            "Orientation is repository-specific and not inferred from a branch name.",
        ),
        (
            "radial_velocity",
            "radial_velocity",
            "dot((vx,vy), e_r)",
            "scripts/explicit_controller_phase21_orbital_transfer_planner.py:basis",
            ("velocity_x", "velocity_y", "radial_unit_vector"),
            "m/s",
            "invalid basis or velocity propagates invalid evidence",
            "signed; inward velocity remains negative",
            ("test_radial_outward", "test_radial_inward", "test_mixed_velocity"),
            "A single signed value does not classify progress.",
        ),
        (
            "tangential_velocity",
            "tangential_velocity",
            "dot((vx,vy), e_t)",
            "scripts/explicit_controller_phase21_orbital_transfer_planner.py:basis",
            ("velocity_x", "velocity_y", "tangential_unit_vector"),
            "m/s",
            "invalid basis or velocity propagates invalid evidence",
            "signed counterclockwise convention",
            ("test_positive_tangential_velocity", "test_negative_tangential_velocity"),
            "Signed component is not an action or task-success label.",
        ),
        (
            "target_circular_speed",
            "target_circular_speed",
            "sqrt(mu / target_radius)",
            "scripts/explicit_controller_phase21_orbital_transfer_planner.py:orbital_diagnostics",
            ("mu", "target_radius"),
            "m/s",
            "mu and target radius must be finite and positive",
            "no fallback denominator or guessed target",
            ("test_target_circular_speed_matches_repository_formula",),
            "Current simplified simulator target only.",
        ),
        (
            "target_error_ratios",
            "radius_error_ratio,radial_velocity_ratio,tangential_velocity_error_ratio",
            "(r-r_target)/r_target; v_r/(v_circ+1e-12); (v_t-v_circ)/(v_circ+1e-12)",
            "runtime_assurance/recovery_experiment_runner.py:_observe",
            ("radius", "target_radius", "radial_velocity", "tangential_velocity", "target_circular_speed"),
            "dimensionless",
            "nonpositive denominator is invalid",
            "signed ratios retained",
            ("test_target_ratios", "test_invalid_denominator"),
            "Ratios are components, not a combined recovery score.",
        ),
        (
            "realized_speed_ratio",
            "realized_speed_ratio",
            "speed / (target_circular_speed + speed_ratio_denominator_epsilon)",
            "runtime_assurance/recovery_experiment_runner.py:_observe",
            ("speed_magnitude", "target_circular_speed"),
            "dimensionless",
            "nonpositive denominator is invalid",
            "strict overspeed comparator remains > 1.90",
            ("test_realized_speed_ratio", "test_exact_threshold_is_not_overspeed"),
            "Below one hazard threshold does not establish safety.",
        ),
        (
            "overspeed_headroom",
            "overspeed_headroom",
            "1.90 - speed_ratio",
            "runtime_assurance/final_veto_monitor.py:OVERSPEED_THRESHOLD",
            ("realized_speed_ratio",),
            "dimensionless",
            "missing ratio remains not_evaluated",
            "positive below, zero at, negative above strict threshold",
            ("test_headroom_signs",),
            "Headroom is relative to one diagnostic threshold, not formal safety margin.",
        ),
        (
            "specific_orbital_energy_proxy",
            "specific_orbital_energy",
            "0.5*speed^2 - mu/(radius+1e-12)",
            "scripts/explicit_controller_phase21_orbital_transfer_planner.py:orbital_diagnostics",
            ("speed_magnitude", "mu", "radius"),
            "J/kg",
            "supported gravity model and positive radius required",
            "declared diagnostic proxy; not exact softened-force invariant",
            ("test_specific_energy_proxy", "test_unsupported_energy_model"),
            "Specific energy is not mass-scaled, fuel use, or exact conserved energy.",
        ),
        (
            "phase34_recoverability",
            "phase34_compatible_recoverability",
            "abs(r_ratio)<=0.0025 and abs(vr_ratio)<=0.02 and abs(vt_ratio)<=0.25",
            "runtime_assurance/recovery_evaluators.py:evaluate_phase34_compatible_recoverability",
            ("radius_error_ratio", "radial_velocity_ratio", "tangential_velocity_error_ratio"),
            "boolean",
            "all three finite components required",
            "inclusive component bounds",
            ("test_recoverability_boundaries", "test_recoverability_equivalence"),
            "Crossing, simulator success, and hazard avoidance remain separate.",
        ),
        (
            "target_radius_crossing",
            "target_radius_crossing",
            "previous_error>0 and current_error<=0, or previous_error<0 and current_error>=0",
            "scripts/explicit_controller_phase34_post_cross_sync.py:rollout_phase34_case crossing test",
            ("previous_state", "current_state", "target_radius"),
            "boolean",
            "both measured states and valid target required",
            "previous exactly on target is not a new crossing; current exactly on target can be",
            ("test_crossing_directions", "test_exact_target_crossing_semantics"),
            "No interpolation is performed.",
        ),
        (
            "threshold_free_progress",
            "delta_*",
            "current component minus previous component",
            "docs/architecture/staged_recovery_architecture_v0.md:No-Progress Detection",
            ("previous_orbital_state", "current_orbital_state"),
            "component-specific",
            "missing and invalid components propagate independently",
            "no threshold or progress class is applied",
            ("test_raw_progress_deltas", "test_progress_has_no_classification"),
            "Short-term component changes do not imply eventual recovery.",
        ),
        (
            "action_geometry",
            "action radial/tangential components and saturation margin",
            "dot(action,e_r); dot(action,e_t); 1-max(abs(action components))",
            "analysis/recovery_action_branching_nonformal_v0/manifest.json:coordinate_convention and simulator/phase34_35_transition.py action bounds",
            ("explicit_action", "orbital_basis", "action_component_limit"),
            "normalized_action",
            "explicit finite action and valid basis required",
            "abort and rejection remain distinct from physical zero action",
            ("test_action_geometry", "test_explicit_abort_has_no_action"),
            "Normalized action is not delta-v or a generated policy action.",
        ),
    )
    keys = (
        "derivation_id",
        "output_field",
        "formula_description",
        "exact_repository_source",
        "input_fields",
        "units",
        "evidence_status",
        "invalid_conditions",
        "boundary_semantics",
        "tests_covering_it",
        "limitations",
    )
    return tuple(
        dict(zip(keys, row[:6] + ("derived",) + row[6:])) for row in rows
    )


def derivation_traceability_document() -> dict[str, object]:
    payload: dict[str, object] = {
        "derivation_traceability_schema_version": DERIVATION_TRACEABILITY_SCHEMA_VERSION,
        "instrumentation_id": INSTRUMENTATION_ID,
        "completed_date": COMPLETED_DATE,
        "derivations": _to_json_value(derivation_traceability()),
    }
    payload["canonical_payload_hash"] = canonical_sha256(payload)
    return payload


CLAIM_RESTRICTIONS = (
    "no_runtime_instrumentation_completeness_claim",
    "no_staged_recovery_execution_readiness_claim",
    "no_recovery_controller_or_phase_action_claim",
    "no_phase_guard_or_threshold_claim",
    "no_no_progress_or_hysteresis_parameter_claim",
    "no_task_recovery_claim",
    "no_formal_safety_claim",
    "no_hardware_or_deployment_claim",
)


def instrumentation_manifest_payload() -> dict[str, object]:
    coverage = dict(architecture_signal_coverage())
    direct = [key for key, value in coverage.items() if value == COVERAGE_DIRECT_INPUT]
    pure = [key for key, value in coverage.items() if value == COVERAGE_PURE_DERIVATION]
    unsupported_runtime = [
        key
        for key, value in coverage.items()
        if value
        in {
            COVERAGE_RUNTIME_INTEGRATION,
            COVERAGE_FUTURE_EVALUATOR,
            COVERAGE_NOT_SUPPORTED,
        }
    ]
    return {
        "instrumentation_id": INSTRUMENTATION_ID,
        "manifest_schema_version": INSTRUMENTATION_MANIFEST_SCHEMA_VERSION,
        "record_schema_version": INSTRUMENTATION_SCHEMA_VERSION,
        "completed_date": COMPLETED_DATE,
        "source_architecture": {
            "architecture_id": ARCHITECTURE_ID,
            "commit": SOURCE_ARCHITECTURE_COMMIT,
            "canonical_hash": SOURCE_ARCHITECTURE_CANONICAL_HASH,
            "artifact_hashes": [list(item) for item in SOURCE_ARCHITECTURE_ARTIFACT_HASHES],
        },
        "coordinate_convention": {
            "frame": "inertial_cartesian_2d",
            "state_order": ["x", "y", "vx", "vy"],
            "radial_unit_vector": "e_r = (x/r, y/r)",
            "positive_tangential_unit_vector": "e_t = (-e_r_y, e_r_x)",
            "positive_tangential_orientation": "counterclockwise_90_degree_rotation_of_e_r",
        },
        "units": {
            "position": "m",
            "velocity": "m/s",
            "specific_energy": "J/kg",
            "time": "s",
            "ratios": "dimensionless",
            "actions": "normalized_action",
        },
        "zero_tolerances": {
            "position_norm_invalid_at_or_below": POSITION_NORM_ZERO_TOLERANCE,
            "vector_and_tangential_error_zero_tolerance": VECTOR_ZERO_TOLERANCE,
            "ratio_denominator_epsilon": RATIO_DENOMINATOR_EPSILON,
            "speed_ratio_denominator_epsilon": SPEED_RATIO_DENOMINATOR_EPSILON,
            "specific_energy_radius_epsilon": SPECIFIC_ENERGY_RADIUS_EPSILON,
        },
        "supported_direct_inputs": direct,
        "supported_pure_derivations": pure,
        "unsupported_runtime_fields": unsupported_runtime,
        "architecture_signal_coverage": [list(item) for item in architecture_signal_coverage()],
        "coverage_counts": [list(item) for item in coverage_counts()],
        "phase34_compatible_thresholds": {
            "radius_error_ratio_max": PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
            "radial_velocity_ratio_max": PHASE34_RECOVERABLE_VR_RATIO_MAX,
            "tangential_velocity_error_ratio_max": PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
            "comparator": "inclusive_absolute_component_bounds",
        },
        "overspeed_contract": {
            "threshold": OVERSPEED_THRESHOLD,
            "comparator": OVERSPEED_COMPARATOR,
            "headroom": "1.90 - speed_ratio",
        },
        "adverse_stop_priority": list(FROZEN_ADVERSE_STOP_PRIORITY),
        "canonical_field_order": list(CANONICAL_FIELD_ORDER),
        "canonicalization": {
            "encoding": "utf-8",
            "sort_keys": True,
            "separators": [",", ":"],
            "allow_nan": False,
            "hash_algorithm": "sha256",
            "self_hash_field_excluded": True,
            "volatile_provenance_timestamp_excluded_from_record_hash": True,
        },
        "pure_derivation_status": PURE_DERIVATION_IMPLEMENTED,
        "runtime_logger_integration": RUNTIME_LOGGER_NOT_IMPLEMENTED,
        "staged_recovery_execution": STAGED_EXECUTION_NOT_AUTHORIZED,
        "execution_authorization_reason": EXECUTION_NOT_AUTHORIZED_REASON,
        "claim_restrictions": list(CLAIM_RESTRICTIONS),
    }


def instrumentation_manifest_document() -> dict[str, object]:
    payload = instrumentation_manifest_payload()
    payload["canonical_payload_hash"] = canonical_sha256(payload)
    return payload


def canonical_instrumentation_manifest_sha256(
    document: Mapping[str, object] | None = None,
) -> str:
    current = dict(document or instrumentation_manifest_document())
    current.pop("canonical_payload_hash", None)
    return canonical_sha256(current)


def validate_instrumentation_contract(
    *,
    manifest: Mapping[str, object] | None = None,
    catalog: Mapping[str, object] | None = None,
    traceability: Mapping[str, object] | None = None,
) -> InstrumentationValidationReport:
    manifest_document = dict(manifest or instrumentation_manifest_document())
    catalog_document = dict(catalog or field_catalog_document())
    trace_document = dict(traceability or derivation_traceability_document())
    errors: list[str] = []
    if manifest_document.get("manifest_schema_version") != INSTRUMENTATION_MANIFEST_SCHEMA_VERSION:
        errors.append("manifest_schema_version_mismatch")
    if not canonical_document_hash_is_valid(manifest_document):
        errors.append("manifest_canonical_hash_mismatch")
    source = manifest_document.get("source_architecture")
    if not isinstance(source, dict) or (
        source.get("architecture_id") != ARCHITECTURE_ID
        or source.get("commit") != SOURCE_ARCHITECTURE_COMMIT
        or source.get("canonical_hash") != SOURCE_ARCHITECTURE_CANONICAL_HASH
        or source.get("artifact_hashes")
        != [list(item) for item in SOURCE_ARCHITECTURE_ARTIFACT_HASHES]
    ):
        errors.append("source_architecture_hash_mismatch")
    if manifest_document.get("staged_recovery_execution") != STAGED_EXECUTION_NOT_AUTHORIZED:
        errors.append("staged_execution_must_remain_unauthorized")
    if manifest_document.get("runtime_logger_integration") != RUNTIME_LOGGER_NOT_IMPLEMENTED:
        errors.append("runtime_logger_must_remain_not_implemented")
    overspeed = manifest_document.get("overspeed_contract")
    if not isinstance(overspeed, dict) or (
        overspeed.get("threshold") != OVERSPEED_THRESHOLD
        or overspeed.get("comparator") != OVERSPEED_COMPARATOR
    ):
        errors.append("overspeed_contract_drift")
    thresholds = manifest_document.get("phase34_compatible_thresholds")
    expected_thresholds = {
        "radius_error_ratio_max": PHASE34_RECOVERABLE_R_ERROR_RATIO_MAX,
        "radial_velocity_ratio_max": PHASE34_RECOVERABLE_VR_RATIO_MAX,
        "tangential_velocity_error_ratio_max": PHASE34_RECOVERABLE_VT_ERROR_RATIO_MAX,
        "comparator": "inclusive_absolute_component_bounds",
    }
    if thresholds != expected_thresholds:
        errors.append("phase34_threshold_drift")
    if manifest_document.get("adverse_stop_priority") != list(FROZEN_ADVERSE_STOP_PRIORITY):
        errors.append("adverse_stop_priority_drift")
    if manifest_document.get("canonical_field_order") != list(CANONICAL_FIELD_ORDER):
        errors.append("manifest_canonical_field_order_mismatch")
    if manifest_document.get("architecture_signal_coverage") != [
        list(item) for item in architecture_signal_coverage()
    ]:
        errors.append("manifest_architecture_coverage_mismatch")
    coverage = dict(architecture_signal_coverage())
    if set(coverage) != set(ARCHITECTURE_SIGNAL_IDS):
        errors.append("architecture_signal_coverage_mismatch")
    if any(value not in COVERAGE_CLASSIFICATIONS for value in coverage.values()):
        errors.append("unknown_coverage_classification")
    if catalog_document.get("field_catalog_schema_version") != FIELD_CATALOG_SCHEMA_VERSION:
        errors.append("field_catalog_schema_version_mismatch")
    if not canonical_document_hash_is_valid(catalog_document):
        errors.append("field_catalog_canonical_hash_mismatch")
    if catalog_document.get("architecture_signal_coverage") != [
        list(item) for item in architecture_signal_coverage()
    ]:
        errors.append("field_catalog_architecture_coverage_mismatch")
    fields_document = catalog_document.get("fields")
    if not isinstance(fields_document, list):
        errors.append("field_catalog_is_missing")
        fields_document = []
    field_ids = [item.get("field_id") for item in fields_document if isinstance(item, dict)]
    if len(field_ids) != len(set(field_ids)):
        errors.append("duplicate_field_id")
    if field_ids != list(CANONICAL_FIELD_ORDER):
        errors.append("canonical_field_order_mismatch")
    if not set(ARCHITECTURE_SIGNAL_IDS).issubset(set(field_ids)):
        errors.append("architecture_signal_missing_from_catalog")
    required_field_definition_keys = {
        field.name for field in fields(InstrumentationFieldDefinition)
    }
    if any(
        not isinstance(item, dict)
        or not required_field_definition_keys.issubset(item)
        for item in fields_document
    ):
        errors.append("field_catalog_entry_is_incomplete")
    if trace_document.get(
        "derivation_traceability_schema_version"
    ) != DERIVATION_TRACEABILITY_SCHEMA_VERSION:
        errors.append("derivation_traceability_schema_version_mismatch")
    if not canonical_document_hash_is_valid(trace_document):
        errors.append("derivation_traceability_canonical_hash_mismatch")
    derivations = trace_document.get("derivations")
    if not isinstance(derivations, list) or not derivations:
        errors.append("derivation_traceability_is_missing")
    else:
        required_trace_fields = {
            "derivation_id",
            "output_field",
            "formula_description",
            "exact_repository_source",
            "input_fields",
            "units",
            "evidence_status",
            "invalid_conditions",
            "boundary_semantics",
            "tests_covering_it",
            "limitations",
        }
        if any(
            not isinstance(item, dict) or not required_trace_fields.issubset(item)
            for item in derivations
        ):
            errors.append("derivation_traceability_entry_is_incomplete")
    return InstrumentationValidationReport(
        valid=not errors,
        errors=tuple(sorted(set(errors))),
        coverage_counts=coverage_counts(),
        architecture_signal_count=len(ARCHITECTURE_SIGNAL_IDS),
        catalog_field_count=len(CANONICAL_FIELD_ORDER),
    )


__all__ = [
    "ARCHITECTURE_SIGNAL_IDS",
    "CANONICAL_FIELD_ORDER",
    "COMPLETED_DATE",
    "COVERAGE_CLASSIFICATIONS",
    "COVERAGE_DIRECT_INPUT",
    "COVERAGE_FUTURE_EVALUATOR",
    "COVERAGE_NOT_SUPPORTED",
    "COVERAGE_PREDICTED_STATE",
    "COVERAGE_PREVIOUS_STATE",
    "COVERAGE_PURE_DERIVATION",
    "COVERAGE_RUNTIME_INTEGRATION",
    "CartesianState2D",
    "ActionGeometry",
    "CrossingEvent",
    "CURRENT_GRAVITY_MODEL_ID",
    "DERIVATION_TRACEABILITY_SCHEMA_VERSION",
    "EVIDENCE_STATUS_TO_ARCHITECTURE_STATUS",
    "EXECUTION_NOT_AUTHORIZED_REASON",
    "FIELD_CATALOG_SCHEMA_VERSION",
    "INSTRUMENTATION_ID",
    "INSTRUMENTATION_MANIFEST_SCHEMA_VERSION",
    "INSTRUMENTATION_SCHEMA_VERSION",
    "InstrumentationContractError",
    "InstrumentationEvidenceStatus",
    "InstrumentationFieldDefinition",
    "InstrumentationValidationReport",
    "InstrumentedValue",
    "NORMALIZED_ACTION_COMPONENT_LIMIT",
    "OrbitalBasis2D",
    "OrbitalConfiguration",
    "OrbitalDerivedState",
    "PURE_DERIVATION_IMPLEMENTED",
    "POSITION_NORM_ZERO_TOLERANCE",
    "RATIO_DENOMINATOR_EPSILON",
    "RUNTIME_LOGGER_NOT_IMPLEMENTED",
    "RecoverabilityComponents",
    "RecoveryProgressSample",
    "SOURCE_ARCHITECTURE_CANONICAL_HASH",
    "SOURCE_ARCHITECTURE_COMMIT",
    "SOURCE_ARCHITECTURE_ARTIFACT_HASHES",
    "SPECIFIC_ENERGY_MODEL_STATUS",
    "SPEED_RATIO_DENOMINATOR_EPSILON",
    "STAGED_EXECUTION_NOT_AUTHORIZED",
    "StagedRecoveryInstrumentationRecord",
    "VECTOR_ZERO_TOLERANCE",
    "architecture_signal_coverage",
    "build_instrumentation_record",
    "canonical_instrumentation_manifest_sha256",
    "canonical_document_hash_is_valid",
    "canonical_json_bytes",
    "canonical_record_sha256",
    "canonical_sha256",
    "coverage_counts",
    "derive_action_geometry",
    "derive_crossing_event",
    "derive_orbital_basis",
    "derive_orbital_state",
    "derive_phase34_recoverability",
    "derive_predicted_hazard_state",
    "derive_progress_sample",
    "derivation_traceability",
    "derivation_traceability_document",
    "derived_value",
    "field_catalog",
    "field_catalog_document",
    "instrumentation_manifest_document",
    "instrumentation_manifest_payload",
    "invalid_value",
    "measured_value",
    "not_evaluated_value",
    "predicted_value",
    "validate_instrumentation_contract",
    "with_canonical_record_hash",
]

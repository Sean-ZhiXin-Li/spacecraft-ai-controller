from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.final_veto_artifacts import (  # noqa: E402
    DECISION_FIELDNAMES,
    DECISION_SCHEMA_VERSION,
    JsonlEventWriter,
)


LOG_MODE_COMPACT = "compact"
LOG_MODE_FULL_TRACE = "full_trace"
VALID_DECISION_LOG_MODES = (LOG_MODE_COMPACT, LOG_MODE_FULL_TRACE)
FORMAL_DEFAULT_DECISION_LOG_MODE = LOG_MODE_COMPACT

MAX_DECISION_STEPS = 100_000
MAX_COMPACT_RECORDS_PER_RUN = 1_024
MAX_COMPACT_RECORD_BYTES = 4_096
FORMAL_PUBLIC_LOG_BYTE_BUDGET = 64 * 1024 * 1024
EXPECTED_COMPACT_RECORDS_PER_MONITOR_ARM = 3


class CompactDecisionLogError(RuntimeError):
    pass


def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def update_canonical_stream_digest(
    digest: "hashlib._Hash",
    event: Mapping[str, object],
) -> None:
    # A newline is a deterministic record boundary for the logical JSONL stream.
    digest.update(canonical_json_bytes(event))
    digest.update(b"\n")


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return bool(value)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def logging_configuration_errors(
    *,
    mode: str,
    is_formal_experiment: bool,
    full_trace_path: Path | None,
    repository_root: Path,
    digest_enabled: bool = True,
) -> list[str]:
    errors: list[str] = []
    if mode not in VALID_DECISION_LOG_MODES:
        errors.append(f"unsupported decision log mode: {mode}")
        return errors
    if not digest_enabled:
        errors.append("deterministic decision-stream digest is disabled")
    if is_formal_experiment and mode != LOG_MODE_COMPACT:
        errors.append("formal execution requires compact decision logging")
    if mode == LOG_MODE_FULL_TRACE:
        if is_formal_experiment:
            errors.append("full per-step tracing is explicitly nonformal only")
        if full_trace_path is None:
            errors.append("full_trace mode requires an explicit user-supplied output path")
        elif _is_within(full_trace_path, repository_root):
            errors.append("full_trace output must be outside the repository")
    elif full_trace_path is not None:
        errors.append("--full-trace-path is valid only with full_trace mode")
    return errors


@dataclass(frozen=True, slots=True)
class CompactLoggingPlanEstimate:
    logging_mode: str
    digest_enabled: bool
    monitor_on_jobs: int
    maximum_steps_per_monitor_arm: int
    maximum_logical_events: int
    segmentation_or_exception_upper_bound_per_run: int
    terminal_records_per_run: int
    semantic_record_upper_bound_without_policy_cap: int
    enforced_record_limit_per_run: int
    maximum_public_records: int
    maximum_public_bytes: int
    expected_public_records: int
    expected_serialized_bytes: int


def dry_run_expected_compact_bytes_per_monitor_arm(job: object) -> int:
    def identity(field: str, default: object) -> object:
        return getattr(job, field, default)

    base = {
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "experiment_id": identity("experiment_id", "final_veto_overspeed_ablation_v0"),
        "run_id": identity("run_id", "representative_monitor_on_run"),
        "paired_run_id": identity("paired_run_id", "representative_pair"),
        "case_id": identity("case_id", "representative_case"),
        "subset_id": identity("subset_id", "representative_subset"),
        "arm_id": "monitor_on",
        "monitor_id": "one_step_overspeed_veto_v0",
        "phase": "DESCENT",
        "active_stage": "radial_energy_push",
        "invalid_evaluation": False,
        "fallback_failure": False,
        "terminal_state": False,
        "hazard_threshold": 1.90,
        "hazard_comparator": ">",
        "is_formal_experiment": True,
        "false_negative": False,
    }
    allow_event = {
        **base,
        "step": 1,
        "decision_type": "continue",
        "decision_reason": "predicted_nominal_within_threshold",
        "veto_status": "allow",
        "fallback_executed": False,
        "nominal_proposed_action": [-0.06896833813278117, 0.13390994373224901],
        "executed_action": [-0.06896833813278117, 0.13390994373224901],
        "predicted_nominal_speed_ratio": 1.8906024454524528,
        "predicted_fallback_speed_ratio": None,
        "realized_executed_speed_ratio": 1.8906024454524528,
    }
    veto_event = {
        **base,
        "step": MAX_DECISION_STEPS,
        "decision_type": "veto_action",
        "decision_reason": "predicted_nominal_exceeds_overspeed_threshold",
        "veto_status": "veto",
        "fallback_executed": True,
        "nominal_proposed_action": [-0.0007711865733707108, 0.11305563567553462],
        "executed_action": [0.0, 0.0],
        "predicted_nominal_speed_ratio": 1.9183887199363643,
        "predicted_fallback_speed_ratio": 1.8906024003603095,
        "realized_executed_speed_ratio": 1.8906024003603095,
    }
    terminal = {
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "event_kind": "terminal_transition",
        "experiment_id": base["experiment_id"],
        "run_id": base["run_id"],
        "paired_run_id": base["paired_run_id"],
        "case_id": base["case_id"],
        "subset_id": base["subset_id"],
        "arm_id": "monitor_on",
        "monitor_id": base["monitor_id"],
        "step": MAX_DECISION_STEPS,
        "terminal_state": True,
        "terminal_label": "no_crossing",
        "termination_reason": "max_steps",
        "is_formal_experiment": True,
    }
    records = (_new_segment(allow_event), _new_segment(veto_event), terminal)
    return sum(len(canonical_json_bytes(record)) + 1 for record in records)


def estimate_compact_logging_plan(
    jobs: Iterable[object],
    *,
    mode: str = FORMAL_DEFAULT_DECISION_LOG_MODE,
    digest_enabled: bool = True,
    maximum_steps: int = MAX_DECISION_STEPS,
) -> CompactLoggingPlanEstimate:
    planned_jobs = list(jobs)
    monitor_jobs = [job for job in planned_jobs if bool(getattr(job, "monitor_enabled", False))]
    monitor_on_jobs = len(monitor_jobs)
    logical_events = monitor_on_jobs * maximum_steps
    per_run_unbounded_by_data = maximum_steps
    terminal_records_per_run = 1
    semantic_upper_bound = monitor_on_jobs * (
        per_run_unbounded_by_data + terminal_records_per_run
    )
    maximum_records = monitor_on_jobs * MAX_COMPACT_RECORDS_PER_RUN
    maximum_bytes = maximum_records * (MAX_COMPACT_RECORD_BYTES + 1)
    expected_records = min(
        maximum_records,
        monitor_on_jobs * EXPECTED_COMPACT_RECORDS_PER_MONITOR_ARM,
    )
    dry_run_bytes_per_arm = max(
        (dry_run_expected_compact_bytes_per_monitor_arm(job) for job in monitor_jobs),
        default=0,
    )
    return CompactLoggingPlanEstimate(
        logging_mode=mode,
        digest_enabled=digest_enabled,
        monitor_on_jobs=monitor_on_jobs,
        maximum_steps_per_monitor_arm=maximum_steps,
        maximum_logical_events=logical_events,
        segmentation_or_exception_upper_bound_per_run=per_run_unbounded_by_data,
        terminal_records_per_run=terminal_records_per_run,
        semantic_record_upper_bound_without_policy_cap=semantic_upper_bound,
        enforced_record_limit_per_run=MAX_COMPACT_RECORDS_PER_RUN,
        maximum_public_records=maximum_records,
        maximum_public_bytes=maximum_bytes,
        expected_public_records=expected_records,
        expected_serialized_bytes=monitor_on_jobs * dry_run_bytes_per_arm,
    )


def compact_logging_preflight_errors(
    jobs: Iterable[object],
    *,
    mode: str,
    digest_enabled: bool = True,
) -> list[str]:
    planned = list(jobs)
    estimate = estimate_compact_logging_plan(
        planned,
        mode=mode,
        digest_enabled=digest_enabled,
    )
    errors: list[str] = []
    if mode != LOG_MODE_COMPACT:
        errors.append("formal public decision logging would be unbounded per-step logging")
    if not digest_enabled:
        errors.append("formal compact logging requires the deterministic stream digest")
    if estimate.maximum_public_records <= 0:
        errors.append("formal compact logging has no finite public record bound")
    if estimate.maximum_public_bytes > FORMAL_PUBLIC_LOG_BYTE_BUDGET:
        errors.append(
            "formal compact log bound exceeds the public byte budget: "
            f"{estimate.maximum_public_bytes} > {FORMAL_PUBLIC_LOG_BYTE_BUDGET}"
        )
    return errors


class DecisionStreamStatistics:
    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.event_count = 0
        self.allow_count = 0
        self.veto_count = 0
        self.fallback_count = 0
        self.invalid_evaluation_count = 0
        self.false_negative_count = 0
        self.fallback_failure_count = 0
        self.first_veto_step: int | None = None
        self.last_veto_step: int | None = None
        self.longest_consecutive_veto_steps = 0
        self.longest_consecutive_allow_steps = 0
        self.veto_segment_count = 0
        self.allow_segment_count = 0
        self.decision_state_transition_count = 0
        self.phase_change_count = 0
        self.active_stage_change_count = 0
        self._current_status: str | None = None
        self._current_streak = 0
        self._previous_step: int | None = None
        self._previous_state: tuple[object, ...] | None = None
        self._previous_segment_behavior: tuple[object, ...] | None = None
        self._previous_phase: object = None
        self._previous_stage: object = None

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    @property
    def last_event_step(self) -> int | None:
        return self._previous_step

    @property
    def intervention_rate(self) -> float | None:
        valid = self.allow_count + self.veto_count
        return self.veto_count / valid if valid else None

    @property
    def allow_rate(self) -> float | None:
        valid = self.allow_count + self.veto_count
        return self.allow_count / valid if valid else None

    @property
    def fallback_rate(self) -> float | None:
        valid = self.allow_count + self.veto_count
        return self.fallback_count / valid if valid else None

    def observe(self, event: Mapping[str, object]) -> None:
        update_canonical_stream_digest(self._digest, event)
        self.event_count += 1
        invalid = bool(event.get("invalid_evaluation"))
        decision_type = str(event.get("decision_type", ""))
        status: str | None = None
        if not invalid and decision_type == "continue":
            status = "allow"
            self.allow_count += 1
        elif not invalid and decision_type == "veto_action":
            status = "veto"
            self.veto_count += 1
        if bool(event.get("fallback_executed")):
            self.fallback_count += 1
        if invalid:
            self.invalid_evaluation_count += 1
        if bool(event.get("false_negative")):
            self.false_negative_count += 1
        if bool(event.get("fallback_failure")):
            self.fallback_failure_count += 1

        step = int(event.get("step", 0) or 0)
        consecutive = self._previous_step is not None and step == self._previous_step + 1
        segment_behavior = _segment_key(event) + (exceptional_event_reasons(event),)
        if status is not None and (
            not consecutive or segment_behavior != self._previous_segment_behavior
        ):
            if status == "allow":
                self.allow_segment_count += 1
            else:
                self.veto_segment_count += 1
        if status is None:
            self._current_status = None
            self._current_streak = 0
        elif status == self._current_status and consecutive:
            self._current_streak += 1
        else:
            self._current_status = status
            self._current_streak = 1
        if status == "allow":
            self.longest_consecutive_allow_steps = max(
                self.longest_consecutive_allow_steps,
                self._current_streak,
            )
        elif status == "veto":
            self.first_veto_step = step if self.first_veto_step is None else self.first_veto_step
            self.last_veto_step = step
            self.longest_consecutive_veto_steps = max(
                self.longest_consecutive_veto_steps,
                self._current_streak,
            )

        state = (
            event.get("decision_type"),
            event.get("veto_status"),
            bool(event.get("fallback_executed")),
            invalid,
            bool(event.get("fallback_failure")),
            bool(event.get("terminal_state", False)),
        )
        if self._previous_state is not None and state != self._previous_state:
            self.decision_state_transition_count += 1
        if self._previous_step is not None and event.get("phase") != self._previous_phase:
            self.phase_change_count += 1
        if self._previous_step is not None and event.get("active_stage") != self._previous_stage:
            self.active_stage_change_count += 1
        self._previous_step = step
        self._previous_state = state
        self._previous_segment_behavior = segment_behavior
        self._previous_phase = event.get("phase")
        self._previous_stage = event.get("active_stage")


def _segment_key(event: Mapping[str, object]) -> tuple[object, ...]:
    return (
        event.get("decision_schema_version"),
        event.get("experiment_id"),
        event.get("run_id"),
        event.get("paired_run_id"),
        event.get("case_id"),
        event.get("subset_id"),
        event.get("arm_id"),
        event.get("phase"),
        event.get("active_stage"),
        event.get("decision_type"),
        event.get("decision_reason"),
        event.get("veto_status"),
        bool(event.get("fallback_executed")),
        bool(event.get("invalid_evaluation")),
        bool(event.get("fallback_failure")),
        bool(event.get("terminal_state", False)),
        event.get("monitor_id"),
        event.get("hazard_threshold"),
        event.get("hazard_comparator"),
        event.get("is_formal_experiment"),
    )


def _minimum(current: float | None, value: object) -> float | None:
    parsed = _optional_float(value)
    if parsed is None:
        return current
    return parsed if current is None else min(current, parsed)


def _maximum(current: float | None, value: object) -> float | None:
    parsed = _optional_float(value)
    if parsed is None:
        return current
    return parsed if current is None else max(current, parsed)


def _new_segment(event: Mapping[str, object]) -> dict[str, object]:
    step = int(event["step"])
    nominal = _optional_float(event.get("predicted_nominal_speed_ratio"))
    realized = _optional_float(event.get("realized_executed_speed_ratio"))
    fallback = _optional_float(event.get("predicted_fallback_speed_ratio"))
    return {
        "decision_schema_version": event.get("decision_schema_version", DECISION_SCHEMA_VERSION),
        "event_kind": "decision_segment",
        "experiment_id": event.get("experiment_id"),
        "run_id": event.get("run_id"),
        "paired_run_id": event.get("paired_run_id"),
        "case_id": event.get("case_id"),
        "subset_id": event.get("subset_id"),
        "arm_id": event.get("arm_id"),
        "monitor_id": event.get("monitor_id"),
        "start_step": step,
        "end_step": step,
        "step_count": 1,
        "phase": event.get("phase"),
        "active_stage": event.get("active_stage"),
        "decision_type": event.get("decision_type"),
        "decision_reason": event.get("decision_reason"),
        "veto_status": event.get("veto_status"),
        "fallback_executed": bool(event.get("fallback_executed")),
        "invalid_evaluation": bool(event.get("invalid_evaluation")),
        "fallback_failure": bool(event.get("fallback_failure")),
        "terminal_state": bool(event.get("terminal_state", False)),
        "fallback_failure_count": int(bool(event.get("fallback_failure"))),
        "false_negative_count": int(bool(event.get("false_negative"))),
        "first_nominal_action": event.get("nominal_proposed_action"),
        "last_nominal_action": event.get("nominal_proposed_action"),
        "first_executed_action": event.get("executed_action"),
        "last_executed_action": event.get("executed_action"),
        "first_predicted_nominal_speed_ratio": nominal,
        "last_predicted_nominal_speed_ratio": nominal,
        "minimum_predicted_nominal_speed_ratio": nominal,
        "maximum_predicted_nominal_speed_ratio": nominal,
        "first_realized_executed_speed_ratio": realized,
        "last_realized_executed_speed_ratio": realized,
        "minimum_realized_executed_speed_ratio": realized,
        "maximum_realized_executed_speed_ratio": realized,
        "minimum_predicted_fallback_speed_ratio": fallback,
        "maximum_predicted_fallback_speed_ratio": fallback,
        "hazard_threshold": event.get("hazard_threshold"),
        "hazard_comparator": event.get("hazard_comparator"),
        "is_formal_experiment": event.get("is_formal_experiment"),
    }


def _extend_segment(segment: dict[str, object], event: Mapping[str, object]) -> None:
    segment["end_step"] = int(event["step"])
    segment["step_count"] = int(segment["step_count"]) + 1
    segment["last_nominal_action"] = event.get("nominal_proposed_action")
    segment["last_executed_action"] = event.get("executed_action")
    segment["last_predicted_nominal_speed_ratio"] = _optional_float(
        event.get("predicted_nominal_speed_ratio")
    )
    segment["last_realized_executed_speed_ratio"] = _optional_float(
        event.get("realized_executed_speed_ratio")
    )
    segment["minimum_predicted_nominal_speed_ratio"] = _minimum(
        _optional_float(segment["minimum_predicted_nominal_speed_ratio"]),
        event.get("predicted_nominal_speed_ratio"),
    )
    segment["maximum_predicted_nominal_speed_ratio"] = _maximum(
        _optional_float(segment["maximum_predicted_nominal_speed_ratio"]),
        event.get("predicted_nominal_speed_ratio"),
    )
    segment["minimum_realized_executed_speed_ratio"] = _minimum(
        _optional_float(segment["minimum_realized_executed_speed_ratio"]),
        event.get("realized_executed_speed_ratio"),
    )
    segment["maximum_realized_executed_speed_ratio"] = _maximum(
        _optional_float(segment["maximum_realized_executed_speed_ratio"]),
        event.get("realized_executed_speed_ratio"),
    )
    segment["minimum_predicted_fallback_speed_ratio"] = _minimum(
        _optional_float(segment["minimum_predicted_fallback_speed_ratio"]),
        event.get("predicted_fallback_speed_ratio"),
    )
    segment["maximum_predicted_fallback_speed_ratio"] = _maximum(
        _optional_float(segment["maximum_predicted_fallback_speed_ratio"]),
        event.get("predicted_fallback_speed_ratio"),
    )
    segment["fallback_failure_count"] = int(segment["fallback_failure_count"]) + int(
        bool(event.get("fallback_failure"))
    )
    segment["false_negative_count"] = int(segment["false_negative_count"]) + int(
        bool(event.get("false_negative"))
    )


def exceptional_event_reasons(event: Mapping[str, object]) -> tuple[str, ...]:
    reasons: list[str] = []
    if bool(event.get("invalid_evaluation")):
        reasons.append("invalid_monitor_evaluation")
    if bool(event.get("false_negative")):
        reasons.append("false_negative")
    if bool(event.get("fallback_failure")):
        reasons.append("fallback_failure")
    if bool(event.get("terminal_state", False)):
        reasons.append("terminal_transition")
    if bool(event.get("fallback_executed")):
        predicted = _optional_float(event.get("predicted_fallback_speed_ratio"))
        realized = _optional_float(event.get("realized_executed_speed_ratio"))
        threshold = _optional_float(event.get("hazard_threshold"))
        if predicted is not None and realized is not None and threshold is not None:
            if (predicted > threshold) != (realized > threshold):
                reasons.append("fallback_classification_mismatch")
    return tuple(reasons)


class DecisionLogStream:
    def __init__(
        self,
        record_sink: Callable[[Mapping[str, object]], None],
        *,
        mode: str,
        is_formal_experiment: bool,
        maximum_records: int = MAX_COMPACT_RECORDS_PER_RUN,
        maximum_record_bytes: int = MAX_COMPACT_RECORD_BYTES,
    ) -> None:
        if mode not in VALID_DECISION_LOG_MODES:
            raise CompactDecisionLogError(f"unsupported decision log mode: {mode}")
        if is_formal_experiment and mode != LOG_MODE_COMPACT:
            raise CompactDecisionLogError("formal execution requires compact decision logging")
        self.record_sink = record_sink
        self.mode = mode
        self.is_formal_experiment = is_formal_experiment
        self.maximum_records = maximum_records
        self.maximum_record_bytes = maximum_record_bytes
        self.statistics = DecisionStreamStatistics()
        self.output_record_count = 0
        self.max_buffered_logical_events = 0
        self._segment: dict[str, object] | None = None
        self._segment_key: tuple[object, ...] | None = None
        self._last_segment_step: int | None = None
        self._closed = False

    @property
    def decision_stream_event_count(self) -> int:
        return self.statistics.event_count

    @property
    def decision_stream_sha256(self) -> str:
        return self.statistics.sha256

    @property
    def compact_decision_record_count(self) -> int:
        return self.output_record_count if self.mode == LOG_MODE_COMPACT else 0

    @property
    def buffered_logical_event_count(self) -> int:
        return int(self._segment is not None)

    def _write_compact_record(self, record: Mapping[str, object]) -> None:
        payload_size = len(canonical_json_bytes(record))
        if payload_size > self.maximum_record_bytes:
            raise CompactDecisionLogError(
                f"compact decision record exceeds {self.maximum_record_bytes} bytes"
            )
        if self.output_record_count >= self.maximum_records:
            raise CompactDecisionLogError(
                f"compact decision record limit exceeded: {self.maximum_records}"
            )
        self.record_sink(record)
        self.output_record_count += 1

    def _flush_segment(self) -> None:
        if self._segment is not None:
            self._write_compact_record(self._segment)
        self._segment = None
        self._segment_key = None
        self._last_segment_step = None

    def consume(self, event: Mapping[str, object]) -> None:
        if self._closed:
            raise CompactDecisionLogError("decision log stream is closed")
        if bool(event.get("is_formal_experiment")) != self.is_formal_experiment:
            raise CompactDecisionLogError("event formality does not match decision log stream")
        self.statistics.observe(event)
        if self.mode == LOG_MODE_FULL_TRACE:
            self.record_sink(event)
            self.output_record_count += 1
            return
        reasons = exceptional_event_reasons(event)
        if reasons:
            self._flush_segment()
            dedicated = dict(event)
            dedicated["event_kind"] = "decision_event"
            dedicated["exception_reasons"] = list(reasons)
            self._write_compact_record(dedicated)
            return
        key = _segment_key(event)
        step = int(event["step"])
        contiguous = self._last_segment_step is not None and step == self._last_segment_step + 1
        if self._segment is None or key != self._segment_key or not contiguous:
            self._flush_segment()
            self._segment = _new_segment(event)
            self._segment_key = key
        else:
            _extend_segment(self._segment, event)
        self._last_segment_step = step
        self.max_buffered_logical_events = max(self.max_buffered_logical_events, 1)

    def finish_run(self, terminal_record: Mapping[str, object] | None = None) -> None:
        if self._closed:
            return
        self._flush_segment()
        if terminal_record is not None:
            record = dict(terminal_record)
            record["event_kind"] = "terminal_transition"
            if self.mode == LOG_MODE_COMPACT:
                self._write_compact_record(record)
        self._closed = True

    def close(self) -> None:
        self.finish_run()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def infer_terminal_outcome(
    row: Mapping[str, object],
    *,
    maximum_steps: int = MAX_DECISION_STEPS,
) -> str:
    explicit = str(row.get("termination_reason", "")).strip()
    if explicit:
        return explicit
    if _as_bool(row.get("invalid_simulation", False)):
        return "invalid_simulation"
    if _as_bool(row.get("overspeed", False)):
        return "overspeed"
    if int(row.get("steps", 0) or 0) >= maximum_steps:
        return "max_steps"
    if _as_bool(row.get("final_simulator_success", False)):
        return "success"
    return str(row.get("terminal_label", "unknown")) or "unknown"


def terminal_transition_record(
    row: Mapping[str, object],
    *,
    maximum_steps: int = MAX_DECISION_STEPS,
) -> dict[str, object]:
    return {
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "experiment_id": row.get("experiment_id"),
        "run_id": row.get("run_id"),
        "paired_run_id": row.get("paired_run_id"),
        "case_id": row.get("case_id"),
        "subset_id": row.get("subset_id"),
        "arm_id": row.get("arm_id"),
        "monitor_id": row.get("monitor_id"),
        "step": int(row.get("steps", 0) or 0),
        "terminal_state": True,
        "terminal_label": row.get("terminal_label"),
        "termination_reason": infer_terminal_outcome(row, maximum_steps=maximum_steps),
        "is_formal_experiment": _as_bool(row.get("is_formal_experiment", False)),
    }


def _counter_matches_arm(
    statistics: DecisionStreamStatistics,
    row: Mapping[str, object],
) -> bool:
    monitor_evaluations = int(row.get("monitor_evaluation_count", 0) or 0)
    invalid_evaluations = int(row.get("invalid_monitor_evaluation_count", 0) or 0)
    return all(
        (
            statistics.event_count == monitor_evaluations + invalid_evaluations,
            statistics.allow_count == int(row.get("allow_count", 0) or 0),
            statistics.veto_count == int(row.get("veto_count", 0) or 0),
            statistics.fallback_count == int(row.get("fallback_count", 0) or 0),
            statistics.false_negative_count
            == int(row.get("false_negative_count", 0) or 0),
            statistics.fallback_failure_count
            == int(row.get("fallback_failure_count", 0) or 0),
            statistics.invalid_evaluation_count == invalid_evaluations,
            statistics.allow_count + statistics.veto_count == monitor_evaluations,
        )
    )


def _validate_full_trace_input_event(
    event: Mapping[str, object],
    *,
    line_number: int,
    arm_row: Mapping[str, object] | None,
) -> None:
    prefix = f"full-trace event at line {line_number}"
    missing = [field for field in DECISION_FIELDNAMES if field not in event]
    if missing:
        raise CompactDecisionLogError(f"{prefix} is missing fields: {missing}")
    if "event_kind" in event:
        raise CompactDecisionLogError(f"{prefix} is already a compact record")
    if event.get("is_formal_experiment") is not False:
        raise CompactDecisionLogError(
            f"{prefix} is not explicitly marked is_formal_experiment=false"
        )
    if event.get("decision_schema_version") != DECISION_SCHEMA_VERSION:
        raise CompactDecisionLogError(f"{prefix} has the wrong decision schema")
    if event.get("arm_id") != "monitor_on":
        raise CompactDecisionLogError(f"{prefix} is not a monitor_on decision")
    if event.get("hazard_threshold") != 1.90 or event.get("hazard_comparator") != ">":
        raise CompactDecisionLogError(f"{prefix} drifts from strict > 1.90")
    if arm_row is None:
        raise CompactDecisionLogError(f"{prefix} has no matching arm result row")
    expected_identity = {
        "experiment_id": arm_row.get("experiment_id"),
        "run_id": arm_row.get("run_id"),
        "paired_run_id": arm_row.get("paired_run_id"),
        "case_id": arm_row.get("case_id"),
        "subset_id": arm_row.get("subset_id"),
        "arm_id": arm_row.get("arm_id"),
        "monitor_id": arm_row.get("monitor_id"),
    }
    drifted = [
        field
        for field, expected in expected_identity.items()
        if str(event.get(field, "")) != str(expected or "")
    ]
    if drifted:
        raise CompactDecisionLogError(
            f"{prefix} disagrees with its arm identity fields: {drifted}"
        )
    if _as_bool(arm_row.get("is_formal_experiment", False)):
        raise CompactDecisionLogError(f"{prefix} matches a formal arm result")
    if not _as_bool(event.get("invalid_evaluation", False)):
        expected_mapping = {
            "continue": ("predicted_nominal_within_threshold", "allow", False),
            "veto_action": (
                "predicted_nominal_exceeds_overspeed_threshold",
                "veto",
                True,
            ),
        }
        decision_type = str(event.get("decision_type", ""))
        if decision_type not in expected_mapping:
            raise CompactDecisionLogError(f"{prefix} has an invalid decision type")
        expected_reason, expected_status, expected_fallback = expected_mapping[decision_type]
        if (
            event.get("decision_reason") != expected_reason
            or event.get("veto_reason") != expected_reason
            or event.get("veto_status") != expected_status
            or _as_bool(event.get("fallback_executed", False)) != expected_fallback
        ):
            raise CompactDecisionLogError(
                f"{prefix} has inconsistent decision reason, status, or fallback mapping"
            )


def convert_full_trace_to_compact(
    input_path: Path,
    output_path: Path,
    *,
    arm_results_path: Path | None = None,
) -> dict[str, object]:
    if arm_results_path is None:
        raise CompactDecisionLogError(
            "offline conversion requires arm results for identity and counter audit"
        )
    arm_rows = read_csv_rows(arm_results_path)
    arms_by_run = {str(row.get("run_id")): row for row in arm_rows}
    if len(arms_by_run) != len(arm_rows):
        raise CompactDecisionLogError("arm results contain duplicate run_id values")
    paired_arms: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in arm_rows:
        paired_arms.setdefault(str(row.get("paired_run_id")), {})[
            str(row.get("arm_id"))
        ] = row

    audit_statistics = DecisionStreamStatistics()
    unique_structures: set[tuple[object, ...]] = set()
    per_run_statistics: dict[str, DecisionStreamStatistics] = {}
    active_run_id: str | None = None
    active_stream: DecisionLogStream | None = None
    compact_record_count = 0
    compact_kind_counts = {
        "decision_segment": 0,
        "decision_event": 0,
        "terminal_transition": 0,
    }
    completed_run_ids: set[str] = set()
    last_input_step_by_run: dict[str, int] = {}
    expected_monitor_run_ids = {
        run_id
        for run_id, row in arms_by_run.items()
        if str(row.get("arm_id")) == "monitor_on"
    }
    counter_equality = False

    with JsonlEventWriter(output_path.parent, output_path.name) as writer:
        def write_compact_record(record: Mapping[str, object]) -> None:
            event_kind = str(record.get("event_kind", ""))
            if event_kind in compact_kind_counts:
                compact_kind_counts[event_kind] += 1
            writer.write_event(record)

        def finish_active() -> None:
            nonlocal active_stream, compact_record_count
            if active_stream is None or active_run_id is None:
                return
            row = arms_by_run.get(active_run_id)
            terminal = terminal_transition_record(row) if row is not None else None
            active_stream.finish_run(terminal)
            compact_record_count += active_stream.compact_decision_record_count
            per_run_statistics[active_run_id] = active_stream.statistics
            completed_run_ids.add(active_run_id)
            active_stream = None

        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CompactDecisionLogError(
                        f"logical decision at line {line_number} is not an object"
                    )
                run_id = str(value.get("run_id", ""))
                _validate_full_trace_input_event(
                    value,
                    line_number=line_number,
                    arm_row=arms_by_run.get(run_id),
                )
                try:
                    step = int(value.get("step", 0))
                except (TypeError, ValueError) as exc:
                    raise CompactDecisionLogError(
                        f"full-trace event at line {line_number} has an invalid step"
                    ) from exc
                expected_step = last_input_step_by_run.get(run_id, 0) + 1
                if step != expected_step:
                    raise CompactDecisionLogError(
                        f"full-trace event at line {line_number} is not step-contiguous"
                    )
                last_input_step_by_run[run_id] = step
                if active_run_id is not None and run_id != active_run_id:
                    finish_active()
                if active_stream is None and run_id in completed_run_ids:
                    raise CompactDecisionLogError(
                        f"input run is not contiguous in the logical stream: {run_id}"
                    )
                if active_stream is None:
                    active_run_id = run_id
                    active_stream = DecisionLogStream(
                        write_compact_record,
                        mode=LOG_MODE_COMPACT,
                        is_formal_experiment=False,
                    )
                audit_statistics.observe(value)
                unique_structures.add(_segment_key(value))
                active_stream.consume(value)
        finish_active()
        from scripts.check_final_veto_results import read_jsonl, validate_decision_events

        compact_validation_errors = validate_decision_events(
            read_jsonl(writer.staged_path_for_validation())
        )
        if compact_validation_errors:
            raise CompactDecisionLogError(
                "generated compact log failed validation: "
                + "; ".join(compact_validation_errors)
            )
        input_run_ids = set(per_run_statistics)
        counter_equality = (
            bool(input_run_ids)
            and input_run_ids == expected_monitor_run_ids
            and all(
                _counter_matches_arm(per_run_statistics[run_id], arms_by_run[run_id])
                for run_id in input_run_ids
            )
        )
        if not counter_equality:
            raise CompactDecisionLogError(
                "logical decision counters do not equal the supplied monitor_on arm rows"
            )

    terminal_transitions: list[str] = []
    for arms in paired_arms.values():
        if "monitor_off" in arms and "monitor_on" in arms:
            terminal_transitions.append(
                f"{infer_terminal_outcome(arms['monitor_off'])} -> "
                f"{infer_terminal_outcome(arms['monitor_on'])}"
            )
    original_size = input_path.stat().st_size
    compact_size = output_path.stat().st_size
    average = original_size / audit_statistics.event_count if audit_statistics.event_count else 0.0
    preservation_monitor_on_arms = 8
    stress_monitor_on_arms = 5
    estimated_unbounded_preservation = int(
        round(preservation_monitor_on_arms * MAX_DECISION_STEPS * average)
    )
    estimated_unbounded_stress = int(
        round(stress_monitor_on_arms * MAX_DECISION_STEPS * average)
    )
    estimated_unbounded_formal = (
        estimated_unbounded_preservation + estimated_unbounded_stress
    )
    return {
        "original_event_count": audit_statistics.event_count,
        "original_byte_size": original_size,
        "average_bytes_per_event": average,
        "compact_record_count": compact_record_count,
        "compact_segment_count": compact_kind_counts["decision_segment"],
        "compact_dedicated_event_count": compact_kind_counts["decision_event"],
        "compact_terminal_record_count": compact_kind_counts["terminal_transition"],
        "compact_byte_size": compact_size,
        "compression_ratio": original_size / compact_size if compact_size else None,
        "full_stream_sha256": audit_statistics.sha256,
        "allow_event_count": audit_statistics.allow_count,
        "veto_event_count": audit_statistics.veto_count,
        "fallback_event_count": audit_statistics.fallback_count,
        "invalid_evaluation_count": audit_statistics.invalid_evaluation_count,
        "false_negative_count": audit_statistics.false_negative_count,
        "fallback_failure_count": audit_statistics.fallback_failure_count,
        "decision_state_transition_count": audit_statistics.decision_state_transition_count,
        "phase_change_count": audit_statistics.phase_change_count,
        "active_stage_change_count": audit_statistics.active_stage_change_count,
        "unique_repeated_event_structures": len(unique_structures),
        "first_veto_step": audit_statistics.first_veto_step,
        "last_veto_step": audit_statistics.last_veto_step,
        "longest_consecutive_veto_steps": audit_statistics.longest_consecutive_veto_steps,
        "longest_consecutive_allow_steps": audit_statistics.longest_consecutive_allow_steps,
        "veto_segment_count": audit_statistics.veto_segment_count,
        "allow_segment_count": audit_statistics.allow_segment_count,
        "intervention_rate": audit_statistics.intervention_rate,
        "allow_rate": audit_statistics.allow_rate,
        "fallback_rate": audit_statistics.fallback_rate,
        "counter_equality": counter_equality,
        "terminal_outcome_transitions": terminal_transitions,
        "worst_case_preservation_monitor_on_arms": preservation_monitor_on_arms,
        "worst_case_stress_monitor_on_arms": stress_monitor_on_arms,
        "worst_case_formal_monitor_on_arms": (
            preservation_monitor_on_arms + stress_monitor_on_arms
        ),
        "worst_case_formal_logical_events": (
            preservation_monitor_on_arms + stress_monitor_on_arms
        )
        * MAX_DECISION_STEPS,
        "estimated_unbounded_preservation_byte_size": estimated_unbounded_preservation,
        "estimated_unbounded_stress_byte_size": estimated_unbounded_stress,
        "estimated_unbounded_formal_byte_size": estimated_unbounded_formal,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a nonformal full Final Veto decision trace to compact JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--arm-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = convert_full_trace_to_compact(
            args.input,
            args.output,
            arm_results_path=args.arm_results,
        )
    except (CompactDecisionLogError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2))
    return 0 if bool(report["counter_equality"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CompactDecisionLogError",
    "CompactLoggingPlanEstimate",
    "DecisionLogStream",
    "DecisionStreamStatistics",
    "FORMAL_DEFAULT_DECISION_LOG_MODE",
    "FORMAL_PUBLIC_LOG_BYTE_BUDGET",
    "LOG_MODE_COMPACT",
    "LOG_MODE_FULL_TRACE",
    "MAX_COMPACT_RECORD_BYTES",
    "MAX_COMPACT_RECORDS_PER_RUN",
    "MAX_DECISION_STEPS",
    "VALID_DECISION_LOG_MODES",
    "canonical_json_bytes",
    "compact_logging_preflight_errors",
    "convert_full_trace_to_compact",
    "dry_run_expected_compact_bytes_per_monitor_arm",
    "estimate_compact_logging_plan",
    "exceptional_event_reasons",
    "infer_terminal_outcome",
    "logging_configuration_errors",
    "terminal_transition_record",
    "update_canonical_stream_digest",
]

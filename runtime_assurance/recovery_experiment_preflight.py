from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Sequence

import runtime_assurance.recovery_branch_executor as recovery_branch_executor

from runtime_assurance.recovery_branch_executor import (
    ACTION_MAGNITUDE,
    SUPPORTED_BRANCH_IDS,
    generate_explicit_abort,
    generate_tangential_correction_action,
    generate_velocity_opposed_action,
    generate_zero_action,
)
from runtime_assurance.recovery_evaluators import (
    EVALUATION_INVALID,
    EVALUATION_NOT_EVALUATED,
    FROZEN_RECOVERY_HORIZON,
    INSTABILITY_EVALUATOR_ID,
    PHASE34_RECOVERABILITY_EVALUATOR_ID,
    RECOVERY_SUCCESS_EVALUATOR_ID,
    UNSAFE_STATE_EVALUATOR_ID,
    evaluate_instability,
    evaluate_phase34_compatible_recoverability,
    evaluate_recovery_success_v0,
    evaluate_unsafe_state,
)
from runtime_assurance.recovery_experiment_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    PUBLISHED_ARTIFACT_FILENAMES,
    RecoveryArtifactContract,
    RecoveryArtifactError,
    publish_recovery_experiment_bundle,
    validate_publication_target,
    validate_recovery_experiment_bundle,
)
from runtime_assurance.recovery_stop_conditions import (
    FROZEN_TERMINATION_PRIORITY,
)
from scripts.check_recovery_action_branching_manifest import (
    EXPECTED_BRANCH_IDS,
    EXPECTED_EXPERIMENT_ID,
    EXPECTED_SCHEMA_VERSION,
    ManifestValidationError,
    validate_manifest_data,
)
from scripts.check_recovery_branch_state import (
    BranchStateValidationError,
    canonical_json_bytes,
    canonical_sha256,
    validate_branch_state_data,
)
from simulator.phase34_35_transition import CartesianState2D


PREFLIGHT_SCHEMA_VERSION = "recovery_experiment_preflight_v0"
MANIFEST_RELATIVE_PATH = Path(
    "analysis/recovery_action_branching_nonformal_v0/manifest.json"
)
BRANCH_STATE_RELATIVE_PATH = Path(
    "analysis/recovery_action_branching_nonformal_v0/branch_state.json"
)

# These hashes pin the exact frozen repository inputs. They are canonical JSON
# hashes, not byte hashes, and are deliberately separate from implementation HEAD.
FROZEN_MANIFEST_CANONICAL_HASH = (
    "e9cb96eae714bc0d8ed66d1a85f29baed2819d0d425a3ce9742b7e77ac236bad"
)
FROZEN_BRANCH_STATE_CANONICAL_HASH = (
    "8b017254a8db2584a6732bcd086447ba405cf949d9e932cf03e71543b2cdb898"
)

EXPECTED_RUNTIME_STOP_PRIORITY = (
    "invalid_simulation",
    "invalid_recovery_evaluation",
    "overspeed",
    "instability",
    "unsafe_state",
    "action_rejected",
    "explicit_abort",
    "recovery_success",
    "recovery_horizon_exhausted",
    "total_horizon_exhausted",
)

_MANIFEST_STOP_TO_RUNTIME = {
    "realized_overspeed": "overspeed",
    "recovery_action_rejection": "action_rejected",
    "recovery_horizon_exhaustion": "recovery_horizon_exhausted",
    "total_horizon_exhaustion": "total_horizon_exhausted",
}

_PROTECTED_HASH_GROUPS = {
    "historical_phase34_37": (
        "analysis/phase34_post_cross_sync",
        "analysis/phase35_crossing_basin_expansion",
        "analysis/phase36b_transfer_family_benchmark",
        "analysis/phase36c_non_crossing_geometry_diagnosis",
        "analysis/phase37a_radial_commit_timing",
        "analysis/phase37b_weak_tangential_subset",
        "scripts/check_phase_results.py",
    ),
    "final_veto_v0": ("analysis/final_veto_ablation_v0",),
    "frozen_recovery_inputs": (
        "analysis/recovery_action_branching_nonformal_v0/manifest.json",
        "analysis/recovery_action_branching_nonformal_v0/branch_state.json",
    ),
}


@dataclass(frozen=True, slots=True)
class RecoveryExperimentPreflightCheck:
    check_id: str
    status: str
    required: bool
    message: str
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed", "not_applicable"}:
            raise ValueError(f"unsupported preflight status: {self.status!r}")
        if not self.check_id or not self.message:
            raise ValueError("preflight checks require an ID and message")
        keys = tuple(key for key, _ in self.details)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("preflight check details must be uniquely key-sorted")


@dataclass(frozen=True, slots=True)
class RecoveryExperimentPreflightReport:
    ready: bool
    checks: tuple[RecoveryExperimentPreflightCheck, ...]
    failed_check_ids: tuple[str, ...]
    manifest_hash: str
    branch_state_hash: str
    implementation_commit: str
    working_tree_clean: bool
    protected_evidence_hashes: tuple[tuple[str, str], ...]
    preflight_schema_version: str = PREFLIGHT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class RecoveryRepositoryState:
    is_git_worktree: bool
    head_commit: str
    tracked_clean: bool
    staged_clean: bool
    untracked_paths: tuple[str, ...] = ()


class RecoveryExperimentPreflightError(ValueError):
    pass


def find_repository_root(start: Path | None = None) -> Path:
    candidate = (start or Path(__file__)).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists() and (directory / "runtime_assurance").is_dir():
            return directory
    raise FileNotFoundError("could not locate repository root")


def load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryExperimentPreflightError(
            f"invalid UTF-8 JSON at {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RecoveryExperimentPreflightError(f"JSON root must be an object: {path}")
    return value


def _git(
    repository_root: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository_root.resolve().as_posix()}",
            *arguments,
        ],
        cwd=repository_root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=check,
    )


def inspect_repository_state(repository_root: Path) -> RecoveryRepositoryState:
    root = repository_root.resolve()
    worktree = _git(root, "rev-parse", "--is-inside-work-tree", check=False)
    if worktree.returncode != 0 or worktree.stdout.strip() != "true":
        return RecoveryRepositoryState(False, "", False, False, ())
    head = _git(root, "rev-parse", "HEAD", check=False)
    tracked = _git(root, "diff", "--quiet", check=False)
    staged = _git(root, "diff", "--cached", "--quiet", check=False)
    untracked = _git(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        check=False,
    )
    return RecoveryRepositoryState(
        is_git_worktree=True,
        head_commit=head.stdout.strip() if head.returncode == 0 else "",
        tracked_clean=tracked.returncode == 0,
        staged_clean=staged.returncode == 0,
        untracked_paths=tuple(
            sorted(line.strip() for line in untracked.stdout.splitlines() if line.strip())
        ),
    )


def _aggregate_paths_hash(repository_root: Path, relative_paths: Sequence[str]) -> str:
    root = repository_root.resolve()
    files: list[Path] = []
    for relative in relative_paths:
        path = (root / PurePosixPath(relative)).resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(child for child in path.rglob("*") if child.is_file())
    lines: list[str] = []
    for path in sorted(files, key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{relative}|{digest}")
    return hashlib.sha256(os.linesep.join(lines).encode("utf-8")).hexdigest()


def protected_evidence_hashes(repository_root: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, _aggregate_paths_hash(repository_root, paths))
        for name, paths in _PROTECTED_HASH_GROUPS.items()
    )


def _manifest_branch_ids(manifest: Mapping[str, object]) -> tuple[str, ...]:
    branches = manifest.get("branches")
    if not isinstance(branches, list):
        return ()
    result: list[str] = []
    for branch in branches:
        if isinstance(branch, dict) and isinstance(branch.get("branch_id"), str):
            result.append(branch["branch_id"])
    return tuple(result)


def _manifest_result_paths(manifest: Mapping[str, object]) -> tuple[str, ...]:
    output = manifest.get("output_contract")
    if not isinstance(output, dict) or not isinstance(output.get("future_artifacts"), list):
        return ()
    paths: list[str] = []
    for artifact in output["future_artifacts"]:
        if not isinstance(artifact, dict):
            continue
        path = artifact.get("path")
        if isinstance(path, str) and Path(path).name != "branch_state.json":
            paths.append(path)
    return tuple(paths)


def _normalize_manifest_stop_priority(manifest: Mapping[str, object]) -> tuple[str, ...]:
    stop_conditions = manifest.get("stop_conditions")
    if not isinstance(stop_conditions, dict):
        return ()
    priority = stop_conditions.get("priority_order")
    if not isinstance(priority, list) or not all(isinstance(item, str) for item in priority):
        return ()
    return tuple(_MANIFEST_STOP_TO_RUNTIME.get(item, item) for item in priority)


def build_artifact_contract(
    manifest: Mapping[str, object],
    branch_state: Mapping[str, object],
    *,
    implementation_commit: str,
) -> RecoveryArtifactContract:
    source_case = manifest.get("source_case")
    horizons = manifest.get("horizons")
    hazard = manifest.get("hazard")
    output = manifest.get("output_contract")
    ordering = branch_state.get("branch_ordering")
    if not all(
        isinstance(value, dict)
        for value in (source_case, horizons, hazard, output, ordering)
    ):
        raise RecoveryExperimentPreflightError(
            "manifest or branch state lacks a required contract object"
        )
    result_paths = _manifest_result_paths(manifest)
    output_filenames = tuple(Path(path).name for path in result_paths)
    protected = manifest.get("protected_paths")
    prohibited = manifest.get("prohibited_claims")
    if not isinstance(protected, list) or not all(isinstance(item, str) for item in protected):
        raise RecoveryExperimentPreflightError("manifest protected_paths are invalid")
    if not isinstance(prohibited, list) or not all(isinstance(item, str) for item in prohibited):
        raise RecoveryExperimentPreflightError("manifest prohibited_claims are invalid")
    return RecoveryArtifactContract(
        experiment_id=str(manifest.get("experiment_id")),
        case_id=str(source_case.get("case_id")),
        seed=int(source_case.get("seed")),
        branch_ids=_manifest_branch_ids(manifest),
        branch_state_hash=str(branch_state.get("canonical_branch_state_hash")),
        manifest_hash=canonical_sha256(dict(manifest)),
        implementation_commit=implementation_commit,
        case_configuration_hash=str(branch_state.get("case_configuration_hash")),
        simulator_configuration_hash=str(
            branch_state.get("simulator_configuration_hash")
        ),
        simulator_constants_hash=str(branch_state.get("simulator_constants_hash")),
        branch_step=int(branch_state.get("branch_step")),
        nominal_prefix_transition_count=int(ordering.get("realized_prefix_transition_count")),
        recovery_horizon=int(horizons.get("recovery_horizon_realized_transitions")),
        total_horizon=int(horizons.get("total_episode_horizon_realized_transitions")),
        hazard_threshold=float(hazard.get("threshold")),
        hazard_comparator=str(hazard.get("comparator")),
        stop_priority=_normalize_manifest_stop_priority(manifest),
        output_directory=str(output.get("base_directory")),
        output_filenames=output_filenames,
        protected_paths=tuple(protected),
        prohibited_claims=tuple(prohibited),
    )


def _details(**values: object) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in values.items()))


def _passed(
    check_id: str, message: str, *, required: bool = True, **details: object
) -> RecoveryExperimentPreflightCheck:
    return RecoveryExperimentPreflightCheck(
        check_id, "passed", required, message, _details(**details)
    )


def _failed(
    check_id: str, message: str, *, required: bool = True, **details: object
) -> RecoveryExperimentPreflightCheck:
    return RecoveryExperimentPreflightCheck(
        check_id, "failed", required, message, _details(**details)
    )


def _not_applicable(
    check_id: str, message: str, **details: object
) -> RecoveryExperimentPreflightCheck:
    return RecoveryExperimentPreflightCheck(
        check_id, "not_applicable", False, message, _details(**details)
    )


def _check_call(
    check_id: str,
    message: str,
    operation: Callable[[], object],
) -> RecoveryExperimentPreflightCheck:
    try:
        operation()
    except Exception as exc:
        return _failed(check_id, f"{message}: {exc}")
    return _passed(check_id, message)


def _state_from_branch_document(branch_state: Mapping[str, object]) -> CartesianState2D:
    state = branch_state.get("state")
    if not isinstance(state, dict):
        raise RecoveryExperimentPreflightError("branch state lacks named state")
    values = (
        state.get("position_x"),
        state.get("position_y"),
        state.get("velocity_x"),
        state.get("velocity_y"),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise RecoveryExperimentPreflightError("branch state is not finite")
    return CartesianState2D(*(float(value) for value in values))


def _check_branch_configuration_hashes(branch_state: Mapping[str, object]) -> None:
    case_configuration = branch_state.get("case_configuration")
    simulator_configuration = branch_state.get("simulator_configuration")
    if not isinstance(case_configuration, dict) or not isinstance(
        simulator_configuration, dict
    ):
        raise RecoveryExperimentPreflightError(
            "branch state lacks case or simulator configuration"
        )
    simulator_constants = simulator_configuration.get("simulator_constants")
    if not isinstance(simulator_constants, dict):
        raise RecoveryExperimentPreflightError("branch state lacks simulator constants")
    expected = (
        ("case_configuration_hash", case_configuration),
        ("simulator_configuration_hash", simulator_configuration),
        ("simulator_constants_hash", simulator_constants),
    )
    for field, payload in expected:
        if branch_state.get(field) != canonical_sha256(payload):
            raise RecoveryExperimentPreflightError(f"{field} does not recompute")


def _check_action_implementations(
    branch_state: Mapping[str, object], expected_branch_ids: Sequence[str]
) -> None:
    if SUPPORTED_BRANCH_IDS != frozenset(expected_branch_ids):
        raise RecoveryExperimentPreflightError("executor branch set has drifted")
    if generate_zero_action() != (0.0, 0.0):
        raise RecoveryExperimentPreflightError("zero-action implementation drifted")
    state = _state_from_branch_document(branch_state)
    velocity_action = generate_velocity_opposed_action(state)
    if not math.isclose(math.hypot(*velocity_action), ACTION_MAGNITUDE, rel_tol=0.0, abs_tol=1e-15):
        raise RecoveryExperimentPreflightError("velocity-opposed magnitude drifted")
    simulator = branch_state.get("simulator_configuration")
    if not isinstance(simulator, dict) or not isinstance(simulator.get("simulator_constants"), dict):
        raise RecoveryExperimentPreflightError("simulator constants are unavailable")
    target_speed = float(simulator["simulator_constants"].get("target_circular_speed"))
    tangential_action = generate_tangential_correction_action(state, target_speed)
    if not math.isclose(math.hypot(*tangential_action), ACTION_MAGNITUDE, rel_tol=0.0, abs_tol=1e-15):
        raise RecoveryExperimentPreflightError("tangential-correction magnitude drifted")
    if generate_explicit_abort() is not None:
        raise RecoveryExperimentPreflightError("explicit abort must execute no action")


def _check_evaluator_semantics() -> None:
    if evaluate_instability(instability_flag=None).status != EVALUATION_NOT_EVALUATED:
        raise RecoveryExperimentPreflightError("missing instability evidence is optimistic")
    if evaluate_unsafe_state(unsafe_state_flag=None).status != EVALUATION_NOT_EVALUATED:
        raise RecoveryExperimentPreflightError("missing unsafe-state evidence is optimistic")
    missing_recovery = evaluate_recovery_success_v0(
        declared_overspeed_occurred=None,
        simulation_valid=None,
        recovery_evaluation_valid=None,
        target_radius_crossing=None,
        phase34_compatible_recoverable_crossing=None,
        branch_transition_count=None,
        recovery_horizon=None,
        branch_step=None,
        crossing_step=None,
        explicit_abort=None,
        action_rejected=None,
    )
    if missing_recovery.status != EVALUATION_NOT_EVALUATED:
        raise RecoveryExperimentPreflightError("missing recovery evidence is optimistic")
    if evaluate_instability(instability_flag=float("nan")).status != EVALUATION_INVALID:
        raise RecoveryExperimentPreflightError("malformed instability evidence is not invalid")
    if evaluate_unsafe_state(unsafe_state_flag="false").status != EVALUATION_INVALID:
        raise RecoveryExperimentPreflightError("malformed unsafe-state evidence is not invalid")
    malformed_recovery = evaluate_recovery_success_v0(
        declared_overspeed_occurred=False,
        simulation_valid=True,
        recovery_evaluation_valid=True,
        target_radius_crossing=False,
        phase34_compatible_recoverable_crossing=False,
        branch_transition_count=float("nan"),
        recovery_horizon=FROZEN_RECOVERY_HORIZON,
        branch_step=28,
        crossing_step=None,
        explicit_abort=False,
        action_rejected=False,
    )
    if malformed_recovery.status != EVALUATION_INVALID:
        raise RecoveryExperimentPreflightError("malformed recovery evidence is not invalid")
    phase34_missing = evaluate_phase34_compatible_recoverability(
        r_error_ratio=None,
        vr_ratio=None,
        vt_error_ratio=None,
    )
    if phase34_missing.status != EVALUATION_NOT_EVALUATED:
        raise RecoveryExperimentPreflightError("missing recoverability evidence is optimistic")


def _check_output_collisions(
    repository_root: Path, contract: RecoveryArtifactContract
) -> None:
    output = (repository_root / PurePosixPath(contract.output_directory)).resolve()
    collisions = [name for name in contract.output_filenames if (output / name).exists()]
    if collisions:
        raise RecoveryExperimentPreflightError(
            f"reserved experiment outputs already exist: {collisions}"
        )


def _check_untracked_collisions(
    repository_state: RecoveryRepositoryState,
    contract: RecoveryArtifactContract,
) -> None:
    expected = {
        str(PurePosixPath(contract.output_directory) / filename)
        for filename in contract.output_filenames
    }
    collisions = sorted(expected.intersection(repository_state.untracked_paths))
    if collisions:
        raise RecoveryExperimentPreflightError(
            f"untracked paths collide with reserved outputs: {collisions}"
        )


def _check_protected_target_refusal(
    repository_root: Path, contract: RecoveryArtifactContract
) -> None:
    for protected in contract.protected_paths:
        path = repository_root / PurePosixPath(protected.rstrip("/"))
        try:
            validate_publication_target(
                path,
                repository_root=repository_root,
                contract=contract,
                synthetic=False,
            )
        except RecoveryArtifactError:
            continue
        raise RecoveryExperimentPreflightError(
            f"artifact writer accepted protected path: {protected}"
        )


def run_recovery_experiment_preflight(
    *,
    repository_root: Path | None = None,
    manifest_data: Mapping[str, object] | None = None,
    branch_state_data: Mapping[str, object] | None = None,
    repository_state: RecoveryRepositoryState | None = None,
    require_clean_repository: bool,
    expected_manifest_hash: str = FROZEN_MANIFEST_CANONICAL_HASH,
    expected_branch_state_hash: str = FROZEN_BRANCH_STATE_CANONICAL_HASH,
    expected_runtime_stop_priority: tuple[str, ...] = EXPECTED_RUNTIME_STOP_PRIORITY,
    available_evaluator_ids: frozenset[str] | None = None,
) -> tuple[RecoveryExperimentPreflightReport, RecoveryArtifactContract | None]:
    root = (repository_root or find_repository_root()).resolve()
    manifest_path = root / MANIFEST_RELATIVE_PATH
    branch_state_path = root / BRANCH_STATE_RELATIVE_PATH
    try:
        manifest = copy.deepcopy(
            dict(manifest_data) if manifest_data is not None else load_json_object(manifest_path)
        )
    except Exception:
        manifest = {}
    try:
        branch_state = copy.deepcopy(
            dict(branch_state_data)
            if branch_state_data is not None
            else load_json_object(branch_state_path)
        )
    except Exception:
        branch_state = {}
    repo = repository_state or inspect_repository_state(root)
    manifest_hash = canonical_sha256(manifest) if manifest else ""
    branch_state_hash = str(branch_state.get("canonical_branch_state_hash", ""))
    evidence_hashes = protected_evidence_hashes(root)
    checks: list[RecoveryExperimentPreflightCheck] = []

    checks.append(
        _passed("git_worktree", "execution is inside a Git worktree")
        if repo.is_git_worktree
        else _failed("git_worktree", "execution is not inside a Git worktree")
    )
    head_valid = len(repo.head_commit) == 40 and all(
        character in "0123456789abcdef" for character in repo.head_commit
    )
    checks.append(
        _passed("committed_head", "HEAD is a full commit", head=repo.head_commit)
        if head_valid
        else _failed("committed_head", "HEAD is not a full commit", head=repo.head_commit)
    )
    if require_clean_repository:
        checks.append(
            _passed("tracked_tree_clean", "tracked working tree is clean")
            if repo.tracked_clean
            else _failed("tracked_tree_clean", "tracked working tree has changes")
        )
        checks.append(
            _passed("staging_area_clean", "staging area is clean")
            if repo.staged_clean
            else _failed("staging_area_clean", "staged changes are present")
        )
    else:
        checks.extend(
            (
                _not_applicable(
                    "tracked_tree_clean",
                    "tracked cleanliness is required only by experiment-preflight mode",
                ),
                _not_applicable(
                    "staging_area_clean",
                    "staging cleanliness is required only by experiment-preflight mode",
                ),
            )
        )

    checks.append(
        _check_call(
            "manifest_contract",
            "frozen recovery manifest passes its repository validator",
            lambda: validate_manifest_data(manifest),
        )
    )
    branch_selection_api_absent = not any(
        hasattr(recovery_branch_executor, name)
        for name in (
            "optimize_recovery_action",
            "rank_recovery_branches",
            "select_recovery_branch",
        )
    )
    checks.append(
        _passed(
            "branch_selection_absent",
            "executor exposes no branch ranking, selection, or optimization API",
        )
        if branch_selection_api_absent
        else _failed(
            "branch_selection_absent",
            "executor exposes undeclared branch selection or optimization",
        )
    )
    checks.append(
        _passed("manifest_hash", "canonical manifest hash matches frozen input", hash=manifest_hash)
        if manifest_hash == expected_manifest_hash
        else _failed(
            "manifest_hash",
            "canonical manifest hash differs from frozen input",
            actual=manifest_hash,
            expected=expected_manifest_hash,
        )
    )
    identity_ok = (
        manifest.get("manifest_schema_version") == EXPECTED_SCHEMA_VERSION
        and manifest.get("experiment_id") == EXPECTED_EXPERIMENT_ID
        and manifest.get("experiment_status") == "design_frozen_not_run"
        and manifest.get("is_formal_experiment") is False
    )
    checks.append(
        _passed("manifest_identity", "manifest identity and nonformal freeze status match")
        if identity_ok
        else _failed("manifest_identity", "manifest identity or status drifted")
    )

    try:
        contract = build_artifact_contract(
            manifest,
            branch_state,
            implementation_commit=repo.head_commit,
        )
    except Exception as exc:
        contract = None
        checks.append(_failed("artifact_contract_build", f"artifact contract cannot be built: {exc}"))
    else:
        checks.append(_passed("artifact_contract_build", "artifact contract is derived from frozen inputs"))

    checks.append(
        _check_call(
            "branch_state_contract",
            "frozen branch state passes its repository validator",
            lambda: validate_branch_state_data(branch_state),
        )
    )
    checks.append(
        _passed(
            "branch_state_hash",
            "canonical branch-state hash matches frozen input",
            hash=branch_state_hash,
        )
        if branch_state_hash == expected_branch_state_hash
        else _failed(
            "branch_state_hash",
            "canonical branch-state hash differs from frozen input",
            actual=branch_state_hash,
            expected=expected_branch_state_hash,
        )
    )
    try:
        recomputed_branch = copy.deepcopy(branch_state)
        supplied_branch_hash = recomputed_branch.pop("canonical_branch_state_hash")
        branch_hash_internal = supplied_branch_hash == canonical_sha256(recomputed_branch)
    except Exception:
        branch_hash_internal = False
    checks.append(
        _passed("branch_state_hash_recomputation", "stored branch-state hash recomputes")
        if branch_hash_internal
        else _failed("branch_state_hash_recomputation", "stored branch-state hash does not recompute")
    )

    source_case = manifest.get("source_case") if isinstance(manifest.get("source_case"), dict) else {}
    source_ok = (
        branch_state.get("case_id") == source_case.get("case_id")
        and branch_state.get("seed") == source_case.get("seed")
    )
    checks.append(
        _passed("source_case", "manifest and branch state use the same frozen case and seed")
        if source_ok
        else _failed("source_case", "manifest and branch state source case differ")
    )
    ordering = branch_state.get("branch_ordering") if isinstance(branch_state.get("branch_ordering"), dict) else {}
    step = branch_state.get("branch_step")
    ordering_ok = (
        isinstance(step, int)
        and not isinstance(step, bool)
        and ordering.get("realized_prefix_transition_count") == step - 1
        and ordering.get("before_nominal_action_execution") is True
        and ordering.get("before_final_veto_fallback_execution") is True
        and ordering.get("nominal_action_executed") is False
        and ordering.get("final_veto_fallback_executed") is False
    )
    checks.append(
        _passed("branch_boundary", "branch point precedes nominal and fallback execution")
        if ordering_ok
        else _failed("branch_boundary", "branch ordering or prefix count is inconsistent")
    )
    trigger_ok = (
        isinstance(branch_state.get("predicted_speed_ratio"), (int, float))
        and not isinstance(branch_state.get("predicted_speed_ratio"), bool)
        and math.isfinite(float(branch_state.get("predicted_speed_ratio")))
        and float(branch_state.get("predicted_speed_ratio")) > 1.90
        and branch_state.get("threshold") == 1.90
        and branch_state.get("comparator") == ">"
    )
    checks.append(
        _passed("branch_trigger", "branch prediction remains a strict > 1.90 veto")
        if trigger_ok
        else _failed("branch_trigger", "branch prediction no longer satisfies strict > 1.90")
    )
    checks.append(
        _check_call(
            "branch_state_resumable",
            "branch state is finite and resumable",
            lambda: _state_from_branch_document(branch_state),
        )
    )
    checks.append(
        _check_call(
            "branch_configuration_hashes",
            "case, simulator, and constants hashes recompute",
            lambda: _check_branch_configuration_hashes(branch_state),
        )
    )

    branch_ids = _manifest_branch_ids(manifest)
    expected_ids = frozenset(EXPECTED_BRANCH_IDS)
    checks.append(
        _passed("four_branch_manifest", "manifest contains exactly four unique frozen branches")
        if set(branch_ids) == expected_ids and len(branch_ids) == len(expected_ids)
        else _failed(
            "four_branch_manifest",
            "manifest branch set or order drifted",
            actual=branch_ids,
            expected=sorted(expected_ids),
        )
    )
    checks.append(
        _passed("four_branch_executor", "executor supports exactly the frozen branch set")
        if SUPPORTED_BRANCH_IDS == expected_ids
        else _failed("four_branch_executor", "executor branch support drifted")
    )
    checks.append(
        _check_call(
            "branch_action_implementations",
            "pure branch actions match the frozen definitions",
            lambda: _check_action_implementations(branch_state, branch_ids),
        )
    )
    rejection = manifest.get("recovery_action_rejection") if isinstance(manifest.get("recovery_action_rejection"), dict) else {}
    rejection_ok = (
        rejection.get("execute_rejected_action") is False
        and rejection.get("substitute_zero_action") is False
        and rejection.get("recursive_branch_selection") is False
        and rejection.get("terminate_current_branch") is True
    )
    checks.append(
        _passed("action_rejection", "rejected recovery actions terminate without fallback or recursion")
        if rejection_ok
        else _failed("action_rejection", "recovery-action rejection handling drifted")
    )

    horizons = manifest.get("horizons") if isinstance(manifest.get("horizons"), dict) else {}
    horizon_ok = (
        horizons.get("recovery_horizon_realized_transitions") == 10_000
        and horizons.get("total_episode_horizon_realized_transitions") == 100_000
    )
    checks.append(
        _passed("scientific_horizons", "recovery and total horizons remain 10000 and 100000")
        if horizon_ok
        else _failed("scientific_horizons", "frozen scientific horizons drifted")
    )
    hazard = manifest.get("hazard") if isinstance(manifest.get("hazard"), dict) else {}
    hazard_ok = (
        hazard.get("threshold") == 1.90
        and hazard.get("comparator") == ">"
        and hazard.get("prediction_horizon_steps") == 1
    )
    checks.append(
        _passed("hazard_contract", "hazard remains one-step strict > 1.90")
        if hazard_ok
        else _failed("hazard_contract", "hazard threshold, comparator, or horizon drifted")
    )
    normalized_priority = _normalize_manifest_stop_priority(manifest)
    checks.append(
        _passed("manifest_stop_priority", "manifest stop priority normalizes to frozen runtime labels")
        if normalized_priority == expected_runtime_stop_priority
        else _failed(
            "manifest_stop_priority",
            "manifest stop priority drifted",
            actual=normalized_priority,
            expected=expected_runtime_stop_priority,
        )
    )
    checks.append(
        _passed("runtime_stop_priority", "runtime stop priority matches the frozen order")
        if FROZEN_TERMINATION_PRIORITY == expected_runtime_stop_priority
        else _failed(
            "runtime_stop_priority",
            "runtime stop priority drifted",
            actual=FROZEN_TERMINATION_PRIORITY,
            expected=expected_runtime_stop_priority,
        )
    )

    required_evaluators = frozenset(
        {
            RECOVERY_SUCCESS_EVALUATOR_ID,
            PHASE34_RECOVERABILITY_EVALUATOR_ID,
            INSTABILITY_EVALUATOR_ID,
            UNSAFE_STATE_EVALUATOR_ID,
        }
    )
    evaluator_ids = (
        required_evaluators
        if available_evaluator_ids is None
        else available_evaluator_ids
    )
    checks.append(
        _passed("evaluator_imports", "all required evaluator IDs are available")
        if evaluator_ids == required_evaluators
        else _failed(
            "evaluator_imports",
            "required evaluator set is unavailable or contains drift",
            actual=sorted(evaluator_ids),
            expected=sorted(required_evaluators),
        )
    )
    checks.append(
        _passed("evaluator_horizon", "Recovery Success v0 horizon remains 10000")
        if FROZEN_RECOVERY_HORIZON == 10_000
        else _failed("evaluator_horizon", "Recovery Success v0 horizon drifted")
    )
    checks.append(
        _check_call(
            "evaluator_missing_invalid_semantics",
            "missing evidence remains not_evaluated and malformed evidence remains invalid",
            _check_evaluator_semantics,
        )
    )

    if contract is not None:
        expected_filenames = PUBLISHED_ARTIFACT_FILENAMES
        checks.append(
            _passed("reserved_artifacts", "reserved result bundle filenames match the manifest")
            if contract.output_filenames == expected_filenames
            else _failed(
                "reserved_artifacts",
                "reserved result bundle filenames drifted",
                actual=contract.output_filenames,
                expected=expected_filenames,
            )
        )
        checks.append(
            _check_call(
                "output_location",
                "frozen recovery output directory does not overlap protected evidence",
                lambda: validate_publication_target(
                    root / PurePosixPath(contract.output_directory),
                    repository_root=root,
                    contract=contract,
                    synthetic=False,
                ),
            )
        )
        checks.append(
            _check_call(
                "output_collisions",
                "no recovery result artifact already exists",
                lambda: _check_output_collisions(root, contract),
            )
        )
        checks.append(
            _check_call(
                "untracked_output_collisions",
                "untracked files do not collide with reserved outputs",
                lambda: _check_untracked_collisions(repo, contract),
            )
        )
        checks.append(
            _check_call(
                "protected_output_refusal",
                "artifact publication rejects all manifest-protected locations",
                lambda: _check_protected_target_refusal(root, contract),
            )
        )
        artifact_api_ready = callable(validate_recovery_experiment_bundle) and callable(
            publish_recovery_experiment_bundle
        )
        checks.append(
            _passed(
                "artifact_publication_boundary",
                "four-branch validation and staged publication APIs are available",
                schema=ARTIFACT_SCHEMA_VERSION,
            )
            if artifact_api_ready
            else _failed("artifact_publication_boundary", "artifact publication APIs are unavailable")
        )

    checks.append(
        _passed(
            "protected_evidence_inventory",
            "protected evidence hashes were recorded read-only",
            groups=len(evidence_hashes),
        )
    )

    failed = tuple(
        check.check_id
        for check in checks
        if check.required and check.status != "passed"
    )
    report = RecoveryExperimentPreflightReport(
        ready=not failed,
        checks=tuple(checks),
        failed_check_ids=failed,
        manifest_hash=manifest_hash,
        branch_state_hash=branch_state_hash,
        implementation_commit=repo.head_commit,
        working_tree_clean=repo.tracked_clean and repo.staged_clean,
        protected_evidence_hashes=evidence_hashes,
    )
    return report, contract


def preflight_report_lines(report: RecoveryExperimentPreflightReport) -> tuple[str, ...]:
    lines = [
        f"PREFLIGHT_SCHEMA {report.preflight_schema_version}",
        f"IMPLEMENTATION_COMMIT {report.implementation_commit}",
        f"MANIFEST_HASH {report.manifest_hash}",
        f"BRANCH_STATE_HASH {report.branch_state_hash}",
    ]
    for check in report.checks:
        lines.append(
            f"{check.status.upper()} {check.check_id}: {check.message}"
        )
    for name, digest in report.protected_evidence_hashes:
        lines.append(f"PROTECTED_HASH {name} {digest}")
    lines.append(f"READY {'true' if report.ready else 'false'}")
    return tuple(lines)


__all__ = [
    "BRANCH_STATE_RELATIVE_PATH",
    "EXPECTED_RUNTIME_STOP_PRIORITY",
    "FROZEN_BRANCH_STATE_CANONICAL_HASH",
    "FROZEN_MANIFEST_CANONICAL_HASH",
    "MANIFEST_RELATIVE_PATH",
    "PREFLIGHT_SCHEMA_VERSION",
    "RecoveryExperimentPreflightCheck",
    "RecoveryExperimentPreflightError",
    "RecoveryExperimentPreflightReport",
    "RecoveryRepositoryState",
    "build_artifact_contract",
    "find_repository_root",
    "inspect_repository_state",
    "load_json_object",
    "preflight_report_lines",
    "protected_evidence_hashes",
    "run_recovery_experiment_preflight",
]

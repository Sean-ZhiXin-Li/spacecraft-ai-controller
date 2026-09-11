from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import scipy


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "stage2b_numerical_validation_v0"
    / "manifest.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "stage2b_numerical_validation_v0"
    / "results.json"
)


FROZEN_FILES = (
    "analysis/stage2a_post_veto_alternative_audit_v0/"
    "exact_state_comparisons.json",

    "analysis/stage2b_numerical_validation_v0/"
    "results.json",

    "analysis/stage2b_numerical_validation_v0/"
    "summary.md",

    "simulator/phase34_35_transition.py",

    "runtime_assurance/dop853_validation.py",

    "scripts/run_stage2b_numerical_validation_v0.py",

    "scripts/check_stage2b_numerical_validation_v0.py",

    "scripts/build_stage2b_numerical_validation_manifest_v0.py",

    "Tests/test_stage2b_numerical_validation.py",

    "environment.yml",
)


def sha256_file(
    path: Path,
) -> str:
    text = path.read_text(
        encoding="utf-8",
    )

    normalized_text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


def canonical_json_bytes(
    value: object,
) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def canonical_hash(
    value: object,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value)
    ).hexdigest()


def repository_head() -> str:
    completed = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return completed.stdout.strip()


def main() -> None:
    results = json.loads(
        RESULTS_PATH.read_text(
            encoding="utf-8",
        )
    )

    file_hashes: dict[str, str] = {}

    for relative_path in FROZEN_FILES:
        absolute_path = (
            PROJECT_ROOT
            / relative_path
        )

        if not absolute_path.is_file():
            raise FileNotFoundError(
                f"required Stage 2B file missing: "
                f"{relative_path}"
            )

        file_hashes[relative_path] = (
            sha256_file(
                absolute_path
            )
        )

    bundle_hash = canonical_hash(
        file_hashes
    )

    manifest = {
        "schema_version": (
            "stage2b_numerical_validation_manifest_v0"
        ),
        "validation_id": (
            "stage2b_numerical_propagation_"
            "independence_validation_v0"
        ),
        "completed_date": "2026-09-11",
        "validation_classification": (
            "offline_numerical_validation"
        ),
        "source_repository_head": (
            repository_head()
        ),
        "baseline_propagator": (
            "phase34_35_semi_implicit_euler"
        ),
        "independent_propagator": (
            "scipy_solve_ivp_DOP853"
        ),
        "validation_scope": (
            "one_step_exact_state"
        ),
        "controller_authority": (
            "none"
        ),
        "stage2a_runtime_modified": False,
        "controller_policy_modified": False,
        "controller_decision_interval_seconds": (
            100.0
        ),
        "dop853_internal_steps_change_"
        "controller_frequency": False,
        "overspeed_comparator": ">",
        "overspeed_threshold": 1.90,
        "exact_state_count": (
            results[
                "exact_state_count"
            ]
        ),
        "nominal_proposal_count": (
            results[
                "nominal_proposal_count"
            ]
        ),
        "physical_alternative_count": (
            results[
                "physical_alternative_count"
            ]
        ),
        "total_evaluated_proposal_count": (
            results[
                "total_evaluated_proposal_count"
            ]
        ),
        "classification_match_count": (
            results[
                "classification_match_count"
            ]
        ),
        "all_classifications_match": (
            results[
                "all_classifications_match"
            ]
        ),
        "maximum_absolute_speed_ratio_difference": (
            results[
                "maximum_absolute_speed_ratio_difference"
            ]
        ),
        "runtime_environment_observed": {
            "python_version": (
                platform.python_version()
            ),
            "scipy_version": (
                scipy.__version__
            ),
        },
        "claim_restrictions": [
            "no_global_integrator_independence_claim",
            "no_multi_step_independence_claim",
            "no_physical_model_fidelity_claim",
            "no_real_spacecraft_safety_claim",
            "no_controller_superiority_claim",
            "no_recovery_success_claim",
        ],
        "frozen_files": list(
            FROZEN_FILES
        ),
        "file_hash_method": (
            "sha256_utf8_lf_normalized_text"
        ),
        "file_hashes": (
            file_hashes
        ),
        "bundle_hash": (
            bundle_hash
        ),
    }

    manifest["canonical_manifest_hash"] = (
        canonical_hash(
            manifest
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "manifest:",
        OUTPUT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "source_repository_head:",
        manifest[
            "source_repository_head"
        ],
    )

    print(
        "python_version:",
        manifest[
            "runtime_environment_observed"
        ][
            "python_version"
        ],
    )

    print(
        "scipy_version:",
        manifest[
            "runtime_environment_observed"
        ][
            "scipy_version"
        ],
    )

    print(
        "frozen_file_count:",
        len(FROZEN_FILES),
    )

    print(
        "bundle_hash:",
        bundle_hash,
    )

    print(
        "canonical_manifest_hash:",
        manifest[
            "canonical_manifest_hash"
        ],
    )


if __name__ == "__main__":
    main()

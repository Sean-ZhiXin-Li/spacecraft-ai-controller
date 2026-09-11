from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "stage2b_numerical_validation_v0"
    / "results.json"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "stage2b_numerical_validation_v0"
    / "manifest.json"
)

EXPECTED_SCHEMA_VERSION = (
    "stage2b_numerical_validation_v0"
)

EXPECTED_MANIFEST_SCHEMA_VERSION = (
    "stage2b_numerical_validation_manifest_v0"
)

EXPECTED_SCOPE = "one_step_exact_state"

EXPECTED_BASELINE_PROPAGATOR = (
    "phase34_35_semi_implicit_euler"
)

EXPECTED_INDEPENDENT_PROPAGATOR = (
    "scipy_solve_ivp_DOP853"
)

EXPECTED_THRESHOLD = 1.90

EXPECTED_EXACT_STATE_COUNT = 4
EXPECTED_NOMINAL_COUNT = 4
EXPECTED_ALTERNATIVE_COUNT = 10
EXPECTED_TOTAL_COUNT = 14
EXPECTED_FROZEN_FILE_COUNT = 10

REQUIRED_FROZEN_FILES = {
    "analysis/stage2a_post_veto_alternative_audit_v0/"
    "exact_state_comparisons.json",
    "analysis/stage2b_numerical_validation_v0/results.json",
    "analysis/stage2b_numerical_validation_v0/summary.md",
    "simulator/phase34_35_transition.py",
    "runtime_assurance/dop853_validation.py",
    "scripts/run_stage2b_numerical_validation_v0.py",
    "scripts/check_stage2b_numerical_validation_v0.py",
    "scripts/build_stage2b_numerical_validation_manifest_v0.py",
    "Tests/test_stage2b_numerical_validation.py",
    "environment.yml",
}


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


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


def main() -> None:
    document = json.loads(
        RESULTS_PATH.read_text(
            encoding="utf-8",
        )
    )

    require(
        document["schema_version"]
        == EXPECTED_SCHEMA_VERSION,
        "unexpected schema version",
    )

    require(
        document["validation_scope"]
        == EXPECTED_SCOPE,
        "unexpected validation scope",
    )

    require(
        document["baseline_propagator"]
        == EXPECTED_BASELINE_PROPAGATOR,
        "unexpected baseline propagator",
    )

    require(
        document["independent_propagator"]
        == EXPECTED_INDEPENDENT_PROPAGATOR,
        "unexpected independent propagator",
    )

    require(
        document["overspeed_threshold"]
        == EXPECTED_THRESHOLD,
        "overspeed threshold drifted",
    )

    require(
        document["exact_state_count"]
        == EXPECTED_EXACT_STATE_COUNT,
        "unexpected exact-state count",
    )

    require(
        document["nominal_proposal_count"]
        == EXPECTED_NOMINAL_COUNT,
        "unexpected nominal proposal count",
    )

    require(
        document["physical_alternative_count"]
        == EXPECTED_ALTERNATIVE_COUNT,
        "unexpected physical alternative count",
    )

    require(
        document["total_evaluated_proposal_count"]
        == EXPECTED_TOTAL_COUNT,
        "unexpected total proposal count",
    )

    records = document["records"]

    require(
        len(records)
        == EXPECTED_TOTAL_COUNT,
        "record count does not match total",
    )

    nominal_records = [
        record
        for record in records
        if record["proposal_type"]
        == "nominal"
    ]

    alternative_records = [
        record
        for record in records
        if record["proposal_type"]
        == "physical_alternative"
    ]

    require(
        len(nominal_records)
        == EXPECTED_NOMINAL_COUNT,
        "nominal record count mismatch",
    )

    require(
        len(alternative_records)
        == EXPECTED_ALTERNATIVE_COUNT,
        "alternative record count mismatch",
    )

    recomputed_match_count = 0
    maximum_absolute_difference = 0.0

    for record in records:
        euler_ratio = float(
            record[
                "recomputed_euler_speed_ratio"
            ]
        )

        dop853_ratio = float(
            record[
                "dop853_speed_ratio"
            ]
        )

        frozen_ratio = float(
            record[
                "frozen_euler_speed_ratio"
            ]
        )

        stored_difference = float(
            record[
                "dop853_minus_euler"
            ]
        )

        require(
            math.isfinite(euler_ratio),
            "non-finite Euler ratio",
        )

        require(
            math.isfinite(dop853_ratio),
            "non-finite DOP853 ratio",
        )

        require(
            math.isfinite(frozen_ratio),
            "non-finite frozen Euler ratio",
        )

        require(
            math.isclose(
                euler_ratio,
                frozen_ratio,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            (
                "Euler reproduction mismatch: "
                f"{record['case_id']} / "
                f"{record['action_identity']}"
            ),
        )

        expected_difference = (
            dop853_ratio - euler_ratio
        )

        require(
            math.isclose(
                stored_difference,
                expected_difference,
                rel_tol=0.0,
                abs_tol=1.0e-18,
            ),
            (
                "stored propagation difference "
                "does not recompute"
            ),
        )

        euler_veto = (
            euler_ratio
            > EXPECTED_THRESHOLD
        )

        dop853_veto = (
            dop853_ratio
            > EXPECTED_THRESHOLD
        )

        require(
            record["euler_veto"]
            == euler_veto,
            "Euler classification does not recompute",
        )

        require(
            record["dop853_veto"]
            == dop853_veto,
            "DOP853 classification does not recompute",
        )

        classification_match = (
            euler_veto == dop853_veto
        )

        require(
            record["classification_match"]
            == classification_match,
            "classification_match does not recompute",
        )

        if classification_match:
            recomputed_match_count += 1

        maximum_absolute_difference = max(
            maximum_absolute_difference,
            abs(expected_difference),
        )

    require(
        recomputed_match_count
        == EXPECTED_TOTAL_COUNT,
        "not all classifications match",
    )

    require(
        document["classification_match_count"]
        == recomputed_match_count,
        "stored classification match count mismatch",
    )

    require(
        document["all_classifications_match"]
        is True,
        "all_classifications_match is not true",
    )

    require(
        math.isclose(
            float(
                document[
                    "maximum_absolute_speed_ratio_difference"
                ]
            ),
            maximum_absolute_difference,
            rel_tol=0.0,
            abs_tol=1.0e-18,
        ),
        "maximum propagation difference does not recompute",
    )

    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8",
        )
    )

    require(
        manifest["schema_version"]
        == EXPECTED_MANIFEST_SCHEMA_VERSION,
        "unexpected manifest schema version",
    )

    require(
        manifest["file_hash_method"]
        == "sha256_utf8_lf_normalized_text",
        "unexpected file hash method",
    )

    require(
        manifest["validation_scope"]
        == EXPECTED_SCOPE,
        "manifest validation scope mismatch",
    )

    require(
        manifest["baseline_propagator"]
        == EXPECTED_BASELINE_PROPAGATOR,
        "manifest baseline propagator mismatch",
    )

    require(
        manifest["independent_propagator"]
        == EXPECTED_INDEPENDENT_PROPAGATOR,
        "manifest independent propagator mismatch",
    )

    require(
        manifest["overspeed_threshold"]
        == EXPECTED_THRESHOLD,
        "manifest overspeed threshold drifted",
    )

    require(
        manifest["all_classifications_match"]
        is True,
        "manifest does not preserve PASS result",
    )

    require(
        manifest["classification_match_count"]
        == recomputed_match_count,
        "manifest classification count mismatch",
    )

    require(
        manifest["total_evaluated_proposal_count"]
        == EXPECTED_TOTAL_COUNT,
        "manifest total proposal count mismatch",
    )

    supplied_manifest_hash = manifest.get(
        "canonical_manifest_hash"
    )

    require(
        isinstance(
            supplied_manifest_hash,
            str,
        )
        and len(supplied_manifest_hash) == 64,
        "canonical manifest hash missing or invalid",
    )

    manifest_without_hash = dict(
        manifest
    )

    manifest_without_hash.pop(
        "canonical_manifest_hash"
    )

    recomputed_manifest_hash = (
        canonical_hash(
            manifest_without_hash
        )
    )

    require(
        supplied_manifest_hash
        == recomputed_manifest_hash,
        "canonical manifest hash mismatch",
    )

    frozen_files = manifest[
        "frozen_files"
    ]

    file_hashes = manifest[
        "file_hashes"
    ]

    require(
        isinstance(frozen_files, list),
        "manifest frozen_files must be a list",
    )

    require(
        isinstance(file_hashes, dict),
        "manifest file_hashes must be an object",
    )

    require(
        len(frozen_files)
        == EXPECTED_FROZEN_FILE_COUNT,
        "unexpected frozen-file count",
    )

    require(
        len(file_hashes)
        == EXPECTED_FROZEN_FILE_COUNT,
        "unexpected file-hash count",
    )

    require(
        len(set(frozen_files))
        == len(frozen_files),
        "duplicate frozen-file entry",
    )

    require(
        set(frozen_files)
        == REQUIRED_FROZEN_FILES,
        "frozen-file set mismatch",
    )

    require(
        set(file_hashes)
        == REQUIRED_FROZEN_FILES,
        "file-hash key set mismatch",
    )

    for relative_path in frozen_files:
        path = (
            PROJECT_ROOT
            / relative_path
        )

        require(
            path.is_file(),
            (
                "frozen file missing: "
                f"{relative_path}"
            ),
        )

        actual_hash = sha256_file(
            path
        )

        require(
            actual_hash
            == file_hashes[
                relative_path
            ],
            (
                "frozen file hash mismatch: "
                f"{relative_path}"
            ),
        )

    recomputed_bundle_hash = (
        canonical_hash(
            file_hashes
        )
    )

    require(
        recomputed_bundle_hash
        == manifest["bundle_hash"],
        "bundle hash mismatch",
    )

    print(
        "schema_version:",
        document["schema_version"],
    )

    print(
        "record_count:",
        len(records),
    )

    print(
        "nominal_record_count:",
        len(nominal_records),
    )

    print(
        "alternative_record_count:",
        len(alternative_records),
    )

    print(
        "classification_match_count:",
        recomputed_match_count,
    )

    print(
        "maximum_absolute_speed_ratio_difference:",
        maximum_absolute_difference,
    )

    print(
        "stage2b_numerical_validation_v0: PASS"
    )

    print(
        "frozen_file_count:",
        len(frozen_files),
    )

    print(
        "bundle_hash:",
        recomputed_bundle_hash,
    )

    print(
        "canonical_manifest_hash:",
        recomputed_manifest_hash,
    )

    print(
        "stage2b_manifest_integrity: PASS"
    )


if __name__ == "__main__":
    main()

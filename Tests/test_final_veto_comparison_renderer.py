from __future__ import annotations

import ast
import contextlib
import csv
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from scripts.check_final_veto_results import (
    main as result_validator_main,
    validate_formal_artifact_completeness,
)
from scripts.render_final_veto_comparison import (
    EXPECTED_EXPERIMENT_ID,
    FIGURE_DPI,
    PRESERVATION_SUBSET_ID,
    STRESS_SUBSET_ID,
    ComparisonRenderError,
    inspect_png,
    load_comparison_data,
    render_comparison_plot,
    validate_output_path,
)
from scripts.run_final_veto_ablation import (
    RunnerContractError,
    execute_jobs_to_directory,
    load_frozen_manifest,
    require_complete_formal_package,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIRECTORY = PROJECT_ROOT / "analysis" / "final_veto_ablation_v0"
RESULTS = FORMAL_DIRECTORY / "results.csv"
PAIRED_RESULTS = FORMAL_DIRECTORY / "paired_results.csv"
ORIGINAL_FORMAL_HASHES = {
    "manifest.json": "5e4387ed375855e0eb79d3b01599c421360ec33235000d4be7fe076794cda3a3",
    "results.csv": "1d41f5af976d4c2408c6eb0d11540b78a5d4b971e749aaf04bb081b77a933a61",
    "paired_results.csv": "723a2e069d56cb762ca44ff25524414b7d044e80cca5d2ab87b05acaef8fdd11",
    "decision_log.jsonl": "8926598ea30981076adc5c851055b01480b55c425dbc310d6f1e45fe7019b72f",
    "summary.md": "84f5f0e4968dbe250fc6eb2cd23c7c63bbb2573496a6a20609c1d99f26a8f979",
}
PROTECTED_DIRECTORIES = (
    PROJECT_ROOT / "analysis" / "phase34_post_cross_sync",
    PROJECT_ROOT / "analysis" / "phase35_crossing_basin_expansion",
    PROJECT_ROOT / "analysis" / "phase36b_transfer_family_benchmark",
    PROJECT_ROOT / "analysis" / "phase36c_non_crossing_geometry_diagnosis",
    PROJECT_ROOT / "analysis" / "phase37a_radial_commit_timing",
    PROJECT_ROOT / "analysis" / "phase37b_weak_tangential_subset",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_files(directories: tuple[Path, ...]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for directory in directories:
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            snapshot[path.relative_to(PROJECT_ROOT).as_posix()] = sha256(path)
    return snapshot


def copy_csv_with_mutation(
    source: Path,
    destination: Path,
    mutation,
) -> None:
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    mutation(rows)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def copy_formal_package(repository_root: Path, *, include_plot: bool = False) -> Path:
    destination = repository_root / "analysis" / "final_veto_ablation_v0"
    destination.mkdir(parents=True)
    for filename in ORIGINAL_FORMAL_HASHES:
        shutil.copyfile(FORMAL_DIRECTORY / filename, destination / filename)
    if include_plot:
        render_comparison_plot(
            destination / "results.csv",
            destination / "paired_results.csv",
            destination / "comparison.png",
        )
    return destination


def completeness_report(
    repository_root: Path,
    *,
    allow_missing_plot: bool = False,
):
    directory = repository_root / "analysis" / "final_veto_ablation_v0"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return validate_formal_artifact_completeness(
        manifest,
        manifest_path=manifest_path,
        results_path=directory / "results.csv",
        paired_results_path=directory / "paired_results.csv",
        decision_log_path=directory / "decision_log.jsonl",
        repository_root=repository_root,
        allow_missing_comparison_plot=allow_missing_plot,
    )


class ComparisonDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_comparison_data(RESULTS, PAIRED_RESULTS)

    def test_exact_arm_and_pair_counts_are_required(self) -> None:
        self.assertEqual(len(self.data.arm_rows), 26)
        self.assertEqual(len(self.data.pair_rows), 13)
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            shortened_results = temporary / "results.csv"
            copy_csv_with_mutation(RESULTS, shortened_results, lambda rows: rows.pop())
            with self.assertRaisesRegex(ComparisonRenderError, "exactly 26"):
                load_comparison_data(shortened_results, PAIRED_RESULTS)

            shortened_pairs = temporary / "pairs.csv"
            copy_csv_with_mutation(PAIRED_RESULTS, shortened_pairs, lambda rows: rows.pop())
            with self.assertRaisesRegex(ComparisonRenderError, "exactly 13"):
                load_comparison_data(RESULTS, shortened_pairs)

    def test_missing_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "pairs.csv"
            copy_csv_with_mutation(PAIRED_RESULTS, path, lambda rows: rows.pop(0))
            with self.assertRaises(ComparisonRenderError):
                load_comparison_data(RESULTS, path)

    def test_malformed_csv_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "malformed.csv"
            path.write_text('experiment_id,"unterminated', encoding="utf-8")
            with self.assertRaises(ComparisonRenderError):
                load_comparison_data(path, PAIRED_RESULTS)

    def test_wrong_experiment_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "results.csv"

            def mutate(rows):
                rows[0]["experiment_id"] = "wrong_experiment"

            copy_csv_with_mutation(RESULTS, path, mutate)
            with self.assertRaisesRegex(ComparisonRenderError, EXPECTED_EXPERIMENT_ID):
                load_comparison_data(path, PAIRED_RESULTS)

    def test_case_order_is_stable_and_subsets_remain_separate(self) -> None:
        repeated = load_comparison_data(RESULTS, PAIRED_RESULTS)
        self.assertEqual(self.data.case_order, repeated.case_order)
        subsets = [record.subset_id for record in self.data.monitor_on_cases]
        self.assertEqual(subsets[:8], [PRESERVATION_SUBSET_ID] * 8)
        self.assertEqual(subsets[8:], [STRESS_SUBSET_ID] * 5)

    def test_hazard_and_preservation_counts_match_formal_rows(self) -> None:
        self.assertEqual(self.data.hazard_counts["preservation"], (0, 0))
        self.assertEqual(self.data.hazard_counts["diagnostic stress"], (5, 0))
        self.assertEqual(self.data.preservation_counts["crossing"], (8, 8))
        self.assertEqual(
            self.data.preservation_counts["recoverable crossing"],
            (8, 8),
        )
        self.assertEqual(self.data.preservation_counts["simulator success"], (8, 8))

    def test_intervention_rate_is_computed_from_counts(self) -> None:
        self.assertEqual(self.data.total_monitor_evaluations, 511327)
        self.assertEqual(self.data.total_vetoes, 499877)
        self.assertEqual(
            self.data.overall_intervention_rate,
            499877 / 511327,
        )

    def test_terminal_transitions_do_not_infer_task_recovery(self) -> None:
        transitions = Counter(
            transition.terminal_transition for transition in self.data.stress_transitions
        )
        self.assertEqual(transitions, Counter({"overspeed -> max_steps": 5}))
        self.assertTrue(all(item.hazard_avoided for item in self.data.stress_transitions))
        self.assertFalse(any(item.task_recovered for item in self.data.stress_transitions))


class ComparisonRenderTests(unittest.TestCase):
    def test_renderer_source_has_no_rollout_or_controller_imports(self) -> None:
        source_path = PROJECT_ROOT / "scripts" / "render_final_veto_comparison.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        banned = ("explicit_controller", "controller", "envs", "simulator", "run_final_veto")
        self.assertFalse(any(any(token in name for token in banned) for name in imported))

    def test_renderer_writes_a_nonempty_readable_png_without_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            output = Path(temporary_name) / "comparison.png"
            with mock.patch(
                "scripts.run_final_veto_ablation.execute_job"
            ) as execute_job:
                data, metadata = render_comparison_plot(RESULTS, PAIRED_RESULTS, output)
            execute_job.assert_not_called()
            self.assertEqual((metadata.width, metadata.height), (2400, 1500))
            self.assertEqual(FIGURE_DPI, 150)
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(inspect_png(output), metadata)
            self.assertEqual(len(data.arm_rows), 26)

    def test_renderer_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            directory = Path(temporary_name)
            first = directory / "first.png"
            second = directory / "second.png"
            render_comparison_plot(RESULTS, PAIRED_RESULTS, first)
            render_comparison_plot(RESULTS, PAIRED_RESULTS, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_renderer_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            output = Path(temporary_name) / "comparison.png"
            output.write_bytes(b"existing")
            with self.assertRaisesRegex(ComparisonRenderError, "overwrite"):
                render_comparison_plot(RESULTS, PAIRED_RESULTS, output)
            self.assertEqual(output.read_bytes(), b"existing")

    def test_renderer_is_atomic_and_leaves_no_partial_file_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            directory = Path(temporary_name)
            output = directory / "comparison.png"

            def fail_after_partial_write(_data, temporary_path):
                temporary_path.write_bytes(b"partial")
                raise RuntimeError("synthetic render failure")

            with mock.patch(
                "scripts.render_final_veto_comparison._draw_figure",
                side_effect=fail_after_partial_write,
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic render failure"):
                    render_comparison_plot(RESULTS, PAIRED_RESULTS, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(directory.glob(".comparison.png.*.tmp")), [])

    def test_output_traversal_and_protected_directories_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            traversal = Path(temporary_name) / "nested" / ".." / "comparison.png"
            with self.assertRaisesRegex(ComparisonRenderError, "traversal"):
                validate_output_path(traversal)
        with self.assertRaisesRegex(ComparisonRenderError, "protected"):
            validate_output_path(
                PROJECT_ROOT / "analysis" / "phase34_post_cross_sync" / "comparison.png"
            )

    def test_render_does_not_change_protected_historical_artifacts(self) -> None:
        before = snapshot_files(PROTECTED_DIRECTORIES)
        with tempfile.TemporaryDirectory() as temporary_name:
            render_comparison_plot(
                RESULTS,
                PAIRED_RESULTS,
                Path(temporary_name) / "comparison.png",
            )
        self.assertEqual(snapshot_files(PROTECTED_DIRECTORIES), before)


class FormalArtifactCompletenessTests(unittest.TestCase):
    def test_formal_validator_rejects_missing_plot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            copy_formal_package(root)
            report = completeness_report(root)
            self.assertFalse(report.complete)
            self.assertTrue(any("comparison.png" in error for error in report.errors))

    def test_plot_pending_mode_allows_only_the_missing_plot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            directory = copy_formal_package(root)
            report = completeness_report(root, allow_missing_plot=True)
            self.assertFalse(report.complete)
            self.assertTrue(report.comparison_plot_pending)
            (directory / "summary.md").unlink()
            report = completeness_report(root, allow_missing_plot=True)
            self.assertFalse(report.comparison_plot_pending)
            self.assertTrue(any("summary.md" in error for error in report.errors))

    def test_formal_validator_rejects_zero_byte_and_fake_png(self) -> None:
        for payload in (b"", b"not a png"):
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as temporary_name:
                    root = Path(temporary_name)
                    directory = copy_formal_package(root)
                    (directory / "comparison.png").write_bytes(payload)
                    report = completeness_report(root)
                    self.assertFalse(report.complete)
                    self.assertTrue(report.errors)

    def test_formal_validator_accepts_a_complete_synthetic_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            copy_formal_package(root, include_plot=True)
            report = completeness_report(root)
            self.assertTrue(report.complete, report.errors)
            self.assertFalse(report.comparison_plot_pending)

    def test_artifact_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            directory = copy_formal_package(root)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["output_contract"]["future_artifacts"][0]["path"] = (
                "analysis/final_veto_ablation_v0/../../escaped.csv"
            )
            report = validate_formal_artifact_completeness(
                manifest,
                manifest_path=manifest_path,
                results_path=directory / "results.csv",
                paired_results_path=directory / "paired_results.csv",
                decision_log_path=directory / "decision_log.jsonl",
                repository_root=root,
            )
            self.assertFalse(report.complete)
            self.assertTrue(any("escapes" in error for error in report.errors))

    def test_nonformal_validation_does_not_require_a_plot(self) -> None:
        with mock.patch(
            "scripts.check_final_veto_results.validate_formal_artifact_completeness"
        ) as completeness:
            with contextlib.redirect_stdout(io.StringIO()):
                result = result_validator_main(
                    [
                        "--results",
                        str(RESULTS),
                        "--paired-results",
                        str(PAIRED_RESULTS),
                        "--decision-log",
                        str(FORMAL_DIRECTORY / "decision_log.jsonl"),
                    ]
                )
        self.assertEqual(result, 0)
        completeness.assert_not_called()

    def test_future_runner_refuses_a_four_of_five_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            directory = copy_formal_package(Path(temporary_name))
            with self.assertRaisesRegex(RunnerContractError, "comparison_plot"):
                require_complete_formal_package(directory)

    def test_formal_execution_wrapper_cannot_succeed_without_rendered_plot(self) -> None:
        manifest = load_frozen_manifest()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            output = root / "analysis" / "final_veto_ablation_v0"

            def stage_four_artifacts(_jobs, staged, _manifest, **_kwargs):
                staged.mkdir(parents=True, exist_ok=True)
                for filename in (
                    "results.csv",
                    "paired_results.csv",
                    "decision_log.jsonl",
                    "summary.md",
                ):
                    shutil.copyfile(FORMAL_DIRECTORY / filename, staged / filename)
                return [], []

            with mock.patch(
                "scripts.run_final_veto_ablation.PROJECT_ROOT",
                root,
            ), mock.patch(
                "scripts.run_final_veto_ablation._execute_jobs_to_artifact_directory",
                side_effect=stage_four_artifacts,
            ), mock.patch(
                "scripts.run_final_veto_ablation.render_comparison_plot"
            ):
                with self.assertRaisesRegex(RunnerContractError, "comparison_plot"):
                    execute_jobs_to_directory(
                        [],
                        output,
                        manifest,
                        is_formal_experiment=True,
                    )
            self.assertFalse((output / "results.csv").exists())

    def test_original_formal_artifact_hashes_remain_frozen(self) -> None:
        actual = {
            filename: sha256(FORMAL_DIRECTORY / filename)
            for filename in ORIGINAL_FORMAL_HASHES
        }
        self.assertEqual(actual, ORIGINAL_FORMAL_HASHES)


if __name__ == "__main__":
    unittest.main()

"""Batch-export print_library Markdown files to PDF with Typora.

Python is the controller: it scans files, manages resume/skip, retries, logging,
progress, and PDF stability checks. AutoHotkey v2 is used only for Typora GUI
automation.

Example:
    python tools/typora_pdf_export.py
    python tools/typora_pdf_export.py --binder Binder03 --limit 10
    python tools/typora_pdf_export.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRINT_ROOT = REPO_ROOT / "print_library"
DEFAULT_TYPORA = Path(r"E:\Typora\Typora.exe")
DEFAULT_AHK_SCRIPT = REPO_ROOT / "tools" / "typora_export_gui.ahk"
DEFAULT_LOG = DEFAULT_PRINT_ROOT / "export.log"
DEFAULT_STATE = DEFAULT_PRINT_ROOT / "export_state.json"


@dataclass(frozen=True)
class ExportItem:
    markdown: Path
    pdf: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export print_library Markdown files to PDF using Typora native PDF export.",
    )
    parser.add_argument("--root", default=str(DEFAULT_PRINT_ROOT), help="Markdown root to scan.")
    parser.add_argument("--typora", default=str(DEFAULT_TYPORA), help="Typora executable path.")
    parser.add_argument("--ahk", default=None, help="AutoHotkey v2 executable path.")
    parser.add_argument("--ahk-script", default=str(DEFAULT_AHK_SCRIPT), help="AHK v2 helper script.")
    parser.add_argument("--binder", default=None, help="Only export paths under a binder, e.g. Binder03.")
    parser.add_argument("--force", action="store_true", help="Re-export even when the PDF already exists.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of pending files to export.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per Markdown file.")
    parser.add_argument("--load-wait", type=float, default=2.5, help="Seconds to wait after Typora opens a file.")
    parser.add_argument("--dialog-timeout", type=int, default=30, help="Seconds to wait for the Save dialog.")
    parser.add_argument("--pdf-timeout", type=int, default=180, help="Seconds to wait for each PDF to appear.")
    parser.add_argument("--stable-seconds", type=float, default=3.0, help="PDF size must remain stable this long.")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="Export log path.")
    parser.add_argument("--state", default=str(DEFAULT_STATE), help="Resume state JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="List pending files without opening Typora.")
    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def find_ahk(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_path = os.environ.get("AHK_EXE")
    if env_path:
        candidates.append(Path(env_path))
    which = shutil.which("AutoHotkey64.exe") or shutil.which("AutoHotkey.exe")
    if which:
        candidates.append(Path(which))
    candidates.extend(
        [
            Path(r"C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe"),
            Path(r"C:\Program Files\AutoHotkey\v2\AutoHotkey.exe"),
            Path(r"C:\Program Files\AutoHotkey\AutoHotkey64.exe"),
            Path(r"C:\Program Files\AutoHotkey\AutoHotkey.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "AutoHotkey v2 executable not found. Install AutoHotkey v2 or pass --ahk."
    )


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"started": None, "updated": None, "succeeded": {}, "failed": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        backup = path.with_suffix(path.suffix + ".corrupt")
        path.replace(backup)
        logging.warning("State file was corrupt and was moved to %s", backup)
        return {"started": None, "updated": None, "succeeded": {}, "failed": {}}


def save_state(path: Path, state: dict) -> None:
    state["updated"] = datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def is_inside_binder(path: Path, print_root: Path, binder: str | None) -> bool:
    if not binder:
        return True
    try:
        rel = path.relative_to(print_root)
    except ValueError:
        return False
    first = rel.parts[0] if rel.parts else ""
    return first.lower().startswith(binder.lower())


def scan_markdown(print_root: Path, binder: str | None) -> list[ExportItem]:
    files = []
    for md in sorted(print_root.rglob("*.md")):
        if not md.is_file():
            continue
        if not is_inside_binder(md, print_root, binder):
            continue
        files.append(ExportItem(markdown=md, pdf=md.with_suffix(".pdf")))
    return files


def pending_items(items: Iterable[ExportItem], force: bool) -> list[ExportItem]:
    if force:
        return list(items)
    return [item for item in items if not item.pdf.exists()]


def launch_typora_once(typora: Path) -> subprocess.Popen:
    if not typora.exists():
        raise FileNotFoundError(f"Typora executable not found: {typora}")
    logging.info("Launching Typora: %s", typora)
    proc = subprocess.Popen([str(typora)], cwd=str(REPO_ROOT))
    time.sleep(4)
    return proc


def log_ahk_output(stdout: str, stderr: str) -> None:
    for line in stdout.splitlines():
        if line.strip():
            logging.info(line)
    for line in stderr.splitlines():
        if line.strip():
            logging.warning("AHK stderr: %s", line)


def run_ahk(ahk: Path, script: Path, args: list[str], timeout: int) -> None:
    command = [str(ahk), str(script), *args]
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    log_ahk_output(result.stdout, result.stderr)
    if result.returncode != 0:
        details = "\n".join(part for part in [result.stdout, result.stderr] if part)
        raise RuntimeError(f"AHK failed with code {result.returncode}: {details}")


def wait_for_pdf_stable(pdf: Path, timeout: int, stable_seconds: float) -> int:
    deadline = time.monotonic() + timeout
    last_size = -1
    stable_since: float | None = None
    while time.monotonic() < deadline:
        if pdf.exists():
            size = pdf.stat().st_size
            if size > 0 and size == last_size:
                if stable_since is None:
                    stable_since = time.monotonic()
                if time.monotonic() - stable_since >= stable_seconds:
                    logging.info("PDF appeared and size stabilized: %s (%d bytes)", pdf, size)
                    return size
            else:
                last_size = size
                stable_since = None
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for stable PDF: {pdf}")


def progress(current: int, total: int, label: str) -> None:
    width = 30
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total} {label[:70]:70}", end="", flush=True)
    if current == total:
        print()


def export_one(
    item: ExportItem,
    ahk: Path,
    ahk_script: Path,
    typora: Path,
    args: argparse.Namespace,
) -> int:
    if args.force and item.pdf.exists():
        item.pdf.unlink()

    load_wait_ms = str(int(args.load_wait * 1000))
    run_ahk(
        ahk,
        ahk_script,
        [
            "open_export",
            str(typora),
            str(item.markdown),
            load_wait_ms,
            str(args.dialog_timeout),
        ],
        timeout=args.dialog_timeout + int(args.load_wait) + 45,
    )
    size = wait_for_pdf_stable(item.pdf, args.pdf_timeout, args.stable_seconds)
    run_ahk(ahk, ahk_script, ["close", str(typora), "500"], timeout=20)
    return size


def main() -> int:
    args = parse_args()
    print_root = Path(args.root).resolve()
    typora = Path(args.typora).resolve()
    ahk_script = Path(args.ahk_script).resolve()
    log_path = Path(args.log).resolve()
    state_path = Path(args.state).resolve()

    setup_logging(log_path)
    logging.info("Typora PDF export started")
    logging.info("print_root=%s binder=%s force=%s limit=%s", print_root, args.binder, args.force, args.limit)

    if not print_root.exists():
        raise FileNotFoundError(f"Print root not found: {print_root}")
    if not ahk_script.exists():
        raise FileNotFoundError(f"AHK helper script not found: {ahk_script}")

    all_items = scan_markdown(print_root, args.binder)
    todo = pending_items(all_items, args.force)
    if args.limit is not None:
        todo = todo[: args.limit]

    logging.info("Markdown files found: %d", len(all_items))
    logging.info("Pending exports: %d", len(todo))

    if args.dry_run:
        for item in todo:
            print(item.markdown)
        return 0
    if not todo:
        logging.info("Nothing to export")
        return 0

    ahk = find_ahk(args.ahk)
    logging.info("AutoHotkey executable: %s", ahk)

    state = load_state(state_path)
    if not state.get("started"):
        state["started"] = datetime.now().isoformat(timespec="seconds")
    state.setdefault("succeeded", {})
    state.setdefault("failed", {})
    save_state(state_path, state)

    launch_typora_once(typora)
    run_ahk(ahk, ahk_script, ["activate", str(typora)], timeout=35)

    failures: list[tuple[Path, str]] = []
    for index, item in enumerate(todo, start=1):
        rel = item.markdown.relative_to(print_root)
        progress(index - 1, len(todo), str(rel))
        logging.info("Exporting %s -> %s", item.markdown, item.pdf)
        success = False
        last_error = ""
        for attempt in range(1, args.retries + 1):
            try:
                size = export_one(item, ahk, ahk_script, typora, args)
                state["succeeded"][str(item.markdown)] = {
                    "pdf": str(item.pdf),
                    "bytes": size,
                    "attempt": attempt,
                    "time": datetime.now().isoformat(timespec="seconds"),
                }
                state["failed"].pop(str(item.markdown), None)
                save_state(state_path, state)
                logging.info("Exported %s (%d bytes)", item.pdf, size)
                success = True
                break
            except Exception as exc:  # noqa: BLE001 - batch exporter logs all failures
                last_error = str(exc)
                logging.warning("Attempt %d/%d failed for %s: %s", attempt, args.retries, item.markdown, last_error)
                try:
                    run_ahk(ahk, ahk_script, ["close", str(typora), "500"], timeout=20)
                except Exception as close_exc:  # noqa: BLE001
                    logging.debug("Close after failure also failed: %s", close_exc)
                time.sleep(1)
        if not success:
            state["failed"][str(item.markdown)] = {
                "pdf": str(item.pdf),
                "error": last_error,
                "time": datetime.now().isoformat(timespec="seconds"),
            }
            save_state(state_path, state)
            failures.append((item.markdown, last_error))
        progress(index, len(todo), str(rel))

    logging.info("Export complete: %d succeeded, %d failed", len(todo) - len(failures), len(failures))
    if failures:
        logging.error("Failed files:")
        for path, error in failures:
            logging.error("  %s :: %s", path, error)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the command to resume; existing PDFs are skipped unless --force is used.")
        raise SystemExit(130)

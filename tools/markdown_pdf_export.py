"""Non-GUI PDF export for print_library Markdown files.

This script is a fallback for Typora GUI automation. It does not require Typora
and never modifies Markdown sources.

Engine priority for --engine auto:
1. pandoc
2. quarto
3. Chrome or Edge headless print-to-PDF via generated HTML
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "print_library"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "print_pdf_library"
DEFAULT_THEME_ROOT = REPO_ROOT / "tools" / "pdf_themes"


@dataclass(frozen=True)
class ExportItem:
    source: Path
    relative: Path
    output: Path


@dataclass
class ExportResult:
    source: Path
    output: Path
    engine: str
    status: str
    message: str
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export print_library Markdown files to PDF without Typora.",
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="Markdown root to scan.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="PDF output root.")
    parser.add_argument("--binder", default=None, help="Only export one binder, e.g. Binder01_Research_Constitution.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of pending files to export.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing PDFs.")
    parser.add_argument(
        "--theme",
        choices=["binder", "research", "paper", "github"],
        default="binder",
        help="Print CSS theme used by the browser engine.",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "pandoc", "quarto", "chrome"],
        default="auto",
        help="PDF conversion engine.",
    )
    parser.add_argument("--summary", default=None, help="CSV summary path. Defaults to output-root/export_summary.csv.")
    return parser.parse_args()


def which_any(names: Iterable[str]) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def detect_engine(requested: str) -> tuple[str, str]:
    engines = {
        "pandoc": lambda: which_any(["pandoc.exe", "pandoc"]),
        "quarto": lambda: which_any(["quarto.exe", "quarto"]),
        "chrome": find_browser,
    }
    if requested != "auto":
        executable = engines[requested]()
        if executable:
            return requested, executable
        raise RuntimeError(f"Requested engine '{requested}' is not available on PATH or common install paths.")

    for name in ["pandoc", "quarto", "chrome"]:
        executable = engines[name]()
        if executable:
            return name, executable
    raise RuntimeError(
        "No PDF export engine found. Install one of: pandoc, quarto, Google Chrome, or Microsoft Edge."
    )


def find_browser() -> str | None:
    path = which_any(["chrome.exe", "msedge.exe", "chromium.exe", "google-chrome", "chromium", "msedge"])
    if path:
        return path
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def binder_matches(relative: Path, binder: str | None) -> bool:
    if binder is None:
        return True
    if not relative.parts:
        return False
    requested = binder.lower()
    first = relative.parts[0].lower()
    return first == requested or first.startswith(requested)


def scan_items(input_root: Path, output_root: Path, binder: str | None, force: bool) -> list[ExportItem]:
    items: list[ExportItem] = []
    for source in sorted(input_root.rglob("*.md")):
        if not source.is_file():
            continue
        relative = source.relative_to(input_root)
        if not binder_matches(relative, binder):
            continue
        output = (output_root / relative).with_suffix(".pdf")
        if output.exists() and not force:
            continue
        items.append(ExportItem(source=source, relative=relative, output=output))
    return items


def run_command(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if result.returncode != 0:
        details = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
        raise RuntimeError(details or f"Command failed: {' '.join(command)}")


def export_with_pandoc(item: ExportItem, executable: str) -> None:
    item.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        str(item.source),
        "--standalone",
        "--from",
        "markdown+pipe_tables+fenced_code_blocks+yaml_metadata_block",
        "--output",
        str(item.output),
    ]
    run_command(command, cwd=REPO_ROOT)


def export_with_quarto(item: ExportItem, executable: str) -> None:
    item.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="quarto_pdf_") as tmp_name:
        tmp = Path(tmp_name)
        temp_md = tmp / item.source.name
        temp_md.write_text(item.source.read_text(encoding="utf-8"), encoding="utf-8")
        command = [executable, "render", str(temp_md), "--to", "pdf", "--output-dir", str(tmp)]
        run_command(command, cwd=REPO_ROOT)
        produced = temp_md.with_suffix(".pdf")
        if not produced.exists():
            candidates = sorted(tmp.glob("*.pdf"))
            if not candidates:
                raise RuntimeError("Quarto completed but did not produce a PDF.")
            produced = candidates[0]
        shutil.copy2(produced, item.output)


def available_markdown_parser() -> str | None:
    if importlib.util.find_spec("markdown") is not None:
        return "python-markdown"
    if importlib.util.find_spec("markdown_it") is not None:
        return "markdown-it-py"
    if importlib.util.find_spec("mistune") is not None:
        return "mistune"
    return None


def render_markdown_body(markdown_text: str) -> tuple[str, str]:
    """Render Markdown to HTML with a real parser.

    Browser export must never print raw Markdown. If no supported parser is
    installed, fail clearly instead of falling back to escaped preformatted text.
    """

    parser = available_markdown_parser()
    if parser == "python-markdown":
        import markdown  # type: ignore

        return (
            markdown.markdown(
                markdown_text,
                extensions=["extra", "tables", "fenced_code", "sane_lists", "toc"],
                output_format="html5",
            ),
            parser,
        )

    if parser == "markdown-it-py":
        from markdown_it import MarkdownIt  # type: ignore

        md = MarkdownIt("commonmark", {"html": True, "typographer": True})
        for extension in ["table", "strikethrough"]:
            try:
                md.enable(extension)
            except Exception:
                pass
        return md.render(markdown_text), parser

    if parser == "mistune":
        import mistune  # type: ignore

        plugin_sets = [
            ["table", "strikethrough", "url"],
            ["table", "strikethrough"],
            ["table"],
        ]
        last_error: Exception | None = None
        for plugins in plugin_sets:
            try:
                renderer = mistune.create_markdown(escape=False, plugins=plugins)
                return renderer(markdown_text), parser
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"mistune is installed but could not render Markdown: {last_error}")

    raise RuntimeError(
        "Browser PDF export requires a Markdown parser. Install one of: "
        "python-markdown, markdown-it-py, or mistune. Refusing to print raw Markdown."
    )


def load_theme_css(theme: str) -> str:
    theme_path = DEFAULT_THEME_ROOT / f"{theme}.css"
    if not theme_path.exists():
        raise FileNotFoundError(f"Theme CSS not found: {theme_path}")
    return theme_path.read_text(encoding="utf-8")


def binder_name_for(relative: Path) -> str:
    if relative.parts and relative.parts[0].startswith("Binder"):
        return relative.parts[0]
    return "No binder"


def css_string(value: str) -> str:
    return value.replace("\\", "/").replace('"', '\\"')


def dynamic_print_css(repo_path: str, binder_name: str) -> str:
    _ = (repo_path, binder_name)
    return ""


def markdown_to_html(markdown_text: str, title: str, repo_path: str, binder_name: str, theme: str) -> tuple[str, str]:
    body, parser = render_markdown_body(markdown_text)
    theme_css = load_theme_css(theme)
    print_css = dynamic_print_css(repo_path, binder_name)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
{theme_css}
{print_css}
</style>
</head>
<body>
<div class="print-meta" data-repo-path="{html.escape(repo_path)}" data-binder-name="{html.escape(binder_name)}"></div>
<div class="markdown-body">
{body}
</div>
</body>
</html>
""", parser


def export_with_browser(item: ExportItem, executable: str, theme: str) -> None:
    item.output.parent.mkdir(parents=True, exist_ok=True)
    markdown_text = item.source.read_text(encoding="utf-8")
    repo_path = item.relative.as_posix()
    binder_name = binder_name_for(item.relative)
    document_html, parser = markdown_to_html(markdown_text, item.source.stem, repo_path, binder_name, theme)
    print(f"Rendering Markdown with {parser}, theme={theme}: {item.relative}")
    with tempfile.TemporaryDirectory(prefix="md_pdf_") as tmp_name:
        tmp = Path(tmp_name)
        html_path = tmp / (item.source.stem + ".html")
        profile_path = tmp / "browser_profile"
        profile_path.mkdir(parents=True, exist_ok=True)
        html_path.write_text(document_html, encoding="utf-8")
        command = [
            executable,
            "--headless=new",
            "--disable-gpu",
            "--disable-gpu-sandbox",
            "--disable-gpu-compositing",
            "--disable-dev-shm-usage",
            "--disable-features=UseSkiaRenderer,VizDisplayCompositor",
            "--no-sandbox",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            f"--user-data-dir={profile_path}",
            f"--print-to-pdf={item.output.resolve()}",
            html_path.as_uri(),
        ]
        run_command(command, cwd=REPO_ROOT)


def export_item(item: ExportItem, engine: str, executable: str, theme: str) -> None:
    if engine == "pandoc":
        export_with_pandoc(item, executable)
    elif engine == "quarto":
        export_with_quarto(item, executable)
    elif engine == "chrome":
        export_with_browser(item, executable, theme)
    else:
        raise ValueError(f"Unsupported engine: {engine}")


def write_summary(path: Path, results: list[ExportResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "source", "output", "engine", "status", "seconds", "message"],
        )
        writer.writeheader()
        timestamp = datetime.now().isoformat(timespec="seconds")
        for result in results:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "source": str(result.source),
                    "output": str(result.output),
                    "engine": result.engine,
                    "status": result.status,
                    "seconds": f"{result.seconds:.2f}",
                    "message": result.message,
                }
            )


def progress(current: int, total: int, label: str) -> None:
    width = 30
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total} {label[:70]:70}", end="", flush=True)
    if current == total:
        print()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    summary_path = Path(args.summary).resolve() if args.summary else output_root / "export_summary.csv"

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    engine, executable = detect_engine(args.engine)
    print(f"Using engine: {engine} ({executable})")
    print(f"Using theme: {args.theme}")

    items = scan_items(input_root, output_root, args.binder, args.force)
    if args.limit is not None:
        items = items[: args.limit]

    if not items:
        print("No pending Markdown files to export.")
        return 0

    results: list[ExportResult] = []
    failures = 0
    for index, item in enumerate(items, start=1):
        progress(index - 1, len(items), str(item.relative))
        start = time.monotonic()
        try:
            export_item(item, engine, executable, args.theme)
            if not item.output.exists() or item.output.stat().st_size <= 0:
                raise RuntimeError("PDF was not created or is empty.")
            elapsed = time.monotonic() - start
            results.append(
                ExportResult(item.source, item.output, engine, "success", "ok", elapsed)
            )
        except Exception as exc:  # noqa: BLE001 - batch exporter reports per-file failures
            failures += 1
            elapsed = time.monotonic() - start
            results.append(
                ExportResult(item.source, item.output, engine, "failure", str(exc), elapsed)
            )
            print(f"\nFAILED: {item.relative}: {exc}", file=sys.stderr)
        progress(index, len(items), str(item.relative))

    write_summary(summary_path, results)
    print(f"Summary written: {summary_path}")
    print(f"Exported: {len(results) - failures}; Failed: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

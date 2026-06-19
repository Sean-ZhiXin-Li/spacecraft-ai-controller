"""Build a merged Research Library PDF release from print_library.

The builder exports ordered Markdown staging files to individual PDFs, merges
them binder-by-binder, generates a library cover and index, writes a release
manifest and summary, and packages release artifacts into a zip file.

This is a publication/packaging tool only. It does not modify research content
or Markdown source files.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
PRINT_ROOT = REPO_ROOT / "print_library"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "release_print"

if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from markdown_pdf_export import (  # noqa: E402
    ExportItem,
    available_markdown_parser,
    detect_engine,
    export_with_browser,
)


BINDER_PURPOSES = {
    "Binder01_Research_Constitution": "Permanent scientific rules, scope boundaries, claim safety, benchmark rules, and repository philosophy.",
    "Binder02_Paper_Writing": "Documents for writing AI4Sci papers, future papers, quad charts, and presentations.",
    "Binder03_Scientific_Evidence": "Core evidence chain for Phase34, Phase36, Phase37, Phase38, Phase39, and major postmortems.",
    "Binder04_Experiment_Design": "Hypotheses, variable ranking, failure signatures, go/no-go reviews, and future experiment discipline.",
    "Binder05_Metrics_Reproducibility": "Metric definitions, benchmark grids, artifact maps, reproducibility notes, logging schema, and dataset contracts.",
    "Binder06_Literature_Review": "Reference plans, verified reference candidates, citation follow-up notes, and future literature planning.",
    "Binder07_Submission": "AI4Sci-specific schedule, repository checklist, figure plan, quad plan, public portfolio notes, and final submission notes.",
    "Binder08_Research_History": "Selected project logs and summaries that explain major research transitions.",
    "Binder09_Research_Philosophy": "Private philosophy, personal reflection, and long-term vision.",
    "Binder10_Research_Handbook": "Source packet for a future consolidated project handbook.",
}

PDF_KEYWORDS = "AI, Orbital Control, AI4Sci, Research Library, Benchmark, Reproducibility"
PDF_AUTHOR = "Zhixin (Sean) Li"


@dataclass
class SequenceEntry:
    binder: str
    index: int
    label: str
    relative_path: Path
    source_path: Path
    individual_pdf_path: Path
    status: str = "pending"
    warning: str = ""
    page_count: int = 0
    start_page: int = 0
    included: bool = False
    generated: bool = False


@dataclass
class BinderResult:
    binder: str
    pdf_path: Path
    entries: list[SequenceEntry] = field(default_factory=list)
    document_count: int = 0
    cover_count: int = 0
    page_count: int = 0
    missing_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    merged: bool = False


@dataclass(frozen=True)
class ReleaseProvenance:
    generated_timestamp: str
    git_branch: str
    git_commit: str
    git_dirty_status: str
    python_version: str
    pdf_theme: str
    markdown_parser: str
    pdf_merge_backend: str
    browser_executable: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the printable Research Library release package.")
    parser.add_argument("--theme", choices=["binder", "research", "paper", "github"], default="binder")
    parser.add_argument("--force", action="store_true", help="Re-export existing individual PDFs.")
    parser.add_argument("--binder", default=None, help="Build only one binder, e.g. Binder01_Research_Constitution.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--version", default="v1.0")
    return parser.parse_args()


def load_pdf_backend() -> tuple[str, Any, Any]:
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore

        return "pypdf", PdfReader, PdfWriter
    except ImportError:
        pass
    try:
        from PyPDF2 import PdfReader, PdfWriter  # type: ignore

        return "PyPDF2", PdfReader, PdfWriter
    except ImportError as exc:
        raise RuntimeError(
            "PDF merge dependency missing. Install it with: pip install pypdf"
        ) from exc


def run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", *args],
            cwd=str(REPO_ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=10,
        )
    except Exception:
        return "unavailable"
    if result.returncode != 0:
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def collect_provenance(theme: str, pdf_merge_backend: str, browser_executable: str) -> ReleaseProvenance:
    branch = run_git(["branch", "--show-current"])
    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    dirty_status = "dirty" if status and status != "unavailable" else "clean"
    if status == "unavailable":
        dirty_status = "unavailable"
    parser = available_markdown_parser() or "unavailable"
    return ReleaseProvenance(
        generated_timestamp=datetime.now().isoformat(timespec="seconds"),
        git_branch=branch,
        git_commit=commit,
        git_dirty_status=dirty_status,
        python_version=sys.version.replace("\n", " "),
        pdf_theme=theme,
        markdown_parser=parser,
        pdf_merge_backend=pdf_merge_backend,
        browser_executable=browser_executable,
    )


def select_binders(requested: str | None) -> list[Path]:
    binders = sorted(path for path in PRINT_ROOT.iterdir() if path.is_dir() and path.name.startswith("Binder"))
    if not requested:
        return binders
    matches = [path for path in binders if path.name == requested or path.name.lower().startswith(requested.lower())]
    if not matches:
        raise FileNotFoundError(f"No binder matched: {requested}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Binder selector is ambiguous: {requested}. Matches: {names}")
    return matches


def binder_display_name(binder: str) -> str:
    match = re.match(r"Binder(\d+)_(.+)", binder)
    if not match:
        return binder.replace("_", " ")
    return match.group(2).replace("_", " ")


def binder_number(binder: str) -> str:
    match = re.match(r"Binder(\d+)_", binder)
    return match.group(1) if match else ""


def document_title(relative_path: Path) -> str:
    stem = relative_path.stem
    title = re.sub(r"^\d+_COVER_", "", stem)
    title = title.replace("_", " ").replace("-", " ")
    title = re.sub(r"\s+", " ", title).strip()
    if title.lower() == "readme":
        return "README"
    return title.title() if title else relative_path.name


def make_generated_entry(
    binder: str,
    label: str,
    relative_path: Path,
    markdown_path: Path,
    output_root: Path,
) -> SequenceEntry:
    return SequenceEntry(
        binder=binder,
        index=0,
        label=label,
        relative_path=relative_path,
        source_path=markdown_path,
        individual_pdf_path=output_root / "temp" / "individual" / binder / relative_path.with_suffix(".pdf"),
        generated=True,
    )


def write_document_cover_markdown(path: Path, binder: str, document_relative: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binder_name = binder_display_name(binder)
    lines = [
        '<div class="document-cover">',
        "",
        '<p class="cover-kicker">Spacecraft AI Controller</p>',
        "",
        '<p class="cover-subtitle">Research Library</p>',
        "",
        f'<p class="cover-accent">{binder_name}</p>',
        "",
        '<hr class="cover-rule" />',
        "",
        f'<h1 class="cover-title">{document_title(document_relative)}</h1>',
        "",
        '<p class="cover-kicker">June 2026</p>',
        "",
        "</div>",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_binder_cover_markdown(path: Path, binder: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    number = binder_number(binder)
    binder_name = binder_display_name(binder)
    lines = [
        '<div class="document-cover">',
        "",
        '<p class="cover-kicker">Spacecraft AI Controller</p>',
        "",
        '<h1 class="cover-title">Research Library</h1>',
        "",
        '<hr class="cover-rule" />',
        "",
        f'<p class="cover-subtitle">Binder {number}</p>',
        "",
        f'<p class="cover-accent">{binder_name}</p>',
        "",
        '<p class="cover-subtitle">Zhixin (Sean) Li</p>',
        "",
        '<p class="cover-kicker">June 2026</p>',
        "",
        "</div>",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_sequence_entries(raw_entries: list[SequenceEntry], binder_path: Path, output_root: Path) -> list[SequenceEntry]:
    binder = binder_path.name
    normalized: list[SequenceEntry] = []
    seen_sources: set[str] = set()

    binder_cover_source = binder_path / "BINDER_COVER.md"
    if not binder_cover_source.exists():
        normalized.append(
            SequenceEntry(
                binder=binder,
                index=1,
                label="Binder Cover",
                relative_path=Path("BINDER_COVER.md"),
                source_path=binder_cover_source,
                individual_pdf_path=output_root / "temp" / "individual" / binder / "BINDER_COVER.pdf",
            )
        )
    else:
        cover_md = generated_markdown_path(output_root, f"{binder}_BINDER_COVER.md")
        write_binder_cover_markdown(cover_md, binder)
        normalized.append(
            make_generated_entry(
                binder,
                "Binder Cover",
                Path("BINDER_COVER.md"),
                cover_md,
                output_root,
            )
        )

    contents_md = generated_markdown_path(output_root, f"{binder}_TABLE_OF_CONTENTS.md")
    contents_entry = make_generated_entry(
        binder,
        "Generated Contents",
        Path("TABLE_OF_CONTENTS.md"),
        contents_md,
        output_root,
    )
    normalized.append(contents_entry)

    index = 0
    while index < len(raw_entries):
        entry = raw_entries[index]
        rel = entry.relative_path.as_posix()
        if rel in {"BINDER_COVER.md", "PRINT_SEQUENCE.md"}:
            index += 1
            continue

        if entry.label == "Cover" and index + 1 < len(raw_entries) and raw_entries[index + 1].label == "Document":
            document_entry = raw_entries[index + 1]
            source_key = str(document_entry.source_path.resolve()).lower()
            if source_key in seen_sources:
                document_entry.status = "skipped_duplicate"
                document_entry.warning = "Duplicate document source skipped by v2 sequence normalizer."
                index += 2
                continue
            seen_sources.add(source_key)
            cover_relative = Path("Generated_Cover_Pages") / f"{len(normalized):03d}_COVER_{document_entry.relative_path.stem}.md"
            cover_md = generated_markdown_path(output_root, str(Path(binder) / cover_relative).replace("\\", "_"))
            write_document_cover_markdown(cover_md, binder, document_entry.relative_path)
            normalized.append(make_generated_entry(binder, "Cover", cover_relative, cover_md, output_root))
            normalized.append(document_entry)
            index += 2
            continue

        source_key = str(entry.source_path.resolve()).lower()
        if source_key in seen_sources:
            entry.status = "skipped_duplicate"
            entry.warning = "Duplicate source skipped by v2 sequence normalizer."
            index += 1
            continue
        seen_sources.add(source_key)
        normalized.append(entry)
        index += 1

    for sequence_index, entry in enumerate(normalized, start=1):
        entry.index = sequence_index
    return normalized


def parse_print_sequence(binder_path: Path, output_root: Path) -> list[SequenceEntry]:
    sequence_path = binder_path / "PRINT_SEQUENCE.md"
    if not sequence_path.exists():
        raise FileNotFoundError(f"Missing PRINT_SEQUENCE.md for {binder_path.name}")

    pattern = re.compile(r"(Binder Cover|Binder README|Table of Contents|Cover|Document):\s+`([^`]+)`")
    entries: list[SequenceEntry] = []
    for line in sequence_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        label = match.group(1)
        relative = Path(match.group(2))
        source = binder_path / relative
        individual_pdf = output_root / "temp" / "individual" / binder_path.name / relative.with_suffix(".pdf")
        entries.append(
            SequenceEntry(
                binder=binder_path.name,
                index=len(entries) + 1,
                label=label,
                relative_path=relative,
                source_path=source,
                individual_pdf_path=individual_pdf,
            )
        )
    return entries


def export_entry(entry: SequenceEntry, executable: str, theme: str, force: bool) -> None:
    if not entry.source_path.exists():
        entry.status = "missing"
        entry.warning = f"Missing source file: {entry.source_path}"
        return
    if entry.individual_pdf_path.exists() and not force:
        entry.status = "skipped_existing"
        return
    relative_for_export = Path(entry.binder) / entry.relative_path
    export_item = ExportItem(source=entry.source_path, relative=relative_for_export, output=entry.individual_pdf_path)
    export_with_browser(export_item, executable, theme)
    if not entry.individual_pdf_path.exists() or entry.individual_pdf_path.stat().st_size <= 0:
        raise RuntimeError(f"PDF was not created or is empty: {entry.individual_pdf_path}")
    entry.status = "exported"


def calculate_start_pages(entries: list[SequenceEntry]) -> None:
    current_page = 1
    for entry in entries:
        entry.start_page = current_page
        current_page += max(entry.page_count, 0)


def contents_rows(entries: list[SequenceEntry]) -> list[tuple[int, str, int]]:
    rows: list[tuple[int, str, int]] = []
    row_index = 1
    for entry in entries:
        if entry.label in {"Binder Cover", "Generated Contents", "Cover"}:
            continue
        title = document_title(entry.relative_path)
        rows.append((row_index, title, entry.start_page))
        row_index += 1
    return rows


def write_binder_contents_markdown(path: Path, binder: str, entries: list[SequenceEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binder_name = binder_display_name(binder)
    rows = contents_rows(entries)
    lines = [
        "# Contents",
        "",
        f"**Binder {binder_number(binder)}: {binder_name}**",
        "",
        "| # | Document | Page |",
        "|---:|---|---:|",
    ]
    for row_index, title, page in rows:
        lines.append(f"| {row_index} | {title} | {page} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_sequence_with_contents(
    result: BinderResult,
    executable: str,
    theme: str,
    force: bool,
    PdfReader: Any,
) -> None:
    contents_entry = next((entry for entry in result.entries if entry.label == "Generated Contents"), None)
    if contents_entry is None:
        raise RuntimeError(f"{result.binder}: missing generated contents entry")

    for entry in result.entries:
        if entry.label == "Generated Contents":
            continue
        try:
            export_entry(entry, executable, theme, force)
            if entry.individual_pdf_path.exists():
                entry.page_count = read_page_count(entry.individual_pdf_path, PdfReader)
        except Exception as exc:  # noqa: BLE001 - release builder records per-file export failure
            entry.status = "failure"
            entry.warning = str(exc)
            result.warnings.append(f"{entry.relative_path}: {exc}")

    previous_contents_pages = -1
    for _ in range(3):
        contents_entry.page_count = max(contents_entry.page_count, 1)
        calculate_start_pages(result.entries)
        write_binder_contents_markdown(contents_entry.source_path, result.binder, result.entries)
        export_entry(contents_entry, executable, theme, True)
        contents_entry.page_count = read_page_count(contents_entry.individual_pdf_path, PdfReader)
        if contents_entry.page_count == previous_contents_pages:
            break
        previous_contents_pages = contents_entry.page_count

    calculate_start_pages(result.entries)


def read_page_count(pdf_path: Path, PdfReader: Any) -> int:
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def approximate_text_width(text: str, font_size: float) -> float:
    return len(text) * font_size * 0.48


def add_font_resource(page: Any) -> None:
    try:
        from pypdf.generic import DictionaryObject, NameObject  # type: ignore
    except ImportError:
        from PyPDF2.generic import DictionaryObject, NameObject  # type: ignore

    resources = page.get("/Resources")
    if resources is None:
        resources = DictionaryObject()
        page[NameObject("/Resources")] = resources
    font = resources.get("/Font")
    if font is None:
        font = DictionaryObject()
        resources[NameObject("/Font")] = font
    font[NameObject("/FPRINT")] = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )


def append_footer_stream(page: Any, writer: Any, binder_name: str, page_index: int, page_total: int) -> None:
    try:
        from pypdf.generic import ArrayObject, DecodedStreamObject, NameObject  # type: ignore
    except ImportError:
        from PyPDF2.generic import ArrayObject, DecodedStreamObject, NameObject  # type: ignore

    add_font_resource(page)
    width = float(page.mediabox.width)
    left_x = 54.0
    right_margin = 54.0
    y = 32.0
    line_y = 49.0
    font_size = 8.2
    footer_text = f"Spacecraft AI Controller | {binder_name} | Page {page_index} of {page_total}"
    text_x = max(left_x, (width - approximate_text_width(footer_text, font_size)) / 2.0)
    content = f"""
q
0.72 0.78 0.84 RG
0.5 w
{left_x:.2f} {line_y:.2f} m
{(width - right_margin):.2f} {line_y:.2f} l
S
0.34 0.42 0.50 rg
BT
/FPRINT {font_size:.2f} Tf
{text_x:.2f} {y:.2f} Td
({pdf_escape(footer_text)}) Tj
ET
Q
"""
    stream = DecodedStreamObject()
    stream.set_data(content.encode("latin-1"))
    stream_ref = writer._add_object(stream)
    existing = page.get("/Contents")
    if existing is None:
        page[NameObject("/Contents")] = stream_ref
    elif isinstance(existing, ArrayObject):
        existing.append(stream_ref)
    else:
        page[NameObject("/Contents")] = ArrayObject([existing, stream_ref])


def rewrite_pdf_with_metadata_and_footer(
    pdf_path: Path,
    title: str,
    footer_label: str,
    PdfReader: Any,
    PdfWriter: Any,
) -> None:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    total_pages = len(reader.pages)
    for index, page in enumerate(reader.pages, start=1):
        writer.add_page(page)
        append_footer_stream(writer.pages[-1], writer, footer_label, index, total_pages)
    writer.add_metadata(
        {
            "/Title": title,
            "/Author": PDF_AUTHOR,
            "/Subject": "Spacecraft AI Controller Research Library",
            "/Keywords": PDF_KEYWORDS,
        }
    )
    tmp = pdf_path.with_suffix(".final.tmp.pdf")
    with tmp.open("wb") as handle:
        writer.write(handle)
    tmp.replace(pdf_path)


def add_pdf_metadata(pdf_path: Path, title: str, PdfReader: Any, PdfWriter: Any) -> None:
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata(
        {
            "/Title": title,
            "/Author": PDF_AUTHOR,
            "/Subject": "Spacecraft AI Controller Research Library",
            "/Keywords": PDF_KEYWORDS,
        }
    )
    tmp = pdf_path.with_suffix(".metadata.tmp.pdf")
    with tmp.open("wb") as handle:
        writer.write(handle)
    tmp.replace(pdf_path)


def merge_binder_pdf(result: BinderResult, PdfReader: Any, PdfWriter: Any) -> None:
    writer = PdfWriter()
    for entry in result.entries:
        if entry.status == "skipped_duplicate":
            result.warnings.append(entry.warning)
            continue
        if entry.status == "missing":
            result.missing_files.append(str(entry.relative_path).replace("\\", "/"))
            if entry.relative_path.as_posix() == "BINDER_COVER.md":
                result.warnings.append("Critical file missing: BINDER_COVER.md")
                raise FileNotFoundError(f"{result.binder}: missing BINDER_COVER.md")
            continue
        if not entry.individual_pdf_path.exists():
            entry.status = "missing_pdf"
            entry.warning = f"Missing individual PDF: {entry.individual_pdf_path}"
            result.warnings.append(entry.warning)
            continue
        reader = PdfReader(str(entry.individual_pdf_path))
        entry.page_count = len(reader.pages)
        for page in reader.pages:
            writer.add_page(page)
        entry.included = True

    writer.add_metadata(
        {
            "/Title": result.binder,
            "/Author": PDF_AUTHOR,
            "/Subject": "Spacecraft AI Controller Research Library",
            "/Keywords": PDF_KEYWORDS,
        }
    )
    result.pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with result.pdf_path.open("wb") as handle:
        writer.write(handle)

    rewrite_pdf_with_metadata_and_footer(
        result.pdf_path,
        f"{result.binder} {binder_display_name(result.binder)}",
        binder_display_name(result.binder),
        PdfReader,
        PdfWriter,
    )
    result.page_count = read_page_count(result.pdf_path, PdfReader)
    result.merged = True


def generated_markdown_path(output_root: Path, filename: str) -> Path:
    path = output_root / "temp" / "generated" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def export_generated_markdown(
    markdown_path: Path,
    pdf_path: Path,
    relative: Path,
    executable: str,
    theme: str,
    PdfReader: Any,
    PdfWriter: Any,
    title: str,
) -> int:
    item = ExportItem(source=markdown_path, relative=relative, output=pdf_path)
    export_with_browser(item, executable, theme)
    rewrite_pdf_with_metadata_and_footer(pdf_path, title, "Research Library", PdfReader, PdfWriter)
    return read_page_count(pdf_path, PdfReader)


def write_library_cover(output_root: Path, version: str) -> Path:
    timestamp = datetime.now().isoformat(timespec="seconds")
    path = generated_markdown_path(output_root, "Research_Library_Cover.md")
    path.write_text(
        "\n".join(
            [
                "# Spacecraft AI Controller",
                "",
                "## Research Library",
                "",
                f"**Version:** {version}",
                "",
                "**Owner:** Zhixin (Sean) Li",
                "",
                f"**Generated:** {timestamp}",
                "",
                "**Source directory:** `print_library`",
                "",
                f"**Output directory:** `{output_root.name}`",
                "",
                "This release package is a printable copy of selected research-library staging documents. It is not a new source of scientific evidence.",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_library_index(output_root: Path, binder_results: list[BinderResult]) -> Path:
    path = generated_markdown_path(output_root, "Research_Library_Index.md")
    lines = [
        "# Research Library Index",
        "",
        "| Binder | Purpose | PDF filename | Document count | Page count | Notes |",
        "|---|---|---|---:|---:|---|",
    ]
    for result in binder_results:
        notes = "; ".join(result.warnings) if result.warnings else "OK"
        lines.append(
            f"| `{result.binder}` | {BINDER_PURPOSES.get(result.binder, '')} | `{result.pdf_path.name}` | {result.document_count} | {result.page_count} | {notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_manifest(path: Path, binder_results: list[BinderResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "binder",
                "sequence_index",
                "source_markdown_path",
                "individual_pdf_path",
                "included_in_merged_pdf",
                "merged_pdf_path",
                "page_count",
                "status",
                "warning",
            ],
        )
        writer.writeheader()
        for result in binder_results:
            for entry in result.entries:
                writer.writerow(
                    {
                        "binder": result.binder,
                        "sequence_index": entry.index,
                        "source_markdown_path": str(entry.source_path),
                        "individual_pdf_path": str(entry.individual_pdf_path),
                        "included_in_merged_pdf": "yes" if entry.included else "no",
                        "merged_pdf_path": str(result.pdf_path),
                        "page_count": entry.page_count,
                        "status": entry.status,
                        "warning": entry.warning,
                    }
                )


def write_summary(
    path: Path,
    version: str,
    provenance: ReleaseProvenance,
    binder_results: list[BinderResult],
    cover_pdf: Path,
    index_pdf: Path,
    zip_path: Path,
    command: str,
) -> None:
    total_entries = sum(len(result.entries) for result in binder_results)
    generated_pdfs = sum(1 for result in binder_results for entry in result.entries if entry.individual_pdf_path.exists())
    merged_pdfs = sum(1 for result in binder_results if result.merged)
    lines = [
        "# Research Library Release Summary",
        "",
        f"- Version: `{version}`",
        f"- Generated timestamp: `{provenance.generated_timestamp}`",
        f"- Git branch: `{provenance.git_branch}`",
        f"- Git commit hash: `{provenance.git_commit}`",
        f"- Git dirty status: `{provenance.git_dirty_status}`",
        f"- Python version: `{provenance.python_version}`",
        f"- PDF theme: `{provenance.pdf_theme}`",
        f"- Markdown parser: `{provenance.markdown_parser}`",
        f"- PDF merge backend: `{provenance.pdf_merge_backend}`",
        f"- Browser executable: `{provenance.browser_executable}`",
        f"- Total binders: {len(binder_results)}",
        f"- Total source Markdown files: {total_entries}",
        f"- Total generated individual PDFs: {generated_pdfs}",
        f"- Total merged binder PDFs: {merged_pdfs}",
        f"- Library cover PDF: `{cover_pdf}`",
        f"- Library index PDF: `{index_pdf}`",
        f"- Zip package path: `{zip_path}`",
        f"- Reproduction command: `{command}`",
        "",
        "## Per-Binder Summary",
        "",
        "| Binder | Document count | Cover count | PDF filename | Page count | Missing files | Warnings |",
        "|---|---:|---:|---|---:|---|---|",
    ]
    for result in binder_results:
        missing = "<br>".join(result.missing_files) if result.missing_files else "None"
        warnings = "<br>".join(result.warnings) if result.warnings else "None"
        lines.append(
            f"| `{result.binder}` | {result.document_count} | {result.cover_count} | `{result.pdf_path.name}` | {result.page_count} | {missing} | {warnings} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_zip(zip_path: Path, files: list[Path], output_root: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            if not file_path.exists():
                continue
            archive.write(file_path, arcname=file_path.relative_to(output_root))


def build_binder(
    binder_path: Path,
    output_root: Path,
    executable: str,
    theme: str,
    force: bool,
    PdfReader: Any,
    PdfWriter: Any,
) -> BinderResult:
    result = BinderResult(
        binder=binder_path.name,
        pdf_path=output_root / f"{binder_path.name}.pdf",
    )
    raw_entries = parse_print_sequence(binder_path, output_root)
    result.entries = normalize_sequence_entries(raw_entries, binder_path, output_root)
    result.document_count = sum(
        1 for entry in result.entries if entry.label in {"Document", "Binder README", "Table of Contents", "Binder Cover", "Generated Contents"}
    )
    result.cover_count = sum(1 for entry in result.entries if entry.label == "Cover")

    export_sequence_with_contents(result, executable, theme, force, PdfReader)
    merge_binder_pdf(result, PdfReader, PdfWriter)
    return result


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    backend_name, PdfReader, PdfWriter = load_pdf_backend()
    print(f"Using PDF merge backend: {backend_name}")
    engine, executable = detect_engine("chrome")
    print(f"Using browser PDF engine: {executable}")
    print(f"Using theme: {args.theme}")
    provenance = collect_provenance(args.theme, backend_name, executable)

    binders = select_binders(args.binder)
    binder_results: list[BinderResult] = []
    for binder_path in binders:
        print(f"Building binder: {binder_path.name}")
        binder_results.append(
            build_binder(binder_path, output_root, executable, args.theme, args.force, PdfReader, PdfWriter)
        )

    cover_md = write_library_cover(output_root, args.version)
    cover_pdf = output_root / "Research_Library_Cover.pdf"
    export_generated_markdown(
        cover_md,
        cover_pdf,
        Path("Research_Library_Cover.md"),
        executable,
        args.theme,
        PdfReader,
        PdfWriter,
        "Research Library Cover",
    )

    index_md = write_library_index(output_root, binder_results)
    index_pdf = output_root / "Research_Library_Index.pdf"
    export_generated_markdown(
        index_md,
        index_pdf,
        Path("Research_Library_Index.md"),
        executable,
        args.theme,
        PdfReader,
        PdfWriter,
        "Research Library Index",
    )

    manifest_path = output_root / "release_manifest.csv"
    summary_path = output_root / "release_summary.md"
    zip_path = output_root / f"Research_Library_{args.version}.zip"
    command = "python tools\\build_print_release.py " + " ".join(sys.argv[1:])

    write_manifest(manifest_path, binder_results)
    write_summary(summary_path, args.version, provenance, binder_results, cover_pdf, index_pdf, zip_path, command)

    package_files = [cover_pdf, index_pdf, summary_path, manifest_path]
    package_files.extend(result.pdf_path for result in binder_results if result.pdf_path.exists())
    create_zip(zip_path, package_files, output_root)

    print(f"Release summary: {summary_path}")
    print(f"Release manifest: {manifest_path}")
    print(f"Zip package: {zip_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

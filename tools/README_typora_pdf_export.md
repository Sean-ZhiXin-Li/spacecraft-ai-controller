# Typora PDF Export Tool

This tool exports every Markdown file under `print_library/` to a PDF beside the Markdown file using Typora's native PDF exporter.

## Requirements

- Windows
- Typora at `E:\Typora\Typora.exe`
- AutoHotkey v2 installed
- Typora export settings already configured for PDF

The Python script controls scanning, resume, skipping, retries, logging, progress, and PDF stability checks. AutoHotkey v2 is used only for Typora GUI automation.

## Commands

Export all pending Markdown files:

```powershell
python tools\typora_pdf_export.py
```

Export one binder:

```powershell
python tools\typora_pdf_export.py --binder Binder03
```

Export only the first 10 pending files:

```powershell
python tools\typora_pdf_export.py --limit 10
```

Re-export even if PDFs already exist:

```powershell
python tools\typora_pdf_export.py --force
```

If AutoHotkey is not auto-detected:

```powershell
python tools\typora_pdf_export.py --ahk "C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe"
```

## Outputs

- PDF beside each Markdown file.
- `print_library/export.log`
- `print_library/export_state.json`

## Resume Behavior

Rerun the same command after interruption. Existing PDFs are skipped unless `--force` is specified.

## Important Notes

- Do not use the keyboard or mouse while the exporter is running.
- Keep Typora's previous export settings set to PDF.
- The script expects Typora's Save dialog default location/name to produce `<markdown filename>.pdf` beside the Markdown file.
- Markdown files are never modified by this tool.

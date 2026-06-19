from pathlib import Path
import subprocess
import urllib.parse

ROOT = Path(r"E:\spacecraft_ai_project")
SRC = ROOT / "print_library"
OUT = ROOT / "pdf_output"

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

for md in SRC.rglob("*.md"):
    rel = md.relative_to(SRC)
    pdf = OUT / rel.with_suffix(".pdf")
    pdf.parent.mkdir(parents=True, exist_ok=True)

    html = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{
          font-family: "Segoe UI", Arial, sans-serif;
          max-width: 900px;
          margin: 40px auto;
          line-height: 1.6;
        }}
        pre {{
          background: #f6f8fa;
          padding: 12px;
          overflow-x: auto;
        }}
        code {{
          font-family: Consolas, monospace;
        }}
      </style>
    </head>
    <body>
    <pre>{md.read_text(encoding="utf-8", errors="ignore")}</pre>
    </body>
    </html>
    """

    temp_html = pdf.with_suffix(".html")
    temp_html.write_text(html, encoding="utf-8")

    url = "file:///" + urllib.parse.quote(str(temp_html).replace("\\", "/"))

    print(f"Exporting {md} -> {pdf}")

    subprocess.run([
        EDGE,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf}",
        url
    ])

print("Done.")
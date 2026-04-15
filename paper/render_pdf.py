"""
Render paper/main.html → paper/prism_paper.pdf using WeasyPrint.
Run from the repo root: python3 paper/render_pdf.py
"""
from weasyprint import HTML, CSS
from pathlib import Path

html_path = Path(__file__).parent / "main.html"
pdf_path  = Path(__file__).parent / "prism_paper.pdf"

HTML(filename=str(html_path)).write_pdf(
    str(pdf_path),
    stylesheets=[CSS(string="""
        @page { size: A4; margin: 2.2cm 2.4cm 2.4cm 2.4cm; }
    """)]
)
print(f"Written: {pdf_path}")

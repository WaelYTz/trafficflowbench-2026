"""report.md -> report.html -> TrafficFlowBench_WaelYT_report.pdf (headless Edge).

  python report/build_report.py ["Author line"]
"""
import base64, os, re, subprocess, sys
from pathlib import Path
import markdown

HERE = Path(__file__).resolve().parent
AUTHOR = sys.argv[1] if len(sys.argv) > 1 else "Zitouni Ouail Imad eddine · Team WaelYT · ESI Algiers"
EDGE = os.environ.get("EDGE", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")  # any Chromium works

md = (HERE / "report.md").read_text(encoding="utf-8").replace("{{AUTHOR}}", AUTHOR)
body = markdown.markdown(md, extensions=["tables", "sane_lists"])
body = re.sub(r'src="([^"]+\.png)"', lambda m: 'src="data:image/png;base64,' + base64.b64encode((HERE / m.group(1)).read_bytes()).decode() + '"', body)
css = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 17pt; margin: 0 0 4pt; line-height: 1.2; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 10pt 0 3pt; break-after: avoid; }
p, li { margin: 3pt 0; }
ul, ol { padding-left: 18pt; margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 8.8pt; break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #eef1f5; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #f3f3f3; padding: 0 2pt; }
img { display: block; max-width: 82%; margin: 8pt auto 2pt; }
em { color: #444; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><title>TrafficFlowBench WaelYT report</title><style>{css}</style></head><body>{body}</body></html>"
(HERE / "report.html").write_text(html, encoding="utf-8")
pdf = HERE / "TrafficFlowBench_WaelYT_report.pdf"
subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                (HERE / "report.html").as_uri()], check=True, timeout=120)
print("wrote", pdf, pdf.stat().st_size // 1024, "KB")

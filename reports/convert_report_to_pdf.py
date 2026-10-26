import os
import json
import subprocess
import shutil

base_dir = r"C:\Users\Ankit Umadi\Documents\big\reports"
md_path = os.path.join(base_dir, "technical_report.md")
html_path = os.path.join(base_dir, "technical_report_printable.html")
pdf_output = os.path.join(base_dir, "technical_report.pdf")
chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

with open(md_path, "r", encoding="utf-8") as f:
    md_content = f.read()

md_json = json.dumps(md_content)

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BDSRAE Technical Report - Topic 19</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.1/github-markdown.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  @page {{
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
  }}
  body {{
    background-color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: #1f2937;
    line-height: 1.5;
    padding: 0;
    margin: 0;
  }}
  .markdown-body {{
    box-sizing: border-box;
    max-width: 950px;
    margin: 0 auto;
    font-size: 10.5pt;
  }}
  table {{
    font-size: 9pt;
  }}
  pre {{
    font-size: 8pt;
  }}
</style>
</head>
<body>
<div class="markdown-body" id="rendered-content"></div>
<script>
  const rawMarkdown = {md_json};
  document.getElementById('rendered-content').innerHTML = marked.parse(rawMarkdown);
  renderMathInElement(document.getElementById('rendered-content'), {{
    delimiters: [
      {{ left: '$$', right: '$$', display: true }},
      {{ left: '$', right: '$', display: false }}
    ],
    throwOnError: false
  }});
</script>
</body>
</html>"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_template)

cmd = [
    chrome_path,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--virtual-time-budget=9000",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={pdf_output}",
    html_path
]

print("Executing Chrome headless PDF render...")
res = subprocess.run(cmd, capture_output=True, text=True)
if os.path.exists(pdf_output):
    print(f"Success! PDF generated at: {pdf_output} ({os.path.getsize(pdf_output)/1024:.1f} KB)")
else:
    print("PDF generation failed:", res.stdout, res.stderr)

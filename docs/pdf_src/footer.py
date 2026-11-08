"""Add 'title ... page / total' footers and PDF metadata: python footer.py raw.pdf out.pdf"""
import io, sys
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

TITLE = "penspin-graph — the project explained step by step"
src, dst = sys.argv[1], sys.argv[2]
reader = PdfReader(src)
n = len(reader.pages)
pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
writer = PdfWriter()
for i, page in enumerate(reader.pages):
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(w, h))
    c.setFont("DejaVu", 7.5)
    c.setFillColorRGB(0.42, 0.45, 0.5)
    c.drawString(40, 22, TITLE)
    c.drawRightString(w - 40, 22, f"{i + 1} / {n}")
    c.setStrokeColorRGB(0.85, 0.87, 0.9)
    c.line(40, 32, w - 40, 32)
    c.save()
    buf.seek(0)
    page.merge_page(PdfReader(buf).pages[0])
    writer.add_page(page)
writer.add_metadata({"/Title": TITLE, "/Author": "Zijun Xiang (with Claude)", "/Subject": "Wuji Hand 2 pen spinning: motion-graph pipeline vs Wuji official"})
with open(dst, "wb") as f:
    writer.write(f)
print(f"{dst}: {n} pages")

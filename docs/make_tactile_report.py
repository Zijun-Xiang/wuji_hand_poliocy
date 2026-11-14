#!/usr/bin/env python3
"""One-page PDF: does the connected Wuji Hand 2 have tactile sensors? Verdict + evidence."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
OUT = HERE / "Wuji_Hand2_tactile_sensor_verdict.pdf"
PHOTO = HERE / "thumb_fingertip_flex.png"

pdfmetrics.registerFont(TTFont("CJK", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("CJKB", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", subfontIndex=0))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("CJK", normal="CJK", bold="CJKB", italic="CJK", boldItalic="CJKB")

INK, MUTED, RED, GREEN, BAND = colors.HexColor("#15181f"), colors.HexColor("#5c6370"), colors.HexColor("#b42318"), colors.HexColor("#067647"), colors.HexColor("#f3f4f6")
base = ParagraphStyle("base", fontName="CJK", fontSize=9.2, leading=12.2, textColor=INK, alignment=TA_LEFT)
small = ParagraphStyle("small", parent=base, fontSize=8.0, leading=10.2, textColor=MUTED)
h = ParagraphStyle("h", parent=base, fontName="CJKB", fontSize=10.5, leading=13, spaceBefore=5, spaceAfter=2)
title = ParagraphStyle("title", parent=base, fontName="CJKB", fontSize=15, leading=18)
cell = ParagraphStyle("cell", parent=base, fontSize=8.4, leading=10.6)
cellb = ParagraphStyle("cellb", parent=cell, fontName="CJKB")


def P(t, s=base):
    return Paragraph(t, s)


doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=11 * mm, bottomMargin=9 * mm,
                        title="Wuji Hand 2 tactile sensor verdict", author="penspin-graph deploy diagnostics")
W = letter[0] - 28 * mm
story = []
story.append(P("Does the connected Wuji Hand 2 have tactile sensors? 这只手有没有触觉传感器？", title))
story.append(P("Device: Wuji Hand 2.2 (Beta), right hand, serial WH2KA01260824017, hw 0.2.0 · firmware v2.5.1 → v3.0.0 (upgraded 2026-11-13) · "
               "wuji_sdk 2026.11.9 · Ethernet/UDP 192.168.1.111 · tests run 2026-11-13, 18:38–20:04 EST", small))
story.append(Spacer(1, 4))

verdict = Table([[P("<font color='#b42318' size='12'>Verdict: NO — no usable tactile sensor (无可用的触觉传感器).</font> The hand's controller never finds fingertip tactile electronics "
                    "on any of the five finger buses, so no tactile data can be read. The fingertips do carry sensor-style pads with a flex circuit "
                    "(柔性电路板, FPC), so this unit is either the <b>non-tactile build</b> of the 2.2 Beta or has its fingertip electronics "
                    "disconnected on all five fingers — from the computer's side both mean the same: no tactile sensing.", base)]],
                colWidths=[W])
verdict.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1.2, RED), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4f2")),
                             ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
story.append(verdict)

story.append(P("<font color=\"#1f3a93\">How it was identified (鉴定方法)</font>", h))
story.append(P("The Wuji Hand 2 puts each fingertip tactile module on the finger's internal bus as <b>node 5</b> (nodes 1–4 are the joints). "
               "Six independent checks were read through the official SDK and Wuji CLI, repeated with motors off and on, after a software "
               "reboot, and again after flashing the current firmware v3.0.0 — the hand's own bookkeeping, not a guess from the data stream.", base))
story.append(Spacer(1, 3))

rows = [[P("Check (检查项)", cellb), P("Expected if tactile modules exist", cellb), P("Observed — v2.5.1 and v3.0.0, all identical", cellb)],
        [P("<b>node_online</b> bitmask (bus enumeration, 总线枚举)", cell), P("bit of node 5 set on each finger", cell),
         P("0xF7BDEF: joints 1–4 online on all 5 fingers, <b>node 5 absent on all 5</b>", cell)],
        [P("<b>slave_version</b> table", cell), P("25 entries (20 joints + 5 tactile)", cell), P("20 entries — joints only", cell)],
        [P("<b>fingertip_online()</b> / comm <b>tactile_online_mask</b>", cell), P("True / bits set", cell), P("[False]×5 / 0", cell)],
        [P("<b>tactile_timeout_total</b> (polls that failed)", cell), P("&gt; 0 if a module were fitted but dead", cell), P("[0,0,0,0,0] — the master never polls: nothing was enumerated at boot", cell)],
        [P("<b>fingertip_&lt;f&gt;_status / info</b>", cell), P("model + state, 40/34-point descriptor", cell),
         P("v2.5.1: InternalError; v3.0.0: <b>NodeOfflineError “node not on the bus; check that the fingertip module is connected”</b>", cell)],
        [P("<b>fingertip/&lt;f&gt;/data</b> streams, 3 s, pads pressed", cell), P("~100 Hz frames", cell), P("0 frames on all five", cell)],
        [P("Joints (control check, 对照)", cell), P("20/20 online", cell), P("20/20 online; playback lap mean joint error 1.85° — the hand itself is healthy", cell)]]
t = Table(rows, colWidths=[W * 0.30, W * 0.26, W * 0.44], repeatRows=1)
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BAND), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
                       ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#d0d3da")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                       ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
story.append(t)

story.append(P("<font color=\"#1f3a93\">Why the answer is “no” (理由)</font>", h))
reasons = [
    "<b>1. The electronics are absent from the bus, not merely silent.</b> A fitted-but-faulty module makes the controller poll it and time out; here the "
    "timeout counters stay at 0 and node 5 is never listed. All five fingers show the same, twice (before/after reboot) and on two firmware versions.",
    "<b>2. Software is ruled out.</b> Firmware v3.0.0 pairs with the installed SDK 2026.11.9 (Wuji's current pairing); the tactile API requires only "
    "firmware ≥ v2.1.0 and SDK ≥ 2026.7.21. Motors enabled vs. disabled made no difference. Wuji CLI's <i>doctor</i> finds the hand but has no tactile verdict for Hand 2.",
    "<b>3. The hardware is consistent with the non-tactile 2.2 Beta build.</b> Wuji's current Hand 2.2 (Beta) manual is explicitly the "
    "“Non-Tactile” version and that variant has fingertip soft pads; the tactile variant adds a 40-point (thumb) / 34-point (other fingers) 3-axis "
    "force array per fingertip at 100 Hz. The pads on this unit carry a flex circuit (photo), so sensor pads may be mounted — but no fingertip "
    "electronics answer. Only Wuji can say whether this is a non-tactile assembly or a disconnected one; both give no tactile data.",
]
for r in reasons:
    story.append(P(r, base))
    story.append(Spacer(1, 1.5))

if PHOTO.is_file():
    img = Image(str(PHOTO), width=56 * mm, height=48.2 * mm)
    cap = P("Thumb fingertip with the silicone pad pulled back: metal core with an orange flex circuit — sensor-style hardware, yet node 5 never appears on the bus.", small)
    bottom = Table([[img, cap, P("<b>Evidence files</b> (runs/tactile/ in the <i>wuji_hand_poliocy</i> repo):<br/>probe_2026-11-13_184739.txt · reboot_probe_2026-11-13_192609.txt · "
                                 "wuji_cli_2026-11-13_195354.txt · wuji_upgrade_2026-11-13_200001.txt · WUJI_SUPPORT_SUMMARY.md<br/><br/>"
                                 "<b>What would change the verdict:</b> fingertip_online() returning True for any finger, or node 5 appearing in node_online. "
                                 "The live viewer (deploy/tactile_viewer.py) re-checks every 2 s and will show the data at once.", small)]],
                   colWidths=[59 * mm, 47 * mm, W - 106 * mm])
    bottom.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4)]))
    story.append(Spacer(1, 2))
    story.append(bottom)

doc.build(story)
print("wrote", OUT)

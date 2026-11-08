"""Hand-built SVG figures for the walkthrough PDF (measured text, no overlaps)."""
from pathlib import Path
from PIL import ImageFont

FONT_DIR = Path("/usr/share/fonts/opentype/inter")
_fonts = {}


def text_w(s, size, weight="regular"):
    key = (size, weight)
    if key not in _fonts:
        name = {"regular": "Inter-Regular.otf", "semibold": "Inter-SemiBold.otf", "bold": "Inter-Bold.otf"}[weight]
        _fonts[key] = ImageFont.truetype(str(FONT_DIR / name), size=size * 4)
    return _fonts[key].getlength(s) / 4.0


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


PH = {  # phase colours: stroke, fill, text
    "data": ("#2563eb", "#eff6ff", "#1e3a8a"),
    "graph": ("#4f46e5", "#eef2ff", "#312e81"),
    "phys": ("#059669", "#ecfdf5", "#064e3b"),
    "ctrl": ("#d97706", "#fffbeb", "#78350f"),
    "wuji": ("#64748b", "#f1f5f9", "#1e293b"),
    "none": ("#94a3b8", "#ffffff", "#64748b"),
}
FAM = "Inter, 'Noto Sans CJK SC', sans-serif"
CHECK = []


def fit(s, size, weight, maxw, where):
    w = text_w(s, size, weight)
    if w > maxw:
        CHECK.append(f"OVERFLOW {where}: '{s}' {w:.0f}>{maxw:.0f}")
    return s


def arrow_defs():
    return (
        "<defs>"
        "<marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
        "<path d='M0,0 L10,5 L0,10 z' fill='#334155'/></marker>"
        "<marker id='ahr' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
        "<path d='M0,0 L10,5 L0,10 z' fill='#dc2626'/></marker>"
        "</defs>"
    )


# --------------------------------------------------------------------------- #
# Figure 1: our pipeline as a snake of 11 numbered boxes
# --------------------------------------------------------------------------- #
def fig_flow():
    W, BW, BH, GAP = 720, 158, 78, 20
    xs = [14 + i * (BW + GAP) for i in range(4)]
    ys = [10, 132, 254]
    boxes = [
        (1, "data", "Load clips", ["40 clips · 25,885 frames", "into the wrist frame"]),
        (2, "data", "Embed frames", ["each frame → 67-D point", "z-score × block weight"]),
        (3, "graph", "Propose jumps", ["kNN (K = 12) + 5-frame", "window: 101,934"]),
        (4, "graph", "Chain tricks", ["hold of clip A → start", "of clip B: 600"]),
        (5, "graph", "Kinematic filter", ["≤ 5.81 rad/s, blend 2–10", "5,403 removed"]),
        (6, "phys", "Wuji tracking policy", ["released ONNX, unchanged", "residual + EMA"]),
        (7, "phys", "Validate in MuJoCo", ["policy 6 in the loop", "38,624 / 40,208 pass"]),
        (8, "phys", "Loops + SCC", ["core: 15,215 frames", "replay, prune, repeat"]),
        (9, "ctrl", "Controller", ["nearest point → plan", "→ circle a loop"]),
        (10, "ctrl", "Policy", ["PPO export (ready)", "BC (trained)"]),
        (11, "ctrl", "Evaluate", ["18.0 s → 84.7 s", "1,476° → 8,709°"]),
    ]
    pos = {1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0), 5: (3, 1), 6: (2, 1), 7: (1, 1), 8: (0, 1), 9: (0, 2), 10: (1, 2), 11: (2, 2)}
    H = ys[2] + BH + 6
    out = [f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' font-family=\"{FAM}\" width='100%'>", arrow_defs()]
    for n, ph, title, lines in boxes:
        c, r = pos[n]
        x, y = xs[c], ys[r]
        stroke, fill, tc = PH[ph]
        dash = " stroke-dasharray='5 3'" if n == 6 else ""
        out.append(f"<rect x='{x}' y='{y}' width='{BW}' height='{BH}' rx='9' fill='{fill}' stroke='{stroke}' stroke-width='1.6'{dash}/>")
        out.append(f"<circle cx='{x + 17}' cy='{y + 17}' r='11' fill='{stroke}'/>")
        out.append(f"<text x='{x + 17}' y='{y + 21}' font-size='11' font-weight='700' fill='white' text-anchor='middle'>{n}</text>")
        out.append(f"<text x='{x + 34}' y='{y + 22}' font-size='11.5' font-weight='700' fill='{tc}'>{esc(fit(title, 11.5, 'bold', BW - 40, f'box{n}'))}</text>")
        for i, ln in enumerate(lines):
            out.append(f"<text x='{x + 10}' y='{y + 47 + i * 16}' font-size='10' fill='#334155'>{esc(fit(ln, 10, 'regular', BW - 18, f'box{n}'))}</text>")
        if n == 6:
            tag = "= Wuji"
            tw = text_w(tag, 8.5, "bold") + 10
            out.append(f"<rect x='{x + BW - tw - 6}' y='{y - 8}' width='{tw}' height='15' rx='7' fill='#059669'/>")
            out.append(f"<text x='{x + BW - tw / 2 - 6}' y='{y + 3}' font-size='8.5' font-weight='700' fill='white' text-anchor='middle'>{tag}</text>")
    cy = lambda r: ys[r] + BH / 2

    def harrow(c0, c1, r):
        if c1 > c0:
            x0, x1 = xs[c0] + BW + 2, xs[c1] - 2
        else:
            x0, x1 = xs[c0] - 2, xs[c1] + BW + 2
        out.append(f"<line x1='{x0}' y1='{cy(r)}' x2='{x1}' y2='{cy(r)}' stroke='#334155' stroke-width='1.6' marker-end='url(#ah)'/>")

    harrow(0, 1, 0); harrow(1, 2, 0); harrow(2, 3, 0)
    xd = xs[3] + BW / 2
    out.append(f"<line x1='{xd}' y1='{ys[0] + BH + 2}' x2='{xd}' y2='{ys[1] - 2}' stroke='#334155' stroke-width='1.6' marker-end='url(#ah)'/>")
    harrow(3, 2, 1); harrow(2, 1, 1); harrow(1, 0, 1)
    xd = xs[0] + BW / 2
    out.append(f"<line x1='{xd}' y1='{ys[1] + BH + 2}' x2='{xd}' y2='{ys[2] - 2}' stroke='#334155' stroke-width='1.6' marker-end='url(#ah)'/>")
    harrow(0, 1, 2); harrow(1, 2, 2)
    x8, x7 = xs[0] + BW / 2, xs[1] + BW / 2
    out.append(f"<path d='M{x8},{ys[1] - 2} Q{(x8 + x7) / 2},{ys[1] - 40} {x7},{ys[1] - 2}' fill='none' stroke='#dc2626' stroke-width='1.5' stroke-dasharray='4 3' marker-end='url(#ahr)'/>")
    lab = "prune failed jumps, search again"
    out.append(f"<text x='{(x8 + x7) / 2}' y='{ys[1] - 27}' font-size='9.5' font-weight='600' fill='#dc2626' text-anchor='middle'>{lab}</text>")
    lx, ly = xs[3], ys[2]
    out.append(f"<rect x='{lx}' y='{ly}' width='{BW}' height='{BH}' rx='9' fill='white' stroke='#cbd5e1'/>")
    items = [("data", "data"), ("graph", "graph building"), ("phys", "physics test"), ("ctrl", "control + learning")]
    for i, (ph, name) in enumerate(items):
        stroke, fill, _ = PH[ph]
        yy = ly + 15 + i * 14.5
        out.append(f"<rect x='{lx + 10}' y='{yy - 9}' width='14' height='11' rx='3' fill='{fill}' stroke='{stroke}' stroke-width='1.4'/>")
        out.append(f"<text x='{lx + 30}' y='{yy}' font-size='9.5' fill='#334155'>{esc(name)}</text>")
    out.append(f"<text x='{lx + 10}' y='{ly + 72}' font-size='9' fill='#334155'>dashed box = shared with Wuji</text>")
    fit("dashed box = shared with Wuji", 9, "regular", BW - 14, "legend")
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Figure 2: side-by-side lanes, Wuji official vs ours, one row per stage
# --------------------------------------------------------------------------- #
def fig_lanes():
    W = 720
    LX, RX, BW = 14, 432, 274
    MID = (LX + BW + RX) / 2
    top, RH, BH = 58, 66, 52
    rows = [
        ("same", ("Recorded clips", "40 clips, wrist-local, one fixed wrist mount"), ("Same clips, wrist frame", "+ velocities, shaft axis, resting flags")),
        ("new", ("(no state embedding)", "reference = clip c, frame k (by the clock)"), ("Every frame → 67-D point", "z-score × weights; distance in steps δ")),
        ("new", ("Only t → t+1 inside a clip", "clip end = episode end (reference_end)"), ("Candidate jumps", "kNN + 5-frame window, cost ≤ 6: 101,934")),
        ("new", ("Each clip ends in a hold", "next trick = new run from its frame 0"), ("Hold → start of another clip", "slow 0.5–1 s blends: 600 candidates")),
        ("new", ("Clips used as recorded", "safety only in the action path"), ("Kinematic filter + blends", "≤ 5.81 rad/s, 2–10 frames: 5,403 removed")),
        ("same", ("PPO tracking policy (trained)", "4096 envs, DR, ~10k iterations, 5 groups"), ("The same released policies", "weights unchanged, same obs/action path")),
        ("new", ("Checked only by training on clips", "terminates at 5 cm / 60° pen error"), ("Each jump tested in MuJoCo", "5 cm/60° + ≤ 1.5× baseline: 38,624 pass")),
        ("new", ("No cycles", "longest motion = one clip (18.0 s here)"), ("Loops + SCC, replay & prune", "core: 15,215 frames (59%), 151 loops")),
        ("changed", ("Deploy: frame k = time since start", "starts at frame 0, stops at the last frame"), ("Nearest point → plan → circle", "re-match when pen > 3 cm or > 40°")),
        ("changed", ("PPO per motion group", "needs a reference clip at run time"), ("PPO export + BC", "stitched clips ready; BC needs no reference")),
        ("changed", ("Goal: track one clip to its end", "best here: 18.0 s, 1,476°"), ("Goal: keep spinning from any start", "best: 84.7 s, 8,709°")),
    ]
    H = top + RH * len(rows) + 6
    out = [f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' font-family=\"{FAM}\" width='100%'>", arrow_defs()]
    out.append(f"<rect x='{LX}' y='8' width='{BW}' height='34' rx='8' fill='#475569'/>")
    out.append(f"<text x='{LX + BW / 2}' y='30' font-size='12' font-weight='700' fill='white' text-anchor='middle'>Wuji official (wuji-mjlab) · 官方方法</text>")
    out.append(f"<rect x='{RX}' y='8' width='{BW}' height='34' rx='8' fill='#1d4ed8'/>")
    out.append(f"<text x='{RX + BW / 2}' y='30' font-size='12' font-weight='700' fill='white' text-anchor='middle'>This project (penspin-graph) · 本项目</text>")
    out.append(f"<text x='{MID}' y='30' font-size='10.5' font-weight='700' fill='#334155' text-anchor='middle'>stage</text>")
    chip = {"same": ("#d1fae5", "#065f46", "SAME"), "changed": ("#fef3c7", "#92400e", "CHANGED"), "new": ("#dbeafe", "#1e40af", "NEW")}
    for i, (verdict, (lt, ld), (rt, rd)) in enumerate(rows):
        y = top + i * RH
        none = lt.startswith("(")
        s, f, tc = PH["none"] if none else PH["wuji"]
        dash = " stroke-dasharray='4 3'" if none else ""
        out.append(f"<rect x='{LX}' y='{y}' width='{BW}' height='{BH}' rx='8' fill='{f}' stroke='{s}' stroke-width='1.3'{dash}/>")
        out.append(f"<text x='{LX + 10}' y='{y + 20}' font-size='11' font-weight='700' fill='{tc}'>{esc(fit(lt, 11, 'bold', BW - 18, f'L{i}'))}</text>")
        out.append(f"<text x='{LX + 10}' y='{y + 39}' font-size='9.8' fill='#334155'>{esc(fit(ld, 9.8, 'regular', BW - 18, f'L{i}'))}</text>")
        s, f, tc = PH["data"]
        out.append(f"<rect x='{RX}' y='{y}' width='{BW}' height='{BH}' rx='8' fill='{f}' stroke='{s}' stroke-width='1.4'/>")
        out.append(f"<text x='{RX + 10}' y='{y + 20}' font-size='11' font-weight='700' fill='{tc}'>{esc(fit(rt, 11, 'bold', BW - 18, f'R{i}'))}</text>")
        out.append(f"<text x='{RX + 10}' y='{y + 39}' font-size='9.8' fill='#334155'>{esc(fit(rd, 9.8, 'regular', BW - 18, f'R{i}'))}</text>")
        # centre: number + verdict chip, joined to both boxes by thin lines
        cyy = y + BH / 2
        out.append(f"<line x1='{LX + BW}' y1='{cyy}' x2='{RX}' y2='{cyy}' stroke='#cbd5e1' stroke-width='1'/>")
        out.append(f"<circle cx='{MID}' cy='{y + 15}' r='11' fill='#1e3a8a'/>")
        out.append(f"<text x='{MID}' y='{y + 19}' font-size='10.5' font-weight='700' fill='white' text-anchor='middle'>{i + 1}</text>")
        bg, fg, word = chip[verdict]
        cw = text_w(word, 8.5, "bold") + 14
        out.append(f"<rect x='{MID - cw / 2}' y='{y + 31}' width='{cw}' height='15' rx='7.5' fill='{bg}' stroke='{fg}' stroke-width='0.6'/>")
        out.append(f"<text x='{MID}' y='{y + 42}' font-size='8.5' font-weight='700' fill='{fg}' text-anchor='middle'>{word}</text>")
        if i < len(rows) - 1:
            for x in (LX + BW / 2, RX + BW / 2):
                out.append(f"<line x1='{x}' y1='{y + BH + 1}' x2='{x}' y2='{y + RH - 1}' stroke='#334155' stroke-width='1.3' marker-end='url(#ah)'/>")
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    Path("fig_flow.svg").write_text(fig_flow())
    Path("fig_lanes.svg").write_text(fig_lanes())
    print("\n".join(CHECK) if CHECK else "all figure text fits")

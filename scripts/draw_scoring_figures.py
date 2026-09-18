#!/usr/bin/env python3
"""Rebuild the static, illustrative scoring figures as hand-authored SVG.

Standard library only (no matplotlib). No run data is used: the numbers are
the illustrative examples from scripts/backups/prepare_scoring_examples.py.

Output: assets/images/weirdml-v3/scoring/{submissions,submissions-mobile,
area-early,area-late,hints,hints-mobile}.svg
"""
from math import log10
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'assets/images/weirdml-v3/scoring'

# Design tokens (assets/css/weirdml-v3-design.css).
INK, INK2, MUTED, FAINT = '#17212b', '#3a4a58', '#6a7987', '#93a1ad'
ACCENT, ACCENT_SOFT, LINE, LINE_STRONG = '#0e6f88', '#e6f2f5', '#e4e8ec', '#cfd6dc'
SURFACE, SURFACE2 = '#ffffff', '#f4f5f6'
ROSE = '#b9536b'  # worse attempts only
SANS = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'JetBrains Mono', ui-monospace, Menlo, monospace"

STYLE = f"""
.t{{font-family:{SANS};fill:{INK2};font-size:12px}}
.m{{font-family:{MONO};fill:{MUTED};font-size:12px}}
.h{{paint-order:stroke;stroke:{SURFACE};stroke-width:4px;stroke-linejoin:round}}
.lab{{fill:{MUTED}}}
.strong{{fill:{INK};font-weight:600}}
.grid{{stroke:{LINE};stroke-width:1}}
.axis{{stroke:{LINE_STRONG};stroke-width:1}}
.step{{fill:none;stroke:{ACCENT};stroke-width:2.25;stroke-linejoin:round;stroke-linecap:round}}
.num{{font-family:{SANS};font-weight:600;fill:{SURFACE};text-anchor:middle}}
.numw{{fill:{ROSE}}}
"""


class SVG:
    def __init__(self, w, h, title):
        self.w, self.h = w, h
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
                      f'role="img" aria-label="{title}">',
                      f'<style>{STYLE}</style>',
                      f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>']

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, cls='t', anchor='start', size=None, extra=''):
        sz = f' font-size="{size}"' if size else ''
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{sz}{extra}>{s}</text>')

    def write(self, name):
        self.add('</svg>')
        (OUT / f'{name}.svg').write_text('\n'.join(self.parts) + '\n')


# --------------------------------------------------------------------------
# Shared axis frame for the log-token charts.
# --------------------------------------------------------------------------
class Frame:
    def __init__(self, svg, x0, y0, x1, y1, tmin, tmax, ticks, fs=12, xtitle=True):
        self.svg, self.x0, self.y0, self.x1, self.y1 = svg, x0, y0, x1, y1
        self.lmin, self.lmax = log10(tmin), log10(tmax)
        self.fs = fs
        # Horizontal hairlines and y labels.
        for v in (0, .25, .5, .75, 1):
            y = self.y(v)
            svg.add(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>')
            svg.text(x0 - 10, y + fs * .36, {0: '0', 1: '1'}.get(v, f'{v:.2f}'[1:]),
                     'm', 'end', fs)
        # Baseline slightly stronger.
        svg.add(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis"/>')
        for t, label in ticks:
            x = self.x(t)
            svg.add(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y1 + 5}" class="axis"/>')
            anchor = 'start' if t == tmin else 'end' if t == tmax else 'middle'
            svg.text(x, y1 + fs + 12, label, 'm', anchor, fs)
        if xtitle:
            svg.text((x0 + x1) / 2, y1 + fs * 2 + 24, 'Total tokens (log scale)', 't lab',
                     'middle', fs)

    def x(self, t):
        return self.x0 + (log10(t) - self.lmin) / (self.lmax - self.lmin) * (self.x1 - self.x0)

    def y(self, v):
        return self.y1 - v * (self.y1 - self.y0)

    def step_path(self, pts):
        # pts: [(token, value)], value holds until the next point.
        d = f'M{self.x(pts[0][0]):.1f},{self.y(pts[0][1]):.1f}'
        for (t, v), (t2, _v2) in zip(pts, pts[1:]):
            d += f' H{self.x(t2):.1f} V{self.y(_v2):.1f}'
        return d


# --------------------------------------------------------------------------
# Figure 1: ten submissions, best-so-far.
# --------------------------------------------------------------------------
SCORES = [.12, .31, .26, .47, .41, .62, .58, .55, .73, .68]
TOKENS = [t * 1e6 for t in (.15, .35, .7, 1.4, 2.8, 5, 8, 14, 23, 38)]  # illustrative


def submissions(mobile=False):
    W, H = (420, 470) if mobile else (900, 430)
    fs = 13 if mobile else 12
    svg = SVG(W, H, 'Ten illustrative submissions and the best-so-far curve')
    x0, x1 = (44, W - 30) if mobile else (48, W - 24)
    y0 = 84 if mobile else 48
    y1 = H - (fs * 2 + 34)

    # Header row: y-axis title (left) and legend (right / second row on mobile).
    svg.text(x0 - 34 if mobile else x0 - 38, 20, 'Effective score', 't strong', 'start', fs)
    lx = x0 - 34 if mobile else None
    ly = 50 if mobile else 20
    items = [('line', ACCENT, 'Best so far'), ('dot', ACCENT, 'New best'), ('dot', ROSE, 'Worse attempt')]
    widths = [fs * 0.58 * len(s) + 30 for _, _, s in items]
    x = lx if mobile else x1 - sum(widths) - 12 * (len(items) - 1)
    for (kind, color, label), w in zip(items, widths):
        if kind == 'line':
            svg.add(f'<line x1="{x}" y1="{ly - 4}" x2="{x + 18}" y2="{ly - 4}" stroke="{color}" stroke-width="2.25" stroke-linecap="round"/>')
        elif color == ROSE:
            svg.add(f'<circle cx="{x + 9}" cy="{ly - 4}" r="4.75" fill="{SURFACE}" stroke="{color}" stroke-width="1.5"/>')
        else:
            svg.add(f'<circle cx="{x + 9}" cy="{ly - 4}" r="5.5" fill="{color}"/>')
        svg.text(x + 26, ly, label, 't', 'start', fs)
        x += w + 12

    fr = Frame(svg, x0, y0, x1, y1, 1e5, 5e7,
               [(1e5, '100k'), (5e5, '500k'), (5e6, '5M'), (5e7, '50M')], fs)

    # Scoring window 500k-50M as a faint band with a quiet label.
    bx = fr.x(5e5)
    svg.add(f'<rect x="{bx:.1f}" y="{y0}" width="{x1 - bx:.1f}" height="{y1 - y0}" fill="{ACCENT}" fill-opacity="0.045"/>')
    svg.add(f'<line x1="{bx:.1f}" y1="{y0}" x2="{bx:.1f}" y2="{y1}" stroke="{ACCENT}" stroke-opacity="0.35" stroke-width="1" stroke-dasharray="2 3"/>')
    svg.text(bx + 8, y0 + fs + 2, 'Scoring window 500k – 50M', 't lab', 'start', fs - 1)

    best, pts = 0, [(1e5, 0)]
    for t, s in zip(TOKENS, SCORES):
        if s > best:
            best = s
            pts.append((t, s))
    pts.append((5e7, best))
    svg.add(f'<path d="{fr.step_path(pts)}" class="step"/>')

    # Worse attempts: dotted drop from the retained best down to the attempt.
    running = 0
    r = 11 if mobile else 10
    for i, (t, s) in enumerate(zip(TOKENS, SCORES), 1):
        running = max(running, s)
        worse = s < running
        cx, cy = fr.x(t), fr.y(s)
        if worse:
            svg.add(f'<line x1="{cx:.1f}" y1="{fr.y(running):.1f}" x2="{cx:.1f}" y2="{cy:.1f}" stroke="{ROSE}" stroke-width="1.25" stroke-dasharray="2 3"/>')
    for i, (t, s) in enumerate(zip(TOKENS, SCORES), 1):
        worse = s < max(SCORES[:i])
        cx, cy = fr.x(t), fr.y(s)
        # Worse attempts are hollow (shape, not only hue, separates the two kinds).
        svg.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r + 2}" fill="{SURFACE}"/>')
        if worse:
            svg.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r - 0.75}" fill="{SURFACE}" stroke="{ROSE}" stroke-width="1.5"/>')
            svg.text(cx, cy + 3.6, str(i), 'num numw', 'middle', 10.5)
        else:
            svg.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{ACCENT}"/>')
            svg.text(cx, cy + 3.6, str(i), 'num', 'middle', 10.5)

    # Annotations: best carried forward, and the last attempt.
    yb = fr.y(.73)
    svg.text(x1, yb - 14, 'Best stays at <tspan class="m strong" fill="' + INK + '">0.73</tspan>', 't h', 'end', fs)
    x10, y10 = fr.x(TOKENS[-1]), fr.y(SCORES[-1])
    ax, anchor, afs = (W - 4, 'end', fs - 1) if mobile else (x10, 'middle', fs)
    svg.text(ax, y10 + r + afs + 14, 'Last attempt', 't h', anchor, afs)
    svg.text(ax, y10 + r + afs * 2 + 18, '0.68', 'm strong h', anchor, afs)
    svg.write('submissions-mobile' if mobile else 'submissions')


# --------------------------------------------------------------------------
# Figure 2: area under the best-so-far curve (two panels).
# --------------------------------------------------------------------------
def area(token, name, area_value):
    W, H = 440, 300
    fs = 12
    svg = SVG(W, H, f'Best-so-far curve stepping from 0.20 to 0.80 at {token / 1e6:g}M tokens')
    x0, x1, y0, y1 = 48, W - 20, 40, H - (fs * 2 + 34)
    svg.text(x0 - 38, 20, 'Best so far', 't strong', 'start', fs)
    ticks = [(5e5, '500k'), (5e6, '5M'), (5e7, '50M')]
    if token not in [t for t, _ in ticks]:
        ticks.insert(-1, (token, f'{token / 1e6:g}M'))
    fr = Frame(svg, x0, y0, x1, y1, 5e5, 5e7, ticks, fs)

    pts = [(5e5, .2), (token, .8), (5e7, .8)]
    d = fr.step_path(pts)
    fill = d + f' V{y1} H{fr.x(5e5):.1f} Z'
    svg.add(f'<path d="{fill}" fill="{ACCENT}" fill-opacity="0.10"/>')
    svg.add(f'<path d="{d}" class="step"/>')
    # Step position drop line + marker.
    sx = fr.x(token)
    svg.add(f'<line x1="{sx:.1f}" y1="{fr.y(.8):.1f}" x2="{sx:.1f}" y2="{y1}" stroke="{ACCENT}" stroke-opacity="0.45" stroke-width="1" stroke-dasharray="2 3"/>')
    svg.add(f'<circle cx="{sx:.1f}" cy="{fr.y(.8):.1f}" r="6.5" fill="{SURFACE}"/>')
    svg.add(f'<circle cx="{sx:.1f}" cy="{fr.y(.8):.1f}" r="4.5" fill="{ACCENT}"/>')
    # Direct labels on the two levels, and the shaded area's value.
    svg.text(fr.x(5e5) + 6, fr.y(.2) - 8, '0.20', 'm strong', 'start', fs)
    svg.text(x1 - 2, fr.y(.8) - 10, 'Final best 0.80', 't strong h', 'end', fs)
    svg.text((x0 + x1) / 2, fr.y(.1) + fs * .36, f'Area {area_value}', 't strong h', 'middle', fs)
    svg.write(name)


# --------------------------------------------------------------------------
# Figure 3: risk-free hints flow.
# --------------------------------------------------------------------------
def box(svg, x, y, w, h, title, lines, kind='main', fs=12, mono_title=False):
    if kind == 'main':
        style = f'fill="{ACCENT_SOFT}" stroke="{ACCENT}" stroke-width="1"'
    elif kind == 'side':
        style = f'fill="{SURFACE}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 3"'
    else:  # discard
        style = f'fill="{SURFACE2}" stroke="{LINE_STRONG}" stroke-width="1" stroke-dasharray="4 3"'
    svg.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" {style}/>')
    n = 1 + len(lines)
    lh = fs + 6
    top = y + h / 2 - (lh * n) / 2 + fs
    cls = 'm strong' if mono_title else 't strong'
    svg.text(x + w / 2, top, title, cls, 'middle', fs + (0 if mono_title else 1))
    for i, ln in enumerate(lines, 1):
        svg.text(x + w / 2, top + lh * i, ln, 't lab' if kind != 'main' else 't', 'middle', fs)


def arrow(svg, d, kind='main', marker='a'):
    if kind == 'main':
        svg.add(f'<path d="{d}" fill="none" stroke="{ACCENT}" stroke-width="1.75" stroke-linejoin="round" marker-end="url(#{marker})"/>')
    else:
        svg.add(f'<path d="{d}" fill="none" stroke="{FAINT}" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#{marker}d)"/>')


def defs(svg):
    svg.add(f'<defs>'
            f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M1,1 L9,5 L1,9 Z" fill="{ACCENT}"/></marker>'
            f'<marker id="ad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M1,1 L9,5 L1,9 Z" fill="{FAINT}"/></marker>'
            f'</defs>')


def ledger(svg, x, y, w, fs=12):
    """Worked example: a kept ×0.9 hint scales later submissions only."""
    h = fs * 2 + 66
    svg.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{SURFACE}" stroke="{LINE}"/>')
    svg.text(x + 14, y + 22, 'Example: keep a ×0.9 hint', 't strong', 'start', fs)
    rows = [('Earlier best', '0.35', '0.35', 'unchanged'),
            ('Later submission', '0.80', '0.72', '\u00d70.9')]
    for i, (lab, a, b, note) in enumerate(rows):
        yy = y + 22 + (fs + 12) * (i + 1)
        svg.text(x + 14, yy, lab, 't lab', 'start', fs)
        svg.text(x + w - 14, yy,
                 f'<tspan class="m">{a}</tspan><tspan class="t lab"> \u2192 </tspan>'
                 f'<tspan class="m strong">{b}</tspan><tspan class="t lab">  {note}</tspan>',
                 't', 'end', fs, ' xml:space="preserve"')
    return h


def hints(mobile=False):
    fs = 13 if mobile else 12
    if not mobile:
        W, H = 900, 440
        svg = SVG(W, H, 'Hint browsing branches into a side-conversation; only selected hints rejoin the main run')
        defs(svg)
        # Main lane.
        box(svg, 32, 48, 168, 56, 'Main run', ['model + context'], 'main', fs)
        arrow(svg, 'M200,76 H246')
        box(svg, 248, 48, 200, 56, 'browse_hints()', ['tool call'], 'main', fs, mono_title=True)
        arrow(svg, 'M448,76 H698')
        svg.text(573, 68, 'main context retained unchanged', 't lab h', 'middle', fs)
        box(svg, 700, 48, 168, 56, 'Main run resumes', ['selected hints applied'], 'main', fs)
        # Branch down to the side-conversation.
        arrow(svg, 'M348,104 V188')
        svg.text(360, 150, 'copy of model + context', 't lab', 'start', fs)
        box(svg, 248, 190, 312, 136, 'Side-conversation',
            ['reads the full text of every hint,', 'with its price, and decides',
             'which hints to keep'], 'side', fs)
        # Selected hints rejoin the main run.
        arrow(svg, 'M560,216 H784 V106')
        svg.text(672, 206, 'selected hints only', 't h', 'middle', fs)
        # Preview and rejected hints are discarded.
        arrow(svg, 'M404,326 V370', 'discard')
        box(svg, 274, 372, 260, 46, 'Discarded', ['preview and rejected hints'], 'discard', fs)
        ledger(svg, 600, 250, 268, fs)
        svg.write('hints')
        return

    W = 420
    svg = SVG(W, 700, 'Hint browsing branches into a side-conversation; only selected hints rejoin the main run')
    defs(svg)
    cx = W / 2
    box(svg, 60, 20, 300, 56, 'Main run', ['model + context'], 'main', fs)
    arrow(svg, f'M{cx},76 V104')
    box(svg, 60, 106, 300, 56, 'browse_hints()', ['tool call'], 'main', fs, mono_title=True)
    arrow(svg, f'M{cx},162 V214')
    svg.text(cx + 12, 193, 'copy of model + context', 't lab', 'start', fs)
    box(svg, 40, 216, 340, 132, 'Side-conversation',
        ['reads the full text of every hint,', 'with its price, and decides', 'which hints to keep'], 'side', fs)
    # Two outcomes: discard (left, dashed) and rejoin (right, accent).
    arrow(svg, 'M120,348 V420', 'discard')
    box(svg, 24, 422, 186, 72, 'Discarded', ['preview and', 'rejected hints'], 'discard', fs)
    arrow(svg, 'M300,348 V420')
    svg.text(312, 390, 'selected', 't h', 'start', fs)
    svg.text(312, 390 + fs + 4, 'hints only', 't h', 'start', fs)
    box(svg, 226, 422, 170, 72, 'Main run', ['resumes with', 'selected hints'], 'main', fs)
    h = ledger(svg, 24, 540, 372, fs)
    svg.parts[0] = svg.parts[0].replace('0 0 420 700', f'0 0 420 {540 + h + 20}')
    svg.parts[2] = f'<rect width="420" height="{540 + h + 20}" fill="{SURFACE}"/>'
    svg.write('hints-mobile')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    submissions(); submissions(mobile=True)
    area(5e6, 'area-early', '0.500'); area(2e7, 'area-late', '0.319')
    hints(); hints(mobile=True)
    print(f'Wrote 6 illustrative SVG figures to {OUT}')

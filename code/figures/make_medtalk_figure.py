#!/usr/bin/env python3
"""
make_medtalk_figure.py -- MEDTalk's intensity signal beside its brows. Post hoc, not pre-registered.

    python code/figures/make_medtalk_figure.py [--out DIR]   recompute the values and draw the figure into DIR
    python code/figures/make_medtalk_figure.py --check       recompute the values and compare them with the
                                                             released CSV; writes nothing

Reads data/derived/audit_results.csv. Outputs, into --out only (default figures_out/; never into data/ or
media/, which the script refuses):
    figure_medtalk_values.csv        every plotted value
    medtalk_intensity_vs_brow.png    the figure: 300 dpi, white background, 180 mm wide
The released copies, data/derived/figure_medtalk_values.csv and media/medtalk_intensity_vs_brow.png, were
made this way and copied into place by hand. A plain run also compares its values with the released CSV and
exits non-zero if they disagree. The CSV comes out byte for byte the same as the released one; the PNG's
bytes can differ under another matplotlib or FreeType version, which is why it is written to --out and
never over the released copy.

WHAT IS MEASURED
MEDTalk predicts a scalar intensity signal, s_t, for every frame: emotion2vec features of the audio attend
to the Whisper transcript of the same audio, and the transcript reaches the animation decoder only through
s_t. The audit stored the mean of s_t inside each item's speech span (audit_results.csv: source
"medtalk_st", channel "intensity", column m.mean_in_span). The brow readout is MEDTalk's inner and outer
brow-raise amplitude (source "medtalk", channels brow_inner and brow_outer, arm "clean", column m.amp:
p95 - p5 inside the speech span), combined as their geometric mean. Every change is the ln ratio of an
item to the c100 item of its own cell.

  (a) For 11 voice steps, the mean over the 12 cells of the brow ln ratio and of the s_t ln ratio, each
      with a 95 % percentile interval from a bootstrap over cells (B = 2000, numpy default_rng seed
      20260914).
  (b) The 192 items (16 steps x 12 cells): s_t ln ratio against brow ln ratio, one symbol per item. The
      dashed line is the within-cell slope of ln brow amplitude on ln s_t, fitted on cell-centred values
      over all 17 items of each cell (c100 included, so 204 items), with a 95 % percentile interval from
      a cluster bootstrap over cells (B = 2000, seed 0). It is drawn through the origin, the c100 item of
      every cell; the items themselves are not cell-centred, so a plain fit through the plotted points
      would give a different slope.

Both bootstraps are the audit analysis code's own, written out here so that this script needs numpy,
pandas and matplotlib only. Their draws depend on the order of the cells, which is the order in which the
cells first appear in audit_results.csv (the ladder manifest's order); do not sort them.

HOW TO READ IT
Post hoc; not part of the pre-registration. The intervals are over 12 cells from three voices, so every
value here is conditional on these three voices. The slope in (b) is descriptive: brow amplitude and s_t
both respond to the register of the voice, so their association across steps should not be read as the
gain of a path from s_t to the brows. Which internal path carries the fall in brow amplitude was not
tested.

VALUES CSV
One row per plotted element; columns that do not apply to a row are empty.
    panel a, row "step mean"          rung; brow, brow_ci_lo, brow_ci_hi; s_t, s_t_ci_lo, s_t_ci_hi;
                                      n_cells = n_items = 12
    panel b, row "item"               rung, cell, voice (zh, en, ja); brow and s_t are that item's ln ratios
    panel b, row "within-cell slope"  slope, slope_ci_lo, slope_ci_hi; n_cells = 12, n_items = 204

--check recomputes every value and exits non-zero if a row is missing, extra or repeated, if a value is
empty, unreadable or not finite, or if any value differs from the released CSV by more than 1e-6.

LAYOUT CHECK
Before the PNG is saved, every label is tested against every other label, against every drawn line (the
zero lines, the whiskers, the dashed slope line, the axis lines) and against every symbol, with 1 pt of
clearance; the script stops if anything touches or leaves the figure.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.markers import MarkerStyle  # noqa: E402
from matplotlib.transforms import Affine2D, blended_transform_factory, offset_copy  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "derived" / "audit_results.csv"
VALUES = ROOT / "data" / "derived" / "figure_medtalk_values.csv"      # the released copy, read by --check
PNG_NAME = "medtalk_intensity_vs_brow.png"
OUT_DIR = ROOT / "figures_out"
PROTECTED = (ROOT / "data", ROOT / "media")                           # hold the released copies

N_BOOT = 2000
SEED_STEP = 20260914        # bootstrap over cells for the step means (panel a)
SEED_SLOPE = 0              # cluster bootstrap over cells for the within-cell slope (panel b)
TOL = 1e-6

# panel (a): the steps drawn, top to bottom, grouped by what they change in the voice
STEP_GROUPS = [
    ("null", ["tgt100", "src"]),
    ("span", ["k050", "k167"]),
    ("loudness range", ["g060", "g150"]),
    ("rate", ["rate090"]),
    ("register", ["r+2", "r+4"]),
    ("effort", ["effort0", "effort"]),
]
ANCHOR = "c100"
VOICES = ["zh", "en", "ja"]
VOICE_OF_LANG = {"cn": "zh", "en": "en", "jp": "ja"}
VOICE_NAME = {"zh": "zh  ESD 0002", "en": "en  RAVDESS 02", "ja": "ja  JVNV F2"}

COLUMNS = ["panel", "row", "rung", "cell", "voice",
           "brow", "brow_ci_lo", "brow_ci_hi", "s_t", "s_t_ci_lo", "s_t_ci_hi",
           "slope", "slope_ci_lo", "slope_ci_hi", "n_cells", "n_items"]
NUMERIC = COLUMNS[5:14]
KEY = ("panel", "row", "rung", "cell")


def rel(p):
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------- values ----
def load_measures(path):
    """Cell x rung tables of the brow amplitude (geometric mean of inner and outer) and of mean s_t."""
    R = pd.read_csv(path, low_memory=False)
    st_rows = R[(R.source == "medtalk_st") & (R.channel == "intensity") & (R.arm == "clean")]
    cells = list(dict.fromkeys(st_rows.cell))
    rungs = list(dict.fromkeys(st_rows.rung))

    def table(source, channel, metric):
        s = R[(R.source == source) & (R.channel == channel) & (R.arm == "clean")]
        return s.pivot(index="cell", columns="rung", values=metric).reindex(index=cells, columns=rungs)

    bi = table("medtalk", "brow_inner", "m.amp")
    bo = table("medtalk", "brow_outer", "m.amp")
    st = table("medtalk_st", "intensity", "m.mean_in_span")
    problems = []
    if len(cells) != 12 or len(rungs) != 17 or ANCHOR not in rungs or len(st_rows) != 204:
        problems.append("expected 12 cells x 17 rungs = 204 medtalk_st rows, found %d cells, %d rungs, %d rows"
                        % (len(cells), len(rungs), len(st_rows)))
    for name, t in (("brow_inner m.amp", bi), ("brow_outer m.amp", bo), ("s_t m.mean_in_span", st)):
        v = t.to_numpy(float)
        if not (np.all(np.isfinite(v)) and np.all(v > 0)):
            problems.append("%s has missing or non-positive values" % name)
    if problems:
        sys.exit("STOP: " + "; ".join(problems))
    brow = np.exp(0.5 * (np.log(bi) + np.log(bo)))
    lang = st_rows.drop_duplicates("cell").set_index("cell")["lang"]
    voice = {c: VOICE_OF_LANG[lang[c]] for c in cells}
    return cells, rungs, brow, st, voice


def ln_ratio(t, cell, rung):
    return math.log(t.loc[cell, rung] / t.loc[cell, ANCHOR])


def boot_mean(vals, n_boot=N_BOOT, seed=SEED_STEP):
    """Mean over cells with a percentile bootstrap over cells (the audit analysis's boot_mean)."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), len(v)


def within_cell_slope(x, y, groups, n_boot=N_BOOT, seed=SEED_SLOPE, ci=0.95):
    """Slope of ln y on x within groups (sum xc*yc / sum xc^2 on group-centred values), with a cluster
    bootstrap over groups: the audit analysis's dose_response with group= and log_y=True."""
    x = np.asarray(x, float)
    yy = np.log(np.asarray(y, float))
    g = np.asarray(groups)
    ug = list(dict.fromkeys(g.tolist()))
    idx = {k: np.flatnonzero(g == k) for k in ug}
    cen = {k: (x[idx[k]] - x[idx[k]].mean(), yy[idx[k]] - yy[idx[k]].mean()) for k in ug}

    def slope_of(keys):
        xc = np.concatenate([cen[k][0] for k in keys])
        yc = np.concatenate([cen[k][1] for k in keys])
        d = float(np.sum(xc * xc))
        return float(np.sum(xc * yc) / d) if d > 1e-12 else np.nan

    est = slope_of(ug)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        s = slope_of([ug[i] for i in rng.integers(0, len(ug), len(ug))])
        if np.isfinite(s):
            bs.append(s)
    a = (1.0 - ci) / 2.0
    return est, float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a)), len(ug), len(x)


def compute(path=RESULTS):
    cells, rungs, brow, st, voice = load_measures(path)
    rows = []
    for _family, steps in STEP_GROUPS:
        for rung in steps:
            b = boot_mean([ln_ratio(brow, c, rung) for c in cells])
            s = boot_mean([ln_ratio(st, c, rung) for c in cells])
            rows.append(dict(panel="a", row="step mean", rung=rung, cell="", voice="",
                             brow=b[0], brow_ci_lo=b[1], brow_ci_hi=b[2],
                             s_t=s[0], s_t_ci_lo=s[1], s_t_ci_hi=s[2], n_cells=b[3], n_items=b[3]))
    for c in cells:
        for rung in rungs:
            if rung != ANCHOR:
                rows.append(dict(panel="b", row="item", rung=rung, cell=c, voice=voice[c],
                                 brow=ln_ratio(brow, c, rung), s_t=ln_ratio(st, c, rung)))
    xs, ys, gs = [], [], []
    for c in cells:
        for rung in rungs:
            xs.append(float(np.log(st.loc[c, rung])))
            ys.append(float(brow.loc[c, rung]))
            gs.append(c)
    est, lo, hi, n_groups, n_items = within_cell_slope(xs, ys, gs)
    rows.append(dict(panel="b", row="within-cell slope", rung="all 17 steps", cell="", voice="",
                     slope=est, slope_ci_lo=lo, slope_ci_hi=hi, n_cells=n_groups, n_items=n_items))
    return rows


def as_text(row):
    out = {}
    for col in COLUMNS:
        v = row.get(col, "")
        if col in NUMERIC:
            if v == "" or v is None:
                out[col] = ""
            else:
                s = "%.8f" % v
                out[col] = "0.00000000" if s == "-0.00000000" else s
        else:
            out[col] = str(v)
    return out


def write_values(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(as_text(r))


def read_values(path):
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != COLUMNS:
            sys.exit("STOP: %s does not have the expected columns" % rel(path))
        return list(rd)


def check(rows, path):
    """Compare recomputed rows with a values CSV; return (problems, largest difference)."""
    if not Path(path).is_file():
        return ["%s is missing" % rel(path)], float("nan")
    problems = []
    have_rows = read_values(path)
    want_rows = [as_text(r) for r in rows]
    have, want = {}, {}
    for i, r in enumerate(have_rows, start=2):
        if None in r or any(v is None for v in r.values()):
            problems.append("line %d of the CSV does not have exactly %d fields" % (i, len(COLUMNS)))
            continue
        k = tuple(r[c] for c in KEY)
        if k in have:
            problems.append("row in the CSV more than once: %s" % (k,))
        have[k] = r
    for r, t in zip(rows, want_rows):
        k = tuple(t[c] for c in KEY)
        if k in want:
            sys.exit("STOP: the recomputed rows repeat the key %s" % (k,))
        want[k] = (r, t)
    if len(have_rows) != len(want_rows):
        problems.append("the CSV has %d rows, %d were recomputed" % (len(have_rows), len(want_rows)))
    problems += ["row in the CSV but not recomputed: %s" % (k,) for k in have if k not in want]
    problems += ["row recomputed but not in the CSV: %s" % (k,) for k in want if k not in have]
    worst = 0.0
    for k, (r, t) in want.items():
        if k not in have:
            continue
        h = have[k]
        for col in COLUMNS:
            if col not in NUMERIC:
                if h[col] != t[col]:
                    problems.append("%s %s: CSV %r, recomputed %r" % (k, col, h[col], t[col]))
                continue
            if (h[col].strip() == "") != (t[col] == ""):
                problems.append("%s %s: CSV %r, recomputed %r" % (k, col, h[col], t[col]))
                continue
            if t[col] == "":
                continue
            try:
                hv = float(h[col])
            except ValueError:
                problems.append("%s %s: CSV %r is not a number" % (k, col, h[col]))
                continue
            d = abs(hv - float(r[col]))
            if not math.isfinite(d) or d > TOL:
                problems.append("%s %s: CSV %s, recomputed %.8f (|diff| %.2e)" % (k, col, h[col], r[col], d))
            elif d > worst:
                worst = d
    return problems, worst


# --------------------------------------------------------------------------- figure ----
FIG_W_MM, FIG_H_MM = 180.0, 88.0
FS = 10.0                   # every label, at the printed size of 180 mm
FS_SMALL = 9.0              # step-group names and the slope note
FS_SUB = 9.0                # the subscript t of s_t: drawn as its own 9 pt text, not a shrunk math subscript
SUB_DROP = 2.6              # pt the subscript sits below the baseline
CLEAR_PT = 1.0              # the least room the layout check allows between a label and anything else

INK = "#0b0b0b"
SECOND = "#52514e"
BASELINE = "#c3c2b7"
BROW_C = "#3b3a37"          # brow: filled dark circle
ST_C = "#4a3aa7"            # s_t: open violet circle
VOICE_C = {"zh": "#2a78d6", "en": "#eb6834", "ja": "#1baf7a"}
VOICE_SHAPE = {"zh": "^", "en": "s", "ja": "D"}          # the release's per-voice symbols
SHAPE_SCALE = {"^": 1.12, "s": 0.88, "D": 0.80}          # equalise the visual size of the shapes
SLOPE_LS = (0, (4, 2.5))
MINUS = "\u2212"

# axes rectangles in mm from the bottom-left corner: left, bottom, width, height
RECT_A = (37.0, 24.0, 51.0, 56.0)
RECT_B = (111.0, 24.0, 65.0, 56.0)
Y_XLABEL, Y_KEY, Y_TITLE = 13.0, 3.5, 83.5      # baselines, mm from the bottom
# the slope note sits in the lower left of panel (b), where no item falls and no line runs
NOTE_X, NOTE_Y, NOTE_STEP = RECT_B[0] + 1.8, RECT_B[1] + 2.2, 4.1     # mm: left edge, lowest baseline, pitch
NOTE_DASH_PT, NOTE_GAP_PT = 10.5, 3.0

plt.rcParams.update({
    "font.family": ["STIXGeneral", "DejaVu Serif", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "xtick.major.size": 3.0,
    "ytick.major.width": 0.8, "ytick.major.size": 3.0,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.5,
    "axes.edgecolor": BASELINE, "xtick.color": BASELINE, "ytick.color": BASELINE,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.unicode_minus": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def num(x, fmt="%.2f", sign=False):
    s = (("%+" + fmt[1:]) if sign else fmt) % x
    return s.replace("-", MINUS)


class Layout:
    """Places runs of text pieces, markers and line samples on one baseline, and remembers what it placed
    so the layout check can tell pieces of one label (allowed to touch) from separate labels (not
    allowed)."""

    def __init__(self, fig):
        self.fig = fig
        self.mm = Affine2D().scale(1.0 / 25.4) + fig.dpi_scale_trans
        self.renderer = fig.canvas.get_renderer()
        self.groups = []            # lists of Text artists that form one label

    def width_pt(self, artist):
        return artist.get_window_extent(self.renderer).width * 72.0 / self.fig.dpi

    def run(self, x_mm, y_mm, pieces, ha="left", color=INK, mid_pt=0.34 * FS):
        """pieces: ("text", s, size, style) | ("sub", s) | ("gap", pt) | ("marker", dict of Line2D kwargs)
        | ("line", pt, dict of Line2D kwargs). The run sits on the baseline y_mm; markers and line samples
        are centred mid_pt above it, on the lower-case letters."""
        made = []
        for p in pieces:
            if p[0] == "text":
                t = self.fig.text(0, 0, p[1], transform=self.mm, fontsize=p[2], fontstyle=p[3], color=color,
                                  ha="left", va="baseline")
                made.append(("text", t, self.width_pt(t), 0.0))
            elif p[0] == "sub":
                t = self.fig.text(0, 0, p[1], transform=self.mm, fontsize=FS_SUB, fontstyle="italic",
                                  color=color, ha="left", va="baseline")
                made.append(("text", t, self.width_pt(t), -SUB_DROP))
            elif p[0] == "gap":
                made.append(("gap", None, p[1], 0.0))
            elif p[0] == "line":
                made.append(("line", Line2D([0, 0], [0, 0], **p[2]), p[1], 0.0))
            else:
                kw = dict(p[1])
                made.append(("marker", Line2D([0], [0], ls="none", **kw), kw.get("ms", 5.0), 0.0))
        total = sum(m[2] for m in made)
        x = {"left": 0.0, "center": -0.5 * total, "right": -total}[ha]
        texts = []
        for kind, art, w, dy in made:
            if kind == "text":
                art.set_position((x_mm, y_mm))
                art.set_transform(offset_copy(self.mm, fig=self.fig, x=x, y=dy, units="points"))
                texts.append(art)
            elif kind == "marker":
                art.set_data([x_mm], [y_mm])
                art.set_transform(offset_copy(self.mm, fig=self.fig, x=x + 0.5 * w, y=mid_pt, units="points"))
                self.fig.add_artist(art)
            elif kind == "line":
                art.set_data([x_mm, x_mm + w * 25.4 / 72.0], [y_mm, y_mm])
                art.set_transform(offset_copy(self.mm, fig=self.fig, x=x, y=mid_pt, units="points"))
                self.fig.add_artist(art)
            x += w
        self.groups.append(texts)
        return texts


def draw(rows):
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=300)
    L = Layout(fig)

    def axes(rect):
        return fig.add_axes([rect[0] / FIG_W_MM, rect[1] / FIG_H_MM, rect[2] / FIG_W_MM, rect[3] / FIG_H_MM])

    steps = {r["rung"]: r for r in rows if r["panel"] == "a"}
    items = [r for r in rows if r["panel"] == "b" and r["row"] == "item"]
    slope = [r for r in rows if r["panel"] == "b" and r["row"] == "within-cell slope"][0]
    f = lambda r, k: float(r[k])   # noqa: E731

    # ---------------------------------------------------------------- (a) step means
    ax = axes(RECT_A)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ypos, centre, y = {}, {}, 0.0
    for gi, (family, names) in enumerate(STEP_GROUPS):
        if gi:
            y -= 0.5
        ys = []
        for rung in names:
            ypos[rung] = y
            ys.append(y)
            y -= 1.0
        centre[family] = float(np.mean(ys))
    ax.set_ylim(y + 1.0 - 0.6, 0.6)
    XA = (-0.165, 0.07)
    ax.set_xlim(*XA)
    ax.axvline(0, color=BASELINE, lw=0.8, zorder=1)
    for rung, y in ypos.items():
        r = steps[rung]
        for key, dy, style in (("brow", 0.19, "brow"), ("s_t", -0.19, "s_t")):
            lo, hi, est = f(r, key + "_ci_lo"), f(r, key + "_ci_hi"), f(r, key)
            if not (XA[0] < lo and hi < XA[1]):
                sys.exit("STOP: panel (a) %s %s interval leaves the axis" % (rung, key))
            c = BROW_C if style == "brow" else ST_C
            ax.plot([lo, hi], [y + dy, y + dy], color=c, lw=1.3, solid_capstyle="butt", zorder=3)
            if style == "brow":
                ax.plot([est], [y + dy], ls="none", marker="o", ms=5.2, mfc=BROW_C, mec="white", mew=0.6, zorder=4)
            else:
                ax.plot([est], [y + dy], ls="none", marker="o", ms=4.8, mfc="white", mec=ST_C, mew=1.3, zorder=4)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(list(ypos.keys()))
    ax.tick_params(axis="y", length=0)
    xt = [-0.15, -0.10, -0.05, 0.0, 0.05]
    ax.set_xticks(xt)
    ax.set_xticklabels(["0" if v == 0 else num(v) for v in xt])
    fam_tr = blended_transform_factory(L.mm, ax.transData)
    for family, yc in centre.items():
        t = fig.text(3.0, yc, family, transform=fam_tr, ha="left", va="center", fontsize=FS_SMALL, color=SECOND)
        L.groups.append([t])
    xa_mid = RECT_A[0] + 0.5 * RECT_A[2]
    L.run(xa_mid, Y_XLABEL, [("text", "ln ratio vs c100", FS, "normal")], ha="center")
    L.run(3.0, Y_TITLE, [("text", "(a) each step: mean of 12 cells, 95 % CI", FS, "normal")])
    L.run(xa_mid, Y_KEY, [
        ("marker", dict(marker="o", ms=5.2, mfc=BROW_C, mec="white", mew=0.6)), ("gap", 3.0),
        ("text", "brow", FS, "normal"), ("gap", 12.0),
        ("marker", dict(marker="o", ms=4.8, mfc="white", mec=ST_C, mew=1.3)), ("gap", 3.0),
        ("text", "s", FS, "italic"), ("sub", "t")], ha="center")

    # ---------------------------------------------------------------- (b) items
    bx = axes(RECT_B)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    XB, YB = (-0.092, 0.052), (-0.33, 0.21)
    bx.set_xlim(*XB)
    bx.set_ylim(*YB)
    bx.axhline(0, color=BASELINE, lw=0.8, zorder=1)
    bx.axvline(0, color=BASELINE, lw=0.8, zorder=1)
    for v in VOICES:
        pts = [(f(r, "s_t"), f(r, "brow")) for r in items if r["voice"] == v]
        for px, py in pts:
            if not (XB[0] < px < XB[1] and YB[0] < py < YB[1]):
                sys.exit("STOP: panel (b) item (%.4f, %.4f) leaves the axis" % (px, py))
        mk = VOICE_SHAPE[v]
        bx.plot([p[0] for p in pts], [p[1] for p in pts], ls="none", marker=mk, ms=4.6 * SHAPE_SCALE[mk],
                mfc=VOICE_C[v], mec="white", mew=0.45, zorder=3)
    b = f(slope, "slope")
    xx = np.array(XB)
    bx.plot(xx, b * xx, color=SECOND, lw=1.0, ls=SLOPE_LS, zorder=2)
    xt = [-0.08, -0.04, 0.0, 0.04]
    bx.set_xticks(xt)
    bx.set_xticklabels(["0" if v == 0 else num(v) for v in xt])
    bx.set_yticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2])
    bx.set_yticklabels(["0" if v == 0 else num(v, "%.1f") for v in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2]])
    bx.set_ylabel("brow ln ratio vs c100", color=INK, labelpad=3)
    # the slope note, bottom line first; lines 2 and 3 start where the text of line 1 starts
    indent = NOTE_DASH_PT + NOTE_GAP_PT
    mid = 0.34 * FS_SMALL
    L.run(NOTE_X, NOTE_Y + 2 * NOTE_STEP,
          [("line", NOTE_DASH_PT, dict(color=SECOND, lw=1.0, ls=SLOPE_LS)), ("gap", NOTE_GAP_PT),
           ("text", "within-cell slope %s" % num(b), FS_SMALL, "normal")], color=SECOND, mid_pt=mid)
    L.run(NOTE_X, NOTE_Y + NOTE_STEP,
          [("gap", indent), ("text", "95 %% CI [%s, %s]" % (num(f(slope, "slope_ci_lo")),
                                                            num(f(slope, "slope_ci_hi"))), FS_SMALL, "normal")],
          color=SECOND, mid_pt=mid)
    L.run(NOTE_X, NOTE_Y, [("gap", indent), ("text", "descriptive", FS_SMALL, "normal")], color=SECOND, mid_pt=mid)
    xb_mid = RECT_B[0] + 0.5 * RECT_B[2]
    L.run(xb_mid, Y_XLABEL, [("text", "s", FS, "italic"), ("sub", "t"), ("gap", 3.0),
                         ("text", "ln ratio vs c100", FS, "normal")], ha="center")
    L.run(RECT_B[0] - 16.0, Y_TITLE, [("text", "(b) 192 items: 16 steps \u00d7 12 cells", FS, "normal")])
    key = []
    for i, v in enumerate(VOICES):
        mk = VOICE_SHAPE[v]
        if i:
            key.append(("gap", 10.0))
        key += [("marker", dict(marker=mk, ms=4.6 * SHAPE_SCALE[mk], mfc=VOICE_C[v], mec="white", mew=0.45)),
                ("gap", 2.5), ("text", VOICE_NAME[v], FS, "normal")]
    L.run(xb_mid - 6.0, Y_KEY, key, ha="center")
    return fig, L, (ax, bx)


def _seg_hits_box(p, q, box):
    """Liang-Barsky: does the segment p-q meet the box (x0, y0, x1, y1)?"""
    x0, y0 = p
    dx, dy = q[0] - x0, q[1] - y0
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, x0 - box[0]), (dx, box[2] - x0), (-dy, y0 - box[1]), (dy, box[3] - y0)):
        if pp == 0.0:
            if qq < 0.0:
                return False
        else:
            t = qq / pp
            if pp < 0.0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def _marks(fig, axs):
    """Every drawn line and symbol, in display pixels: ("seg", p, q, half width) and
    ("sym", centre, half width, half height). Symbol extents come from the marker path and its stroke,
    because matplotlib's window extent of a marker ignores both."""
    px = fig.dpi / 72.0
    out = []
    lines = [ln for a in axs for ln in a.get_lines()] + [a for a in fig.artists if isinstance(a, Line2D)]
    for ln in lines:
        if not ln.get_visible():
            continue
        xy = np.column_stack([np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)])
        P = ln.get_transform().transform(xy)
        if ln.get_linestyle() not in ("None", "none", "", " "):
            hw = 0.5 * ln.get_linewidth() * px
            for i in range(len(P) - 1):
                out.append(("seg", P[i], P[i + 1], hw, ln))
        if ln.get_marker() not in (None, "None", "none", "", " "):
            m = MarkerStyle(ln.get_marker())
            v = m.get_path().transformed(m.get_transform()).vertices
            ms, mew = ln.get_markersize(), ln.get_markeredgewidth()
            hx = (np.abs(v[:, 0]).max() * ms + 0.5 * mew) * px
            hy = (np.abs(v[:, 1]).max() * ms + 0.5 * mew) * px
            for c in P:
                out.append(("sym", c, hx, hy, ln))
    for a in axs:
        for sp in a.spines.values():
            if sp.get_visible():
                V = sp.get_path().transformed(sp.get_transform()).vertices
                hw = 0.5 * sp.get_linewidth() * px
                for i in range(len(V) - 1):
                    out.append(("seg", V[i], V[i + 1], hw, sp))
    return out


def layout_check(fig, L, axs):
    """No label may leave the figure, touch another label, or come within CLEAR_PT of a line or a symbol."""
    fig.canvas.draw()
    r = L.renderer
    fb = fig.bbox
    clear = CLEAR_PT * fig.dpi / 72.0
    labels = [grp for grp in L.groups if grp]
    for a in axs:
        for t in a.get_xticklabels() + a.get_yticklabels():
            if t.get_visible() and t.get_text():
                labels.append([t])
        if a.yaxis.label.get_text():
            labels.append([a.yaxis.label])
    boxes = [[t.get_window_extent(r) for t in grp] for grp in labels]
    problems = []
    for grp, bb in zip(labels, boxes):
        for t, b in zip(grp, bb):
            if b.x0 < fb.x0 or b.y0 < fb.y0 or b.x1 > fb.x1 or b.y1 > fb.y1:
                problems.append("leaves the figure: %r" % t.get_text())
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            for bi in boxes[i]:
                for bj in boxes[j]:
                    if bi.overlaps(bj):
                        problems.append("touch: %r and %r" % (labels[i][0].get_text(), labels[j][0].get_text()))
    marks = _marks(fig, axs)
    for grp, bb in zip(labels, boxes):
        for t, b in zip(grp, bb):
            for mk in marks:
                if mk[0] == "seg":
                    e = clear + mk[3]
                    hit = _seg_hits_box(mk[1], mk[2], (b.x0 - e, b.y0 - e, b.x1 + e, b.y1 + e))
                    what = "a line"
                else:
                    (cx, cy), hx, hy = mk[1], mk[2], mk[3]
                    hit = (b.x0 - clear < cx + hx and cx - hx < b.x1 + clear and
                           b.y0 - clear < cy + hy and cy - hy < b.y1 + clear)
                    what = "a symbol"
                if hit:
                    problems.append("%r is within %.1f pt of %s (%s)" % (t.get_text(), CLEAR_PT, what,
                                                                        mk[4].__class__.__name__))
    return sorted(set(problems))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="recompute the values and compare them with the released CSV; write nothing")
    ap.add_argument("--out", default=str(OUT_DIR), help="output folder (default: %(default)s)")
    a = ap.parse_args()

    rows = compute(RESULTS)
    if a.check:
        problems, worst = check(rows, VALUES)
        if problems:
            print("STOP: %d disagreement(s) between %s and the values recomputed from %s"
                  % (len(problems), rel(VALUES), rel(RESULTS)))
            for p in problems[:20]:
                print("  " + p)
            if len(problems) > 20:
                print("  ... and %d more" % (len(problems) - 20))
            sys.exit(1)
        print("OK: %d rows of %s agree with the values recomputed from %s (largest |diff| %.1e, tolerance %.0e)"
              % (len(rows), rel(VALUES), rel(RESULTS), worst, TOL))
        return

    out = Path(a.out).resolve()
    for p in PROTECTED:
        if out == p or p in out.parents:
            sys.exit("STOP: --out %s is inside %s/, which holds the released copies; write somewhere else "
                     "(default figures_out/)" % (a.out, rel(p)))
    out.mkdir(parents=True, exist_ok=True)
    values_out, png_out = out / VALUES.name, out / PNG_NAME

    write_values(rows, values_out)
    back = read_values(values_out)          # the figure is drawn from the file, so the two cannot drift apart
    problems, _ = check(rows, values_out)
    if problems:
        sys.exit("STOP: %s did not read back as written" % rel(values_out))
    fig, L, axs = draw(back)
    problems = layout_check(fig, L, axs)
    if problems:
        plt.close(fig)
        sys.exit("STOP: layout check failed:\n  " + "\n  ".join(problems))
    fig.savefig(png_out, dpi=300, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    slope = [r for r in back if r["row"] == "within-cell slope"][0]
    print("wrote %s (%d rows) and %s" % (rel(values_out), len(back), rel(png_out)))
    print("  within-cell slope %s [%s, %s], %s cells, %s items"
          % (slope["slope"], slope["slope_ci_lo"], slope["slope_ci_hi"], slope["n_cells"], slope["n_items"]))
    for r in back:
        if r["panel"] == "a":
            print("  %-8s brow %s [%s, %s]   s_t %s [%s, %s]" % (r["rung"], r["brow"], r["brow_ci_lo"], r["brow_ci_hi"],
                                                                r["s_t"], r["s_t_ci_lo"], r["s_t_ci_hi"]))
    problems, worst = check(rows, VALUES)
    if problems:
        print("DISAGREES with the released %s (%d problem(s)); run --check for the list" % (rel(VALUES), len(problems)))
        sys.exit(1)
    print("agrees with the released %s (largest |diff| %.1e)" % (rel(VALUES), worst))


if __name__ == "__main__":
    main()

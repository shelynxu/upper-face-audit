#!/usr/bin/env python3
"""
make_figures.py -- the two audit figures: the protocol diagram and the dose response.

Run (needs only matplotlib):
    python code/figures/make_figures.py [output directory]   # default: the current directory

Writes:
    fig1_protocol.pdf   the protocol diagram (Fig. 1)
    fig2_dose.pdf       the dose response (Fig. 2)

Both are vector PDFs, one column wide (86 mm), with every label at 9 pt at print size and
the fonts embedded as TrueType (no Type 3), as the paper's style requires.

Fig. 1 is a diagram and reads nothing. Fig. 2 is drawn from
data/derived/figure_dose_values.csv, which holds every plotted number together with the
table it came from: the twelve-cell estimate and interval per model and follower, the null
floors, the human prediction and its x[0.5, 2] band, the observed human ratio, and the
per-voice values that are drawn as small symbols under each row. That file is the figure's
single source of numbers, so the figure and the released values cannot drift apart.
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.markers import MarkerStyle  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VALUES = os.path.join(HERE, os.pardir, os.pardir, "data", "derived", "figure_dose_values.csv")

# ---------------------------------------------------------------- style ----
FS = 9                      # every label is 9 pt at print size
LS = 1.12                   # line spacing inside the diagram boxes
COLW = 86.0 / 25.4          # one column = 86 mm

INK = "#0b0b0b"             # primary ink
SECOND = "#52514e"          # secondary ink
MUTED = "#898781"           # muted ink (null-floor ticks)
BASELINE = "#c3c2b7"        # zero line / axis
BAND = "#e4e3dc"            # the human prediction's interval (grey band)
MODEL_C = "#3b3a37"         # audited models: dark ink
FOLLOW_C = "#2a78d6"        # constructed followers: blue
# The blue and the dark ink separate by dE 27 for normal vision and under simulated colour
# vision deficiency, and both clear 3:1 contrast on white. The dark ink is a de-emphasis
# tone rather than a second hue, and marker shape repeats every distinction colour makes.

plt.rcParams.update({
    "font.family": "STIXGeneral",     # Times-compatible, ships with matplotlib
    "mathtext.fontset": "stix",
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # TrueType, embedded; no Type 3
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 0, "xtick.major.pad": 1.5, "ytick.major.pad": 3,
    "axes.labelpad": 2, "axes.titlepad": 3,
    "axes.edgecolor": SECOND, "xtick.color": SECOND,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "savefig.dpi": 300,
})

# ----------------------------------------------------------- Fig. 2 data ---
SOURCES = [  # key, row label, group, marker
    ("emoface", "EmoFace", "model", "o"),
    ("emotalk", "EmoTalk", "model", "s"),
    ("emote", "EMOTE", "model", "^"),
    ("medtalk", "MEDTalk", "model", "D"),
    ("f0_follower", "F0 follower", "follower", "P"),
    ("energy_follower", "Energy follower", "follower", "X"),
]
VOICES = ["zh", "en", "ja"]        # the three speakers, one per language


def load_values(path=VALUES):
    """The twelve-cell values (d) and the per-voice values (pv) from the released CSV."""
    d = {"effort": {}, "floor": {}, "span": {}, "dyn": {}}
    pv = {"effort": {}, "span": {}, "dyn": {}}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m, src, voice = r["measure"], r["source"], r["voice"]
            est = float(r["estimate"]) if r["estimate"] else None
            ci = (float(r["ci_lo"]), float(r["ci_hi"])) if r["ci_lo"] else (None, None)
            if m == "null_floor":
                d["floor"][src] = est
            elif m in ("effort_ln_ratio", "span_slope", "dyn_slope"):
                fam = m.split("_")[0]
                if voice == "all":
                    d[fam][src] = (est,) + ci
                else:
                    pv[fam].setdefault(src, {})[voice] = est
            elif m == "human_prediction":
                d["pred"] = (est,) + ci
            elif m == "decision_band":
                d["band"] = ci
            elif m == "human_observed":
                d["human"] = (est,) + ci
    missing = [k for k, _, _, _ in SOURCES if k not in d["effort"]]
    if missing or "pred" not in d:
        sys.exit("%s is incomplete (missing %s)" % (path, missing or "the human prediction"))
    return d, pv


# -------------------------------------------------------------- Fig. 2 -----
# Row positions: the four models, then the two followers, then the human reference, each
# group separated by a wider gap than the rows inside it.
Y = {"emoface": 0.0, "emotalk": 1.0, "emote": 2.0, "medtalk": 3.0,
     "f0_follower": 4.35, "energy_follower": 5.35}
Y_HUMAN = 6.7

FIG_H = 3.1864                     # inches (80.93 mm)
YA = (-0.45, 7.2)                  # panel (a) y range, in row units
YB = (-0.5, 6.25)                  # panel (b) y range
BOT, GAP_AB, TOP = 0.19, 0.53, 0.20                    # inches
PITCH = (FIG_H - BOT - GAP_AB - TOP) / ((YA[1] - YA[0]) + (YB[1] - YB[0]))
L_COL, R_MARG, GAP_SD = 0.94, 0.07, 0.17               # label column, right margin, panel gap

MS_A, MS_B = 5.2, 4.4              # main markers (pt)
PV_MS_A, PV_MS_B = 3.1, 2.8        # per-voice markers (pt), before the per-shape factor
# Per-voice symbols sit on a strip just below their CI line (level 0). A symbol that would
# cover more than PV_OVERLAP of a symbol already on that level drops to the next level, so
# their vertical position carries no data.
PV_LEVELS_A = (3.8, 6.0, 8.2)      # pt below the CI line
PV_LEVELS_B = (3.4, 5.5, 7.5)
PV_OVERLAP = 0.15
PV_SHAPE = {"zh": "^", "en": "s", "ja": "D"}
PV_SCALE = {"^": 1.12, "s": 0.88, "D": 0.80}   # equalise the visual size of the shapes
PV_COLOR = "#77766f"
PV_MEW = 0.6

XLIM_A = (-0.2, 0.62)
XLIM_SPAN = (-0.16, 0.70)
XLIM_DYN = (-0.46, 0.84)


def style_axes(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def ci_row(ax, y, est, lo, hi, color, marker, hollow=False, ms=5.6):
    ax.plot([lo, hi], [y, y], color=color, lw=1.0, solid_capstyle="butt", zorder=3)
    if hollow:
        ax.plot([est], [y], ls="none", marker=marker, ms=ms - 0.4, mfc="white",
                mec=color, mew=1.0, zorder=4)
    else:
        ax.plot([est], [y], ls="none", marker=marker, ms=ms, mfc=color,
                mec="white", mew=0.7, zorder=4)


def color_of(group):
    return MODEL_C if group == "model" else FOLLOW_C


def pv_marker(ax, x, y, v, base_ms):
    # drawn above the main markers: a white-filled symbol may clip the lower edge of a main
    # marker it sits under, but no per-voice value is hidden and no CI line is touched
    mk = PV_SHAPE[v]
    ax.plot([x], [y], ls="none", marker=mk, ms=base_ms * PV_SCALE[mk], mfc="white",
            mec=PV_COLOR, mew=PV_MEW, zorder=4.5)


def marker_half_height(marker, ms):
    """Half of a marker's vertical extent in pt: 'D' and '^' are taller than 'o'."""
    st = MarkerStyle(marker)
    ext = st.get_path().transformed(st.get_transform()).get_extents()
    return 0.5 * (ext.y1 - ext.y0) * ms


def row_shift(main_marker, ms_main):
    """Extra drop (pt) of the per-voice strip under a main marker taller than a circle."""
    return max(0.0, marker_half_height(main_marker, ms_main) - 0.5 * ms_main)


def pv_levels(values, xlim, width_pt, base_ms):
    """Level index per voice for one row: the lowest level where the symbol does not cover
    more than PV_OVERLAP of a symbol already placed there."""
    placed, out = {}, {}
    scale = width_pt / (xlim[1] - xlim[0])
    for v in VOICES:
        x = (values[v] - xlim[0]) * scale
        wv = base_ms * PV_SCALE[PV_SHAPE[v]]
        for lvl in range(len(PV_LEVELS_A)):
            if all(abs(x - x2) >= (1 - PV_OVERLAP) * (wv + w2) / 2
                   for x2, w2 in placed.get(lvl, [])):
                break
        else:
            sys.exit("per-voice symbols need more than %d levels" % len(PV_LEVELS_A))
        placed.setdefault(lvl, []).append((x, wv))
        out[v] = lvl
    return out


def fig2(d, pv):
    """Brow ln ratio: (a) the effort step against the human prediction, (b) the span and
    dynamics slopes. Small open symbols under each row are that row's three speakers."""
    W = COLW
    ha, hb = (YA[1] - YA[0]) * PITCH, (YB[1] - YB[0]) * PITCH
    H = BOT + hb + GAP_AB + ha + TOP
    fig = plt.figure(figsize=(W, H))
    pt2u = 1.0 / (PITCH * 72.0)        # points -> row units (the same pitch in both panels)

    def add(x0, y0, w, h):
        return fig.add_axes([x0 / W, y0 / H, w / W, h / H])

    axA = add(L_COL, BOT + hb + GAP_AB, W - L_COL - R_MARG, ha)
    wb = (W - L_COL - R_MARG - GAP_SD) / 2
    axS = add(L_COL, BOT, wb, hb)
    axD = add(L_COL + wb + GAP_SD, BOT, wb, hb)

    # ---- (a) the effort step against the human prediction
    ax = axA
    style_axes(ax)
    p, plo, phi = d["pred"]
    b0, b1 = d["band"]
    ax.axvspan(plo, phi, color=BAND, lw=0, zorder=0)
    ax.axvline(0.0, color=BASELINE, lw=0.6, zorder=1)
    ax.axvline(p, color=SECOND, lw=0.8, zorder=1)
    for x in (b0, b1):
        ax.axvline(x, color=SECOND, lw=0.7, ls=(0, (3, 2)), zorder=1)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x, s in ((b0, "\u00d70.5"), (p, "\u00d71"), (b1, "\u00d72")):
        ax.annotate(s, xy=(x, 1.0), xycoords=trans, xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS, color=INK)
    for key, _, grp, mk in SOURCES:
        est, lo, hi = d["effort"][key]
        y = Y[key]
        ax.plot([d["floor"][key]], [y], ls="none", marker="|", ms=6.5, mew=0.9,
                color=MUTED, zorder=2)
        ci_row(ax, y, est, lo, hi, color_of(grp), mk, ms=MS_A)
        lv = pv_levels(pv["effort"][key], XLIM_A, (W - L_COL - R_MARG) * 72, PV_MS_A)
        for v in VOICES:
            lev = PV_LEVELS_A[lv[v]] + row_shift(mk, MS_A)
            pv_marker(ax, pv["effort"][key][v], y + lev * pt2u, v, PV_MS_A)
    est, lo, hi = d["human"]
    ci_row(ax, Y_HUMAN, est, lo, hi, INK, "o", hollow=True, ms=MS_A + 0.4)
    ax.annotate("one cell", xy=(d["effort"]["emotalk"][2], Y["emotalk"]), xytext=(4, 0),
                textcoords="offset points", ha="left", va="center", fontsize=FS,
                color=SECOND, bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"),
                zorder=5)
    ax.set_xlim(*XLIM_A)
    ax.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(["\u22120.2", "0", "0.2", "0.4", "0.6"])
    ax.set_ylim(YA[1], YA[0])
    ax.set_yticks([Y[k] for k, _, _, _ in SOURCES] + [Y_HUMAN])
    ax.set_yticklabels([lab for _, lab, _, _ in SOURCES] + ["Human"])
    ax.set_xlabel("Brow ln ratio, effort step vs c100")

    # ---- (b) the cue-specific slopes
    for ax, fam, title, xlim, xt in (
            (axS, "span", "Span, per ln factor", XLIM_SPAN, [0.0, 0.3, 0.6]),
            (axD, "dyn", "Dynamics, per ln factor", XLIM_DYN, [-0.4, 0.0, 0.4, 0.8])):
        style_axes(ax)
        ax.axvline(0.0, color=BASELINE, lw=0.6, zorder=1)
        for key, _, grp, mk in SOURCES:
            est, lo, hi = d[fam][key]
            y = Y[key]
            ci_row(ax, y, est, lo, hi, color_of(grp), mk, ms=MS_B)
            lv = pv_levels(pv[fam][key], xlim, wb * 72, PV_MS_B)
            for v in VOICES:
                lev = PV_LEVELS_B[lv[v]] + row_shift(mk, MS_B)
                pv_marker(ax, pv[fam][key][v], y + lev * pt2u, v, PV_MS_B)
        ax.set_xlim(*xlim)
        ax.set_xticks(xt)
        ax.set_xticklabels([("\u2212%g" % -v) if v < 0 else ("%g" % v) for v in xt])
        ax.set_ylim(YB[1], YB[0])
        ax.set_title(title, fontsize=FS)
    axS.set_yticks([Y[k] for k, _, _, _ in SOURCES])
    axS.set_yticklabels([lab for _, lab, _, _ in SOURCES])
    axD.set_yticks([])

    # ---- key for the per-voice symbols, in the empty top right of the dynamics panel
    handles = [Line2D([], [], ls="none", marker=PV_SHAPE[v], ms=PV_MS_A * PV_SCALE[PV_SHAPE[v]],
                      mfc="white", mec=PV_COLOR, mew=PV_MEW, label=v) for v in VOICES]
    leg = axD.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0),
                     frameon=True, fontsize=FS, handlelength=0.7, handletextpad=0.35,
                     labelspacing=0.12, borderpad=0.3, borderaxespad=0.15, fancybox=False)
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor(BASELINE)

    fig.text(0.03 / W, (H - 0.03) / H, "(a) Effort step", ha="left", va="top",
             fontsize=FS, weight="bold")
    fig.text(0.03 / W, (BOT + hb + 0.045) / H, "(b) Cue slopes", ha="left",
             va="bottom", fontsize=FS, weight="bold")
    return fig


# -------------------------------------------------------------- Fig. 1 -----
# The protocol diagram. Box shape and outline tell the kinds apart in grayscale; colour
# repeats the same distinction:
#   data      recorded human material   folded corner, thin solid, green tint
#   op        manipulation or analysis  square corners, thin solid, white
#   model     audited model or stage    square corners, thick dark edge, orange tint
#   follower  constructed control       dashed blue edge (Fig. 2's follower blue), pale blue
#   measure   measured or compared      large rounded corners, thin solid, lavender
FIG1_ROWS = [
    ("(a)", [("Human speech\n12 cells\n3 voices", "data"),
             ("span \u00d70.50\u2013\u00d71.67;  register +2, +4 st\n"
              "loudness \u00d70.60, \u00d71.50;  rate \u00d70.90\n"
              "effort = register + rate + level\n"
              "c100 = unchanged resynthesis", "op")]),
    # a list = model box over follower box, joined by a bracket into one arrow
    ("(b)", [(["EmoFace, EmoTalk,\nEMOTE, MEDTalk", "F0 and energy\nfollowers"],
              ["model", "follower"]),
             ("brow\nlid\njaw", "measure"),
             ("ln ratio\nvs null;\nslope\nper cue", "measure")]),
    ("(c)", [("RAVDESS strong\nvs normal: 670\npairs, 24 actors", "data"),
             ("exploratory\nregression\non 4 voice cues", "op"),
             ("estimated\nbrow ratio\nper step", "measure")]),
    ("(d)", [("decoder input", "model"), ("model output", "model")]),
]
FIG1_COMPARE_LABEL = "compare: \u00d7[0.5, 2]"
# The height the page layout is set for (243.780 x 183.294 pt = 86.0 x 64.662 mm). The row
# gaps are scaled, by at most 10 %, so the figure comes out exactly this high.
FIG1_TARGET_H_PT = 183.2944


def fig1():
    """The protocol diagram: the ladder and the models (a, b), the human reference (c), and
    the two model-internal stages probed (d), below a hairline."""
    # lw in pt; pad = horizontal text padding in inches
    KIND = {
        "data":     dict(fc="#dff0e6", ec="#2b7a55", lw=0.7, ls="solid", pad=0.095),
        "op":       dict(fc="#ffffff", ec="#77766f", lw=0.6, ls="solid", pad=0.055),
        "model":    dict(fc="#fbe3c6", ec="#3b3a37", lw=1.25, ls="solid", pad=0.060),
        "follower": dict(fc="#eef4fc", ec="#2a78d6", lw=1.0, ls=(0, (3.0, 1.7)), pad=0.060),
        "measure":  dict(fc="#ece2f4", ec="#6a4a8a", lw=0.7, ls="solid", pad=0.080),
    }
    W = COLW
    FOLD, FOLD_FC = 0.070, "#8cc2a4"  # data: folded-corner size (in) and flap fill
    ROUND_MEASURE = 0.085             # measure: corner radius, in
    PADY = 0.045                      # vertical text padding, in
    PADC, GAPC = 0.036, 0.035         # model/follower pair: vertical padding, gap between boxes
    BRK = 0.055                       # merge bracket to the right of the pair, in
    TAGW, RM, TOPM, BOTM = 0.25, 0.035, 0.030, 0.030
    MIN_GAP = 0.14                    # smallest horizontal gap between boxes (room for an arrow)
    GAP_AFTER = {"(a)": 0.175, "(b)": 0.21, "(c)": 0.14}

    # ---- measure every text and size the boxes
    mfig = plt.figure(figsize=(W, 3))
    ren = mfig.canvas.get_renderer()

    def measure(s):
        t = mfig.text(0, 0, s, fontsize=FS, linespacing=LS, ha="left", va="bottom",
                      multialignment="center")
        bb = t.get_window_extent(renderer=ren)
        t.remove()
        return bb.width / mfig.dpi, bb.height / mfig.dpi

    usable = W - TAGW - RM
    layout = []
    for tag, specs in FIG1_ROWS:
        boxes = []
        for text, kind in specs:
            if isinstance(text, list):                      # model box over follower box
                sz = [measure(p) for p in text]
                pad = max(KIND[k]["pad"] for k in kind)
                hs = [th + 2 * PADC for _, th in sz]
                boxes.append(dict(w=max(w for w, _ in sz) + 2 * pad, nat_h=sum(hs) + GAPC,
                                  parts=list(zip(text, kind, hs, sz)), kind=kind[0]))
            else:
                tw, th = measure(text)
                boxes.append(dict(w=tw + 2 * KIND[kind]["pad"], nat_h=th + 2 * PADY,
                                  text=text, kind=kind, tsz=(tw, th)))
        n = len(boxes)
        if sum(b["w"] for b in boxes) + MIN_GAP * (n - 1) > usable + 1e-9:
            raise RuntimeError("Fig. 1 row %s is too wide for one column" % tag)
        layout.append(dict(tag=tag, boxes=boxes,
                           h=max(b["nat_h"] for b in boxes),                    # one height per row
                           gap=(usable - sum(b["w"] for b in boxes)) / (n - 1),  # justified
                           gap_after=GAP_AFTER.get(tag, 0.0)))
    plt.close(mfig)

    # ---- pin the height by scaling the row gaps
    gaps = sum(r["gap_after"] for r in layout)
    fixed = TOPM + BOTM + sum(r["h"] for r in layout)
    scale = (FIG1_TARGET_H_PT / 72.0 - fixed) / gaps
    if not 0.9 <= scale <= 1.1:
        print("WARNING Fig. 1: height not pinned (gap scale %.3f); drawn at its natural height"
              % scale, file=sys.stderr)
        scale = 1.0
    y = TOPM
    for r in layout:
        r["gap_after"] *= scale
        r["y"] = y
        x = TAGW
        for b in r["boxes"]:
            b.update(x0=x, x1=x + b["w"], yt=y, h=r["h"])
            x += b["w"] + r["gap"]
        y += r["h"] + r["gap_after"]
    H = y + BOTM

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                                       # y grows downward, inches
    ax.axis("off")

    def put_text(s, xc, yc):
        ax.text(xc, yc, s, ha="center", va="center", fontsize=FS, linespacing=LS,
                multialignment="center", color=INK, zorder=6)

    def arrow(x0, y0, x1, y1, both=False):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=5,
                    arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", lw=0.8,
                                    color=INK, shrinkA=0, shrinkB=0, mutation_scale=7))

    def half(b):                                            # half the outline width, in
        return KIND[b["kind"]]["lw"] / 144

    def draw_box(b):
        x0, x1, yt, h = b["x0"], b["x1"], b["yt"], b["h"]
        yb = yt + h
        k = KIND[b["kind"]]
        if "parts" in b:            # model box over follower box, merged by a bracket
            (t_top, k_top, h0, sz0), (t_bot, k_bot, h1, sz1) = b["parts"]
            extra = (h - (h0 + h1 + GAPC)) / 2              # > 0 only if another box is taller
            spans = ((t_top, KIND[k_top], yt, yt + h0 + extra),
                     (t_bot, KIND[k_bot], yb - h1 - extra, yb))
            cx = (x0 + x1) / 2
            mids = []
            for s, kk, ya, yz in spans:
                ax.add_patch(Rectangle((x0, ya), x1 - x0, yz - ya, fc=kk["fc"], ec=kk["ec"],
                                       lw=kk["lw"], ls=kk["ls"], joinstyle="miter", zorder=3))
                mids.append((ya + yz) / 2)
                put_text(s, cx, mids[-1])
            xb = x1 + BRK
            ax.add_line(Line2D([x1 + KIND[k_top]["lw"] / 144, xb, xb, x1 + KIND[k_bot]["lw"] / 144],
                               [mids[0], mids[0], mids[1], mids[1]], color=INK, lw=0.8,
                               solid_joinstyle="miter", solid_capstyle="butt", zorder=5))
            b["out_x"] = xb
            return
        if b["kind"] == "data":
            ax.add_patch(Polygon([(x0, yt), (x1 - FOLD, yt), (x1, yt + FOLD), (x1, yb), (x0, yb)],
                                 closed=True, fc=k["fc"], ec=k["ec"], lw=k["lw"],
                                 joinstyle="miter", zorder=3))
            ax.add_patch(Polygon([(x1 - FOLD, yt), (x1 - FOLD, yt + FOLD), (x1, yt + FOLD)],
                                 closed=True, fc=FOLD_FC, ec=k["ec"], lw=k["lw"],
                                 joinstyle="miter", zorder=4))
        elif b["kind"] == "measure":
            r = min(ROUND_MEASURE, h / 2, (x1 - x0) / 2)
            ax.add_patch(FancyBboxPatch((x0, yt), x1 - x0, h,
                                        boxstyle="round,pad=0,rounding_size=%g" % r,
                                        fc=k["fc"], ec=k["ec"], lw=k["lw"], zorder=3))
        else:                                               # op, model: square corners
            ax.add_patch(Rectangle((x0, yt), x1 - x0, h, fc=k["fc"], ec=k["ec"], lw=k["lw"],
                                   ls=k["ls"], joinstyle="miter", zorder=3))
        put_text(b["text"], (x0 + x1) / 2, yt + h / 2)

    for row in layout:
        yc = row["y"] + row["h"] / 2
        ax.text(0.03, yc, row["tag"], ha="left", va="center", fontsize=FS, weight="bold",
                color=INK)
        for b in row["boxes"]:
            draw_box(b)
        for b0, b1 in zip(row["boxes"][:-1], row["boxes"][1:]):
            arrow(b0.get("out_x", b0["x1"] + half(b0)), yc, b1["x0"] - half(b1), yc)

    A, B, C, D = layout
    # hairline separating the model-internal probe from the dose protocol
    ysep = D["y"] - C["gap_after"] / 2
    ax.plot([0.03, W - RM], [ysep, ysep], color=BASELINE, lw=0.6, zorder=1)

    # the ladder feeds the models: the manipulation box down to the model box
    a2, b1 = A["boxes"][1], B["boxes"][0]
    lo, hi = max(a2["x0"], b1["x0"]), min(a2["x1"], b1["x1"])
    if hi - lo > 0.12:
        xa = (lo + hi) / 2
        arrow(xa, a2["yt"] + a2["h"] + half(a2), xa, b1["yt"] - half(b1))
    else:
        arrow(a2["x0"] + 0.1, a2["yt"] + a2["h"] + half(a2), b1["x1"] - 0.1, b1["yt"] - half(b1))
    # the model response against the human estimate
    b3, c3 = B["boxes"][-1], C["boxes"][-1]
    lo, hi = max(b3["x0"], c3["x0"]), min(b3["x1"], c3["x1"])
    xc = (lo + hi) / 2
    yb3, yc3 = b3["yt"] + b3["h"] + half(b3), c3["yt"] - half(c3)
    arrow(xc, yb3, xc, yc3, both=True)
    ax.text(xc - 0.07, (yb3 + yc3) / 2, FIG1_COMPARE_LABEL, ha="right", va="center",
            fontsize=FS, color=INK)
    return fig


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isdir(out):
        os.makedirs(out)
    d, pv = load_values()
    for name, fig in (("fig1_protocol", fig1()), ("fig2_dose", fig2(d, pv))):
        path = os.path.join(out, name + ".pdf")
        fig.savefig(path, metadata={"Creator": "make_figures.py", "CreationDate": None})
        w, h = fig.get_size_inches()
        plt.close(fig)
        print("%s  %.1f x %.1f mm" % (path, w * 25.4, h * 25.4))


if __name__ == "__main__":
    main()

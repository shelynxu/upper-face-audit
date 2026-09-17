#!/usr/bin/env python3
"""
make_perception_figure.py -- the perception figure: what the perception test showed
viewers (panel a), and what they judged (panel b).

    python code/figures/make_perception_figure.py [--frames DIR] [--check-scored]

Panel (b), the figure's result, is redrawn from data/derived/figure_perception_values.csv:
one row per estimate, square = viewer mean, whisker = 95 % bootstrap CI, the Holm-adjusted
two-sided p of the three hypotheses in the right-hand column, and the mean [95 % CI] printed
beside it so the interval can be read without the text. On each row sits one small open
symbol per listener group at that group's viewer mean (triangle = Mandarin link, square =
English link, diamond = Japanese link, named in the key under the axis). No group CI and no
group p is drawn: the per-group hypotheses are secondary and the group differences
exploratory, so the figure shows where the groups sit and nothing that reads as a
confirmatory group test.

Panel (a) is the stimulus: three crops of one frame of one stimulus pair -- E (the model as
shipped), H (the same render with the actor's own brow motion added) and the pixel
difference |H - E|. The two renders come from the same rig, scene, camera and audio and
differ only in the brow channel, so the difference map localises exactly what the viewer saw
change. Those frames come out of the renders, which are not part of this release; without
--frames the panel is drawn empty and a note says so. Empty rather than dropped, because the
rest of the figure then keeps exactly the size and position it has in the paper, and panel
(b) can be compared with the published figure pixel for pixel.

--check-scored recomputes the four viewer means and the twelve listener-group means from
data/perception/perception_scored.csv (participant means over the primary sample) and stops
on any disagreement with the released values.

Two things are checked on every run. The numbers the figure prints must match the strings the
values CSV printed for the paper (printed_mean_ci, printed_holm), and each group symbol must
land on the level the values CSV recorded for it. The row pitch is not fixed: the figure is
drawn at the first pitch at which the layout self-check finds nothing colliding -- no label,
number, whisker or symbol touching another or leaving the figure -- and the height is then
trimmed to the lowest text. The released values draw at 11 pt per row.

Outputs, into --out only (never into data/):
    perception_figure.pdf   vector PDF, fonts embedded as TrueType (Type 42)
    perception_figure.png   the same figure at 300 dpi, for checking by eye

Print rules: one column wide (86 mm), every label 9 pt at print size (include with
width=\\columnwidth, scale 1.0), no Type 3 fonts, no colour-only encoding -- the groups are
told apart by shape and named in the key, and the single accent marks the silent check.
"""
import argparse
import csv
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.markers import MarkerStyle  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

VALUES_CSV = os.path.join(ROOT, "data", "derived", "figure_perception_values.csv")
SCORED_CSV = os.path.join(ROOT, "data", "perception", "perception_scored.csv")
OUT_DIR = os.path.join(ROOT, "figures_out")

# ---------------------------------------------------------------- style ----
FS = 9
COLW = 86.0 / 25.4
W_PT = COLW * 72.0

INK = "#0b0b0b"
SECOND = "#52514e"
BASELINE = "#c3c2b7"
HYP = INK
PASSBAR = "#2a78d6"
PANEL = "#f6f6f3"
MINUS = "−"

# listener-group layer, the same symbols and the same muted ink as the dose figure's
# per-voice layer
GROUPS = ["cn", "en", "jp"]
GROUP_KEY = {"cn": "zh", "en": "en", "jp": "ja"}          # ISO 639-1 codes, as in the key
PV_SHAPE = {"cn": "^", "en": "s", "jp": "D"}
PV_SCALE = {"^": 1.12, "s": 0.88, "D": 0.80}              # equalise the visual size of the shapes
PV_COLOR = "#77766f"                                      # muted ink (contrast 4.6:1 on white)
PV_MEW = 0.6
PV_MS = 2.8                                               # pt, before the per-shape factor
# Symbol levels, pt from the whisker line, positive = below. A symbol that would overlap an
# already placed symbol on its level moves to the next level. The second level sits ABOVE the
# whisker, not in a second strip below: a second strip below lands nearer the NEXT row's line
# than its own. The third level (three group means within a symbol width) is that second strip
# below. The vertical position carries no data.
PV_LEVELS = (4.3, -4.3, 7.5)
CAP_HALF_PT = 1.87          # whisker cap half-height in pt
PV_OVERLAP = 0.15
LEVEL_TOUCH_PT = 0.6        # symbols on adjacent levels of one row may touch by this much (strokes)
# pt per row; the first that passes the layout check is used
PITCHES = (11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
KEY_GAP_PT = 1.5            # axis title -> key line
KEY_TITLE = "group means (exploratory):"

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "xtick.major.size": 2.5,
    "xtick.minor.width": 0.5, "xtick.minor.size": 1.5,
    "ytick.major.size": 0, "xtick.major.pad": 1.5, "ytick.major.pad": 3,
    "axes.labelpad": 2, "axes.titlepad": 3,
    "axes.edgecolor": SECOND, "xtick.color": SECOND,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.unicode_minus": True,
    "savefig.dpi": 300,
})

ROW_ORDER = ["mute", "P1", "P2", "P3"]        # the silent check first, then the three hypotheses
HOLM_ROWS = ["P1", "P2", "P3"]                # only the hypothesis family carries a Holm p
# the contrast each row is the participant mean of, for --check-scored
SCORED_CONTRAST = {"mute": "mute:H_vs_E@strong", "P2": "H_vs_R@strong"}
# the released values carry three decimals, so a recomputed mean can sit exactly on half of
# that last digit; the slack keeps that boundary from failing on binary representation alone
SCORED_TOL = 5e-4
SCORED_SLACK = 1e-9


# -------------------------------------------------------------- numbers ----
def fmt2(x, sign=False):
    q = Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if q == 0:
        q = Decimal("0.00")
    s = ("%+.2f" if sign else "%.2f") % q
    return s.replace("-", MINUS)


def fmt_p(p):
    p = float(p)
    if p < 0.001:
        return "<0.001"
    q = Decimal(repr(p))
    if p >= 0.0995:
        return str(q.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    return str(q.quantize(Decimal(1).scaleb(q.adjusted() - 1), rounding=ROUND_HALF_UP))


def read_blocks(path):
    """The values CSV holds three blocks separated by a blank line: the pooled rows, the
    listener-group rows, and one row naming the frame behind panel (a). Each block is
    returned as a list of dicts keyed by its own header."""
    blocks, rows = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for rec in csv.reader(fh):
            if not any(f.strip() for f in rec):
                if rows:
                    blocks.append(rows)
                rows = []
            else:
                rows.append(rec)
    if rows:
        blocks.append(rows)
    out = {}
    for b in blocks:
        head = b[0][0]
        if head in ("key", "group_row"):
            out[head] = [dict(zip(b[0], r)) for r in b[1:]]
        elif head.startswith("panel"):
            out["panel"] = b[0]
    return out


def load_values(path):
    b = read_blocks(path)
    for want in ("key", "group_row"):
        if want not in b:
            sys.exit("%s: no '%s' block found" % (path, want))

    pooled, labels, holm, printed, ns = {}, {}, {}, {}, set()
    for r in b["key"]:
        key = r["key"]
        pooled[key] = (float(r["mean"]), float(r["ci_lo"]), float(r["ci_hi"]),
                       float(r["dz"]), int(r["n"]))
        labels[key] = r["label"]
        printed[key] = (r["printed_mean_ci"], r["printed_holm"])
        ns.add(int(r["n"]))
        if r["holm_p_two_sided"]:
            holm[key] = float(r["holm_p_two_sided"])
    missing = [k for k in ROW_ORDER if k not in pooled]
    if missing:
        sys.exit("%s: rows missing: %s" % (path, ", ".join(missing)))
    if sorted(holm) != sorted(HOLM_ROWS):
        sys.exit("%s: Holm p expected for %s, found %s" % (path, HOLM_ROWS, sorted(holm)))
    if len(ns) != 1:
        sys.exit("%s: the pooled rows disagree about n: %s" % (path, sorted(ns)))
    n = ns.pop()

    groups = {"n": {}, "mean": {}, "n_row": {}, "level": {}}
    for key in ROW_ORDER:
        groups["mean"][key], groups["n_row"][key], groups["level"][key] = {}, {}, {}
    for r in b["group_row"]:
        key, g = r["group_row"], r["group"]
        if key not in pooled:
            sys.exit("%s: group row for unknown row %r" % (path, key))
        if g not in GROUPS:
            sys.exit("%s: unknown listener group %r" % (path, g))
        groups["mean"][key][g] = float(r["mean"])
        groups["n_row"][key][g] = int(r["n_row"])
        groups["n"][g] = int(r["n_group"])
        if r.get("symbol_level", "") != "":
            groups["level"][key][g] = int(r["symbol_level"])
    if sum(groups["n"].values()) != n:
        sys.exit("%s: group n %s do not sum to n = %d" % (path, groups["n"], n))
    for key in ROW_ORDER:
        if sorted(groups["mean"][key]) != sorted(groups["n"]):
            sys.exit("%s: row %s has group means for %s only"
                     % (path, key, sorted(groups["mean"][key])))

    # the figure must print exactly the numbers the released table prints
    for key in ROW_ORDER:
        m, lo, hi, _, _ = pooled[key]
        drawn = "%s [%s, %s]" % (fmt2(m, sign=True), fmt2(lo), fmt2(hi))
        want_mean, want_holm = printed[key]
        if drawn != want_mean:
            sys.exit("STOP: %s would print %s, the values CSV prints %s" % (key, drawn, want_mean))
        if want_holm and fmt_p(holm[key]) != want_holm:
            sys.exit("STOP: %s Holm p would print %s, the values CSV prints %s"
                     % (key, fmt_p(holm[key]), want_holm))

    panel_a = {}
    if "panel" in b and len(b["panel"]) >= 6:
        p = b["panel"]
        panel_a = {"cell": p[1], "peak": p[3], "fps": p[5]}
    return pooled, labels, holm, n, groups, panel_a


def check_scored(path, pooled, groups):
    """Recompute every mean the figure draws from the per-screen scores and stop on a
    disagreement. P1 is the mean of the normal and strong H-vs-E viewer means, P2 the H-vs-R
    strong mean, P3 strong minus normal, and the silent check the muted H-vs-E pairs -- the
    frozen analyser's definitions, over the primary sample."""
    per, group_of = {}, {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["in_primary"] != "1" or r["score_H"] == "":
                continue
            p = r["participant"]
            per.setdefault(p, {}).setdefault(r["contrast"], []).append(float(r["score_H"]))
            group_of[p] = r["listener_group"]

    def mean(xs):
        return sum(xs) / float(len(xs))

    viewer = {k: {} for k in ROW_ORDER}
    for p, by in per.items():
        hn, hs = by.get("H_vs_E@normal"), by.get("H_vs_E@strong")
        if hn and hs:
            viewer["P1"][p] = (mean(hn) + mean(hs)) / 2.0
            viewer["P3"][p] = mean(hs) - mean(hn)
        for key, contrast in SCORED_CONTRAST.items():
            if by.get(contrast):
                viewer[key][p] = mean(by[contrast])

    print("  --check-scored against %s" % os.path.relpath(path, ROOT))
    worst = 0.0
    for key in ROW_ORDER:
        rows = [(None, sorted(viewer[key]), pooled[key][0], pooled[key][4])]
        for g in GROUPS:
            rows.append((g, [p for p in sorted(viewer[key]) if group_of[p] == g],
                         groups["mean"][key][g], groups["n_row"][key][g]))
        for g, ps, want, n_want in rows:
            what = "pooled" if g is None else g
            if not ps:
                sys.exit("STOP: %s %s has no scored participant" % (key, what))
            got = mean([viewer[key][p] for p in ps])
            d = abs(got - want)
            worst = max(worst, d)
            if d > SCORED_TOL + SCORED_SLACK:
                sys.exit("STOP: %s %s recomputes to %.6f, the values CSV holds %.4f (%.6f apart)"
                         % (key, what, got, want, d))
            if len(ps) != n_want:
                sys.exit("STOP: %s %s has %d participants, the values CSV holds %d"
                         % (key, what, len(ps), n_want))
        print("    %-4s ok (pooled %+.4f; %s)" % (
            key, pooled[key][0],
            "  ".join("%s %+.4f" % (GROUP_KEY[g], groups["mean"][key][g]) for g in GROUPS)))
    print("    largest difference %.6f; the tolerance is %.4f, half of the last decimal the "
          "released values carry" % (worst, SCORED_TOL))


# --------------------------------------------------------------- frames ----
# the crop of the render the panel shows: the brow band and the eyes only
CROP = (128, 332, 50, 510)
ARO = (CROP[1] - CROP[0]) / float(CROP[3] - CROP[2])       # crop aspect (height / width)


def load_frames(src_dir, cell):
    """The two rendered frames and their difference, as written from the mp4 pair."""
    path = os.path.join(src_dir, "peak_%s.npz" % cell)
    if not os.path.isfile(path):
        sys.exit("--frames: %s not found" % path)
    z = np.load(path)
    r0, r1, c0, c1 = CROP
    E, H, d = z["E"][r0:r1, c0:c1], z["H"][r0:r1, c0:c1], z["diff"][r0:r1, c0:c1]
    # the panel box is sized from CROP, so a frame the crop does not fit would be drawn stretched
    if E.shape[:2] != (r1 - r0, c1 - c0):
        sys.exit("--frames: %s holds %s frames, too small for the crop %s"
                 % (path, z["E"].shape[:2], CROP))
    # the E brow line: the darkest row inside the band that carries the change
    lum = E.astype(np.float32).mean(axis=2)
    band = d.mean(axis=1)
    b0, b1 = int(np.argmax(band)) - 30, int(np.argmax(band)) + 30
    brow_row = b0 + int(np.argmin(lum[max(b0, 0):b1].mean(axis=1)))
    return E, H, d, brow_row


def hot(E, d):
    """|H - E| as a warm overlay on a desaturated E, gamma-lifted so the small off-brow
    render noise stays visibly smaller than the brow change."""
    dn = np.clip(d / max(d.max(), 1e-9), 0, 1) ** 0.55
    base = E.astype(np.float32).mean(axis=2, keepdims=True).repeat(3, axis=2)
    out = base * 0.5
    out[..., 0] += 250 * dn
    out[..., 1] += 105 * dn
    out[..., 2] += 25 * dn
    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- draw ----
# panel (a) was laid out against this reference height; holding its offsets in pt keeps the
# panel the same size and in the same place whatever height the layout check settles on
H_REF_PT = 2.18 * 72.0
A_TOP_PT = (1.0 - 0.915) * H_REF_PT
A_LABEL_PT = (1.0 - 0.995) * H_REF_PT
A_X0, A_GAP = 0.008, 0.006
A_W = (0.995 - 0.008) / 3.0 - A_GAP

B_GAP = 5.0
LBL_X = A_X0 * W_PT
INDENT = 5.0
GAP_LBL, GAP_NUM, R_MARGIN = 5.0, 9.0, 1.0
Y_HEAD1, Y_MC, Y_HEAD2 = 0.0, -1.0, -2.3
Y_HYP = {"P1": -3.3, "P2": -4.3, "P3": -5.3}
Y_TOP, Y_BOT_MIN = 0.5, -5.85        # Y_BOT drops further if the last row's symbols need it
BOTTOM_MARGIN_PT = 1.5
HOLM_CLEAR_PT = 2.0


def _w_pt(fig, renderer, s, **kw):
    t = fig.text(0, 0, s, fontsize=FS, **kw)
    w = t.get_window_extent(renderer=renderer).width * 72.0 / fig.dpi
    t.remove()
    return w


def _half_h(marker, ms):
    st = MarkerStyle(marker)
    ext = st.get_path().transformed(st.get_transform()).get_extents()
    return 0.5 * (ext.y1 - ext.y0) * ms


def _half_w(marker, ms):
    st = MarkerStyle(marker)
    ext = st.get_path().transformed(st.get_transform()).get_extents()
    return 0.5 * (ext.x1 - ext.x0) * ms


def pv_levels(values, xlo, scale):
    """Level per group for one row: the lowest level where the symbol does not overlap an
    already placed same-level symbol by more than PV_OVERLAP of their mean width."""
    placed, out = {}, {}
    for g in GROUPS:
        if g not in values:
            continue
        x = (values[g] - xlo) * scale
        mk = PV_SHAPE[g]
        wv = 2 * _half_w(mk, PV_MS * PV_SCALE[mk])
        for lvl in range(len(PV_LEVELS)):
            if all(abs(x - x2) >= (1 - PV_OVERLAP) * (wv + w2) / 2 for x2, w2 in placed.get(lvl, [])):
                break
        else:
            sys.exit("STOP: group symbols need more than %d levels" % len(PV_LEVELS))
        placed.setdefault(lvl, []).append((x, wv))
        out[g] = lvl
    return out


class LayoutError(Exception):
    pass


def draw(pooled, labels, holm, n, groups, frames, H_pt, pitch):
    fig = plt.figure(figsize=(COLW, H_pt / 72.0))
    renderer = fig.canvas.get_renderer()

    def fx(pt):
        return pt / W_PT

    def fy(pt):
        return 1.0 - pt / H_pt

    # ---- panel (a): three crops, or three empty frames without --frames ---------
    a_h_pt = A_W * ARO * W_PT
    titles = ["E (as shipped)", "H (brows added)", r"$|$H$-$E$|$"]
    if frames is not None:
        E, H, d, brow_row = frames
        images = [E, H, hot(E, d)]
    for i, ttl in enumerate(titles):
        ax = fig.add_axes([A_X0 + i * (A_W + A_GAP), fy(A_TOP_PT + a_h_pt), A_W, a_h_pt / H_pt])
        if frames is not None:
            ax.imshow(images[i], interpolation="lanczos", aspect="auto")
            if i < 2:   # shared rule at the E brow line, so H's raise can be read off it
                ax.axhline(brow_row, color="#ffffff", lw=1.0, alpha=0.9, zorder=3)
                ax.axhline(brow_row, color=INK, lw=0.5, ls=(0, (2.2, 1.6)), alpha=0.95, zorder=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(BASELINE)
            s.set_linewidth(0.5)
        ax.set_title(ttl, fontsize=FS, color=INK, pad=2.0)
    fig.text(A_X0, fy(A_LABEL_PT), "(a)", fontsize=FS, fontweight="bold", color=INK, va="top")

    # ---- panel (b): geometry ---------------------------------------------------
    num = {k: "%s [%s, %s]" % (fmt2(m, sign=True), fmt2(lo), fmt2(hi))
           for k, (m, lo, hi, _, _) in pooled.items()}
    head_num = r"mean [95$\,$% CI]"
    num_w = max([_w_pt(fig, renderer, s) for s in num.values()] + [_w_pt(fig, renderer, head_num)])
    lbl_w = max(_w_pt(fig, renderer, labels[k]) for k in ROW_ORDER)
    NUM_X = W_PT - R_MARGIN - num_w
    PLOT_R = NUM_X - GAP_NUM
    PLOT_L = LBL_X + INDENT + lbl_w + GAP_LBL
    plot_w = PLOT_R - PLOT_L

    gvals = [v for by in groups["mean"].values() for v in by.values()]
    ghyp = [v for k, by in groups["mean"].items() if k in holm for v in by.values()]
    sym_half = max(_half_w(PV_SHAPE[g], PV_MS * PV_SCALE[PV_SHAPE[g]]) for g in GROUPS)
    lo_all = min([r[1] for r in pooled.values()] + gvals)
    hi_all = max([r[2] for r in pooled.values()] + gvals)
    hi_hyp = max([pooled[k][2] for k in holm] + ghyp)
    XLO = min(-0.10, lo_all - 0.05)
    # leave room on the right for the Holm column, so no whisker or symbol runs into it
    c = max(_w_pt(fig, renderer, fmt_p(holm[k])) for k in holm) + 2.0 * HOLM_CLEAR_PT + sym_half
    XHI = max(1.22, hi_all + 0.05, (plot_w * hi_hyp - c * XLO) / (plot_w - c))
    scale = plot_w / (XHI - XLO)

    levels = {k: pv_levels(groups["mean"].get(k, {}), XLO, scale) for k in ROW_ORDER}
    below = [PV_LEVELS[v] for v in levels["P3"].values() if PV_LEVELS[v] > 0]
    y_bot = Y_BOT_MIN
    if below:
        need = (max(below) + max(_half_h(PV_SHAPE[g], PV_MS * PV_SCALE[PV_SHAPE[g]])
                                 for g in GROUPS) + PV_MEW / 2.0 + 1.0) / pitch
        y_bot = min(Y_BOT_MIN, Y_HYP["P3"] - need)

    b_top_pt = A_TOP_PT + a_h_pt + B_GAP
    ax_h_pt = (Y_TOP - y_bot) * pitch
    ax = fig.add_axes([fx(PLOT_L), fy(b_top_pt + ax_h_pt), plot_w / W_PT, ax_h_pt / H_pt])
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(y_bot, Y_TOP)

    # the silent check answers a different question; band it rather than colour it alone
    ax.add_patch(Rectangle((fx(LBL_X - 1.5), Y_MC - 0.5), fx(W_PT - 0.5) - fx(LBL_X - 1.5), 1.0,
                           transform=tr, facecolor=PANEL, edgecolor="none", zorder=0,
                           clip_on=False))
    for y0, y1 in ((Y_MC - 0.5, Y_MC + 0.5), (y_bot, Y_HEAD2 - 0.42)):
        ax.plot([0, 0], [y0, y1], color=SECOND, lw=0.8, zorder=1, solid_capstyle="butt")

    fig.text(fx(LBL_X), 0, "(b)", transform=tr, fontsize=FS, fontweight="bold", color=INK,
             va="center", ha="left")
    b_w = _w_pt(fig, renderer, "(b)", fontweight="bold")
    fig.text(fx(LBL_X + b_w + 3.5), Y_HEAD1, "no sound: visibility check", transform=tr,
             fontsize=FS, color=SECOND, style="italic", va="center", ha="left")
    fig.text(fx(NUM_X), Y_HEAD1, head_num, transform=tr, fontsize=FS, color=SECOND,
             va="center", ha="left")
    fig.text(fx(LBL_X), Y_HEAD2, "with sound: hypotheses", transform=tr, fontsize=FS,
             color=SECOND, style="italic", va="center", ha="left")
    ax.text(XHI, Y_HEAD2, r"Holm $p$", fontsize=FS, color=SECOND, va="center", ha="right")

    # marks for the layout check: dicts with kind line / cap / square / sym / key; point-like
    # marks carry their half extents in pt (from the marker path, stroke included) because
    # matplotlib's window extent of a marker ignores its shape and stroke
    marks = []
    for key in ROW_ORDER:
        m, lo, hi, _, _ = pooled[key]
        y = Y_MC if key == "mute" else Y_HYP[key]
        col = PASSBAR if key == "mute" else HYP
        fig.text(fx(LBL_X + INDENT), y, labels[key], transform=tr, fontsize=FS, color=INK,
                 va="center", ha="left")
        for a in ax.plot([lo, hi], [y, y], color=col, lw=1.0, zorder=3, solid_capstyle="butt"):
            marks.append({"kind": "line", "row": key, "art": a})
        for x in (lo, hi):
            cap = CAP_HALF_PT / pitch
            for a in ax.plot([x, x], [y - cap, y + cap], color=col, lw=1.0, zorder=3):
                marks.append({"kind": "cap", "row": key, "art": a})
        for a in ax.plot([m], [y], marker="s", ms=4.2, mfc=col, mec="white", mew=0.7,
                         ls="none", zorder=4):
            marks.append({"kind": "square", "row": key, "art": a, "xy": (m, y),
                          "half": (_half_w("s", 4.2) + 0.35, _half_h("s", 4.2) + 0.35)})
        fig.text(fx(NUM_X), y, num[key], transform=tr, fontsize=FS, color=INK, va="center", ha="left")
        if key in holm:
            ax.text(XHI, y, fmt_p(holm[key]), fontsize=FS, color=INK, va="center", ha="right")
        for g in GROUPS:
            if g not in groups["mean"].get(key, {}):
                continue
            lvl = levels[key][g]
            mk = PV_SHAPE[g]
            ms = PV_MS * PV_SCALE[mk]
            yy = y - PV_LEVELS[lvl] / pitch
            art = ax.plot([groups["mean"][key][g]], [yy], ls="none", marker=mk, ms=ms,
                          mfc="white", mec=PV_COLOR, mew=PV_MEW, zorder=4.5)[0]
            marks.append({"kind": "sym", "row": key, "group": g, "level": lvl, "art": art,
                          "xy": (groups["mean"][key][g], yy),
                          "half": (_half_w(mk, ms) + PV_MEW / 2.0, _half_h(mk, ms) + PV_MEW / 2.0)})

    ticks = [t for t in (0.0, 0.5, 1.0, 1.5) if XLO <= t <= XHI]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0" if t == 0 else ("%.1f" % t) for t in ticks])
    ax.set_xticks([t for t in np.arange(-0.25, XHI, 0.25) if XLO <= t and t not in ticks], minor=True)
    ax.set_yticks([])
    ax.set_xlabel(r"comparison score toward H ($-3$ to $+3$); $n=%d$ viewers" % n, fontsize=FS, color=INK)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_bounds(XLO, XHI)

    # ---- key line under the axis title ----------------------------------------
    fig.canvas.draw()
    k = 72.0 / fig.dpi
    xl = ax.xaxis.label.get_window_extent(renderer=renderer)
    key_top_pt = H_pt - xl.y0 * k + KEY_GAP_PT          # pt from the top
    parts = [(None, KEY_TITLE)] + [(g, "%s (%d)" % (GROUP_KEY[g], groups["n"][g]))
                                   for g in GROUPS if g in groups["n"]]
    SYM_GAP, ITEM_GAP, TITLE_GAP = 2.5, 6.5, 4.0
    widths = []
    for g, s in parts:
        w = _w_pt(fig, renderer, s, style="italic" if g is None else "normal")
        sw = 0.0 if g is None else 2 * _half_w(PV_SHAPE[g], PV_MS * PV_SCALE[PV_SHAPE[g]]) + SYM_GAP
        widths.append((sw, w))
    total = sum(sw + w for sw, w in widths) + TITLE_GAP + ITEM_GAP * (len(parts) - 2)
    cx = (xl.x0 + xl.x1) / 2.0 * k
    x = min(max(cx - total / 2.0, LBL_X), W_PT - R_MARGIN - total)
    key_arts = []
    t_title = None
    for (g, s), (sw, w) in zip(parts, widths):
        if g is not None:
            mk = PV_SHAPE[g]
            ms = PV_MS * PV_SCALE[mk]
            key_arts.append((g, x + _half_w(mk, ms), ms, mk))
            x += sw
        t = fig.text(fx(x), fy(key_top_pt), s, fontsize=FS, va="top", ha="left",
                     color=SECOND if g is None else INK, style="italic" if g is None else "normal")
        if g is None:
            t_title = t
        x += w + (TITLE_GAP if g is None else ITEM_GAP)
    fig.canvas.draw()
    tb = t_title.get_window_extent(renderer=renderer)
    y_mid = (tb.y0 + tb.y1) / 2.0 / fig.bbox.height
    for g, xpt, ms, mk in key_arts:
        art = Line2D([fx(xpt)], [y_mid], transform=fig.transFigure, ls="none", marker=mk, ms=ms,
                     mfc="white", mec=PV_COLOR, mew=PV_MEW)
        fig.add_artist(art)
        marks.append({"kind": "key", "row": "key", "group": g, "art": art, "fig_xy": (fx(xpt), y_mid),
                      "half": (_half_w(mk, ms) + PV_MEW / 2.0, _half_h(mk, ms) + PV_MEW / 2.0)})
    info = {"XLO": XLO, "XHI": XHI, "y_bot": y_bot, "levels": levels,
            "scale_pt_per_unit": scale, "pitch": pitch}
    return fig, ax, marks, info


class _Box:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    def hits(self, o, gap_x=0.0, gap_y=0.0):
        return (self.x0 < o.x1 + gap_x and o.x0 - gap_x < self.x1 and
                self.y0 < o.y1 + gap_y and o.y0 - gap_y < self.y1)

    def v_overlap(self, o):
        return min(self.y1, o.y1) - max(self.y0, o.y0)

    def h_overlap(self, o):
        return min(self.x1, o.x1) - max(self.x0, o.x0)


def layout_check(fig, ax, marks):
    """Raise LayoutError on: text/text overlap; text within HOLM_CLEAR_PT of a whisker, square
    or group symbol; a group symbol touching another row's marks, its own row's caps, a symbol
    of another row, or a same-row symbol beyond the allowed touch; a key symbol touching text;
    anything outside the figure; a group symbol below the axis line. All boxes in pt.
    Returns the lowest text edge in pt above the figure bottom."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = 72.0 / fig.dpi
    W, H = fig.bbox.width * k, fig.bbox.height * k
    tol = 0.3

    def ext_box(artist):
        b = artist.get_window_extent(renderer=r)
        return _Box(b.x0 * k, b.y0 * k, b.x1 * k, b.y1 * k)

    texts = [t for t in fig.texts if t.get_text()]
    for a in fig.axes:
        texts += [t for t in a.texts if t.get_text()]
        texts += [a.title] if a.title.get_text() else []
    texts += [ax.xaxis.label] + [t for t in ax.get_xticklabels() if t.get_text()]
    boxes = [(t.get_text(), ext_box(t)) for t in texts]

    for mk in marks:
        if "half" in mk:
            if mk["kind"] == "key":
                cx, cy = mk["fig_xy"][0] * W, mk["fig_xy"][1] * H
            else:
                px, py = ax.transData.transform(mk["xy"])
                cx, cy = px * k, py * k
            hw, hh = mk["half"]
            mk["box"] = _Box(cx - hw, cy - hh, cx + hw, cy + hh)
        else:
            mk["box"] = ext_box(mk["art"])

    problems = []
    for s, b in boxes:
        if b.x0 < -tol or b.y0 < -tol or b.x1 > W + tol or b.y1 > H + tol:
            problems.append("outside figure: %r" % s)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (si, bi), (sj, bj) = boxes[i], boxes[j]
            if bi.hits(bj, -tol, -tol):
                problems.append("text overlap: %r / %r" % (si, sj))
    for mk in marks:
        b = mk["box"]
        tag = "%s:%s:%s" % (mk["kind"], mk["row"], mk.get("group", ""))
        if b.x0 < -tol or b.y0 < -tol or b.x1 > W + tol or b.y1 > H + tol:
            problems.append("mark outside figure: %s" % tag)
        for s, tb in boxes:
            if mk["kind"] == "key":
                if b.hits(tb):
                    problems.append("key symbol %s touches text %r" % (tag, s))
            elif b.hits(tb, gap_x=HOLM_CLEAR_PT):
                problems.append("text within %.0f pt of %s: %r" % (HOLM_CLEAR_PT, tag, s))

    axis_y = ax.transData.transform((0, ax.get_ylim()[0]))[1] * k
    syms = [m for m in marks if m["kind"] == "sym"]
    for s in syms:
        b = s["box"]
        tag = "%s:%s" % (s["row"], s["group"])
        if b.y0 < axis_y + 0.5:
            problems.append("group symbol %s reaches the axis line" % tag)
        for o in marks:
            if o is s or o["kind"] == "key":
                continue
            ob = o["box"]
            if not b.hits(ob):
                continue
            if o["kind"] == "sym":
                if o["row"] != s["row"]:
                    problems.append("group symbol %s touches symbol %s:%s" % (tag, o["row"], o["group"]))
                elif o["level"] == s["level"]:
                    if b.h_overlap(ob) > PV_OVERLAP * min(b.x1 - b.x0, ob.x1 - ob.x0) + tol:
                        problems.append("group symbols %s and %s:%s overlap on one level"
                                        % (tag, o["row"], o["group"]))
                elif b.v_overlap(ob) > LEVEL_TOUCH_PT:
                    problems.append("group symbols %s and %s:%s overlap across levels"
                                    % (tag, o["row"], o["group"]))
            elif o["row"] != s["row"]:
                problems.append("group symbol %s touches the %s of row %s" % (tag, o["kind"], o["row"]))
            elif o["kind"] == "cap":
                problems.append("group symbol %s touches its own row's cap" % tag)
            # own row's square or whisker line: allowed, a symbol may clip a main marker
    if problems:
        raise LayoutError("; ".join(sorted(set(problems))))
    return min(b.y0 for _, b in boxes)


def build(pooled, labels, holm, n, groups, frames):
    """Draw at the first row pitch that passes the layout check, trimming the figure height to
    the lowest text. Two passes per pitch: the first is drawn tall so nothing is clipped while
    the height is measured, the second is drawn at that measured height."""
    for pitch in PITCHES:
        try:
            H_pt = 300.0
            fig, ax, marks, info = draw(pooled, labels, holm, n, groups, frames, H_pt, pitch)
            low = layout_check(fig, ax, marks)
            plt.close(fig)
            H_pt = H_pt - (low - BOTTOM_MARGIN_PT)
            fig, ax, marks, info = draw(pooled, labels, holm, n, groups, frames, H_pt, pitch)
            low = layout_check(fig, ax, marks)
            return fig, info, H_pt, low
        except LayoutError as e:
            plt.close("all")
            print("  row pitch %.0f pt fails the layout check: %s" % (pitch, e))
    sys.exit("STOP (layout): no row pitch in %s passes" % (PITCHES,))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--values", default=VALUES_CSV, help="the plotted values (default: %(default)s)")
    ap.add_argument("--frames", default=None,
                    help="folder holding peak_<cell>.npz for panel (a); without it the panel "
                         "is drawn empty")
    ap.add_argument("--check-scored", action="store_true",
                    help="recompute every mean from the per-screen scores before drawing")
    ap.add_argument("--scored", default=SCORED_CSV, help="per-screen scores (default: %(default)s)")
    ap.add_argument("--out", default=OUT_DIR, help="output folder (default: %(default)s)")
    a = ap.parse_args()

    pooled, labels, holm, n, groups, panel_a = load_values(a.values)
    if a.check_scored:
        check_scored(a.scored, pooled, groups)

    frames = None
    if a.frames:
        frames = load_frames(a.frames, panel_a.get("cell", ""))

    fig, info, H_pt, low = build(pooled, labels, holm, n, groups, frames)

    # the released table records which level each group symbol was drawn at; the layout engine
    # has to put them back in the same places
    for key in ROW_ORDER:
        for g, want in groups["level"].get(key, {}).items():
            if info["levels"][key][g] != want:
                sys.exit("STOP: %s %s symbol drawn at level %d, the values CSV records %d"
                         % (key, g, info["levels"][key][g], want))

    os.makedirs(a.out, exist_ok=True)
    pdf = os.path.join(a.out, "perception_figure.pdf")
    fig.savefig(pdf)
    fig.savefig(os.path.join(a.out, "perception_figure.png"), dpi=300)
    plt.close(fig)

    print("wrote %s (+ .png)" % pdf)
    print("  values: %s" % a.values)
    if frames is None:
        print("  panel (a) left empty: the rendered frames are not part of this release; pass")
        print("           --frames DIR with peak_%s.npz to draw it" % panel_a.get("cell", "<cell>"))
    else:
        print("  panel (a): %s frame %s of %s fps, from %s"
              % (panel_a.get("cell", "?"), panel_a.get("peak", "?"), panel_a.get("fps", "?"), a.frames))
    print("  size: %.2f x %.2f mm = %.1f pt tall (bottom margin %.2f pt); row pitch %.0f pt; "
          "x range [%.2f, %.2f], %.1f pt per unit"
          % (W_PT * 25.4 / 72, H_pt * 25.4 / 72, H_pt, low, info["pitch"], info["XLO"], info["XHI"],
             info["scale_pt_per_unit"]))
    print("  groups n: %s (pooled n %d)" % (groups["n"], n))
    for key in ROW_ORDER:
        m, lo, hi, _, nk = pooled[key]
        print("  %-5s %+.3f [%+.3f, %+.3f] n %d Holm %-6s | %s" % (
            key, m, lo, hi, nk, fmt_p(holm[key]) if key in holm else "-",
            "  ".join("%s %+.3f(L%d)" % (GROUP_KEY[g], groups["mean"][key][g], info["levels"][key][g])
                      for g in GROUPS if g in groups["mean"].get(key, {}))))


if __name__ == "__main__":
    main()

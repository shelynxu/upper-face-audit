# -*- coding: utf-8 -*-
"""analyse2.py -- dose tables, null floors, G2 effort comparison, E3 metric validation, figures.  Env: GPTSoVits.  CPU, 1 process.

Reads (lane): results.csv, injection.csv, g2_measures.json, score_meta.json, refs.json
Reads (read-only): human2/g2_coefficients.json + g2_boot_draws.npz (through the lane copy of g2_predict.py),
                   human2/follow_records.csv (human follow index, 'audit' path = the same f0_peaks/detect_events rule),
                   human/human_audit_metrics.csv (human layer_coupling)
Writes: dose_tables.csv, ratio_tables.csv, null_floor.csv, g2_predictions.csv, g2_compare.csv, e3_follow.csv,
        e3_injection.csv, e3_separation.csv, audit2_key.json, audit2_tables.md, fig_*.png

DOSE (metrics.dose_response, group = cell: within-cell slope, cluster bootstrap over cells, Spearman on cell-centred values,
      mean within-cell Spearman, fraction of cells perfectly monotone)
  span      rungs k050 k075 c100 k133 k167, x = ln realised k (Praat; span_pyin: pyin), c100 x = 0
  dyn       g060 c100 g150, x = ln realised g on the frozen IP frames (realised_g_ip)
  combined  less c100 more, x = -1 / 0 / +1
  register  c100 r+2 r+4, x = realised register (st, Praat)
  rate      c100 rate090 (time-normalised frames), x = ln(1/0.9)
  y = amp (ln), vel_rms (ln), event_rate (raw). brow_mean = geometric mean of the brow_inner and brow_outer amplitudes
  (its ln ratio = the plan's primary endpoint: mean ln ratio over the two brows).
  swing_frac = median over cells of (max - min over the family's rungs) / c100 value (d1_audit/protocol.py t4_stats rule).
  r_within = Pearson r of cell-centred x and cell-centred ln y (reported beside swing_frac).
RATIO  per cell ln(y_rung / y_c100); mean over cells, cell bootstrap 95 % CI; lvl+6 = gain diagnostic; src / tgt100 = null floor.
Per-language rows are DESCRIPTIVE (3-5 cells).
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import importlib.util
import io
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
LANE = Path("${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/audit2")
sys.path.insert(0, str(LANE))
import common2 as C  # noqa: E402
import g2_predict as G2P  # noqa: E402  (lane copy; coefficients/draws read from human2)

_spec = importlib.util.spec_from_file_location("vd_metrics", "${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/audit/metrics.py")
MX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MX)

HUMAN = Path("${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/human")
HUMAN2 = Path(C.HUMAN2)
SRC_ORDER = ["emoface", "emotalk", "emote", "medtalk", "f0_follower", "energy_follower"]
SRC_LABEL = {"emoface": "EmoFace", "emotalk": "EmoTalk", "emote": "EMOTE", "medtalk": "MEDTalk",
             "f0_follower": "F0-height follower", "energy_follower": "Energy follower", "medtalk_st": "MEDTalk s_t"}
LOGM = {"m.amp", "m.vel_rms", "m.mean_in_span"}
FAMS = {
    "span": {"rungs": ["k050", "k075", "c100", "k133", "k167"], "x": "x.k_praat", "tf": "ln"},
    "span_pyin": {"rungs": ["k050", "k075", "c100", "k133", "k167"], "x": "x.k_pyin", "tf": "ln"},
    "dyn": {"rungs": ["g060", "c100", "g150"], "x": "x.g_ip", "tf": "ln"},
    "combined": {"rungs": ["less", "c100", "more"], "x": "ordinal", "tf": ""},
    "register": {"rungs": ["c100", "r+2", "r+4"], "x": "x.reg_praat", "tf": "lin"},
    "rate": {"rungs": ["c100", "rate090"], "x": "rate", "tf": ""},
}
RATIO_RUNGS = ["src", "tgt100", "lvl+6", "effort0", "effort", "rate090", "r+2", "r+4", "k050", "k167", "g060", "g150", "less", "more"]
G2_RUNGS = ["effort", "effort0", "rate090", "r+2", "r+4", "lvl+6", "g150", "k167", "more"]
N_BOOT = 2000
RNG_SEED = 20260914


def boot_mean(vals, n_boot=N_BOOT, seed=RNG_SEED):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), np.nan, np.nan, len(v)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), len(v)


def boot_cluster_mean(vals, groups, n_boot=N_BOOT, seed=RNG_SEED):
    """mean of clip values; CI by resampling clusters (actors)."""
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    ok = np.isfinite(vals)
    vals, groups = vals[ok], groups[ok]
    ug = np.unique(groups)
    idx = [np.flatnonzero(groups == g) for g in ug]
    rng = np.random.default_rng(seed)
    sums = np.array([vals[i].sum() for i in idx])
    cnts = np.array([len(i) for i in idx])
    pick = rng.integers(0, len(ug), (n_boot, len(ug)))
    bs = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(vals.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), int(len(vals)), int(len(ug)), bs


# ------------------------------------------------------------------ load
R = pd.read_csv(LANE / "results.csv", low_memory=False)
ITEMS = C.items()
LANG = {i["cell"]: i["lang"] for i in ITEMS}
CELLS = list(dict.fromkeys(i["cell"] for i in ITEMS))
VERD = {i["id"]: i["verdict"] for i in ITEMS}
X = R.drop_duplicates("id").set_index("id")
_CACHE = {}


def get(src, ch, arm, metric):
    k = (src, ch, arm, metric)
    if k in _CACHE:
        return _CACHE[k]
    if ch == "brow_mean":
        a, b = get(src, "brow_inner", arm, metric), get(src, "brow_outer", arm, metric)
        out = np.exp(0.5 * (np.log(a.where(a > 0)) + np.log(b.where(b > 0)))) if metric in LOGM else 0.5 * (a + b)
    else:
        s = R[(R.source == src) & (R.channel == ch) & (R.arm == arm)]
        out = s.pivot(index="cell", columns="rung", values=metric) if len(s) else pd.DataFrame()
    _CACHE[k] = out
    return out


def x_of(fam, cell, rung):
    if rung == "c100":
        return 0.0
    f = FAMS[fam]
    if f["x"] == "ordinal":
        return {"less": -1.0, "more": 1.0}[rung]
    if f["x"] == "rate":
        return math.log(1 / 0.9)
    v = X.loc[cell + "__" + rung, f["x"]]
    if not np.isfinite(v):
        return np.nan
    return math.log(v) if f["tf"] == "ln" and v > 0 else (float(v) if f["tf"] == "lin" else np.nan)


def dose_row(src, ch, arm, fam, metric, cells, excl_fail, n_boot):
    Y = get(src, ch, arm, metric)
    xs, ys, gs, swings = [], [], [], []
    if Y.empty:
        return None
    for cell in cells:
        if cell not in Y.index:
            continue
        vals = []
        for rung in FAMS[fam]["rungs"]:
            if excl_fail and VERD.get(cell + "__" + rung) == "fail":
                continue
            if rung not in Y.columns:
                continue
            y, x = Y.loc[cell, rung], x_of(fam, cell, rung)
            if not (np.isfinite(y) and np.isfinite(x)):
                continue
            xs.append(x)
            ys.append(y)
            gs.append(cell)
            vals.append(y)
        y0 = Y.loc[cell, "c100"] if "c100" in Y.columns else np.nan
        if len(vals) >= 2 and np.isfinite(y0) and y0 > 0:
            swings.append((max(vals) - min(vals)) / y0)
    if len(xs) < 3:
        return None
    log_y = metric in LOGM
    dr = MX.dose_response(xs, ys, gs, log_y=log_y, n_boot=n_boot, seed=0)
    xs, ys, gs = np.asarray(xs), np.asarray(ys, float), np.asarray(gs)
    yy = np.log(np.where(ys > 0, ys, np.nan)) if dr.get("log_y_used") else ys
    xc, yc = [], []
    for g in np.unique(gs):
        m = (gs == g) & np.isfinite(yy)
        if m.sum() >= 2:
            xc += list(xs[m] - xs[m].mean())
            yc += list(yy[m] - yy[m].mean())
    r_within = float(np.corrcoef(xc, yc)[0, 1]) if len(xc) > 3 and np.std(xc) > 0 and np.std(yc) > 0 else np.nan
    return dict(source=src, channel=ch, arm=arm, family=fam, metric=metric, excl_fail=excl_fail, slope=dr["slope"],
                ci_lo=dr["ci_lo"], ci_hi=dr["ci_hi"], p_boot_le0=dr["p_boot_le0"], rho=dr["rho"], rho_items_mean=dr["rho_items_mean"],
                frac_monotone=dr["frac_monotone"], r_within=r_within, swing_frac=float(np.median(swings)) if swings else np.nan,
                n_items=dr["n_cells"], n_cells=dr["n_groups"], log_y_used=dr["log_y_used"])


def ratio_vals(src, ch, arm, rung, metric, cells):
    Y = get(src, ch, arm, metric)
    out = {}
    if Y.empty or rung not in Y.columns:
        return out
    for cell in cells:
        if cell not in Y.index:
            continue
        y0, y1 = Y.loc[cell, "c100"], Y.loc[cell, rung]
        if metric in LOGM:
            if np.isfinite(y0) and np.isfinite(y1) and y0 > 0 and y1 > 0:
                out[cell] = math.log(y1 / y0)
        elif np.isfinite(y0) and np.isfinite(y1):
            out[cell] = y1 - y0
    return out


def main():
    subsets = [("all", CELLS, False, N_BOOT), ("excl_fail", CELLS, True, N_BOOT)]
    subsets += [("lang:%s (descriptive)" % lg, [c for c in CELLS if LANG[c] == lg], False, 1000) for lg in ("cn", "en", "jp")]
    sources = SRC_ORDER + ["medtalk_st"]
    # ---------------- dose tables
    rows = []
    for src in sources:
        chans = ["intensity"] if src == "medtalk_st" else (["brow_mean", "brow_inner", "brow_outer", "lid_upper", "jaw", "brow_inner_net", "brow_outer_net"]
                                                          if src in ("emoface", "medtalk", "emotalk") else
                                                          (["brow_mean", "brow_inner", "brow_outer", "lid_upper", "jaw"] if src == "emote" else ["brow_mean"]))
        arms = ["clean"] if src == "medtalk_st" else ["clean", "noise"]
        for ch in chans:
            for arm in arms:
                for fam in FAMS:
                    metrics = ["m.mean_in_span", "m.amp"] if src == "medtalk_st" else ["m.amp", "m.vel_rms", "m.event_rate"]
                    for metric in metrics:
                        for sname, cells, excl, nb in subsets:
                            if metric != metrics[0] and not sname.startswith(("all", "excl")):
                                continue
                            if metric in ("m.vel_rms", "m.event_rate") and ch not in ("brow_mean", "lid_upper", "jaw"):
                                continue
                            if sname == "excl_fail" and fam not in ("span", "span_pyin", "combined"):
                                continue
                            r = dose_row(src, ch, arm, fam, metric, cells, excl, nb)
                            if r:
                                r["subset"] = sname
                                rows.append(r)
    DT = pd.DataFrame(rows)
    DT.to_csv(LANE / "dose_tables.csv", index=False, float_format="%.5g")
    print("dose_tables", DT.shape, flush=True)

    # ---------------- ratio tables + null floor
    rrows, frows = [], []
    for src in sources:
        chans = ["intensity"] if src == "medtalk_st" else (["brow_mean", "brow_inner", "brow_outer", "lid_upper", "jaw"] if src in REALS else ["brow_mean"])
        arms = ["clean"] if src == "medtalk_st" else ["clean", "noise"]
        for ch in chans:
            for arm in arms:
                metric = "m.mean_in_span" if src == "medtalk_st" else "m.amp"
                for sname, cells, _e, nb in [s for s in subsets if s[0] != "excl_fail"]:
                    for rung in RATIO_RUNGS:
                        v = ratio_vals(src, ch, arm, rung, metric, cells)
                        est, lo, hi, n = boot_mean(list(v.values()), nb)
                        rrows.append(dict(source=src, channel=ch, arm=arm, rung=rung, metric=metric, subset=sname, mean_ln_ratio=est,
                                          ci_lo=lo, ci_hi=hi, median=float(np.median(list(v.values()))) if v else np.nan,
                                          n_pos=int(sum(1 for x in v.values() if x > 0)), n_cells=n,
                                          per_cell=json.dumps({k: round(x, 4) for k, x in v.items()})))
                nulls = [abs(x) for rung in ("src", "tgt100") for x in ratio_vals(src, ch, arm, rung, metric, CELLS).values()]
                frows.append(dict(source=src, channel=ch, arm=arm, n=len(nulls), floor_median_abs=float(np.median(nulls)) if nulls else np.nan,
                                  floor_p90_abs=float(np.percentile(nulls, 90)) if nulls else np.nan,
                                  floor_max_abs=float(np.max(nulls)) if nulls else np.nan))
    RT, NF = pd.DataFrame(rrows), pd.DataFrame(frows)
    RT.to_csv(LANE / "ratio_tables.csv", index=False, float_format="%.5g")
    NF.to_csv(LANE / "null_floor.csv", index=False, float_format="%.5g")

    # ---------------- G2: human-predicted interval per rung
    gm = {r["id"]: r for r in json.loads((LANE / "g2_measures.json").read_text())}
    J = dict(json_path=HUMAN2 / "g2_coefficients.json", draws_path=HUMAN2 / "g2_boot_draws.npz")

    def deltas(cell, rung, praat_register=False):
        a, b = gm[cell + "__" + rung], gm[cell + "__c100"]
        d = {"d_level_db": a["level_db"] - b["level_db"], "d_register_st": 12 * math.log2(a["f0_med_hz"] / b["f0_med_hz"]),
             "d_env_sd_db": a["env_sd_db"] - b["env_sd_db"], "d_ln_dur": math.log(a["speech_dur_s"] / b["speech_dur_s"])}
        if praat_register:
            d["d_register_st"] = float(X.loc[cell + "__" + rung, "x.reg_praat"])
        return d

    nominal = {"effort": {"d_level_db": 7.0, "d_register_st": 4.7, "d_env_sd_db": 0.0, "d_ln_dur": math.log(1 / 0.9)},
               "effort0": {"d_register_st": 4.7, "d_ln_dur": math.log(1 / 0.9)}, "rate090": {"d_ln_dur": math.log(1 / 0.9)},
               "r+2": {"d_register_st": 2.0}, "r+4": {"d_register_st": 4.0}, "lvl+6": {"d_level_db": 6.0}}
    prow, per_cell_pred = [], {}
    for rung in G2_RUNGS:
        for mk, win in (("E4_common", "W2_full"), ("E4_common", "W2_pm"), ("E7_common", "W2_full")):
            for outcome in ("brow_mean", "lid"):
                g = G2P.G2(mk, win, outcome, **J)
                variants = {"g2_realised": [deltas(c, rung) for c in CELLS], "praat_register": [deltas(c, rung, True) for c in CELLS]}
                if rung in nominal:
                    variants["nominal"] = [nominal[rung]]
                for vname, dl in variants.items():
                    md = {k: float(np.mean([d.get(k, 0.0) for d in dl])) for k in ("d_level_db", "d_register_st", "d_env_sd_db", "d_ln_dur")}
                    for wi in (False, True):
                        p = g.predict(with_intercept=wi, **md)
                        prow.append(dict(rung=rung, model=mk, window=win, outcome=outcome, variant=vname, with_intercept=wi,
                                         estimate=p["estimate"], ci_lo=p["ci95_percentile"][0], ci_hi=p["ci95_percentile"][1], **md))
                    if mk == "E4_common" and win == "W2_full" and vname == "g2_realised":
                        per_cell_pred[(rung, outcome)] = {c: g.predict(**deltas(c, rung))["estimate"] for c in CELLS}
    PR = pd.DataFrame(prow)
    PR.to_csv(LANE / "g2_predictions.csv", index=False, float_format="%.5g")

    crow = []
    for src in SRC_ORDER:
        for arm in ("clean", "noise"):
            for ch, outcome in (("brow_mean", "brow_mean"), ("lid_upper", "lid")):
                if src not in REALS and ch != "brow_mean":
                    continue
                fl = NF[(NF.source == src) & (NF.channel == ch) & (NF.arm == arm)]
                floor = float(fl.floor_median_abs.iloc[0]) if len(fl) else np.nan
                for rung in G2_RUNGS:
                    m = RT[(RT.source == src) & (RT.channel == ch) & (RT.arm == arm) & (RT.rung == rung) & (RT.subset == "all")]
                    if not len(m):
                        continue
                    m = m.iloc[0]
                    p = PR[(PR.rung == rung) & (PR.model == "E4_common") & (PR.window == "W2_full") & (PR.outcome == outcome)
                           & (PR.variant == "g2_realised") & (~PR.with_intercept)].iloc[0]
                    pe = p.estimate
                    crow.append(dict(source=src, arm=arm, channel=ch, rung=rung, model_ln_ratio=m.mean_ln_ratio, model_ci_lo=m.ci_lo,
                                     model_ci_hi=m.ci_hi, n_pos=m.n_pos, n_cells=m.n_cells, human_pred=pe, human_ci_lo=p.ci_lo,
                                     human_ci_hi=p.ci_hi, ratio_model_over_pred=(m.mean_ln_ratio / pe) if abs(pe) > 1e-9 else np.nan,
                                     in_0p5_2x=bool(pe > 0 and 0.5 * pe <= m.mean_ln_ratio <= 2 * pe),
                                     ci_overlaps_pred_ci=bool(m.ci_lo <= p.ci_hi and p.ci_lo <= m.ci_hi),
                                     null_floor_median_abs=floor, above_floor=bool(m.mean_ln_ratio > floor and m.ci_lo > 0)))
    CP = pd.DataFrame(crow)
    CP.to_csv(LANE / "g2_compare.csv", index=False, float_format="%.5g")

    # ---------------- E3: follow index / events -- humans
    FR = pd.read_csv(HUMAN2 / "follow_records.csv", low_memory=False)
    hb = FR[(FR.path == "audit") & (FR.thr == "ref") & (FR.cond == "base") & (FR.n_pk > 0)].copy()
    hb["recall"] = hb.hits / hb.n_pk
    hb["exc_circ"] = (hb.hits - hb.chance_circ) / hb.n_pk
    hb["exc_cross"] = (hb.hits - hb.chance_cross) / hb.n_pk
    hb["events_per_f0"] = hb.n_ev / hb.n_pk
    hb["event_rate"] = hb.n_ev / hb.span_dur
    E4 = ["hap", "sad", "ang", "sur"]
    erows, boots = [], {}
    for win, wname in (("plan", "plan"), ("aud", "post")):
        for ch in ("brow_inner", "brow_outer"):
            for sub, mask in (("human_all", np.ones(len(hb), bool)), ("human_E4", hb.emotion.isin(E4).values),
                              ("human_E4_normal", (hb.emotion.isin(E4) & (hb.intensity == 1)).values),
                              ("human_E4_strong", (hb.emotion.isin(E4) & (hb.intensity == 2)).values)):
                s = hb[mask & (hb.win == win).values & (hb.ch == ch).values]
                r = dict(source=sub, arm="human", channel=ch, window=wname, unit="clip (actor bootstrap)")
                for met in ("recall", "exc_circ", "exc_cross", "events_per_f0", "event_rate"):
                    est, lo, hi, n, ng, bs = boot_cluster_mean(s[met], s.actor)
                    r.update({met: est, met + "_lo": lo, met + "_hi": hi, met + "_p05": float(np.nanpercentile(s[met], 5)),
                              met + "_p95": float(np.nanpercentile(s[met], 95))})
                    boots[(sub, ch, wname, met)] = (s[met].values, s.actor.values)
                r["n"], r["n_clusters"] = n, ng
                erows.append(r)
    ham = pd.read_csv(HUMAN / "human_audit_metrics.csv", low_memory=False)
    hum_coup = {}
    for ch in ("brow_inner", "brow_outer"):
        c1, c2 = ch + ".coupling.r_phrase", ch + ".coupling.r_phrase_rev"
        if c1 in ham and c2 in ham:
            d = (ham[c1] - ham[c2]).values
            hum_coup[ch] = boot_cluster_mean(d, ham.actor.values)[:4]
    # models + followers (c100 primary; all non-rate rungs pooled secondary)
    for src in SRC_ORDER:
        for arm in ("clean", "noise"):
            for ch in ("brow_inner", "brow_outer"):
                for scope, rungs in (("c100", ["c100"]), ("all_nonrate_rungs", [r for r in R.rung.unique() if r not in C.RATE_RUNGS])):
                    s = R[(R.source == src) & (R.arm == arm) & (R.channel == ch) & (R.rung.isin(rungs)) & (R["fplan.n_f0_peaks"] > 0)]
                    for wname, pre in (("plan", "fplan"), ("post", "fpost")):
                        r = dict(source=src, arm=arm, channel=ch, window=wname, unit="%s item (cell bootstrap)" % scope, scope=scope)
                        cols = {"recall": pre + ".recall", "exc_circ": pre + ".recall_excess", "events_per_f0": pre + ".events_per_f0",
                                "event_rate": "m.event_rate", "prec_exc": pre + ".precision_excess"}
                        if wname == "plan":
                            cols["exc_cross"] = "fplan.recall_excess_cross"
                        for met, col in cols.items():
                            est, lo, hi, n, ng, bs = boot_cluster_mean(s[col], s.cell)
                            r.update({met: est, met + "_lo": lo, met + "_hi": hi})
                            if scope == "c100":
                                boots[(src + "|" + arm, ch, wname, met)] = (s[col].values, s.cell.values)
                        if arm == "clean":
                            est, lo, hi, *_ = boot_cluster_mean((s["coup.r_phrase"] - s["coup.r_phrase_rev"]).values, s.cell.values)
                            r.update(coup_minus_rev=est, coup_minus_rev_lo=lo, coup_minus_rev_hi=hi)
                        r["n"] = len(s)
                        r["n_clusters"] = s.cell.nunique()
                        erows.append(r)
    EF = pd.DataFrame(erows)
    EF.to_csv(LANE / "e3_follow.csv", index=False, float_format="%.5g")

    # separation: source (c100) minus human_all, independent bootstraps; AUC of cells vs clips
    srows = []
    rng = np.random.default_rng(RNG_SEED)
    for src in SRC_ORDER:
        for arm in ("clean", "noise"):
            for ch in ("brow_inner", "brow_outer"):
                for wname in ("plan",):
                    for met in ("exc_circ", "exc_cross", "events_per_f0", "event_rate"):
                        ka, kh = (src + "|" + arm, ch, wname, met), ("human_all", ch, wname, met)
                        if ka not in boots or kh not in boots:
                            continue
                        va, ga = boots[ka]
                        vh, gh = boots[kh]
                        ma = boot_cluster_mean(va, ga, seed=1)
                        mh = boot_cluster_mean(vh, gh, seed=2)
                        diff = ma[-1] - mh[-1]
                        a_ok, h_ok = va[np.isfinite(va)], vh[np.isfinite(vh)]
                        auc = float(((a_ok[:, None] > h_ok[None, :]).mean() + 0.5 * (a_ok[:, None] == h_ok[None, :]).mean())) if len(a_ok) and len(h_ok) else np.nan
                        srows.append(dict(source=src, arm=arm, channel=ch, window=wname, metric=met, source_mean=ma[0], human_mean=mh[0],
                                          diff_est=ma[0] - mh[0], diff_lo=float(np.quantile(diff, 0.025)), diff_hi=float(np.quantile(diff, 0.975)),
                                          auc_source_gt_human=auc, n_source=ma[3], n_human=mh[3]))
    SP = pd.DataFrame(srows)
    SP.to_csv(LANE / "e3_separation.csv", index=False, float_format="%.5g")

    # injection: humans (human2 follow_records) and sources (injection.csv)
    irows = []
    hp = FR[(FR.path == "audit") & (FR.thr == "ref") & (FR.win == "plan")]
    for ch in ("brow_inner", "brow_outer"):
        base = hp[(hp.cond == "base") & (hp.ch == ch)]
        hb_sum = base.hits.sum()
        for cond in ("lock_q1_s0.5", "lock_q1_s1", "lock_q1_s2", "rand_q1_s1"):
            s = hp[(hp.cond == cond) & (hp.ch == ch)]
            den = s.ideal_hits.sum() - hb_sum
            irows.append(dict(source="human_all", arm="human", channel=ch, cond=cond.replace("_q1", ""), n=len(s),
                              recall_base=hb_sum / base.n_pk.sum(), recall_cond=s.hits.sum() / s.n_pk.sum(),
                              recovery=(s.hits.sum() - hb_sum) / den if den > 0 else np.nan,
                              excess_cond=float(((s.hits - s.chance_circ) / s.n_pk.where(s.n_pk > 0)).mean())))
    IJ = pd.read_csv(LANE / "injection.csv")
    for (src, ch, arm), g in IJ.groupby(["source", "channel", "arm"]):
        base = g[g.cond == "base"]
        for cond in ("lock_s0.5", "lock_s1", "lock_s2", "rand_s1"):
            s = g[g.cond == cond]
            den = s.ideal.sum() - s.hits_base.sum()
            irows.append(dict(source=src, arm=arm, channel=ch, cond=cond, n=len(s), recall_base=base.hits.sum() / max(base.n_pk.sum(), 1),
                              recall_cond=s.hits.sum() / max(s.n_pk.sum(), 1),
                              recovery=(s.hits.sum() - s.hits_base.sum()) / den if den > 0 else np.nan,
                              excess_cond=float((s.hits / s.n_pk.where(s.n_pk > 0) - s.recall_chance).mean()),
                              excess_base=float((base.hits / base.n_pk.where(base.n_pk > 0) - base.recall_chance).mean())))
    IR = pd.DataFrame(irows)
    IR.to_csv(LANE / "e3_injection.csv", index=False, float_format="%.5g")

    key = {"human_coupling_minus_rev": hum_coup, "n_items": int(R.id.nunique()), "n_rows": int(len(R))}
    (LANE / "audit2_key.json").write_text(json.dumps(key, indent=1, default=float))
    write_md(DT, RT, NF, PR, CP, EF, SP, IR, hum_coup)
    figures(DT, RT, NF, PR, CP, EF, IR, per_cell_pred)
    print("DONE analyse2", flush=True)


REALS = ["emoface", "emotalk", "emote", "medtalk"]


def f3(v, nd=3):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else ("%+.*f" % (nd, v))


def ci(e, lo, hi, nd=3):
    return "%s [%s, %s]" % (f3(e, nd), f3(lo, nd), f3(hi, nd))


def write_md(DT, RT, NF, PR, CP, EF, SP, IR, hum_coup):
    L = ["# audit2 tables (generated by analyse2.py; every number [measured])", ""]
    L += ["## T1. Dose slopes, amplitude, all 12 cells, clean arm", "",
          "slope of ln amp per unit x (span/dyn: per ln k / ln g; register: per st; combined: per rung; rate: per ln(1/rate)); "
          "cell bootstrap 95 % CI; rho = Spearman on cell-centred values; mono = fraction of cells perfectly monotone; "
          "r_w = within-cell Pearson r; swing = median (max-min)/c100.", "",
          "| source | channel | family | slope [95% CI] | rho | mono | r_w | swing | n cells |", "|---|---|---|---|---|---|---|---|---|"]
    d = DT[(DT.metric == "m.amp") & (DT.arm == "clean") & (DT.subset == "all")]
    for src in SRC_ORDER + ["medtalk_st"]:
        for ch in ("brow_mean", "brow_inner", "brow_outer", "lid_upper", "jaw", "intensity"):
            for fam in FAMS:
                s = d[(d.source == src) & (d.channel == ch) & (d.family == fam)]
                if not len(s):
                    continue
                s = s.iloc[0]
                L.append("| %s | %s | %s | %s | %s | %s | %s | %.3f | %d |" % (SRC_LABEL[src], ch, fam, ci(s.slope, s.ci_lo, s.ci_hi), f3(s.rho, 2),
                                                                      "" if not np.isfinite(s.frac_monotone) else "%.2f" % s.frac_monotone,
                                                                      f3(s.r_within, 2), s.swing_frac, s.n_cells))
    L += ["", "## T2. Span / combined slopes without the 3 ladder2 FAIL items (en_rav02_take1_sad k133, k167, more), clean", "",
          "| source | channel | family | all cells | excl. FAIL |", "|---|---|---|---|---|"]
    for src in SRC_ORDER:
        for ch in ("brow_mean", "lid_upper", "jaw"):
            for fam in ("span", "span_pyin", "combined"):
                a = DT[(DT.source == src) & (DT.channel == ch) & (DT.family == fam) & (DT.metric == "m.amp") & (DT.arm == "clean")]
                sa, se = a[a.subset == "all"], a[a.subset == "excl_fail"]
                if len(sa) and len(se):
                    sa, se = sa.iloc[0], se.iloc[0]
                    L.append("| %s | %s | %s | %s | %s |" % (SRC_LABEL[src], ch, fam, ci(sa.slope, sa.ci_lo, sa.ci_hi), ci(se.slope, se.ci_lo, se.ci_hi)))
    L += ["", "## T3. ln amplitude ratio vs c100 (mean over 12 cells [cell bootstrap CI], n cells > 0), clean arm", "",
          "| source | channel | " + " | ".join(RATIO_RUNGS) + " |", "|---|---|" + "---|" * len(RATIO_RUNGS)]
    for src in SRC_ORDER + ["medtalk_st"]:
        for ch in ("brow_mean", "lid_upper", "jaw", "intensity"):
            s = RT[(RT.source == src) & (RT.channel == ch) & (RT.arm == "clean") & (RT.subset == "all")]
            if not len(s):
                continue
            cells = []
            for rung in RATIO_RUNGS:
                q = s[s.rung == rung]
                cells.append("" if not len(q) else "%s [%s, %s] %d/%d" % (f3(q.iloc[0].mean_ln_ratio), f3(q.iloc[0].ci_lo), f3(q.iloc[0].ci_hi),
                                                                         q.iloc[0].n_pos, q.iloc[0].n_cells))
            L.append("| %s | %s | %s |" % (SRC_LABEL[src], ch, " | ".join(cells)))
    L += ["", "## T4. Null floor: |ln amp ratio| of src and tgt100 vs c100 (24 contrasts)", "",
          "| source | channel | arm | median | p90 | max |", "|---|---|---|---|---|---|"]
    for _, s in NF.iterrows():
        L.append("| %s | %s | %s | %.3f | %.3f | %.3f |" % (SRC_LABEL.get(s.source, s.source), s.channel, s.arm, s.floor_median_abs, s.floor_p90_abs, s.floor_max_abs))
    L += ["", "## T5. Human-predicted ln ratio (G2), mean realised delta over the 12 cells", "",
          "| rung | model | window | outcome | variant | form | estimate [95% CI] | d_level | d_register | d_env | d_ln_dur |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, p in PR.iterrows():
        L.append("| %s | %s | %s | %s | %s | %s | %s | %.2f | %.2f | %.2f | %.3f |" % (p.rung, p.model, p.window, p.outcome, p.variant,
                                                                               "with intercept" if p.with_intercept else "slope-only",
                                                                               ci(p.estimate, p.ci_lo, p.ci_hi), p.d_level_db, p.d_register_st,
                                                                               p.d_env_sd_db, p.d_ln_dur))
    L += ["", "## T6. Model ln ratio vs human-predicted (E4_common, W2_full, slope-only, g2-realised deltas)", "",
          "| source | arm | channel | rung | model [95% CI] | n>0 | human pred [95% CI] | model/pred | in [0.5,2]x | CIs overlap | null floor | above floor |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, c in CP.iterrows():
        L.append("| %s | %s | %s | %s | %s | %d/%d | %s | %s | %s | %s | %.3f | %s |" % (SRC_LABEL[c.source], c.arm, c.channel, c.rung,
                                                                                  ci(c.model_ln_ratio, c.model_ci_lo, c.model_ci_hi), c.n_pos, c.n_cells,
                                                                                  ci(c.human_pred, c.human_ci_lo, c.human_ci_hi), f3(c.ratio_model_over_pred, 2),
                                                                                  c.in_0p5_2x, c.ci_overlaps_pred_ci, c.null_floor_median_abs, c.above_floor))
    L += ["", "## T7. E3 follow index and event statistics (plan G3 window -0.25..+0.30 s unless 'post')", "",
          "| source | arm | channel | window | unit | recall | excess vs circular [CI] | excess vs cross-utt [CI] | events/F0 peak | events/s | n (clusters) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, e in EF.iterrows():
        L.append("| %s | %s | %s | %s | %s | %.3f | %s | %s | %.2f | %.2f | %d (%d) |" % (
            SRC_LABEL.get(e.source, e.source), e.arm, e.channel, e.window, e.unit, e.recall, ci(e.exc_circ, e.exc_circ_lo, e.exc_circ_hi),
            ci(e.get("exc_cross", np.nan), e.get("exc_cross_lo", np.nan), e.get("exc_cross_hi", np.nan)), e.events_per_f0, e.event_rate, e.n, e.n_clusters))
    L += ["", "Human coupling r_phrase - r_phrase_rev (human_audit_metrics.csv, actor CI): " +
          "; ".join("%s %s" % (ch, ci(*v[:3])) for ch, v in hum_coup.items()), ""]
    L += ["## T8. E3 separation: source c100 items minus human clips (plan window)", "",
          "| source | arm | channel | metric | source mean | human mean | diff [95% CI] | AUC(source > human) |", "|---|---|---|---|---|---|---|---|"]
    for _, s in SP.iterrows():
        L.append("| %s | %s | %s | %s | %.3f | %.3f | %s | %.2f |" % (SRC_LABEL[s.source], s.arm, s.channel, s.metric, s.source_mean, s.human_mean,
                                                                 ci(s.diff_est, s.diff_lo, s.diff_hi), s.auc_source_gt_human))
    L += ["", "## T9. Injection positive control (plan window): recovery = (hits - base hits) / (ideal - base hits), pooled", "",
          "| source | arm | channel | condition | recall base | recall injected | recovery | excess (injected) |", "|---|---|---|---|---|---|---|---|"]
    for _, s in IR.iterrows():
        L.append("| %s | %s | %s | %s | %.3f | %.3f | %s | %s |" % (SRC_LABEL.get(s.source, s.source), s.arm, s.channel, s.cond, s.recall_base,
                                                             s.recall_cond, f3(s.recovery, 2), f3(s.excess_cond, 3)))
    L += ["", "## T10. Per-language dose slopes and effort ratio (DESCRIPTIVE; 5 cn / 3 en / 4 jp cells), brow_mean and jaw, clean", "",
          "| source | channel | subset | span slope | dyn slope | register slope | effort ln ratio | lvl+6 ln ratio |", "|---|---|---|---|---|---|---|---|"]
    for src in SRC_ORDER:
        for ch in ("brow_mean", "jaw"):
            for sub in ("lang:cn (descriptive)", "lang:en (descriptive)", "lang:jp (descriptive)"):
                vals = []
                for fam in ("span", "dyn", "register"):
                    q = DT[(DT.source == src) & (DT.channel == ch) & (DT.family == fam) & (DT.metric == "m.amp") & (DT.arm == "clean") & (DT.subset == sub)]
                    vals.append(ci(q.iloc[0].slope, q.iloc[0].ci_lo, q.iloc[0].ci_hi, 2) if len(q) else "")
                for rung in ("effort", "lvl+6"):
                    q = RT[(RT.source == src) & (RT.channel == ch) & (RT.rung == rung) & (RT.arm == "clean") & (RT.subset == sub)]
                    vals.append(ci(q.iloc[0].mean_ln_ratio, q.iloc[0].ci_lo, q.iloc[0].ci_hi, 2) if len(q) else "")
                if any(vals):
                    L.append("| %s | %s | %s | %s |" % (SRC_LABEL[src], ch, sub, " | ".join(vals)))
    (LANE / "audit2_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ figures
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#dcdbd5", "#fcfcfb"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"


def _style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figures(DT, RT, NF, PR, CP, EF, IR, per_cell_pred):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    srcs = SRC_ORDER
    ypos = {s: i for i, s in enumerate(srcs[::-1])}
    # Fig 1: dose forest
    panels = [("span", "slope", "span: d ln amp / d ln k"), ("dyn", "slope", "dynamics: d ln amp / d ln g"),
              ("register", "slope", "register: d ln amp / st"), ("rate090", "ratio", "rate 0.90: ln ratio"),
              ("effort", "ratio", "effort composite: ln ratio"), ("lvl+6", "ratio", "+6 dB (gain diagnostic): ln ratio")]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.8), sharey=True, facecolor=SURF)
    for ax, (key, kind, title) in zip(axes, panels):
        _style(ax)
        ax.axvline(0, color=INK2, lw=0.8)
        for ch, col, off, mk in (("brow_mean", S1, 0.12, "o"), ("jaw", S2, -0.12, "s")):
            for s in srcs:
                if ch == "jaw" and s not in REALS:
                    continue
                if kind == "slope":
                    q = DT[(DT.source == s) & (DT.channel == ch) & (DT.family == key) & (DT.metric == "m.amp") & (DT.arm == "clean") & (DT.subset == "all")]
                    if not len(q):
                        continue
                    e, lo, hi = q.iloc[0].slope, q.iloc[0].ci_lo, q.iloc[0].ci_hi
                else:
                    q = RT[(RT.source == s) & (RT.channel == ch) & (RT.rung == key) & (RT.arm == "clean") & (RT.subset == "all")]
                    if not len(q):
                        continue
                    e, lo, hi = q.iloc[0].mean_ln_ratio, q.iloc[0].ci_lo, q.iloc[0].ci_hi
                y = ypos[s] + off
                ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
                ax.plot([e], [y], marker=mk, ms=6, color=col, mec=SURF, mew=1.5, ls="none")
        if key in ("effort", "rate090", "lvl+6"):
            p = PR[(PR.rung == key) & (PR.model == "E4_common") & (PR.window == "W2_full") & (PR.outcome == "brow_mean") & (PR.variant == "g2_realised") & (~PR.with_intercept)]
            if len(p):
                ax.axvspan(p.iloc[0].ci_lo, p.iloc[0].ci_hi, color=S1, alpha=0.10, lw=0)
                ax.axvline(p.iloc[0].estimate, color=S1, lw=1, ls="--")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    axes[0].set_yticks([ypos[s] for s in srcs])
    axes[0].set_yticklabels([SRC_LABEL[s] for s in srcs], color=INK, fontsize=9)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=S1, marker="o", lw=2), Line2D([], [], color=S2, marker="s", lw=2),
                matplotlib.patches.Patch(color=S1, alpha=0.15)],
               ["brow (mean of inner/outer), 95% cell-bootstrap CI", "jaw (positive control)", "human-predicted brow ln ratio (G2, 95% CI; dashed = estimate)"],
               loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(LANE / "fig_dose_forest.png", dpi=160, facecolor=SURF)
    plt.close(fig)

    # Fig 2: effort composite per cell vs human prediction
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURF)
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.grid(axis="x", visible=False)
    p = PR[(PR.rung == "effort") & (PR.model == "E4_common") & (PR.window == "W2_full") & (PR.outcome == "brow_mean") & (PR.variant == "g2_realised") & (~PR.with_intercept)].iloc[0]
    ax.axhspan(p.ci_lo, p.ci_hi, color=S1, alpha=0.10, lw=0)
    ax.axhline(p.estimate, color=S1, lw=1, ls="--")
    ax.axhline(0.5 * p.estimate, color=INK2, lw=0.7, ls=":")
    ax.axhline(2 * p.estimate, color=INK2, lw=0.7, ls=":")
    ax.axhline(0, color=INK2, lw=0.8)
    cols = {"cn": S1, "en": S2, "jp": S3}
    for i, s in enumerate(srcs):
        q = RT[(RT.source == s) & (RT.channel == "brow_mean") & (RT.rung == "effort") & (RT.arm == "clean") & (RT.subset == "all")]
        if not len(q):
            continue
        q = q.iloc[0]
        pc = json.loads(q.per_cell)
        rng = np.random.default_rng(i)
        for cell, v in pc.items():
            ax.plot(i + rng.uniform(-0.18, 0.18), v, "o", ms=5, color=cols[LANG[cell]], mec=SURF, mew=1.0, alpha=0.9)
        ax.plot([i + 0.3, i + 0.3], [q.ci_lo, q.ci_hi], color=INK, lw=2, solid_capstyle="round")
        ax.plot([i + 0.3], [q.mean_ln_ratio], "D", ms=6, color=INK, mec=SURF, mew=1.2)
    ax.set_xticks(range(len(srcs)))
    ax.set_xticklabels([SRC_LABEL[s] for s in srcs], fontsize=8, color=INK)
    ax.set_ylabel("ln(brow amp effort / c100)", fontsize=9, color=INK2)
    ax.set_title("Effort composite (+4.7 st, rate 0.90, +7 dB): model response vs human-predicted interval", fontsize=9, color=INK, loc="left")
    ax.legend([Line2D([], [], color=S1, marker="o", ls="none"), Line2D([], [], color=S2, marker="o", ls="none"),
               Line2D([], [], color=S3, marker="o", ls="none"), Line2D([], [], color=INK, marker="D", lw=2),
               matplotlib.patches.Patch(color=S1, alpha=0.15), Line2D([], [], color=INK2, ls=":")],
              ["cn cell", "en cell", "jp cell", "mean, 95% CI", "human-predicted (G2 slope-only, 95% CI)", "0.5x / 2x predicted"],
              fontsize=7, frameon=False, labelcolor=INK2, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(LANE / "fig_effort_vs_human.png", dpi=160, facecolor=SURF)
    plt.close(fig)

    # Fig 3: E3 follow excess + injection recovery
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), facecolor=SURF, sharey=True)
    rows_y = [("human_all", "human", "Human RAVDESS (all clips)")] + [(s, a, "%s (%s)" % (SRC_LABEL[s], a)) for s in srcs for a in ("clean", "noise")]
    yy = {k[:2]: len(rows_y) - 1 - i for i, k in enumerate(rows_y)}
    for ax, (met, title) in zip(axes, (("exc_circ", "follow excess vs circular shift (recall - chance)"),
                                       ("events_per_f0", "events per F0 peak"))):
        _style(ax)
        ax.axvline(0 if met == "exc_circ" else 1, color=INK2, lw=0.8)
        for ch, col, off in (("brow_inner", S1, 0.13), ("brow_outer", S2, -0.13)):
            for src, arm, _lab in rows_y:
                if src == "human_all":
                    q = EF[(EF.source == src) & (EF.channel == ch) & (EF.window == "plan")]
                else:
                    q = EF[(EF.source == src) & (EF.arm == arm) & (EF.channel == ch) & (EF.window == "plan") & (EF.scope == "c100")]
                if not len(q):
                    continue
                q = q.iloc[0]
                y = yy[(src, arm)] + off
                if met + "_lo" in q and np.isfinite(q[met + "_lo"]):
                    ax.plot([q[met + "_lo"], q[met + "_hi"]], [y, y], color=col, lw=2, solid_capstyle="round")
                ax.plot([q[met]], [y], "o", ms=5, color=col, mec=SURF, mew=1.2)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax = axes[2]
    _style(ax)
    for ch, col, off in (("brow_inner", S1, 0.13), ("brow_outer", S2, -0.13)):
        for src, arm, _lab in rows_y:
            cond = "lock_s1"
            q = IR[(IR.source == src) & (IR.channel == ch) & (IR.cond == cond) & ((IR.arm == arm) | (src == "human_all"))]
            if not len(q):
                continue
            ax.plot([0, q.iloc[0].recovery], [yy[(src, arm)] + off] * 2, color=col, lw=2, solid_capstyle="round")
            ax.plot([q.iloc[0].recovery], [yy[(src, arm)] + off], "o", ms=5, color=col, mec=SURF, mew=1.2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title("injection recovery (median-size raises locked to F0 peaks)", fontsize=9, color=INK, loc="left")
    axes[0].set_yticks(list(yy.values()))
    axes[0].set_yticklabels([lab for _s, _a, lab in rows_y], fontsize=8, color=INK)
    fig.legend([Line2D([], [], color=S1, marker="o", lw=2), Line2D([], [], color=S2, marker="o", lw=2)],
               ["brow inner", "brow outer"], loc="lower center", ncol=2, frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(LANE / "fig_e3_follow.png", dpi=160, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    main()

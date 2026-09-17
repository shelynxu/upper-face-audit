# -*- coding: utf-8 -*-
"""score.py -- per-item metrics for the audit2 lane (plan 3.2 E1 + E3).  Env: GPTSoVits.  CPU, 2 worker processes.

Metric code: voicedose/audit/metrics.py, imported read-only by path (never edited).

SOURCES (item time) -> CHANNELS (the brow-inner / brow-outer / upper-lid / jaw equivalents; mapping stated in audit2.md)
  emoface   post-step rig (measure lane path), 60 fps: brow_inner = mean cols 31/100 (CTRL_[LR]_brow_raiseIn),
            brow_outer = 32/101 (raiseOut), lid_upper = 39/108 (eye_eyelidU), jaw = 4 (CTRL_C_jaw-translateY);
            secondary brow_*_net = raise - brow_down (29/98)
  medtalk   raw decoder output, 174 cols in EmoFace's order (probe_medtalk.py docstring), 60 fps: same columns
  emotalk   ARKit-52, 30 fps: brow_inner = browInnerUp (2), brow_outer = mean browOuterUp L/R (3,4),
            lid_upper = mean eyeWide L/R (20,21) - mean eyeBlink L/R (8,9), jaw = jawOpen (24);
            secondary brow_*_net = the raise - mean browDown L/R (0,1)
  emote     FLAME 68 landmarks on the mean face (probe.landmarks68), 25 fps, y up, lengths / IOD (|lm36 - lm45|):
            brow_inner = (mean y lm21,22 - eye line) / IOD, brow_outer = (mean y lm17,26 - eye line) / IOD,
            eye line = mean y lm36,39,42,45; lid_upper = mean |37-41|,|38-40|,|43-47|,|44-46| / IOD; jaw = |lm30 - lm8| / IOD
            (the human pose-normalised definitions, human/analyze.py build_signals, in iBUG indices)
  f0_follower      constructed: Gaussian(60 ms) of the phrase layer re the CELL's c100 voiced median F0, rectified at 0 st
  energy_follower  constructed: Gaussian(60 ms) of [30-ms frame dBFS - (c100 speech-span median dBFS - 20 dB)]+,
                   zeroed outside the frozen speech span (speaker/cell-referenced, so level, dynamics and register all pass)
  medtalk_st       MEDTalk's per-frame audio-predicted intensity s_t (probe.stages 'B_intensity'), channel 'intensity'
  (followers: brow_inner and brow_outer carry the same follower track)

FROZEN STRUCTURE (ladder2 rule: computed once on c100, reused on every rung)
  span   = metrics.speech_mask(c100 logE, 60 fps, 30 dB) -> (s0, s1) s  (the window W2 the human G2 coefficients use)
  peaks  = metrics.f0_peaks(c100 phrase layer, >= 1 st) inside the span (human2/clipdata.py 'audit' path definition)
  acoustic grid = lilthead/extract_prosody grid(analyse(y16), 60) (stage_acoustics.py)
  RATE RUNGS (rate090, effort0, effort): every track (model output, follower, acoustic layer) is re-sampled onto the
  c100 time axis, x_tn(t) = x_item(t / rate) (ladder2 timemap rule t_c100 = rate * t_item), then scored like any rung.

ARMS  clean; noise = + white Gaussian noise, sigma = (human median noise_sigma / human median amp, per channel,
      human/human_audit_metrics.csv) x (this source's median c100 amp) x sqrt(fps / 29.97) (equal smoothed-noise SD
      at the detector's 40-ms smoothing), seeded per item x source x channel. Thresholds are recomputed per arm.
THRESHOLD  metrics.reference_threshold over the 12 c100 items per (source, channel, arm), held fixed for every rung.
FOLLOW  plan G3 window (-0.25, +0.30) s and the post-peak window (0, +0.30) s: metrics.follow_index (circular-shift null,
        200 shifts) + cross-utterance chance (events of the SAME rung of the other cells of the same language, aligned at
        span onset; mean hit count / n peaks), human2/clipdata.py's two nulls.
Also: layer_coupling (lag -0.20..+0.50 s as human_audit_metrics), accent_responses median, mean in span.
INJECTION (E3; c100 items, human2 recipe): Gaussian raises (sigma 0.10 s) peaking +0.10 s after every frozen F0 peak, size
  s x (human median event size / human median W2 amplitude) x source median c100 amplitude, s in 0.5/1/2; rand = s 1 at
  uniform times; detected at the source's fixed threshold.
Writes: results.csv, injection.csv, refs.json, score_meta.json.
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
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

LANE = Path("${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/audit2")
sys.path.insert(0, str(LANE))
import common2 as C  # noqa: E402

_spec = importlib.util.spec_from_file_location("vd_metrics", "${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/audit/metrics.py")
MX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MX)

HUMAN = Path("${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/human")
HUMAN2 = Path(C.HUMAN2)
SOURCES = ["emoface", "emotalk", "emote", "medtalk", "f0_follower", "energy_follower", "medtalk_st"]
REAL = ["emoface", "emotalk", "emote", "medtalk"]
BROWS = ["brow_inner", "brow_outer"]
COUPLE = ["brow_inner", "brow_outer", "lid_upper", "jaw"]
WIN_PLAN, WIN_POST = (-0.25, 0.30), (0.0, 0.30)
LAG_COUP = (-0.20, 0.50)
N_PERM = 200
LAG_INJ, SIG_INJ = 0.10, 0.10
G = {}


# ------------------------------------------------------------------ loading
def load_ac(iid):
    z = np.load(LANE / "acoustic" / (iid + ".npz"))
    return {k: (z[k] if z[k].ndim else float(z[k])) for k in z.files}


def cell_ref(cell):
    a = load_ac(cell + "__c100")
    span = MX.speech_mask(a["logE_db"], 60.0)
    idx = np.flatnonzero(span)
    s0, s1 = idx[0] / 60.0, (idx[-1] + 1) / 60.0
    pk = MX.f0_peaks(np.asarray(a["phrase_st"], float), np.asarray(a["voiced"]) > 0.5, 60.0, span, min_prom_st=1.0)
    return {"s0": s0, "s1": s1, "pk_t": pk["t"].tolist(), "pk_height": pk["height_st"].tolist(), "f0_med_hz": a["f0_med_hz"],
            "abs_floor_db": float(np.median(np.asarray(a["abs_db"], float)[span])) - 20.0, "dur_s": a["dur_s"],
            "n_frames60": int(len(a["logE_db"]))}


def channels_of(src, iid, ac, ref, rate):
    """-> {channel: (x on ITEM time, fps)}"""
    if src in ("emoface", "medtalk"):
        z = np.load(LANE / "outputs" / src / (iid + ".npz"))
        R = (z["post"][:, :174] if src == "emoface" else z["M"]).astype(float)
        bd = R[:, [29, 98]].mean(1)
        bi, bo = R[:, [31, 100]].mean(1), R[:, [32, 101]].mean(1)
        return {"brow_inner": (bi, 60.0), "brow_outer": (bo, 60.0), "lid_upper": (R[:, [39, 108]].mean(1), 60.0),
                "jaw": (R[:, 4], 60.0), "brow_inner_net": (bi - bd, 60.0), "brow_outer_net": (bo - bd, 60.0)}
    if src == "emotalk":
        M = np.load(LANE / "outputs" / "emotalk" / (iid + ".npz"))["M"].astype(float)
        bd = M[:, [0, 1]].mean(1)
        bi, bo = M[:, 2], M[:, [3, 4]].mean(1)
        return {"brow_inner": (bi, 30.0), "brow_outer": (bo, 30.0),
                "lid_upper": (M[:, [20, 21]].mean(1) - M[:, [8, 9]].mean(1), 30.0), "jaw": (M[:, 24], 30.0),
                "brow_inner_net": (bi - bd, 30.0), "brow_outer_net": (bo - bd, 30.0)}
    if src == "emote":
        L = np.load(LANE / "outputs" / "emote" / (iid + ".npz"))["lmk68"].astype(float)
        iod = np.linalg.norm(L[:, 36] - L[:, 45], axis=1)
        eye = L[:, [36, 39, 42, 45], 1].mean(1)
        lid = np.mean([np.linalg.norm(L[:, p] - L[:, q], axis=1) for p, q in ((37, 41), (38, 40), (43, 47), (44, 46))], axis=0)
        return {"brow_inner": ((L[:, [21, 22], 1].mean(1) - eye) / iod, 25.0),
                "brow_outer": ((L[:, [17, 26], 1].mean(1) - eye) / iod, 25.0),
                "lid_upper": (lid / iod, 25.0), "jaw": (np.linalg.norm(L[:, 30] - L[:, 8], axis=1) / iod, 25.0)}
    if src == "f0_follower":
        off = 12.0 * math.log2(ac["f0_med_hz"] / ref["f0_med_hz"])
        x = MX.gsmooth(np.clip(np.asarray(ac["phrase_st"], float) + off, 0.0, None), 60.0, 0.06)
        return {"brow_inner": (x, 60.0), "brow_outer": (x, 60.0)}
    if src == "energy_follower":
        e = np.asarray(ac["abs_db"], float)
        tm = np.arange(len(e)) / 60.0
        m = (tm >= ref["s0"] / rate) & (tm < ref["s1"] / rate)
        x = MX.gsmooth(np.clip(e - ref["abs_floor_db"], 0.0, None) * m, 60.0, 0.06)
        return {"brow_inner": (x, 60.0), "brow_outer": (x, 60.0)}
    if src == "medtalk_st":
        s = np.load(LANE / "outputs" / "medtalk" / (iid + ".npz"))["s_t"].astype(float)
        return {"intensity": (s, 60.0)}
    raise KeyError(src)


def warp(x, fps, rate, dur_c100):
    if rate == 1.0:
        return np.asarray(x, float)
    T = int(round(dur_c100 * fps))
    t = np.arange(T) / fps
    return np.interp(t / rate, np.arange(len(x)) / fps, MX._nanfill(x))


def noised(x, fps, src, ch, iid):
    sd = G["sigma"].get("%s|%s" % (src, ch))
    if not sd or not np.isfinite(sd):
        return x
    rng = np.random.default_rng(zlib.crc32(("%s|%s|%s" % (iid, src, ch)).encode()))
    return x + rng.normal(0.0, sd, len(x))


def span_mask(n, fps, ref):
    tm = np.arange(n) / fps
    return (tm >= ref["s0"]) & (tm < ref["s1"])


def bumps(T, fps, times, h):
    t = np.arange(T) / fps
    y = np.zeros(T)
    for tc in times:
        y += h * np.exp(-0.5 * ((t - tc) / SIG_INJ) ** 2)
    return y


def hit_flags(pk, ev, win):
    pk, ev = np.asarray(pk, float), np.asarray(ev, float)
    if len(pk) == 0:
        return np.zeros(0, bool)
    if len(ev) == 0:
        return np.zeros(len(pk), bool)
    d = ev[None, :] - pk[:, None]
    return ((d >= win[0]) & (d <= win[1])).any(1)


# ------------------------------------------------------------------ per item
def work(it):
    iid, cell = it["id"], it["cell"]
    ref = G["refs"][cell]
    rate = it["rate"] if it["rung"] in C.RATE_RUNGS else 1.0
    ac = load_ac(iid)
    acw = {k: warp(np.asarray(ac[k], float), 60.0, rate, ref["dur_s"]) for k in ("phrase_st", "tone_st", "voiced", "logE_db")}
    n60 = len(acw["logE_db"])
    sp60 = span_mask(n60, 60.0, ref)
    v60 = acw["voiced"] > 0.5
    pk = np.asarray(ref["pk_t"], float)
    seed = zlib.crc32(iid.encode())
    base = {k: it[k] for k in ("id", "cell", "lang", "label", "rung", "family", "role", "verdict")}
    base["rate_warp"] = rate
    rows, events = [], {}
    for src in SOURCES:
        chs = channels_of(src, iid, ac, ref, rate)
        for ch, (x, fps) in chs.items():
            xw = warp(x, fps, rate, ref["dur_s"])
            for arm in ("clean", "noise"):
                if src == "medtalk_st" and arm == "noise":
                    continue
                xx = xw if arm == "clean" else noised(xw, fps, src, ch, iid)
                thr = G["thr"]["%s|%s|%s" % (src, ch, arm)]
                m = span_mask(len(xx), fps, ref)
                cm = MX.channel_metrics(xx, fps, m, thr=thr, sign=1.0)
                ev = cm.pop("_events")
                r = dict(base, source=src, channel=ch, arm=arm, fps=fps, n_frames=len(xx))
                r.update({"m." + k: v for k, v in cm.items()})
                r["m.mean_in_span"] = float(np.mean(xx[m])) if m.sum() else np.nan
                for tag, win in (("fplan", WIN_PLAN), ("fpost", WIN_POST)):
                    fo = MX.follow_index(pk, ev["t"], (ref["s0"], ref["s1"]), win, n_perm=N_PERM, seed=seed)
                    r.update({"%s.%s" % (tag, k): v for k, v in fo.items()})
                    r[tag + ".hits"] = int(hit_flags(pk, ev["t"], win).sum())
                if arm == "clean" and ch in COUPLE:
                    xa = MX.resample(xx, fps, 60.0, n60)
                    cp = MX.layer_coupling(xa, 60.0, acw["phrase_st"], acw["tone_st"], v60, sp60, LAG_COUP)
                    r.update({"coup." + k: v for k, v in cp.items()})
                    resp = MX.accent_responses(xa, 60.0, pk)
                    r["accent_resp_median"] = float(np.nanmedian(resp)) if len(resp) and np.isfinite(resp).any() else np.nan
                rows.append(r)
                events["%s|%s|%s" % (src, ch, arm)] = ev["t"].tolist()
    return iid, rows, events


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    only = [c for c in os.environ.get("AUDIT2_CELLS", "").split(",") if c]
    items = C.items(only or None)
    cells = list(dict.fromkeys(i["cell"] for i in items))
    G["refs"] = {c: cell_ref(c) for c in cells}
    # human noise-to-amplitude ratios (audit definitions on the human tracks)
    ham = pd.read_csv(HUMAN / "human_audit_metrics.csv", low_memory=False)
    ratio = {}
    for hc, mc in (("brow_inner", "brow_inner"), ("brow_outer", "brow_outer"), ("lid", "lid_upper")):
        ratio[mc] = float(np.nanmedian(ham[hc + ".motion.noise_sigma"]) / np.nanmedian(ham[hc + ".motion.amp"]))
    ratio["jaw"] = ratio["brow_inner_net"] = ratio["brow_outer_net"] = 0.5 * (ratio["brow_inner"] + ratio["brow_outer"])
    # human injection size relative to the human W2 amplitude (human2 clipdata)
    cd = pd.read_csv(HUMAN2 / "clipdata.csv", low_memory=False)
    meta = json.loads((HUMAN2 / "clipdata_meta.json").read_text())
    inj_frac = {ch: float(meta["median_event_size"][ch] / np.nanmedian(cd["amp_%s_W2_full" % ch])) for ch in BROWS}
    # c100 tracks -> sigma, thresholds
    c100 = [i for i in items if i["rung"] == "c100"]
    tracks = {}
    for it in c100:
        ref = G["refs"][it["cell"]]
        ac = load_ac(it["id"])
        for src in SOURCES:
            for ch, (x, fps) in channels_of(src, it["id"], ac, ref, 1.0).items():
                tracks.setdefault("%s|%s" % (src, ch), []).append((it["id"], it["cell"], np.asarray(x, float), fps))
    G["sigma"], G["thr"], G["amp_c100_median"] = {}, {}, {}
    for key, L in tracks.items():
        src, ch = key.split("|")
        amps = [MX.span_pct(x, fps, span_mask(len(x), fps, G["refs"][cell])) for _i, cell, x, fps in L]
        G["amp_c100_median"][key] = float(np.nanmedian(amps))
        fps0 = L[0][3]
        G["sigma"][key] = ratio.get(ch, np.nan) * G["amp_c100_median"][key] * math.sqrt(fps0 / 29.97) if src != "medtalk_st" else np.nan
        masks = [span_mask(len(x), fps, G["refs"][cell]) for _i, cell, x, fps in L]
        G["thr"][key + "|clean"] = MX.reference_threshold([x for _i, _c, x, _f in L], [f for *_a, f in L], masks)
        if src != "medtalk_st":
            G["thr"][key + "|noise"] = MX.reference_threshold([noised(x, fps, src, ch, iid) for iid, _c, x, fps in L],
                                                              [f for *_a, f in L], masks)
    print("refs/thresholds %.0fs" % (time.time() - t0), flush=True)

    with Pool(2) as pool:
        out = pool.map(work, items, chunksize=3)
    rows, EV = [], {}
    for iid, rr, ev in out:
        rows += rr
        EV[iid] = ev
    D = pd.DataFrame(rows)
    # cross-utterance chance: the same rung of the other cells of the same language, aligned at span onset
    lang_of = {i["cell"]: i["lang"] for i in items}
    cc = np.full(len(D), np.nan)
    for j, r in enumerate(D[["id", "cell", "rung", "source", "channel", "arm"]].itertuples(index=False)):
        ref = G["refs"][r.cell]
        pk = np.asarray(ref["pk_t"], float)
        if len(pk) == 0:
            continue
        key = "%s|%s|%s" % (r.source, r.channel, r.arm)
        vals = []
        for oc in cells:
            if oc == r.cell or lang_of[oc] != lang_of[r.cell]:
                continue
            if "%s__%s" % (oc, r.rung) not in EV:
                continue
            ev = np.asarray(EV["%s__%s" % (oc, r.rung)][key], float) - G["refs"][oc]["s0"] + ref["s0"]
            vals.append(hit_flags(pk, ev, WIN_PLAN).sum())
        cc[j] = float(np.mean(vals)) / len(pk) if vals else np.nan
    D["fplan.recall_chance_cross"] = cc
    D["fplan.recall_excess_cross"] = D["fplan.recall"] - cc
    # realised doses (ladder2 manifest)
    dose = {}
    for it in items:
        rz = it["realised"]
        dose[it["id"]] = {"x.k_praat": rz.get("realised_k_praat"), "x.k_pyin": rz.get("realised_k_pyin"),
                          "x.g_ip": rz.get("realised_g_ip"), "x.reg_praat": rz.get("register_st_praat"),
                          "x.reg_pyin": rz.get("register_st_pyin"), "x.d_rms_speech_db": rz.get("d_rms_speech_db")}
    for k in ("x.k_praat", "x.k_pyin", "x.g_ip", "x.reg_praat", "x.reg_pyin", "x.d_rms_speech_db"):
        D[k] = D["id"].map(lambda i: dose[i][k])
    suf = "_subset" if only else ""
    D.to_csv(LANE / ("results%s.csv" % suf), index=False, float_format="%.6g")
    print("results.csv", D.shape, "%.0fs" % (time.time() - t0), flush=True)

    # injection positive control on the c100 items (brows, real models + followers)
    inj = []
    conds = [("lock_s0.5", 0.5, "lock"), ("lock_s1", 1.0, "lock"), ("lock_s2", 2.0, "lock"), ("rand_s1", 1.0, "rand")]
    for key, L in tracks.items():
        src, ch = key.split("|")
        if ch not in BROWS or src == "medtalk_st":
            continue
        for iid, cell, x, fps in L:
            ref = G["refs"][cell]
            pk = np.asarray(ref["pk_t"], float)
            m = span_mask(len(x), fps, ref)
            for arm in ("clean", "noise"):
                xa = x if arm == "clean" else noised(x, fps, src, ch, iid)
                thr = G["thr"]["%s|%s" % (key, arm)]
                ev0 = MX.detect_events(xa, fps, thr, m, 1.0)["t"]
                hb = hit_flags(pk, ev0, WIN_PLAN)
                fo0 = MX.follow_index(pk, ev0, (ref["s0"], ref["s1"]), WIN_PLAN, n_perm=N_PERM, seed=1)
                inj.append(dict(source=src, channel=ch, arm=arm, cell=cell, cond="base", n_pk=len(pk), hits=int(hb.sum()),
                                hits_base=int(hb.sum()), ideal=int(hb.sum()), recall_chance=fo0["recall_chance"],
                                n_ev=len(ev0), size=0.0))
                h = inj_frac[ch] * G["amp_c100_median"][key]
                for cond, s, kind in conds:
                    rng = np.random.default_rng([zlib.crc32(iid.encode()), conds.index((cond, s, kind)), BROWS.index(ch)])
                    times = pk + LAG_INJ if kind == "lock" else rng.uniform(ref["s0"], ref["s1"], len(pk))
                    ev = MX.detect_events(xa + bumps(len(xa), fps, times, s * h), fps, thr, m, 1.0)["t"]
                    hf = hit_flags(pk, ev, WIN_PLAN)
                    ideal = (hb | True) if kind == "lock" else hb
                    fo = MX.follow_index(pk, ev, (ref["s0"], ref["s1"]), WIN_PLAN, n_perm=N_PERM, seed=1)
                    inj.append(dict(source=src, channel=ch, arm=arm, cell=cell, cond=cond, n_pk=len(pk), hits=int(hf.sum()),
                                    hits_base=int(hb.sum()), ideal=int(np.sum(ideal)), recall_chance=fo["recall_chance"],
                                    n_ev=len(ev), size=float(s * h)))
    pd.DataFrame(inj).to_csv(LANE / ("injection%s.csv" % suf), index=False, float_format="%.6g")
    (LANE / ("refs%s.json" % suf)).write_text(json.dumps(G["refs"], indent=1))
    (LANE / ("score_meta%s.json" % suf)).write_text(json.dumps({"noise_ratio_human": ratio, "inj_frac_human": inj_frac,
                                                      "sigma": G["sigma"], "thr": G["thr"], "amp_c100_median": G["amp_c100_median"],
                                                      "windows": {"plan": WIN_PLAN, "post": WIN_POST}, "lag_coupling": LAG_COUP,
                                                      "n_perm": N_PERM, "seconds": time.time() - t0}, indent=1, default=float))
    print("DONE score %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

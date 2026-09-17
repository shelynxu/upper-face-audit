# -*- coding: utf-8 -*-
"""clipdata.py -- per-clip inputs for human2 (E2): face amplitudes in three windows with equal-length variants,
per-actor event thresholds, and the F0-peak follow-index INJECTION positive control.

Reads (read-only): human/ravdess_face3d, human/acoustic2 (through h1_analyze.py = verbatim copy of human/analyze.py,
whose load_all/build_signals are the pose-normalised brow/lid tracks), human/human_audit_metrics.csv (median event size).
Writes: clipdata.csv (one row per clip), follow_records.csv (one row per clip x path x threshold x channel x condition x
window), clipdata_meta.json.

WINDOWS (video frames, wav t = 0 = frame 0)
  W1  the human lane's speech span v2 (acoustic.py; level_def.py)             -- human_dose.md amplitudes
  W2  metrics.speech_mask(logE 60 fps, 30 dB below max) = the span utterance_metrics scores MODEL outputs on
  W3  extract_prosody VAD (30 fps) dilated 15 frames = audit_compat `brow_*|amp` (the G1 numbers in the plan)
AMPLITUDE = metrics.span_pct (p95 - p5 on a 240-Hz linear re-grid) of the raw pose-normalised track.
EQUAL-LENGTH VARIANTS (W1, W2; contiguous windows [ta, tb])
  full     p95-p5 over [ta, tb]
  fix1s    centred 1.0-s window (every W1 span >= 1.03 s)
  slide05  mean of p95-p5 over sliding 0.5-s windows (hop 1/30 s) inside [ta, tb]
  pm       pair-matched: centred crop of length min(own, partner) duration (partner = other intensity, same actor x
           emotion x statement x repetition); NaN for neutral
  tn256    time-normalised: [ta, tb] re-sampled to exactly 256 samples (percentile range is nearly invariant to this;
           reported to show that time-normalising does NOT equalise sampling opportunity)

FOLLOW INDEX + INJECTION (brow_inner, brow_outer)
  path 'audit': metrics.f0_peaks (phrase layer, >= 1 st, >= 0.25 s apart, voiced) inside W2; events =
     metrics.detect_events(raw, thr = metrics.reference_threshold over the actor's normal-intensity clips, mask W2) --
     exactly utterance_metrics; windows (0, 0.30) (metrics default) and (-0.25, +0.30) (plan G3).
  path 'lane': human_dose.md section 4 -- phrase-layer peaks (find_peaks prom 0.25 / >= 1 st, >= 150 ms, voiced, in W1);
     events = analyze.events on Savitzky-Golay(5,2) at thr 'jit' (3 x white-noise sigma) or 'rest' (p95 of rest-window
     prominences); window (-0.10, +0.50).
  conditions: base; lock_q1_s{0.5,1,2,3} = a Gaussian raise (sigma 0.10 s) of s x the human median event size peaking
     LAG = +0.10 s after EVERY peak the path uses; lock_q0.5_s1 = each peak w.p. 0.5; rand_q1_s1 = the same number of
     raises at uniform random times in the span (not locked). Raises are added to the RAW track before smoothing.
  per record: n peaks, hits, ideal hits (base hits OR injected), circular-shift chance (200 shifts, common random numbers
     across conditions), cross-utterance chance (events of the same actor x statement's other clips in the same condition,
     aligned at span onset).
RUN: clipdata.sh (env d1_emote; single process)
"""
import importlib.util
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_filter, find_peaks

LANE = Path("${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/human2")
HUMAN = Path("${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/human")
sys.argv = sys.argv[:1]                    # h1_analyze reads sys.argv[1] at import (actor list); keep it empty
sys.path.insert(0, str(LANE))
import h1_analyze as AN  # noqa: E402

sys.path.insert(0, "${PROJECT_ROOT}/pipeline/lilthead")
import extract_prosody as EP  # noqa: E402

EP.OUT = LANE / "_unused_prosody_dir"
_spec = importlib.util.spec_from_file_location("metrics_v11", str(LANE / "metrics_v11.py"))
MX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MX)

CH = ["brow_inner", "brow_outer", "lid", "head_pitch"]
BROWS = ["brow_inner", "brow_outer"]
K_JIT = 3.0                                 # human_dose.md section 1 (K = 3) [measured by the human lane]
UP = 240.0
LAG, SIG = 0.10, 0.10
N_PERM = 200
WIN_AUD, WIN_PLAN, WIN_LANE = (0.0, 0.30), (-0.25, 0.30), (-0.10, 0.50)
CONDS = [("base", 0.0, 0.0, "lock"), ("lock_q1_s0.5", 1.0, 0.5, "lock"), ("lock_q1_s1", 1.0, 1.0, "lock"),
         ("lock_q1_s2", 1.0, 2.0, "lock"), ("lock_q1_s3", 1.0, 3.0, "lock"), ("lock_q0.5_s1", 0.5, 1.0, "lock"),
         ("rand_q1_s1", 1.0, 1.0, "rand")]


def p9505(v):
    v = np.asarray(v, float)
    return float(np.percentile(v, 95) - np.percentile(v, 5)) if len(v) >= 3 else float("nan")


def regrid(x, fps, ta, tb, n=None):
    t = np.arange(len(x)) / fps
    tu = np.linspace(ta, tb, n) if n else np.arange(ta, tb + 1e-9, 1.0 / UP)
    return np.interp(tu, t, MX._nanfill(x))


def bounds(mask, fps):
    idx = np.flatnonzero(mask)
    return (idx[0] / fps, idx[-1] / fps) if len(idx) >= 3 else (np.nan, np.nan)


def amp_variants(x, fps, ta, tb):
    out = {}
    if not np.isfinite(ta):
        return dict(full=np.nan, fix1s=np.nan, slide05=np.nan, tn256=np.nan)
    v = regrid(x, fps, ta, tb)
    out["full"] = p9505(v)
    c = 0.5 * (ta + tb)
    a, b = max(ta, c - 0.5), min(tb, c + 0.5)
    out["fix1s"] = p9505(regrid(x, fps, a, b))
    w = int(round(0.5 * UP))
    if len(v) <= w:
        out["slide05"] = out["full"]
    else:
        S = np.lib.stride_tricks.sliding_window_view(v, w)[::8]
        out["slide05"] = float(np.mean(np.percentile(S, 95, axis=1) - np.percentile(S, 5, axis=1)))
    out["tn256"] = p9505(regrid(x, fps, ta, tb, n=256))
    return out


def bumps(T, fps, times, h):
    t = np.arange(T) / fps
    y = np.zeros(T)
    for tc in times:
        y += h * np.exp(-0.5 * ((t - tc) / SIG) ** 2)
    return y


def hit_flags(pk, ev, win):
    if len(pk) == 0:
        return np.zeros(0, bool)
    if len(ev) == 0:
        return np.zeros(len(pk), bool)
    d = np.asarray(ev)[None, :] - np.asarray(pk)[:, None]
    return ((d >= win[0]) & (d <= win[1])).any(1)


def circ_chance(pk, ev, span, win, u):
    """mean hits (count, not fraction) over circular shifts u (n_perm,) of the events inside span."""
    if len(pk) == 0 or len(ev) == 0:
        return 0.0
    a, b = span
    L = max(b - a, 1e-6)
    sh = a + np.mod(np.asarray(ev)[None, :] - a + (u * L)[:, None], L)          # (P, E)
    d = sh[:, None, :] - np.asarray(pk)[None, :, None]                            # (P, K, E)
    return float(((d >= win[0]) & (d <= win[1])).any(2).sum(1).mean())


def main():
    t0 = time.time()
    clips, missing = AN.load_all(None)
    actors = sorted({c["actor"] for c in clips.values()})
    AN.build_signals(clips, actors)
    print("signals", len(clips), "missing", len(missing), "%.0fs" % (time.time() - t0), flush=True)
    am = pd.read_csv(HUMAN / "human_audit_metrics.csv", low_memory=False)
    MED = {ch: float(np.nanmedian(am[f"{ch}.motion.prom_median"])) for ch in BROWS}
    print("median event size (audit detector, all clips):", MED, flush=True)

    # ---------------- acoustic grids + windows
    for name, c in clips.items():
        za = np.load(AN.AC / ("03" + name[2:] + ".npz"))
        A = {"f0_hz_5ms": za["f0_hz_5ms"], "rms_db_10ms": za["rms_db_10ms"], "nuclei_s": za["nuclei_s"], "dur_s": c["S"]["file_dur_s"]}
        G60, G30 = EP.grid(A, 60), EP.grid(A, 30)
        T, fps = c["T"], c["fps"]
        tv = np.arange(T) / fps
        W1 = np.zeros(T, bool)
        W1[c["i0"]:c["i1"] + 1] = True
        e60 = np.asarray(G60["logE_db"], float)
        sp60 = MX.speech_mask(e60, 60.0, 30.0)
        ii = np.flatnonzero(sp60)
        span_s = (ii[0] / 60.0, (ii[-1] + 1) / 60.0)
        W2 = (tv >= span_s[0]) & (tv < span_s[1])
        vad30 = np.asarray(G30["vad"]) > 0.5
        valid30 = binary_dilation(vad30, iterations=15)
        W3 = valid30[np.clip(np.round(tv * 30.0).astype(int), 0, len(valid30) - 1)]
        c["W"] = {"W1": W1, "W2": W2, "W3": W3}
        c["span_s"] = span_s
        v60 = np.asarray(G60["voiced"]) > 0.5
        c["pkA"] = MX.f0_peaks(np.asarray(G60["phrase_st"], float), v60, 60.0, sp60, min_prom_st=1.0)["t"]
        ph, vo = c["phrase5"], c["voiced5"]
        pk, pr = find_peaks(ph, prominence=0.25, distance=AN.F0PK_DIST)
        tpk = pk * 0.005
        keep = (tpk >= c["S"]["t_on"]) & (tpk <= c["S"]["t_off"]) & vo[np.clip(pk, 0, len(vo) - 1)]
        c["pkL"] = tpk[keep][pr["prominences"][keep] >= AN.F0PK_PROM]
    print("grids", "%.0fs" % (time.time() - t0), flush=True)

    # ---------------- thresholds
    thr_ref, thr_jit, thr_rest = {}, {}, {}
    for a in actors:
        cs = [c for c in clips.values() if c["actor"] == a]
        ref = [c for c in cs if c["intensity"] == 1]
        for ch in CH:
            thr_ref[(a, ch)] = MX.reference_threshold([c["raw"][ch] for c in ref], [c["fps"] for c in ref], [c["W"]["W2"] for c in ref])
        for ch in BROWS:
            sig = AN.robust_sigma_d2([c["raw"][ch][c["r0"]:c["r1"] + 1] for c in cs if c["rest_ok"]])
            proms = []
            for c in cs:
                if not c["rest_ok"]:
                    continue
                pk, pr = find_peaks(c["sm"][ch], prominence=0, distance=AN.MIN_DIST, wlen=AN.WLEN)
                m = (pk >= c["r0"]) & (pk <= c["r1"])
                proms.append(pr["prominences"][m])
            proms = np.concatenate(proms) if proms else np.array([])
            thr_jit[(a, ch)] = K_JIT * sig
            thr_rest[(a, ch)] = float(np.percentile(proms, 95)) if proms.size >= 20 else np.nan
    for ch in BROWS:
        med = np.nanmedian([thr_rest[(a, ch)] for a in actors])
        for a in actors:
            if not np.isfinite(thr_rest[(a, ch)]):
                thr_rest[(a, ch)] = med
            thr_rest[(a, ch)] = max(thr_rest[(a, ch)], thr_jit[(a, ch)])

    # ---------------- per-clip amplitudes
    rows = []
    for name, c in clips.items():
        fps = c["fps"]
        r = dict(clip=name, actor=c["actor"], emotion=c["emotion"], intensity=c["intensity"], statement=c["statement"],
                 repetition=c["repetition"], fps=fps, T=c["T"], n_pkA=len(c["pkA"]), n_pkL=len(c["pkL"]))
        for wn in ("W1", "W2", "W3"):
            ta, tb = bounds(c["W"][wn], fps)
            r[f"{wn}_on"], r[f"{wn}_dur"] = ta, (tb - ta) if np.isfinite(ta) else np.nan
        for ch in CH:
            x = c["raw"][ch]
            r[f"amp_{ch}_W3_full"] = MX.span_pct(x, fps, c["W"]["W3"])
            for wn in ("W1", "W2"):
                ta, tb = bounds(c["W"][wn], fps)
                for k, v in amp_variants(x, fps, ta, tb).items():
                    r[f"amp_{ch}_{wn}_{k}"] = v
            seg = c["sm"][ch][c["i0"]:c["i1"] + 1]
            r[f"laneSG_{ch}"] = p9505(seg)
            r[f"thr_ref_{ch}"] = thr_ref[(c["actor"], ch)]
        for ch in BROWS:
            r[f"thr_jit_{ch}"], r[f"thr_rest_{ch}"] = thr_jit[(c["actor"], ch)], thr_rest[(c["actor"], ch)]
        rows.append(r)
    D = pd.DataFrame(rows)
    key = ["actor", "emotion", "statement", "repetition"]
    part = {tuple(k): n for n, k in zip(D["clip"], D[key + ["intensity"]].values.tolist())}
    for wn in ("W1", "W2"):
        for ch in CH:
            D[f"amp_{ch}_{wn}_pm"] = np.nan
    for i, rr_ in D.iterrows():
        if rr_["emotion"] == "neu":
            continue
        other = part.get((rr_["actor"], rr_["emotion"], rr_["statement"], rr_["repetition"], 3 - rr_["intensity"]))
        if other is None:
            continue
        c, o = clips[rr_["clip"]], clips[other]
        for wn in ("W1", "W2"):
            ta, tb = bounds(c["W"][wn], c["fps"])
            oa, ob = bounds(o["W"][wn], o["fps"])
            Lm = min(tb - ta, ob - oa)
            cc = 0.5 * (ta + tb)
            for ch in CH:
                D.at[i, f"amp_{ch}_{wn}_pm"] = p9505(regrid(c["raw"][ch], c["fps"], cc - Lm / 2, cc + Lm / 2))
    D.to_csv(LANE / "clipdata.csv", index=False, float_format="%.6g")
    print("clipdata.csv", D.shape, "%.0fs" % (time.time() - t0), flush=True)

    # ---------------- follow index + injection
    recs, evstore = [], {}
    for name, c in clips.items():
        fps, T, a = c["fps"], c["T"], c["actor"]
        seed = zlib.crc32(name.encode())
        u = np.random.default_rng(seed).uniform(0, 1, N_PERM)
        paths = {"audit": (c["pkA"], c["span_s"]), "lane": (c["pkL"], (c["S"]["t_on"], c["S"]["t_off"]))}
        for path, (pk, span) in paths.items():
            for ch in BROWS:
                base_hits = {}
                for cond, q, s, kind in CONDS:
                    rng = np.random.default_rng([seed, CONDS.index((cond, q, s, kind)), BROWS.index(ch), 0 if path == "audit" else 1])
                    if kind == "lock":
                        sel = np.ones(len(pk), bool) if q >= 1 else (rng.uniform(0, 1, len(pk)) < q)
                        times = pk[sel] + LAG
                    else:
                        sel = np.zeros(len(pk), bool)
                        times = rng.uniform(span[0], span[1], len(pk))
                    x = c["raw"][ch] + (bumps(T, fps, times, s * MED[ch]) if s > 0 else 0.0)
                    if path == "audit":
                        ev = MX.detect_events(x, fps, thr_ref[(a, ch)], c["W"]["W2"], 1.0)["t"]
                        evsets = {"ref": ev}
                        wins = {"aud": WIN_AUD, "plan": WIN_PLAN}
                    else:
                        sm = savgol_filter(x, AN.SG_WIN, AN.SG_ORD)
                        evsets = {}
                        for tn, thr in (("jit", thr_jit[(a, ch)]), ("rest", thr_rest[(a, ch)])):
                            t_, _ = AN.events(sm, thr, c["i0"], c["i1"], False)
                            evsets[tn] = t_ / fps
                        wins = {"lane": WIN_LANE}
                    for tn, ev in evsets.items():
                        for wl, win in wins.items():
                            hf = hit_flags(pk, ev, win)
                            bk = (tn, wl)
                            if cond == "base":
                                base_hits[bk] = hf
                            ideal = (base_hits[bk] | sel) if len(pk) else hf
                            recs.append(dict(clip=name, actor=a, emotion=c["emotion"], intensity=c["intensity"],
                                             statement=c["statement"], path=path, thr=tn, ch=ch, cond=cond, win=wl,
                                             n_pk=len(pk), n_inj=int(sel.sum()) if kind == "lock" else len(times),
                                             n_ev=len(ev), hits=int(hf.sum()), ideal_hits=int(ideal.sum()),
                                             chance_circ=circ_chance(pk, ev, span, win, u), span_on=span[0],
                                             span_dur=span[1] - span[0]))
                        evstore[(name, path, tn, ch, cond)] = ev
    R = pd.DataFrame(recs)
    print("follow records", R.shape, "%.0fs" % (time.time() - t0), flush=True)

    # cross-utterance chance: other clips of the same actor x statement, same condition, aligned at span onset
    groups = {}
    for name, c in clips.items():
        groups.setdefault((c["actor"], c["statement"]), []).append(name)
    onset = {(name, p): (clips[name]["span_s"][0] if p == "audit" else clips[name]["S"]["t_on"]) for name in clips for p in ("audit", "lane")}
    cross = np.full(len(R), np.nan)
    wmap = {"aud": WIN_AUD, "plan": WIN_PLAN, "lane": WIN_LANE}
    for j, rec in enumerate(R.itertuples(index=False)):
        if rec.n_pk == 0:
            cross[j] = 0.0
            continue
        pk = clips[rec.clip]["pkA"] if rec.path == "audit" else clips[rec.clip]["pkL"]
        others = [o for o in groups[(rec.actor, rec.statement)] if o != rec.clip]
        evs, own = [], []
        for k, o in enumerate(others):
            e = evstore[(o, rec.path, rec.thr, rec.ch, rec.cond)] - onset[(o, rec.path)] + onset[(rec.clip, rec.path)]
            evs.append(e)
            own.append(np.full(len(e), k))
        if not others:
            continue
        ev = np.concatenate(evs) if evs else np.array([])
        ow = np.concatenate(own).astype(int) if own else np.array([], int)
        win = wmap[rec.win]
        if len(ev) == 0:
            cross[j] = 0.0
            continue
        d = ev[None, :] - pk[:, None]
        h = ((d >= win[0]) & (d <= win[1])).astype(np.int32)                      # (K, E)
        per_other = np.zeros((len(pk), len(others)), np.int32)
        np.add.at(per_other.T, ow, h.T)
        cross[j] = float((per_other > 0).sum(0).mean())                           # mean over others of hit counts
    R["chance_cross"] = cross
    R.to_csv(LANE / "follow_records.csv", index=False, float_format="%.6g")
    meta = dict(median_event_size=MED, K_jit=K_JIT, lag_s=LAG, sigma_s=SIG, n_perm=N_PERM, conds=CONDS,
                windows=dict(aud=WIN_AUD, plan=WIN_PLAN, lane=WIN_LANE), metrics_version=MX.__version__,
                thr_ref_median={ch: float(np.median([thr_ref[(a, ch)] for a in actors])) for ch in CH},
                thr_jit_median={ch: float(np.median([thr_jit[(a, ch)] for a in actors])) for ch in BROWS},
                thr_rest_median={ch: float(np.median([thr_rest[(a, ch)] for a in actors])) for ch in BROWS},
                n_clips=len(clips), missing=missing)
    (LANE / "clipdata_meta.json").write_text(json.dumps(meta, indent=1))
    print("DONE clipdata %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

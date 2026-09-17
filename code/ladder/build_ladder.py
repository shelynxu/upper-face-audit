# -*- coding: utf-8 -*-
"""build_ladder.py -- voice dose ladder v2 (lane voicedose/ladder2): ACTOR speech in cn / en / jp.

EDITED COPY of voicedose/ladder/build_ladder.py (the verbatim original is _orig_build_ladder.py in this lane).
Spec: the voice-dose plan, section 3.1 (+ section 2.2 "structure frozen across rungs").

ENGINE -- reused by import, nothing copy-edited (as ladder/):
    h2_prosody_probe3.highpass + rms_normalize   source conditioning (80 Hz HPF, -26 dBFS)
    h2_prosody_probe2.analyse_constrained        WORLD analysis -- ANALYSIS ONLY
    h2_prosody_probe.manipulate                   target-contour math (k_phrase, k_tone = 1.0, reg_st); its WORLD
                                                  waveform is DISCARDED, never written
    h2_prosody_probe3.psola_resynth               Praat TD-PSOLA: null, pitch targets, and the rate DurationTier
    h2_prosody_probe3.match_level / wave_dyn / peak_guard
    h2_prosody_probe2.track_against_target / measures / design_beta
    h2_voicefix_gen.speech_edges                  speech extent (35-dB gate) for the -26 dBFS level rule
    lilthead/extract_prosody.features             LiltHead v1 input tracks (model view)
    model/draft/features_v2.raw_tracks / profile_from_raw / structure   the v2 structure (IPs, nuclei, syllables)
    human/acoustic.speech_span                    the human lane's speech span (its one() writes into the human lane,
                                                  so its scalar formulas are re-expressed in human_cues() below)

WHAT CHANGED vs ladder/
  sources   the 12 pool_set.json tier-C ACTOR cells (ESD 0002 short3 x 5, RAVDESS Actor 02 take1 x 3, JVNV F2 reg1 x 4);
            ladder/ used TTS clones for en / jp.
  rungs     + r+2 r+4    register: psola_resynth(f0_target = analysis F0 x 2^(st/12)) = manipulate(A, reg_st=st)
            + rate090    PSOLA null with a constant DurationTier 1/0.90 (slower, pitch kept); time map stored
            + effort0    register +4.7 st AND rate 0.90 in ONE PSOLA pass, at the common -26 dBFS (diagnostic twin)
            + effort     effort0 on disk x 10^(7/20)  (= register +4.7 st, rate 0.90, then +7 dB)
            + tgt100     reference: PSOLA with the UNCHANGED analysis contour imposed as target -- the floor of every
                         target rung (c100 keeps Praat's own pitch tier)
  dynamics  wave_dyn is applied only inside the frozen c100 IPs intersected with the c100 speech extent, with
            20-ms raised-cosine ramps inside each IP edge; everywhere else the output IS the base signal sample for
            sample. ladder/ applied it to the whole file (non-speech +12.24 dB under g060, ladder.md section 3).
  frozen    structure/<cell>.json: features_v2.structure computed ONCE on the on-disk c100 with a c100-only
            SpeakerProfile, LiltHead v1 nuclei, and the same spans mapped onto the rate rungs.
  measures  every F0 / envelope / LiltHead track of a rate item is mapped onto c100's time axis
            (t_item = t_c100 / rate, nearest frame) BEFORE any comparison. New: realised register st, realised time
            scale (envelope-correlation search) and nucleus alignment, adjacent-syllable register-step RMS on the frozen
            syllables, non-speech and internal-pause RMS vs c100, human-lane cue values with deltas vs c100.

LEVEL. Two-gain rule for every item except lvl+6 / effort (which are pure gains on the on-disk c100 / effort0):
RMS over the c100 speech extent (mapped by 1/rate on rate items) = -26 dBFS, and RMS of the non-speech samples
(outside the frozen c100 IPs) = c100's non-speech RMS, the gain moving along 20-ms ramps inside the IP edges
(pin() in build_cell). The smoke run showed why: with ladder/'s one scalar gain the non-speech of k167 sat +4.5 dB
and of `more` +2.6 dB off c100 (cn_esd0002_short3_hap), a pause-level cue for any VAD / energy-floor front end.

RUN (WSL): bash run_build.sh   (GPTSoVits env, CPU, <= 2 worker processes)
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse          # noqa: E402
import hashlib           # noqa: E402
import json              # noqa: E402
import math              # noqa: E402
import time              # noqa: E402
import traceback         # noqa: E402
from math import gcd     # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np       # noqa: E402
import soundfile as sf   # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

LANE = "${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/ladder2"
VD = "${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose"
WAV_DIR = LANE + "/wav"
STRUCT_DIR = LANE + "/structure"
TMAP_DIR = LANE + "/timemap"
POOL_SET = "${PROJECT_ROOT}/pipeline/prosody_ladder/pool_set.json"
CODE_FILES = {"features_v2": VD + "/model/draft/features_v2.py", "extract_prosody": "${PROJECT_ROOT}/pipeline/lilthead/extract_prosody.py",
              "human_acoustic": VD + "/human/acoustic.py", "h2_prosody_probe3": "${PROJECT_ROOT}/pipeline/h2_prosody_probe3.py",
              "h2_prosody_probe": "${PROJECT_ROOT}/pipeline/h2_prosody_probe.py", "h2_prosody_probe2": "${PROJECT_ROOT}/pipeline/h2_prosody_probe2.py"}
sys.path.insert(0, "${PROJECT_ROOT}/pipeline")
sys.path.insert(0, "${PROJECT_ROOT}/pipeline/lilthead")
sys.path.insert(0, VD + "/model/draft")
sys.path.insert(0, VD + "/human")
import h2_prosody_probe as P1        # noqa: E402
import h2_prosody_probe2 as P2       # noqa: E402
import h2_prosody_probe3 as P3       # noqa: E402
import h2_voicefix_gen as VFG        # noqa: E402
import extract_prosody as EP         # noqa: E402
import features_v2 as FV             # noqa: E402
import acoustic as HA                # noqa: E402

# Hard rule: no reused module may write outside this lane. None of the functions called below writes,
# but the module-level output constants are redirected anyway.
for _mod, _names in ((P1, ("OUT_DIR", "SERVE_DIR")), (P2, ("OUT_DIR", "SERVE_DIR")),
                     (P3, ("OUT_DIR", "ARCHIVE", "SERVE_DIR"))):
    for _n in _names:
        setattr(_mod, _n, LANE + "/_unused_" + _n.lower())
EP.OUT = Path(LANE + "/_unused_lilthead_out")
HA.OUT = Path(LANE + "/_unused_human_out")
HA.LANE = Path(LANE)

CELLS = ["cn_esd0002_short3_ang", "cn_esd0002_short3_hap", "cn_esd0002_short3_sad", "cn_esd0002_short3_sur",
         "cn_esd0002_short3_neu", "en_rav02_take1_hap", "en_rav02_take1_sad", "en_rav02_take1_ang",
         "jp_jvnvF2_reg1_ang", "jp_jvnvF2_reg1_sad", "jp_jvnvF2_reg1_hap", "jp_jvnvF2_reg1_sur"]

TARGET_DBFS = -26.0
LEVEL_DB = 6.0
EFFORT_REG_ST, EFFORT_RATE, EFFORT_LEVEL_DB = 4.7, 0.90, 7.0
GUARD = 0.98                       # P3.peak_guard's threshold
XFADE_MS = 20.0
RUNGS = [
    ("c100", "null", {}),
    ("tgt100", "target_null", {}),
    ("k050", "span", dict(k=0.50)),
    ("k075", "span", dict(k=0.75)),
    ("k133", "span", dict(k=1.33)),
    ("k167", "span", dict(k=1.67)),
    ("g060", "dyn", dict(g=0.60)),
    ("g150", "dyn", dict(g=1.50)),
    ("lvl+6", "level", dict(level_db=LEVEL_DB)),
    ("less", "combined", dict(k=0.75, g=0.60)),
    ("more", "combined", dict(k=1.33, g=1.50)),
    ("r+2", "register", dict(reg_st=2.0)),
    ("r+4", "register", dict(reg_st=4.0)),
    ("rate090", "rate", dict(rate=0.90)),
    ("effort0", "effort", dict(reg_st=EFFORT_REG_ST, rate=EFFORT_RATE)),
    ("effort", "effort", dict(reg_st=EFFORT_REG_ST, rate=EFFORT_RATE, level_db=EFFORT_LEVEL_DB)),
]
ROLE = {"src": "reference", "c100": "null", "tgt100": "reference", "lvl+6": "diagnostic", "effort0": "diagnostic"}
PITCH_RUNGS = ("k050", "k075", "k133", "k167", "less", "more")
REG_RUNGS = ("r+2", "r+4", "effort0", "effort")
F0_UNTOUCHED = ("g060", "g150", "lvl+6", "src", "tgt100", "rate090")
ST = 12.0 / math.log(2.0)
WIN = max(3, int(round(P1.PHRASE_SMOOTH_MS / P1.FRAME_MS)) | 1)
EAR_PASS_ST, EAR_REJECT_ST = 3.43, 5.72    # h2_wave1_ladder.solve_k docstring: the listening check on p180 cells
TEXT_SYL = {"en": (7, "RAVDESS statements 01/02: 7 written syllables [V-abs, as cited in human/human_dose.md section 0]")}


def rnd(x, n=3):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def layers(f0):
    """log-F0 -> median + 400-ms phrase layer + tone residual (natural-log units), as d2/design_beta."""
    lf0, v = P1.interp_unvoiced(np.asarray(f0, float))
    if lf0 is None or v.sum() < 10:
        return None
    med = float(np.median(lf0[v]))
    phrase = P1.smooth(lf0 - med, WIN)
    return dict(med=med, phrase=phrase, tone=(lf0 - med) - phrase, v=v)


def pspan(L):
    if not L:
        return None
    ph = L["phrase"][L["v"]]
    return float((np.percentile(ph, 90) - np.percentile(ph, 10)) * ST)


def restrict(f0, v_ref):
    """Keep only frames voiced in the reference (analysis) contour: one frame population per cell."""
    f = np.asarray(f0, float).copy()
    n = min(len(f), len(v_ref))
    f[:n][~v_ref[:n]] = 0.0
    f[n:] = 0.0
    return f


def slope(L, Lref):
    """LS slope of an item's phrase layer on the reference's, frames voiced in both = realised k."""
    if not L or not Lref:
        return None
    n = min(len(L["v"]), len(Lref["v"]))
    m = L["v"][:n] & Lref["v"][:n]
    if m.sum() < 20:
        return None
    x = Lref["phrase"][:n][m] - Lref["phrase"][:n][m].mean()
    y = L["phrase"][:n][m] - L["phrase"][:n][m].mean()
    return float(np.dot(x, y) / max(np.dot(x, x), 1e-12))


def tone_cmp(L, Lref):
    if not L or not Lref:
        return None, None
    n = min(len(L["v"]), len(Lref["v"]))
    m = L["v"][:n] & Lref["v"][:n]
    if m.sum() < 20 or np.std(L["tone"][:n][m]) < 1e-9 or np.std(Lref["tone"][:n][m]) < 1e-9:
        return None, None
    return (float(np.corrcoef(L["tone"][:n][m], Lref["tone"][:n][m])[0, 1]),
            float(np.std(L["tone"][:n][m]) / np.std(Lref["tone"][:n][m])))


def delivery(f, t):
    """|realised - target| in st on frames voiced in both."""
    n = min(len(f), len(t))
    f, t = f[:n], t[:n]
    m = (f > 0) & (t > 0)
    if m.sum() < 10:
        return None, None
    e = np.abs(12 * np.log2(f[m] / t[m]))
    return float(np.median(e)), float(np.mean(e > 1.0))


def reg_shift(f, fc):
    """Realised register shift (st): median frame-wise 12*log2(item / c100) on frames voiced in both."""
    n = min(len(f), len(fc))
    f, fc = f[:n], fc[:n]
    m = (f > 0) & (fc > 0)
    if m.sum() < 20:
        return None
    return float(np.median(12 * np.log2(f[m] / fc[m])))


def praat_f0(y, sr, lo, hi, n):
    """Second-opinion tracker: Praat autocorrelation pitch, 5-ms step, onto the harvest frame grid."""
    snd = P3.parselmouth.Sound(np.asarray(y, np.float64), sampling_frequency=sr)
    p = snd.to_pitch_ac(time_step=P1.FRAME_MS / 1000.0, pitch_floor=lo, pitch_ceiling=hi)
    out = np.zeros(n)
    idx = np.clip(np.round(np.asarray(p.xs()) / (P1.FRAME_MS / 1000.0)).astype(int), 0, n - 1)
    out[idx] = p.selected_array["frequency"]
    return out


def ref_index(n, rate, m):
    """c100-grid frame i -> item frame round(i / rate) (t_item = t_c100 / rate); ok = inside the item."""
    idx = np.round(np.arange(n) / rate).astype(int)
    return np.minimum(idx, max(m - 1, 0)), idx < m


def to_ref_grid(f, rate, n):
    """An item track (any uniform grid shared with c100) -> c100's time axis, nearest frame, length n."""
    f = np.asarray(f, float)
    idx, ok = ref_index(n, rate, len(f))
    out = np.zeros(n)
    out[ok] = f[idx[ok]]
    return out


def remap_G(G, rate, T):
    out = {}
    for k, v in G.items():
        v = np.asarray(v)
        idx, ok = ref_index(T, rate, len(v))
        o = np.zeros(T, v.dtype)
        o[ok] = v[idx[ok]]
        out[k] = o
    return out


def fit_len(y, n):
    y = np.asarray(y, np.float32)
    d = len(y) - n
    if d > 0:
        y = y[:n]
    elif d < 0:
        y = np.concatenate([y, np.zeros(-d, np.float32)])
    return y, int(d)


def db(x):
    return 20.0 * math.log10(max(float(x), 1e-12))


def rms(y):
    y = np.asarray(y, np.float64)
    return float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0


def dyn_clip_frac(y, sr, g, si, ei):
    """How often wave_dyn's +/-12 dB gain clip engages (same env/median math as P3.wave_dyn)."""
    env = P1.rms_env(y, sr)
    edb = 20 * np.log10(np.maximum(env, 1e-9))
    m = edb > (edb.max() - 40)
    med = float(np.median(edb[m])) if m.any() else float(edb.max())
    hit = np.abs((g - 1.0) * (edb - med)) > 12.0
    return dict(speech=rnd(hit[si:ei].mean(), 4), whole=rnd(hit.mean(), 4))


def speech_weight(n, sr, ips_idx, si, ei):
    """Sample weight for the speech-only dynamics: 1 inside each frozen c100 IP intersected with the c100 speech
    extent, 20-ms raised-cosine ramps INSIDE each edge, 0 elsewhere. Also the hard (ramp-free) mask."""
    hop = int(sr * FV.E_HOP)
    w = np.zeros(n)
    hard = np.zeros(n, bool)
    xf = int(round(sr * XFADE_MS / 1000.0))
    spans = [(a * hop, b * hop) for a, b in ips_idx] or [(si, ei)]
    for a, b in spans:
        s0, s1 = max(si, a), min(ei, b, n)
        if s1 - s0 < 8:
            continue
        w[s0:s1] = 1.0
        hard[s0:s1] = True
        m = min(xf, (s1 - s0) // 2)
        if m > 0:
            r = 0.5 - 0.5 * np.cos(np.pi * (np.arange(m) + 0.5) / m)
            w[s0:s0 + m] = r
            w[s1 - m:s1] = r[::-1]
    return w, hard


def dyn_in_speech(base, sr, g, w, si, ei):
    """P3.wave_dyn on the base, blended in only where w > 0; the dynamics part is scaled so the blended speech extent
    has RMS = TARGET_DBFS, while w == 0 samples stay the base's samples exactly."""
    b = np.asarray(base, np.float64)
    yd = P3.wave_dyn(np.asarray(base, np.float32), sr, g).astype(np.float64)
    s = 1.0
    for _ in range(12):
        mix = b + w * (s * yd - b)
        s *= 10 ** ((TARGET_DBFS - db(rms(mix[si:ei]))) / 20.0)
    mix = b + w * (s * yd - b)
    return mix.astype(np.float32), db(s)


def env_sd_ref(y, ref_db, sr, rate, si, ei, ip_frames=None):
    """h2_prosody_probe2.measures(mask_ref=c100)'s env SD (30-ms RMS env, 5-ms frames, frames within 40 dB of the c100
    max) with the item's envelope mapped onto c100's time axis; also restricted to the c100 speech extent, and to the
    frozen c100 IP frames (the samples the speech-only dynamics actually touch)."""
    step = max(1, int(sr * P1.FRAME_MS / 1000.0))
    e = 20 * np.log10(np.maximum(P1.rms_env(y, sr)[::step], 1e-9))
    n = len(ref_db)
    idx, ok = ref_index(n, rate, len(e))
    keep = (ref_db > ref_db.max() - 40.0) & ok
    fr = np.arange(n) * step
    ext = keep & (fr >= si) & (fr < ei)
    ipk = (keep & ip_frames[:n]) if ip_frames is not None else np.zeros(n, bool)
    em = e[idx]
    return (float(np.std(em[keep])) if keep.sum() > 4 else None,
            float(np.std(em[ext])) if ext.sum() > 4 else None,
            float(np.std(em[ipk])) if ipk.sum() > 4 else None)


def realised_scale(ec, yi, sr, si, ei):
    """Time scale s (t_item = t_c100 / s) maximising the correlation of the 10-ms dB envelopes over the c100 speech
    extent; grid 0.800..1.100 step 0.002. Returns s, r at s, r at s = 1."""
    e2 = FV.energy_db(yi, sr)
    tc = np.arange(len(ec)) * FV.E_HOP
    t2 = np.arange(len(e2)) * FV.E_HOP
    seg = (tc >= si / sr) & (tc < ei / sr)
    if seg.sum() < 20:
        return None, None, None
    x0 = ec[seg]
    best_s, best_r, r1 = None, -2.0, None
    for s in np.round(np.arange(0.800, 1.1001, 0.002), 3):
        x = np.interp(tc[seg] / s, t2, e2)
        if np.std(x) < 1e-9:
            continue
        r = float(np.corrcoef(x0, x)[0, 1])
        if abs(s - 1.0) < 1e-9:
            r1 = r
        if r > best_r:
            best_s, best_r = float(s), r
    return best_s, best_r, r1


def nucleus_align(nuc_c, nuc_i, rate, tol=0.08):
    """c100 nuclei (s) mapped by 1/rate, matched to the nearest item nucleus within tol."""
    nuc_c, nuc_i = np.asarray(nuc_c, float), np.asarray(nuc_i, float)
    if nuc_c.size == 0 or nuc_i.size == 0:
        return dict(n_c100=int(nuc_c.size), n_item=int(nuc_i.size), n_matched=0), []
    pairs = []
    for t in nuc_c:
        j = int(np.argmin(np.abs(nuc_i - t / rate)))
        if abs(nuc_i[j] - t / rate) <= tol:
            pairs.append((float(t), float(nuc_i[j])))
    out = dict(n_c100=int(nuc_c.size), n_item=int(nuc_i.size), n_matched=len(pairs))
    if pairs:
        P = np.array(pairs)
        off = (P[:, 1] - P[:, 0] / rate) * 1000.0
        out["median_abs_offset_ms"] = rnd(np.median(np.abs(off)), 1)
        out["median_offset_ms"] = rnd(np.median(off), 1)
        if len(pairs) >= 3 and np.ptp(P[:, 0]) > 0.2:
            out["slope_item_on_c100"] = rnd(np.polyfit(P[:, 0], P[:, 1], 1)[0], 4)
    return out, pairs


def syl_steps(f0, syls, syl_ip):
    """Adjacent-syllable register steps: syllable register = mean st over its voiced 5-ms frames (>= 3), missing
    syllables interpolated; returns RMS of steps between consecutive syllables of the same IP, and over all adjacent
    syllables (review_phonetics/probe.py contrast())."""
    f0 = np.asarray(f0, float)
    reg = []
    for a, b in syls:
        i0 = int(round(a * FV.E_HOP / FV.F0_HOP))
        i1 = int(round(b * FV.E_HOP / FV.F0_HOP))
        v = f0[i0:i1]
        v = v[v > 0]
        reg.append(float(np.mean(12 * np.log2(v))) if v.size >= 3 else np.nan)
    reg = np.array(reg)
    ok = np.isfinite(reg)
    if ok.sum() < 3:
        return None, None
    r = np.interp(np.arange(len(reg)), np.flatnonzero(ok), reg[ok])
    ips = np.asarray(syl_ip)
    d = np.diff(r)
    same = ips[1:] == ips[:-1]
    within = float(np.sqrt(np.mean(d[same] ** 2))) if same.sum() >= 2 else None
    alladj = float(np.sqrt(np.mean(d ** 2))) if d.size >= 2 else None
    return within, alladj


def human_cues(y, sr):
    """human/acoustic.one()'s scalars, re-expressed verbatim (one() writes npz into the human lane): 16-kHz resample,
    extract_prosody.analyse + nuclei, acoustic.speech_span (imported), same formulas."""
    y = np.asarray(y, np.float64)
    SR = int(HA.SR)
    if int(sr) != SR:
        g = gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g)
    A = EP.analyse(y.astype(np.float32), SR)
    A["nuclei_s"] = EP.nuclei(A["rms_db_10ms"], A["f0_hz_5ms"])
    f0 = np.asarray(A["f0_hz_5ms"], float)
    rdb = np.asarray(A["rms_db_10ms"], float)
    t_on, t_off = HA.speech_span(rdb, f0)
    if t_on is None:
        return None
    hop, half = int(SR * 0.010), int(SR * 0.015)
    n = int(len(y) / hop) + 1
    c2 = np.concatenate([[0.0], np.cumsum(y ** 2)])
    cen = np.arange(n) * hop
    a_ = np.clip(cen - half, 0, len(y))
    b_ = np.clip(cen + half, 0, len(y))
    rr = np.sqrt(np.maximum((c2[b_] - c2[a_]) / np.maximum(b_ - a_, 1), 0.0))
    dbfs = 20 * np.log10(np.maximum(rr, 1e-9))
    lf0, voiced = P1.interp_unvoiced(f0)
    t5 = np.arange(len(f0)) * HA.FRAME_MS / 1000.0
    in5 = (t5 >= t_on) & (t5 <= t_off)
    S = {}
    if lf0 is not None and (voiced & in5).sum() >= 20:
        med = float(np.median(lf0[voiced & in5]))
        st = (lf0 - med) * 12.0 / np.log(2.0)
        win = max(3, int(round(P1.PHRASE_SMOOTH_MS / HA.FRAME_MS)) | 1)
        phrase = P1.smooth(st, win)
        vv = voiced & in5
        fv = f0[vv]
        p10, p90 = np.percentile(fv, [10, 90])
        S.update(f0_med_hz=float(np.exp(med)), f0_span_st=float(12 * np.log2(p90 / p10)),
                 phrase_span_st=float(np.percentile(phrase[vv], 95) - np.percentile(phrase[vv], 5)),
                 tone_sd_st=float(np.std((st - phrase)[vv])), voiced_frac=float(vv.sum() / max(in5.sum(), 1)))
    t10 = np.arange(len(rdb)) * 0.010
    in10 = (t10 >= t_on) & (t10 <= t_off)
    g = rdb[in10]
    g = g[g > g.max() - 40.0]
    gf = dbfs[:len(rdb)][in10]
    gf = gf[gf > gf.max() - 40.0]
    nuc = np.asarray(A["nuclei_s"], float)
    nuc_in = nuc[(nuc >= t_on) & (nuc <= t_off)]
    S.update(t_on=float(t_on), t_off=float(t_off), speech_dur_s=float(t_off - t_on),
             env_sd_db=float(np.std(g)), env_range_db=float(np.percentile(g, 90) - np.percentile(g, 10)),
             level_dbfs=float(np.median(gf)), n_nuclei=int(nuc_in.size),
             nuc_rate_sps=float(nuc_in.size / max(t_off - t_on, 1e-3)))
    return {k: rnd(v, 4) for k, v in S.items()}


def human_deltas(H, Hc):
    if not H or not Hc:
        return None
    d = {}
    if H.get("level_dbfs") is not None and Hc.get("level_dbfs") is not None:
        d["d_level_db"] = rnd(H["level_dbfs"] - Hc["level_dbfs"], 3)
    if H.get("f0_med_hz") and Hc.get("f0_med_hz"):
        d["d_register_st"] = rnd(12 * math.log2(H["f0_med_hz"] / Hc["f0_med_hz"]), 3)
    for k in ("speech_dur_s", "phrase_span_st", "env_sd_db", "nuc_rate_sps", "f0_span_st"):
        if H.get(k) and Hc.get(k):
            d["ratio_" + k] = rnd(H[k] / Hc[k], 4)
    if H.get("env_sd_db") is not None and Hc.get("env_sd_db") is not None:
        d["d_env_sd_db"] = rnd(H["env_sd_db"] - Hc["env_sd_db"], 3)
    return d


def lilthead_view(A, G, Gc):
    """The LiltHead model's own input tracks (its harvest, its p99 energy reference), 60 fps (rate items already mapped
    onto c100's frames). realised_k / realised_g are against c100's tracks on c100's voiced / vad frames."""
    v = G["voiced"] > 0.5
    vad = G["vad"] > 0.5
    ph = G["phrase_st"][v]
    out = dict(phrase_span_st=rnd(np.percentile(ph, 90) - np.percentile(ph, 10), 3) if ph.size > 5 else None,
               tone_sd_st=rnd(np.std(G["tone_st"][v]), 3) if v.sum() > 5 else None,
               logE_sd_db_vad=rnd(np.std(G["logE_db"][vad]), 3) if vad.sum() > 5 else None,
               vad_frac=rnd(vad.mean(), 4), voiced_frac=rnd(v.mean(), 4),
               n_nuclei=int(len(A["nuclei_s"])),
               rate_sps_vad=rnd(np.mean(G["rate_sps"][vad]), 3) if vad.sum() else None,
               prominence_sd_vad=rnd(np.std(G["prominence"][vad]), 3) if vad.sum() > 5 else None)
    if Gc is not None:
        n = min(len(G["phrase_st"]), len(Gc["phrase_st"]))
        m = v[:n] & (Gc["voiced"][:n] > 0.5)
        if m.sum() > 20:
            x = Gc["phrase_st"][:n][m] - Gc["phrase_st"][:n][m].mean()
            y = G["phrase_st"][:n][m] - G["phrase_st"][:n][m].mean()
            out["realised_k"] = rnd(np.dot(x, y) / max(np.dot(x, x), 1e-12), 3)
            if np.std(G["tone_st"][:n][m]) > 1e-9 and np.std(Gc["tone_st"][:n][m]) > 1e-9:
                out["tone_r_vs_c100"] = rnd(np.corrcoef(G["tone_st"][:n][m], Gc["tone_st"][:n][m])[0, 1], 3)
        mv = Gc["vad"][:n] > 0.5
        if mv.sum() > 20 and np.std(Gc["logE_db"][:n][mv]) > 1e-9:
            out["realised_g"] = rnd(np.std(G["logE_db"][:n][mv]) / np.std(Gc["logE_db"][:n][mv]), 3)
    return out


def spans_s(pairs, scale=1.0):
    return [[rnd(a * FV.E_HOP * scale, 3), rnd(b * FV.E_HOP * scale, 3)] for a, b in pairs]


# --------------------------------------------------------------------------- build one cell
def build_cell(item):
    cell = item["cell"]
    t0 = time.time()
    out = dict(cell=cell, lang=item["lang"], label=item["label"], source_wav=item["wav"],
               source_wav_win=item["wav"].replace("/mnt/d/", "D:/"), text=item.get("text", ""),
               speaker=item.get("speaker"), tier=item.get("tier"), source=item.get("source"),
               items=[], failures=[], notes=[])
    y_raw, sr = sf.read(item["wav"], dtype="float32")
    if y_raw.ndim > 1:
        y_raw = y_raw.mean(axis=1)
    n0 = len(y_raw)
    out.update(sr=int(sr), n_samples=int(n0), dur_s=rnd(n0 / sr, 4), source_sha256=sha256(item["wav"]),
               raw_rms_dbfs=rnd(db(rms(y_raw)), 2), raw_peak=rnd(np.abs(y_raw).max(), 4))

    y0 = P3.rms_normalize(P3.highpass(y_raw, sr), P3.TARGET_RMS_DBFS)
    A = P2.analyse_constrained(y0, sr)
    fmin, fmax = float(A["f0_lo"]), float(A["f0_hi"])
    Af0 = np.asarray(A["f0"], float)
    v_ref = Af0 > 0
    n_fr = len(Af0)
    out.update(analysis_f0_range_hz=[rnd(fmin, 1), rnd(fmax, 1)], analysis_f0_median_hz=rnd(A["f0_med"], 1),
               analysis_voiced_frac=rnd(v_ref.mean(), 3))

    y_null = P3.psola_resynth(y0, sr, fmin, fmax)
    if y_null is None:
        out["failures"].append("PSOLA null failed -- cell skipped")
        return out
    y_null, dn = fit_len(y_null, n0)
    if dn:
        out["notes"].append("PSOLA null length differed from source by %+d samples; trimmed/padded" % dn)
    y_null = P3.match_level(y_null, y0, sr)

    s_sec, e_sec = VFG.speech_edges(y_null, sr)
    si, ei = int(round(s_sec * sr)), int(round(e_sec * sr))
    out.update(speech_extent_s=[rnd(s_sec, 3), rnd(e_sec, 3)], speech_extent_samples=[si, ei],
               level_rule=("one scalar gain: RMS over the c100 speech extent [%d:%d] (x 1/rate on rate items) = %.1f dBFS"
                           % (si, ei, TARGET_DBFS)))

    def ext(rate, n):
        return min(n, int(round(si / rate))), min(n, int(round(ei / rate)))

    def to_target(y, rate=1.0):
        sa, ea = ext(rate, len(y))
        return (np.asarray(y, np.float64) * 10 ** ((TARGET_DBFS - db(rms(y[sa:ea]))) / 20.0)).astype(np.float32)

    A_lay = layers(Af0)
    a_pspan = pspan(A_lay)
    tcache = {}

    def target_for(k=1.0, reg=0.0):
        key = (round(k, 4), round(reg, 4))
        if key not in tcache:
            if key == (1.0, 0.0):
                tcache[key] = Af0.copy()
            else:
                _, _, f0t = P1.manipulate(A, k_phrase=k, k_tone=1.0, reg_st=reg)   # target math; WORLD audio discarded
                tcache[key] = np.asarray(f0t, float)
        return tcache[key]

    pcache = {}

    def psola(k=1.0, reg=0.0, rate=1.0, null=False):
        key = ("null", rate) if null else (round(k, 4), round(reg, 4), rate)
        if key not in pcache:
            if null and rate == 1.0:
                pcache[key] = (y_null, 0)
            else:
                y = P3.psola_resynth(y0, sr, fmin, fmax, f0_target=None if null else target_for(k, reg), rate=rate)
                if y is None:
                    pcache[key] = (None, None)
                else:
                    y, d = fit_len(y, int(round(n0 / rate)))
                    pcache[key] = (P3.match_level(y, y0, sr), d)
        return pcache[key]

    written = []          # (rid, family, params, engine, f0_target, build_extra, path, rate)
    finals = {}

    def write(rid, fam, prm, y, engine, f0t, extra, rate=1.0):
        y, peak_pre = P3.peak_guard(np.asarray(y, np.float32))
        extra = dict(extra)
        extra["peak_before_guard"] = rnd(peak_pre, 4)
        extra["peak_guard_engaged"] = bool(peak_pre > GUARD)
        path = "%s/%s__%s.wav" % (WAV_DIR, cell, rid)
        sf.write(path, y, sr, subtype="PCM_16")
        written.append((rid, fam, prm, engine, f0t, extra, path, rate))
        finals[rid] = y
        return y

    c100_final = write("c100", "null", {}, to_target(y_null), "psola_null", Af0, {})

    # ---- frozen structure: ONCE, on the on-disk c100
    c_disk, _ = sf.read(written[-1][6], dtype="float32")
    raw_c = FV.raw_tracks(c_disk, sr)
    prof = FV.profile_from_raw([raw_c])
    S0 = FV.structure(raw_c, prof)
    if not S0["ips"]:
        out["notes"].append("features_v2.structure found no IP on c100: dynamics mask falls back to the speech extent")
    w_dyn, hard_ip = speech_weight(n0, sr, S0["ips"], si, ei)
    pause_mask = np.zeros(n0, bool)
    pause_mask[si:ei] = True
    pause_mask &= ~hard_ip
    ns_c = w_dyn <= 0
    ns_ref_db = db(rms(c100_final[ns_c])) if ns_c.sum() > 0.05 * sr else None
    if ns_ref_db is None:
        out["notes"].append("less than 50 ms of non-speech in c100: non-speech level not pinned")
    out["c100_nonspeech_rms_dbfs"] = rnd(ns_ref_db, 2)

    def weight_for(rate, n):
        if rate == 1.0 and n == n0:
            return w_dyn
        return w_dyn[np.minimum(np.round(np.arange(n) * rate).astype(int), n0 - 1)]

    def pin(y, rate=1.0):
        """Two-gain level rule (smoke 09-14: the one-scalar rule moved non-speech by up to +4.5 dB on k167 and
        +2.6 dB on `more`): non-speech samples (w == 0: outside the frozen c100 IPs, mapped by 1/rate) -> c100's
        non-speech RMS; speech extent RMS -> TARGET_DBFS; the gain moves along the same 20-ms ramps inside the IPs."""
        y = np.asarray(y, np.float64)
        n = len(y)
        w = weight_for(rate, n)
        ns = w <= 0
        sa, ea = ext(rate, n)
        if ns_ref_db is None or ns.sum() < 0.05 * sr:
            return y.astype(np.float32), None
        # PER CONTIGUOUS NON-SPEECH RUN (full build 09-14: one gain for all non-speech left jp_jvnvF2_reg1_sur's
        # lead/trail -2 dB while its internal pauses sat +0.2 dB): each run -> c100's RMS over the same run
        # (mapped by rate); runs < 50 ms take the all-non-speech gain.
        g_all_raw = ns_ref_db - db(rms(y[ns]))
        edges = np.flatnonzero(np.diff(np.concatenate([[0], ns.astype(np.int8), [0]])))
        G = np.full(n, np.nan)
        runs, clipped, tot, acc = [], False, 0, 0.0
        for a_, b_ in zip(edges[::2], edges[1::2]):
            ca, cb = int(round(a_ * rate)), min(n0, int(round(b_ * rate)))
            raw = (db(rms(c100_final[ca:cb])) - db(rms(y[a_:b_]))) if (b_ - a_ >= 0.05 * sr and cb > ca) else g_all_raw
            clipped |= abs(raw) > 12.0
            gdb = float(np.clip(raw, -12.0, 12.0))
            G[a_:b_] = gdb
            runs.append([rnd(a_ / sr, 3), rnd(b_ / sr, 3), rnd(gdb, 2)])
            tot += b_ - a_
            acc += (b_ - a_) * gdb
        known = np.flatnonzero(~np.isnan(G))
        G = np.interp(np.arange(n), known, G[known])      # only the ramp samples inside IP edges use the in-between values
        gns = 10 ** (G / 20.0)
        s = 1.0
        for _ in range(12):
            yy = y * (w * s + (1 - w) * gns)
            s *= 10 ** ((TARGET_DBFS - db(rms(yy[sa:ea]))) / 20.0)
        yy = y * (w * s + (1 - w) * gns)
        return yy.astype(np.float32), dict(nonspeech_gain_db=rnd(acc / max(tot, 1), 3), nonspeech_runs_s_gain_db=runs,
                                           speech_gain_db=rnd(db(s), 3), clipped_at_12db=bool(clipped))

    # the conditioned source without the PSOLA pass: the null-floor check, not a rung
    y_src, pin_src = pin(to_target(y0))
    write("src", "source", {}, y_src, "none (HPF + level only) + non-speech pinned to c100", Af0, {"level_pin": pin_src})

    for rid, fam, prm in RUNGS[1:]:
        k, g = prm.get("k", 1.0), prm.get("g")
        reg, rate = prm.get("reg_st", 0.0), prm.get("rate", 1.0)
        extra = {}
        if fam == "level":
            y = c100_final.astype(np.float64) * 10 ** (LEVEL_DB / 20.0)
            engine, f0t = "c100 on disk x gain %+.0f dB" % LEVEL_DB, Af0
        elif rid == "effort":
            if "effort0" not in finals:
                out["failures"].append("effort: no effort0 to scale")
                continue
            y = finals["effort0"].astype(np.float64) * 10 ** (EFFORT_LEVEL_DB / 20.0)
            engine, f0t = "effort0 on disk x gain %+.0f dB" % EFFORT_LEVEL_DB, target_for(1.0, reg)
        else:
            if fam == "dyn":
                base, engine, f0t = c100_final.astype(np.float64), "c100 (psola_null)", Af0
            else:
                null = fam == "rate"
                yp, d = psola(k=k, reg=reg, rate=rate, null=null)
                if yp is None:
                    out["failures"].append("%s: PSOLA failed" % rid)
                    continue
                if d:
                    extra["len_fix_samples"] = d
                base = to_target(yp, rate)
                f0t = Af0 if null else target_for(k, reg)
                engine = ("psola_null" if null else "psola_target") + ("+rate %.2f" % rate if rate != 1.0 else "")
            if g is not None:
                extra["dyn_gain_clip_frac"] = dyn_clip_frac(base, sr, g, si, ei)
                y, inside_db = dyn_in_speech(base, sr, g, w_dyn, si, ei)
                extra["dyn_inside_scale_db"] = rnd(inside_db, 3)
                engine += "+wave_dyn(speech only: frozen c100 IPs within the speech extent, %g-ms ramps)" % XFADE_MS
            else:
                y = base
            y, extra["level_pin"] = pin(y, rate)
            engine += " + non-speech pinned to c100"
        write(rid, fam, prm, y, engine, f0t, extra, rate)

    # ----------------------------------------------------------------------- verification (read back)
    disk = {}
    for rid, fam, prm, engine, f0t, extra, path, rate in written:
        yy, srr = sf.read(path, dtype="float32")
        disk[rid] = yy
        if srr != sr:
            out["failures"].append("%s: read-back sr %d != %d" % (rid, srr, sr))
    c = disk["c100"]
    allv = np.concatenate([w[4][w[4] > 0] for w in written])
    trk_lo = max(45.0, float(allv.min()) / (2 ** (5 / 12.0)))
    trk_hi = min(950.0, float(allv.max()) * (2 ** (5 / 12.0)))
    out["tracker"] = ("PRIMARY pyworld harvest via h2_prosody_probe2.track_against_target, ONE range for every item "
                      "of the cell = union of all rung targets +/-5 st = [%.1f, %.1f] Hz, clean_f0, 5-ms frames; "
                      "rate items mapped onto c100's frames (t_item = t_c100 / rate) BEFORE the frame set is frozen "
                      "to the analysis contour's voiced frames. SECOND OPINION Praat To Pitch (ac), same range and "
                      "frame set; THIRD pyin (verify_pyin.py)." % (trk_lo, trk_hi))
    out["tracker_range_hz"] = [rnd(trk_lo, 1), rnd(trk_hi, 1)]

    step = max(1, int(sr * P1.FRAME_MS / 1000.0))
    ref_db = 20 * np.log10(np.maximum(P1.rms_env(c, sr)[::step], 1e-9))
    ip_fr = hard_ip[np.minimum(np.arange(len(ref_db)) * step, n0 - 1)]
    ec10 = FV.energy_db(c, sr)
    c_out = np.concatenate([c[:si], c[ei:]])
    nuc_c_s = np.asarray(S0["nuclei"], float) * FV.E_HOP
    syls, syl_ip = S0["syllables"], S0["syl_ip"]

    meas = {}
    order = ["c100"] + [w[0] for w in written if w[0] != "c100"]
    wmap = {w[0]: w for w in written}
    for rid in order:
        _, fam, prm, engine, f0t, extra, path, rate = wmap[rid]
        yy = disk[rid]
        n_it = int(len(yy) / sr / (P1.FRAME_MS / 1000.0)) + 1
        f0n = P2.track_against_target(yy, sr, allv)
        f0c = to_ref_grid(P1.clean_f0(f0n), rate, n_fr)
        f0p = to_ref_grid(praat_f0(yy, sr, trk_lo, trk_hi, n_it), rate, n_fr)
        mm = P2.measures(yy, sr, f0n, mask_ref=c) if rate == 1.0 else P2.measures(yy, sr, f0n)
        sa, ea = ext(rate, len(yy))
        outside = np.concatenate([yy[:sa], yy[ea:]])
        EA, EG = EP.features(yy, sr, 60)
        Si = FV.structure(FV.raw_tracks(yy, sr), prof)
        esd, esd_x, esd_ip = env_sd_ref(yy, ref_db, sr, rate, si, ei, ip_fr)
        M = dict(Lfix=layers(restrict(f0c, v_ref)), Lown=layers(f0c), Lpr=layers(restrict(f0p, v_ref)),
                 f0c=f0c, f0p=f0p, mm=mm, EA=EA, EG=EG, S=Si, env=esd, env_x=esd_x, env_ip=esd_ip,
                 floor_rel_db=(db(rms(outside)) - db(rms(yy[sa:ea]))) if len(outside) > 0.05 * sr else None,
                 nonspeech_vs_c100=(db(rms(outside)) - db(rms(c_out))) if (len(outside) > 0.05 * sr and len(c_out) > 0.05 * sr) else None,
                 pause_vs_c100=(db(rms(yy[pause_mask])) - db(rms(c[pause_mask]))) if (rate == 1.0 and pause_mask.sum() > 0.05 * sr) else None,
                 human=human_cues(yy, sr), scale=realised_scale(ec10, yy, sr, si, ei),
                 align=nucleus_align(nuc_c_s, np.asarray(Si["nuclei"], float) * FV.E_HOP, rate),
                 steps_pr=syl_steps(restrict(f0p, v_ref), syls, syl_ip),
                 steps_hv=syl_steps(restrict(f0c, v_ref), syls, syl_ip))
        meas[rid] = M
    C = meas["c100"]
    Tc = len(C["EG"]["phrase_st"])
    for rid in order:
        if wmap[rid][7] != 1.0:
            meas[rid]["EG"] = remap_G(meas[rid]["EG"], wmap[rid][7], Tc)

    mc = C["mm"]
    c_ps = pspan(C["Lfix"])
    c_env = mc.get("env_sd_db")
    c_env_ref, c_env_x = C["env"], C["env_x"]
    c_rms_sx = db(rms(c[si:ei]))
    a_steps = syl_steps(restrict(Af0, v_ref), syls, syl_ip)

    L_h2 = layers(P1.clean_f0(P2.track_against_target(c, sr, Af0)))
    out["c100_phrase_span_st"] = rnd(c_ps, 3)
    out["c100_phrase_span_st_own_voicing"] = rnd(pspan(C["Lown"]), 3)
    out["c100_phrase_span_st_h2_own_range_tracker"] = rnd(pspan(L_h2), 3)
    out["c100_phrase_span_st_praat"] = rnd(pspan(C["Lpr"]), 3)
    out["analysis_phrase_span_st"] = rnd(a_pspan, 3)
    out["c100_env_sd_db"] = c_env
    out["c100_adj_step_rms_st_praat"] = rnd(C["steps_pr"][0], 3)
    out["analysis_adj_step_rms_st"] = rnd(a_steps[0], 3)

    # ---- the frozen structure file
    text = item.get("text", "") or ""
    if item["lang"] == "cn":
        n_txt = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        txt_src = "count of CJK characters in pool_set.json text %r" % text
    elif item["lang"] in TEXT_SYL:
        n_txt, txt_src = TEXT_SYL[item["lang"]]
    else:
        n_txt, txt_src = None, "no transcript found for JVNV (searched ${CORPUS_ROOT}/JVNV to depth 3) [U]"
    ip_dur = sum(b - a for a, b in S0["ips"]) * FV.E_HOP
    ips = S0["ips"]
    struct = dict(
        cell=cell, lang=item["lang"], label=item["label"], text=text, computed_on="c100",
        frozen_rule=("computed ONCE on the on-disk c100 (features_v2.structure with a SpeakerProfile from c100 alone); "
                     "every later analysis of every rung reuses these spans -- on rate rungs the *_s spans mapped by "
                     "1/rate (rate_maps) -- and never re-segments a rung"),
        c100_path=wmap["c100"][6], c100_path_win=wmap["c100"][6].replace("/mnt/d/", "D:/"), c100_sha256=sha256(wmap["c100"][6]),
        source_wav=item["wav"], code_sha256={k: sha256(p) for k, p in CODE_FILES.items()},
        grid=dict(energy_hop_s=FV.E_HOP, f0_hop_s=FV.F0_HOP, sr=int(sr),
                  note="*_idx = 10-ms energy-grid indices (end-exclusive); *_s = seconds on c100's time axis"),
        profile=prof.as_dict(),
        speech_extent_s=[rnd(s_sec, 3), rnd(e_sec, 3)], speech_extent_samples=[si, ei],
        ips_idx=[list(map(int, p)) for p in ips], ips_s=spans_s(ips),
        pauses_s=[[rnd(ips[i][1] * FV.E_HOP, 3), rnd(ips[i + 1][0] * FV.E_HOP, 3)] for i in range(len(ips) - 1)],
        nuclei_idx=[int(p) for p in S0["nuclei"]], nuclei_s=[rnd(p * FV.E_HOP, 3) for p in S0["nuclei"]],
        syllables_idx=[list(map(int, p)) for p in syls], syllables_s=spans_s(syls), syl_ip=[int(q) for q in syl_ip],
        counts=dict(n_ip=len(ips), n_nuclei=len(S0["nuclei"]), n_syl=len(syls),
                    syl_rate_sps=rnd(len(syls) / max(ip_dur, 1e-3), 3), ip_speech_s=rnd(ip_dur, 3)),
        text_syllables=n_txt, text_syllables_source=txt_src,
        n_syl_minus_text=(len(syls) - n_txt) if n_txt else None,
        v1_nuclei_s=[rnd(t, 3) for t in C["EA"]["nuclei_s"]],
        v1_nuclei_note="lilthead/extract_prosody.nuclei on c100 (LiltHead v1's own detector; p99-relative gate)",
        dyn_mask=dict(rule="w = 1 inside each c100 IP intersected with the c100 speech extent, %g-ms raised-cosine ramps inside each edge" % XFADE_MS,
                      speech_samples_weighted=int((w_dyn > 0).sum()), internal_pause_samples=int(pause_mask.sum())),
        rate_maps={})
    for rid, fam, prm, engine, f0t, extra, path, rate in written:
        if rate != 1.0:
            sc = 1.0 / rate
            struct["rate_maps"][rid] = dict(rate=rate, rule="t_item_s = t_c100_s / rate",
                                           speech_extent_s=[rnd(s_sec * sc, 3), rnd(e_sec * sc, 3)],
                                           ips_s=spans_s(ips, sc), syllables_s=spans_s(syls, sc),
                                           nuclei_s=[rnd(p * FV.E_HOP * sc, 3) for p in S0["nuclei"]])
    struct["rung_resegmentation"] = {rid: [len(meas[rid]["S"]["ips"]), len(meas[rid]["S"]["nuclei"]), len(meas[rid]["S"]["syllables"])]
                                     for rid in order}
    struct["rung_resegmentation_note"] = ("[n_ip, n_nuclei, n_syl] if features_v2.structure were re-run on each rung with the frozen "
                                          "c100 profile -- shown ONLY to document why the structure is frozen; do not use")
    spath = "%s/%s.json" % (STRUCT_DIR, cell)
    jdump(struct, spath)
    out["structure_file"] = spath
    out["structure_counts"] = struct["counts"]
    out["text_syllables"] = n_txt

    Hc = C["human"]
    c_cnt = [len(ips), len(S0["nuclei"]), len(syls)]
    for rid in order:
        _, fam, prm, engine, f0t, extra, path, rate = wmap[rid]
        yy = disk[rid]
        M = meas[rid]
        mm = M["mm"]
        T = layers(f0t)
        tgt_ps = pspan(T)
        tm = P1.measures_f0_only(f0t)
        rec = dict(id="%s__%s" % (cell, rid), cell=cell, lang=item["lang"], label=item["label"], rung=rid,
                   family=fam, role=ROLE.get(rid, "rung"), k=prm.get("k", 1.0), g=prm.get("g", 1.0),
                   level_db=prm.get("level_db", 0.0), reg_st=prm.get("reg_st", 0.0), rate=rate,
                   k_tone=1.0, engine=engine, path=path, path_win=path.replace("/mnt/d/", "D:/"), sr=int(sr))
        if fam == "source":
            rec.update(k=None, g=None, level_db=None, reg_st=None)
        ps = pspan(M["Lfix"])
        tr_h, tsd_h = tone_cmp(M["Lfix"], C["Lfix"])
        tr_p, tsd_p = tone_cmp(M["Lpr"], C["Lpr"])
        dh = delivery(M["f0c"], np.asarray(f0t)[:n_fr])
        dp = delivery(M["f0p"], np.asarray(f0t)[:n_fr])
        n_exp = int(round(n0 / rate))
        env = M["env"] if rate != 1.0 else mm.get("env_sd_db")
        sa, ea = ext(rate, len(yy))
        s_best, r_best, r_one = M["scale"]
        al, pairs = M["align"]
        stp, stp_all = M["steps_pr"]
        sth, _ = M["steps_hv"]
        realised = dict(
            n_samples=int(len(yy)), expected_n_samples=n_exp, len_ok=bool(abs(len(yy) - n_exp) <= 1),
            same_len_as_c100=bool(len(yy) == len(c)), dur_s=rnd(len(yy) / sr, 4),
            realised_k=rnd(slope(M["Lfix"], C["Lfix"]), 3),
            realised_k_praat=rnd(slope(M["Lpr"], C["Lpr"]), 3),
            phrase_span_st=rnd(ps, 3),
            span_ratio=rnd(ps / c_ps, 3) if (ps is not None and c_ps) else None,
            d_phrase_span_st=rnd(ps - c_ps, 3) if (ps is not None and c_ps is not None) else None,
            phrase_span_st_own_voicing=rnd(pspan(M["Lown"]), 3),
            phrase_span_st_praat=rnd(pspan(M["Lpr"]), 3),
            tone_r_vs_c100=rnd(tr_h, 3), tone_sd_ratio_vs_c100=rnd(tsd_h, 3),
            tone_r_vs_c100_praat=rnd(tr_p, 3), tone_sd_ratio_vs_c100_praat=rnd(tsd_p, 3),
            delivery_err_med_st=rnd(dh[0], 3), delivery_frac_gt1st=rnd(dh[1], 3),
            delivery_err_med_st_praat=rnd(dp[0], 3), delivery_frac_gt1st_praat=rnd(dp[1], 3),
            register_st_harvest=rnd(reg_shift(restrict(M["f0c"], v_ref), restrict(C["f0c"], v_ref)), 3),
            register_st_praat=rnd(reg_shift(restrict(M["f0p"], v_ref), restrict(C["f0p"], v_ref)), 3),
            f0_span_st=mm.get("f0_span_st"),
            d_f0_span_st=rnd((mm.get("f0_span_st") or 0) - (mc.get("f0_span_st") or 0), 2),
            f0_median_hz=mm.get("f0_median_hz"),
            d_f0_median_st=rnd(12 * math.log2(mm["f0_median_hz"] / mc["f0_median_hz"]), 3)
            if (mm.get("f0_median_hz") and mc.get("f0_median_hz")) else None,
            voiced_frac=mm.get("voiced_frac"), voiced_frac_praat=rnd(np.mean(M["f0p"] > 0), 3),
            env_sd_db=rnd(env, 2) if env is not None else None,
            env_sd_method=("h2_prosody_probe2.measures(mask_ref=c100)" if rate == 1.0 else
                           "envelope mapped onto c100 frames, c100 gate (env_sd_ref)"),
            realised_g=rnd(env / c_env, 3) if (env and c_env) else None,
            env_sd_db_ref=rnd(M["env"], 3), env_sd_db_ref_extent=rnd(M["env_x"], 3),
            realised_g_extent=rnd(M["env_x"] / c_env_x, 3) if (M["env_x"] and c_env_x) else None,
            env_sd_db_ref_ip=rnd(M["env_ip"], 3),
            realised_g_ip=rnd(M["env_ip"] / C["env_ip"], 3) if (M["env_ip"] and C["env_ip"]) else None,
            dyn_range_db=mm.get("dyn_range_db"),
            d_med_level_db=rnd(mm["med_level_db"] - mc["med_level_db"], 2)
            if (mm.get("med_level_db") is not None and mc.get("med_level_db") is not None) else None,
            rms_speech_dbfs=rnd(db(rms(yy[sa:ea])), 2),
            d_rms_speech_db=rnd(db(rms(yy[sa:ea])) - c_rms_sx, 2),
            rms_whole_dbfs=rnd(db(rms(yy)), 2),
            peak_dbfs=rnd(db(np.abs(yy).max()), 2),
            n_full_scale_samples=int(np.sum(np.abs(yy) >= 32767 / 32768.0)),
            floor_rel_db=rnd(M["floor_rel_db"], 2),
            nonspeech_rms_vs_c100_db=rnd(M["nonspeech_vs_c100"], 2),
            pause_rms_vs_c100_db=rnd(M["pause_vs_c100"], 2),
            realised_time_scale_env=rnd(s_best, 3), env_corr_at_scale=rnd(r_best, 4), env_corr_at_1=rnd(r_one, 4),
            nucleus_align=al,
            adj_step_rms_st_praat=rnd(stp, 3), adj_step_ratio_praat=rnd(stp / C["steps_pr"][0], 3) if (stp and C["steps_pr"][0]) else None,
            adj_step_rms_all_st_praat=rnd(stp_all, 3),
            adj_step_ratio_harvest=rnd(sth / C["steps_hv"][0], 3) if (sth and C["steps_hv"][0]) else None,
            struct_counts=[len(M["S"]["ips"]), len(M["S"]["nuclei"]), len(M["S"]["syllables"])],
            struct_resegments=bool([len(M["S"]["ips"]), len(M["S"]["nuclei"]), len(M["S"]["syllables"])] != c_cnt),
        )
        dsteps = syl_steps(restrict(np.asarray(f0t, float), v_ref), syls, syl_ip)
        dm = (np.asarray(f0t) > 0) & v_ref
        target = dict(phrase_span_st=rnd(tgt_ps, 3),
                      target_k=rnd(tgt_ps / a_pspan, 3) if (tgt_ps and a_pspan) else None,
                      f0_span_st=tm.get("f0_span_st"), f0_median_hz=tm.get("f0_median_hz"),
                      design_beta=P2.design_beta(Af0, f0t),
                      design_register_st=rnd(np.median(12 * np.log2(np.asarray(f0t)[dm] / Af0[dm])), 3) if dm.sum() > 10 else None,
                      design_adj_step_ratio=rnd(dsteps[0] / a_steps[0], 3) if (dsteps[0] and a_steps[0]) else None,
                      n_frames_capped_900hz=int(np.sum(np.asarray(f0t) >= 899.999)),
                      frac_voiced_outside_analysis_range=rnd(
                          np.mean((f0t[f0t > 0] < fmin) | (f0t[f0t > 0] > fmax)), 4) if np.any(f0t > 0) else None)
        H = M["human"]
        rec.update(build=extra, realised=realised, target=target,
                   lilthead_view=lilthead_view(M["EA"], M["EG"], C["EG"]),
                   human_cues=dict(values=H, vs_c100=human_deltas(H, Hc)))
        if rate != 1.0:
            fr = {}
            for fps in (30, 60):
                T_i, T_c = int(len(yy) / sr * fps), int(n0 / sr * fps)
                fr[str(fps)] = dict(n_item_frames=T_i, n_c100_frames=T_c,
                                    c100_frame_for_item_frame=[int(min(round(j * rate), T_c - 1)) for j in range(T_i)],
                                    item_frame_for_c100_frame=[int(min(round(i / rate), T_i - 1)) for i in range(T_c)])
            tmap = dict(id=rec["id"], cell=cell, rung=rid, rate=rate, sr=int(sr),
                        design="t_c100_s = rate * t_item_s (h2_prosody_probe3.psola_resynth: Praat DurationTier constant 1/rate over the whole file)",
                        n_samples_item=int(len(yy)), n_samples_c100=int(n0),
                        duration_ratio_item_over_c100=rnd(len(yy) / n0, 5),
                        realised_time_scale_env=rnd(s_best, 3), env_corr_at_scale=rnd(r_best, 4), env_corr_at_1=rnd(r_one, 4),
                        nucleus_align=al, nucleus_pairs_s=[[rnd(a, 3), rnd(b, 3)] for a, b in pairs],
                        frames=fr)
            tpath = "%s/%s__%s.json" % (TMAP_DIR, cell, rid)
            jdump(tmap, tpath)
            rec["time_map"] = tpath
        out["items"].append(rec)

    out["flags"] = cell_flags(out)
    out["seconds"] = rnd(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- flags (build stage; verdicts in verify_pyin)
def cell_flags(out):
    F = []
    it = {r["rung"]: r for r in out["items"]}

    def R(rid, key, block="realised"):
        return ((it.get(rid) or {}).get(block) or {}).get(key)

    c_env = out.get("c100_env_sd_db")
    for r in out["items"]:
        re_ = r["realised"]
        if not re_["len_ok"]:
            F.append("DURATION %s: %d samples vs expected %d" % (r["rung"], re_["n_samples"], re_["expected_n_samples"]))
        if r["build"].get("len_fix_samples"):
            F.append("LENFIX %s: PSOLA output %+d samples vs expected, trimmed/padded" % (r["rung"], r["build"]["len_fix_samples"]))
    seq = [("k050", 0.50), ("k075", 0.75), ("c100", 1.00), ("k133", 1.33), ("k167", 1.67)]
    for name, key, block, weak in (("harvest", "realised_k", "realised", True),
                                   ("praat", "realised_k_praat", "realised", False),
                                   ("lilthead", "realised_k", "lilthead_view", False)):
        ks = [R(rid, key, block) for rid, _ in seq]
        if any(x is None for x in ks):
            F.append("SPAN-UNMEASURED [%s]: %s" % (name, ks))
            continue
        for i in range(len(seq) - 1):
            designed = seq[i + 1][1] - seq[i][1]
            got = ks[i + 1] - ks[i]
            if got <= 0:
                F.append("SPAN-NONMONOTONE [%s] %s->%s: realised k %.3f -> %.3f" % (name, seq[i][0], seq[i + 1][0], ks[i], ks[i + 1]))
            elif weak and got < 0.5 * designed:
                F.append("SPAN-WEAK-STEP [%s] %s->%s: realised k step %+.3f vs designed %+.2f"
                         % (name, seq[i][0], seq[i + 1][0], got, designed))
    for rid in PITCH_RUNGS:
        kh, kp = R(rid, "realised_k"), R(rid, "realised_k_praat")
        if kh is not None and kp is not None and abs(kh - kp) > 0.15:
            F.append("TRACKER-DISAGREE %s: realised k harvest %.3f vs praat %.3f" % (rid, kh, kp))
    seqg = [("g060", 0.60), ("c100", 1.00), ("g150", 1.50)]
    ev = [R(rid, "env_sd_db") for rid, _ in seqg]
    if c_env and all(e is not None for e in ev):
        for i in range(len(seqg) - 1):
            designed = (seqg[i + 1][1] - seqg[i][1]) * c_env
            got = ev[i + 1] - ev[i]
            if got < 0.5 * designed:
                F.append("DYN-COLLAPSE %s->%s: realised env-SD step %+.2f dB vs designed %+.2f dB"
                         % (seqg[i][0], seqg[i + 1][0], got, designed))
    for rid, sgn in (("less", -1), ("more", +1)):
        rk, rg = R(rid, "realised_k"), R(rid, "realised_g")
        if rk is not None and sgn * (rk - 1.0) <= 0:
            F.append("COMBINED %s: realised k %.3f (wrong way)" % (rid, rk))
        if rg is not None and sgn * (rg - 1.0) <= 0:
            F.append("COMBINED %s: realised g %.3f (wrong way)" % (rid, rg))
    d_lvl = R("lvl+6", "d_rms_speech_db")
    if d_lvl is not None and d_lvl < LEVEL_DB - 0.1:
        F.append("LEVEL-SHORT lvl+6: delivered %+.2f dB (peak guard)" % d_lvl)
    d_eff = R("effort", "d_rms_speech_db")
    if d_eff is not None and d_eff < EFFORT_LEVEL_DB - 0.1:
        F.append("LEVEL-SHORT effort: delivered %+.2f dB vs c100 (peak guard)" % d_eff)
    for rid in F0_UNTOUCHED:
        rk = R(rid, "realised_k")
        if rk is not None and abs(rk - 1.0) > 0.05:
            F.append("F0-LEAK %s: realised k %.3f with no F0-span change (tracker/voicing%s)"
                     % (rid, rk, "; src = the PSOLA null floor" if rid == "src" else ""))
    for rid in PITCH_RUNGS:
        b = R(rid, "design_beta", "target")
        if b is not None and not (0.80 <= b <= 1.25):
            F.append("TONE %s: design beta %.3f outside 0.80-1.25" % (rid, b))
        trp = R(rid, "tone_r_vs_c100_praat")
        if trp is not None and trp < 0.80:
            F.append("TONE-REALISED %s: praat tone-layer r vs c100 %.3f < 0.80 (informational)" % (rid, trp))
    for rid in REG_RUNGS:
        if rid not in it:
            continue
        want = it[rid]["reg_st"]
        for name in ("praat", "harvest"):
            got = R(rid, "register_st_" + name)
            if got is None:
                F.append("REGISTER-UNMEASURED %s [%s]" % (rid, name))
            elif abs(got - want) > 0.5:
                F.append("REGISTER %s [%s]: realised %+.2f st vs designed %+.2f" % (rid, name, got, want))
        kp = R(rid, "realised_k_praat")
        if kp is not None and abs(kp - 1.0) > 0.10:
            F.append("REGISTER-SPAN %s: realised k (praat) %.3f -- span not held" % (rid, kp))
    for rid in ("rate090", "effort0", "effort"):
        if rid not in it:
            continue
        s = R(rid, "realised_time_scale_env")
        if s is None or abs(s - EFFORT_RATE) > 0.02:
            F.append("RATE %s: realised envelope time scale %s vs designed %.2f" % (rid, s, EFFORT_RATE))
        al = R(rid, "nucleus_align") or {}
        if (al.get("median_abs_offset_ms") or 0) > 25:
            F.append("RATE-ALIGN %s: nuclei %.0f ms off the mapped c100 nuclei (median)" % (rid, al["median_abs_offset_ms"]))
    for rid in [r for r in it if it[r]["rate"] == 1.0 and r != "c100"]:
        s = R(rid, "realised_time_scale_env")
        if s is not None and abs(s - 1.0) > 0.01:
            F.append("TIME-SCALE %s: envelope scale %.3f on a rate-1 item" % (rid, s))
    for rid in [x for x in it if x not in ("c100", "lvl+6", "effort")]:
        d = R(rid, "nonspeech_rms_vs_c100_db")
        if d is not None and abs(d) > 1.0:
            F.append("NONSPEECH %s: RMS outside the speech extent %+.2f dB vs c100 (limit +/-1)" % (rid, d))
        p = R(rid, "pause_rms_vs_c100_db")
        if p is not None and abs(p) > 1.0:
            F.append("PAUSE %s: RMS in internal pauses %+.2f dB vs c100" % (rid, p))
    for r in out["items"]:
        t = r["target"]
        if t.get("n_frames_capped_900hz"):
            F.append("CAP %s: %d target frames hit manipulate()'s 900 Hz clip" % (r["rung"], t["n_frames_capped_900hz"]))
        if (t.get("frac_voiced_outside_analysis_range") or 0) > 0.01:
            F.append("RANGE %s: %.1f%% of voiced target frames outside the analysis F0 range"
                     % (r["rung"], 100 * t["frac_voiced_outside_analysis_range"]))
        if (t.get("f0_span_st") or 0) > 18.0:
            F.append("WIDE %s: target F0 span %.1f st > 18 st guide (not capped)" % (r["rung"], t["f0_span_st"]))
    for rid in PITCH_RUNGS:
        d = R(rid, "d_f0_span_st")
        if d is not None and d > EAR_PASS_ST + 1.0:
            F.append("EAR %s: realised F0-span change %+.2f st (the listening check passed p180 cells at mean %+.2f, rejected at %+.2f)"
                     % (rid, d, EAR_PASS_ST, EAR_REJECT_ST))
    for r in out["items"]:
        if r["family"] in ("null", "dyn", "level", "source", "target_null", "register", "rate", "effort"):
            b = r["target"].get("design_beta")
            if b is not None and abs(b - 1.0) > 1e-3:
                F.append("SELFTEST %s: design beta %.3f != 1.000" % (r["rung"], b))
        if r["build"].get("peak_guard_engaged"):
            F.append("PEAK-GUARD %s: pre-guard peak %.3f -> scaled to 0.98 (level %+.2f dB vs c100)"
                     % (r["rung"], r["build"]["peak_before_guard"], r["realised"]["d_rms_speech_db"]))
        if r["realised"]["n_full_scale_samples"]:
            F.append("CLIP %s: %d full-scale samples" % (r["rung"], r["realised"]["n_full_scale_samples"]))
        if r["rung"] not in ("lvl+6", "effort") and abs(r["realised"]["rms_speech_dbfs"] - TARGET_DBFS) > 0.15:
            F.append("RMS %s: speech-extent RMS %.2f dBFS (target %.1f)" % (r["rung"], r["realised"]["rms_speech_dbfs"], TARGET_DBFS))
    return F


# --------------------------------------------------------------------------- main
def _safe(item):
    try:
        return build_cell(item)
    except Exception:
        return dict(cell=item["cell"], lang=item["lang"], label=item["label"], items=[], flags=[],
                    failures=["EXCEPTION: " + traceback.format_exc()], notes=[])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--manifest", default=LANE + "/manifest.json")
    a = ap.parse_args()
    for d in (WAV_DIR, STRUCT_DIR, TMAP_DIR):
        os.makedirs(d, exist_ok=True)
    pool = {it["cell"]: it for it in json.load(open(POOL_SET, encoding="utf-8"))["items"]}
    want = [c for c in (a.cells.split(",") if a.cells else CELLS) if c]
    missing = [c for c in want if c not in pool]
    if missing:
        print("cells not in pool_set.json:", missing, flush=True)
        return 2
    items = [pool[c] for c in want]
    t0 = time.time()
    res = {}
    procs = max(1, min(2, a.procs))

    def report(r):
        res[r["cell"]] = r
        print("done %-24s %6.1fs items %d flags %d failures %d" % (
            r["cell"], r.get("seconds") or 0, len(r["items"]), len(r["flags"]), len(r["failures"])), flush=True)
        for fl in r["failures"]:
            print("   FAILURE", fl[:3000], flush=True)

    if procs == 1:
        for it in items:
            report(_safe(it))
    else:
        with get_context("fork").Pool(procs) as P:
            for r in P.imap_unordered(_safe, items):
                report(r)
    cells = [res[c] for c in want]
    man = dict(
        built=time.strftime("%Y-%m-%d %H:%M:%S %z"), builder=LANE + "/build_ladder.py", lane="voicedose/ladder2",
        spec="the voice-dose plan, section 3.1",
        engine=("Praat TD-PSOLA via h2_prosody_probe3.psola_resynth (null, pitch targets, DurationTier rate); WORLD analysis "
                "only (h2_prosody_probe2.analyse_constrained; target contours from h2_prosody_probe.manipulate(k_phrase, "
                "k_tone=1.0, reg_st), its WORLD waveform discarded); dynamics = h2_prosody_probe3.wave_dyn blended in "
                "inside the frozen c100 IPs only; level = scalar gain"),
        conditioning=("h2_prosody_probe3.highpass (80 Hz, 4th-order zero-phase) + rms_normalize(-26 dBFS whole file); then "
                      "every item one scalar gain to -26 dBFS RMS over the c100 speech extent (h2_voicefix_gen.speech_edges, "
                      "35-dB gate, frozen per cell; x 1/rate on rate items); lvl+6 = on-disk c100 x 10^(6/20); effort = "
                      "on-disk effort0 x 10^(7/20); 0.98 peak guard (flagged when engaged); PCM_16 at the source sample rate"),
        phrase_layer="log-F0, unvoiced gaps interpolated, 400-ms Hann smooth around the voiced median; tone = residual; k_tone = 1.0",
        rungs=[dict(rung=r, family=f, role=ROLE.get(r, "rung"), **p) for r, f, p in RUNGS],
        references=dict(src="conditioned source without PSOLA (null floor)",
                        tgt100="PSOLA with the unchanged analysis contour imposed as target (floor of every target rung)",
                        effort0="effort composite at the common -26 dBFS (level-equalised twin of effort; diagnostic)",
                        **{"lvl+6": "gain-knob diagnostic, not a counted rung (plan section 2.2 G2)"}),
        frozen_structure=STRUCT_DIR + "/<cell>.json (features_v2.structure on c100, computed once; rate_maps for rate rungs)",
        time_maps=TMAP_DIR + "/<cell>__<rung>.json (rate rungs: design map, measured envelope scale, nucleus pairs, frame index arrays at 30 and 60 fps)",
        measures=dict(
            realised_k="LS slope of the item's phrase layer on c100's; harvest (union range) + clean_f0; frames voiced in the "
                       "analysis contour and in both tracks; rate items mapped onto c100's frames first",
            realised_k_praat="the same with Praat To Pitch (ac)",
            register_st_praat="median frame-wise 12*log2(item/c100), Praat, frozen frame set (harvest: register_st_harvest)",
            realised_time_scale_env="s maximising corr(c100 10-ms dB envelope, item envelope at t/s) over the c100 speech extent; rate items should read 0.90",
            nucleus_align="frozen c100 v2 nuclei mapped by 1/rate vs nuclei detected on the item with the frozen profile (nearest within 80 ms)",
            adj_step_rms_st_praat="RMS of register steps between consecutive syllables of the same IP (frozen c100 syllables; "
                                  "syllable register = mean st of Praat-voiced frames); adj_step_ratio_praat = / c100; "
                                  "target.design_adj_step_ratio = same on the target contour vs the analysis contour",
            phrase_span_st="p90-p10 of the phrase layer over the frozen frame set (st); span_ratio = / c100",
            env_sd_db="rate 1: h2_prosody_probe2.measures(mask_ref=c100); rate items: envelope mapped onto c100 frames with the c100 gate; realised_g = / c100",
            env_sd_db_ref_extent="the same restricted to the c100 speech extent (realised_g_extent)",
            nonspeech_rms_vs_c100_db="RMS outside the (mapped) c100 speech extent, item minus c100 (dB)",
            pause_rms_vs_c100_db="RMS inside the c100 speech extent but outside the frozen c100 IPs, item minus c100 (rate-1 items)",
            tone_r_vs_c100="Pearson r of the tone layer vs c100's on frames voiced in both (harvest; _praat = Praat)",
            design_beta="h2_prosody_probe2.design_beta(analysis F0, target F0); F0-shape-untouched items must be 1.000",
            delivery_err_med_st="median |realised - target| st on frames voiced in both",
            human_cues="human/acoustic.one() scalar definitions (16 kHz, extract_prosody F0, acoustic.speech_span) on the item; vs_c100 = deltas / ratios",
            lilthead_view="lilthead/extract_prosody.features(y, sr, 60): the v1 model's own tracker and p99 energy reference; "
                          "rate items mapped onto c100's frames; realised_k / realised_g against c100's tracks"),
        seconds=round(time.time() - t0, 1), cells=cells)
    jdump(man, a.manifest)
    nfail = sum(len(c["failures"]) for c in cells)
    print("manifest -> %s  (%d items, %d failures, %.0fs)" % (a.manifest, sum(len(c["items"]) for c in cells), nfail, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

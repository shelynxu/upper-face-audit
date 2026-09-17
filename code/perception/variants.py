# -*- coding: utf-8 -*-
"""variants.py -- the training-free LiltFace rungs as transforms on RAW EmoFace rigs (T, 174).

Pure numpy (no torch), so the same file serves the emoface env (torch 1.10) and pl_medtalk.
Every transform takes the model's raw controller output BEFORE emoface_cli's post-step
(savgol, blinks, gaze, eye x1.8 / mouth x1.3, head motion), which screen_rigs.py then applies
identically to every variant with one seed -- so a before/after pair differs only in what the
model (or the rung) produced.

    rung 4  brow_accent_layer   AU1/AU2 (inner/outer brow raise) += gain * accent envelope
                                (prosody_brow_demo.py generalised: half-wave-rectified F0 rise
                                in semitones re the utterance median, smoothed ~120 ms, max 1)
    rung 2  compose_upper       EmoFace mouth + MEDTalk upper face (the 28 upper columns)
    rung 3  intensity_gate      s_t for the in-model gate F_t = base_t + e_neu + s_t (e_lab - e_neu)
                                (3a) emotion2vec frame posterior of the label class
                                (3b) prosodic prominence z(F0) + z(E)
                                both smoothed, mean-one over the utterance, clipped to [lo, hi]

Cue tracks come from prosody_ladder/cue_bank.py (one .npz per wav, 60 fps).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CUE_DIR = HERE / "cue_bank"
FPS = 60.0

# column blocks of the 174-control rig (Emoface/visualization/metahuman_attr_names.txt),
# identical to probe_emoface.py / emoface_cli.py
AU1 = (31, 100)                       # CTRL_{L,R}_brow_raiseIn
AU2 = (32, 101)                       # CTRL_{L,R}_brow_raiseOut
BROW_DOWN = (29, 98)                  # CTRL_{L,R}_brow_down (AU4; anger) -- never touched here
UPPER_COLS = [29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
              98, 99, 100, 101, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
MOUTH_COLS = list(range(3, 13)) + list(range(46, 92)) + list(range(115, 161))
LOWER_COLS = MOUTH_COLS + list(range(13, 29))

# emotion2vec_plus_base class order (tokens.txt) -> the project's canonical labels
E2V_CANON = {"生气/angry": "ang", "厌恶/disgusted": "dis", "恐惧/fearful": "fea", "开心/happy": "hap",
             "中立/neutral": "neu", "其他/other": "oth", "难过/sad": "sad", "吃惊/surprised": "sur"}


def cue_key(wav):
    return hashlib.sha1(str(Path(wav).resolve()).encode("utf-8")).hexdigest()[:16]


def load_cue(wav):
    p = CUE_DIR / (cue_key(wav) + ".npz")
    if not p.is_file():
        raise FileNotFoundError("no cue-bank entry for %s (run prosody_ladder/cue_bank.py --wav ...)" % wav)
    d = np.load(p, allow_pickle=True)
    out = {k: d[k] for k in d.files if k != "meta"}
    out["meta"] = json.loads(str(d["meta"]))
    out["e2v_canon"] = [E2V_CANON.get(str(x), str(x)) for x in out["e2v_labels"]]
    return out


def fit_T(x, T):
    """Resample a (N,) or (N, D) frame track to exactly T frames (linear; identity when N == T)."""
    x = np.asarray(x, float)
    n = x.shape[0]
    if n == T:
        return x
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, T)
    if x.ndim == 1:
        return np.interp(dst, src, x)
    return np.stack([np.interp(dst, src, x[:, j]) for j in range(x.shape[1])], axis=1)


def gauss_smooth(x, ms, fps=FPS):
    from scipy.ndimage import gaussian_filter1d
    sigma = (ms / 1000.0) * fps / 2.0
    return gaussian_filter1d(np.asarray(x, float), sigma=max(sigma, 1e-6))


# ------------------------------------------------------------------ rung 4
def speech_span(cue, T):
    """(first, last) speech frame on the T-frame grid (energy within 30 dB of the file maximum), or None."""
    e = fit_T(cue["logE_db"], T).astype(float)
    idx = np.nonzero(e > (e.max() - 30.0))[0]
    return (int(idx[0]), int(idx[-1])) if len(idx) else None


# non-articulatory lower-face controls that carry the expression (smile, dimple, corner depress, stretch, upper-lip raise,
# nose wrinkle / flare, nasolabial fold); jaw, lips, purse, funnel, towards, sticky and tongue articulate and stay free
EXPR_LOWER_COLS = [52, 53, 56, 77, 82, 91, 94, 95, 96, 97, 121, 122, 125, 146, 151, 160, 163, 164, 165, 166]
BLINK_COLS = [36, 105]
# brow down / lateral / raise inner / raise outer and the upper + lower eyelid, both sides (r5b: where LiltFace's gate acts)
BROW_LID_COLS = [29, 30, 31, 32, 38, 39, 98, 99, 100, 101, 107, 108]


def hold_expression(R, cue, ref=None, inside_frames=10, ease_frames=9):
    """Hold the facial EXPRESSION through the silent lead-in and lead-out; returns a copy (the `_h` conditions, v2).

    OBSERVED (2026-09-14, on top of the end-of-clip stare): clips start or end with frames that have no expression --
    the face snaps from an expression to neutral, or only the eyes spring wide open. Measured on the 12 study clips:
    in the silence EmoFace relaxes the non-articulatory controls toward neutral (cn_hap: squint 0.25 / cheek raise 0.02 on
    the first frame vs 1.47 / 0.71 in speech; smile 0.14 vs 0.47; en_hap smile 0.29 vs 0.63), drives the blink control
    negative (eyes wide), and ramps the expression back in over the first ~10 frames of speech. pad_rig then holds those
    neutral first / last frames through the visible pads.
    FIX: every expression control (UPPER_COLS except the blink, plus EXPR_LOWER_COLS) takes its median over the first
    (lead-in) or second (lead-out) half of the speech frames of `ref` (the EmoFace array of the same clip and condition, so lead-in / lead-out are identical across arms)
    outside speech, and hands over to the model with a raised-cosine ramp INSIDE the first and last `inside_frames` of
    speech -- after the model's own ramp-in, so there is no dip at onset (v1 eased outside speech and dipped there).
    The blink keeps v1's cosine ease OUTSIDE speech (an inserted blink at speech onset stays whole). Articulators are
    untouched: the mouth closes in silence as the model has it."""
    R = np.array(R, dtype=np.float32, copy=True)
    src = R if ref is None else np.asarray(ref, dtype=np.float32)
    T = R.shape[0]
    span = speech_span(cue, T)
    if span is None:
        return R
    a, b = span
    R = hold_outside_speech(R, cue, ref=src, cols=BLINK_COLS, ease_frames=ease_frames)
    cols = [c for c in UPPER_COLS if c not in BLINK_COLS] + EXPR_LOWER_COLS
    m = (a + b) // 2                     # lead-in holds the median of the first half of speech, lead-out of the second
    med_on = np.median(src[a:m + 1][:, cols], axis=0)      # half: an expression that grows over the utterance (cn_hap
    med_off = np.median(src[m:b + 1][:, cols], axis=0)     # smile 0.47 -> 0.65) is not pulled back at its end
    n = max(1, min(int(inside_frames), (b - a + 1) // 2))
    for t in range(T):
        med = med_on if t <= m else med_off
        if t < a or t > b:
            w = 0.0
        else:
            d = min(t - a, b - t)                   # frames inside speech from the nearer edge
            w = 0.5 * (1.0 - np.cos(np.pi * d / n)) if d < n else 1.0
        if w < 1.0:
            R[t, cols] = w * R[t, cols] + (1.0 - w) * med
    return R


def hold_outside_speech(R, cue, ref=None, cols=None, ref_ms=150.0, ease_frames=9):
    """Freeze the upper face outside speech at its speech-edge state; returns a copy.

    ROOT CAUSE (observed 2026-09-13, happy clips ending in a wide-eyed smiling stare): in the silence before and after
    the voice EmoFace drives CTRL_{L,R}_eye_blink NEGATIVE (eyes wide open) while the emotion label keeps the smile --
    en_hap -0.12 in speech -> -0.53 on the last frame, en_ang -0.17 -> -0.99, cn_sur -0.78 -> -1.27 (post-step values) --
    the post-step's eye x1.8 amplifies it and pad_rig holds that last frame through the lead-out. EmoFace itself has it,
    so every arm does.
    FIX: outside [first, last] speech frame the upper-face controls (UPPER_COLS: brows, blink, lids, squint, cheek raise)
    are replaced by their median over ALL speech frames, blended in with a cosine ease over ease_frames so nothing
    jumps. (A first version used the last 150 ms of speech: that window caught the post-step's inserted blinks and the
    end-of-speech eye closure, so the fixed clips ended with the eyes +0.2..+1.2 more closed -- the opposite artefact.
    The whole-speech median ignores brief blinks and accent peaks.) `ref` (the EmoFace array of the same clip and
    condition) supplies the reference values, so the lead-in and lead-out are identical across arms and any arm
    difference lives inside speech. The mouth is untouched. `ref_ms` is kept for call compatibility and unused."""
    R = np.array(R, dtype=np.float32, copy=True)
    src = R if ref is None else np.asarray(ref, dtype=np.float32)
    cols = list(UPPER_COLS if cols is None else cols)
    T = R.shape[0]
    span = speech_span(cue, T)
    if span is None:
        return R
    a, b = span
    on_ref = off_ref = np.median(src[a:b + 1][:, cols], axis=0)

    def keep(d):                                    # weight of the model's own value at distance d frames from speech
        return 0.5 * (1.0 + np.cos(np.pi * d / ease_frames)) if d < ease_frames else 0.0

    for t in range(0, a):
        w = keep(a - t)
        R[t, cols] = w * R[t, cols] + (1.0 - w) * on_ref
    for t in range(b + 1, T):
        w = keep(t - b)
        R[t, cols] = w * R[t, cols] + (1.0 - w) * off_ref
    return R


def speech_weight(cue, T, n=9):
    """0 outside speech (energy within 30 dB of the file maximum), ramping 0 -> 1 over n frames (150 ms) into the first
    speech frame and 1 -> 0 over the last n speech frames. 2026-09-13 judge finding: without it the gate relaxes the face
    in the lead-in/lead-out pads and the phrase envelope puts up to 76 % of its mass outside speech, so the pads differed
    from EmoFace by 8-13 grey levels on every frame ('LiltFace starts calmer'). Every LiltFace term is multiplied by it."""
    e = fit_T(cue["logE_db"], T).astype(float)
    sp = e > (e.max() - 30.0)
    idx = np.nonzero(sp)[0]
    w = np.zeros(T)
    if len(idx) == 0:
        return w
    a, b = int(idx[0]), int(idx[-1])
    w[a:b + 1] = 1.0
    ramp = np.linspace(0.0, 1.0, n + 1)[1:]
    w[a:min(a + n, T)] = ramp[:min(n, T - a)]
    lo_ = max(b - n + 1, 0)
    w[lo_:b + 1] = np.minimum(w[lo_:b + 1], ramp[::-1][-(b + 1 - lo_):])
    return w


def taper_gate(s, w):
    """A mean-one gate track pulled back to 1 outside speech: 1 + (s - 1) * w."""
    return 1.0 + (np.asarray(s, float) - 1.0) * np.asarray(w, float)


def accent_envelope(cue, T, smooth_ms=120.0, layer="f0", weight=None):
    """Half-wave-rectified pitch rise, smoothed, scaled to max 1 (prosody_brow_demo.accent).

    layer="f0": semitones re the utterance median with unvoiced frames at 0 (the demo's
    signal; in Mandarin this also follows lexical tone). layer="phrase": the 400-ms intonation
    layer only (tone removed), the tone-safe choice for cn.
    """
    if layer == "phrase":
        st = fit_T(cue["phrase_st"], T)
    elif layer == "prom":
        # prominence = z(F0) + z(energy): the syllable-level emphasis track. IEMOCAP markers (09-13) put real
        # brows on ENERGY more than on pitch, and it fires per stressed syllable, so it gives more events than
        # the 400-ms intonation layer.
        st = fit_T(cue["prominence"], T)
    else:
        st = fit_T(cue["f0_st"], T)
        v = fit_T(cue["voiced"].astype(float), T) > 0.5
        st = np.where(v, st, 0.0)
    a = gauss_smooth(np.clip(st, 0, None), smooth_ms)
    m = float(a.max())
    a = a / m if m > 1e-9 else a
    return a * np.asarray(weight, float) if weight is not None else a


def brow_accent_layer(raw, cue, gain=0.20, au2_ratio=0.5, smooth_ms=120.0, layer="f0"):
    """Rung 4: add the accent envelope to the inner (and half to the outer) brow-raise controls.
    gain is in RAW units; the post-step's eye x1.8 makes 0.20 a peak of 0.36 on the rig."""
    B = np.array(raw, dtype=np.float32, copy=True)
    env = accent_envelope(cue, B.shape[0], smooth_ms, layer)
    for c in AU1:
        B[:, c] += gain * env
    for c in AU2:
        B[:, c] += gain * au2_ratio * env
    return B, env


EYELID_U = (39, 108)                  # CTRL_{L,R}_eye_eyelidU: upper-lid raise (eye widening)
EYE_GAIN = 1.8                        # emoface_cli's shipped eye amplification; the rig clamps at 1
# Head columns appended by the post-step: 174 rotateX, 175 rotateY, 176 rotateZ. Rendered through
# Maya on Hadley_full_rig (2026-09-13, +15 deg stills): rotateX TURNS the head (yaw), rotateY tilts
# it sideways (roll), rotateZ PITCHES it and +15 is chin-up. emoface_cli's comments call X the
# pitch; on this rig they are wrong. A nod is therefore a NEGATIVE rotateZ.
HEAD_ROT_X, HEAD_ROT_Y, HEAD_ROT_Z = 174, 175, 176
NOD_COL, NOD_DOWN_SIGN = 2, -1.0      # index into the (T, 3) head block, and the sign that nods down


def accent_layer_v2(raw, cue, gain_brow=0.20, gain_lid=0.10, nod_deg=2.5, nod_lag_ms=100.0, nod_sign=1.0,
                    smooth_ms=120.0, layer="f0", weight=None):
    """Rung 4 v2: the accent layer through channels that still have headroom on every face, plus the
    head nod, the most reliable human marker of a strong accent (Swerts & Krahmer 2010: nods on
    89.6 % of strongly accented words, brows on 70.1 %).

    On the rendered EmoFace rigs the inner-brow raise is already clamped at 1 for happy and sad
    faces (raw 0.6-1.0 x the shipped eye gain 1.8), so the v1 layer changed nothing visible there.
    Here:  inner brow  += gain_brow * env, capped where the rendered value would exceed 1
           outer brow  += gain_brow * env          (headroom ~0.7 on every clip)
           upper lid   += gain_lid  * env          (eye widening; headroom ~0.94)
           head pitch  -= nod_deg * env delayed by nod_lag_ms, on rotateZ (see NOD_COL; returned
                          separately: the head columns are appended by the post-step, so the caller
                          adds this to columns 174:177 afterwards; the shipped idle motion is ~0 deg
                          on these clips, so the nod is the only head movement)
    Returns (raw', env, head_add (T, 3))."""
    B = np.array(raw, dtype=np.float32, copy=True)
    T = B.shape[0]
    env = accent_envelope(cue, T, smooth_ms, layer, weight=weight)
    cap = 1.0 / EYE_GAIN
    for c in AU1:
        B[:, c] = np.minimum(np.maximum(B[:, c], B[:, c] + gain_brow * env), np.maximum(B[:, c], cap))
    for c in AU2:
        B[:, c] += gain_brow * env
    for c in EYELID_U:
        B[:, c] += gain_lid * env
    lag = int(round(nod_lag_ms / 1000.0 * FPS))
    env_lag = np.concatenate([np.zeros(lag), env[:T - lag]]) if lag > 0 else env
    head = np.zeros((T, 3), dtype=np.float32)
    head[:, NOD_COL] = nod_sign * NOD_DOWN_SIGN * nod_deg * gauss_smooth(env_lag, 60.0)
    return B, env, head


# ------------------------------------------------------------------ rung 2
def compose_upper(raw_emoface, raw_upper_source, cols=UPPER_COLS):
    """Rung 2: EmoFace's rig with the upper-face columns taken from another model's rig on the
    same audio (MEDTalk, same 174-column order). Frame counts are matched by truncation."""
    A = np.array(raw_emoface, dtype=np.float32, copy=True)
    U = np.asarray(raw_upper_source, dtype=np.float32)
    n = min(len(A), len(U))
    if abs(len(A) - len(U)) > 2:
        U = fit_T(U, len(A)).astype(np.float32)
        n = len(A)
    A[:n, cols] = U[:n, cols]
    return A[:n]


# ------------------------------------------------------------------ rung 3
def intensity_gate(cue, T, label, mode="e2v", lo=0.4, hi=1.6, smooth_ms=150.0, prom_gain=0.25):
    """s_t in [lo, hi], mean one over the utterance.

    mode="e2v":  the emotion2vec frame posterior of the label's class (canonical ang/hap/sad/...),
                 smoothed, divided by its utterance mean.  For 'neu' the gate has nothing to scale
                 (e_lab - e_neu = 0) and s_t is returned as ones.
    mode="prom": 1 + prom_gain * smoothed prominence (z(F0) + z(E)), then mean-one.
    The clip is applied AFTER mean-one normalisation and the mean is restored once more, so the
    utterance-average expression stays at the label's trained strength.
    """
    if label == "neu":
        return np.ones(T)
    if mode == "e2v":
        idx = cue["e2v_canon"].index(label)
        s = fit_T(cue["e2v_post"][:, idx], T)
        s = gauss_smooth(s, smooth_ms)
        s = s / max(float(s.mean()), 1e-6)
    elif mode == "prom":
        p = gauss_smooth(fit_T(cue["prominence"], T), smooth_ms)
        s = 1.0 + prom_gain * p
        s = s / max(float(s.mean()), 1e-6)
    elif mode == "prom_acc":
        # rung 5 "accent" form of the gate: RECTIFIED prominence. Between accents the face sits at the
        # label's trained strength (s = 1); on a prominent syllable the emotion direction is
        # pushed further (up to hi). No mean-one normalisation, so the utterance mean is > 1 by
        # construction: an accent layer in embedding space, not a redistribution.
        p = gauss_smooth(fit_T(cue["prominence"], T), smooth_ms)
        return np.clip(1.0 + prom_gain * np.clip(p, 0.0, None), 1.0, hi)
    else:
        raise ValueError("mode must be e2v, prom or prom_acc")
    # clip, restore the mean, clip again: bounded in [lo, hi] with the mean within a few
    # percent of one (a single clip-then-renormalise can leave values far above hi)
    s = np.clip(s, lo, hi)
    s = np.clip(s / max(float(s.mean()), 1e-6), lo, hi)
    return s


def describe(name):
    return {
        "before": "EmoFace as shipped (raw model output + the identical post-step)",
        "r1_medtalk": "rung 1: MEDTalk as shipped (raw output + the identical post-step)",
        "r2_compose": "rung 2: EmoFace mouth + MEDTalk upper face (28 upper columns)",
        "r3_e2v": "rung 3a: in-model per-frame intensity gate from emotion2vec posteriors",
        "r3_prom": "rung 3b: in-model per-frame intensity gate from prosodic prominence",
        "r3_prom_up": "rung 3b, upper face only: EmoFace mouth + the prominence-gated model's upper face",
        "r4_f0": "rung 4: post-hoc AU1/AU2 accent layer from the rectified F0 rise",
        "r4_phrase": "rung 4: post-hoc AU1/AU2 accent layer from the intonation layer only",
        "r4_nod": "rung 4 v2: accent layer on outer brow + upper lid (+ inner brow where not clamped) "
                  "and a 2.5-degree head nod lagging the pitch rise by 100 ms",
        "r4_up": "rung 4 v2 without the nod: F0 accent on outer brow + upper lid (+ inner brow where not clamped)",
        "r5_lilt": "LiltFace: prominence gate (upper face, in-model) + intonation-layer accent on outer brow, "
                   "upper lid and head nod (tone-safe: the lexical-tone layer never drives the face)",
        "r5_lilt_nonod": "LiltFace without the head nod",
        "r5_dir": "rung 5 in-model accent: rectified prominence pushes the upper face along EmoFace's own "
                  "emotion direction (e_lab - e_neu) at accented syllables; mouth from the shipped model",
        "r5s_lilt_nonod": "LiltFace (study, speech-masked): r5_lilt_nonod with the gate deviation and the accent envelope "
                          "multiplied by a speech weight (0 in the lead-in / lead-out, 150-ms ramps), so the pads equal EmoFace",
        "r6_xl": "LiltFace-XL: deeper prominence gate (brows relax to 0.1x between accents, up to 2.4x on them), "
                 "accent from the syllable-level prominence track on outer brow + upper lid + inner brow where not "
                 "clamped, and a 3-degree head nod lagging the accent by 100 ms; mouth untouched",
    }.get(name, name)

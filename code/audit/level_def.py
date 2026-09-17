# -*- coding: utf-8 -*-
"""level_def.py -- THE one level / envelope / speech-extent definition of the human2 lane (numpy + scipy only).

Every level number in human2.md uses LEVEL_DB below. It is the human lane's acoustic.py definition, re-implemented
without extract_prosody so it can run on ANY wav (ladder rungs, model inputs) in any env:

  1. mono, resampled to 16 kHz (scipy resample_poly, as acoustic.py / extract_prosody do)
  2. 30-ms RMS frames at a 10-ms hop (frame k centred on sample 160 k; extract_prosody.analyse's vectorised RMS)
  3. SPEECH EXTENT (acoustic.py speech_span v2): frame dB re the file's 99th-percentile frame; core frames > -20 dB in
     runs >= 30 ms; core runs merged across gaps < 0.5 s; clusters holding >= 5 % of the loudest cluster's energy and
     >= 100 ms kept; first..last kept cluster, edges extended through contiguous frames > -35 dB.
     Gain-invariant (relative to p99), so a scalar gain never moves the extent.
  4. LEVEL_DB = median over extent frames within 40 dB of the extent's loudest frame of 20 log10(RMS)   [dBFS]
     (median of frame dB, not energy mean: it moves 1:1 with a scalar gain and is not mechanically inflated by a
     wider envelope, which is a separate predictor)
  5. ENV_SD_DB = SD over the same frame set of the frame dB re p99 (gain-invariant)
  6. SPEECH_DUR_S = extent end - start

Alternatives computed only for the reconciliation table (never used as predictors):
  wav_level_dbfs    audit/metrics.py (median sliding 30-ms dB within 40 dB of the FILE max; measure + audit lanes)
  ladder_rms_dbfs   ladder/build_ladder.py rule: energy RMS over h2_voicefix_gen.speech_edges (10-ms frames > peak - 35 dB)
  span_energy_dbfs  energy-mean RMS over the LEVEL_DB extent
"""
import math

import numpy as np
from scipy.signal import resample_poly

SR = 16000


def to16k(y, sr):
    y = np.asarray(y, np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if int(sr) != SR:
        g = math.gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g)
    return y


def frame_rms(y16):
    hop, half = int(SR * 0.010), int(SR * 0.015)
    n = int(len(y16) / hop) + 1
    c2 = np.concatenate([[0.0], np.cumsum(y16 ** 2)])
    cen = np.arange(n) * hop
    a_ = np.clip(cen - half, 0, len(y16))
    b_ = np.clip(cen + half, 0, len(y16))
    return np.sqrt(np.maximum((c2[b_] - c2[a_]) / np.maximum(b_ - a_, 1), 0.0)), a_, b_


def _runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def speech_span_v2(r):
    """acoustic.py speech_span, same logic. r = frame dB re p99 (10-ms hop). Returns (t_on, t_off) in s or (None, None)."""
    r = np.asarray(r, float)
    core = [(a, b) for a, b in _runs(r > -20.0) if b - a >= 3]
    if not core:
        return None, None
    clusters = [[core[0][0], core[0][1]]]
    for a, b in core[1:]:
        if a - clusters[-1][1] < 50:
            clusters[-1][1] = b
        else:
            clusters.append([a, b])
    lin = 10 ** (r / 10.0)
    en = [lin[a_:b_].sum() for a_, b_ in clusters]
    kept = [ab for ab, e_ in zip(clusters, en) if e_ >= 0.05 * max(en) and ab[1] - ab[0] >= 10]
    if not kept:
        kept = [clusters[int(np.argmax(en))]]
    a, b = kept[0][0], kept[-1][1]
    while a > 0 and r[a - 1] > -35.0:
        a -= 1
    while b < len(r) and r[b] > -35.0:
        b += 1
    return a * 0.010, b * 0.010


def measure(y, sr, span=None, alternatives=True):
    """-> dict(level_db, env_sd_db, speech_dur_s, t_on, t_off [, wav_level_dbfs, ladder_rms_dbfs, span_energy_dbfs]).
    span=(t_on, t_off) overrides the extent (e.g. to measure a rung on its reference's extent)."""
    y16 = to16k(y, sr)
    rms, a_, b_ = frame_rms(y16)
    rms_e = rms.copy()
    rms_e[b_ <= a_] = 0.0                                     # extract_prosody.analyse sets empty frames to 0
    ref = float(np.percentile(rms_e, 99)) or 1.0
    rdb = 20 * np.log10(np.maximum(rms_e / ref, 1e-3))
    t_on, t_off = span if span is not None else speech_span_v2(rdb)
    out = dict(t_on=t_on, t_off=t_off)
    if t_on is None:
        return dict(out, level_db=np.nan, env_sd_db=np.nan, speech_dur_s=np.nan)
    t10 = np.arange(len(rdb)) * 0.010
    in10 = (t10 >= t_on) & (t10 <= t_off)
    dbfs = 20 * np.log10(np.maximum(rms, 1e-9))
    gf = dbfs[in10]
    gf = gf[gf > gf.max() - 40.0]
    g = rdb[in10]
    g = g[g > g.max() - 40.0]
    out.update(level_db=float(np.median(gf)), env_sd_db=float(np.std(g)), speech_dur_s=float(t_off - t_on))
    if not alternatives:
        return out
    n = max(1, int(SR * 0.030))
    p = np.convolve(y16 ** 2, np.ones(n) / n, mode="same")
    edb = 10 * np.log10(np.maximum(p, 1e-18))
    out["wav_level_dbfs"] = float(np.median(edb[edb > edb.max() - 40.0]))
    hop = int(SR * 0.010)
    nf = max(1, (len(y16) - hop) // hop + (1 if (len(y16) - hop) % hop else 0))
    fr = np.array([np.sqrt((y16[i:i + hop] ** 2).mean() + 1e-12) for i in range(0, max(1, len(y16) - hop), hop)])
    fdb = 20 * np.log10(fr / max(fr.max(), 1e-12))
    on = np.where(fdb > -35.0)[0]
    si, ei = (on[0] * hop, min(len(y16), (on[-1] + 1) * hop)) if len(on) else (0, len(y16))
    out["ladder_rms_dbfs"] = float(20 * np.log10(max(np.sqrt(np.mean(y16[si:ei] ** 2)), 1e-12)))
    out["span_energy_dbfs"] = float(10 * np.log10(max(np.mean(rms[in10] ** 2), 1e-18)))
    del nf
    return out


def measure_file(path, span=None, alternatives=True):
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float64")
    return measure(y, sr, span, alternatives)

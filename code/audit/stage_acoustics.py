# -*- coding: utf-8 -*-
"""stage_acoustics.py -- acoustic tracks + G2 cue measures for every ladder2 wav.  Env: GPTSoVits (pyworld).  CPU.

Per wav (16-kHz mono via level_def.to16k, the human2 resampler):
  (1) lilthead/extract_prosody.analyse -> grid(A, 60): f0_st, voiced, phrase_st, tone_st, logE_db (re p99) at 60 fps --
      the SAME extractor human2/clipdata.py used for the RAVDESS F0 peaks and W2 spans (EP.grid on the stored analysis),
      so model and human follow indices share the peak definition. nuclei_s is left empty (not used here).
      Also stored: f0_med_hz of the grid's voiced frames (to re-reference F0 to the c100 median) and absolute 30-ms
      frame dBFS at 60 fps (energy follower; logE_db is peak-referenced and level-blind).
  (2) the G2 cue measures exactly as human2/g2_predict.G2.measure_wav (copied into this lane): LEVEL_DB, ENV_SD_DB,
      SPEECH_DUR_S (level_def.py) and f0_med_hz / phrase_span_st (extract_prosody analysis + h2_prosody_probe smoother).
Writes: <lane>/acoustic/<id>.npz, <lane>/g2_measures.json.  Resumable.  <= 2 worker processes, 1 thread each.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import io
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

LANE = Path("${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/audit2")
sys.path.insert(0, str(LANE))
for p in ("${PROJECT_ROOT}/pipeline", "${PROJECT_ROOT}/pipeline/lilthead"):
    if p not in sys.path:
        sys.path.insert(0, p)
import common2 as C                                   # noqa: E402
import level_def as LD                                # noqa: E402  (lane copy of human2/level_def.py)
import extract_prosody as EP                          # noqa: E402

EP.OUT = LANE / "_unused_prosody_dir"                 # analyse/grid do not write; patched anyway
OUT = LANE / "acoustic"


def abs_db_60(y16, sr=16000, fps=60.0, win_s=0.030):
    x = np.asarray(y16, np.float64)
    T = int(len(x) / sr * fps)
    cen = np.round(np.arange(T) * sr / fps).astype(int)
    half = int(sr * win_s / 2)
    c2 = np.concatenate([[0.0], np.cumsum(x ** 2)])
    a = np.clip(cen - half, 0, len(x))
    b = np.clip(cen + half, 0, len(x))
    rms = np.sqrt(np.maximum((c2[b] - c2[a]) / np.maximum(b - a, 1), 0.0))
    return 20 * np.log10(np.maximum(rms, 1e-7))


def work(it):
    out = OUT / (it["id"] + ".npz")
    rec = {"id": it["id"]}
    try:
        import g2_predict as G2P                      # lane copy
        y, sr = sf.read(it["wav"], dtype="float64")
        y16 = LD.to16k(y, sr)
        if not out.is_file():
            A = EP.analyse(y16, LD.SR)
            A["nuclei_s"] = np.zeros(0, np.float32)
            G = EP.grid(A, 60)
            f0 = np.asarray(A["f0_hz_5ms"], float)
            T = len(G["f0_st"])
            tt = np.arange(T) / 60.0
            t5 = np.arange(len(f0)) * 0.005
            idx = np.clip(np.round(tt / 0.005).astype(int), 0, len(f0) - 1)
            vg = f0[idx] > 0
            f0g = np.interp(tt, t5, f0)
            med = float(np.median(f0g[vg])) if vg.sum() >= 3 else float("nan")
            np.savez_compressed(out.with_suffix(".tmp.npz"), f0_st=G["f0_st"], voiced=G["voiced"], phrase_st=G["phrase_st"],
                                tone_st=G["tone_st"], logE_db=G["logE_db"], abs_db=abs_db_60(y16)[:T].astype(np.float32),
                                f0_med_hz=med, f0_lo=A["f0_lo"], f0_hi=A["f0_hi"], dur_s=A["dur_s"], fps=60.0, sr_src=sr)
            os.replace(out.with_suffix(".tmp.npz"), out)
        m = G2P.G2.measure_wav(it["wav"])
        rec.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()
                    if np.isscalar(v)})
    except Exception as ex:                           # noqa: BLE001
        rec["error"] = str(ex)[:300]
        rec["tb"] = traceback.format_exc()[-800:]
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = C.items()
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if lim:
        items = items[:lim]
    t0 = time.time()
    with Pool(int(os.environ.get("AUDIT2_PROCS", "2"))) as pool:
        recs = pool.map(work, items, chunksize=2)
    bad = [r for r in recs if "error" in r]
    name = "g2_measures.json" if not lim else "g2_measures_smoke.json"
    io.open(LANE / name, "w", encoding="utf-8").write(json.dumps(recs, indent=1, default=float))
    print("DONE n=%d fail=%d %.0fs" % (len(recs), len(bad), time.time() - t0), flush=True)
    for r in bad[:5]:
        print(r["id"], r["error"], r["tb"], flush=True)


if __name__ == "__main__":
    main()

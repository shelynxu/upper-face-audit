# -*- coding: utf-8 -*-
"""run_ext.py -- EmoTalk / EMOTE / MEDTalk on every ladder2 item through the D1 audit probes (d1_audit/probes, read-only).  CPU.

  python run_ext.py --probe emotalk   (env d1_emotalk)  run(wav)            label-free; level=1, person=0 (demo defaults)
  python run_ext.py --probe emote     (env d1_emote)    run(wav, label)     EMOTE_v2, intensity 2, identity M003 (probe defaults);
                                                        + FLAME 68 landmarks on the mean face (probe.landmarks68)
  python run_ext.py --probe medtalk   (env pl_medtalk)  run(wav, label)     asr 'zh' (as shipped), raw decoder output (savgol 0),
                                                        + per-frame intensity s_t (probe.stages); the probe's ASR cache is
                                                        monkeypatched into <lane>/asr_cache_medtalk (never d1_audit/results)
Writes <lane>/outputs/<probe>/<id>.npz (M, fps, T [, lmk68, s_t, transcript]) and <lane>/run_<probe>.json. Resumable.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import argparse
import io
import json
import sys
import time
import traceback
from pathlib import Path

LANE = Path("${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/audit2")
sys.path.insert(0, str(LANE))
sys.path.insert(0, "${PROJECT_ROOT}/pipeline/d1_audit")
import common2 as C                                   # noqa: E402
import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, choices=["emotalk", "emote", "medtalk"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    items = C.items()
    if a.ids:
        want = set(a.ids.split(","))
        items = [i for i in items if i["id"] in want]
    if a.limit:
        items = items[:a.limit]
    out_dir = LANE / "outputs" / a.probe
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.probe == "medtalk":
        import probes.probe_medtalk as PM             # noqa: E402
        PM.ASR_CACHE = LANE / "asr_cache_medtalk"
    from probes import load_probe                     # noqa: E402
    t0 = time.time()
    P = load_probe(a.probe, device="cpu")
    torch.set_num_threads(a.threads)
    log = {"probe": a.probe, "n": len(items), "load_s": round(time.time() - t0, 1), "fail": [], "info": {}}
    try:
        log["info"] = json.loads(json.dumps(P.info(), default=str))
    except Exception as ex:                           # noqa: BLE001
        log["info"] = {"error": str(ex)[:200]}
    print("loaded %s in %.1fs" % (a.probe, time.time() - t0), flush=True)
    t1 = time.time()
    per = []
    for k, it in enumerate(items, 1):
        out = out_dir / (it["id"] + ".npz")
        if out.is_file():
            continue
        try:
            ts = time.time()
            extra = {}
            if a.probe == "emotalk":
                M = P.run(it["wav"])
            elif a.probe == "emote":
                M = P.run(it["wav"], emotion=it["label"])
                extra["lmk68"] = np.asarray(P.inner.landmarks68(M), np.float32)
            else:
                M = P.run(it["wav"], emotion=it["label"])
                st = P.stages(it["wav"], it["label"])
                extra["s_t"] = np.asarray(st["B_intensity"], np.float32).reshape(-1)
                extra["transcript"] = np.array(P._transcript(it["wav"]))
            M = np.asarray(M, np.float32)
            tmp = out.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, M=M, fps=float(P.frame_rate), T=int(M.shape[0]), **extra)
            os.replace(tmp, out)
            per.append(time.time() - ts)
        except Exception as ex:                       # noqa: BLE001
            log["fail"].append({"id": it["id"], "err": str(ex)[:300], "tb": traceback.format_exc()[-1200:]})
            print("FAIL", it["id"], ex, flush=True)
        if k % 10 == 0 or k == len(items):
            print("[%d/%d] %.0fs" % (k, len(items), time.time() - t1), flush=True)
    log["seconds"] = round(time.time() - t0, 1)
    log["sec_per_item_median"] = float(np.median(per)) if per else None
    io.open(LANE / ("run_%s%s.json" % (a.probe, "_smoke" if (a.limit or a.ids) else "")), "w", encoding="utf-8").write(
        json.dumps(log, indent=1, default=str))
    print("DONE %s fail=%d %.0fs" % (a.probe, len(log["fail"]), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

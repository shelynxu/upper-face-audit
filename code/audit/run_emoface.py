# -*- coding: utf-8 -*-
"""run_emoface.py -- EmoFace on every ladder2 item, exactly the measure lane's "before" path (measure/run_faces.py).  Env: emoface.  CPU.

  raw  = prosody_ladder/pool_rigs.Model().decode(label)   (d1_audit probe emoface_p, gate none, device cpu = plain EmoFace)
  post = screen_rigs.post_step(raw, n_blinks=0, eye=1.0, mouth=1.0)  (savgol 15/3, no blinks, seeded gaze + idle head)
Label = the cell's emotion label. Writes <lane>/outputs/emoface/<id>.npz (raw, post, T, fps=60). Resumable.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["SPJ_RIG_THREADS"] = "3"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import io
import json
import sys
import time
import traceback
from pathlib import Path

LANE = Path("${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/audit2")
sys.path.insert(0, str(LANE))
sys.path.insert(0, "${PROJECT_ROOT}/pipeline/prosody_ladder")
import common2 as C                                   # noqa: E402
import pool_rigs as PR                                # noqa: E402
import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

SR = PR.SR
OUT = LANE / "outputs" / "emoface"


def main():
    torch.set_num_threads(3)
    items = C.items()
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if lim:
        items = items[:lim]
    OUT.mkdir(parents=True, exist_ok=True)
    M = PR.Model()
    log = {"n": len(items), "fail": [], "checks": {}}
    t0 = time.time()
    for k, it in enumerate(items, 1):
        out = OUT / (it["id"] + ".npz")
        if out.is_file():
            continue
        try:
            M.prepare(it["wav"])
            raw = M.decode(it["label"])
            if not log["checks"]:
                full = np.asarray(M.p.run(it["wav"], it["label"]), np.float32)
                log["checks"]["decode_vs_probe_run_max_abs_diff"] = float(np.abs(full - raw).max())
            post = SR.post_step(raw, n_blinks=0, eye=1.0, mouth=1.0)
            tmp = out.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, raw=raw.astype(np.float32), post=post.astype(np.float32), T=int(M.T), fps=60.0)
            os.replace(tmp, out)
        except Exception as ex:                       # noqa: BLE001
            log["fail"].append({"id": it["id"], "err": str(ex)[:300], "tb": traceback.format_exc()[-800:]})
            print("FAIL", it["id"], ex, flush=True)
        if k % 20 == 0 or k == len(items):
            print("[%d/%d] %.0fs" % (k, len(items), time.time() - t0), flush=True)
    log["seconds"] = time.time() - t0
    io.open(LANE / ("run_emoface%s.json" % ("_smoke" if lim else "")), "w", encoding="utf-8").write(json.dumps(log, indent=1))
    print("DONE fail=%d checks=%s" % (len(log["fail"]), log["checks"]), flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""g2_predict.py -- predicted HUMAN ln(amp strong / amp normal) for a realised acoustic delta, from g2_coefficients.json and
g2_boot_draws.npz (human2 lane). Library + CLI. numpy only for predictions; measuring deltas from wavs needs pyworld
(env GPTSoVits or pl_medtalk) for the register cue.

  from g2_predict import G2
  g = G2()                                            # primary: model E4_common, window W2_full, outcome brow_mean
  g.predict(d_level_db=7, d_register_st=4.7, d_env_sd_db=0.3, d_ln_dur=0.105)                 # slope-only (plan G2 wording)
  g.predict(..., with_intercept=True)
  g.delta_from_wavs("rung.wav", "c100.wav")          # realised deltas, same definitions as the human regression

CLI:
  python g2_predict.py --d_level_db 7 --d_register_st 4.7 --d_env_sd_db 0 --d_ln_dur 0.105 [--model E4_fe] [--outcome brow_inner]
  python g2_predict.py --rung rung.wav --ref c100.wav            (env with pyworld)
  python g2_predict.py --selftest                                 (re-derives the worked examples stored in the json)
Returns dict(estimate, ci95_percentile, ci95_normal, coef_vector, key).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CUES = ["d_level_db", "d_register_st", "d_env_sd_db", "d_ln_dur", "d_span_st"]


class G2:
    def __init__(self, model="E4_common", window="W2_full", outcome="brow_mean", json_path=None, draws_path=None):
        self.J = json.loads(Path(json_path or HERE / "g2_coefficients.json").read_text(encoding="utf-8"))
        self.blk = self.J["models"][model][window][outcome]
        self.key = f"{model}|{window}|{outcome}"
        dp = Path(draws_path or HERE / "g2_boot_draws.npz")
        self.draws = np.load(dp)[self.key].astype(float) if dp.is_file() else None

    def vector(self, delta, with_intercept=False):
        names = self.blk["coef_names"]
        v = np.zeros(len(names))
        for k, val in delta.items():
            if k in names:
                v[names.index(k)] = float(val)
            elif abs(float(val)) > 0 and k in CUES:
                raise KeyError(f"{k} is not a predictor of {self.key}")
        if with_intercept:
            if "intercept" in names:
                v[names.index("intercept")] = 1.0
            else:
                idx = [i for i, n in enumerate(names) if n.startswith("a_")]
                v[idx] = 1.0 / max(len(idx), 1)
        return v

    def predict(self, with_intercept=False, **delta):
        v = self.vector(delta, with_intercept)
        b = np.asarray(self.blk["coef"], float)
        C = np.asarray(self.blk["cov_boot"], float)
        est = float(b @ v)
        se = float(math.sqrt(max(v @ C @ v, 0.0)))
        out = {"key": self.key, "delta": delta, "with_intercept": with_intercept, "estimate": est, "x_ratio": math.exp(est),
               "ci95_normal": [est - 1.96 * se, est + 1.96 * se], "coef_vector": v.tolist()}
        if self.draws is not None:
            d = self.draws @ v
            out["ci95_percentile"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        return out

    @staticmethod
    def measure_wav(path):
        """LEVEL_DB, ENV_SD_DB, SPEECH_DUR_S (level_def.py) + f0_med_hz and phrase_span_st exactly as human/acoustic.py."""
        import soundfile as sf
        sys.path.insert(0, str(HERE))
        import level_def as LD
        y, sr = sf.read(str(path), dtype="float64")
        m = LD.measure(y, sr, alternatives=False)
        sys.path.insert(0, "${PROJECT_ROOT}/pipeline/lilthead")
        sys.path.insert(0, "${PROJECT_ROOT}/pipeline")
        import extract_prosody as EP
        import h2_prosody_probe as P1
        EP.OUT = HERE / "_unused_prosody_dir"
        y16 = LD.to16k(y, sr)
        A = EP.analyse(y16.astype(np.float32), LD.SR)
        f0 = np.asarray(A["f0_hz_5ms"], float)
        lf0, voiced = P1.interp_unvoiced(f0)
        t5 = np.arange(len(f0)) * 0.005
        in5 = (t5 >= m["t_on"]) & (t5 <= m["t_off"])
        m["f0_med_hz"] = m["phrase_span_st"] = float("nan")
        if lf0 is not None and (voiced & in5).sum() >= 20:
            med = float(np.median(lf0[voiced & in5]))
            st = (lf0 - med) * 12.0 / np.log(2.0)
            win = max(3, int(round(P1.PHRASE_SMOOTH_MS / 5.0)) | 1)
            ph = P1.smooth(st, win)
            vv = voiced & in5
            m["f0_med_hz"] = float(np.exp(med))
            m["phrase_span_st"] = float(np.percentile(ph[vv], 95) - np.percentile(ph[vv], 5))
        return m

    def delta_from_wavs(self, rung, ref):
        a, b = self.measure_wav(rung), self.measure_wav(ref)
        return {"d_level_db": a["level_db"] - b["level_db"], "d_register_st": 12 * math.log2(a["f0_med_hz"] / b["f0_med_hz"]),
                "d_env_sd_db": a["env_sd_db"] - b["env_sd_db"], "d_ln_dur": math.log(a["speech_dur_s"] / b["speech_dur_s"]),
                "d_span_st": a["phrase_span_st"] - b["phrase_span_st"], "rung": a, "ref": b}


def selftest():
    g = G2()
    bad = 0
    for k, ex in g.J["worked_examples"].items():
        d = {c: v for c, v in ex["delta"].items() if c in CUES}
        for mode, flag in (("slope_only", False), ("with_intercept", True)):
            r = g.predict(with_intercept=flag, **d)
            ok = abs(r["estimate"] - ex[mode][0]) < 1e-4 and all(abs(a - b) < 1e-4 for a, b in zip(r["ci95_percentile"], ex[mode][1:]))
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {k:70s} {mode:15s} {r['estimate']:+.4f} {r['ci95_percentile']}")
    print("selftest", "PASS" if bad == 0 else f"FAIL ({bad})")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    for c in CUES:
        ap.add_argument("--" + c, type=float, default=0.0)
    ap.add_argument("--model", default="E4_common")
    ap.add_argument("--window", default="W2_full")
    ap.add_argument("--outcome", default="brow_mean")
    ap.add_argument("--with_intercept", action="store_true")
    ap.add_argument("--rung")
    ap.add_argument("--ref")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    g = G2(a.model, a.window, a.outcome)
    if a.rung and a.ref:
        d = g.delta_from_wavs(a.rung, a.ref)
        delta = {c: d[c] for c in CUES[:4]}
        print(json.dumps({k: d[k] for k in CUES}, indent=1))
    else:
        delta = {c: getattr(a, c) for c in CUES if getattr(a, c) != 0.0}
    print(json.dumps(g.predict(with_intercept=a.with_intercept, **delta), indent=1))


if __name__ == "__main__":
    main()

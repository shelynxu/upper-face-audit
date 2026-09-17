# -*- coding: utf-8 -*-
"""mixed_model_family.py -- the LMM stated as the FROZEN family, not as its components.

mixed_model.py fits the three per-screen contrasts. The frozen family is derived from them:
    P1 = H vs E pooled over voice level   -> the INTERCEPT of the H_vs_E screens
    P2 = H vs R at the strong voice        -> the intercept of the H_vs_R@strong screens
    P3 = strong minus normal               -> the LEVEL coefficient on the H_vs_E screens
so P1 and P3 come from one model with level as a fixed effect, which is the correct crossed
analogue of the two participant-level quantities the plan defines.

Also refits with a second optimizer, because the lbfgs run warned about convergence.

RUN: python mixed_model_family.py
"""
import os
import sys
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mixed_model as L6  # noqa: E402

from statsmodels.regression.mixed_linear_model import MixedLM  # noqa: E402


def fit(formula, d, tag):
    d = d.copy()
    d["g"] = 1
    best = None
    for meth in ("lbfgs", "bfgs", "cg", "powell"):
        try:
            f = MixedLM.from_formula(
                formula, groups="g", data=d,
                vc_formula={"participant": "0 + C(participant)",
                            "voice_pair": "0 + C(voice_pair)"}).fit(reml=True, method=meth)
            conv = bool(getattr(f, "converged", False))
            if best is None or (conv and not best[1]):
                best = (f, conv, meth)
            if conv:
                break
        except Exception:
            continue
    if best is None:
        return None
    f, conv, meth = best
    return {"fit": f, "converged": conv, "method": meth, "tag": tag}


def zp(f, term):
    b, se = f.params[term], f.bse[term]
    z = b / se
    return b, se, z, 2 * (1 - stats.norm.cdf(abs(z)))


def holm(ps):
    ps = np.asarray(ps, float); k = len(ps); adj = np.empty(k); run = 0.0
    for rank, idx in enumerate(np.argsort(ps)):
        run = max(run, ps[idx] * (k - rank)); adj[idx] = min(run, 1.0)
    return adj


def main():
    df = L6.trial_frame()
    he = df[df.contrast.isin(["H_vs_E@normal", "H_vs_E@strong"])].copy()
    he["strong"] = (he.contrast == "H_vs_E@strong").astype(float)
    hr = df[df.contrast == "H_vs_R@strong"].copy()

    print("H vs E screens %d | H vs R screens %d | participants %d | voice pairs %d\n"
          % (len(he), len(hr), df.participant.nunique(), df.voice_pair.nunique()))

    m1 = fit("score_H ~ 1 + strong", he, "P1/P3")
    m2 = fit("score_H ~ 1", hr, "P2")
    if not m1 or not m2:
        print("model did not fit")
        return 1

    # P1 = the pooled H-vs-E effect. With `strong` coded 0/1 the intercept is the NORMAL level,
    # so centre it: the intercept of the centred model is the mean over the two levels, which is
    # exactly the plan's P1.
    hec = he.copy()
    hec["strong_c"] = hec.strong - 0.5
    m1c = fit("score_H ~ 1 + strong_c", hec, "P1/P3 centred")

    out = {}
    b, se, z, p = zp(m1c["fit"], "Intercept")
    out["P1"] = (b, se, p)
    b, se, z, p = zp(m1c["fit"], "strong_c")
    out["P3"] = (b, se, p)
    b, se, z, p = zp(m2["fit"], "Intercept")
    out["P2"] = (b, se, p)

    print("crossed LMM  score_H ~ 1 + level + (1|participant) + (1|voice_pair)")
    print("  optimizer %s converged=%s / %s converged=%s"
          % (m1c["method"], m1c["converged"], m2["method"], m2["converged"]))
    print()
    frozen = {"P1": (0.117, 0.201), "P2": (0.066, 0.355), "P3": (0.181, 0.201)}
    ps = [out[k][2] for k in ("P1", "P2", "P3")]
    hp = holm(ps)
    fp = holm([frozen[k][1] for k in ("P1", "P2", "P3")])
    print("%-5s %-28s %-28s %-14s %s" % ("", "FROZEN (by-subject t)", "CROSSED LMM", "Holm frozen", "Holm LMM"))
    for i, k in enumerate(("P1", "P2", "P3")):
        fb, fpv = frozen[k]
        b, se, p = out[k]
        print("%-5s %-28s %-28s %-14s %s"
              % (k, "%+.3f  p %.4f" % (fb, fpv),
                 "%+.3f (SE %.3f)  p %.4f" % (b, se, p),
                 "%.4f" % fp[i], "%.4f %s" % (hp[i], "SUPPORTED" if hp[i] < .05 else "not supported")))

    print("\nmute / visibility check, same model")
    mu = df[df.contrast == "mute:H_vs_E@strong"]
    mm = fit("score_H ~ 1", mu, "MC")
    b, se, z, p = zp(mm["fit"], "Intercept")
    print("  MC  %+.3f (SE %.3f)  p %.3g   [converged=%s]" % (b, se, p, mm["converged"]))

    print("\nverdict")
    print("  frozen supports : %s" % ([k for i, k in enumerate(('P1','P2','P3')) if fp[i] < .05] or "NONE"))
    print("  LMM supports    : %s" % ([k for i, k in enumerate(('P1','P2','P3')) if hp[i] < .05] or "NONE"))


if __name__ == "__main__":
    raise SystemExit(main())

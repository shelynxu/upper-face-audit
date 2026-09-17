# -*- coding: utf-8 -*-
"""mixed_model.py -- would a linear mixed-effects model change the v6 perception conclusion, and does
the DV need per-participant normalisation?

Two questions, answered by fitting the models rather than by opinion.

WHAT THE FROZEN ANALYSIS DOES
    per participant, the mean of their screens in a contrast -> one-sample t -> Holm over P1-P3.
    This is the classic "F1" / by-subject analysis. It treats the 12 voice pairs as FIXED.

WHY AN LMM MIGHT BE BETTER
    Each screen is one of 12 voice pairs, crossed with participants. If the paper wants to
    generalise beyond these 12 voices -- and it does, it says "in English acted speech" -- then
    items are a random factor too (Clark 1973, the language-as-fixed-effect fallacy). The
    by-subject test alone is anticonservative for that claim. A crossed model
        score_H ~ 1 + (1 | participant) + (1 | voice_pair)
    is the modern answer; min-F' is the classical one and needs no fitting.

ON NORMALISATION -- THE TRAP WORTH NAMING
    score_H is a SIGNED comparison score, already oriented toward H, on -3..+3, and ZERO MEANS
    "the two faces look the same". The hypothesis IS "the mean is above zero".
    So a per-participant z-score -- subtracting each person's own mean -- SUBTRACTS EXACTLY THE
    QUANTITY UNDER TEST and forces the group mean to ~0 by construction. That is not a
    conservative choice, it is a destroyed one. It is included below only to show the damage.
    The defensible ways to handle scale-use differences keep the zero anchor:
      scale-divided : divide by the participant's own SD over ALL their scored screens (spread
                      only, no centring)
      rank / sign   : Wilcoxon and the sign test -- scale-free by construction; the frozen
                      analyser already reports both
      ordinal       : treat the 7 points as ordered categories

RUN: python mixed_model.py
"""
import collections
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robustness_specification_curve as R  # noqa: E402   (reuses the score_H derivation validated against the analyser)

PRIMARY = R.PRIMARY
CONTRASTS = list(PRIMARY) + ["mute:H_vs_E@strong"]


def trial_frame():
    """One row per SCORED screen, with participant, voice pair and contrast."""
    files = [f for f in sorted(os.listdir(R.W))
             if f.endswith(".csv") and "_all_part" not in f and not f.lower().startswith("test")]
    catch_ok = {a["code"] for a in R.AUDIT if a["catch_pass"] >= a["catch_n"]}
    grp = {a["code"]: a["group"] for a in R.AUDIT}
    rows = []
    for f in files:
        code = f.split("_")[0][:24]
        if code not in catch_ok:
            continue
        for r in R.rows_of(f):
            if r.get("block") in ("pair_catch", "practice"):
                continue
            if (r.get("revision") or "0").strip() not in ("", "0"):
                continue           # first response only, as the analyser does
            c = r.get("voice_arm") or ""
            if c not in CONTRASTS:
                continue
            if (r.get("focus_hidden") or "").strip() not in ("", "0"):
                continue           # the plan's primary screen rule
            sh = R.score_H(r)
            if sh is None:
                continue
            rows.append({
                "participant": code, "group": grp.get(code, ""),
                "voice_pair": "%s_%s" % (r.get("voice_id"), r.get("voice_emotion")),
                "contrast": c, "score_H": sh,
            })
    return pd.DataFrame(rows)


def by_unit_t(df, unit):
    m = df.groupby(unit)["score_H"].mean()
    t, p = stats.ttest_1samp(m, 0)
    return len(m), m.mean(), m.std(ddof=1), t, p


def min_f_prime(df):
    """Clark's min-F'. Conservative test of the intercept against BOTH random factors, from the
    two one-way analyses the frozen analyser already runs. No model fitting."""
    n1, m1, s1, t1, p1 = by_unit_t(df, "participant")
    n2, m2, s2, t2, p2 = by_unit_t(df, "voice_pair")
    f1, f2 = t1 ** 2, t2 ** 2
    if f1 + f2 == 0:
        return None
    mf = (f1 * f2) / (f1 + f2)
    dfn = 1
    dfd = ((f1 + f2) ** 2) / (f1 ** 2 / (n2 - 1) + f2 ** 2 / (n1 - 1))
    p = 1 - stats.f.cdf(mf, dfn, dfd)
    return {"F1": f1, "p1": p1, "n1": n1, "F2": f2, "p2": p2, "n2": n2,
            "minF": mf, "df": dfd, "p_minF": p}


def lmm(df):
    """score_H ~ 1 + (1|participant) + (1|voice_pair), crossed, via a dummy group + vc."""
    from statsmodels.regression.mixed_linear_model import MixedLM
    d = df.copy()
    d["g"] = 1
    md = MixedLM.from_formula(
        "score_H ~ 1", groups="g", data=d,
        vc_formula={"participant": "0 + C(participant)", "voice_pair": "0 + C(voice_pair)"})
    fit = md.fit(reml=True, method="lbfgs")
    b = fit.params["Intercept"]
    se = fit.bse["Intercept"]
    z = b / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    vc = {k: v for k, v in fit.params.items() if "Var" in k}
    return {"beta": b, "se": se, "z": z, "p": p, "vc": vc,
            "n_obs": len(d), "n_p": d.participant.nunique(), "n_i": d.voice_pair.nunique()}


def holm(ps):
    ps = np.asarray(ps, float)
    k = len(ps)
    adj = np.empty(k)
    run = 0.0
    for rank, idx in enumerate(np.argsort(ps)):
        run = max(run, ps[idx] * (k - rank))
        adj[idx] = min(run, 1.0)
    return adj


def main():
    df = trial_frame()
    print("scored screens %d | participants %d | voice pairs %d | groups %s\n"
          % (len(df), df.participant.nunique(), df.voice_pair.nunique(),
             dict(collections.Counter(df.drop_duplicates('participant').group))))

    # ---------------- Q1: does the model choice change the answer? ----------------
    print("=" * 108)
    print("Q1  BY-SUBJECT (frozen)  vs  BY-ITEM  vs  min-F'  vs  CROSSED LMM")
    print("=" * 108)
    print("%-22s %-26s %-26s %-22s %s"
          % ("contrast", "by-subject (frozen)", "by-item", "min-F'", "crossed LMM"))
    raw_p, lmm_p = {}, {}
    for c in CONTRASTS:
        d = df[df.contrast == c]
        n1, m1, s1, t1, p1 = by_unit_t(d, "participant")
        n2, m2, s2, t2, p2 = by_unit_t(d, "voice_pair")
        mf = min_f_prime(d)
        L = lmm(d)
        raw_p[c], lmm_p[c] = p1, L["p"]
        print("%-22s %-26s %-26s %-22s %s"
              % (c[:22],
                 "%+.3f  p %.4f (n%d)" % (m1, p1, n1),
                 "%+.3f  p %.4f (n%d)" % (m2, p2, n2),
                 "p %.4f" % mf["p_minF"],
                 "%+.3f (SE %.3f) p %.4f" % (L["beta"], L["se"], L["p"])))

    print("\nHolm over the three confirmatory hypotheses")
    keys = list(PRIMARY)
    h_raw = holm([raw_p[k] for k in keys])
    h_lmm = holm([lmm_p[k] for k in keys])
    print("%-22s %14s %14s   %s" % ("", "frozen Holm p", "LMM Holm p", "supported?"))
    for k, a, b in zip(keys, h_raw, h_lmm):
        print("%-22s %14.4f %14.4f   frozen %-3s  LMM %s"
              % (k[:22], a, b, "YES" if a < .05 else "no", "YES" if b < .05 else "no"))

    print("\nvariance components of the crossed LMM (how much is person, how much is item)")
    for c in CONTRASTS:
        L = lmm(df[df.contrast == c])
        vc = L["vc"]
        tot = sum(vc.values())
        print("  %-22s %s" % (c[:22], "  ".join("%s %.3f (%.0f%%)" % (k.replace(" Var", ""), v, 100 * v / tot)
                                                for k, v in vc.items())))

    # ---------------- Q2: normalisation ----------------
    print("\n" + "=" * 108)
    print("Q2  PER-PARTICIPANT NORMALISATION OF THE PERCEPTUAL RESPONSES")
    print("=" * 108)
    sd_all = df.groupby("participant")["score_H"].std(ddof=1)
    mu_all = df.groupby("participant")["score_H"].mean()
    print("participants' own scale use: SD over all their scored screens  min %.2f  med %.2f  max %.2f"
          % (sd_all.min(), sd_all.median(), sd_all.max()))
    print("                             mean over all their screens       min %+.2f med %+.2f max %+.2f"
          % (mu_all.min(), mu_all.median(), mu_all.max()))

    variants = {}
    variants["raw (frozen)"] = df.assign(y=df.score_H)
    d2 = df.copy()
    d2["y"] = d2.score_H / d2.participant.map(sd_all)          # spread only, zero kept
    variants["scale-divided (SD, no centring)"] = d2
    d3 = df.copy()
    d3["y"] = (d3.score_H - d3.participant.map(mu_all)) / d3.participant.map(sd_all)
    variants["z-scored (centred -- DESTROYS IT)"] = d3

    print("\n%-38s %s" % ("variant", "  ".join("%-20s" % c[:20] for c in CONTRASTS)))
    for name, d in variants.items():
        cells = []
        for c in CONTRASTS:
            g = d[d.contrast == c].groupby("participant")["y"].mean()
            t, p = stats.ttest_1samp(g, 0)
            cells.append("%+.3f p %.4f" % (g.mean(), p))
        print("%-38s %s" % (name, "  ".join("%-20s" % x for x in cells)))

    print("\nscale-free tests on the frozen DV (already in the analyser, repeated here)")
    for c in CONTRASTS:
        g = df[df.contrast == c].groupby("participant")["score_H"].mean()
        try:
            w = stats.wilcoxon(g)[1]
        except Exception:
            w = float("nan")
        pos = int((g > 0).sum()); neg = int((g < 0).sum())
        sgn = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
        print("  %-22s Wilcoxon p %.4f | sign %d+/%d- p %.4f" % (c[:22], w, pos, neg, sgn))


if __name__ == "__main__":
    main()

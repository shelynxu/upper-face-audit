# -*- coding: utf-8 -*-
"""robustness_specification_curve.py -- does the pre-registered v6 conclusion survive removing the least engaged
participants? A specification curve, not a new primary analysis.

READ engagement_audit.py's header first. prereg/ANALYSIS_PLAN.md section 5 item 5 forbids exclusion
by response time, replays or response pattern, so NONE of the sets below may replace the primary
sample. The only legitimate reading is: "the null survives / does not survive" -- and a set that
turned the null positive would be an unusable, outcome-dependent artefact, not a finding.

SELF-VALIDATION. Before reporting anything the script re-derives the FROZEN analyser's published
numbers from the raw CSVs on the same 33-participant input (P1 +0.122, P2 +0.098, P3 +0.215,
MC +0.876) and refuses to continue if it cannot. That is what makes the rest of the table
trustworthy without re-running the hashed analyser.

The plan's own screen and participant rules are implemented here as written:
  screen rule      drop a scored screen with focus_hidden >= 1 (primary)
  participant rule < 75 % retained in ANY of the three primary AV contrasts -> drop entirely
  P1  H_vs_E pooled: per participant the mean of its normal and strong means
  P2  H_vs_R@strong
  P3  H_vs_E@strong - H_vs_E@normal
  MC  mute:H_vs_E@strong
  support = Holm p < .05 over {P1,P2,P3}, two-sided, AND mean > 0

RUN: python robustness_specification_curve.py
"""
import collections
import csv
import io
import json
import os

import numpy as np
from scipy import stats

W = "${PROJECT_ROOT}/webapp/data/lilt_link/whole_response"
HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = json.load(io.open(os.path.join(HERE, "engagement_audit.json"), encoding="utf-8"))
BY_CODE = {a["code"]: a for a in AUDIT}

PRIMARY = ("H_vs_E@normal", "H_vs_E@strong", "H_vs_R@strong")


def rows_of(f):
    with io.open(os.path.join(W, f), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def score_H(r):
    """The raw CSV has no score_H column -- the frozen analyser derives it. Same derivation here:
    cue_manip says which side carried H ("L:H|R:E" -> left), response_emotion is the 7-point
    comparison as L1/L2/L3 | S0 | R1/R2/R3, and the score is signed TOWARD H."""
    cm = r.get("cue_manip") or ""
    resp = (r.get("response_emotion") or "").strip()
    if not resp or resp == "S0":
        return 0.0
    if resp[0] not in "LR" or not resp[1:].isdigit():
        return None
    mag = float(resp[1:])
    chose = "left" if resp[0] == "L" else "right"
    h_side = "left" if cm.startswith("L:H") else ("right" if "|R:H" in cm else None)
    if h_side is None:
        return None
    return mag if chose == h_side else -mag


def participant_vectors(files, use_focus_rule=True):
    """-> {code: (P1, P2, P3, MC or nan)} applying the plan's screen and participant rules."""
    out, dropped = {}, []
    for f in files:
        rows = rows_of(f)
        code = f.split("_")[0][:24]
        keep = collections.defaultdict(list)
        served = collections.Counter()
        mute = []
        for r in rows:
            c = r.get("voice_arm") or ""          # the raw CSV calls the contrast voice_arm
            if r.get("block") in ("pair_catch", "practice"):
                continue
            # the DV is the FIRST response: 17 screens across the sample were revised
            # (revision=1) and the analyser scores revision 0 only.
            if (r.get("revision") or "0").strip() not in ("", "0"):
                continue
            hidden = (r.get("focus_hidden") or "").strip()
            lost = hidden not in ("", "0") if use_focus_rule else False
            sh = score_H(r)
            if sh is None:
                continue
            if c in PRIMARY:
                served[c] += 1
                if not lost:
                    keep[c].append(sh)
            elif c == "mute:H_vs_E@strong":
                if not lost:
                    mute.append(sh)
        # participant rule: >= 75 % retained in every primary contrast
        bad = [c for c in PRIMARY if served[c] and len(keep[c]) / served[c] < 0.75]
        if bad:
            dropped.append((code, bad))
            continue
        if not all(keep[c] for c in PRIMARY):
            dropped.append((code, ["empty"]))
            continue
        n_, s_, r_ = (np.mean(keep["H_vs_E@normal"]), np.mean(keep["H_vs_E@strong"]),
                      np.mean(keep["H_vs_R@strong"]))
        out[code] = ((n_ + s_) / 2.0, r_, s_ - n_,
                     float(np.mean(mute)) if mute else float("nan"))
    return out, dropped


def holm(X):
    ps, ms = [], []
    for i in range(3):
        t, p = stats.ttest_1samp(X[:, i], 0)
        ps.append(p); ms.append(X[:, i].mean())
    ps = np.array(ps); adj = np.empty(3); run = 0.0
    for rank, idx in enumerate(np.argsort(ps)):
        run = max(run, ps[idx] * (3 - rank)); adj[idx] = min(run, 1.0)
    return np.array(ms), adj, (adj < .05) & (np.array(ms) > 0)


def report(label, codes, vecs):
    sub = [vecs[c] for c in codes if c in vecs]
    if len(sub) < 5:
        print("%-42s n=%-3d (too small)" % (label, len(sub)))
        return None
    A = np.array(sub)
    ms, adj, sup = holm(A[:, :3])
    mc = A[:, 3]; mc = mc[~np.isnan(mc)]
    tmc, pmc = stats.ttest_1samp(mc, 0)
    grp = collections.Counter(BY_CODE[c]["group"] for c in codes if c in vecs)
    print("%-42s n=%-3d %-18s P1 %+0.3f(p%.3f) P2 %+0.3f(p%.3f) P3 %+0.3f(p%.3f) | MC %+0.3f (p %.1e) | supported: %s"
          % (label, len(sub), "en%d/cn%d/jp%d" % (grp.get("en", 0), grp.get("cn", 0), grp.get("jp", 0)),
             ms[0], adj[0], ms[1], adj[1], ms[2], adj[2], mc.mean(), pmc,
             ["P1", "P2", "P3"][int(np.argmax(sup))] if sup.any() else "NONE"))
    return sup.any()


def main():
    files = [f for f in sorted(os.listdir(W))
             if f.endswith(".csv") and "_all_part" not in f and not f.lower().startswith("test")]
    vecs, dropped = participant_vectors(files)
    print("participants with usable vectors: %d   dropped by the plan's focus rule: %s\n"
          % (len(vecs), dropped or "none"))

    catch_ok = [a["code"] for a in AUDIT if a["catch_pass"] >= a["catch_n"]]
    jp = [a["code"] for a in AUDIT if a["group"] == "jp"]

    # ---- self-validation against the frozen analyser (the pre-jp sample, N=33) ----
    pre_jp = [c for c in catch_ok if c not in jp]
    A = np.array([vecs[c] for c in pre_jp if c in vecs])
    ms, adj, _ = holm(A[:, :3])
    mc = A[:, 3][~np.isnan(A[:, 3])]
    want = {"n": 33, "P1": 0.122, "P2": 0.098, "P3": 0.215, "MC": 0.876}
    got = {"n": len(A), "P1": round(ms[0], 3), "P2": round(ms[1], 3),
           "P3": round(ms[2], 3), "MC": round(mc.mean(), 3)}
    print("SELF-VALIDATION against the frozen analyser (pre-jp sample)")
    print("  frozen analyser: %s" % want)
    print("  recomputed here: %s" % got)
    if got != want:
        raise SystemExit("STOP: this reimplementation does not reproduce the frozen analyser -- "
                         "do not trust anything below it.")
    print("  MATCH -- the sensitivity table below is computed the same way.\n")

    q = {a["code"]: a["quality_flags"] for a in AUDIT}
    notpc = [c for c in catch_ok if "not-a-computer" in q.get(c, [])]
    fast = [c for c in catch_ok if "fast-session" in q.get(c, [])]
    anyflag = [c for c in catch_ok if q.get(c)]
    # the borderline case the a-priori thresholds each just missed
    border = [a["code"] for a in AUDIT
              if a["catch_pass"] >= a["catch_n"]
              and a["same_rate"] > 0.6 and a["max_run"] >= 15 and (a["frac_fast"] or 0) > 0.2]
    focusflag = [a["code"] for a in AUDIT
                 if a["catch_pass"] >= a["catch_n"] and (a["focus_lost"] or 0) > 0]

    print("exclusion sets (each nested inside the next)")
    print("  not-a-computer : %s" % notpc)
    print("  fast-session   : %s" % fast)
    print("  borderline     : %s  (same>0.6 AND run>=15 AND >20%% sub-450ms; each threshold alone missed it)" % border)
    print("  focus_lost>0   : %s\n" % focusflag)

    print("=== SPECIFICATION CURVE ===")
    report("S0 PRE-REGISTERED PRIMARY (catch-passers)", catch_ok, vecs)
    report("S1  - not-a-computer", [c for c in catch_ok if c not in notpc], vecs)
    report("S2  - fast-session", [c for c in catch_ok if c not in fast], vecs)
    report("S3  - any quality flag", [c for c in catch_ok if c not in anyflag], vecs)
    report("S4  - any flag - borderline", [c for c in catch_ok if c not in anyflag + border], vecs)
    report("S5  - any flag - borderline - focus", [c for c in catch_ok if c not in anyflag + border + focusflag], vecs)
    report("S6  English viewers only", [c for c in catch_ok if BY_CODE[c]["group"] == "en"], vecs)
    report("S7  PC users only, any group", [c for c in catch_ok if BY_CODE[c]["device"] == "PC"], vecs)

    print("\n=== the cn group's device problem ===")
    for g in ("cn", "en", "jp"):
        a = [x for x in AUDIT if x["group"] == g and x["catch_pass"] >= x["catch_n"]]
        pc = sum(1 for x in a if x["device"] == "PC")
        print("  %s: %d valid, %d on a computer, %d on a phone/tablet (%.0f%%)"
              % (g, len(a), pc, len(a) - pc, 100.0 * (len(a) - pc) / max(len(a), 1)))


if __name__ == "__main__":
    main()

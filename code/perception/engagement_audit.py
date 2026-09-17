# -*- coding: utf-8 -*-
"""engagement_audit.py -- a SCORE-BLIND engagement audit of every completed v6 session.

WHY THIS IS A SENSITIVITY ANALYSIS AND NOTHING ELSE
prereg/ANALYSIS_PLAN.md section 5 item 5 is explicit and frozen:
    "Nothing else: no exclusion by response time, replays, response pattern or outcome."
So no flag in this file may ever move the PRIMARY sample. The primary sample stays exactly what
the pre-registration says: not-a-participant / not-eligible / catch-failed / focus rules. What a
quality audit can legitimately do is ask ONE question:

    does the pre-registered conclusion survive when the least engaged participants are removed?

That question has an asymmetric answer space, and it is important to say so before running it:
  * If the null SURVIVES every exclusion, that is a real strengthening statement and it can go
    in the paper as one robustness clause.
  * If an exclusion TURNS the null into a positive, that result is unusable -- it would be an
    outcome-dependent, post-hoc, plan-violating exclusion, and reporting it as a finding would
    be worse than reporting nothing.
So this audit can only ever confirm the null or produce an unreportable artefact. It is worth
running for the first outcome.

SCORE-BLINDNESS. Every flag below is computed from metadata or from response STYLE, never from
whether the hypothesised arm won. Specifically, side bias is |left - right| / n, which is blind
to which side H was on (H's side is rotated within participant, 6 of 12 per contrast), and the
"same" rate and the emotion-response entropy say nothing about H either.

THRESHOLDS are fixed here before any sensitivity model is fitted, and each is anchored to
something external (the design's own estimate, a physical floor, or the observed reference
distribution's own tails), not chosen to produce a result.

RUN: python engagement_audit.py
"""
import collections
import csv
import io
import json
import math
import os
import statistics as st

W = "${PROJECT_ROOT}/webapp/data/lilt_link/whole_response"
M = "${PROJECT_ROOT}/webapp/data/lilt_link/middle_response"

# --- thresholds, fixed before any model is fitted ---------------------------
DESIGN_MIN = 19.0      # manifest est_minutes: the design's own time estimate
FAST_TASK_MIN = 9.5    # half the design estimate: not enough time to watch 81 clips
RT_FLOOR_MS = 450      # a pair screen needs both clips watched; under this is not a judgement
FAST_RT_FRac = 0.30    # flagged if >30 % of scored screens are under the floor
SIDE_BIAS_MAX = 0.55   # |L-R|/n above this is a side habit, not a comparison
SAME_MAX = 0.80        # answering "they are the same" on >80 % of pairs is a non-answer
RUN_MAX = 20           # longest identical-chosen_side run
ENTROPY_MIN = 1.0      # bits, over the 7-choice emotion answers on single/voice screens


def load_rows(p):
    with io.open(os.path.join(W, p), encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def infos_for(csvname):
    stem = csvname.split("__")[0]
    p = os.path.join(M, stem + "_infos.json")
    if os.path.isfile(p):
        return json.load(io.open(p, encoding="utf-8"))
    return {}


def entropy(vals):
    if not vals:
        return 0.0
    c = collections.Counter(vals)
    n = sum(c.values())
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def longest_run(seq):
    best = cur = 0
    prev = None
    for x in seq:
        cur = cur + 1 if x == prev else 1
        prev = x
        best = max(best, cur)
    return best


def audit():
    files = [f for f in sorted(os.listdir(W))
             if f.endswith(".csv") and "_all_part" not in f and not f.lower().startswith("test")]
    out = []
    for f in files:
        rows = load_rows(f)
        j = infos_for(f)
        rs = j.get("response_summary") or {}
        tm = j.get("timing") or {}
        cn = j.get("counts") or {}
        bg = j.get("background") or {}
        dev = j.get("device") or {}
        cons = j.get("consistency") or {}

        scored = [r for r in rows if r.get("block") not in ("practice", "pair_catch")]
        pairs = [r for r in scored if r.get("chosen_side")]
        side = collections.Counter(r["chosen_side"] for r in pairs)
        npair = sum(side.values()) or 1
        rts = [int(r["rt_ms"]) for r in scored if (r.get("rt_ms") or "").strip().isdigit()]
        emo = [r.get("response_emotion") for r in rows
               if r.get("block") in ("single", "voice_only") and (r.get("response_emotion") or "").strip()]

        catch = [r for r in rows if r.get("block") == "pair_catch"]
        cpass = sum(1 for r in catch
                    if r.get("chosen_side") and "correct:" + r["chosen_side"] == (r.get("voice_arm") or ""))

        rec = {
            "file": f,
            "code": f.split("_")[0][:24],
            "group": (rows[0].get("culture") if rows else "") or "",
            "n_rows": len(rows),
            "catch_pass": cpass, "catch_n": len(catch),
            "task_min": tm.get("task_minutes"),
            "total_min": tm.get("total_minutes"),
            "median_rt": rs.get("median_rt_ms") or (st.median(rts) if rts else None),
            "frac_fast": (sum(1 for x in rts if x < RT_FLOOR_MS) / len(rts)) if rts else None,
            "replays": rs.get("total_replays"),
            "focus_lost": rs.get("focus_lost_events"),
            "focus_away_s": rs.get("focus_away_seconds"),
            "revisions": rs.get("revisions"),
            "reentries": cn.get("reloads_or_reentries"),
            "side_bias": abs(side.get("left", 0) - side.get("right", 0)) / npair,
            "same_rate": side.get("same", 0) / npair,
            "max_run": longest_run([r["chosen_side"] for r in pairs]),
            "emo_entropy": entropy(emo),
            "native": bg.get("native_lang"), "hearing": bg.get("hearing"), "vision": bg.get("vision"),
            "device": dev.get("kind"), "viewport": dev.get("viewport"),
            "cons_review": cons.get("review"),
        }
        flags = []
        if rec["catch_pass"] < rec["catch_n"]:
            flags.append("CATCH(prereg)")
        if rec["task_min"] is not None and rec["task_min"] < FAST_TASK_MIN:
            flags.append("fast-session")
        if rec["frac_fast"] is not None and rec["frac_fast"] > FAST_RT_FRac:
            flags.append("fast-rt")
        if rec["side_bias"] > SIDE_BIAS_MAX:
            flags.append("side-habit")
        if rec["same_rate"] > SAME_MAX:
            flags.append("all-same")
        if rec["max_run"] > RUN_MAX:
            flags.append("long-run")
        if rec["emo_entropy"] < ENTROPY_MIN:
            flags.append("low-entropy")
        if (rec["focus_lost"] or 0) > 0:
            flags.append("focus(prereg-sens)")
        if rec["native"] not in ("cn", "en", "jp"):
            flags.append("ELIGIBILITY(prereg)")
        if rec["hearing"] == "impaired" or rec["vision"] == "other":
            flags.append("ELIGIBILITY(prereg)")
        if rec["device"] not in ("PC", None, ""):
            flags.append("not-a-computer")
        rec["flags"] = flags
        rec["quality_flags"] = [x for x in flags if "prereg" not in x]
        out.append(rec)
    return out


def main():
    a = audit()
    print("completed non-test sessions: %d   by group: %s\n"
          % (len(a), dict(collections.Counter(r["group"] for r in a))))

    print("=== reference distribution of each measure (all %d) ===" % len(a))
    for k, fmt in (("task_min", "%.1f"), ("median_rt", "%.0f"), ("frac_fast", "%.3f"),
                   ("replays", "%.0f"), ("side_bias", "%.2f"), ("same_rate", "%.2f"),
                   ("max_run", "%.0f"), ("emo_entropy", "%.2f")):
        v = sorted(x[k] for x in a if x[k] is not None)
        if not v:
            continue
        print("  %-12s min %s  p10 %s  med %s  p90 %s  max %s"
              % (k, fmt % v[0], fmt % v[len(v)//10], fmt % v[len(v)//2],
                 fmt % v[9*len(v)//10], fmt % v[-1]))

    print("\n=== per participant ===")
    hdr = ("code", "grp", "rows", "catch", "task_m", "rt_med", "%fast", "repl", "focus",
           "sideb", "same", "run", "H(emo)", "dev", "flags")
    print("%-24s %-3s %4s %5s %6s %6s %5s %5s %5s %5s %5s %4s %6s %-6s %s" % hdr)
    for r in sorted(a, key=lambda x: (-len(x["quality_flags"]), x["group"], x["code"])):
        print("%-24s %-3s %4d %2d/%-2d %6s %6s %5s %5s %5s %5.2f %5.2f %4d %6.2f %-6s %s"
              % (r["code"], r["group"], r["n_rows"], r["catch_pass"], r["catch_n"],
                 ("%.1f" % r["task_min"]) if r["task_min"] is not None else "-",
                 ("%.0f" % r["median_rt"]) if r["median_rt"] is not None else "-",
                 ("%.2f" % r["frac_fast"]) if r["frac_fast"] is not None else "-",
                 r["replays"] if r["replays"] is not None else "-",
                 r["focus_lost"] if r["focus_lost"] is not None else "-",
                 r["side_bias"], r["same_rate"], r["max_run"], r["emo_entropy"],
                 (r["device"] or "?")[:6], ",".join(r["flags"]) or "-"))

    print("\n=== summary ===")
    prereg_out = [r for r in a if any("prereg)" in f and "sens" not in f for f in r["flags"])]
    qual = [r for r in a if r["quality_flags"]]
    print("pre-registered exclusions (catch / eligibility): %d -> %s"
          % (len(prereg_out), [r["code"] for r in prereg_out]))
    print("score-blind QUALITY flags (sensitivity only): %d" % len(qual))
    for r in qual:
        print("   %-24s %s" % (r["code"], ",".join(r["quality_flags"])))
    io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "engagement_audit.json"),
            "w", encoding="utf-8").write(json.dumps(a, ensure_ascii=False, indent=1))
    print("\nwrote engagement_audit.json")


if __name__ == "__main__":
    main()

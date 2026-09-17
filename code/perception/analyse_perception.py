# -*- coding: utf-8 -*-
"""analyse_lilt_v6.py -- analysis of the v6 perception study (build_lilt_v6.py, /root/lilt_cn). From analyse_lilt_v2a.py.

Reads webapp/data/lilt_link/whole_response/*.csv. Kept rows: mode lilt, not test*, not practice, revision 0 (the first
response), one row per participant x stimulus, and only the files of the v6 manifest (--manifest, default the live link
manifest when build_lilt_v6.py built it). The manifest is authoritative for arm, voice pair, level, actor and emotion;
the CSV columns (voice_arm, cue_manip, renderer, voice_emotion, face_motion) are the fallback.

  pairs     CCR L3..S0..R3 -> v in -3..+3 (+ = right face better); cue_manip "L:x|R:y" gives score_H = +v when H was on
            the right, -v when on the left. Contrasts H_vs_E@normal, H_vs_E@strong, H_vs_R@strong, mute:H_vs_E@strong.
            Per contrast: trial mean, preference / tie rates; PARTICIPANT level (one mean per participant): mean, sd, dz,
            participant-bootstrap 95 % CI (percentile), one-sample t (two-sided and one-sided H > 0), Wilcoxon, sign test;
            VOICE-PAIR level (one mean per voice pair) with a voice-pair bootstrap CI.
  P1        per participant: mean of its H_vs_E@normal and H_vs_E@strong means (= the pooled mean when complete) > 0
  P2        per participant: mean score_H in H_vs_R@strong > 0
  P3        per participant: mean over the voice pairs answered at both levels of score_H(strong) - score_H(normal) > 0
            (the builder keeps H on the same side at both levels of a voice pair, so side bias cancels)
  family    Holm over P1, P2, P3 (participant-level t). --sided two (default) or one (directional, p/2 in the predicted
            direction); both Holm sets are reported. Which is primary is fixed in her analysis plan [U].
  MC        manipulation check: mute:H_vs_E@strong ("which face's movement is more lively") > 0; not in the family.
  singles   per arm E | H | R: vividness (CSV `intensity`), naturalness, emotion accuracy (response == voice_emotion,
            e.g. "fear") -- participant means with bootstrap CIs, within-participant differences H-E, R-E, H-R; accuracy
            per emotion x arm.
  catch     one per participant: pass = the moving side. Every result for two samples: catch_pass (primary) and all.
  OPTIONAL  (build_lilt_v6.py --hr-normal / --voice-check, default OFF; analysed only when rows exist, never in the
            Holm family): P2b = per participant mean score_H in H_vs_R@normal > 0, P3R = (H-R)@strong - (H-R)@normal;
            voice check = within participant, over the voice pairs heard at both levels, emotion intensity (CSV
            `intensity`) of the strong take minus the normal take > 0, plus 7-AFC accuracy per level and per emotion
            (results["samples"][s]["optional"], ["voice_check"]; lilt_v6_voice_long.csv).
  EXCLUSIONS (ANALYSIS_PLAN_V6.md section 5, in this order; every count in results["flow"]):
            1 not a participant: test* codes, files outside whole_response/ (completion), revision > 0, duplicates;
            2 not eligible: the background dict of the participant's *_infos.json beside the CSV (native_lang != cn,
              hearing == impaired, vision == other; "na" / missing kept and counted);
            3 catch: the primary sample = catch passers; sample "all" = everyone surviving 1, 2 and 4;
            4 focus, per scored screen (single, pair, pair_mute, voice_only; never the catch): PRIMARY drops screens with
              focus_hidden >= 1; the SENSITIVITY run (results["sensitivity_focus_lost"]) drops focus_lost >= 1 and
              re-reports P1-P3 + MC. Participant rule after the screen drop: < 75 % retained of the served screens in
              ANY primary AV contrast -> out of every analysis; < 4 of 6 silent screens -> out of MC only; (options) < 75 %
              in H_vs_R@normal -> out of P2b/P3R only; < 4 voice pairs with both takes -> out of the voice check only.
  P2 sens.  pre-registered sensitivity subset for P2 (plan section 8): H_vs_R@strong restricted to the cells whose R is a
            clean timing control (stimuli/build/p2_sensitivity_cells.json: shown R/H amplitude in [0.8, 1.25] and
            r(H, R) < 0.3 on both brows) -> results["samples"][s]["p2_sensitivity"]; descriptive, outside the family.
  freeze    the report starts by re-checking study_v6/PLAN_FREEZE.sha256 (sha256sum -c equivalent; paths /d/..., /mnt/d/...
            and D:/... are all resolved); the outcome is printed and stored in results["plan_freeze"], never enforced.
  describe  per emotion, per actor and per voice pair: every contrast and P3 (trial means, participant means + CI).
  balance   realised_balance (DESCRIPTIVE audit; no test, estimate, exclusion or stopping rule depends on it): H-left share
            per contrast, single-arm counts per voice pair, participants sharing one arm/side vector, and the rotation
            audit. The server (LILTROT v6) assigns list k in 0..P-1 (P = lcm of arm counts, 6) at session start by
            least-filled allocation per link and build (completed + started < lilt_rotation_active_min, default 60 min,
            non-test sessions; ties -> lowest k) and stamps it in data/sessions/<key>.json -> lilt_rotation. The audit
            reads those stamps (--sessions-dir; joined to CSV participants by session_id, usernames never read out),
            reports counts per (link, build, offset) for completed non-test sessions and for the analysed participants,
            falls back to reconstructing k from the served files when no stamp exists, and warns only beyond the
            least-filled allowance: > ceil(n_link / P) + 2 on one list or vector, single-arm split > 4 per link
            (COLLAPSED when more than half of a link's n >= 4 share one list); P from lilt_rotation.period.
  long CSV  one row per scored pair: participant, voice_pair, actor, emotion, level, contrast, H_side, ccr_raw,
            score_H, rt_ms, n_replays, catch_pass -- for score_H ~ contrast + (1|participant) + (1|voice_pair).
            (Exp 2 rows too: exp = 2, stim_language, listener_group, native_stimulus; score_H there = score toward L.)

EXPERIMENT 2 (ANALYSIS_PLAN_V6_EXP2.md; block "pair_ml", served after the AV pairs and before the silent pairs):
  pairs     voice_arm "L_vs_E@<lang>" | "L_vs_Lrev@<lang>", lang = stimulus language cn / en / jp; cue_manip "L:x|R:y" with
            arms L (r5b_lilt_nonod-g2), E (before), Lrev (r5b_lilt_nonod_rev-g2); score_L = +v when L was on the right.
  P4        per participant: the mean over stimulus languages of the per-language mean score_L in L_vs_E > 0
  P5        the same for L_vs_Lrev > 0
  family    Holm over P4, P5 (separate from P1-P3). Per stimulus language: descriptives with participant CIs.
  explor.   listener group (link: cn / en / jp) x stimulus language table; native-stimulus effect = within
            participant mean score_L on the native-language cells minus the mean over the non-native cells; per group
            P4 / P5; side-bias-adjusted P4 / P5 (v minus the participant's mean raw v over the Exp 1 AV pairs); the Lrev
            sensitivity subset (exp2/exp2_cell_rule.json lrev_clean cells).
  exclusion shared with Exp 1 (eligibility, catch, focus_hidden screen drop, Exp 1 retention rule -> out of everything);
            plus < 75 % retained in L_vs_E or in L_vs_Lrev (pooled over languages) -> out of Exp 2 only.
LISTENER GROUPS (three links /root/lilt_cn|_jp|_en serve identical stimuli): primary analyses pooled over groups; per-group
            secondary tables (P1-P3, MC, P4, P5); group differences in every participant DV exploratory (label-permutation
            test of the between-group sum of squares). Group = the LINK the participant entered by (infos `test_url`
            /root/lilt_<cn|en|jp> = the session path; the server rotates per link and the stopping rule counts per link);
            without an infos link: background native_lang, else CSV `culture`, else `ui_language`. native_lang != link is
            kept, counted in flow["link_x_native_lang"] and reported. Eligibility: native_lang in {cn, en, jp}.

RUN (WSL): /home/shelyn/miniconda/envs/GPTSoVits/bin/python analyse_lilt_v6.py [--manifest M] [--out DIR] [--sided two|one]
           ... analyse_lilt_v6.py --selftest          (simulated data only; reads no participant data)
"""
import argparse
import collections
import csv
import glob
import io
import json
import math
import os
import random
import re
import sys
import zlib
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPJ = HERE.parents[1]
DATA = SPJ / "webapp" / "data" / "lilt_link" / "whole_response"
LIVE_MANIFEST = SPJ / "pipeline" / "pilot" / "lilt_link" / "manifest.json"
OUTDIR = HERE / "_explore" / "voicedose" / "study_v6" / "analysis"
CCR = {"L3": -3, "L2": -2, "L1": -1, "S0": 0, "R1": 1, "R2": 2, "R3": 3}
CODES = {v: k for k, v in CCR.items()}
AV_CONTRASTS = ["H_vs_E@normal", "H_vs_E@strong", "H_vs_R@strong"]
MUTE_CONTRAST = "mute:H_vs_E@strong"
PAIR_CONTRASTS = AV_CONTRASTS + [MUTE_CONTRAST]
# ---- the OPTIONAL blocks of build_lilt_v6.py (--hr-normal / --voice-check, default OFF). Analysed only when rows of
# the block exist; never in the Holm family (P1-P3 stay as specified in ANALYSIS_PLAN_V6.md).
HR_NORMAL_CONTRAST = "H_vs_R@normal"            # P2b: per participant mean score_H > 0 (secondary)
VOICE_ARM = "voice_check"                        # voice_only block: strong minus normal emotion INTENSITY (CSV `intensity`)
ALL_PAIR_CONTRASTS = AV_CONTRASTS + [HR_NORMAL_CONTRAST, MUTE_CONTRAST]
FAMILY = ("P1", "P2", "P3")
STUDY_PREFIX = "v6_"
STUDY_DIR = HERE / "_explore" / "voicedose" / "study_v6"
PLAN_FREEZE = STUDY_DIR / "PLAN_FREEZE.sha256"
P2_SENS_FILE = STUDY_DIR / "stimuli" / "build" / "p2_sensitivity_cells.json"
FOCUS_PRIMARY, FOCUS_SENS = "focus_hidden", "focus_lost"      # plan section 5 step 4: primary / sensitivity screen drop
RETAIN_FRAC = 0.75                                            # participant rule: < 75 % retained in any primary AV contrast
MC_MIN_SCREENS = 4                                            # < 4 of 6 silent screens -> out of MC only
VC_MIN_PAIRS = 4                                              # < 4 voice pairs with both takes -> out of the voice check only
LISTENER_GROUPS = ("cn", "en", "jp")                          # the three links; group = the link (infos test_url)
LINK_RE = re.compile(r"/lilt_(cn|en|jp)/?$")
ELIGIBLE = {"native_lang": LISTENER_GROUPS, "hearing_not": ("impaired",), "vision_not": ("other",)}
# ---- EXPERIMENT 2 (ANALYSIS_PLAN_V6_EXP2.md): three-language AV pairs, LiltFace rule layer L vs E and vs Lrev
EXP2_BLOCK = "pair_ml"
EXP2_KINDS = ("L_vs_E", "L_vs_Lrev")                          # P4, P5
EXP2_FAMILY = ("P4", "P5")
STIM_LANGS = ("cn", "en", "jp")
EXP2_RULE_FILE = STUDY_DIR / "exp2" / "exp2_cell_rule.json"   # lrev_clean cells -> the P5 sensitivity subset
EXP2_CELLS = {"cn": ("cn_esd0002_short3_ang", "cn_esd0002_short3_hap", "cn_esd0002_short3_sad"),   # a-priori parity
              "en": ("en_ang_short_sentence", "en_hap_short_sentence", "en_sad_short_sentence"),   # set (plan EXP2
              "jp": ("jp_ang_short_sentence", "jp_hap_short_sentence", "jp_sad_short_sentence")}   # section 2)
SCORED_BLOCKS = ("single", "pair", "pair_mute", "voice_only", EXP2_BLOCK)
PERM_B = 5000                                                 # label permutations for the exploratory group tests
EMOTIONS = ["ang", "sad", "hap", "sur", "fear", "dis", "neu"]
LEVELS = {"n": "normal", "s": "strong"}
FM_RE = re.compile(r"((rav\d{2})_([a-z]+)(?:_r\d)?)_(n|s)(?:__|$)")      # rav<AA>_<emo>[_r<rep>]_<n|s>
SIM_COLUMNS = ["response_ts", "session_id", "participant_code", "culture", "ui_language", "mode", "block", "trial_index",
               "is_practice", "is_catch", "revision", "rt_ms", "n_replays", "stimulus_id", "stimulus_file",
               "stim_language", "voice_emotion", "face_emotion", "congruency", "text_content", "voice_id", "presentation",
               "cue_manip", "face_deg", "voice_arm", "chosen_side", "response_emotion", "intensity", "naturalness",
               "correct", "catch_correct", "renderer", "face_motion",
               "focus_lost", "focus_hidden", "focus_blurred", "focus_away_ms"]


# ---------------------------------------------------------------------------------------------------- small stats
def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    if len(xs) < 2:
        return None
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def r3(x):
    return None if x is None else round(float(x), 3)


def sig4(x):
    return None if x is None else float("%.4g" % x)


def seed_of(name):
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def boot_ci(vals, B, name):
    """Percentile 95 % CI of the mean by resampling the units in `vals` (participants or voice pairs)."""
    vals = [float(v) for v in vals if v is not None]
    n = len(vals)
    if n < 2 or B < 100:
        return None
    try:
        import numpy as np
        rng = np.random.default_rng(seed_of(name))
        a = np.asarray(vals)
        m = a[rng.integers(0, n, size=(B, n))].mean(axis=1)
        lo, hi = np.percentile(m, [2.5, 97.5])
        return [r3(lo), r3(hi)]
    except ImportError:
        rng = random.Random(seed_of(name))
        ms = sorted(sum(rng.choice(vals) for _ in range(n)) / n for _ in range(B))
        return [r3(ms[int(0.025 * (B - 1))]), r3(ms[int(0.975 * (B - 1))])]


def _betacf(a, b, x):
    """Continued fraction for the regularised incomplete beta (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-12:
            break
    return h


def _t_two_sided(t, df):
    x = df / (df + t * t)
    lbeta = math.lgamma(df / 2.0) + math.lgamma(0.5) - math.lgamma(df / 2.0 + 0.5)
    front = math.exp(math.log(x) * df / 2.0 + math.log(1 - x) * 0.5 - lbeta) if 0 < x < 1 else (1.0 if x >= 1 else 0.0)
    if x <= 0:
        return 0.0
    if x < (df / 2.0 + 1) / (df / 2.0 + 0.5 + 2):
        ib = front * _betacf(df / 2.0, 0.5, x) / (df / 2.0)
    else:
        ib = 1.0 - front * _betacf(0.5, df / 2.0, 1 - x) / 0.5
    return max(0.0, min(1.0, ib))


def t_p(vals):
    """One-sample two-sided t p against 0 (scipy when present, else the incomplete-beta formula)."""
    vals = [float(v) for v in vals if v is not None]
    if len(vals) < 2 or not sd(vals):
        return None
    try:
        from scipy import stats
        return float(stats.ttest_1samp(vals, 0.0).pvalue)
    except Exception:                                                   # noqa: BLE001
        t = mean(vals) / (sd(vals) / math.sqrt(len(vals)))
        return _t_two_sided(t, len(vals) - 1)


def one_sided(p2, m, direction=1):
    if p2 is None or m is None:
        return None
    return p2 / 2.0 if m * direction > 0 else 1.0 - p2 / 2.0


def wilcoxon_p(vals):
    vals = [float(v) for v in vals if v is not None and v != 0]
    if len(vals) < 6:
        return None
    try:
        from scipy import stats
        return float(stats.wilcoxon(vals).pvalue)
    except Exception:                                                   # noqa: BLE001
        return None


def binom_two_sided(k, n, p=0.5):
    if not n:
        return None
    pk = lambda j: comb(n, j) * p ** j * (1 - p) ** (n - j)            # noqa: E731
    obs = pk(k)
    return min(1.0, sum(pk(j) for j in range(n + 1) if pk(j) <= obs + 1e-12))


def holm(pmap):
    items = sorted((p, k) for k, p in pmap.items() if p is not None)
    m, run, out = len(items), 0.0, {}
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def test_block(vals, B, name, positive="H_better"):
    """Participant-level (or unit-level) inference on a list of per-unit scores."""
    vals = [float(v) for v in vals if v is not None]
    m, s = mean(vals), sd(vals)
    p2 = t_p(vals)
    wins, losses = sum(1 for v in vals if v > 0), sum(1 for v in vals if v < 0)
    return {"n": len(vals), "mean": r3(m), "sd": r3(s), "dz": r3(m / s) if (m is not None and s) else None,
            "boot95": boot_ci(vals, B, name), "p_t_two_sided": sig4(p2), "p_t_one_sided": sig4(one_sided(p2, m)),
            "p_wilcoxon": sig4(wilcoxon_p(vals)), "n_" + positive: wins, "n_other": losses,
            "p_sign": sig4(binom_two_sided(wins, wins + losses)) if wins + losses else None}


# ---------------------------------------------------------------------------------------------------- rows
def item_map(manifest):
    return {it["file"]: it for it in (manifest or {}).get("items", [])}


def keep_row(r, files):
    sf = r.get("stimulus_file") or ""
    if files is not None:
        return sf in files
    return sf.startswith(STUDY_PREFIX)


def load(data_dir, files=None):
    rows, excl, seen = [], collections.Counter(), set()
    for f in sorted(glob.glob(str(Path(data_dir) / "*.csv"))):
        b = os.path.basename(f)
        if b.startswith("_all") or b.lower().startswith("test"):
            excl["file_skipped"] += 1
            continue
        with io.open(f, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                pc = r.get("participant_code") or ""
                if pc.lower().startswith("test"):
                    excl["test"] += 1
                elif (r.get("mode") or "lilt") != "lilt":
                    excl["other_mode"] += 1
                elif r.get("is_practice", "0") in ("1", "1.0") or r.get("block") == "practice":
                    excl["practice"] += 1
                elif (r.get("revision") or "0") not in ("0", "0.0"):
                    excl["revision"] += 1
                elif not keep_row(r, files):
                    excl["not_this_study"] += 1
                else:
                    key = (pc, r.get("stimulus_id") or r.get("stimulus_file"))
                    if key in seen:
                        excl["duplicate"] += 1
                    else:
                        seen.add(key)
                        rows.append(r)
    return rows, dict(excl)


def load_background(data_dir):
    """participant_code -> background dict, from the *_infos.json the server files beside each completed CSV
    (server.py: `background` is copied into the participant's infos JSON). Keyed by the infos' username AND by the CSV
    stem, so either join works. Missing file -> the participant is simply absent here (kept, counted as unknown)."""
    out = {}
    for f in sorted(glob.glob(str(Path(data_dir) / "*_infos.json"))):
        try:
            d = json.load(io.open(f, encoding="utf-8"))
        except Exception:                                              # noqa: BLE001 -- a half-written file is not fatal
            continue
        bg = d.get("background") if isinstance(d.get("background"), dict) else {}
        rec = {"native_lang": bg.get("native_lang", ""), "hearing": bg.get("hearing", ""), "vision": bg.get("vision", ""),
               "second_lang": bg.get("second_lang", ""), "second_level": bg.get("second_level", ""),
               "age": d.get("age", ""), "gender": d.get("gender", ""), "started_utc": d.get("started_utc", ""),
               "link": (LINK_RE.search(str(d.get("test_url") or "")).group(1)
                        if LINK_RE.search(str(d.get("test_url") or "")) else "")}
        if d.get("username"):
            out[str(d["username"])] = rec
        out[os.path.basename(f)[:-len("_infos.json")]] = rec
    return out


def eligibility(pc, bg):
    """(eligible, reason) per plan section 5 step 2. No background record -> eligible, reason 'background_missing';
    'na' (prefer not to say) is kept."""
    if bg is None:
        return True, "background_missing"
    if bg.get("native_lang") and bg["native_lang"] not in ELIGIBLE["native_lang"]:    # 'other' is not a group
        return False, "native_lang=%s" % bg["native_lang"]
    if bg.get("hearing") in ELIGIBLE["hearing_not"]:
        return False, "hearing=%s" % bg["hearing"]
    if bg.get("vision") in ELIGIBLE["vision_not"]:
        return False, "vision=%s" % bg["vision"]
    return True, "" if bg.get("native_lang") else "native_lang_blank"


def focus_count(r, field):
    v = r.get(field)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def screen_kept(r, field):
    """A scored screen is dropped when the focus counter of `field` is >= 1; the catch is never dropped; blank = kept."""
    if r.get("block") not in SCORED_BLOCKS:
        return True
    c = focus_count(r, field)
    return not (c is not None and c >= 1)


def contrast_of_row(r, imap):
    it = imap.get(r.get("stimulus_file") or "") or {}
    va = it.get("voice_arm") or r.get("voice_arm") or ""
    if r.get("block") == "voice_only" or r.get("presentation") == "voice_only":
        return VOICE_ARM
    return va if ("_vs_" in va) else ""


def design_counts(imap):
    """Screens per participant per contrast from the manifest (distinct pick groups); {} without a manifest."""
    d = collections.defaultdict(set)
    for it in imap.values():
        c = it.get("voice_arm") or ""
        if it.get("block") == "voice_only":
            c = VOICE_ARM
        if c and it.get("pick_group"):
            d[c].add(it["pick_group"])
    return {c: len(g) for c, g in d.items()}


def p2_sensitivity_cells(path=P2_SENS_FILE):
    try:
        d = json.load(io.open(path, encoding="utf-8"))
        return set(d.get("clean_strong") or []), d.get("rule", "")
    except Exception:                                                  # noqa: BLE001
        return set(), "file missing: %s" % path


def _resolve_freeze_path(p):
    """sha256sum lines are written in Git Bash (/d/SPJ/...); the analysis may run in WSL (/mnt/d/...) or on Windows."""
    cands = [p]
    m = re.match(r"^/([a-zA-Z])/(.*)$", p)
    if m:
        cands += ["/mnt/%s/%s" % (m.group(1).lower(), m.group(2)), "%s:/%s" % (m.group(1).upper(), m.group(2))]
    m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", p)
    if m:
        cands += ["/%s/%s" % (m.group(1).lower(), m.group(2)), "%s:/%s" % (m.group(1).upper(), m.group(2))]
    m = re.match(r"^([a-zA-Z]):[/\\](.*)$", p)
    if m:
        cands += ["/mnt/%s/%s" % (m.group(1).lower(), m.group(2)), "/%s/%s" % (m.group(1).lower(), m.group(2))]
    for c in cands:
        if Path(c).is_file():
            return Path(c)
    return None


def freeze_check(path=PLAN_FREEZE):
    """Re-check PLAN_FREEZE.sha256 (plan section 11 item 7) the way `sha256sum -c` would. Reported, never enforced."""
    import hashlib
    path = Path(path)
    out = {"file": str(path), "exists": path.is_file(), "lines": [], "all_ok": None}
    if not path.is_file():
        return out
    ok_all = True
    for line in io.open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        m = re.match(r"^\\?([0-9a-fA-F]{64})\s+\*?(.+)$", line)
        if not m:
            out["lines"].append({"line": line, "status": "unparsed"})
            ok_all = False
            continue
        want, name = m.group(1).lower(), m.group(2)
        fp = _resolve_freeze_path(name)
        if fp is None:
            out["lines"].append({"file": name, "status": "MISSING"})
            ok_all = False
            continue
        got = hashlib.sha256(fp.read_bytes()).hexdigest()
        out["lines"].append({"file": name, "status": "OK" if got == want else "FAILED", "sha256": got})
        ok_all = ok_all and got == want
    out["all_ok"] = ok_all
    side = path.with_suffix(".frozen_at")
    if side.is_file():
        out["frozen_at"] = io.open(side, encoding="utf-8").read().strip()
    return out


def ident(r, imap):
    """voice pair, level, actor, emotion code of a row: manifest first, then face_motion (source stem), then CSV."""
    it = imap.get(r.get("stimulus_file") or "") or {}
    if it.get("base_cell"):
        return {"vp": it["base_cell"], "level": it.get("voice_level", ""), "actor": it.get("actor", ""),
                "emo": it.get("voice_emotion") or r.get("voice_emotion", ""), "arm": it.get("renderer", "")}
    m = FM_RE.search(r.get("face_motion") or "")
    if m:
        return {"vp": m.group(1), "level": LEVELS[m.group(4)], "actor": m.group(2),
                "emo": r.get("voice_emotion") or m.group(3), "arm": r.get("renderer", "")}
    if r.get("block") == EXP2_BLOCK or (r.get("voice_arm") or "").startswith("L_vs_"):   # Exp 2: "<cell>__<tok>...|..."
        stem0 = (r.get("face_motion") or "").split("|")[0]
        cell = stem0.split("__")[0] if "__" in stem0 else ""
        return {"vp": cell or r.get("stimulus_file", ""), "level": r.get("stim_language", ""),
                "actor": r.get("voice_id", ""), "emo": r.get("voice_emotion", ""), "arm": r.get("renderer", "")}
    cm = r.get("cue_manip") or ""                                  # voice_only rows: "voice:normal" | "voice:strong"
    lvl = cm.split(":", 1)[1] if cm.startswith("voice:") else ""
    return {"vp": "", "level": lvl, "actor": r.get("voice_id", ""), "emo": r.get("voice_emotion", ""),
            "arm": r.get("renderer", "")}


def parse_voice(r, imap):
    """(--voice-check) one audio-only row -> participant, voice pair, level, intensity 1-7, accuracy."""
    if r.get("block") != "voice_only" and r.get("presentation") != "voice_only":
        return None
    if r.get("block") in ("practice",):
        return None
    d = ident(r, imap)
    lvl = d["level"]
    if lvl not in ("normal", "strong"):
        return None
    resp = r.get("response_emotion") or ""
    acc = (1.0 if resp == d["emo"] else 0.0) if resp else (float(r["correct"]) if r.get("correct") in ("0", "1") else None)
    vp = d["vp"] or ("%s_%s" % (d["actor"], d["emo"]))
    return {"p": r.get("participant_code"), "vp": vp, "actor": d["actor"], "emo": d["emo"], "level": lvl,
            "intensity": num(r.get("intensity")), "acc": acc, "resp": resp, "rt_ms": r.get("rt_ms")}


def parse_pair(r, imap):
    """One CCR row -> score_H. Any 'A_vs_B@level' voice_arm parses, so the optional H_vs_R@normal block needs nothing
    more than its rows existing."""
    if r.get("presentation") != "AV_pair" or r.get("block") not in ("pair", "pair_mute", EXP2_BLOCK):
        return None
    v = CCR.get(r.get("response_emotion") or "")
    it = imap.get(r.get("stimulus_file") or "") or {}
    va, cm = it.get("voice_arm") or r.get("voice_arm") or "", it.get("cue_manip") or r.get("cue_manip") or ""
    if v is None or "|" not in cm:
        return None
    mute = va.startswith("mute:")
    core = va.split(":", 1)[1] if mute else va
    if "_vs_" not in core or "@" not in core:
        return None
    pair, level = core.split("@", 1)
    a_name, b_name = pair.split("_vs_", 1)
    left, right = cm.split("|")[0].split(":", 1)[-1], cm.split("|")[1].split(":", 1)[-1]
    if right == a_name:
        s, side = v, "right"
    elif left == a_name:
        s, side = -v, "left"
    else:
        return None
    d = ident(r, imap)
    exp2 = r.get("block") == EXP2_BLOCK or a_name == "L"
    lang = (it.get("stim_language") or r.get("stim_language") or "") if not exp2 else level
    return {"p": r.get("participant_code"), "contrast": ("mute:" if mute else "") + core, "A": a_name, "B": b_name,
            "level": level, "v": v, "s": s, "H_side": side, "vp": d["vp"], "actor": d["actor"], "emo": d["emo"],
            "rt_ms": r.get("rt_ms"), "n_replays": r.get("n_replays"), "exp": 2 if exp2 else 1,
            "kind": pair if exp2 else "", "stim_lang": lang or "en"}


def catches(rows):
    per = {}
    for r in rows:
        if r.get("block") != "pair_catch":
            continue
        va = r.get("voice_arm") or ""
        want = va.split(":", 1)[1] if va.startswith("correct:") else None
        resp = r.get("response_emotion") or ""
        got = r.get("chosen_side") or ("left" if resp == "left" or resp[:1] == "L" else
                                       "right" if resp == "right" or resp[:1] == "R" else "same")
        ok = (r.get("correct") in ("1", "1.0")) if r.get("correct") not in (None, "") else (want is not None and got == want)
        d = per.setdefault(r.get("participant_code"), {"n": 0, "passed": 0})
        d["n"] += 1
        d["passed"] += int(bool(ok))
    return per


# ---------------------------------------------------------------------------------------------------- summaries
def unit_means(xs, unit, key="s"):
    d = collections.defaultdict(list)
    for x in xs:
        d[x[unit]].append(x[key])
    return {u: mean(v) for u, v in d.items()}


def summary(xs, B, name):
    if not xs:
        return {"n_trials": 0, "n_participants": 0}
    ss = [x["s"] for x in xs]
    pm = unit_means(xs, "p")
    cm = unit_means(xs, "vp")
    out = {"n_trials": len(xs), "mean_trials": r3(mean(ss)),
           "pref_H": r3(sum(1 for s in ss if s > 0) / len(ss)), "pref_other": r3(sum(1 for s in ss if s < 0) / len(ss)),
           "tie": r3(sum(1 for s in ss if s == 0) / len(ss)), "n_participants": len(pm)}
    out["participants"] = test_block(list(pm.values()), B, name + "|p")
    out["voice_pairs"] = {"n": len(cm), "mean": r3(mean(list(cm.values()))), "boot95": boot_ci(list(cm.values()), B, name + "|c")}
    return out


def p3_values(px):
    """Per participant: mean over the voice pairs answered at both levels of s(strong) - s(normal)."""
    by = {}
    for x in px:
        if x["contrast"] in ("H_vs_E@normal", "H_vs_E@strong"):
            by.setdefault(x["p"], {}).setdefault(x["contrast"], {}).setdefault(x["vp"], []).append(x["s"])
    out = {}
    for p, d in by.items():
        vps = sorted(set(d.get("H_vs_E@normal", {})) & set(d.get("H_vs_E@strong", {})))
        if vps:
            out[p] = mean([mean(d["H_vs_E@strong"][v]) - mean(d["H_vs_E@normal"][v]) for v in vps])
    return out


def p1_values(px):
    n = unit_means([x for x in px if x["contrast"] == "H_vs_E@normal"], "p")
    s = unit_means([x for x in px if x["contrast"] == "H_vs_E@strong"], "p")
    return {p: mean([d[p] for d in (n, s) if p in d]) for p in set(n) | set(s)}


def hypotheses(px, B, name, sided="two"):
    """P1, P2, P3 at the participant level (+ voice-pair level) and the Holm family."""
    H = {}
    cm_n = unit_means([x for x in px if x["contrast"] == "H_vs_E@normal"], "vp")
    cm_s = unit_means([x for x in px if x["contrast"] == "H_vs_E@strong"], "vp")
    cm_r = unit_means([x for x in px if x["contrast"] == "H_vs_R@strong"], "vp")
    cell = {"P1": [mean([d[v] for d in (cm_n, cm_s) if v in d]) for v in set(cm_n) | set(cm_s)],
            "P2": list(cm_r.values()),
            "P3": [cm_s[v] - cm_n[v] for v in cm_s if v in cm_n]}
    part = {"P1": p1_values(px), "P2": unit_means([x for x in px if x["contrast"] == "H_vs_R@strong"], "p"),
            "P3": p3_values(px)}
    defs = {"P1": "H_vs_E pooled over voice levels: per participant the mean of its normal and strong means; > 0",
            "P2": "H_vs_R@strong: per participant mean score_H; > 0",
            "P3": "within participant: mean over voice pairs of score_H(H_vs_E@strong) - score_H(H_vs_E@normal); > 0"}
    for k in FAMILY:
        H[k] = {"definition": defs[k], "participants": test_block(list(part[k].values()), B, "%s|%s|p" % (name, k)),
                "voice_pairs": {"n": len(cell[k]), "mean": r3(mean(cell[k])), "boot95": boot_ci(cell[k], B, "%s|%s|c" % (name, k))}}
    fam2 = {k: H[k]["participants"]["p_t_two_sided"] for k in FAMILY if H[k]["participants"]["n"] >= 2}
    fam1 = {k: H[k]["participants"]["p_t_one_sided"] for k in FAMILY if H[k]["participants"]["n"] >= 2}
    return H, {"sided_primary": sided, "holm_two_sided": {k: sig4(v) for k, v in holm(fam2).items()},
               "holm_one_sided": {k: sig4(v) for k, v in holm(fam1).items()},
               "holm_primary": {k: sig4(v) for k, v in holm(fam2 if sided == "two" else fam1).items()}}


def contrasts_present(px):
    """The pair contrasts in the data, in the canonical order (the optional H_vs_R@normal only when it has rows)."""
    have = {x["contrast"] for x in px}
    return [c for c in ALL_PAIR_CONTRASTS if c in have or c in PAIR_CONTRASTS]


def optional_hypotheses(px, B, name):
    """(--hr-normal) P2b = per participant mean score_H in H_vs_R@normal > 0, and P3R = (H-R)@strong - (H-R)@normal
    within participant over the voice pairs answered at both levels. Both secondary; not in the Holm family. Empty
    when the block has no rows."""
    hr_n = [x for x in px if x["contrast"] == HR_NORMAL_CONTRAST]
    if not hr_n:
        return {}
    pm = unit_means(hr_n, "p")
    cm = unit_means(hr_n, "vp")
    by = {}
    for x in px:
        if x["contrast"] in (HR_NORMAL_CONTRAST, "H_vs_R@strong"):
            by.setdefault(x["p"], {}).setdefault(x["contrast"], {}).setdefault(x["vp"], []).append(x["s"])
    p3r = {}
    for p, d in by.items():
        vps = sorted(set(d.get(HR_NORMAL_CONTRAST, {})) & set(d.get("H_vs_R@strong", {})))
        if vps:
            p3r[p] = mean([mean(d["H_vs_R@strong"][v]) - mean(d[HR_NORMAL_CONTRAST][v]) for v in vps])
    return {"P2b": {"definition": "H_vs_R@normal: per participant mean score_H; > 0 (secondary, outside the Holm family)",
                    "participants": test_block(list(pm.values()), B, "%s|P2b|p" % name),
                    "voice_pairs": {"n": len(cm), "mean": r3(mean(list(cm.values()))),
                                    "boot95": boot_ci(list(cm.values()), B, "%s|P2b|c" % name)}},
            "P3R": {"definition": "within participant: mean over voice pairs of score_H(H_vs_R@strong) - "
                                  "score_H(H_vs_R@normal); the motion-equated companion of P3 (secondary)",
                    "participants": test_block(list(p3r.values()), B, "%s|P3R|p" % name)}}


def voice_check_summary(vx, B, name):
    """(--voice-check) Within participant: mean over the voice pairs heard at both levels of intensity(strong) -
    intensity(normal) > 0 (the P3 premise: Mandarin listeners hear the strong take as stronger); plus emotion
    accuracy (7-AFC) per level and per emotion, and the per-voice-pair difference."""
    if not vx:
        return {}
    by = {}
    for x in vx:
        if x["intensity"] is not None:
            by.setdefault(x["p"], {}).setdefault(x["vp"], {})[x["level"]] = x["intensity"]
    diffs, per_vp = {}, collections.defaultdict(list)
    for p, d in by.items():
        ds = [(vp, lv["strong"] - lv["normal"]) for vp, lv in d.items() if "strong" in lv and "normal" in lv]
        if ds:
            diffs[p] = mean([v for _, v in ds])
            for vp, v in ds:
                per_vp[vp].append(v)
    emo_of_vp = {x["vp"]: x["emo"] for x in vx}
    lvl = {}
    for L in ("normal", "strong"):
        lx = [x for x in vx if x["level"] == L]
        pi = unit_means([x for x in lx if x["intensity"] is not None], "p", "intensity")
        pa = unit_means([x for x in lx if x["acc"] is not None], "p", "acc")
        lvl[L] = {"n_trials": len(lx), "intensity_mean_trials": r3(mean([x["intensity"] for x in lx])),
                  "intensity_participants": {"n": len(pi), "mean": r3(mean(list(pi.values()))),
                                             "boot95": boot_ci(list(pi.values()), B, "%s|voice|%s|int" % (name, L))},
                  "accuracy_trials": r3(mean([x["acc"] for x in lx])),
                  "accuracy_participants": {"n": len(pa), "mean": r3(mean(list(pa.values()))),
                                            "boot95": boot_ci(list(pa.values()), B, "%s|voice|%s|acc" % (name, L))}}
    acc_emo = {}
    for e in sorted({x["emo"] for x in vx}):
        acc_emo[e] = {L: r3(mean([x["acc"] for x in vx if x["emo"] == e and x["level"] == L])) for L in ("normal", "strong")}
        acc_emo[e]["intensity_strong_minus_normal"] = r3(mean(per_vp_v for vp, vs in per_vp.items()
                                                              for per_vp_v in vs if emo_of_vp.get(vp) == e)) if per_vp else None
    conf = collections.Counter((x["emo"], x["resp"]) for x in vx if x["resp"])
    return {"definition": "within participant: mean over voice pairs of intensity(strong) - intensity(normal); > 0. "
                          "Not in the Holm family (manipulation check of the P3 premise)",
            "n_trials": len(vx), "n_participants": len({x["p"] for x in vx}),
            "strong_minus_normal": test_block(list(diffs.values()), B, "%s|voice|diff" % name, positive="strong_higher"),
            "per_voice_pair": {vp: {"n": len(v), "mean_diff": r3(mean(v))} for vp, v in sorted(per_vp.items())},
            "by_level": lvl, "accuracy_by_emotion": acc_emo,
            "confusions": {"%s->%s" % k: v for k, v in sorted(conf.items())}}


# ---------------------------------------------------------------------------------------------------- Exp 2 + groups
def listener_group(bg, prow):
    """cn / en / jp = the LINK (infos test_url, recorded exactly by the server); without it the background native_lang,
    then the CSV `culture` (client: native_lang), then `ui_language`; '?' when none is a group."""
    if bg and bg.get("link") in LISTENER_GROUPS:
        return bg["link"]
    if bg and bg.get("native_lang"):
        return bg["native_lang"] if bg["native_lang"] in LISTENER_GROUPS else "?"
    for f in ("culture", "ui_language"):
        if (prow or {}).get(f) in LISTENER_GROUPS:
            return prow[f]
    return "?"


def exp2_clean_cells(path=EXP2_RULE_FILE):
    """The P5 sensitivity subset: parity cells whose Lrev is a clean motion-equated control (Lrev/L amplitude in
    [0.8, 1.25] and r(L-E, Lrev-E) < 0.3 on both brow channels), from exp2/exp2_cell_rule.json."""
    try:
        rows = json.load(io.open(path, encoding="utf-8"))
        return {r["cell"] for r in rows if r.get("lrev_clean") and r.get("parity_emotion")}, "read %s" % path
    except Exception:                                                  # noqa: BLE001
        return set(), "file missing: %s" % path


def exp2_dv(px2, kind, cells=None):
    """Per participant: mean over stimulus languages of the per-language mean score_L (equal weight per language)."""
    by = {}
    for x in px2:
        if x["kind"] == kind and (cells is None or x["vp"] in cells):
            by.setdefault(x["p"], {}).setdefault(x["stim_lang"], []).append(x["s"])
    return {p: mean([mean(v) for v in d.values()]) for p, d in by.items()}


def perm_groups(vals_by_p, group_of, B, name):
    """EXPLORATORY listener-group difference in one participant DV: between-group sum of squares, label-permutation p
    (+ Kruskal-Wallis when scipy is present). Groups with fewer than 2 participants are dropped."""
    data = [(group_of.get(p), v) for p, v in vals_by_p.items() if v is not None and group_of.get(p) in LISTENER_GROUPS]
    cnt = collections.Counter(g for g, _ in data)
    data = [(g, v) for g, v in data if cnt[g] >= 2]
    labs, ys = [g for g, _ in data], [v for _, v in data]
    gs = sorted(set(labs))
    out = {"n_per_group": {g: cnt[g] for g in sorted(cnt)},
           "means": {g: r3(mean([y for l, y in zip(labs, ys) if l == g])) for g in gs}, "p_perm": None, "B": B}
    if len(gs) < 2:
        return out
    gm = mean(ys)

    def ssb(lab):
        s = 0.0
        for g in gs:
            v = [y for l, y in zip(lab, ys) if l == g]
            s += len(v) * (mean(v) - gm) ** 2
        return s
    obs, rng, lab, k = ssb(labs), random.Random(seed_of(name)), list(labs), 0
    for _ in range(B):
        rng.shuffle(lab)
        k += int(ssb(lab) >= obs - 1e-12)
    out.update({"ssb": r3(obs), "p_perm": sig4((k + 1.0) / (B + 1.0))})
    try:
        from scipy import stats
        out["p_kruskal"] = sig4(float(stats.kruskal(*[[y for l, y in zip(labs, ys) if l == g] for g in gs]).pvalue))
    except Exception:                                                  # noqa: BLE001
        pass
    return out


def exp2_summary(px2, B, name, group_of, sided="two", bias=None, clean_cells=None, descriptives=True):
    """Experiment 2: P4 (L_vs_E), P5 (L_vs_Lrev), Holm over P4-P5; per stimulus language; exploratory native-stimulus
    effect and listener group x stimulus language table; side-bias-adjusted P4/P5; P5 on the clean-Lrev cells."""
    if not px2:
        return {}
    defs = {"P4": "L_vs_E: per participant the mean over stimulus languages of the per-language mean score_L; > 0",
            "P5": "L_vs_Lrev: per participant the mean over stimulus languages of the per-language mean score_L; > 0"}
    out = {"n_trials": len(px2), "n_participants": len({x["p"] for x in px2}), "hypotheses": {}}
    for k, kind in zip(EXP2_FAMILY, EXP2_KINDS):
        pm = exp2_dv(px2, kind)
        cm = unit_means([x for x in px2 if x["kind"] == kind], "vp")
        out["hypotheses"][k] = {"definition": defs[k],
                                "participants": test_block(list(pm.values()), B, "%s|%s|p" % (name, k), positive="L_better"),
                                "cells": {"n": len(cm), "mean": r3(mean(list(cm.values()))),
                                          "boot95": boot_ci(list(cm.values()), B, "%s|%s|c" % (name, k))}}
    fam2 = {k: out["hypotheses"][k]["participants"]["p_t_two_sided"] for k in EXP2_FAMILY
            if out["hypotheses"][k]["participants"]["n"] >= 2}
    fam1 = {k: out["hypotheses"][k]["participants"]["p_t_one_sided"] for k in EXP2_FAMILY
            if out["hypotheses"][k]["participants"]["n"] >= 2}
    out["family"] = {"sided_primary": sided, "holm_two_sided": {k: sig4(v) for k, v in holm(fam2).items()},
                     "holm_one_sided": {k: sig4(v) for k, v in holm(fam1).items()},
                     "holm_primary": {k: sig4(v) for k, v in holm(fam2 if sided == "two" else fam1).items()}}
    if bias:
        adj = [dict(x, s=(x["s"] - bias[x["p"]]) if x["H_side"] == "right" else (x["s"] + bias[x["p"]]))
               for x in px2 if bias.get(x["p"]) is not None]
        out["side_bias_adjusted"] = {k: test_block(list(exp2_dv(adj, kind).values()), B, "%s|%s|adj" % (name, k),
                                                   positive="L_better") for k, kind in zip(EXP2_FAMILY, EXP2_KINDS)}
    if clean_cells:
        sx = [x for x in px2 if x["kind"] == "L_vs_Lrev" and x["vp"] in clean_cells]
        out["P5_sensitivity"] = {"cells": sorted(clean_cells), "n_cells_seen": len({x["vp"] for x in sx}),
                                 "participants": test_block(list(exp2_dv(sx, "L_vs_Lrev").values()), B,
                                                            "%s|P5sens" % name, positive="L_better")}
    if not descriptives:
        return out
    out["per_stim_language"] = {lang: {kind: summary([x for x in px2 if x["kind"] == kind and x["stim_lang"] == lang], B,
                                                     "%s|%s|%s" % (name, kind, lang)) for kind in EXP2_KINDS}
                                for lang in sorted({x["stim_lang"] for x in px2})}
    out["per_emotion"] = {}
    for e in sorted({x["emo"] for x in px2}):
        out["per_emotion"][e] = {}
        for kind in EXP2_KINDS:
            pm = list(unit_means([x for x in px2 if x["kind"] == kind and x["emo"] == e], "p").values())
            out["per_emotion"][e][kind] = {"n_participants": len(pm), "mean": r3(mean(pm)),
                                           "boot95": boot_ci(pm, B, "%s|%s|%s|emo" % (name, kind, e))}
    nat = {}
    for kind in EXP2_KINDS:
        by = {}
        for x in px2:
            g = group_of.get(x["p"])
            if x["kind"] == kind and g in LISTENER_GROUPS:
                by.setdefault(x["p"], {"native": [], "non": []})["native" if x["stim_lang"] == g else "non"].append(x["s"])
        diffs = [mean(d["native"]) - mean(d["non"]) for d in by.values() if d["native"] and d["non"]]
        table = {}
        for g in LISTENER_GROUPS:
            row = {}
            for lang in STIM_LANGS:
                pm = list(unit_means([x for x in px2 if x["kind"] == kind and x["stim_lang"] == lang
                                      and group_of.get(x["p"]) == g], "p").values())
                if pm:
                    row[lang] = {"n": len(pm), "mean": r3(mean(pm)), "boot95": boot_ci(pm, B, "%s|%s|%s|%s" % (name, kind, g, lang))}
            if row:
                table[g] = row
        nat[kind] = {"native_minus_nonnative": test_block(diffs, B, "%s|%s|native" % (name, kind), positive="native_higher"),
                     "group_x_stim_language": table}
    out["native_stimulus_exploratory"] = nat
    return out


def describe(px, B, name, key):
    """Descriptives per emotion / actor / voice pair: every contrast (trials + participant means) and P3."""
    out = {}
    for u in sorted({x[key] for x in px}):
        ux = [x for x in px if x[key] == u]
        d = {}
        for c in contrasts_present(px):
            cx = [x for x in ux if x["contrast"] == c]
            if not cx:
                continue
            pm = list(unit_means(cx, "p").values())
            d[c] = {"n_trials": len(cx), "mean_trials": r3(mean([x["s"] for x in cx])), "n_participants": len(pm),
                    "mean_participants": r3(mean(pm)),
                    "boot95_participants": boot_ci(pm, B, "%s|%s|%s|%s" % (name, key, u, c)) if key != "vp" else None}
        p3 = list(p3_values(ux).values())
        d["P3"] = {"n_participants": len(p3), "mean": r3(mean(p3)),
                   "boot95_participants": boot_ci(p3, B, "%s|%s|%s|P3" % (name, key, u)) if key != "vp" else None}
        out[u] = d
    return out


def singles_summary(rows, imap, B, name):
    xs = [r for r in rows if r.get("block") == "single"]

    def arm(r):
        return ident(r, imap)["arm"] or r.get("renderer") or "?"

    def acc(r):
        want = ident(r, imap)["emo"] or r.get("voice_emotion")
        if r.get("response_emotion"):
            return 1.0 if r.get("response_emotion") == want else 0.0
        if r.get("correct") in ("0", "1"):
            return float(r["correct"])
        return None

    measures = {"vividness": lambda r: num(r.get("intensity")), "naturalness": lambda r: num(r.get("naturalness")),
                "accuracy": acc}
    arms = sorted({arm(r) for r in xs})
    out = {"arms": {}, "differences": {}, "accuracy_by_emotion": {}}
    pm = {}
    for mname, fn in measures.items():
        pm[mname] = {}
        for a in arms:
            d = collections.defaultdict(list)
            for r in xs:
                if arm(r) == a and fn(r) is not None:
                    d[r.get("participant_code")].append(fn(r))
            pm[mname][a] = {p: mean(v) for p, v in d.items()}
    for a in arms:
        ax = [r for r in xs if arm(r) == a]
        out["arms"][a] = {"n_trials": len(ax)}
        for mname, fn in measures.items():
            vals = list(pm[mname][a].values())
            tv = [fn(r) for r in ax if fn(r) is not None]
            out["arms"][a][mname] = {"mean_trials": r3(mean(tv)), "n_participants": len(vals),
                                     "mean_participants": r3(mean(vals)),
                                     "boot95_participants": boot_ci(vals, B, "%s|%s|%s" % (name, a, mname))}
    for a_arm, b_arm in (("H", "E"), ("R", "E"), ("H", "R")):
        if a_arm not in arms or b_arm not in arms:
            continue
        key = "%s-%s" % (a_arm, b_arm)
        out["differences"][key] = {}
        for mname in measures:
            A, Bm = pm[mname][a_arm], pm[mname][b_arm]
            diffs = [A[p] - Bm[p] for p in A if p in Bm]
            out["differences"][key][mname] = {"n_participants": len(diffs), "mean": r3(mean(diffs)),
                                              "boot95": boot_ci(diffs, B, "%s|%s|%s" % (name, key, mname)),
                                              "p_t_two_sided": sig4(t_p(diffs))}
    for e in sorted({ident(r, imap)["emo"] for r in xs}):
        ex = [r for r in xs if ident(r, imap)["emo"] == e]
        out["accuracy_by_emotion"][e] = {a: {"n_trials": sum(1 for r in ex if arm(r) == a),
                                             "accuracy": r3(mean([acc(r) for r in ex if arm(r) == a and acc(r) is not None]))}
                                         for a in arms}
    return out


BALANCE_MULT_EXTRA = 2        # least-filled allowance per link: one list may hold ceil(n_link / P) + 2 participants
BALANCE_SPLIT_ALLOW = 4       # least-filled allowance per link: per-clip single-arm split (max - min) up to 4
BALANCE_BLOCKS = ("single", "pair", "pair_mute", "pair_catch", EXP2_BLOCK)
DEFAULT_PERIOD = 6


def _lcm(a, b):
    return a * b // math.gcd(a, b) if a and b else max(a, b)


def manifest_period(imap):
    """P = lcm of the arm counts over the served pick groups (what the server derives from the manifest it loads);
    None without a manifest."""
    if not imap:
        return None
    sizes = collections.Counter(it.get("pick_group") or ("media/" + it.get("file", f)) for f, it in imap.items())
    P = 1
    for v in sizes.values():
        P = _lcm(P, v)
    return P


def load_rotation_stamps(sessions_dir):
    """DESCRIPTIVE AUDIT ONLY. The LILTROT v6 stamp `lilt_rotation` of every mode "lilt" record in
    <webapp data>/sessions/*.json (least-filled allocation; fields link, build, built, period, offset). The username is
    read only to flag test* sessions and is never returned. Stamps enter no estimate, test or exclusion."""
    out = {"sessions_dir": str(sessions_dir) if sessions_dir else None, "found": False, "stamps": [], "by_session": {},
           "unstamped": {}, "unreadable": 0}
    if not sessions_dir or not Path(sessions_dir).is_dir():
        return out
    out["found"] = True
    unst = collections.Counter()
    for f in sorted(glob.glob(str(Path(sessions_dir) / "*.json"))):
        try:
            d = json.load(io.open(f, encoding="utf-8"))
        except Exception:                                              # noqa: BLE001 -- half-written / broken file
            out["unreadable"] += 1
            continue
        if not isinstance(d, dict) or d.get("mode") != "lilt":
            continue
        is_test = str(d.get("username") or "").strip().lower().startswith("test")
        done = bool(d.get("completed"))
        r = d.get("lilt_rotation") if isinstance(d.get("lilt_rotation"), dict) else {}
        off, per = r.get("offset"), r.get("period")
        if (not isinstance(off, int) or isinstance(off, bool) or not isinstance(per, int) or isinstance(per, bool)
                or per < 1 or not 0 <= off < per):
            unst["test" if is_test else ("completed" if done else "not_completed")] += 1
            continue
        s = {"session_id": str(d.get("session_id") or ""), "link": str(r.get("link") or ""), "build": str(r.get("build") or ""),
             "built": str(r.get("built") or ""), "period": per, "offset": off, "completed": done, "test": is_test}
        out["stamps"].append(s)
        if s["session_id"]:
            out["by_session"][s["session_id"]] = s
    out["unstamped"] = dict(unst)
    return out


def _offset_file_sets(imap, P):
    """Reconstruction without a stamp: the scored files that list k (serve_v6(items, k)) serves, k = 0..P-1."""
    items = list(imap.values())
    return [{x["file"] for x in serve_v6(items, k) if x.get("block") in BALANCE_BLOCKS} for k in range(P)]


def _list_counts_entry(link, build, P, counts, label):
    """One (link, build, P) cell of the audit; warns only beyond the least-filled allowance."""
    n = sum(counts)
    allow = int(math.ceil(n / float(P))) + BALANCE_MULT_EXTRA if n else 0
    e = {"link": link, "build": build, "period": P, "counts_per_offset": list(counts), "n": n,
         "spread": (max(counts) - min(counts)) if counts else 0, "max_per_offset": max(counts) if counts else 0,
         "allowed_max_per_offset": allow, "warning": ""}
    if n and e["max_per_offset"] > allow:
        collapsed = n >= 4 and e["max_per_offset"] > n / 2.0
        e["warning"] = ("%s%s: link %s build %s: %d of %d on one list, counts per offset %s (least-filled allows "
                        "ceil(n/P)+2 = %d, P = %d)" % ("COLLAPSED ROTATION " if collapsed else "rotation beyond the "
                                                       "least-filled allowance ", label, link, build[:12],
                                                       e["max_per_offset"], n, counts, allow, P))
    return e


def rotation_audit(rows_all, imap, group_of, rotation=None):
    """DESCRIPTIVE balance audit of the list assignment (LILTROT v6 least-filled allocation); enters no test, estimate,
    exclusion or stopping rule. Offsets come from the session stamps (rotation = load_rotation_stamps(...), joined to
    the CSV participants by session_id; usernames are never read here), else they are reconstructed from the served
    stimulus_file set (the list k whose serve_v6 set contains it; needs the manifest). Reports counts per (link, build,
    offset) for (a) the completed non-test stamped sessions on the server and (b) the CSV participants analysed, and warns
    only when a list holds more than ceil(n/P)+2 of a (link, build) cell (COLLAPSED when more than half of n >= 4), or
    when a stamp disagrees with the list actually served."""
    rotation = rotation or {}
    stamps = [s for s in rotation.get("stamps") or [] if not s["test"]]
    by_sid = rotation.get("by_session") or {}
    pc = collections.Counter(s["period"] for s in stamps)
    P_man = manifest_period(imap)
    P = pc.most_common(1)[0][0] if pc else (P_man or DEFAULT_PERIOD)
    out = {"rule": "least_filled (allowance: max per offset <= ceil(n/P)+2 per link and build; single-arm split <= 4 "
                   "per link)", "period": P, "period_source": "lilt_rotation.period" if pc else
           ("manifest lcm of arm counts" if P_man else "default 6"), "sessions_dir": rotation.get("sessions_dir"),
           "sessions_found": bool(rotation.get("found")), "unstamped_lilt_sessions": rotation.get("unstamped") or {},
           "unreadable_session_files": rotation.get("unreadable", 0), "server_completed_non_test": [],
           "analysed_participants": [], "offset_source": {}, "stamp_link_differs_from_group": 0,
           "stamp_differs_from_served_list": 0, "warnings": []}
    cells = {}
    for s in stamps:
        if s["completed"]:
            cells.setdefault((s["link"], s["build"], s["period"]), [0] * s["period"])[s["offset"]] += 1
    for (link, build, per), c in sorted(cells.items()):
        e = _list_counts_entry(link, build, per, c, "(server, completed non-test sessions)")
        out["server_completed_non_test"].append(e)
        if e["warning"]:
            out["warnings"].append(e["warning"])
    files_of, sid_of = {}, {}
    for r in rows_all:
        p = r.get("participant_code")
        sid_of.setdefault(p, r.get("session_id") or "")
        if r.get("block") in BALANCE_BLOCKS:
            files_of.setdefault(p, set()).add(r.get("stimulus_file"))
    sets_by_P = {}

    def served_k(p, per):
        if not imap or not files_of.get(p):
            return None
        if per not in sets_by_P:
            sets_by_P[per] = _offset_file_sets(imap, per)
        ks = [k for k, S in enumerate(sets_by_P[per]) if files_of[p] <= S]
        return ks[0] if len(ks) == 1 else None

    src, acells = collections.Counter(), {}
    for p in sorted(sid_of):
        st = by_sid.get(sid_of[p]) if sid_of[p] else None
        if st and not st["test"]:
            link, build, per, k = st["link"], st["build"], st["period"], st["offset"]
            src["stamp"] += 1
            if group_of and group_of.get(p) and link != group_of.get(p):
                out["stamp_link_differs_from_group"] += 1
            if P_man and per == P_man:
                sk = served_k(p, per)
                if sk is not None and sk != k:
                    out["stamp_differs_from_served_list"] += 1
        else:
            per = P_man or P
            k = served_k(p, per)
            if k is None:
                src["unresolved"] += 1
                continue
            link, build = (group_of or {}).get(p, "?"), "(no stamp: reconstructed)"
            src["reconstructed_from_served_files"] += 1
        acells.setdefault((link, build, per), [0] * per)[k] += 1
    for (link, build, per), c in sorted(acells.items()):
        e = _list_counts_entry(link, build, per, c, "(CSV participants analysed)")
        out["analysed_participants"].append(e)
        if e["warning"]:
            out["warnings"].append(e["warning"])
    out["offset_source"] = dict(src)
    if out["stamp_differs_from_served_list"]:
        out["warnings"].append("%d participant(s) were served a different list than their lilt_rotation stamp says "
                               "(manifest analysed != build served?)" % out["stamp_differs_from_served_list"])
    return out


def realised_balance(rows, pairs, imap, group_of=None, period=DEFAULT_PERIOD):
    """Descriptive balance audit (no test depends on it). Thresholds follow the LILTROT v6 least-filled allocation,
    which balances lists per link only up to a residual of about 2 (abandoned places hold a list for W minutes;
    exclusions still consume a list): a vector (list) may be shared by ceil(n_link / P) + 2 participants and a per-clip
    single-arm split may reach 4 per link. group_of (pooled call): slot k on each link serves the same vector, so the
    pooled allowances are the sums over links (sum of ceil(n_link / P) + 2; 4 x number of links). P = period
    (lilt_rotation.period when stamps exist, else the manifest's lcm of arm counts, else 6)."""
    period = int(period or DEFAULT_PERIOD)
    out = {"H_left_share": {}, "single_arm_counts_per_voice_pair": {}, "rotation_period": period,
           "vector_multiplicity": {}, "warnings": []}
    n_links = len({group_of.get(r.get("participant_code"), "?") for r in rows}) if group_of else 1
    split_allow = BALANCE_SPLIT_ALLOW * max(1, n_links)
    out["single_arm_split_allowed"] = split_allow
    for c in contrasts_present(pairs) + sorted({x["contrast"] for x in pairs if x.get("exp") == 2}):
        cx = [x for x in pairs if x["contrast"] == c]
        if not cx:
            continue
        sh = sum(1 for x in cx if x["H_side"] == "left") / float(len(cx))    # Exp 2: the side of L
        out["H_left_share"][c] = {"share": r3(sh), "n_trials": len(cx)}
        if not 0.4 <= sh <= 0.6:
            out["warnings"].append("%s: H on the left in %.0f%% of trials (outside 40-60%%)" % (c, 100 * sh))
    cnt = {}
    for r in rows:
        if r.get("block") == "single":
            d = ident(r, imap)
            cnt.setdefault(d["vp"], collections.Counter())[d["arm"] or "?"] += 1
    for vp, c in sorted(cnt.items()):
        d = {a: c.get(a, 0) for a in ("E", "H", "R")}
        out["single_arm_counts_per_voice_pair"][vp] = d
        if max(d.values()) - min(d.values()) > split_allow:
            out["warnings"].append("single arms of %s unbalanced beyond the least-filled allowance (split > %d): %s"
                                   % (vp, split_allow, d))
    vec = {}
    for r in rows:
        if r.get("block") in BALANCE_BLOCKS:
            vec.setdefault(r.get("participant_code"), set()).add(r.get("stimulus_file"))
    groups = {}
    for p, s in vec.items():
        groups.setdefault(frozenset(s), []).append(p)
    n = len(vec)
    if group_of:
        n_by_g = collections.Counter(group_of.get(p, "?") for p in vec)
        allowed = sum(int(math.ceil(v / float(period))) + BALANCE_MULT_EXTRA for v in n_by_g.values())
    else:
        allowed = int(math.ceil(n / float(period))) + BALANCE_MULT_EXTRA if n else 0
    mult = sorted((len(v) for v in groups.values()), reverse=True)
    out["vector_multiplicity"] = {"n_participants": n, "n_distinct_vectors": len(groups),
                                  "max_multiplicity": mult[0] if mult else 0, "allowed_max": allowed}
    if mult and mult[0] > allowed:
        worst = max(groups.values(), key=len)
        out["warnings"].append("%d participants share one arm/side vector (least-filled allows %d for N=%d, period %d; "
                               "rotation collapsed?): %s" % (len(worst), allowed, n, period, sorted(worst)[:6]))
    return out


def _balance_with_groups(rows_all, imap, group_of, rotation=None):
    """realised_balance pooled, plus per listener group (the server allocates lists per link), plus the descriptive
    rotation audit (stamps from the session JSON when given, else the served-list reconstruction). Audit only."""
    audit = rotation_audit(rows_all, imap, group_of, rotation)
    P = audit["period"]
    allp = [x for x in (parse_pair(r, imap) for r in rows_all) if x]
    out = realised_balance(rows_all, allp, imap, group_of=group_of, period=P)
    out["rotation_audit"] = audit
    out["warnings"].extend(audit["warnings"])
    out["per_listener_group"] = {}
    for g in LISTENER_GROUPS:
        ps = {p for p, gg in group_of.items() if gg == g}
        if not ps:
            continue
        rb = realised_balance([r for r in rows_all if r.get("participant_code") in ps], [x for x in allp if x["p"] in ps], imap,
                              period=P)
        out["per_listener_group"][g] = {"H_left_share": {k: v["share"] for k, v in rb["H_left_share"].items()},
                                        "vector_multiplicity": rb["vector_multiplicity"], "warnings": rb["warnings"]}
    return out


def exclusions(rows, imap, background=None, focus_field=FOCUS_PRIMARY):
    """Plan section 5 steps 2-4 on already-loaded rows. Returns (kept_rows, flow, per_participant)."""
    background = background or {}
    parts = sorted({r.get("participant_code") for r in rows})
    per, inel = {}, {}
    for p in parts:
        ok, why = eligibility(p, background.get(p))
        per[p] = {"eligible": ok, "eligibility_note": why}
        if not ok:
            inel[p] = why
    kept = [r for r in rows if screen_kept(r, focus_field)]
    blank = sum(1 for r in rows if r.get("block") in SCORED_BLOCKS and focus_count(r, focus_field) is None)
    dcount = design_counts(imap)
    served, retained = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for r in rows:
        served[r.get("participant_code")][contrast_of_row(r, imap) or r.get("block")] += 1
    for r in kept:
        retained[r.get("participant_code")][contrast_of_row(r, imap) or r.get("block")] += 1
    voice_pairs_both = collections.Counter()
    for p in parts:                                                    # voice pairs with BOTH takes retained
        seen = collections.defaultdict(set)
        for r in kept:
            if r.get("participant_code") == p and contrast_of_row(r, imap) == VOICE_ARM:
                d = ident(r, imap)
                seen[d["vp"] or d["actor"] + "_" + d["emo"]].add(d["level"])
        voice_pairs_both[p] = sum(1 for v in seen.values() if {"normal", "strong"} <= v)
    low, mc_drop, hr_drop, vc_drop, exp2_drop = {}, {}, {}, {}, {}
    for p in parts:
        det = {}
        for c in AV_CONTRASTS:
            den = max(served[p][c], dcount.get(c, 0))
            det[c] = "%d/%d" % (retained[p][c], den)
            if den and retained[p][c] < RETAIN_FRAC * den:
                low[p] = det
        den_m = max(served[p][MUTE_CONTRAST], dcount.get(MUTE_CONTRAST, 0))
        if den_m and retained[p][MUTE_CONTRAST] < MC_MIN_SCREENS:
            mc_drop[p] = "%d/%d silent screens retained" % (retained[p][MUTE_CONTRAST], den_m)
        den_h = max(served[p][HR_NORMAL_CONTRAST], dcount.get(HR_NORMAL_CONTRAST, 0))
        if den_h and retained[p][HR_NORMAL_CONTRAST] < RETAIN_FRAC * den_h:
            hr_drop[p] = "%d/%d H_vs_R@normal retained" % (retained[p][HR_NORMAL_CONTRAST], den_h)
        den_v = max(served[p][VOICE_ARM], dcount.get(VOICE_ARM, 0))
        if den_v and voice_pairs_both[p] < VC_MIN_PAIRS:
            vc_drop[p] = "%d voice pairs with both takes retained" % voice_pairs_both[p]
        for kind in EXP2_KINDS:                                        # Exp 2: pooled over stimulus languages
            srv = sum(v for c, v in served[p].items() if str(c).startswith(kind + "@"))
            ret = sum(v for c, v in retained[p].items() if str(c).startswith(kind + "@"))
            den = max(srv, sum(v for c, v in dcount.items() if c.startswith(kind + "@")))
            det[kind] = "%d/%d" % (ret, den)
            if den and ret < RETAIN_FRAC * den:
                exp2_drop[p] = "%s %d/%d retained" % (kind, ret, den)
        per[p].update({"served": dict(served[p]), "retained": dict(retained[p]), "retention": det,
                       "voice_pairs_both_takes": voice_pairs_both[p]})
    flow = {"focus_field": focus_field, "participants_completed": len(parts),
            "step2_not_eligible": inel, "background_missing": sorted(p for p in parts if background.get(p) is None),
            "step4_screens_served_scored": sum(1 for r in rows if r.get("block") in SCORED_BLOCKS),
            "step4_screens_dropped_focus": sum(1 for r in rows if not screen_kept(r, focus_field)),
            "step4_focus_field_blank": blank, "step4_participants_low_retention": low,
            "step4_dropped_from_MC_only": mc_drop, "step4_dropped_from_P2b_only": hr_drop,
            "step4_dropped_from_voice_check_only": vc_drop, "step4_dropped_from_exp2_only": exp2_drop,
            "design_counts": dcount}
    return kept, flow, per


def analyse(rows, imap, B=10000, excluded=None, sided="two", descriptives=True, background=None,
            focus_field=FOCUS_PRIMARY, perm_B=PERM_B, rotation=None):
    """All results from already-loaded rows (see load); applies plan section 5 steps 2-4 (exclusions()).
    rotation (load_rotation_stamps) feeds only the descriptive balance audit in results["realised_balance"]."""
    rows_all = list(rows)
    background = background or {}
    rows, flow, per = exclusions(rows_all, imap, background, focus_field)
    parts = sorted({r.get("participant_code") for r in rows_all})
    first_row = {}
    for r in rows_all:
        first_row.setdefault(r.get("participant_code"), r)
    group_of = {p: listener_group(background.get(p), first_row.get(p)) for p in parts}
    catch = catches(rows_all)
    passers = sorted(p for p in parts if catch.get(p, {}).get("n") and catch[p]["passed"] == catch[p]["n"])
    base = [p for p in parts if p not in flow["step2_not_eligible"] and p not in flow["step4_participants_low_retention"]]
    flow["step3_catch_failed_or_missing"] = [p for p in base if p not in passers]
    flow["primary_sample_N"] = len([p for p in base if p in passers])
    flow["sensitivity_sample_N"] = len(base)
    flow["listener_group_of"] = group_of
    flow["listener_group_rule"] = "link (infos test_url); fallback native_lang, CSV culture, ui_language"
    flow["link_x_native_lang"] = {"%s|%s" % k: v for k, v in sorted(collections.Counter(
        ((background.get(p) or {}).get("link") or "?", (background.get(p) or {}).get("native_lang") or "?")
        for p in parts).items())}
    flow["native_lang_differs_from_link"] = sorted(
        p for p in parts if (background.get(p) or {}).get("link") and (background.get(p) or {}).get("native_lang") in LISTENER_GROUPS
        and background[p]["link"] != background[p]["native_lang"])
    flow["primary_N_per_listener_group"] = dict(collections.Counter(group_of[p] for p in base if p in passers))
    flow["exp2_primary_N"] = len([p for p in base if p in passers and p not in flow["step4_dropped_from_exp2_only"]])
    pairs = [x for x in (parse_pair(r, imap) for r in rows) if x]
    pairs2_all = [x for x in pairs if x["exp"] == 2]
    pairs = [x for x in pairs if x["exp"] == 1]                        # Exp 1 only below: P1-P3 never see Exp 2 rows
    voice = [x for x in (parse_voice(r, imap) for r in rows) if x]
    present = contrasts_present(pairs)
    exp2_clean, exp2_clean_note = exp2_clean_cells()
    options = {"hr_normal": HR_NORMAL_CONTRAST in {x["contrast"] for x in pairs}, "voice_check": bool(voice)}
    p2_cells, p2_rule = p2_sensitivity_cells()
    completeness = {}
    for p in parts:
        c = collections.Counter(x["contrast"] for x in pairs if x["p"] == p)
        c2 = collections.Counter(x["kind"] for x in pairs2_all if x["p"] == p)
        completeness[p] = {"single": sum(1 for r in rows if r.get("participant_code") == p and r.get("block") == "single"),
                           **{k: c.get(k, 0) for k in present}, **{k: c2.get(k, 0) for k in EXP2_KINDS},
                           "catch": catch.get(p, {"n": 0, "passed": 0}), "listener_group": group_of[p]}
        if options["voice_check"]:
            completeness[p]["voice_only"] = sum(1 for x in voice if x["p"] == p)
        completeness[p].update(per.get(p, {}))
    res = {"n_rows": len(rows_all), "n_rows_after_focus": len(rows), "excluded_rows": excluded or {}, "flow": flow,
           "participants": parts, "catch_passers": passers,
           "catch_failed_or_missing": [p for p in parts if p not in passers], "completeness": completeness,
           "optional_blocks_present": options,
           "realised_balance": _balance_with_groups(rows_all, imap, group_of, rotation),
           "samples": {}, "primary_sample": "catch_pass",
           "hypotheses_text": {"P1": "H over E pooled over voice levels", "P2": "H over R (time-reversed) at strong",
                               "P3": "H over E larger at strong than normal (within participant, paired by voice pair)",
                               "MC": "silent H over E at strong (visibility)",
                               "P2b": "(optional block) H over R at normal; secondary",
                               "MC2": "(optional block) audio only: strong take rated more intense than normal",
                               "P2_sensitivity": "P2 on the cells whose R is a clean timing control (descriptive)",
                               "P4": "(Exp 2) L over E, pooled over stimulus languages cn/en/jp",
                               "P5": "(Exp 2) L over Lrev (time-reversed L), pooled over stimulus languages",
                               "P5_sensitivity": "(Exp 2) P5 on the cells whose Lrev is a clean timing control"},
           "p2_sensitivity_rule": p2_rule, "p2_sensitivity_cells": sorted(p2_cells),
           "exp2_present": bool(pairs2_all), "exp2_lrev_clean_cells": sorted(exp2_clean),
           "exp2_lrev_clean_note": exp2_clean_note,
           "note": "score_H > 0 = H preferred. Participant bootstrap = percentile CI over participant means. Samples: "
                   "catch_pass = eligible, retention rule, catch passed (PRIMARY); all = eligible + retention rule."}
    mc_drop, hr_drop, vc_drop = (set(flow["step4_dropped_from_MC_only"]), set(flow["step4_dropped_from_P2b_only"]),
                                 set(flow["step4_dropped_from_voice_check_only"]))
    for sname, keep in (("catch_pass", set(p for p in base if p in passers)), ("all", set(base))):
        px = [x for x in pairs if x["p"] in keep]
        vx = [x for x in voice if x["p"] in keep and x["p"] not in vc_drop]
        rx = [r for r in rows if r.get("participant_code") in keep]
        S = {"n_participants": len(keep), "pairs": {}}
        S["side_bias_mean_v_AV"] = r3(mean([x["v"] for x in px if x["contrast"] in AV_CONTRASTS + [HR_NORMAL_CONTRAST]]))
        for c in present:
            cx = [x for x in px if x["contrast"] == c]
            if c == MUTE_CONTRAST:
                cx = [x for x in cx if x["p"] not in mc_drop]
            if c == HR_NORMAL_CONTRAST:
                cx = [x for x in cx if x["p"] not in hr_drop]
            S["pairs"][c] = summary(cx, B, "%s|%s" % (sname, c))
        S["hypotheses"], S["family"] = hypotheses(px, B, sname, sided)
        S["manipulation_check"] = S["pairs"][MUTE_CONTRAST]
        S["optional"] = optional_hypotheses([x for x in px if x["p"] not in hr_drop], B, sname)   # {} unless rows exist
        S["voice_check"] = voice_check_summary(vx, B, sname)          # {} unless voice_only rows exist
        if p2_cells:
            sx = [x for x in px if x["contrast"] == "H_vs_R@strong"
                  and ("%s_%s" % (x["vp"], "s" if x["level"] == "strong" else "n")) in p2_cells]
            S["p2_sensitivity"] = dict(summary(sx, B, "%s|P2sens" % sname), n_cells=len(p2_cells),
                                       n_cells_seen=len({x["vp"] for x in sx}))
        # ---- Experiment 2 (block pair_ml): its own participant rule, its own Holm family
        e2drop = set(flow["step4_dropped_from_exp2_only"])
        px2 = [x for x in pairs2_all if x["p"] in keep and x["p"] not in e2drop]
        bias = unit_means([x for x in px if x["contrast"] in AV_CONTRASTS], "p", key="v")
        S["exp2"] = exp2_summary(px2, B, sname + "|exp2", group_of, sided, bias=bias, clean_cells=exp2_clean,
                                 descriptives=descriptives)
        # ---- listener groups: N always; per-group secondary tables and exploratory group differences with descriptives
        S["listener_group_N"] = dict(collections.Counter(group_of[p] for p in keep))
        if descriptives:
            per_g = {}
            for g in LISTENER_GROUPS:
                kg = {p for p in keep if group_of[p] == g}
                if not kg:
                    continue
                pg = [x for x in px if x["p"] in kg]
                Hg, Fg = hypotheses(pg, B, "%s|g_%s" % (sname, g), sided)
                mcg = summary([x for x in pg if x["contrast"] == MUTE_CONTRAST and x["p"] not in mc_drop], B,
                              "%s|g_%s|MC" % (sname, g))
                e2g = exp2_summary([x for x in px2 if x["p"] in kg], B, "%s|g_%s|exp2" % (sname, g), group_of, sided,
                                   descriptives=False)
                per_g[g] = {"n_participants": len(kg), "P1-P3": {k: Hg[k]["participants"] for k in FAMILY},
                            "holm_exp1_within_group": Fg["holm_primary"], "MC": mcg.get("participants"),
                            "P4-P5": {k: e2g["hypotheses"][k]["participants"] for k in EXP2_FAMILY} if e2g else {},
                            "holm_exp2_within_group": (e2g.get("family") or {}).get("holm_primary") if e2g else {}}
            S["by_listener_group"] = per_g
            dvs = {"P1": p1_values(px), "P2": unit_means([x for x in px if x["contrast"] == "H_vs_R@strong"], "p"),
                   "P3": p3_values(px),
                   "MC": unit_means([x for x in px if x["contrast"] == MUTE_CONTRAST and x["p"] not in mc_drop], "p"),
                   "P4": exp2_dv(px2, "L_vs_E"), "P5": exp2_dv(px2, "L_vs_Lrev")}
            S["group_differences_exploratory"] = {k: perm_groups(v, group_of, perm_B, "%s|grp|%s" % (sname, k))
                                                  for k, v in dvs.items() if v}
            S["per_emotion"] = describe(px, B, sname, "emo")
            S["per_actor"] = describe(px, B, sname, "actor")
            S["per_voice_pair"] = describe(px, B, sname, "vp")
            S["singles"] = singles_summary(rx, imap, B, sname + "|single")
        res["samples"][sname] = S
    long_rows = [{"participant": x["p"], "exp": x["exp"], "voice_pair": x["vp"], "actor": x["actor"], "emotion": x["emo"],
                  "level": x["level"], "contrast": x["contrast"], "stim_language": x["stim_lang"],
                  "listener_group": group_of.get(x["p"], "?"), "native_stimulus": int(x["stim_lang"] == group_of.get(x["p"])),
                  "H_side": x["H_side"], "ccr_raw": x["v"],
                  "score_H": x["s"], "rt_ms": x["rt_ms"], "n_replays": x["n_replays"], "catch_pass": int(x["p"] in passers),
                  "in_primary": int(x["p"] in base and x["p"] in passers
                                    and not (x["exp"] == 2 and x["p"] in flow["step4_dropped_from_exp2_only"]))}
                 for x in pairs + pairs2_all]
    res["voice_long"] = [{"participant": x["p"], "voice_pair": x["vp"], "actor": x["actor"], "emotion": x["emo"],
                          "level": x["level"], "response": x["resp"], "correct": x["acc"], "intensity": x["intensity"],
                          "rt_ms": x["rt_ms"], "catch_pass": int(x["p"] in passers)} for x in voice]
    return res, long_rows


def _fmt_p(p):
    return "-" if p is None else "%.3g" % p


def print_report(res):
    S = res["samples"][res["primary_sample"]]
    fz = res.get("plan_freeze")
    if fz:
        print("PLAN_FREEZE.sha256: %s" % ("not found (NOT FROZEN)" if not fz["exists"] else
                                          ("all OK" if fz["all_ok"] else "FAILED -- a deviation to explain")))
        for l in fz["lines"]:
            print("  %s: %s" % (l.get("file", l.get("line")), l["status"]))
        if fz.get("frozen_at"):
            print("  frozen_at: %s" % fz["frozen_at"].splitlines()[0])
    print("%d rows, %d participants (%d catch passers; failed/missing: %s); excluded %s"
          % (res["n_rows"], len(res["participants"]), len(res["catch_passers"]), res["catch_failed_or_missing"],
             res["excluded_rows"]))
    fl = res.get("flow", {})
    print("  flow (%s): not eligible %s | background missing %d | catch failed %s | scored screens %d, dropped %d, blank %d "
          "| low retention %s | MC-only drop %s | P2b-only drop %s | VC-only drop %s | primary N %s | sensitivity N %s"
          % (fl.get("focus_field"), fl.get("step2_not_eligible"), len(fl.get("background_missing", [])),
             fl.get("step3_catch_failed_or_missing"), fl.get("step4_screens_served_scored", 0),
             fl.get("step4_screens_dropped_focus", 0), fl.get("step4_focus_field_blank", 0),
             fl.get("step4_participants_low_retention"), list(fl.get("step4_dropped_from_MC_only", {})),
             list(fl.get("step4_dropped_from_P2b_only", {})), list(fl.get("step4_dropped_from_voice_check_only", {})),
             fl.get("primary_sample_N"), fl.get("sensitivity_sample_N")))
    fam = S["family"]
    for k in FAMILY:
        d = S["hypotheses"][k]["participants"]
        c = S["hypotheses"][k]["voice_pairs"]
        print("  %s  mean %s CI %s dz %s (n=%d) | t p two %s one %s | Holm(%s) %s | voice pairs %s CI %s"
              % (k, d["mean"], d["boot95"], d["dz"], d["n"], _fmt_p(d["p_t_two_sided"]), _fmt_p(d["p_t_one_sided"]),
                 fam["sided_primary"], _fmt_p(fam["holm_primary"].get(k)), c["mean"], c["boot95"]))
    e2 = S.get("exp2") or {}
    for k in EXP2_FAMILY if e2 else ():
        d = e2["hypotheses"][k]["participants"]
        c = e2["hypotheses"][k]["cells"]
        print("  %s (Exp 2)  mean %s CI %s dz %s (n=%d) | t p two %s one %s | Holm(%s, P4-P5) %s | cells %s CI %s"
              % (k, d["mean"], d["boot95"], d["dz"], d["n"], _fmt_p(d["p_t_two_sided"]), _fmt_p(d["p_t_one_sided"]),
                 e2["family"]["sided_primary"], _fmt_p(e2["family"]["holm_primary"].get(k)), c["mean"], c["boot95"]))
    for lang, dd in sorted((e2.get("per_stim_language") or {}).items()):
        print("    stimulus %s: %s" % (lang, "  ".join("%s %s CI %s" % (kind, dd[kind].get("participants", {}).get("mean"),
                                                                      dd[kind].get("participants", {}).get("boot95"))
                                                 for kind in EXP2_KINDS if dd[kind].get("n_trials"))))
    for kind, dd in sorted((e2.get("native_stimulus_exploratory") or {}).items()):
        q = dd["native_minus_nonnative"]
        print("    exploratory native - non-native stimulus, %s: %s CI %s (n=%d) p %s"
              % (kind, q["mean"], q["boot95"], q["n"], _fmt_p(q["p_t_two_sided"])))
    if e2.get("P5_sensitivity"):
        q = e2["P5_sensitivity"]["participants"]
        print("    P5 sensitivity (clean-Lrev cells %s)  mean %s CI %s (n=%d) -- descriptive"
              % (e2["P5_sensitivity"]["cells"], q["mean"], q["boot95"], q["n"]))
    print("  listener groups = link (N in this sample): %s | link|native_lang %s | native_lang != link: %s"
          % (S.get("listener_group_N"), res.get("flow", {}).get("link_x_native_lang"),
             res.get("flow", {}).get("native_lang_differs_from_link")))
    for g, d in sorted((S.get("by_listener_group") or {}).items()):
        print("    group %s (n=%d, secondary): %s" % (g, d["n_participants"], "  ".join(
            "%s %s %s" % (k, v.get("mean"), v.get("boot95")) for k, v in list(d["P1-P3"].items()) + list(d["P4-P5"].items()))))
    for k, d in sorted((S.get("group_differences_exploratory") or {}).items()):
        print("    exploratory group difference %s: means %s p_perm %s" % (k, d.get("means"), _fmt_p(d.get("p_perm"))))
    ps = S.get("p2_sensitivity")
    if ps and ps.get("n_trials"):
        q = ps["participants"]
        print("  P2 sensitivity (clean-R cells, %d of 12)  mean %s CI %s (n=%d) | t p two %s -- descriptive"
              % (ps["n_cells"], q["mean"], q["boot95"], q["n"], _fmt_p(q["p_t_two_sided"])))
    for k, d in sorted(S.get("optional", {}).items()):
        q = d["participants"]
        print("  %s (optional, secondary)  mean %s CI %s dz %s (n=%d) | t p two %s one %s -- not in the Holm family"
              % (k, q["mean"], q["boot95"], q["dz"], q["n"], _fmt_p(q["p_t_two_sided"]), _fmt_p(q["p_t_one_sided"])))
    vc = S.get("voice_check") or {}
    if vc:
        q = vc["strong_minus_normal"]
        print("  voice check (optional)  intensity strong-normal %s CI %s dz %s (n=%d) | t p two %s one %s | accuracy "
              "normal %s strong %s -- not in the Holm family"
              % (q["mean"], q["boot95"], q["dz"], q["n"], _fmt_p(q["p_t_two_sided"]), _fmt_p(q["p_t_one_sided"]),
                 vc["by_level"]["normal"]["accuracy_participants"]["mean"], vc["by_level"]["strong"]["accuracy_participants"]["mean"]))
    for c in S["pairs"]:
        d = S["pairs"][c]
        if not d.get("n_trials"):
            print("  %-20s no rows" % c)
            continue
        q = d["participants"]
        print("  %-20s score_H participants %s CI %s (n=%d) | voice pairs %s CI %s | pref H/other/tie %s/%s/%s | t p %s"
              % (c, q["mean"], q["boot95"], q["n"], d["voice_pairs"]["mean"], d["voice_pairs"]["boot95"], d["pref_H"],
                 d["pref_other"], d["tie"], _fmt_p(q["p_t_two_sided"])))
    print("  side bias (AV, + = right): %s" % S["side_bias_mean_v_AV"])
    sens = res.get("sensitivity_focus_lost")
    if sens:
        SS = sens["samples"][res["primary_sample"]]
        print("  sensitivity (focus_lost drop; primary N %s): %s | MC mean %s p %s"
              % (sens["flow"].get("primary_sample_N"),
                 " ".join("%s %s p %s" % (k, SS["hypotheses"][k]["participants"]["mean"],
                                          _fmt_p(SS["family"]["holm_primary"].get(k))) for k in FAMILY),
                 SS["manipulation_check"].get("participants", {}).get("mean"),
                 _fmt_p(SS["manipulation_check"].get("participants", {}).get("p_t_two_sided"))))
    rb = res.get("realised_balance", {})
    print("  realised balance: H-left share %s | vectors %s | %s"
          % ({k: v["share"] for k, v in rb.get("H_left_share", {}).items()}, rb.get("vector_multiplicity"),
             ("WARNINGS: %s" % rb["warnings"]) if rb.get("warnings") else "no warnings"))
    ra = rb.get("rotation_audit") or {}
    if ra:
        print("  rotation audit (descriptive; %s): P %s from %s | offsets from %s | sessions dir %s | unstamped lilt "
              "sessions %s | stamp != served list %s | stamp link != group %s"
              % (ra.get("rule"), ra.get("period"), ra.get("period_source"), ra.get("offset_source"),
                 "found" if ra.get("sessions_found") else "not given", ra.get("unstamped_lilt_sessions"),
                 ra.get("stamp_differs_from_served_list"), ra.get("stamp_link_differs_from_group")))
        for lab, key in (("server completed non-test", "server_completed_non_test"), ("CSV analysed", "analysed_participants")):
            for e in ra.get(key) or []:
                print("    %-26s link %s build %s P %d: per offset %s (n %d, spread %d, max %d, allowed %d)%s"
                      % (lab, e["link"], e["build"][:12], e["period"], e["counts_per_offset"], e["n"], e["spread"],
                         e["max_per_offset"], e["allowed_max_per_offset"], "  WARNING" if e["warning"] else ""))
    for arm, d in sorted(S.get("singles", {}).get("arms", {}).items()):
        print("  single %-2s vivid %s %s  natural %s %s  acc %s %s" % (
            arm, d["vividness"]["mean_participants"], d["vividness"]["boot95_participants"],
            d["naturalness"]["mean_participants"], d["naturalness"]["boot95_participants"],
            d["accuracy"]["mean_participants"], d["accuracy"]["boot95_participants"]))
    for k, d in S.get("singles", {}).get("differences", {}).items():
        print("  single %s: vivid %s %s  natural %s %s  acc %s %s" % (
            k, d["vividness"]["mean"], d["vividness"]["boot95"], d["naturalness"]["mean"], d["naturalness"]["boot95"],
            d["accuracy"]["mean"], d["accuracy"]["boot95"]))


def write_outputs(res, long_rows, out, tag=""):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jp = out / ("lilt_v6_results%s.json" % tag)
    with io.open(jp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    lp = out / ("lilt_v6_long%s.csv" % tag)
    with io.open(lp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(long_rows[0]) if long_rows else ["participant"])
        w.writeheader()
        w.writerows(long_rows)
    vrows = res.get("voice_long") or []
    if vrows:                                                          # (--voice-check) one row per audio-only screen
        with io.open(out / ("lilt_v6_voice_long%s.csv" % tag), "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(vrows[0]))
            w.writeheader()
            w.writerows(vrows)
    return jp, lp


def run(data_dir, imap, B=10000, out=None, tag="", quiet=False, sided="two", files_only=True, freeze=PLAN_FREEZE,
        sessions_dir=None):
    """sessions_dir: <webapp data>/sessions for the descriptive rotation audit (lilt_rotation stamps); None = no stamps
    (the audit then reconstructs the list from the served files)."""
    files = set(imap) if (imap and files_only) else None
    rows, excl = load(data_dir, files)
    bg = load_background(data_dir)
    rot = load_rotation_stamps(sessions_dir) if sessions_dir else None
    res, long_rows = analyse(rows, imap or {}, B, excl, sided, background=bg, focus_field=FOCUS_PRIMARY, rotation=rot)
    res["plan_freeze"] = freeze_check(freeze) if freeze else None
    sens, _ = analyse(rows, imap or {}, B, excl, sided, descriptives=False, background=bg, focus_field=FOCUS_SENS)
    res["sensitivity_focus_lost"] = {"flow": sens["flow"], "samples": {
        k: {"n_participants": v["n_participants"], "hypotheses": v["hypotheses"], "family": v["family"],
            "manipulation_check": v["manipulation_check"], "exp2": v.get("exp2")} for k, v in sens["samples"].items()}}
    if not quiet:
        print_report(res)
    if out:
        jp, lp = write_outputs(res, long_rows, out, tag)
        if not quiet:
            print("written", jp, "and", lp)
    return res


# ---------------------------------------------------------------------------------------------------- simulation
def _builder():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import build_lilt_v6 as BV
    return BV


def derive_like_server(row):
    """server._derive for the two presentations of this study (AV single, AV_pair)."""
    resp, pres = row.get("response_emotion") or "", row.get("presentation")
    corr = ""
    if pres == "AV_pair":
        va = str(row.get("voice_arm") or "")
        side = ("left" if resp == "left" or resp[:1] == "L" else "right" if resp == "right" or resp[:1] == "R" else "same")
        row["chosen_side"] = side
        if va.startswith("correct:"):
            corr = 1 if side == va.split(":", 1)[1] else 0
    elif pres == "AV" and resp and row.get("congruency") == "congruent":
        corr = 1 if resp == row.get("voice_emotion") else 0
    elif pres == "voice_only" and resp:                                # the optional voice check: server scores it as
        corr = 1 if resp == row.get("voice_emotion") else 0            # any voice_only row
    row["correct"] = corr
    row["catch_correct"] = corr if row.get("is_catch") else ""
    return row


SERVE_ORDER = ("single", "practice", "pair", "pair_catch", EXP2_BLOCK, "pair_mute", "voice_only")


def serve_v6(items, n_prior):
    """n_prior = the list offset k. Pre-LILTROT servers used k = completed non-test sessions (of the link), so overlapping
    starts shared a list; the LILTROT v6 server picks k by least-filled allocation and stamps it (lilt_rotation.offset).
    List k serves exactly what n_prior = k served before, so this function covers both.
    server.py mode "lilt" as the v6 FINAL design needs it: groups sorted by name, arms[(i + n_prior) % n_arms] with
    arms sorted by media path; block order single, practice, pair + pair_catch, pair_ml (Exp 2, AFTER the English AV
    pairs, BEFORE the silent pairs), pair_mute, voice_only. The live server has no pair_ml / voice_only line until the
    v6 server patch is applied; build_lilt_v6.serve_like_server is the builder's copy of the same rule."""
    groups = {}
    for it in items:
        groups.setdefault(it.get("pick_group") or ("media/" + it["file"]), []).append(it)
    chosen = []
    for i, g in enumerate(sorted(groups)):
        arms = sorted(groups[g], key=lambda z: "media/" + z["file"])
        chosen.append(arms[(i + n_prior) % len(arms)])
    by = {}
    for x in chosen:
        by.setdefault(x["block"], []).append(x)
    return [x for b in SERVE_ORDER for x in by.get(b, [])]


def side_of_A(it):
    """Side of the arm that the score counts toward: L in Exp 2, H in Exp 1."""
    cm = it.get("cue_manip") or ""
    if "|" not in cm:
        return ""
    a = "L" if (it.get("voice_arm") or "").startswith("L_vs_") else "H"
    left, right = cm.split("|")[0].split(":", 1)[-1], cm.split("|")[1].split(":", 1)[-1]
    return "left" if left == a else "right" if right == a else ""


def exp2_items_standin(tok_L="r5b_lilt_nonod-g2", tok_E="before", tok_Lrev="r5b_lilt_nonod_rev-g2",
                       cond="e10m10_d5_h", look="danielle_v9"):
    """SIMULATION ONLY: the Exp 2 items in the contract ANALYSIS_PLAN_V6_EXP2.md section 3 fixes (block pair_ml,
    presentation AV_pair, voice_arm L_vs_E@<lang> / L_vs_Lrev@<lang>, cue_manip L:x|R:y, stim_language, base_cell =
    the pool cell, 2-arm pick groups). Used only while build_lilt_v6.py has no pair_ml block."""
    import hashlib
    toks = {"L": tok_L, "E": tok_E, "Lrev": tok_Lrev}
    items = []
    for lang in STIM_LANGS:
        for cell in EXP2_CELLS[lang]:
            emo = cell.split("_")[-1] if lang == "cn" else cell.split("_")[1]
            for kind, foil in (("L_vs_E", "E"), ("L_vs_Lrev", "Lrev")):
                contrast = "%s@%s" % (kind, lang)
                code = hashlib.sha1(("pair_ml|%s|%s" % (contrast, cell)).encode()).hexdigest()[:10]
                # the 18 pair_ml groups sort consecutively (cn E x3, cn Lrev x3, en ..., jp ...); flipping the en block
                # gives L on the left in 4 or 5 of 9 per contrast and L on opposite sides in a cell's two contrasts
                arms = ((1, (foil, "L")), (2, ("L", foil))) if lang != "en" else ((1, ("L", foil)), (2, (foil, "L")))
                for k, (L, R) in arms:
                    f = "v6_pair_ml_%s_%d.mp4" % (code, k)
                    items.append({"id": f[:-4], "file": f, "block": EXP2_BLOCK, "presentation": "AV_pair",
                                  "stim_language": lang, "voice_emotion": emo, "face_emotion": "", "congruency": "",
                                  "text_content": "", "voice_id": cell, "renderer": "pair", "utterance": cell,
                                  "base_cell": cell, "voice_level": "", "actor": "", "cue_manip": "L:%s|R:%s" % (L, R),
                                  "face_motion": "%s__%s__%s__%s|%s__%s__%s__%s" % (cell, toks[L], cond, look, cell,
                                                                                    toks[R], cond, look),
                                  "voice_arm": contrast, "contrast": contrast,
                                  "pick_group": "pair_ml_%s_%s_%s" % (lang, kind, cell), "face_deg": "d5"})
    return items


def simulate_rows(items, participants, effects=None, singles_eff=None, side_bias=0.0, seed=1, catch_fail=(),
                  always=None, sd_p=0.4, sd_e=1.0, groups=None, per_link=False, offsets=None):
    """Server-rotated sessions (serve_v6) with answers from a latent model:
    score = effects[voice_arm] (else effects[kind]) + native bonus effects["native:<kind>"] when the Exp 2 stimulus
    language is the participant's group + u_participant + e;  CCR v = +/-score + side_bias, rounded, clipped to -3..3.
    per_link: n_prior counts only earlier participants of the same group (the 17:36 server patch counts per link).
    offsets: {participant: list offset k} overrides both (least-filled or collapsed realisations for the balance audit)."""
    BV = _builder()
    effects, singles_eff, always, groups = effects or {}, singles_eff or {}, always or {}, groups or {}
    rng = random.Random(seed)
    out = {}
    seen_link = collections.Counter()
    for n, pc in enumerate(participants):
        grp = groups.get(pc, "cn")
        served = serve_v6(items, (offsets or {}).get(pc, seen_link[grp] if per_link else n))
        seen_link[grp] += 1
        u, uv = rng.gauss(0, sd_p), rng.gauss(0, 0.5)
        rows = []
        for i, it in enumerate(served):
            row = {"response_ts": 1757800000000 + 5000 * i, "session_id": "sim-" + pc, "participant_code": pc,
                   "culture": grp, "ui_language": grp, "mode": "lilt", "block": it["block"], "trial_index": i,
                   "is_practice": 1 if it["block"] == "practice" else 0,
                   "is_catch": 1 if it["block"] == "pair_catch" else 0, "revision": 0, "rt_ms": 1500 + rng.randint(0, 900),
                   "n_replays": 0, "stimulus_id": BV.stim_id(it["file"]), "stimulus_file": it["file"],
                   "stim_language": it.get("stim_language", ""), "voice_emotion": it.get("voice_emotion", ""),
                   "face_emotion": it.get("face_emotion", ""), "congruency": it.get("congruency", ""),
                   "text_content": it.get("text_content", ""), "voice_id": it.get("voice_id", ""),
                   "presentation": it["presentation"], "cue_manip": it.get("cue_manip", ""), "face_deg": it.get("face_deg", ""),
                   "voice_arm": it.get("voice_arm", ""), "renderer": it.get("renderer", ""),
                   "face_motion": it.get("face_motion", ""), "intensity": "", "naturalness": "",
                   "focus_lost": 0, "focus_hidden": 0, "focus_blurred": 0, "focus_away_ms": 0}
            b = it["block"]
            if b == "single":
                arm = it["renderer"]
                acc = singles_eff.get("accuracy", {}).get(arm, 0.6)
                ve = it.get("voice_emotion")
                row["response_emotion"] = ve if rng.random() < acc else rng.choice([e for e in EMOTIONS if e != ve])
                row["intensity"] = max(1, min(7, int(round(4 + singles_eff.get("vivid", {}).get(arm, 0) + uv + rng.gauss(0, 1)))))
                row["naturalness"] = max(1, min(7, int(round(4 + singles_eff.get("natural", {}).get(arm, 0) + rng.gauss(0, 1)))))
            elif b == "practice":
                row["response_emotion"] = "S0"
            elif b in ("pair", "pair_mute", EXP2_BLOCK):
                side = side_of_A(it)
                kind = it["voice_arm"].split("@", 1)[0]
                s = effects.get(it["voice_arm"], effects.get(kind, 0.0)) + u + rng.gauss(0, sd_e)
                if b == EXP2_BLOCK and it.get("stim_language") == grp:
                    s += effects.get("native:" + kind, 0.0)
                v = (s if side == "right" else -s) + side_bias
                code = CODES[max(-3, min(3, int(round(v))))]
                if always.get(pc) == "H3":
                    code = "R3" if side == "right" else "L3"
                elif always.get(pc) == "R3":
                    code = "R3"
                row["response_emotion"] = code
            elif b == "pair_catch":
                want = it["voice_arm"].split(":", 1)[1]
                row["response_emotion"] = want if pc not in catch_fail else ("right" if want == "left" else "left")
            elif b == "voice_only":                                       # (--voice-check) emotion + intensity
                ve = it.get("voice_emotion")
                acc = singles_eff.get("voice_accuracy", {}).get(it.get("voice_level"), 0.6)
                row["response_emotion"] = ve if rng.random() < acc else rng.choice([e for e in EMOTIONS if e != ve])
                bump = effects.get(VOICE_ARM, 0.0) if it.get("voice_level") == "strong" else 0.0
                row["intensity"] = max(1, min(7, int(round(3.5 + bump + uv + rng.gauss(0, 1)))))
            rows.append(derive_like_server(row))
        out[pc] = rows
    return out


def write_sim_csvs(dirpath, sessions, extras=True, background=None, groups=None):
    """background: {pc: dict | None} -> a <pc>_infos.json beside the CSV (as the server files it); default native =
    groups[pc] (else cn); None = no infos file."""
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    for old in list(d.glob("*.csv")) + list(d.glob("*_infos.json")):
        old.unlink()
    background, groups = background or {}, groups or {}
    for pc, rows in sessions.items():
        rows = list(rows)
        bg = background.get(pc, {"native_lang": groups.get(pc, "cn"), "hearing": "normal", "vision": "normal",
                                 "second_lang": "en", "second_level": "inter"})
        if bg is not None:
            with io.open(d / ("%s_infos.json" % pc), "w", encoding="utf-8") as fh:
                json.dump({"username": pc, "age": "30", "gender": "f", "background": bg,
                           "test_url": "/root/lilt_%s" % groups.get(pc, "cn")}, fh)
        if extras and rows:                                               # rows the loader must drop
            pair = next((r for r in rows if r["block"] == "pair"), None)
            if pair:
                rows.append(dict(pair, revision=1, response_emotion="L3", chosen_side="left"))
                rows.append(dict(pair))                                   # a duplicate first response (resume)
        with io.open(d / ("%s.csv" % pc), "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SIM_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    if extras and sessions:
        first = next(iter(sessions.values()))
        with io.open(d / "testzzz.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SIM_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows([dict(r, participant_code="testzzz") for r in first])
        with io.open(d / "inline_testuser.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SIM_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows([dict(r, participant_code="TestInline") for r in first[:5]])


def plan_for_sim(hr_normal=False, voice_check=False):
    BV = _builder()
    sel = BV.placeholder_selection()
    sel["voice_check_pairs"] = list(sel["mute_pairs"])
    items = BV.plan_items(sel, lambda c, a: Path("/sim/%s__%s.mp4" % (c, a)), hr_normal=hr_normal,
                          voice_check=voice_check, wav_of=lambda c: Path("/sim/%s.wav" % c))
    return [it for it in items if it["block"] != EXP2_BLOCK]           # sections A-F test Exp 1 alone


def plan_for_sim_v6():
    """The FINAL 81-screen design: 6 singles, 2 practice, 36 pairs + 1 catch, 18 pair_ml, 6 silent, 12 voice-only.
    Uses the builder's Exp 2 block and singles rule when build_lilt_v6.py has them; otherwise adds the stand-in Exp 2
    items and trims the singles to the complement of the silent subset (ANALYSIS_PLAN_V6.md section 2)."""
    BV = _builder()
    sel = BV.placeholder_selection()
    sel["voice_check_pairs"] = list(sel["mute_pairs"])
    src_of, wav_of = (lambda c, a: Path("/sim/%s__%s.mp4" % (c, a))), (lambda c: Path("/sim/%s.wav" % c))
    try:                                                               # the builder's own contract (--singles 6 --ml)
        sel["single_pairs"] = BV.single_pairs_for(sel, 6)[0]
        sel["ml_cells"] = [c for lang in STIM_LANGS for c in EXP2_CELLS[lang]]
        sel["ml_meta"] = {c: BV.ml_meta({}, c) for c in sel["ml_cells"]}
        items = BV.plan_items(sel, src_of, voice_check=True, wav_of=wav_of, ml=True,
                              src_ml=lambda c, a: Path("/sim/%s__%s__e10m10_d5_h__danielle_v9.mp4" % (c, BV.ML_TOKENS[a])))
        src = "builder plan_items(--singles 6, ml=True, voice_check=True)"
    except (TypeError, AttributeError, KeyError) as e:                 # an older builder: stand-in items
        sel.pop("single_pairs", None)
        items = BV.plan_items(sel, src_of, voice_check=True, wav_of=wav_of)
        src = "builder (no ml: %s)" % e
    if not any(it["block"] == EXP2_BLOCK for it in items):
        items = items + exp2_items_standin()
        src += " + analysis-side Exp 2 stand-in"
    if len({it["pick_group"] for it in items if it["block"] == "single"}) > 6:
        mute = {it["base_cell"] for it in items if it["block"] == "pair_mute"}
        items = [it for it in items if not (it["block"] == "single" and it["base_cell"] not in mute)]
        src += "; singles trimmed to the silent subset (SINGLES_RULE[6])"
    return items, src


def _strrows(sess):
    return [dict((k, "" if v is None else str(v)) for k, v in r.items()) for rs in sess.values() for r in rs
            if r["block"] != "practice"]


def selftest(out, B=2000, coverage_reps=60):
    """Simulated data only. Returns the number of failed checks."""
    items = plan_for_sim()
    imap = {it["file"]: it for it in items}
    root = Path(out) / "_selftest_sim"
    fails, checks = [], []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        if not ok:
            fails.append(name)

    # A. effects present, 2 catch failers, side bias, CSV round trip
    eff = {"H_vs_E@normal": 0.4, "H_vs_E@strong": 0.9, "H_vs_R@strong": 0.6, MUTE_CONTRAST: 1.2}
    seff = {"vivid": {"E": 0.0, "H": 0.7, "R": 0.7}, "natural": {"E": 0.0, "H": -0.4, "R": -0.6},
            "accuracy": {"E": 0.55, "H": 0.7, "R": 0.55}}
    parts = ["sim%03d" % i for i in range(36)]
    sess = simulate_rows(items, parts, eff, seff, side_bias=0.3, seed=11, catch_fail=("sim003", "sim017"))
    write_sim_csvs(root / "A_effects", sess)
    rows, excl = load(root / "A_effects", set(imap))
    res, long_rows = analyse(rows, imap, B, excl)
    write_outputs(res, long_rows, root, "_A_effects")
    S = res["samples"]["catch_pass"]
    check("A loader drops test / revision / duplicate rows",
          excl.get("revision", 0) == 36 and excl.get("duplicate", 0) == 36 and excl.get("file_skipped", 0) >= 1
          and excl.get("test", 0) == 5 and len(res["participants"]) == 36, excl)
    check("A catch failers found", res["catch_failed_or_missing"] == ["sim003", "sim017"] and S["n_participants"] == 34,
          res["catch_failed_or_missing"])
    n_single = len({it["pick_group"] for it in items if it["block"] == "single"})
    check("A completeness %d singles, 12/12/12 pairs, 6 silent, 1 catch per participant" % n_single,
          all(c["single"] == n_single and c["H_vs_E@normal"] == 12 and c["H_vs_E@strong"] == 12 and c["H_vs_R@strong"] == 12
              and c[MUTE_CONTRAST] == 6 and c["catch"]["n"] == 1 for c in res["completeness"].values()),
          list(res["completeness"].values())[:1])
    for c in PAIR_CONTRASTS:
        d = S["pairs"][c]["participants"]
        check("A %s recovered (CI > 0, mean near %.1f)" % (c, eff[c]),
              d["boot95"][0] > 0 and abs(d["mean"] - eff[c]) < 0.35, (d["mean"], d["boot95"]))
    truth = {"P1": 0.65, "P2": 0.6, "P3": 0.5}
    for k in FAMILY:
        d = S["hypotheses"][k]["participants"]
        check("A %s recovered (true %.2f; CI > 0, one-sided p < two-sided p)" % (k, truth[k]),
              d["boot95"][0] > 0 and abs(d["mean"] - truth[k]) < 0.3 and d["p_t_one_sided"] < d["p_t_two_sided"],
              (d["mean"], d["boot95"], d["p_t_two_sided"], d["p_t_one_sided"]))
    check("A Holm family = P1, P2, P3 (the silent check is not in it)",
          sorted(S["family"]["holm_two_sided"]) == ["P1", "P2", "P3"] and sorted(S["family"]["holm_one_sided"]) == ["P1", "P2", "P3"],
          S["family"])
    check("A side bias recovered (true +0.3 in v units)", 0.15 < S["side_bias_mean_v_AV"] < 0.45, S["side_bias_mean_v_AV"])
    vd = S["singles"]["differences"]["H-E"]["vividness"]
    nd = S["singles"]["differences"]["H-E"]["naturalness"]
    ad = S["singles"]["differences"]["H-E"]["accuracy"]
    check("A singles H-E vividness > 0, naturalness < 0, accuracy near +0.15",
          vd["boot95"][0] > 0 and nd["boot95"][1] < 0 and -0.02 < ad["mean"] < 0.3, (vd, nd, ad))
    check("A accuracy scored against response codes (fear present per arm)",
          "fear" in S["singles"]["accuracy_by_emotion"]
          and all(v["n_trials"] > 0 for v in S["singles"]["accuracy_by_emotion"]["fear"].values()),
          S["singles"]["accuracy_by_emotion"].get("fear"))
    check("A descriptives: 4 emotions, 3 actors, 12 voice pairs",
          len(S["per_emotion"]) == 4 and len(S["per_actor"]) == 3 and len(S["per_voice_pair"]) == 12,
          (sorted(S["per_emotion"]), sorted(S["per_actor"])))
    check("A long CSV one row per scored pair", len(long_rows) == 36 * 42, len(long_rows))
    check("A realised balance: 36 sequential participants give no balance warning", not res["realised_balance"]["warnings"],
          res["realised_balance"]["warnings"][:2])
    # A1. no manifest: the CSV columns + face_motion give the same hypotheses
    rows_nm, _ = load(root / "A_effects", None)
    res_nm, _ = analyse(rows_nm, {}, 500, {})
    same = all(res_nm["samples"]["catch_pass"]["hypotheses"][k]["participants"]["mean"] == S["hypotheses"][k]["participants"]["mean"]
               for k in FAMILY)
    check("A1 manifest-less fallback (CSV + face_motion) reproduces P1-P3 means", same,
          {k: res_nm["samples"]["catch_pass"]["hypotheses"][k]["participants"]["mean"] for k in FAMILY})
    # A2. LILTROT v6 least-filled allocation leaves a residual of up to 2 per list (split up to 4) per link (abandoned places hold a list for W
    # minutes, exclusions consume one): such realisations must NOT warn; a collapsed rotation (the old completed-count
    # rule under simultaneous starts) MUST warn. Offsets given explicitly; no stamps -> the audit reconstructs them.
    def _offs(counts, tag):
        names, offs = [], {}
        for k, c in enumerate(counts):
            for j in range(c):
                names.append("sim%s%d_%d" % (tag, k, j))
                offs[names[-1]] = k
        return names, offs

    def _rb(counts, tag, seed, rotation=None):
        nm, of = _offs(counts, tag)
        sess_ = simulate_rows(items, nm, eff, seff, seed=seed, offsets=of)
        return analyse(_strrows(sess_), imap, 200, {}, descriptives=False, rotation=rotation)[0]["realised_balance"], sess_, of

    rb6 = _rb([2, 1, 1, 1, 1, 0], "L", 21)[0]
    check("A2 least-filled-like: 6 participants with one extra on slot 0 (slot 5 empty) do not warn (P = %s from %s)"
          % (rb6["rotation_audit"]["period"], rb6["rotation_audit"]["period_source"]),
          not rb6["warnings"] and rb6["rotation_audit"]["period"] == 6 and rb6["vector_multiplicity"]["max_multiplicity"] == 2,
          (rb6["warnings"][:2], rb6["vector_multiplicity"]))
    rb12, sess12, o12 = _rb([3, 3, 2, 2, 1, 1], "M", 23)
    a12 = rb12["rotation_audit"]
    check("A2 least-filled-like: 12 participants at spread 2 (3/3/2/2/1/1) do not warn; the audit reconstructs the "
          "offsets from the served files",
          not rb12["warnings"] and rb12["vector_multiplicity"]["max_multiplicity"] == 3
          and [e["counts_per_offset"] for e in a12["analysed_participants"]] == [[3, 3, 2, 2, 1, 1]]
          and a12["offset_source"] == {"reconstructed_from_served_files": 12},
          (rb12["warnings"][:2], rb12["vector_multiplicity"], a12["analysed_participants"], a12["offset_source"]))
    rb24 = _rb([5, 4, 3, 4, 5, 3], "F", 26)[0]
    check("A2 least-filled-like: 24 participants at 5/4/3/4/5/3 (simulated W = 60 false alarm under +1 / split 2: arm "
          "classes 9/9/6) do not warn under ceil(n/P)+2 and split <= 4",
          not rb24["warnings"] and rb24["vector_multiplicity"]["max_multiplicity"] == 5
          and rb24["vector_multiplicity"]["allowed_max"] == 6,
          (rb24["warnings"][:2], rb24["vector_multiplicity"], rb24["single_arm_split_allowed"]))
    rbc = _rb([12, 0, 0, 0, 0, 0], "C", 24)[0]
    check("A2 collapsed rotation (old rule, 12 simultaneous starts all on slot 0) warns: shared vector + COLLAPSED ROTATION",
          any("share one arm/side vector" in w for w in rbc["warnings"]) and any("COLLAPSED ROTATION" in w for w in rbc["warnings"]),
          rbc["warnings"][:3])
    rbp = _rb([15, 2, 4, 0, 4, 3], "P", 25)[0]
    check("A2 old-rule realisation 15/2/4/0/4/3 of 28 (rotation README S3) warns", len(rbp["warnings"]) >= 2
          and any("COLLAPSED ROTATION" in w for w in rbp["warnings"]), rbp["warnings"][:3])
    # A3. lilt_rotation stamps read from synthetic data/sessions/<key>.json fixtures (never the live data)
    secret = "PIDSECRETXYZ"

    def _fixtures(sdir, offs, extra=True, override=None):
        sdir.mkdir(parents=True, exist_ok=True)
        for old in sdir.glob("*.json"):
            old.unlink()

        def put(key, sid, user, off, completed=True, link="cn", stamp=True, mode="lilt"):
            d = {"session_id": sid, "username": user, "mode": mode, "completed": completed,
                 "created_utc": "2026-09-14T10:00:00+00:00"}
            if stamp:
                d["lilt_rotation"] = {"offset": off, "period": 6, "build": "bld1", "built": "2026-09-14T09:00", "link": link,
                                      "rule": "least_filled", "active_min": 60, "counts_before": [0] * 6}
            io.open(sdir / (key + ".json"), "w", encoding="utf-8").write(json.dumps(d))
        for pc_, k_ in offs.items():
            put(pc_, "sim-" + pc_, secret + pc_, (override or {}).get(pc_, k_))
        if extra:
            put("t1", "t1", "testrun1", 5)                             # test users: never counted
            put("t2", "t2", "TestRun2", 5)
            put("open1", "open1", secret + "o", 4, completed=False)    # open: not a completed session
            put("old1", "old1", secret + "u", 0, stamp=False)          # unstamped (pre-patch)
            put("en1", "en1", secret + "e", 2, link="en")              # another link
            put("jan1", "jan1", secret + "j", 1, mode="jan")           # another mode
            io.open(sdir / "broken.json", "w", encoding="utf-8").write("{")

    sdirA = root / "A3_sessions" / "sessions"
    _fixtures(sdirA, o12)
    write_sim_csvs(root / "A3_stamped", sess12)
    res3 = run(root / "A3_stamped", imap, B=200, quiet=True, freeze=None, sessions_dir=sdirA)
    buf = io.StringIO()
    _stdout, sys.stdout = sys.stdout, buf
    try:
        print_report(res3)
    finally:
        sys.stdout = _stdout
    a3 = res3["realised_balance"]["rotation_audit"]
    srv = {(e["link"], e["build"]): e["counts_per_offset"] for e in a3["server_completed_non_test"]}
    check("A3 stamps from the session JSON: offsets from 12 stamps joined by session_id; per (link, build, offset) counts "
          "cn/bld1 3/3/2/2/1/1 for completed non-test sessions (test, open, unstamped, other-mode, broken excluded), en "
          "separate; no warning; no username in the results or the printed report",
          a3["offset_source"] == {"stamp": 12} and a3["period_source"] == "lilt_rotation.period"
          and [e["counts_per_offset"] for e in a3["analysed_participants"]] == [[3, 3, 2, 2, 1, 1]]
          and srv == {("cn", "bld1"): [3, 3, 2, 2, 1, 1], ("en", "bld1"): [0, 0, 1, 0, 0, 0]}
          and a3["unstamped_lilt_sessions"] == {"completed": 1} and a3["unreadable_session_files"] == 1
          and a3["stamp_differs_from_served_list"] == 0 and not res3["realised_balance"]["warnings"]
          and "rotation audit" in buf.getvalue() and secret not in buf.getvalue()
          and secret not in json.dumps(res3) and "testrun" not in json.dumps(res3).lower(),
          (a3, res3["realised_balance"]["warnings"][:2]))
    ncol, ocol = _offs([12, 0, 0, 0, 0, 0], "C")
    sdirC = root / "A3_sessions_collapsed" / "sessions"
    _fixtures(sdirC, ocol, extra=False)
    rbc2 = _rb([12, 0, 0, 0, 0, 0], "C", 24, rotation=load_rotation_stamps(sdirC))[0]
    check("A3 collapsed stamps (12 of 12 on offset 0) warn for the server counts and the analysed participants",
          any("COLLAPSED ROTATION (server" in w for w in rbc2["warnings"])
          and any("COLLAPSED ROTATION (CSV" in w for w in rbc2["warnings"])
          and rbc2["rotation_audit"]["offset_source"] == {"stamp": 12}, rbc2["warnings"][:3])
    sdirX = root / "A3_sessions_mismatch" / "sessions"
    first = sorted(o12)[0]
    _fixtures(sdirX, o12, extra=False, override={first: (o12[first] + 1) % 6})
    rbx = analyse(_strrows(sess12), imap, 200, {}, descriptives=False, rotation=load_rotation_stamps(sdirX))[0]["realised_balance"]
    check("A3 a stamp that disagrees with the list actually served is counted and warned",
          rbx["rotation_audit"]["stamp_differs_from_served_list"] == 1
          and any("different list than their lilt_rotation stamp" in w for w in rbx["warnings"]), rbx["warnings"][:3])

    # B. exact sign bookkeeping: always-H-at-3 and always-R3 raters
    sess = simulate_rows(items, ["simAlwaysH", "simAlwaysR3", "simAlwaysH2", "simAlwaysR3b", "simAlwaysH3", "simAlwaysR3c"],
                         seed=3, always={"simAlwaysH": "H3", "simAlwaysH2": "H3", "simAlwaysH3": "H3",
                                         "simAlwaysR3": "R3", "simAlwaysR3b": "R3", "simAlwaysR3c": "R3"})
    write_sim_csvs(root / "B_signs", sess, extras=False)
    rows, excl = load(root / "B_signs", set(imap))
    res, _ = analyse(rows, imap, 200, excl, descriptives=False)
    px = [x for x in (parse_pair(r, imap) for r in rows) if x]
    for c in PAIR_CONTRASTS:
        mv = unit_means([x for x in px if x["contrast"] == c], "p")
        check("B %s: always-H = +3, always-R3 = 0 (balanced sides)" % c,
              all(mv.get(p) == 3 for p in ("simAlwaysH", "simAlwaysH2", "simAlwaysH3"))
              and all(mv.get(p) == 0 for p in ("simAlwaysR3", "simAlwaysR3b", "simAlwaysR3c")), mv)
    hz = res["samples"]["all"]["hypotheses"]
    p3 = p3_values(px)
    check("B P3 of always-H / always-R3 raters is exactly 0 and P1 of always-H is 3",
          all(v == 0 for v in p3.values()) and len(p3) == 6 and p1_values(px)["simAlwaysH"] == 3, (p3, hz["P1"]["participants"]))

    # D. direction: a negative H_vs_R effect gives a small two-sided but a large one-sided p
    sess = simulate_rows(items, ["simD%02d" % i for i in range(24)], {"H_vs_R@strong": -0.8}, {}, seed=41)
    resd = analyse(_strrows(sess), imap, 500, {}, descriptives=False)[0]["samples"]["all"]["hypotheses"]["P2"]["participants"]
    check("D a reversed effect: P2 mean < 0, two-sided p < .05, one-sided p > .5",
          resd["mean"] < 0 and resd["p_t_two_sided"] < 0.05 and resd["p_t_one_sided"] > 0.5, resd)
    check("D the incomplete-beta t formula agrees with scipy (or scipy absent)",
          abs(_t_two_sided(2.1, 23) - 0.0469) < 0.002 and abs(_t_two_sided(0.5, 11) - 0.627) < 0.003,
          (_t_two_sided(2.1, 23), _t_two_sided(0.5, 11)))

    # E. the OPTIONAL blocks (--hr-normal, --voice-check): present -> analysed as secondary, absent -> nothing appears
    check("E flags off: no optional block in the results",
          res["optional_blocks_present"] == {"hr_normal": False, "voice_check": False}
          and S["optional"] == {} and S["voice_check"] == {} and HR_NORMAL_CONTRAST not in S["pairs"]
          and "voice_only" not in list(res["completeness"].values())[0], (res["optional_blocks_present"], list(S["pairs"])))
    items_f = plan_for_sim(hr_normal=True, voice_check=True)
    imap_f = {it["file"]: it for it in items_f}
    eff_f = dict(eff, **{HR_NORMAL_CONTRAST: 0.3, VOICE_ARM: 1.2})
    seff_f = dict(seff, voice_accuracy={"normal": 0.55, "strong": 0.75})
    sess = simulate_rows(items_f, parts, eff_f, seff_f, side_bias=0.3, seed=12, catch_fail=("sim003",))
    write_sim_csvs(root / "E_options", sess)
    rows, excl = load(root / "E_options", set(imap_f))
    res_f, long_f = analyse(rows, imap_f, B, excl)
    write_outputs(res_f, long_f, root, "_E_options")
    SF = res_f["samples"]["catch_pass"]
    check("E flags on: both optional blocks detected", res_f["optional_blocks_present"] == {"hr_normal": True, "voice_check": True},
          res_f["optional_blocks_present"])
    check("E flags on: completeness 12 H_vs_R@normal + 12 voice_only per participant (and the base counts)",
          all(c[HR_NORMAL_CONTRAST] == 12 and c["voice_only"] == 12
              and c["single"] == len({it["pick_group"] for it in items_f if it["block"] == "single"}) and c["H_vs_E@strong"] == 12
              and c[MUTE_CONTRAST] == 6 for c in res_f["completeness"].values()), list(res_f["completeness"].values())[:1])
    d = SF["optional"]["P2b"]["participants"]
    check("E P2b recovered (true 0.3; CI > 0, mean near)", d["boot95"][0] > 0 and abs(d["mean"] - 0.3) < 0.3, d)
    d = SF["optional"]["P3R"]["participants"]
    check("E P3R = (H-R)@strong - (H-R)@normal recovered (true 0.3)", abs(d["mean"] - 0.3) < 0.3, d)
    v = SF["voice_check"]["strong_minus_normal"]
    check("E voice check: strong minus normal intensity recovered (true 1.2; CI > 0)",
          v["boot95"][0] > 0 and abs(v["mean"] - 1.2) < 0.4 and v["n"] == 35, v)
    bl = SF["voice_check"]["by_level"]
    check("E voice check: accuracy by level near 0.55 / 0.75, 6 pairs per participant per level",
          abs(bl["normal"]["accuracy_participants"]["mean"] - 0.55) < 0.12 and abs(bl["strong"]["accuracy_participants"]["mean"] - 0.75) < 0.12
          and bl["normal"]["n_trials"] == 35 * 6 and bl["strong"]["n_trials"] == 35 * 6,
          {L: (bl[L]["accuracy_participants"]["mean"], bl[L]["n_trials"]) for L in bl})
    check("E flags on: the Holm family is still exactly P1, P2, P3", sorted(SF["family"]["holm_two_sided"]) == ["P1", "P2", "P3"],
          SF["family"])
    same_p = all(abs(SF["hypotheses"][k]["participants"]["mean"] - truth[k]) < 0.3 for k in FAMILY)
    check("E flags on: P1-P3 still recovered with the extra blocks present", same_p,
          {k: SF["hypotheses"][k]["participants"]["mean"] for k in FAMILY})
    check("E flags on: H_vs_R@normal side balance and the long CSV (12 x 4 contrasts + 6 silent = 54 per participant)",
          not res_f["realised_balance"]["warnings"] and len(long_f) == 36 * 54 and len(res_f["voice_long"]) == 36 * 12,
          (res_f["realised_balance"]["warnings"][:1], len(long_f), len(res_f["voice_long"])))
    check("E voice-long CSV written", (root / "lilt_v6_voice_long_E_options.csv").is_file())

    # F. plan section 5: eligibility, focus drops, the retention rule, the MC-only / VC-only drops, the freeze check
    partsF = ["simF%02d" % i for i in range(12)]
    sessF = simulate_rows(items_f, partsF, eff_f, seff_f, seed=77)

    def mark(pc, pred, n, field_vals):
        k = 0
        for r in sessF[pc]:
            if pred(r) and k < n:
                r.update(field_vals)
                k += 1
        return k

    hid = {"focus_hidden": 1, "focus_lost": 1}
    blur = {"focus_blurred": 1, "focus_lost": 1}
    mark("simF01", lambda r: r["voice_arm"] == "H_vs_E@normal", 4, hid)          # 8/12 retained -> excluded
    mark("simF02", lambda r: r["voice_arm"] == "H_vs_E@normal", 3, hid)          # 9/12 retained -> kept, 3 screens dropped
    mark("simF03", lambda r: r["voice_arm"] == MUTE_CONTRAST, 3, hid)            # 3/6 silent -> out of MC only
    mark("simF04", lambda r: r["voice_arm"] == "H_vs_R@strong", 5, blur)         # blur only: primary kept, focus_lost run drops
    mark("simF05", lambda r: r["voice_arm"] == HR_NORMAL_CONTRAST, 4, hid)       # 8/12 H_vs_R@normal -> out of P2b only
    mark("simF06", lambda r: r["block"] == "voice_only" and r["cue_manip"] == "voice:strong", 3, hid)  # 3 pairs both -> out of VC
    for r in sessF["simF07"]:                                                     # blank focus fields -> kept, counted
        r.update({"focus_hidden": "", "focus_lost": "", "focus_blurred": ""})
    bgF = {"simF08": {"native_lang": "other", "hearing": "normal", "vision": "normal"},   # en is a listener group now
           "simF09": {"native_lang": "cn", "hearing": "impaired", "vision": "normal"},
           "simF10": {"native_lang": "cn", "hearing": "na", "vision": "na"},      # prefer not to say -> kept
           "simF11": None}                                                        # no infos file -> kept, counted
    write_sim_csvs(root / "F_exclusions", sessF, extras=False, background=bgF)
    rowsF, exclF = load(root / "F_exclusions", set(imap_f))
    bgl = load_background(root / "F_exclusions")
    resF, longF = analyse(rowsF, imap_f, 300, exclF, background=bgl, descriptives=False)
    fl = resF["flow"]
    check("F eligibility: native 'other' and impaired hearing out; 'na' and missing background kept and counted",
          sorted(fl["step2_not_eligible"]) == ["simF08", "simF09"] and fl["background_missing"] == ["simF11"]
          and "simF10" not in fl["step2_not_eligible"], (fl["step2_not_eligible"], fl["background_missing"]))
    check("F retention rule: 8/12 in one AV contrast excluded, 9/12 kept",
          list(fl["step4_participants_low_retention"]) == ["simF01"]
          and resF["completeness"]["simF02"]["retention"]["H_vs_E@normal"] == "9/12",
          (fl["step4_participants_low_retention"], resF["completeness"]["simF02"]["retention"]))
    n_blank = sum(1 for r in sessF["simF07"] if r["block"] in SCORED_BLOCKS)
    check("F screens dropped = 4 + 3 + 3 + 4 + 3 (focus_hidden only; blur kept); simF07's scored blank fields counted",
          fl["step4_screens_dropped_focus"] == 17 and fl["step4_focus_field_blank"] == n_blank,
          (fl["step4_screens_dropped_focus"], fl["step4_focus_field_blank"], n_blank))
    check("F MC-only drop (3/6 silent), VC-only drop (3 pairs with both takes), P2b-only drop (8/12)",
          list(fl["step4_dropped_from_MC_only"]) == ["simF03"] and list(fl["step4_dropped_from_voice_check_only"]) == ["simF06"]
          and list(fl["step4_dropped_from_P2b_only"]) == ["simF05"],
          (fl["step4_dropped_from_MC_only"], fl["step4_dropped_from_voice_check_only"], fl["step4_dropped_from_P2b_only"]))
    SA = resF["samples"]["all"]
    check("F samples: all = 12 - 2 ineligible - 1 low retention = 9; simF03 out of MC only; simF06 out of VC only; simF05 out of P2b only",
          SA["n_participants"] == 9 and SA["hypotheses"]["P1"]["participants"]["n"] == 9
          and SA["manipulation_check"]["participants"]["n"] == 8 and SA["voice_check"]["strong_minus_normal"]["n"] == 8
          and SA["optional"]["P2b"]["participants"]["n"] == 8,
          (SA["n_participants"], SA["hypotheses"]["P1"]["participants"]["n"], SA["manipulation_check"]["participants"]["n"],
           SA["voice_check"]["strong_minus_normal"]["n"], SA["optional"]["P2b"]["participants"]["n"]))
    resS, _ = analyse(rowsF, imap_f, 300, exclF, background=bgl, descriptives=False, focus_field=FOCUS_SENS)
    check("F focus_lost sensitivity run: simF04 (5 blurred H_vs_R@strong) now excluded; drops 17 + 5",
          "simF04" in resS["flow"]["step4_participants_low_retention"] and resS["flow"]["step4_screens_dropped_focus"] == 22,
          (list(resS["flow"]["step4_participants_low_retention"]), resS["flow"]["step4_screens_dropped_focus"]))
    n_f02 = len([1 for x in (parse_pair(r, imap_f) for r in rowsF
                             if r["participant_code"] == "simF02" and screen_kept(r, FOCUS_PRIMARY))
                 if x and x["contrast"] == "H_vs_E@normal"])
    check("F long CSV carries in_primary; simF02 keeps 9 H_vs_E@normal screens (P3 pairs by voice pair over those)",
          bool(longF) and "in_primary" in longF[0] and n_f02 == 9, (len(longF), n_f02))
    import hashlib
    fz_dir = root / "F_freeze"
    fz_dir.mkdir(parents=True, exist_ok=True)
    fa, fb = fz_dir / "a.txt", fz_dir / "b.txt"
    fa.write_text("alpha"), fb.write_text("beta")
    fz = fz_dir / "PLAN_FREEZE.sha256"
    fz.write_text("".join("%s  %s\n" % (hashlib.sha256(f.read_bytes()).hexdigest(), f) for f in (fa, fb)))
    ok1 = freeze_check(fz)
    fb.write_text("beta2")
    ok2 = freeze_check(fz)
    check("F freeze check: all OK before, FAILED on the changed file after, absent file reported as not frozen",
          ok1["all_ok"] is True and ok2["all_ok"] is False and [l["status"] for l in ok2["lines"]] == ["OK", "FAILED"]
          and freeze_check(fz_dir / "none.sha256")["exists"] is False, (ok1["all_ok"], ok2["lines"]))
    check("F P2 sensitivity cells file read (8 clean strong cells) and the subset reported on the sample",
          len(resF["p2_sensitivity_cells"]) == 8 and "p2_sensitivity" in SA, (resF["p2_sensitivity_cells"], list(SA)[:8]))

    # G. the FINAL design: 81-screen sessions from the three links (cn / en / jp listeners), Exp 2 (pair_ml) included
    items_g, src_g = plan_for_sim_v6()
    imap_g = {it["file"]: it for it in items_g}
    want = {"single": 6, "practice": 2, "pair": 36, "pair_catch": 1, EXP2_BLOCK: 18, "pair_mute": 6, "voice_only": 12}
    served_ok = all(len(serve_v6(items_g, n)) == 81 and dict(collections.Counter(x["block"] for x in serve_v6(items_g, n))) == want
                    for n in range(12))
    check("G every rotation slot serves 81 screens: 6 singles, 2 practice, 36 pairs + 1 catch, 18 pair_ml, 6 silent, "
          "12 voice-only [plan source: %s]" % src_g, served_ok,
          dict(collections.Counter(x["block"] for x in serve_v6(items_g, 0))))
    order = [SERVE_ORDER.index(x["block"]) if x["block"] != "pair_catch" else SERVE_ORDER.index("pair")
             for x in serve_v6(items_g, 3)]
    check("G block order single < practice < pairs+catch < pair_ml < pair_mute < voice_only", order == sorted(order), order[:5])
    l_left = [sum(1 for x in serve_v6(items_g, n) if x["block"] == EXP2_BLOCK and x["voice_arm"].startswith(kind + "@")
                  and side_of_A(x) == "left") for n in range(2) for kind in EXP2_KINDS]
    check("G Exp 2 sides: L on the left in 4 or 5 of 9 per contrast; two consecutive slots mirror each other",
          all(v in (4, 5) for v in l_left) and l_left[0] + l_left[2] == 9 and l_left[1] + l_left[3] == 9, l_left)
    flips = []
    for n in range(6):
        sides = collections.defaultdict(dict)
        for x in serve_v6(items_g, n):
            if x["block"] == EXP2_BLOCK:
                sides[x["base_cell"]][x["voice_arm"].split("@")[0]] = side_of_A(x)
        flips.append(sum(1 for d in sides.values() if len(d) == 2 and d["L_vs_E"] != d["L_vs_Lrev"]))
    check("G Exp 2: L on the other side in L_vs_Lrev than in L_vs_E for all 9 cells at every slot", flips == [9] * 6, flips)
    partsG = ["simG%02d" % i for i in range(36)]
    grpG = {pc: LISTENER_GROUPS[i % 3] for i, pc in enumerate(partsG)}
    effG = dict(eff, **{VOICE_ARM: 1.0, "L_vs_E": 0.5, "L_vs_Lrev": 0.3, "native:L_vs_E": 0.6})
    sessG = simulate_rows(items_g, partsG, effG, seff, side_bias=0.3, seed=91, catch_fail=("simG05",), groups=grpG,
                          per_link=True)
    k = 0
    for r in sessG["simG07"]:                                          # 3 of 9 L_vs_E hidden -> 6/9 -> out of Exp 2 only
        if r["voice_arm"].startswith("L_vs_E@") and k < 3:
            r.update(hid)
            k += 1
    bgG = {"simG08": {"native_lang": "other", "hearing": "normal", "vision": "normal"},
           "simG10": {"native_lang": "jp", "hearing": "normal", "vision": "normal"}}     # simG10 enters by /root/lilt_en
    write_sim_csvs(root / "G_three_links", sessG, groups=grpG, background=bgG)
    rowsG, exclG = load(root / "G_three_links", set(imap_g))
    bgGl = load_background(root / "G_three_links")
    resG, longG = analyse(rowsG, imap_g, B, exclG, background=bgGl, perm_B=500)
    write_outputs(resG, longG, root, "_G_three_links")
    SG, flG = resG["samples"]["catch_pass"], resG["flow"]
    compl = [c for p, c in resG["completeness"].items()]
    check("G completeness: 6 singles, 12/12/12, 6 silent, 9 L_vs_E + 9 L_vs_Lrev (simG07: 6 retained), 12 voice-only, 1 catch",
          all(c["single"] == 6 and c["H_vs_E@normal"] == 12 and c["H_vs_R@strong"] == 12 and c[MUTE_CONTRAST] == 6
              and c["L_vs_Lrev"] == 9 and c["voice_only"] == 12 and c["catch"]["n"] == 1 for c in compl)
          and resG["completeness"]["simG07"]["L_vs_E"] == 6 and resG["completeness"]["simG00"]["L_vs_E"] == 9,
          compl[:1])
    check("G listener groups = link: cn 12 / en 12 / jp 12; simG08 (native other) ineligible; simG10 (native jp on the "
          "en link) stays in group en and is listed in native_lang_differs_from_link",
          collections.Counter(flG["listener_group_of"].values()) == collections.Counter({"cn": 12, "en": 12, "jp": 12})
          and flG["step2_not_eligible"] == {"simG08": "native_lang=other"} and flG["listener_group_of"]["simG10"] == "en"
          and flG["native_lang_differs_from_link"] == ["simG10"],
          (collections.Counter(flG["listener_group_of"].values()), flG["step2_not_eligible"],
           flG["native_lang_differs_from_link"]))
    check("G flow: simG05 catch failed, simG07 out of Exp 2 only; primary N 34, Exp 2 primary N 33; P1 n 34, P4 n 33",
          flG["step3_catch_failed_or_missing"] == ["simG05"] and list(flG["step4_dropped_from_exp2_only"]) == ["simG07"]
          and flG["primary_sample_N"] == 34 and flG["exp2_primary_N"] == 33
          and SG["hypotheses"]["P1"]["participants"]["n"] == 34 and SG["exp2"]["hypotheses"]["P4"]["participants"]["n"] == 33,
          (flG["step3_catch_failed_or_missing"], flG["step4_dropped_from_exp2_only"], flG["primary_sample_N"],
           flG["exp2_primary_N"]))
    truthG = {"P4": 0.5 + 0.6 / 3, "P5": 0.3}
    for kk in EXP2_FAMILY:
        d = SG["exp2"]["hypotheses"][kk]["participants"]
        check("G %s recovered (true %.2f; CI > 0)" % (kk, truthG[kk]),
              d["boot95"][0] > 0 and abs(d["mean"] - truthG[kk]) < 0.3, (d["mean"], d["boot95"]))
    check("G Holm families: Exp 1 = P1, P2, P3; Exp 2 = P4, P5",
          sorted(SG["family"]["holm_two_sided"]) == ["P1", "P2", "P3"]
          and sorted(SG["exp2"]["family"]["holm_two_sided"]) == ["P4", "P5"], (SG["family"], SG["exp2"]["family"]))
    check("G Exp 1 P1-P3 still recovered with Exp 2 in the session",
          all(abs(SG["hypotheses"][kk]["participants"]["mean"] - truth[kk]) < 0.3 for kk in FAMILY),
          {kk: SG["hypotheses"][kk]["participants"]["mean"] for kk in FAMILY})
    resG_no = analyse([r for r in rowsG if r.get("block") != EXP2_BLOCK], imap_g, 300, exclG, background=bgGl,
                      descriptives=False)[0]["samples"]["catch_pass"]
    check("G Exp 2 rows do not enter P1-P3 or the MC (identical means with the pair_ml rows removed)",
          all(resG_no["hypotheses"][kk]["participants"]["mean"] == SG["hypotheses"][kk]["participants"]["mean"] for kk in FAMILY)
          and resG_no["manipulation_check"]["participants"]["mean"] == SG["manipulation_check"]["participants"]["mean"],
          {kk: (resG_no["hypotheses"][kk]["participants"]["mean"], SG["hypotheses"][kk]["participants"]["mean"]) for kk in FAMILY})
    psl = SG["exp2"]["per_stim_language"]
    check("G per stimulus language: cn / en / jp, 33 x 3 trials per contrast",
          sorted(psl) == ["cn", "en", "jp"] and all(psl[l][kind]["n_trials"] == 99 for l in psl for kind in EXP2_KINDS),
          {l: {kind: psl[l][kind]["n_trials"] for kind in EXP2_KINDS} for l in psl})
    nat = SG["exp2"]["native_stimulus_exploratory"]

    def within_3se(q, truth_v):                                        # the estimate's own SE; across 12 seeds the
        return abs(q["mean"] - truth_v) < 3.0 * q["sd"] / math.sqrt(q["n"])   # means centre on the truth (exp2/diag_native_sim.log)
    check("G exploratory native-stimulus effect: L_vs_E within 3 SE of +0.6 (CI > 0), L_vs_Lrev within 3 SE of 0; "
          "3 x 3 group x language table",
          nat["L_vs_E"]["native_minus_nonnative"]["boot95"][0] > 0
          and within_3se(nat["L_vs_E"]["native_minus_nonnative"], 0.6)
          and within_3se(nat["L_vs_Lrev"]["native_minus_nonnative"], 0.0)
          and all(len(nat["L_vs_E"]["group_x_stim_language"][g]) == 3 for g in LISTENER_GROUPS),
          (nat["L_vs_E"]["native_minus_nonnative"], nat["L_vs_Lrev"]["native_minus_nonnative"]["mean"]))
    vcG = SG.get("voice_check") or {}
    check("G voice check: strong-minus-normal intensity per emotion reported for every emotion (fear included)",
          bool(vcG) and all(d.get("intensity_strong_minus_normal") is not None for d in vcG["accuracy_by_emotion"].values()),
          {e: d.get("intensity_strong_minus_normal") for e, d in (vcG.get("accuracy_by_emotion") or {}).items()})
    blg = SG["by_listener_group"]
    check("G per-group secondary tables (P1-P3, MC, P4-P5) for cn / en / jp; group differences exploratory with p_perm",
          sorted(blg) == ["cn", "en", "jp"] and all(sorted(blg[g]["P1-P3"]) == ["P1", "P2", "P3"]
                                                   and sorted(blg[g]["P4-P5"]) == ["P4", "P5"] and blg[g]["MC"] for g in blg)
          and sorted(SG["group_differences_exploratory"]) == ["MC", "P1", "P2", "P3", "P4", "P5"]
          and all(0 < d["p_perm"] <= 1 for d in SG["group_differences_exploratory"].values()),
          ({g: blg[g]["n_participants"] for g in blg}, {kk: d.get("p_perm") for kk, d in SG["group_differences_exploratory"].items()}))
    cleanG = SG["exp2"].get("P5_sensitivity") or {}
    check("G P5 sensitivity subset read from exp2/exp2_cell_rule.json (4 clean parity cells) and reported",
          len(resG["exp2_lrev_clean_cells"]) == 4 and cleanG.get("n_cells_seen") == 4 and cleanG["participants"]["n"] == 33,
          (resG["exp2_lrev_clean_cells"], resG["exp2_lrev_clean_note"], cleanG.get("n_cells_seen")))
    adj = SG["exp2"]["side_bias_adjusted"]
    check("G side-bias-adjusted P4 / P5 close to the unadjusted means (sides near balance)",
          all(abs(adj[kk]["mean"] - SG["exp2"]["hypotheses"][kk]["participants"]["mean"]) < 0.2 for kk in EXP2_FAMILY),
          {kk: (adj[kk]["mean"], SG["exp2"]["hypotheses"][kk]["participants"]["mean"]) for kk in EXP2_FAMILY})
    check("G long CSV: 60 scored pairs per participant (36 + 18 + 6) minus simG07's 3 hidden; exp / stim_language / "
          "listener_group / native_stimulus columns; no balance warning",
          len(longG) == 36 * 60 - 3 and all(kk in longG[0] for kk in ("exp", "stim_language", "listener_group", "native_stimulus"))
          and not resG["realised_balance"]["warnings"],
          (len(longG), resG["realised_balance"]["warnings"][:2]))
    pg = resG["realised_balance"]["per_listener_group"]
    check("G per-link rotation: per-group realised balance for cn / en / jp without warnings (H and L shares 0.5 in cn, 12 "
          "sequential participants; en / jp each lose one participant to exclusion rows only)",
          sorted(pg) == ["cn", "en", "jp"] and not pg["cn"]["warnings"]
          and all(v == 0.5 for v in pg["cn"]["H_left_share"].values()),
          {g: (pg[g]["warnings"][:1], pg[g]["vector_multiplicity"]) for g in pg})
    aG = resG["realised_balance"]["rotation_audit"]
    check("G rotation audit without stamps: lists reconstructed from the served files, 2 per offset on each link (P 6 from "
          "the manifest), no per-link warning under the least-filled allowance",
          aG["offset_source"] == {"reconstructed_from_served_files": 36} and aG["period"] == 6
          and sorted((e["link"], e["counts_per_offset"]) for e in aG["analysed_participants"])
          == [(g, [2] * 6) for g in ("cn", "en", "jp")] and not aG["warnings"]
          and not any(pg[g]["warnings"] for g in pg),
          (aG["offset_source"], aG["period_source"], [(e["link"], e["counts_per_offset"]) for e in aG["analysed_participants"]],
           {g: pg[g]["warnings"][:1] for g in pg}))
    sessGs = simulate_rows(items_g, ["simGA%d" % i for i in range(3)] + ["simGR%d" % i for i in range(3)], seed=5,
                           always={"simGA0": "H3", "simGA1": "H3", "simGA2": "H3", "simGR0": "R3", "simGR1": "R3", "simGR2": "R3"},
                           groups={"simGA0": "cn", "simGA1": "en", "simGA2": "jp", "simGR0": "cn", "simGR1": "en", "simGR2": "jp"},
                           per_link=True)
    rGs = analyse(_strrows(sessGs), imap_g, 200, {}, descriptives=False)[0]["samples"]["all"]["exp2"]
    px2s = [x for x in (parse_pair(r, imap_g) for r in _strrows(sessGs)) if x and x["exp"] == 2]
    dvA = {kind: exp2_dv(px2s, kind) for kind in EXP2_KINDS}
    check("G Exp 2 signs: always-L raters score exactly +3 on P4 and P5; always-R3 raters |P4|, |P5| <= 1 and exactly 0 "
          "after the side-bias adjustment",
          all(dvA[kind]["simGA%d" % i] == 3 for kind in EXP2_KINDS for i in range(3))
          and all(abs(dvA[kind]["simGR%d" % i]) <= 1 + 1e-9 for kind in EXP2_KINDS for i in range(3))
          and rGs["side_bias_adjusted"]["P4"]["n"] == 6,
          ({kind: dvA[kind] for kind in EXP2_KINDS}, rGs["side_bias_adjusted"]["P4"]))
    biasGs = unit_means([x for x in (parse_pair(r, imap_g) for r in _strrows(sessGs)) if x and x["exp"] == 1
                         and x["contrast"] in AV_CONTRASTS], "p", key="v")
    adjR = [dict(x, s=(x["s"] - biasGs[x["p"]]) if x["H_side"] == "right" else (x["s"] + biasGs[x["p"]]))
            for x in px2s if x["p"].startswith("simGR")]
    check("G side-bias adjustment removes an always-right responder exactly (score 0)",
          all(v == 0 for v in exp2_dv(adjR, "L_vs_E").values()) and len(exp2_dv(adjR, "L_vs_E")) == 3, exp2_dv(adjR, "L_vs_E"))

    # C. null calibration: coverage of the participant-bootstrap CI when nothing differs (with a side bias), on the
    # FINAL 81-screen sessions so P4 / P5 are covered too
    cover = collections.Counter()
    for rep in range(coverage_reps):
        grpC = {("n%03d" % i): LISTENER_GROUPS[i % 3] for i in range(24)}
        sess = simulate_rows(items_g, ["n%03d" % i for i in range(24)], {}, {}, side_bias=0.3, seed=1000 + rep, groups=grpC,
                             per_link=True)
        S = analyse(_strrows(sess), imap_g, 1000, {}, descriptives=False)[0]["samples"]["all"]
        for c in PAIR_CONTRASTS:
            lo, hi = S["pairs"][c]["participants"]["boot95"]
            cover[c] += int(lo <= 0 <= hi)
        for k in FAMILY:
            lo, hi = S["hypotheses"][k]["participants"]["boot95"]
            cover[k] += int(lo <= 0 <= hi)
        for k in EXP2_FAMILY:
            lo, hi = S["exp2"]["hypotheses"][k]["participants"]["boot95"]
            cover[k] += int(lo <= 0 <= hi)
    for k, v in sorted(cover.items()):
        rate = v / float(coverage_reps)
        check("C null coverage %s = %.2f (want 0.85..1.00; nominal 0.95)" % (k, rate), 0.85 <= rate <= 1.0, rate)

    print("analyse_lilt_v6 self-test:")
    for name, ok, detail in checks:
        print("  %s  %s%s" % ("PASS" if ok else "FAIL", name, "" if ok else "   -> %s" % (detail,)))
    print("%d/%d checks passed; simulated data and results under %s" % (len(checks) - len(fails), len(checks), root))
    return len(fails)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--manifest", default="", help="the v6 manifest (default: the live link manifest if it is a v6 build)")
    ap.add_argument("--out", default=str(OUTDIR))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--sided", choices=("two", "one"), default="two")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--coverage-reps", type=int, default=60)
    ap.add_argument("--sessions-dir", default="auto",
                    help="<webapp data>/sessions for the descriptive rotation audit (lilt_rotation stamps); 'auto' = "
                         "<data-dir>/../../sessions when it exists; 'none' = reconstruct lists from the served files")
    a = ap.parse_args(argv)
    if a.selftest:
        return 1 if selftest(a.out, coverage_reps=a.coverage_reps) else 0
    mp = Path(a.manifest) if a.manifest else LIVE_MANIFEST
    if not mp.is_file():
        print("REFUSED: manifest not found: %s" % mp)
        return 2
    m = json.load(io.open(mp, encoding="utf-8"))
    if m.get("builder") != "build_lilt_v6.py":
        print("REFUSED: %s is not a build_lilt_v6.py manifest (builder %s)" % (mp, m.get("builder")))
        return 2
    if m.get("provisional") or m.get("partial") or (m.get("selection") or {}).get("placeholder"):
        print("WARNING: the manifest is PROVISIONAL / PARTIAL / PLACEHOLDER -- these are not study data")
    sdir = a.sessions_dir
    if sdir == "auto":
        cand = Path(a.data_dir).resolve().parents[1] / "sessions" if len(Path(a.data_dir).resolve().parents) > 1 else None
        sdir = str(cand) if cand is not None and cand.is_dir() else None
    elif sdir.lower() == "none":
        sdir = None
    run(a.data_dir, item_map(m), a.boot, a.out, sided=a.sided, sessions_dir=sdir)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

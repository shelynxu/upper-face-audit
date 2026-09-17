# -*- coding: utf-8 -*-
"""build_lilt_v6.py -- the v6 perception study ("does the missing voice-following upper face matter to viewers?") as a
/root/lilt manifest (server.py mode "lilt"). Copied from build_lilt_v2a.py and changed to the v6 design.

Face: Danielle V9 (pool_render.py --look danielle_v9), condition e10m10_d5_h (EmoFace 1:1, native degree, expression
held through the silent pads). Three versions of one RAVDESS voice clip, every one from the SAME wav:
    E   EmoFace                                                                            token before
    H   E + the SAME actor's own brow motion from the SAME recording, retargeted (inner brow -> brow_raiseIn 31/100,
        outer brow -> brow_raiseOut 32/101; brows only, lids stay EmoFace): an oracle upper face that follows the voice
        as the human did                                                                   token h6_human
    R   E + the same human brow track TIME-REVERSED (same amplitude distribution, misaligned with the voice): a
        motion-equated timing control                                                      token h6_human_rev
Voices: RAVDESS female actors, statement 01, repetition 01, AV mp4 audio; 12 voice pairs (4 emotions x 3 actors) at
intensity normal (cell rav<AA>_<emo>_n) and strong (rav<AA>_<emo>_s). Repetition 02 (cells rav<AA>_<emo>_r2_<n|s>) is
the recorded fallback of the selection rules and is used for two fear pairs: rav20_fea_r2 (rep-01 tracking failure)
and rav12_fea_r2 (rep-01 brow ratio < 1; the rep-02 take passes) [stimuli/SELECTION.md, SELECTION_HISTORY.md].
+ 2 practice clips: tier 1 an actor outside the study, tier 2 a study actor in a non-selected emotion, tier 3 a study
actor's strong clip in a chosen emotion that is NOT that actor's own study emotion -- never a study cell
(SELECTION.md practice rule, amended 14:35; the current practice = rav22_ang_s tier 1 + rav10_hap_s tier 3).
The selection is read from pool_set_v6.json (role / arms fields written by v6_face.py select). Until that selection
exists, only --provisional builds run, on a PLACEHOLDER selection that is marked as such.

v6 FINAL DESIGN (her decisions 2026-09-14 ~17:00; identical for /root/lilt_cn, _jp, _en): 81 screens per participant
  build_lilt_v6.py --singles 6 --ml --voice-check          (--hr-normal stays OFF)
  single 6 | practice 2 | pair 36 + pair_catch 1 | pair_ml 18 | pair_mute 6 | voice_only 12  = 81
  --singles 6   a priori: the singles are the 6 voice pairs of the SILENT subset (mute_rule below): every study emotion
                keeps singles (hap 2, ang 1, fea 2, dis 1), 2 per actor slot, E | H | R rotated 2/2/2 per participant.
  --ml          Experiment 2, block "pair_ml", served AFTER the English pairs and BEFORE the silent pairs (needs
                patches/server_v6.patch as of 17:36, which orders pair_ml; the unpatched server drops the block): cn / en / jp,
                3 cells each (emotions ang, hap, sad in every language: parity) x 2 contrasts L_vs_E@<lang> and
                L_vs_Lrev@<lang>. L = r5b_lilt_nonod-g2 (hand-built LiltFace), Lrev = r5b_lilt_nonod_rev-g2, E = before,
                all e10m10_d5_h, from the ALREADY RENDERED pool/mp4 clips (no new renders). Cell rule computed at build
                time from pool/post arrays: a cell qualifies only if clip(L) - clip(E), L/R averaged, in the cue-bank
                speech span, has p95 - p5 >= 2.0 face-crop px on at least one brow channel (inner 31/100, outer 32/101;
                px per rig unit from gains.json = K x avatar IOD x 0.7778 crop scale: 8.14 / 8.03). The measured table is
                written into the manifest (ml.table) and design_table.md. Same crops / encode / neutral names / pick_group
                left-right rotation as the English pairs; L on the other side in L_vs_Lrev than in L_vs_E of the same cell.
  --voice-check audio-only voice check, served last (unchanged).

Per participant, BASE design (57 screens, block order as served by server.py mode "lilt"):
  single      12  each voice pair once, at STRONG (the only level with all three arms); E | H | R rotated across
                  participants (pick_group of 3, 4/4/4 per person): 7-choice emotion + vividness of the movement (CSV
                  column `intensity`) + naturalness. First exposure.
  practice     2  H vs E on the two practice clips (unscored; left/right rotated)
  pair        36  one shared voice, two face crops side by side, CCR L3..S0..R3 "which face's movement fits the voice":
                    H_vs_E@normal 12 | H_vs_E@strong 12 | H_vs_R@strong 12     (left/right = pick_group of 2, 6/6)
  pair_catch   1  H moving vs the same clip frozen ("which face is moving?"), mixed into the pairs; default source = the
                  first practice clip (no scored clip gets an extra exposure, no extra render)
  pair_mute    6  H vs E at strong WITHOUT sound for 6 voice pairs, "which face's movement is more lively" (served
                  last; visibility / manipulation check). A priori subset: within each emotion the actors sorted by id
                  take slots k = 0,1,2 and the pair is silent when (emotion index + k) is even -> 2,1,2,1 per emotion
                  and 2 per actor slot.
Renders needed: E/H at normal 24 + E/H/R at strong 36 + practice E/H 4 = 64
(pool/mp4/<cell>__<token>__e10m10_d5_h__danielle_v9.mp4).

OPTIONS (her decision pending; both default OFF, nothing depends on the answer):
  --hr-normal    + pair block H_vs_R@normal, 12 screens (groups pair4_Rn, H on the side of H_vs_R@strong): 12 more R
                 renders at the normal voices -> 76 renders; hypothesis P2b (secondary, outside the Holm family).
  --voice-check  + block voice_only, 12 screens served LAST: normal + strong take of 6 voice pairs (stimuli/
                 voice_check_wavs.txt, else the silent-pair subset), audio only under a black 1120x630 frame at -23 LUFS
                 (two-pass loudnorm), 7-choice emotion + emotion intensity 1-7; every participant hears all 12.
                 server.py and app.js need patches/server_v6.patch + app_js_v6.patch (and a guarded restart) first.
  --force-placeholder (with --provisional only): flow-test build on the placeholder selection while the selection in
                 pool_set_v6.json is incomplete.

Left/right bookkeeping [code, checked by rotation_report()]: server.py sorts ALL pick groups by name and serves
arms[(i + n_prior) % n_arms], arms sorted by media path. File names are NEUTRAL (v6_<block>_<10-hex code>_<k>.mp4, the
code = sha1 of block + contrast + cell, k = 1 H on the right / 2 H on the left), so the arm order inside a group is
fixed by k alone. The pair groups are named pair1_En_<jj>_<vp>, pair2_Es_<jj>_<vp>, pair3_Rs_<jj+1 mod 12>_<vp>, so
that within one participant and one voice pair H sits on the SAME side at normal and at strong (side bias cancels in
P3) and on the OTHER side in H_vs_R.

Media conventions are v4's, by import from build_lilt_icassp.py: sbs / sbs_mute / still_of / crop_single (face crops at
1:1 render pixels on the 1120x630 stage; audio = the pool_render mux, loudnorm -23 LUFS, the same wav on both sides).
NOTE [code]: that mux loudness-normalises every clip, so the strong and normal RAVDESS takes play at about the same
integrated loudness; the level cue of vocal intensity is removed, spectral / F0 / duration cues remain. The build
records LUFS per voice pair and level (media_qc.lufs_normal_vs_strong).

Publishing never takes the link down: media are built in --stage-dir, the new files are moved into <out>/media
(names must not collide with the files the current manifest serves), the manifest is replaced atomically, and only
then are the files the new manifest no longer references moved to <out>/_old/<ts>/media.

RUN (WSL, from /mnt/d/SPJ/pipeline/prosody_ladder, CPU-light):
  nice -n 19 taskset -c 0,1,2 /home/shelyn/miniconda/envs/GPTSoVits/bin/python build_lilt_v6.py \
      --out _explore/voicedose/study_v6/study/lilt_link_v6_provisional --provisional
  ... build_lilt_v6.py --check [--newer-than REF]                        # selection + plan + sources, writes nothing
  ... build_lilt_v6.py --out /mnt/d/SPJ/pipeline/pilot/lilt_link --require-complete --layer-label <label> \
      --newer-than <the file the H/R arrays were composed after, e.g. the v6 READY json>
Live-link guards: --require-complete and --newer-than are mandatory, the selection must come from pool_set_v6.json
(no placeholder); every render must be newer than its rig; --replace-same-names into the live link additionally needs
--allow-live-replace.
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPJ = HERE.parents[1]
POOL = HERE / "pool"
STUDY = HERE / "_explore" / "voicedose" / "study_v6"
LIVE = SPJ / "pipeline" / "pilot" / "lilt_link"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import build_lilt_icassp as B                                   # noqa: E402  (v4 media helpers, imported not copied)

ARMS = ("E", "H", "R")
TOKENS = {"E": "before", "H": "h6_human", "R": "h6_human_rev"}
COND = "e10m10_d5_h"
LOOK = "danielle_v9"
LEVELS = {"n": "normal", "s": "strong"}
LEVEL_CODE = {"normal": "n", "strong": "s", "n": "n", "s": "s", "01": "n", "02": "s", "1": "n", "2": "s"}
SINGLE_LEVEL = "s"
CONTRASTS = ("H_vs_E@normal", "H_vs_E@strong", "H_vs_R@strong")
MUTE_CONTRAST = "mute:H_vs_E@strong"
EXPECTED_SERVED = {"single": 12, "practice": 2, "pair": 36, "pair_catch": 1, "pair_mute": 6}
N_VOICE_PAIRS, N_PRACTICE, N_MUTE, N_REQUIRED = 12, 2, 6, 64
# ---- the two OPTIONAL additions (her decision pending; both switchable, default OFF) ------------------------------
# --hr-normal    a fourth AV pair block H_vs_R@normal (12 screens, H on the same side as in H_vs_R@strong, i.e. the
#                other side than in H_vs_E of the same voice pair). Needs 12 more R renders (R at the normal voices):
#                64 -> 76 renders.
# --voice-check  block "voice_only" / presentation "voice_only", served LAST (after the silent pairs): the normal and
#                strong take of N_VOICE_CHECK voice pairs, audio only over a black 1120x630 frame, loudnorm -23 LUFS
#                (two-pass), 7-choice emotion + emotion INTENSITY 1-7 (CSV `intensity`). Tests the premise of P3 that
#                Mandarin listeners hear the strong take as stronger. Every participant hears all 12 (pick_group of 1).
#                Which pairs: stimuli/voice_check_wavs.txt (one rav<AA>_<emo> per line) when it exists, else the
#                silent-pair subset (mute_rule: every emotion present, 2 per actor slot).
#                NOTE: server.py orders single / practice / pair+pair_catch / pair_mute only; the voice_only block needs
#                patches/server_v6.patch (+ a guarded restart) and patches/app_js_v6.patch (intensity label, no
#                naturalness) before it can be served.
HR_NORMAL_CONTRAST = "H_vs_R@normal"
VOICE_CHECK_ARM = "voice_check"
N_VOICE_CHECK = 6
VOICE_CHECK_FILE = STUDY / "stimuli" / "voice_check_wavs.txt"
VOICE_LUFS, VOICE_TP, VOICE_LRA = -23.0, -1.5, 11.0
VOICE_LUFS_TOL = 0.5               # LU: a two-pass loudnorm lands within this of VOICE_LUFS (gate on the voice-only clips)
VOICE_LUFS_FIX_TOL = 0.3           # LU: voice_black re-muxes once with a corrected gain when the clip is further off
ALL_CONTRASTS = CONTRASTS + (HR_NORMAL_CONTRAST,)
# ---- v6 FINAL (her decisions 2026-09-14): --singles 6 --ml --voice-check; --hr-normal stays OFF ------------------------
SINGLES_CHOICES = (12, 6)
SINGLES_RULE = {
    12: "all 12 voice pairs at the strong voice (base design)",
    6: ("a priori (fixed 2026-09-14 before any participant, depends on nothing measured): the 6 voice pairs of the SILENT "
        "subset (mute_rule: within each emotion the actors sorted by id take slots k = 0,1,2; a pair is in the subset when "
        "emotion index + k is even) -> every study emotion keeps singles (hap 2, ang 1, fea 2, dis 1), 2 per actor slot; "
        "E | H | R rotated 2/2/2 per participant; the other 6 voice pairs get no single screen"),
}
ML_BLOCK = "pair_ml"
ML_TOKENS = {"E": "before", "L": "r5b_lilt_nonod-g2", "Lrev": "r5b_lilt_nonod_rev-g2"}
ML_CANDIDATES = list(B.UTTERANCES)          # the fixed v4 study cells (build_lilt_icassp.UTTERANCES) = candidate order
ML_LANGS = ("cn", "en", "jp")
ML_EMOTIONS = ("ang", "hap", "sad")         # parity: the same three emotions in every stimulus language
ML_CONTRASTS = ("L_vs_E", "L_vs_Lrev")
N_ML_CELLS = len(ML_LANGS) * len(ML_EMOTIONS)
ML_MIN_PX = 2.0                             # face-crop px, p95-p5 of clip(L) - clip(E) in speech, on >= 1 brow channel
ML_CHANNELS = {"brow_inner": (31, 100), "brow_outer": (32, 101)}
GAINS_JSON = HERE / "_explore" / "voicedose" / "study_v2a" / "model" / "gains.json"
CROP_SCALE = 0.7778                         # face-crop pair px / full-frame px (v2a_face.CROP_SCALE)
ML_PX_FALLBACK = {"brow_inner": 8.14, "brow_outer": 8.03}
HUMAN_BROW_REF_RIG = {"brow_inner": 0.48, "brow_outer": 0.35}   # English RAVDESS human reference [measured, main session]
RENDER_SETTLE_S = 120                       # provisional builds use a real English render only when older than this
ML_RULE = ("Experiment 2 cells (a priori): for each stimulus language (cn, en, jp) and each parity emotion (ang, hap, sad) "
           "the first cell of build_lilt_icassp.UTTERANCES with that language and label that qualifies; a cell qualifies "
           "only if clip(L) - clip(E) (L = r5b_lilt_nonod-g2, E = before, e10m10_d5_h, pool/post arrays, L/R columns "
           "averaged, frames inside the cue-bank speech span) has p95 - p5 >= %.1f face-crop px on at least one brow "
           "channel (inner 31/100, outer 32/101; px per rig unit = gains.json K x avatar IOD x %.4f). Computed at build "
           "time; no cell is chosen by eye." % (ML_MIN_PX, CROP_SCALE))


def expected_served(hr_normal=False, voice_check=False, ml=False, singles=N_VOICE_PAIRS):
    """Screens per block per participant for a given set of options (keys = manifest["options"])."""
    d = dict(EXPECTED_SERVED)
    d["single"] = int(singles)
    if hr_normal:
        d["pair"] += N_VOICE_PAIRS
    if ml:
        d[ML_BLOCK] = N_ML_CELLS * len(ML_CONTRASTS)
    if voice_check:
        d["voice_only"] = 2 * N_VOICE_CHECK
    return d


def av_contrasts(hr_normal=False):
    return tuple(CONTRASTS) + ((HR_NORMAL_CONTRAST,) if hr_normal else ())
# RAVDESS emotion names (pool_set_v6 labels) -> the response codes of config.json trials.response_options
EMO_RESPONSE = {"hap": "hap", "ang": "ang", "sad": "sad", "sur": "sur", "fea": "fear", "fear": "fear", "dis": "dis",
                "neu": "neu"}
EMO_ORDER = ("hap", "sad", "ang", "fea", "dis", "sur")          # RAVDESS codes 03..08
CELL_RE = re.compile(r"^rav(\d{2})_([a-z]+)(?:_r(\d))?_(n|s)$")   # rav<AA>_<emo>[_r<rep>]_<n|s>; _r2 = repetition 02
STUDY_ROLES = {"study", "main", "scored", "pair", "pairs", "voice_pair", "selected", "test_pair"}
PLACEHOLDER_EMOTIONS = ("ang", "hap", "fea", "sur")               # PROVISIONAL ONLY: not the a-priori choice
PLACEHOLDER_ACTORS = ("rav02", "rav04", "rav06")
PLACEHOLDER_PRACTICE = ("rav08_sad_s", "rav10_dis_s")
LUFS_WINDOW = (-26.0, -19.0)       # gross window; the pool_render mux realises -20.7 .. -23.0 LUFS (measured 2026-09-14)
ARM_LUFS_SPREAD = 0.3              # max LU between the arms of one cell (same wav; AAC re-encode moves it <= 0.1)
ARM_DUR_TOL = 0.034                # s, arms of one cell (the pair mux uses -shortest)
NAME_LEAK = re.compile(r"h6|human|rev|before|rav\d|normal|strong|emoface|lilt|danielle|__", re.I)
HYPOTHESES = {
    "P1": "viewers judge H's movement to fit the voice better than E's: score_H over H_vs_E pooled over both voice "
          "levels > 0 (the deaf upper face matters)",
    "P2": "H is preferred over R: score_H in H_vs_R@strong > 0 (alignment with the voice matters, not motion presence)",
    "P3": "H's advantage over E is larger for strong than for normal voices: within participant, mean score_H "
          "H_vs_E@strong minus H_vs_E@normal > 0 (the upper face should follow vocal intensity; human reference x1.33)",
    "MC": "manipulation check: silent H_vs_E@strong, 'which face's movement is more lively', score_H > 0 (the brow "
          "layer is visible); not in the Holm family",
}
OPTIONAL_HYPOTHESES = {                         # added to the manifest only when the block is built; never in the family
    "P2b": "(--hr-normal) H is preferred over R at the NORMAL voice too: score_H in H_vs_R@normal > 0; secondary, not "
           "in the Holm family; with P2 it gives (H-R)@strong - (H-R)@normal as a motion-equated companion of P3",
    "MC2": "(--voice-check) manipulation check of the P3 premise: audio only, within participant the mean emotion "
           "intensity rating of the strong take minus the normal take of the same voice pair > 0; emotion accuracy "
           "(7-AFC) by level is reported; not in the Holm family",
}
ML_HYPOTHESES = {                               # Experiment 2 (--ml); Holm within Experiment 2 only
    "P4": "(Exp 2) viewers judge LiltFace L's movement to fit the voice better than EmoFace E's: score_L over "
          "L_vs_E pooled over the three stimulus languages > 0",
    "P5": "(Exp 2) L is preferred over Lrev (the same LiltFace driven by the time-reversed prosody): score_L in L_vs_Lrev "
          "pooled over stimulus languages > 0 (alignment with the voice, not motion presence)",
    "X2": "(Exp 2, exploratory) listener group (cn / jp / en link) x stimulus language (native vs non-native stimulus); "
          "per-language descriptives; not in any Holm family",
}


def norm_path(p):
    """Accept D:/X on WSL (-> /mnt/d/X) so one command line works from either shell."""
    s = str(p)
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if m and os.name != "nt":
        return Path("/mnt/%s/%s" % (m.group(1).lower(), m.group(2).replace("\\", "/")))
    return Path(s)


def vp_of(cell):
    """rav02_hap_s -> rav02_hap (the voice pair); anything else unchanged."""
    return cell.rsplit("_", 1)[0] if CELL_RE.match(cell) else cell


def level_of(cell):
    m = CELL_RE.match(cell)
    return LEVELS[m.group(4)] if m else ""


def actor_of(cell):
    m = CELL_RE.match(cell)
    return "rav%s" % m.group(1) if m else ""


def emo_of(cell):
    m = CELL_RE.match(cell)
    return m.group(2) if m else ""


# ---------------------------------------------------------------------------------------------------- selection
def _pool_set_rows(s):
    items = s.get("items", []) if isinstance(s, dict) else s
    if isinstance(items, dict):
        items = [dict(v, cell=v.get("cell", k)) for k, v in items.items() if isinstance(v, dict)]
    return [r for r in (items or []) if isinstance(r, dict) and r.get("cell")]


def mute_rule(vps):
    """The a-priori silent subset: within each emotion (EMO_ORDER) the voice pairs sorted by actor take slots k; silent
    when (emotion index + k) is even. 4 emotions x 3 actors -> 6 pairs (2,1,2,1 per emotion, 2 per slot). Any other
    shape: every other voice pair in sorted order, N_MUTE of them."""
    by = collections.defaultdict(list)
    for vp in vps:
        by[emo_of(vp + "_n")].append(vp)
    emos = sorted(by, key=lambda e: (EMO_ORDER.index(e) if e in EMO_ORDER else 99, e))
    if len(emos) == 4 and all(len(by[e]) == 3 for e in emos):
        return sorted(vp for ei, e in enumerate(emos) for k, vp in enumerate(sorted(by[e])) if (ei + k) % 2 == 0)
    return sorted(vps)[::2][:N_MUTE]


VOICE_WAV_OVERRIDE = {}                          # cell -> wav from voice_check_wavs.txt (4th column), when given


def voice_check_pairs(vps, mute_pairs, path=None, use_file=True):
    """(--voice-check) The voice pairs whose normal + strong takes are heard audio-only. From `path` (default
    stimuli/voice_check_wavs.txt: one rav<AA>_<emo> or rav<AA>_<emo>_<n|s> per line, '#' comments) when it exists;
    otherwise the silent-pair subset (an a-priori 6 with every emotion present -- the first 6 by name would hold no
    'dis' pair at all). Returns (pairs, source, problems)."""
    p = norm_path(path) if path else VOICE_CHECK_FILE
    probs = []
    if p.is_file() and use_file:
        want, wavs = [], {}
        for ln in io.open(p, encoding="utf-8"):
            ln = ln.split("#", 1)[0].strip()
            if not ln:
                continue
            cols = ln.split()                            # the stimuli lane writes cell<TAB>clip<TAB>level<TAB>wav48
            cell = cols[0]
            want.append(vp_of(cell) if CELL_RE.match(cell) else cell)
            if len(cols) >= 4 and CELL_RE.match(cell):
                wavs[cell] = cols[3]
        want = list(dict.fromkeys(want))
        bad = [w for w in want if w not in vps]
        if bad:
            probs.append("voice_check_wavs.txt names pairs outside the study: %s" % bad)
        if len(want) != N_VOICE_CHECK:
            probs.append("voice_check_wavs.txt lists %d pairs, the option needs %d" % (len(want), N_VOICE_CHECK))
        VOICE_WAV_OVERRIDE.update(wavs)
        return sorted(want), str(p), probs
    if p.is_file():
        probs.append("voice_check_wavs.txt ignored (--force-placeholder): its pairs belong to the real selection")
    return sorted(mute_pairs)[:N_VOICE_CHECK], "silent-pair subset (mute_rule)", probs


def voice_wav_of(sel, cell):
    """The RAVDESS wav of a cell for the audio-only clips: pool_set_v6 items carry wav48 / wav (WSL paths); fall back
    to stimuli/wav/<ravdess_clip>_48k.wav. Returns a Path (may not exist -- the caller checks)."""
    r = sel["items"].get(cell, {}) if sel else {}
    if VOICE_WAV_OVERRIDE.get(cell):
        p = norm_path(VOICE_WAV_OVERRIDE[cell])
        if p.is_file():
            return p
    for k in ("wav48", "wav"):
        if r.get(k):
            p = norm_path(r[k])
            if p.is_file():
                return p
    clip = r.get("ravdess_clip")
    if clip:
        return STUDY / "stimuli" / "wav" / ("%s_48k.wav" % clip)
    return STUDY / "stimuli" / "wav" / ("%s_MISSING.wav" % cell)


def placeholder_selection():
    vps = sorted("%s_%s" % (a, e) for e in PLACEHOLDER_EMOTIONS for a in PLACEHOLDER_ACTORS)
    return {"from": "PLACEHOLDER (no selection in pool_set_v6.json; provisional only)", "placeholder": True,
            "valid": True, "notes": [], "voice_pairs": vps, "practice": list(PLACEHOLDER_PRACTICE),
            "mute_pairs": mute_rule(vps), "catch_cell": PLACEHOLDER_PRACTICE[0], "tokens": dict(TOKENS), "cond": COND,
            "items": {}}


def resolve_selection(pool_set_path=None, mute_over=None, catch_over=None, tok_over=None, cond_over=None):
    """Voice pairs / practice / silent subset / catch cell / tokens. Read from pool_set_v6.json when it carries a
    selection (items with role 'study' | 'practice' [| 'catch'], or arms without a role, or selected: true, or a
    top-level 'selection' dict {voice_pairs, practice, mute_pairs, catch_cell}); otherwise the PLACEHOLDER."""
    p = norm_path(pool_set_path) if pool_set_path else None
    s, notes = None, []
    if p is not None and p.is_file():
        try:
            s = json.load(io.open(p, encoding="utf-8"))
        except Exception as e:                                          # noqa: BLE001
            notes.append("pool set unreadable (%s)" % e)
    elif p is not None:
        notes.append("pool set not found: %s" % p)
    rows = _pool_set_rows(s) if s is not None else []
    by_cell = {r["cell"]: r for r in rows}
    explicit = {}
    if isinstance(s, dict) and isinstance(s.get("selection"), dict):
        explicit = s["selection"]
    study, practice, mute_flag, catch_flag = [], [], [], []
    for r in rows:
        role = str(r.get("role") or r.get("block") or r.get("kind") or "").lower()
        if role == "practice" or r.get("practice") is True:
            practice.append(r["cell"])
        elif role == "catch":
            catch_flag.append(r["cell"])
        elif role in STUDY_ROLES or r.get("selected") is True or (not role and r.get("arms")):
            study.append(r["cell"])
        if r.get("mute") is True or r.get("silent") is True:
            mute_flag.append(vp_of(r["cell"]))
    if explicit:
        ev = [vp_of(c) for c in (explicit.get("voice_pairs") or explicit.get("cells") or [])]
        study = study or [("%s_%s" % (v, lv)) for v in ev for lv in ("n", "s")]
        practice = practice or list(explicit.get("practice") or explicit.get("practice_cells") or [])
        mute_flag = mute_flag or [vp_of(c) for c in (explicit.get("mute_pairs") or explicit.get("silent_pairs") or [])]
        catch_flag = catch_flag or ([explicit["catch_cell"]] if explicit.get("catch_cell") else [])
    levels = collections.defaultdict(set)
    for c in study:
        if CELL_RE.match(c):
            levels[vp_of(c)].add(c.rsplit("_", 1)[1])
    vps = sorted(v for v, ls in levels.items() if ls >= {"n", "s"})
    half = sorted(v for v, ls in levels.items() if not ls >= {"n", "s"})
    if not vps and not practice:
        sel = placeholder_selection()
        sel["items"] = by_cell
        sel["notes"] = notes + ["pool_set_v6.json has %d items and no selection fields yet (v6_face.py select not run)"
                                % len(rows)]
    else:
        sel = {"from": str(p), "placeholder": False, "notes": notes, "voice_pairs": vps, "practice": sorted(practice),
               "mute_pairs": sorted(set(mute_flag)) or mute_rule(vps),
               "catch_cell": (catch_flag[0] if catch_flag else (sorted(practice)[0] if practice else "")),
               "tokens": dict(TOKENS), "cond": COND, "items": by_cell}
        if half:
            sel["notes"].append("voice pairs selected at one level only, ignored: %s" % half)
    if isinstance(s, dict):
        t = s.get("tokens")
        if isinstance(t, dict):
            for arm in ARMS:
                if isinstance(t.get(arm), str):
                    sel["tokens"][arm] = t[arm]
        if isinstance(s.get("cond"), str):
            sel["cond"] = s["cond"]
    if mute_over:
        sel["mute_pairs"] = sorted(vp_of(c) for c in mute_over)
    if catch_over:
        sel["catch_cell"] = catch_over
    for arm, tok in (tok_over or {}).items():
        if tok:
            sel["tokens"][arm] = tok
    if cond_over:
        sel["cond"] = cond_over
    sel["valid"], sel["problems"] = validate_selection(sel)
    return sel


def validate_selection(sel):
    probs = []
    vps, pr = sel["voice_pairs"], sel["practice"]
    if len(vps) != N_VOICE_PAIRS:
        probs.append("%d voice pairs, the design needs %d" % (len(vps), N_VOICE_PAIRS))
    emos = collections.Counter(emo_of(v + "_n") for v in vps)
    if len(emos) != 4 or set(emos.values()) != {3}:
        probs.append("voice pairs per emotion %s, the design is 4 emotions x 3 actors" % dict(emos))
    bad_emo = sorted(e for e in emos if e not in EMO_RESPONSE)
    if bad_emo:
        probs.append("emotions without a response code: %s" % bad_emo)
    odd = sorted({actor_of(v + "_n") for v in vps + [vp_of(c) for c in pr]
                  if actor_of(v + "_n") and int(actor_of(v + "_n")[3:]) % 2 == 1})
    if odd:
        probs.append("male RAVDESS actors (odd ids) selected: %s (the avatar is female)" % odd)
    if len(pr) != N_PRACTICE or not all(CELL_RE.match(c) for c in pr):
        probs.append("practice cells %s, the design needs %d rav<AA>_<emo>_<n|s> cells" % (pr, N_PRACTICE))
    # SELECTION.md practice rule (tiers 1-3, amended 2026-09-14 14:35): tier 1 an actor outside the study; tier 2 a
    # study actor in a non-selected emotion; tier 3 a study actor's strong clip in a chosen emotion that is NOT that
    # actor's own study emotion. Refused: a practice clip whose voice pair IS a study voice pair (the clip would be a
    # study cell), or whose emotion is the study emotion of that same actor. (Until 15:2x this refused every study
    # actor x chosen emotion, which contradicted tier 3 and the real selection: rav10_hap_s, actor 10 = fea study actor.)
    own_emos = collections.defaultdict(set)
    for v in vps:
        own_emos[actor_of(v + "_n")].add(emo_of(v + "_n"))
    bad_pr = [c for c in pr if vp_of(c) in vps or emo_of(c) in own_emos.get(actor_of(c), set())]
    if bad_pr:
        probs.append("practice clips may not be a study cell nor a study actor's own study emotion: %s" % bad_pr)
    if len(pr) == N_PRACTICE and len({emo_of(c) for c in pr}) != N_PRACTICE:
        probs.append("practice clips must be two different emotions: %s" % pr)
    if len(sel["mute_pairs"]) != N_MUTE or not set(sel["mute_pairs"]) <= set(vps):
        probs.append("silent pairs %s: need %d of the study voice pairs" % (sel["mute_pairs"], N_MUTE))
    cc = sel["catch_cell"]
    if not cc or not (cc in pr or (vp_of(cc) in vps and CELL_RE.match(cc))):
        probs.append("catch cell %r must be a practice cell or a study cell" % cc)
    if len(set(sel["tokens"].values())) != 3:
        probs.append("tokens not distinct: %s" % sel["tokens"])
    return not probs, probs


def cell_meta(sel, cell):
    """Manifest metadata for one cell, from pool_set_v6.json (falls back to the cell name)."""
    r = sel["items"].get(cell, {}) if sel else {}
    raw = r.get("label") or emo_of(cell)
    return {"lang": r.get("lang") or "en", "label_raw": raw, "voice_emotion": EMO_RESPONSE.get(raw, raw),
            "text": r.get("text", ""), "speaker": r.get("speaker") or actor_of(cell), "actor": actor_of(cell),
            "level": r.get("intensity") or level_of(cell), "ravdess_clip": r.get("ravdess_clip", ""),
            "wav_dur_s": r.get("dur_s"), "source": r.get("source", "actor")}


def selection_summary(sel):
    vps = sel["voice_pairs"]
    return {"from": sel["from"], "placeholder": sel["placeholder"], "valid": sel["valid"], "problems": sel["problems"],
            "notes": sel["notes"], "voice_pairs": vps,
            "emotions": sorted({emo_of(v + "_n") for v in vps},
                               key=lambda e: EMO_ORDER.index(e) if e in EMO_ORDER else 99),
            "actors_per_emotion": {e: sorted(actor_of(v + "_n") for v in vps if emo_of(v + "_n") == e)
                                   for e in sorted({emo_of(v + "_n") for v in vps})},
            "practice": sel["practice"], "mute_pairs": sel["mute_pairs"], "catch_cell": sel["catch_cell"],
            "voice_check_pairs": sel.get("voice_check_pairs") or [], "voice_check_from": sel.get("voice_check_from", ""),
            "single_level": LEVELS[SINGLE_LEVEL], "tokens": sel["tokens"], "cond": sel["cond"]}


# ---------------------------------------------------------------------------------------------------- v6 final: singles + Exp 2
def single_pairs_for(sel, singles):
    """(--singles) The voice pairs that get single screens. 12 -> all; 6 -> the silent subset (SINGLES_RULE[6])."""
    if int(singles) == len(sel["voice_pairs"]):
        return sorted(sel["voice_pairs"]), []
    if int(singles) == N_MUTE:
        sp = sorted(sel["mute_pairs"])
        probs = [] if len(sp) == N_MUTE and set(sp) <= set(sel["voice_pairs"]) else [
            "--singles %d: the silent subset %s is not %d study voice pairs" % (singles, sp, N_MUTE)]
        if sp != mute_rule(sel["voice_pairs"]):
            probs.append("--singles %d: the silent subset %s is not mute_rule(voice pairs) %s (override in use?)"
                         % (singles, sp, mute_rule(sel["voice_pairs"])))
        return sp, probs
    return sorted(sel["voice_pairs"]), ["--singles %s: only %s are defined a priori" % (singles, SINGLES_CHOICES)]


def ml_pool_items():
    try:
        return json.load(io.open(POOL / "manifest.json", encoding="utf-8")).get("items", {})
    except Exception:                                                    # noqa: BLE001
        return {}


def ml_meta(pool_items, cell):
    """Manifest metadata of an Exp 2 cell from pool/manifest.json (lang, label, text, source, wav)."""
    r = pool_items.get(cell) or {}
    parts = cell.split("_")
    lab = r.get("label") or next((p for p in parts if p in ("ang", "hap", "sad", "sur", "neu")), "")
    wav = r.get("wav", "")
    return {"lang": r.get("lang") or parts[0], "label": lab, "text": r.get("text", ""), "source": r.get("source", ""),
            "wav": wav, "voice_id": Path(wav).stem if wav else "", "length": r.get("length", ""), "wav_dur_s": r.get("dur_s")}


def ml_src(cell, arm, cond=COND, look=LOOK):
    return POOL / "mp4" / ("%s__%s__%s__%s.mp4" % (cell, ML_TOKENS[arm], cond, look))


def ml_px_per_rig_unit():
    """Face-crop px per rig unit per brow channel (v2a_face.crop_px_per_rig_unit, recomputed from gains.json)."""
    try:
        g = json.load(io.open(GAINS_JSON, encoding="utf-8"))
        k, iod = g["K_diag_used"], float(g["avatar_iod_px_fullframe"])
        return ({"brow_inner": abs(float(k[0])) * iod * CROP_SCALE, "brow_outer": abs(float(k[1])) * iod * CROP_SCALE},
                str(GAINS_JSON))
    except Exception as e:                                               # noqa: BLE001
        return dict(ML_PX_FALLBACK), "fallback constants (gains.json unreadable: %s)" % e


def ml_measure(cells=None, cond=COND):
    """(--ml) The cell-rule measurement on the pool/post arrays of every candidate: per brow channel, inside the cue-bank
    speech span (variants.speech_span, the span pool_rigs / v2a_face use), p95 - p5 of the arm's own clipped value (L/R
    averaged) for E, L, Lrev, and of the DIFFERENCE clip(L) - clip(E) (the rule) and clip(Lrev) - clip(E), in rig units
    and face-crop px. WSL only (the cue bank is keyed by the WSL wav path). Returns (rows, info)."""
    try:
        import numpy as np
        import variants as VAR                                           # numpy-only module (read-only import)
    except Exception as e:                                               # noqa: BLE001
        return [], {"error": "numpy / variants import failed: %s" % e}
    px, px_from = ml_px_per_rig_unit()
    pool_items = ml_pool_items()

    def c01(x):
        return np.clip(np.asarray(x, np.float64), 0.0, 1.0)

    def p9(v):
        return float(np.percentile(v, 95) - np.percentile(v, 5))

    rows = []
    for cell in (cells or ML_CANDIDATES):
        meta = ml_meta(pool_items, cell)
        row = {"cell": cell, "lang": meta["lang"], "emotion": meta["label"]}
        try:
            arr = {arm: np.load(str(POOL / "post" / ("%s__%s__%s.npy" % (cell, tok, cond)))) for arm, tok in ML_TOKENS.items()}
            T = arr["E"].shape[0]
            if any(x.shape != arr["E"].shape for x in arr.values()):
                raise ValueError("array shapes differ %s" % {k: v.shape for k, v in arr.items()})
            span = VAR.speech_span(VAR.load_cue(meta["wav"]), T)
            if span is None:
                raise ValueError("no speech span")
            sp = slice(span[0], span[1] + 1)
            row.update({"T60": int(T), "speech_a": int(span[0]), "speech_b": int(span[1])})
            for ch, cols in ML_CHANNELS.items():
                own = {arm: c01(arr[arm][sp][:, list(cols)]).mean(axis=1) for arm in ML_TOKENS}
                for arm in ML_TOKENS:
                    row["%s_%s_rig" % (arm, ch)] = round(p9(own[arm]), 3)
                dle, dre = own["L"] - own["E"], own["Lrev"] - own["E"]
                row["LmE_%s_rig" % ch] = round(p9(dle), 3)
                row["LmE_%s_px" % ch] = round(p9(dle) * px[ch], 2)
                row["LrevmE_%s_px" % ch] = round(p9(dre) * px[ch], 2)
                row["r_LmE_LrevmE_%s" % ch] = (round(float(np.corrcoef(dle, dre)[0, 1]), 2)
                                               if dle.std() > 1e-12 and dre.std() > 1e-12 else None)
            row["qualifies_on"] = [ch for ch in ML_CHANNELS if row["LmE_%s_px" % ch] >= ML_MIN_PX]
            row["qualifies"] = bool(row["qualifies_on"])
        except Exception as e:                                           # noqa: BLE001
            row.update({"error": str(e)[:200], "qualifies_on": [], "qualifies": False})
        rows.append(row)
    return rows, {"px_per_rig_unit": {k: round(v, 3) for k, v in px.items()}, "px_from": px_from,
                  "n_candidates": len(rows)}


def ml_select(rows):
    """Apply ML_RULE to the measured rows -> (cells in language x emotion order, problems); annotates row['role']."""
    chosen, probs = [], []
    for lang in ML_LANGS:
        for emo in ML_EMOTIONS:
            cands = [r for r in rows if r["lang"] == lang and r["emotion"] == emo]
            ok = [r for r in cands if r.get("qualifies")]
            if ok:
                chosen.append(ok[0]["cell"])
            else:
                probs.append("Exp 2: no %s %s cell qualifies (>= %.1f px L-E on a brow channel): %s"
                             % (lang, emo, ML_MIN_PX, [(r["cell"], r.get("error") or (r.get("LmE_brow_inner_px"),
                                                                                       r.get("LmE_brow_outer_px")))
                                                        for r in cands] or "no candidate"))
    for r in rows:
        if r["cell"] in chosen:
            r["role"] = "SELECTED"
        elif r["lang"] not in ML_LANGS or r["emotion"] not in ML_EMOTIONS:
            r["role"] = "not eligible (emotion outside the parity set ang/hap/sad)"
        elif r.get("qualifies"):
            r["role"] = "qualifies, not first in candidate order"
        else:
            r["role"] = "FAILS the px rule" + ((" (%s)" % r["error"]) if r.get("error") else "")
    return chosen, probs


def render_queue_status(settle_s=RENDER_SETTLE_S):
    """The English render queue (stimuli/render_list_v6.txt, cell:token:cond): how many exist / are settled."""
    p = STUDY / "stimuli" / "render_list_v6.txt"
    rows = []
    if p.is_file():
        for ln in io.open(p, encoding="utf-8"):
            ln = ln.split("#", 1)[0].strip()
            if ln.count(":") == 2:
                rows.append(tuple(ln.split(":")))
    now = time.time()
    paths = [POOL / "mp4" / ("%s__%s__%s__%s.mp4" % (c, t, cd, LOOK)) for c, t, cd in rows]
    ex = [q for q in paths if q.is_file()]
    return {"listed": len(rows), "present": len(ex), "settled": sum(1 for q in ex if now - q.stat().st_mtime >= settle_s),
            "newest": time.strftime("%H:%M:%S", time.localtime(max(q.stat().st_mtime for q in ex))) if ex else ""}


def settled_real_cells(sel, look, settle_s=RENDER_SETTLE_S):
    """(--provisional) English cells whose EVERY queued render exists, is newer than its rig and has not been written to
    for settle_s seconds: those cells use the real renders; every other cell uses stand-ins for all its arms (both sides
    of a pair then always share one wav)."""
    now, real = time.time(), []
    cells = ([("%s_%s" % (vp, lv), ("E", "H", "R")) for vp in sel["voice_pairs"] for lv in ("n", "s")]
             + [(c, ("E", "H")) for c in sel["practice"]])
    for cell, arms in cells:
        ok = True
        for arm in arms:
            p = POOL / "mp4" / ("%s__%s__%s__%s.mp4" % (cell, sel["tokens"][arm], sel["cond"], look))
            rig = POOL / "rigs" / ("%s__%s__%s.txt" % (cell, sel["tokens"][arm], sel["cond"]))
            if (not p.is_file() or not rig.is_file() or now - p.stat().st_mtime < settle_s
                    or p.stat().st_mtime <= rig.stat().st_mtime):
                ok = False
                break
        if ok:
            real.append(cell)
    return sorted(real)


def l_side(it):
    """Exp 2 pairs: the side LiltFace L is on."""
    cm = it.get("cue_manip") or ""
    if "|" not in cm:
        return ""
    left, right = [x.split(":", 1)[-1] for x in cm.split("|")[:2]]
    return "left" if left == "L" else "right" if right == "L" else ""


def ml_lang_of(it):
    va = it.get("voice_arm") or ""
    return va.split("@", 1)[1] if "@" in va else it.get("stim_language", "")


# ---------------------------------------------------------------------------------------------------- the plan (pure)
def fname(block, key, k):
    """Neutral media name: block + a code that hides cell, level and arm + the orientation index k."""
    return "v6_%s_%s_%d.mp4" % (block, hashlib.sha1(("v6|%s|%s" % (block, key)).encode("utf-8")).hexdigest()[:10], k)


def plan_items(sel, src_of, provisional=False, hr_normal=False, voice_check=False, wav_of=None, ml=False, src_ml=None):
    """THE STUDY as manifest items, each carrying an `op` (what media to make from which sources). No filesystem access
    beyond src_of() / wav_of() / src_ml(). Used by the build, by analyse_lilt_v6.py's simulation and by lilt_selftest_v6.py.
    hr_normal adds the pair block H_vs_R@normal (pair4_Rn groups); voice_check adds the audio-only block "voice_only"
    (wav_of(cell) -> the wav; default voice_wav_of(sel, cell)). Singles cover sel["single_pairs"] (default: every voice
    pair; --singles 6 = the silent subset). ml adds Experiment 2 (block pair_ml) on sel["ml_cells"] with metadata
    sel["ml_meta"] and sources src_ml(cell, arm in E | L | Lrev) (default: ml_src)."""
    vps = sorted(sel["voice_pairs"])
    n = len(vps)
    items = []
    wav_of = wav_of or (lambda c: voice_wav_of(sel, c))
    src_ml = src_ml or (lambda c, arm: ml_src(c, arm, sel.get("cond", COND)))

    def common(cell, block, pres):
        m = cell_meta(sel, cell)
        return {"block": block, "presentation": pres, "stim_language": m["lang"], "text_type": "short",
                "text_content": m["text"], "character": "Danielle", "char_group": "ambiguous",
                "tts_model": "actor" if m["source"] == "actor" else "clone", "voice_id": m["speaker"], "is_static": 0,
                "utterance": cell, "base_cell": vp_of(cell), "voice_level": m["level"], "voice_rung": m["level"],
                "voice_emotion": m["voice_emotion"], "emotion_label": m["label_raw"], "actor": m["actor"],
                "ravdess_clip": m["ravdess_clip"], "face_deg": "d5", "provisional": bool(provisional), "experiment": 1}

    def stem(p):
        return Path(p).stem

    def pair_item(cell, block, L, R, k, contrast, group, op_kind, key):
        pl, pr = src_of(cell, L), src_of(cell, R)
        f = fname(block, key, k)
        it = common(cell, block if block != "practice" else "practice", "AV_pair")
        it.update({"id": f[:-4], "file": f, "face_emotion": "", "congruency": "", "renderer": "pair",
                   "face_motion": "%s|%s" % (stem(pl), stem(pr)), "cue_manip": "L:%s|R:%s" % (L, R),
                   "voice_arm": contrast, "contrast": contrast, "pick_group": group, "source": "",
                   "sources": [str(pl), str(pr)], "op": {"kind": op_kind, "inputs": [str(pl), str(pr)]}})
        return it

    for vp in sorted(sel.get("single_pairs") or vps):                    # ---- singles: E | H | R at strong
        cell = "%s_%s" % (vp, SINGLE_LEVEL)
        for k, arm in enumerate(ARMS, 1):
            p = src_of(cell, arm)
            f = fname("single", cell, k)
            it = common(cell, "single", "AV")
            it.update({"id": f[:-4], "file": f, "face_emotion": it["voice_emotion"], "congruency": "congruent",
                       "renderer": arm, "face_motion": stem(p), "cue_manip": "", "voice_arm": "", "contrast": "",
                       "pick_group": "single_" + vp, "source": str(p), "op": {"kind": "single", "inputs": [str(p)]}})
            items.append(it)

    specs = [("pair1_En", "n", "E", lambda j: j), ("pair2_Es", "s", "E", lambda j: j),
             ("pair3_Rs", "s", "R", lambda j: (j + 1) % n)]             # the +1 flips H's side for H_vs_R (docstring)
    if hr_normal:                                                        # (--hr-normal) 12 groups more, sorted after
        specs.append(("pair4_Rn", "n", "R", lambda j: (j + 1) % n))     # pair3: same H side as H_vs_R@strong, the
    for prefix, lv, foil, slot in specs:                                 # ---- pairs with sound   other than H_vs_E
        contrast = "H_vs_%s@%s" % (foil, LEVELS[lv])
        for j, vp in enumerate(vps):
            cell = "%s_%s" % (vp, lv)
            group = "%s_%02d_%s" % (prefix, slot(j), vp)
            for k, (L, R) in ((1, (foil, "H")), (2, ("H", foil))):      # k = 1: H on the right
                items.append(pair_item(cell, "pair", L, R, k, contrast, group, "sbs", "%s|%s" % (contrast, cell)))

    for cell in sorted(sel["practice"]):                                 # ---- practice: H vs E, unscored
        contrast = "practice:H_vs_E@%s" % level_of(cell)
        for k, (L, R) in ((1, ("E", "H")), (2, ("H", "E"))):
            items.append(pair_item(cell, "practice", L, R, k, contrast, "practice_" + cell, "sbs", contrast + "|" + cell))

    for vp in sorted(sel["mute_pairs"]):                                 # ---- silent pairs at strong, served last
        cell = "%s_s" % vp
        for k, (L, R) in ((1, ("E", "H")), (2, ("H", "E"))):
            items.append(pair_item(cell, "pair_mute", L, R, k, MUTE_CONTRAST, "mute_" + vp, "mute",
                                   MUTE_CONTRAST + "|" + cell))

    cell = sel["catch_cell"]                                             # ---- the one catch: H moving vs frozen
    mov = src_of(cell, "H")
    for k, side in ((1, "left"), (2, "right")):
        f = fname("catch", cell, k)
        it = common(cell, "pair_catch", "AV_pair")
        it.update({"id": f[:-4], "file": f, "face_emotion": "", "congruency": "", "renderer": "pair",
                   "face_motion": stem(mov), "cue_manip": "L:H|R:still" if side == "left" else "L:still|R:H",
                   "voice_arm": "correct:" + side, "contrast": "catch", "pick_group": "pair9_catch_" + cell,
                   "source": str(mov), "op": {"kind": "catch", "inputs": [str(mov)], "side": side}})
        items.append(it)

    if ml:                                                               # ---- (--ml) Experiment 2, three languages
        cells = list(sel.get("ml_cells") or [])
        n_ml = len(cells)
        for ci, c0 in enumerate(ML_CONTRASTS):
            foil = "E" if c0 == "L_vs_E" else "Lrev"
            for j, cell in enumerate(cells):
                mm = (sel.get("ml_meta") or {}).get(cell) or ml_meta({}, cell)
                contrast = "%s@%s" % (c0, mm["lang"])
                # server rule arms[(i + n_prior) % 2]: the L_vs_Lrev group of cell j must sit an ODD number of groups
                # after its L_vs_E group so that L changes side -> slot j when n_ml is odd (offset n_ml), else j + 1
                slot = j if (ci == 0 or n_ml % 2 == 1) else (j + 1) % n_ml
                group = "pair5_ml%d_%s_%02d_%s" % (ci + 1, "LE" if ci == 0 else "LR", slot, cell)
                for k, (Lf, Rt) in ((1, (foil, "L")), (2, ("L", foil))):  # k = 1: L on the right
                    pl, pr = src_ml(cell, Lf), src_ml(cell, Rt)
                    f = fname(ML_BLOCK, "%s|%s" % (contrast, cell), k)
                    items.append({
                        "block": ML_BLOCK, "presentation": "AV_pair", "experiment": 2, "stim_language": mm["lang"],
                        "text_type": mm.get("length") or "short", "text_content": mm.get("text", ""),
                        "character": "Danielle", "char_group": "ambiguous",
                        "tts_model": "actor" if mm.get("source") == "actor" else "clone", "voice_id": mm.get("voice_id", ""),
                        "is_static": 0, "utterance": cell, "base_cell": cell, "voice_level": "", "voice_rung": "",
                        "voice_emotion": EMO_RESPONSE.get(mm["label"], mm["label"]), "emotion_label": mm["label"],
                        "actor": "", "ravdess_clip": "", "face_deg": "d5", "provisional": bool(provisional),
                        "id": f[:-4], "file": f, "face_emotion": "", "congruency": "", "renderer": "pair",
                        "face_motion": "%s|%s" % (stem(pl), stem(pr)), "cue_manip": "L:%s|R:%s" % (Lf, Rt),
                        "voice_arm": contrast, "contrast": contrast, "pick_group": group, "source": "",
                        "sources": [str(pl), str(pr)], "op": {"kind": "sbs", "inputs": [str(pl), str(pr)]}})

    if voice_check:                                                      # ---- (--voice-check) audio only, served last
        for vp in sorted(sel.get("voice_check_pairs") or []):
            for lv in ("n", "s"):
                cell = "%s_%s" % (vp, lv)
                wav = wav_of(cell)
                f = fname("voice", cell, 1)
                it = common(cell, "voice_only", "voice_only")
                it.update({"id": f[:-4], "file": f, "character": "", "char_group": "", "face_deg": "",
                           "face_emotion": "", "congruency": "", "renderer": "voice", "face_motion": "",
                           "cue_manip": "voice:%s" % LEVELS[lv], "voice_arm": VOICE_CHECK_ARM, "contrast": VOICE_CHECK_ARM,
                           "pick_group": "voice_" + cell, "source": str(wav),
                           "op": {"kind": "voice_only", "inputs": [str(wav)]}})
                items.append(it)
    return items


def required_sources(sel, src_of, hr_normal=False):
    """The 64 renders: E/H at normal, E/H/R at strong, E/H for the practice cells (+ H of a catch cell outside these);
    with hr_normal also R at normal (76)."""
    need = []
    for vp in sorted(sel["voice_pairs"]):
        need += [("%s_n" % vp, "E"), ("%s_n" % vp, "H"), ("%s_s" % vp, "E"), ("%s_s" % vp, "H"), ("%s_s" % vp, "R")]
        if hr_normal:
            need.append(("%s_n" % vp, "R"))
    for c in sorted(sel["practice"]):
        need += [(c, "E"), (c, "H")]
    if (sel["catch_cell"], "H") not in need:
        need.append((sel["catch_cell"], "H"))
    return [(c, arm, src_of(c, arm)) for c, arm in need]


def required_wavs(sel, voice_check=False):
    """(--voice-check) the wavs of the audio-only clips: [(cell, path)]."""
    if not voice_check:
        return []
    return [("%s_%s" % (vp, lv), voice_wav_of(sel, "%s_%s" % (vp, lv)))
            for vp in sorted(sel.get("voice_check_pairs") or []) for lv in ("n", "s")]


def required_ml_sources(sel, look=LOOK):
    """(--ml) The Exp 2 renders: E, L, Lrev of every selected cell (already rendered; no new renders)."""
    return [(c, arm, ml_src(c, arm, sel.get("cond", COND), look)) for c in sel.get("ml_cells") or [] for arm in ML_TOKENS]


# ---------------------------------------------------------------------------------------------------- server rotation
def serve_like_server(items, n_prior):
    """server.py mode "lilt" (webapp/server.py ~L2158-2213): groups by pick_group, sorted by name; each group serves
    arms[(i + n_prior) % len(arms)] with arms sorted by _relpath = "media/" + file; block order single, practice,
    pair + pair_catch (shuffled together), pair_mute. Within-block shuffles are not reproduced (order-free checks only).
    lilt_selftest_v6.py verifies the server source still contains these lines."""
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
    return (by.get("single", []) + by.get("practice", []) + by.get("pair", []) + by.get("pair_catch", [])
            + by.get(ML_BLOCK, [])                                       # Exp 2 after the English pairs: server hunk
            + by.get("pair_mute", []) + by.get("voice_only", []))      # voice_only last: patches/server_v6.patch


def plan_options(items):
    """Which optional blocks a plan / manifest item list contains (keys of expected_served)."""
    return {"hr_normal": any(it.get("voice_arm") == HR_NORMAL_CONTRAST for it in items),
            "voice_check": any(it.get("block") == "voice_only" for it in items),
            "ml": any(it.get("block") == ML_BLOCK for it in items),
            "singles": len({it.get("pick_group") for it in items if it.get("block") == "single"})}


def h_side(it):
    cm = it.get("cue_manip") or ""
    if "|" not in cm:
        return ""
    left, right = cm.split("|")[0].split(":", 1)[-1], cm.split("|")[1].split(":", 1)[-1]
    return "left" if left == "H" else "right" if right == "H" else ""


def rotation_report(items, n_participants=12):
    """Per-participant counts and balance under the server rule, for n_prior = 0 .. n_participants-1."""
    problems, per = [], []
    single_cover, orient_cover = {}, {}
    n_vp = len({it["base_cell"] for it in items if it.get("voice_arm") == "H_vs_E@strong"}) or N_VOICE_PAIRS
    n_single = len({it["base_cell"] for it in items if it["block"] == "single"})
    ml_cells = sorted({it["base_cell"] for it in items if it["block"] == ML_BLOCK})
    ml_langs = collections.Counter((it["stim_language"], it["base_cell"]) for it in items if it["block"] == ML_BLOCK)
    ml_cells_per_lang = collections.Counter(lang for lang, _ in ml_langs)
    opts = plan_options(items)
    expected = expected_served(**opts)
    contrasts = av_contrasts(opts["hr_normal"])
    n_voice = len({it["file"] for it in items if it["block"] == "voice_only"})
    for n in range(n_participants):
        served = serve_like_server(items, n)
        cnt = dict(collections.Counter(x["block"] for x in served))
        if cnt != expected:
            problems.append("n_prior=%d served %s, expected %s" % (n, cnt, expected))
        left = {c: sum(1 for x in served if x.get("voice_arm") == c and h_side(x) == "left") for c in contrasts}
        left["mute"] = sum(1 for x in served if x["block"] == "pair_mute" and h_side(x) == "left")
        for c in contrasts:
            if left[c] != n_vp // 2:
                problems.append("n_prior=%d %s has H on the left %d/%d" % (n, c, left[c], n_vp))
        if opts["voice_check"]:
            vo = [x for x in served if x["block"] == "voice_only"]
            if len(vo) != n_voice or len({x["file"] for x in vo}) != n_voice:
                problems.append("n_prior=%d hears %d/%d voice-only clips" % (n, len(vo), n_voice))
            lv = collections.Counter(x["voice_level"] for x in vo)
            if lv.get("normal") != lv.get("strong"):
                problems.append("n_prior=%d voice-only levels %s" % (n, dict(lv)))
        n_mute = sum(1 for x in served if x["block"] == "pair_mute")
        if left["mute"] != n_mute // 2:
            problems.append("n_prior=%d silent pairs have H on the left %d/%d" % (n, left["mute"], n_mute))
        arms = {a: sum(1 for x in served if x["block"] == "single" and x["renderer"] == a) for a in ARMS}
        if sorted(arms.values()) != [n_single // 3] * 3:
            problems.append("n_prior=%d single arms %s" % (n, arms))
        ml_left = {}
        if opts["ml"]:
            mls = [x for x in served if x["block"] == ML_BLOCK]
            n_ml = len(ml_cells)
            for c0 in ML_CONTRASTS:
                xs = [x for x in mls if (x["voice_arm"] or "").startswith(c0 + "@")]
                ml_left[c0] = sum(1 for x in xs if l_side(x) == "left")
                if len(xs) != n_ml or ml_left[c0] not in (n_ml // 2, (n_ml + 1) // 2):
                    problems.append("n_prior=%d Exp 2 %s: %d screens, L on the left %d" % (n, c0, len(xs), ml_left[c0]))
            ml_left["total"] = sum(1 for x in mls if l_side(x) == "left")
            if ml_left["total"] != n_ml:
                problems.append("n_prior=%d Exp 2: L on the left %d/%d" % (n, ml_left["total"], len(mls)))
            by_cell = {}
            for x in mls:
                by_cell.setdefault(x["base_cell"], {})[x["voice_arm"].split("@")[0]] = l_side(x)
            flipped = sum(1 for d in by_cell.values() if d.get("L_vs_E") and d.get("L_vs_Lrev") and d["L_vs_E"] != d["L_vs_Lrev"])
            ml_left["cells_L_side_flipped_LE_vs_LR"] = flipped
            if flipped != n_ml:
                problems.append("n_prior=%d Exp 2: L side flipped between L_vs_E and L_vs_Lrev in %d/%d cells" % (n, flipped, n_ml))
            for lang, nc in ml_cells_per_lang.items():
                ll = sum(1 for x in mls if ml_lang_of(x) == lang and l_side(x) == "left")
                ml_left["left_" + lang] = ll
                if ll != nc:
                    problems.append("n_prior=%d Exp 2 %s: L on the left %d/%d" % (n, lang, ll, 2 * nc))
        sides = {}
        for x in served:
            if x.get("voice_arm") in contrasts or x.get("voice_arm") == MUTE_CONTRAST:
                sides.setdefault(x["base_cell"], {})[x["voice_arm"]] = h_side(x)
        same_level = sum(1 for d in sides.values() if d.get("H_vs_E@normal") and d.get("H_vs_E@normal") == d.get("H_vs_E@strong"))
        flip_rev = sum(1 for d in sides.values() if d.get("H_vs_R@strong") and d.get("H_vs_E@strong") != d.get("H_vs_R@strong"))
        mute_same = sum(1 for d in sides.values() if d.get(MUTE_CONTRAST) and d.get(MUTE_CONTRAST) == d.get("H_vs_E@strong"))
        rn_same = sum(1 for d in sides.values() if d.get(HR_NORMAL_CONTRAST) and d.get(HR_NORMAL_CONTRAST) == d.get("H_vs_R@strong"))
        if same_level != n_vp:
            problems.append("n_prior=%d H on the same side at normal and strong in %d/%d voice pairs" % (n, same_level, n_vp))
        if flip_rev != n_vp:
            problems.append("n_prior=%d H side flipped in H_vs_R in %d/%d voice pairs" % (n, flip_rev, n_vp))
        if opts["hr_normal"] and rn_same != n_vp:
            problems.append("n_prior=%d H on the same side in H_vs_R at normal and strong in %d/%d voice pairs" % (n, rn_same, n_vp))
        audio = collections.Counter(x["utterance"] for x in served if x["block"] in ("single", "pair", "voice_only"))
        ml_audio = collections.Counter(x["utterance"] for x in served if x["block"] == ML_BLOCK)
        for x in served:
            if x["block"] == "single":
                single_cover.setdefault(x["base_cell"], collections.Counter())[x["renderer"]] += 1
            if x["block"] in ("pair", "pair_mute", "practice", "pair_catch", ML_BLOCK):
                orient_cover[x["file"]] = orient_cover.get(x["file"], 0) + 1
        per.append({"n_prior": n, "served": cnt, "n_served": sum(cnt.values()), "H_left": left, "single_arms": arms,
                    "exp2_L_left": ml_left or None, "max_sound_exposures_exp2_cell": max(ml_audio.values() or [0]),
                    "voice_pairs_same_H_side_normal_vs_strong": same_level, "voice_pairs_H_side_flipped_in_H_vs_R": flip_rev,
                    "silent_H_side_same_as_AV_strong": mute_same,
                    "voice_pairs_same_H_side_H_vs_R_normal_vs_strong": rn_same if opts["hr_normal"] else None,
                    "max_sound_exposures_per_scored_clip": {"strong": max([v for k, v in audio.items() if k.endswith("_s")] or [0]),
                                                            "normal": max([v for k, v in audio.items() if k.endswith("_n")] or [0])}})
    if n_participants % 6 == 0:
        k3, k2 = n_participants // 3, n_participants // 2
        for vp, d in single_cover.items():
            if sorted(d.values()) != [k3] * 3 or len(d) != 3:
                problems.append("single %s arm coverage %s over %d participants" % (vp, dict(d), n_participants))
        bad = [it["file"] for it in items if it["block"] in ("pair", "pair_mute", "practice", "pair_catch", ML_BLOCK)
               and orient_cover.get(it["file"], 0) != k2]
        if bad:
            problems.append("%d pair files not served to exactly half of %d participants, e.g. %s" % (len(bad), n_participants, bad[:2]))
    return {"ok": not problems, "n_participants": n_participants, "problems": problems[:20], "per_participant": per[:3],
            "options": opts, "expected_served": expected,
            "voice_pairs_same_H_side_normal_vs_strong": sorted({p["voice_pairs_same_H_side_normal_vs_strong"] for p in per}),
            "voice_pairs_H_side_flipped_in_H_vs_R": sorted({p["voice_pairs_H_side_flipped_in_H_vs_R"] for p in per}),
            "silent_H_side_same_as_AV_strong": sorted({p["silent_H_side_same_as_AV_strong"] for p in per}),
            "voice_pairs_same_H_side_H_vs_R_normal_vs_strong": sorted({p["voice_pairs_same_H_side_H_vs_R_normal_vs_strong"]
                                                                       for p in per if p["voice_pairs_same_H_side_H_vs_R_normal_vs_strong"] is not None}),
            "max_sound_exposures_per_scored_clip": per[0]["max_sound_exposures_per_scored_clip"] if per else {}}


# ---------------------------------------------------------------------------------------------------- provisional stand-ins
def standin_catalog(look):
    cat = {}
    for p in sorted((POOL / "mp4").glob("*__e10m10_d*_h__%s.mp4" % look)):
        parts = p.stem.split("__")
        if len(parts) != 4:
            continue
        m = re.fullmatch(r"e10m10_d(\d+)_h", parts[2])
        if m:
            cat.setdefault(parts[0], []).append({"tok": parts[1], "deg": int(m.group(1)), "path": p})
    return cat


def make_standin_src(cat):
    """PROVISIONAL: existing Danielle V9 1:1 held (e10m10_d*_h) renders. Every arm of one voice pair comes from ONE
    stand-in cell (an English cell of the same emotion when one exists, else any English cell, else any), so both sides
    of a pair share a wav and duration. E = a 'before' clip, H = a non-'before' clip, R = a '_rev' clip when present.
    NOT the study media: no human brows, often another emotion, never RAVDESS."""
    used = {}
    cells = sorted(cat)

    def src(cell, arm):
        vp, emo = vp_of(cell), emo_of(cell)
        lvl = (CELL_RE.match(cell).group(4) if CELL_RE.match(cell) else "s")
        en = [c for c in cells if c.startswith("en_")]
        pool = [c for c in en if ("_%s_" % emo) in c] or en or cells
        sc = pool[int(hashlib.md5(vp.encode("utf-8")).hexdigest(), 16) % len(pool)]
        clips = cat[sc]
        pref = [5, 4, 6, 2, 1] if lvl == "s" else [2, 4, 1, 5, 6]
        order = sorted(clips, key=lambda c: (pref.index(c["deg"]) if c["deg"] in pref else 99, c["tok"]))
        bef = [c for c in order if c["tok"] == "before"] or order
        deg = bef[0]["deg"]

        def at_deg(xs):
            return [c for c in xs if c["deg"] == deg] or xs
        non = at_deg([c for c in order if c["tok"] != "before" and "_rev" not in c["tok"]])
        rev = at_deg([c for c in order if "_rev" in c["tok"]])
        if arm == "E":
            c = bef[0]
        elif arm == "H":
            c = (non or rev or bef)[0]
        else:
            c = (rev or non[1:] or non or bef)[0]
        used[(cell, arm)] = {"stand_in": c["path"].name, "stand_in_cell": sc, "same_emotion": ("_%s_" % emo) in sc}
        return c["path"]
    return src, used


# ---------------------------------------------------------------------------------------------------- media
def media_qc(p):
    """Frame size, audio presence, integrated loudness, and any container tags (exposure check)."""
    r = B.sh(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:stream_tags:format_tags",
              "-of", "json", str(p)])
    try:
        j = json.loads(r.stdout or "{}")
    except ValueError:
        j = {}
    st = j.get("streams", [])
    v = next((s for s in st if s.get("codec_type") == "video"), {})
    tags = dict((j.get("format") or {}).get("tags") or {})
    for s in st:
        tags.update({"%s:%s" % (s.get("codec_type"), k): val for k, val in (s.get("tags") or {}).items()})
    q = {"w": v.get("width"), "h": v.get("height"), "audio": any(s.get("codec_type") == "audio" for s in st)}
    if q["audio"]:
        rr = B.sh(["ffmpeg", "-hide_banner", "-nostats", "-i", str(p), "-af", "ebur128", "-f", "null", "-"])
        m = re.findall(r"I:\s+(-?[\d.]+) LUFS", rr.stderr or "")
        q["lufs"] = float(m[-1]) if m else None
    q["exposing_tags"] = {k: val for k, val in tags.items()
                          if re.search(r"h6|human|_rev|before|lilt|__|rav\d|danielle|emoface", str(val), re.I)}
    return q


def voice_black(wav, out, w=1120, h=630):
    """(--voice-check) The wav under a black w x h frame: the app's voice-only convention (h2_wave1_ladder.wrap_mp4:
    an mp4 behind the audio mask keeps the audio-only screens on the same playback path as every other trial), at the
    stage size of this study and with the pool_render mux loudness (loudnorm I=-23 LUFS), here TWO-PASS so a 3-4 s clip
    lands on the target (single-pass loudnorm on short clips misses by 1-2 LU). Same container as the face clips
    (h264 yuv420p + aac, faststart)."""
    wav, out = Path(wav), Path(out)
    if not wav.is_file():
        return False, "wav missing: %s" % wav
    # 18:30 fix (verifier, verify_design-media/lufs_fix_test*.log): a very quiet wav (rav20_fea_r2 normal, ebur128
    # -45.2 LUFS) landed at -24.0 through loudnorm on the QC meter. Now: measure the wav with the SAME meter as media_qc
    # (ebur128) and apply a plain linear gain when the resulting sample peak stays <= VOICE_TP; loudnorm only otherwise;
    # then ONE corrective re-mux when the muxed clip is still > VOICE_LUFS_FIX_TOL off target (AAC shifts the level).
    I_in, pk_in = _ebur128(wav)
    if I_in is not None and pk_in is not None and pk_in + (VOICE_LUFS - I_in) <= VOICE_TP:
        af, gain = "volume=%.2fdB" % (VOICE_LUFS - I_in), VOICE_LUFS - I_in
    else:
        gain = None
        m1 = B.sh(["ffmpeg", "-hide_banner", "-nostats", "-i", str(wav), "-af",
                   "loudnorm=I=%g:TP=%g:LRA=%g:print_format=json" % (VOICE_LUFS, VOICE_TP, VOICE_LRA), "-f", "null", "-"])
        mm = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", m1.stderr or "", re.S)
        af = "loudnorm=I=%g:TP=%g:LRA=%g" % (VOICE_LUFS, VOICE_TP, VOICE_LRA)
        if mm:
            try:
                j = json.loads(mm.group(0))
                af += (":measured_I=%s:measured_TP=%s:measured_LRA=%s:measured_thresh=%s:offset=%s:linear=true"
                       % (j["input_i"], j["input_tp"], j["input_lra"], j["input_thresh"], j["target_offset"]))
            except (KeyError, ValueError):
                pass

    def mux(filt):
        return B.sh(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=%dx%d:r=30" % (w, h),
                     "-i", str(wav), "-shortest", "-af", filt + ",aresample=48000", "-map", "0:v", "-map", "1:a",
                     "-c:v", "libx264", "-tune", "stillimage", "-crf", "23", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-map_metadata", "-1", str(out)])
    r = mux(af)
    if gain is not None and out.is_file():
        I_out, _ = _ebur128(out)
        if I_out is not None and abs(I_out - VOICE_LUFS) > VOICE_LUFS_FIX_TOL \
                and pk_in + gain + (VOICE_LUFS - I_out) <= VOICE_TP:
            r = mux("volume=%.2fdB" % (gain + VOICE_LUFS - I_out))
    return out.is_file() and out.stat().st_size > 5000, (r.stderr or "")[-160:]


def _ebur128(p):
    """(integrated LUFS, sample peak dBFS) on the media_qc meter (ffmpeg ebur128); (None, None) when unreadable."""
    rr = B.sh(["ffmpeg", "-hide_banner", "-nostats", "-i", str(p), "-af", "ebur128=peak=sample", "-f", "null", "-"])
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", rr.stderr or "")
    pk = re.findall(r"Peak:\s+(-?[\d.]+|-inf) dBFS", rr.stderr or "")
    try:
        return (float(m[-1]) if m else None), (float(pk[-1]) if pk and pk[-1] != "-inf" else None)
    except ValueError:
        return None, None


def execute(items, stage, workers):
    """Make every item's media in `stage`. Returns the failures."""
    stage.mkdir(parents=True, exist_ok=True)
    stills, fails = {}, []
    for it in items:                                                     # the frozen foil first (sequential)
        op = it["op"]
        if op["kind"] == "catch" and op["inputs"][0] not in stills:
            st = stage / ("_still_%s.mp4" % it["utterance"])
            ok, why = B.still_of(Path(op["inputs"][0]), st)
            stills[op["inputs"][0]] = st if ok else None
            if not ok:
                fails.append({"file": st.name, "why": why})

    def one(it):
        op, out = it["op"], stage / it["file"]
        ins = [Path(x) for x in op["inputs"]]
        if op["kind"] == "single":
            return B.crop_single(ins[0], out)
        if op["kind"] == "sbs":
            return B.sbs(ins[0], ins[1], out)
        if op["kind"] == "mute":
            return B.sbs_mute(ins[0], ins[1], out)
        if op["kind"] == "voice_only":
            return voice_black(ins[0], out)
        st = stills.get(op["inputs"][0])
        if st is None:
            return False, "still_of failed"
        return B.sbs(ins[0], st, out) if op["side"] == "left" else B.sbs(st, ins[0], out)

    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for it, (ok, why) in zip(items, ex.map(one, items)):
            it["_ok"] = ok
            if not ok:
                fails.append({"file": it["file"], "why": why})
    for st in stills.values():
        if st is not None:
            st.unlink(missing_ok=True)
            st.with_suffix(".png").unlink(missing_ok=True)
    return fails


def est_minutes(items, served):
    """v4's burden formula (build_lilt_icassp.py): clip + 14 s per single, + 6 s per pair / silent pair / Exp 2 pair,
    + 5 s catch, + 10 s practice, + 9 s per voice-only screen, + 240 s for instructions and the device check."""
    def mean_dur(block):
        ds = [it.get("dur_s") or 0 for it in items if it["block"] == block]
        return sum(ds) / len(ds) if ds else 0.0
    sec = (served.get("single", 0) * (mean_dur("single") + 14) + served.get("pair", 0) * (mean_dur("pair") + 6)
           + served.get("pair_mute", 0) * (mean_dur("pair_mute") + 6)
           + served.get("pair_catch", 0) * (mean_dur("pair_catch") + 5)
           + served.get("practice", 0) * (mean_dur("practice") + 10) + 240
           + served.get("voice_only", 0) * (mean_dur("voice_only") + 9)    # 2 fields (2.7 s each) + 2.5 s fixed + margin
           + served.get(ML_BLOCK, 0) * (mean_dur(ML_BLOCK) + 6))           # Exp 2 pairs: the same CCR screen as a pair
    return round(sec / 60.0, 1)


def client_minutes(n_screens):
    """app.js: Math.max(2, Math.round((n * 11 + 150) / 60)) -- the number the instructions show."""
    return max(2, int((n_screens * 11 + 150) / 60.0 + 0.5))


def ml_summary(rows, cells):
    """Medians over the selected Exp 2 cells (rig units), next to the English human reference."""
    sel_rows = [r for r in rows if r["cell"] in cells and "error" not in r]

    def med(k):
        v = sorted(r[k] for r in sel_rows if r.get(k) is not None)
        return round((v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2.0, 3) if v else None
    return {"n_cells": len(sel_rows),
            "median_own_p95p5_rig": {arm: {ch: med("%s_%s_rig" % (arm, ch)) for ch in ML_CHANNELS} for arm in ML_TOKENS},
            "median_LminusE_p95p5_px": {ch: med("LmE_%s_px" % ch) for ch in ML_CHANNELS},
            "median_LminusE_p95p5_rig": {ch: med("LmE_%s_rig" % ch) for ch in ML_CHANNELS},
            "human_reference_rig": HUMAN_BROW_REF_RIG,
            "note": "own = p95-p5 of the arm's clipped brow value in speech; the English RAVDESS human reference "
                    "(0.48 inner / 0.35 outer rig units) is the main session's measurement [not recomputed here]"}


def design_table(served, sel, opts=None):
    opts = opts or {}
    ns = served.get("single", 0)
    rows = [
        {"block": "single", "per_participant": ns,
         "arms": "E | H | R at the strong voice (pick_group of 3, %s per participant) on %s"
                 % ("/".join([str(ns // 3)] * 3), ", ".join(sel.get("single_pairs") or sel["voice_pairs"])),
         "response": "7-choice emotion + vividness of the facial movement 1-7 (CSV column `intensity`) + naturalness 1-7",
         "purpose": "first-exposure vividness and naturalness per version; emotion recognition by arm. Singles rule: %s"
                    % SINGLES_RULE.get(ns, SINGLES_RULE[12])},
        {"block": "practice", "per_participant": served.get("practice", 0),
         "arms": "H vs E on %s (not study cells; SELECTION.md practice tiers)" % " / ".join(sel["practice"]),
         "response": "CCR -3..+3", "purpose": "teach the comparison scale; unscored"},
        {"block": "pair H_vs_E@normal", "per_participant": 12, "arms": "H vs E, one shared normal-intensity voice",
         "response": "CCR L3..R3 'which face's movement fits the voice better'", "purpose": "P1 (pooled with strong), P3 baseline"},
        {"block": "pair H_vs_E@strong", "per_participant": 12,
         "arms": "H vs E, one shared strong-intensity voice (H on the same side as at normal, per voice pair)",
         "response": "CCR", "purpose": "P1 (pooled with normal); P3 = strong minus normal within participant"},
        {"block": "pair H_vs_R@strong", "per_participant": 12,
         "arms": "H vs R (the same human brow track time-reversed; H on the other side than in H_vs_E@strong)",
         "response": "CCR", "purpose": "P2: alignment with the voice, motion equated"},
        {"block": "pair_catch", "per_participant": served.get("pair_catch", 0),
         "arms": "H moving vs the same clip frozen (%s)" % sel["catch_cell"],
         "response": "left / right: 'one face is frozen -- which face is moving?'", "purpose": "attention (one per participant)"},
        {"block": "pair_mute", "per_participant": served.get("pair_mute", 0),
         "arms": "H vs E at strong WITHOUT sound: %s" % ", ".join(sel["mute_pairs"]),
         "response": "CCR on 'which face's movement is more lively and expressive?'",
         "purpose": "manipulation check (the brow layer is visible), served last%s"
                    % (" before the voice check" if opts.get("voice_check") else "")},
    ]
    if opts.get("hr_normal"):
        rows.insert(5, {"block": "pair H_vs_R@normal (OPTION --hr-normal)", "per_participant": 12,
                        "arms": "H vs R at the normal voice (H on the same side as in H_vs_R@strong)",
                        "response": "CCR", "purpose": "P2b (secondary, not in the Holm family): alignment matters at normal "
                                                      "too; with P2 a motion-equated companion of P3"})
    if opts.get("ml"):
        at = next(i for i, r in enumerate(rows) if r["block"] == "pair_mute")
        cells = sel.get("ml_cells") or []
        n_ml = served.get(ML_BLOCK, 0) // 2
        rows[at:at] = [
            {"block": "pair_ml L_vs_E@<lang> (Experiment 2, --ml)", "per_participant": n_ml,
             "arms": "LiltFace L (%s) vs EmoFace E (%s), one shared voice, cells %s"
                     % (ML_TOKENS["L"], ML_TOKENS["E"], ", ".join(cells)),
             "response": "CCR L3..R3, the same pair question as the English pairs",
             "purpose": "P4: L fits the voice better than E, pooled over stimulus languages cn / en / jp (Holm within "
                        "Exp 2); served after the English pairs, before the silent pairs"},
            {"block": "pair_ml L_vs_Lrev@<lang> (Experiment 2, --ml)", "per_participant": n_ml,
             "arms": "L vs Lrev (%s: LiltFace driven by the time-reversed prosody; L on the other side than in L_vs_E)"
                     % ML_TOKENS["Lrev"],
             "response": "CCR", "purpose": "P5: alignment with the voice, motion equated (Holm within Exp 2); exploratory: "
                                           "listener group x stimulus language (native vs non-native)"}]
    if opts.get("voice_check"):
        rows.append({"block": "voice_only (OPTION --voice-check)", "per_participant": served.get("voice_only", 0),
                     "arms": "audio only (black frame), normal + strong take of %s" % ", ".join(sel.get("voice_check_pairs") or []),
                     "response": "7-choice emotion + emotion INTENSITY 1-7 (CSV column `intensity`); no naturalness",
                     "purpose": "manipulation check of the P3 premise: strong heard as more intense than normal (within "
                                "participant, same voice pair); served LAST (needs patches/server_v6.patch + app_js_v6.patch)"})
    return rows


def design_md(man):
    L = ["# v6 perception study -- design table", "",
         "built %s%s%s" % (man["built"], "  **PROVISIONAL (stand-in media)**" if man["provisional"] else "",
                            "  **PLACEHOLDER SELECTION**" if man["selection"]["placeholder"] else ""), "",
         "| block | per participant | arms | response | purpose |", "|---|---|---|---|---|"]
    for r in man["design"]:
        L.append("| %s | %s | %s | %s | %s |" % (r["block"], r["per_participant"], r["arms"], r["response"], r["purpose"]))
    L += ["", "served per participant: %d %s; est %.1f min (builder formula); instructions show %d min (app.js formula)"
          % (man["n_served_per_participant"], man["served_per_block"], man["est_minutes"],
             client_minutes(man["n_served_per_participant"])), "", "## Hypotheses (freeze in the analysis plan)", ""]
    L += ["- **%s** %s" % (k, v) for k, v in man["hypotheses"].items()]
    opts = man.get("options") or {}
    L += ["", "options: --singles %s | --ml %s | --hr-normal %s | --voice-check %s (v6 final = --singles 6 --ml --voice-check)"
          % (opts.get("singles", 12), "ON" if opts.get("ml") else "off", "ON" if opts.get("hr_normal") else "off",
             "ON" if opts.get("voice_check") else "off")]
    L += ["", "## Voice pairs (slot = pair-group slot)", "",
          "| slot | voice pair | actor | emotion (response code) | silent pair | voice check | statement |",
          "|---|---|---|---|---|---|---|"]
    sel = man["selection"]
    vc = set(sel.get("voice_check_pairs") or [])
    for j, vp in enumerate(sel["voice_pairs"]):
        it = next((x for x in man["items"] if x["base_cell"] == vp and x["block"] == "single"), {})
        L.append("| %02d | %s | %s | %s (%s) | %s | %s | %s |" % (j, vp, actor_of(vp + "_n"), emo_of(vp + "_n"),
                                                                it.get("voice_emotion", ""), "yes" if vp in sel["mute_pairs"] else "",
                                                                "yes" if (vp in vc and opts.get("voice_check")) else "",
                                                                it.get("text_content", "")))
    L += ["", "practice: %s; catch: %s; selection from: %s" % (sel["practice"], sel["catch_cell"], sel["from"])]
    L += ["singles (%d): %s -- %s" % (len(man.get("single_pairs") or []), man.get("single_pairs"),
                                      SINGLES_RULE.get(len(man.get("single_pairs") or []), ""))]
    ml = man.get("ml") or {}
    if ml:
        L += ["", "## Experiment 2 cells (--ml)", "", ml["rule"], "",
              "px per rig unit %s (%s); threshold %.1f px; L-E = p95-p5 of clip(L) - clip(E) in speech; own = p95-p5 of "
              "the arm's clipped value" % (ml["px_per_rig_unit"], ml["px_from"], ml["min_px"]), "",
              "| cell | lang | emo | L-E inner px | L-E outer px | Lrev-E inner px | Lrev-E outer px | own inner E / L / Lrev (rig) "
              "| own outer E / L / Lrev (rig) | role |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in ml["table"]:
            if "error" in r:
                L.append("| %s | %s | %s | | | | | | | %s |" % (r["cell"], r["lang"], r["emotion"], r["role"]))
                continue
            L.append("| %s | %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f / %.2f / %.2f | %.2f / %.2f / %.2f | %s |" % (
                r["cell"], r["lang"], r["emotion"], r["LmE_brow_inner_px"], r["LmE_brow_outer_px"],
                r["LrevmE_brow_inner_px"], r["LrevmE_brow_outer_px"], r["E_brow_inner_rig"], r["L_brow_inner_rig"],
                r["Lrev_brow_inner_rig"], r["E_brow_outer_rig"], r["L_brow_outer_rig"], r["Lrev_brow_outer_rig"], r["role"]))
        L += ["", "selected medians: %s" % json.dumps(ml.get("summary"), ensure_ascii=False)]
    if opts.get("voice_check"):
        L += ["voice-check pairs: %s (from %s)" % (sel.get("voice_check_pairs"), sel.get("voice_check_from"))]
    if sel["problems"]:
        L += ["", "selection problems: %s" % sel["problems"]]
    rc = man["rotation_check"]
    L += ["", "rotation (server rule, %d participants): %s; same H side normal/strong %s; flipped in H_vs_R %s; silent side = AV strong side %s"
          % (rc["n_participants"], "OK" if rc["ok"] else rc["problems"][:3], rc["voice_pairs_same_H_side_normal_vs_strong"],
             rc["voice_pairs_H_side_flipped_in_H_vs_R"], rc["silent_H_side_same_as_AV_strong"])]
    return "\n".join(L) + "\n"


def stim_id(file):
    return hashlib.sha1(("lilt/" + file).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------- publish
def publish(out, stage, man, new_files):
    """New media in, manifest swapped atomically, then the files nothing references any more retired to _old/<ts>."""
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    arch = out / "_old" / ts
    if (out / "manifest.json").is_file():
        arch.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "manifest.json", arch / "manifest.json")
    for f in new_files:
        os.replace(stage / f, media / f)
    tmp = out / "manifest.json.tmp"
    with io.open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, out / "manifest.json")
    keep, moved, locked = set(new_files), 0, []
    for p in sorted(media.iterdir()):
        if p.name in keep or not p.is_file():
            continue
        try:
            (arch / "media").mkdir(parents=True, exist_ok=True)
            os.replace(p, arch / "media" / p.name)
            moved += 1
        except OSError:
            locked.append(p.name)
    return {"archived_to": str(arch) if arch.exists() else "", "retired_files": moved, "retire_failed": locked}


# ---------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="", help="link directory (manifest.json + media/); required unless --check")
    ap.add_argument("--provisional", action="store_true", help="stand-in held Danielle V9 renders, clearly marked")
    ap.add_argument("--require-complete", action="store_true",
                    help="build only when all 64 sources exist, are newer than their rigs and pass --newer-than; "
                         "otherwise --out is not touched")
    ap.add_argument("--check", action="store_true", help="selection + plan + source check only; writes nothing")
    ap.add_argument("--pool-set", default=str(HERE / "pool_set_v6.json"))
    ap.add_argument("--tok-E", default="")
    ap.add_argument("--tok-H", default="")
    ap.add_argument("--tok-R", default="")
    ap.add_argument("--cond", default="")
    ap.add_argument("--look", default=LOOK)
    ap.add_argument("--mute-pairs", default="", help="comma list of 6 voice pairs (rav<AA>_<emo>) overriding the rule")
    ap.add_argument("--catch-cell", default="", help="default: the first practice cell")
    ap.add_argument("--layer-label", default="", help="recorded in the manifest, e.g. h6 compose version / gains")
    ap.add_argument("--newer-than", default="", help="every H / R source must be newer than this file")
    ap.add_argument("--stage-dir", default=str(STUDY / "_stage"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-qc", action="store_true")
    ap.add_argument("--replace-same-names", action="store_true",
                    help="allow new files to overwrite same-named files the current manifest serves (same stimulus ids)")
    ap.add_argument("--allow-live-replace", action="store_true",
                    help="with --replace-same-names into the LIVE link: only when no participant has completed the served build")
    ap.add_argument("--hr-normal", action="store_true",
                    help="OPTION: add the pair block H_vs_R@normal (12 screens; 12 more R renders at the normal voices)")
    ap.add_argument("--voice-check", action="store_true",
                    help="OPTION: add the audio-only block voice_only, served last (needs patches/server_v6.patch + "
                         "app_js_v6.patch applied and a guarded restart before the link serves it)")
    ap.add_argument("--voice-check-pairs", default="",
                    help="file listing the %d voice pairs of the voice check (default stimuli/voice_check_wavs.txt when it "
                         "exists, else the silent-pair subset)" % N_VOICE_CHECK)
    ap.add_argument("--force-placeholder", action="store_true",
                    help="PROVISIONAL ONLY: build on the placeholder selection even though pool_set_v6.json carries a "
                         "(possibly incomplete) selection -- a flow test of the link, never study media")
    ap.add_argument("--singles", type=int, default=12, choices=SINGLES_CHOICES,
                    help="single screens per participant: 12 (base) or 6 (v6 final; a priori = the silent-subset voice "
                         "pairs, SINGLES_RULE)")
    ap.add_argument("--ml", action="store_true",
                    help="Experiment 2: block pair_ml, 18 screens (3 languages x ang/hap/sad x L_vs_E | L_vs_Lrev) from "
                         "the existing pool/mp4 LiltFace renders; cells by ML_RULE computed from pool/post (WSL only); "
                         "needs the server.py pair_ml ordering hunk before the link serves it")
    a = ap.parse_args(argv)

    sel = resolve_selection(a.pool_set, [c for c in a.mute_pairs.split(",") if c] or None, a.catch_cell or None,
                            {"E": a.tok_E, "H": a.tok_H, "R": a.tok_R}, a.cond)
    if a.force_placeholder:
        if not a.provisional:
            print("REFUSED: --force-placeholder is for --provisional flow tests only")
            return 2
        real = sel
        sel = placeholder_selection()
        sel["items"] = real["items"]
        sel["notes"] = ["--force-placeholder: the selection in pool_set_v6.json (%d voice pairs, practice %s, valid %s) "
                        "is NOT used" % (len(real["voice_pairs"]), real["practice"], real["valid"])] + real["problems"]
        sel["valid"], sel["problems"] = validate_selection(sel)
    vc_path = a.voice_check_pairs or None
    sel["voice_check_pairs"], sel["voice_check_from"], vc_probs = voice_check_pairs(
        sel["voice_pairs"], sel["mute_pairs"], vc_path, use_file=not a.force_placeholder)
    if a.voice_check and vc_probs:
        if a.force_placeholder:
            sel["notes"] = sel["notes"] + vc_probs
        else:
            sel["valid"] = False
            sel["problems"] = sel["problems"] + vc_probs
    opts = {"hr_normal": bool(a.hr_normal), "voice_check": bool(a.voice_check), "ml": bool(a.ml), "singles": int(a.singles)}
    sel["single_pairs"], sp_probs = single_pairs_for(sel, a.singles)
    if sp_probs:
        sel["valid"] = False
        sel["problems"] = sel["problems"] + sp_probs
    ml_rows, ml_info = [], {}
    sel["ml_cells"], sel["ml_meta"] = [], {}
    if opts["ml"]:
        ml_rows, ml_info = ml_measure(cond=sel["cond"])
        if not ml_rows:
            sel["valid"] = False
            sel["problems"] = sel["problems"] + ["--ml: the cell rule could not be computed (%s)" % ml_info.get("error")]
        else:
            sel["ml_cells"], ml_probs = ml_select(ml_rows)
            pool_items = ml_pool_items()
            sel["ml_meta"] = {c: ml_meta(pool_items, c) for c in sel["ml_cells"]}
            if ml_probs:
                sel["valid"] = False
                sel["problems"] = sel["problems"] + ml_probs
    print("selection from %s: %d voice pairs %s | practice %s | silent %s | catch %s | valid %s"
          % (sel["from"], len(sel["voice_pairs"]), sel["voice_pairs"], sel["practice"], sel["mute_pairs"],
             sel["catch_cell"], sel["valid"]))
    print("options: --singles %d | --ml %s | --hr-normal %s | --voice-check %s%s" % (
        opts["singles"], opts["ml"], opts["hr_normal"], opts["voice_check"],
        (" (pairs %s from %s)" % (sel["voice_check_pairs"], sel["voice_check_from"])) if opts["voice_check"] else ""))
    print("singles (%d): %s" % (len(sel["single_pairs"]), sel["single_pairs"]))
    if opts["ml"] and ml_rows:
        print("Exp 2 cell rule (px per rig unit %s from %s; threshold %.1f px on >= 1 brow channel):"
              % (ml_info.get("px_per_rig_unit"), ml_info.get("px_from"), ML_MIN_PX))
        print("  %-24s %-3s %-3s %9s %9s | %-17s | %-17s | %s" % ("cell", "lng", "emo", "L-E in px", "L-E out px",
                                                                 "own in E/L/Lrev", "own out E/L/Lrev", "role"))
        for r in ml_rows:
            if "error" in r:
                print("  %-24s %-3s %-3s ERROR %s | %s" % (r["cell"], r["lang"], r["emotion"], r["error"], r["role"]))
                continue
            print("  %-24s %-3s %-3s %9.2f %9.2f | %.2f/%.2f/%.2f    | %.2f/%.2f/%.2f    | %s" % (
                r["cell"], r["lang"], r["emotion"], r["LmE_brow_inner_px"], r["LmE_brow_outer_px"], r["E_brow_inner_rig"],
                r["L_brow_inner_rig"], r["Lrev_brow_inner_rig"], r["E_brow_outer_rig"], r["L_brow_outer_rig"],
                r["Lrev_brow_outer_rig"], r["role"]))
        print("  selected (%d): %s | medians %s" % (len(sel["ml_cells"]), sel["ml_cells"],
                                                    json.dumps(ml_summary(ml_rows, sel["ml_cells"]))))
    for note in sel["notes"] + sel["problems"]:
        print("NOTE:", note)
    out = norm_path(a.out) if a.out else None
    if not a.check and out is None:
        print("REFUSED: --out is required (the live link is D:/SPJ/pipeline/pilot/lilt_link)")
        return 2
    is_live = out is not None and out.resolve() == LIVE.resolve()
    if sel["placeholder"] and not a.provisional:
        print("REFUSED: no study selection in %s yet (v6_face.py select writes role/arms); only --provisional builds "
              "run on the placeholder" % a.pool_set)
        return 2
    if not sel["valid"]:
        print("REFUSED: the selection does not fit the design: %s" % sel["problems"])
        return 2
    if is_live and a.provisional:
        print("REFUSED: a provisional (stand-in) build never goes into the live link %s" % LIVE)
        return 2
    if is_live and not a.check and not a.require_complete:
        print("REFUSED: building into the live link requires --require-complete")
        return 2
    if is_live and not a.check and not a.newer_than:
        print("REFUSED: building into the live link requires --newer-than <the file the H/R layers were composed after>")
        return 2
    if is_live and a.replace_same_names and not a.allow_live_replace:
        print("REFUSED: --replace-same-names into the live link reuses stimulus ids for different media, so rows from both "
              "builds would pool silently. Only if no participant has completed the served build: add --allow-live-replace.")
        return 2

    stand_used, real_cells = {}, []

    def real_src(cell, arm):
        return POOL / "mp4" / ("%s__%s__%s__%s.mp4" % (cell, sel["tokens"][arm], sel["cond"], a.look))
    if a.provisional:
        cat = standin_catalog(a.look)
        if not cat:
            print("NOT BUILT: no held e10m10_d*_h %s render exists to stand in" % a.look)
            return 2
        standin_src, stand_used = make_standin_src(cat)
        real_cells = settled_real_cells(sel, a.look)             # English cells whose real renders are finished

        def src_of(cell, arm):
            return real_src(cell, arm) if cell in real_cells else standin_src(cell, arm)
        print("provisional: real English renders for %d cells %s; stand-ins for the rest" % (len(real_cells), real_cells))
    else:
        src_of = real_src

    need = required_sources(sel, src_of, opts["hr_normal"])
    need_ml = required_ml_sources(sel, a.look) if opts["ml"] else []
    absent = [str(p) for c, arm, p in need + need_ml if not p.is_file()]
    wavs = required_wavs(sel, opts["voice_check"])
    absent += [str(p) for c, p in wavs if not p.is_file()]
    stale = []
    if a.newer_than and not a.provisional:
        ref = norm_path(a.newer_than)
        if not ref.is_file():
            print("NOT BUILT: --newer-than file does not exist: %s" % ref)
            return 2
        stale = [str(p) for c, arm, p in need if arm in ("H", "R") and p.is_file() and p.stat().st_mtime <= ref.stat().st_mtime]
    if not a.provisional:                                  # every render (E included) must be newer than its own rig
        for c, arm, p in need:
            rig = POOL / "rigs" / ("%s__%s__%s.txt" % (c, sel["tokens"][arm], sel["cond"]))
            if p.is_file() and (not rig.is_file() or p.stat().st_mtime <= rig.stat().st_mtime):
                stale.append("%s (not newer than its rig %s)" % (p, rig.name))

    for c, arm, p in need_ml:                             # Exp 2 renders are the real clips in every mode
        rig = POOL / "rigs" / ("%s__%s__%s.txt" % (c, ML_TOKENS[arm], sel["cond"]))
        if p.is_file() and (not rig.is_file() or p.stat().st_mtime <= rig.stat().st_mtime):
            stale.append("%s (not newer than its rig %s)" % (p, rig.name))

    items = plan_items(sel, src_of, provisional=a.provisional, hr_normal=opts["hr_normal"], voice_check=opts["voice_check"],
                       ml=opts["ml"], src_ml=lambda c, arm: ml_src(c, arm, sel["cond"], a.look))
    rot = rotation_report(items, 12)
    rot24 = rotation_report(items, 24)
    leak = sorted({it["file"] for it in items if NAME_LEAK.search(it["file"])})
    print("plan: %d items, %d distinct sources (%d required renders%s%s), tokens %s, cond %s"
          % (len(items), len({s for it in items for s in it["op"]["inputs"]}), len(need),
             "" if opts["hr_normal"] else "; %d with --hr-normal" % (len(need) + N_VOICE_PAIRS),
             (" + %d wavs" % len(wavs)) if wavs else "", sel["tokens"], sel["cond"]))
    print("rotation check (server rule): 12 participants %s %s | 24 participants %s | same H side normal/strong %s | "
          "flipped in H_vs_R %s | max sound exposures per scored clip %s"
          % ("OK" if rot["ok"] else "PROBLEMS", rot["problems"][:3], "OK" if rot24["ok"] else rot24["problems"][:2],
             rot["voice_pairs_same_H_side_normal_vs_strong"], rot["voice_pairs_H_side_flipped_in_H_vs_R"],
             rot["max_sound_exposures_per_scored_clip"]))
    print("neutral file names: %s" % ("OK" if not leak else "LEAK %s" % leak[:3]))
    served0 = dict(collections.Counter(x["block"] for x in serve_like_server(items, 0)))
    exp = expected_served(**opts)
    print("SERVED PER PARTICIPANT (server rule, n_prior=0): %d %s | expected %d %s | %s | instructions show %d min"
          % (sum(served0.values()), served0, sum(exp.values()), exp, "OK" if served0 == exp else "MISMATCH",
             client_minutes(sum(exp.values()))))
    real_need = required_sources(sel, real_src, opts["hr_normal"])
    rq = render_queue_status()
    print("English renders: this design uses %d (E/H normal 24 + E/H/R strong 36 + practice E/H 4%s): %d present in "
          "pool/mp4 | render queue stimuli/render_list_v6.txt: %d listed, %d present, %d settled (>= %d s old), newest %s"
          % (len(real_need), " + R normal 12" if opts["hr_normal"] else "; the 12 queued R@normal renders are unused",
             sum(1 for c, arm, p in real_need if p.is_file()), rq["listed"], rq["present"], rq["settled"],
             RENDER_SETTLE_S, rq["newest"]))
    if a.provisional:
        print("  this provisional build: %d English sources real, %d stand-ins"
              % (sum(1 for c, arm, p in need if c in real_cells), sum(1 for c, arm, p in need if c not in real_cells)))
    if opts["ml"]:
        ml_abs = [p for c, arm, p in need_ml if not p.is_file()]
        print("Exp 2 sources (existing renders, no new renders needed): %d/%d present%s"
              % (len(need_ml) - len(ml_abs), len(need_ml), (", MISSING %s" % [p.name for p in ml_abs]) if ml_abs else ""))
        for c in sel["ml_cells"]:
            print("  %-24s %s" % (c, " ".join("%s:%s" % (arm, "ok" if ml_src(c, arm, sel["cond"], a.look).is_file() else "MISSING")
                                             for arm in ML_TOKENS)))
    if wavs:
        print("voice-check wavs: %d/%d present" % (sum(1 for c, p in wavs if p.is_file()), len(wavs)))
    print("sources: %d/%d present, %d stale" % (len(need) + len(need_ml) + len(wavs) - len(absent),
                                                 len(need) + len(need_ml) + len(wavs), len(stale)))
    if absent:
        print("MISSING (%d), e.g. %s" % (len(absent), [Path(x).name for x in absent[:4]]))
    if stale:
        print("STALE (%d), e.g. %s" % (len(stale), stale[:3]))
    if leak:
        print("REFUSED: file names expose the condition: %s" % leak[:3])
        return 2
    if a.check:
        return 0 if not (absent or stale) and rot["ok"] and rot24["ok"] else 2
    if a.require_complete and (absent or stale or not rot["ok"] or not rot24["ok"]):
        print("NOT BUILT: %d missing, %d stale, rotation %s -- %s untouched"
              % (len(absent), len(stale), "ok" if rot["ok"] and rot24["ok"] else "BAD", out))
        return 2
    partial = bool(absent)
    rot_built = None
    if absent:                                                            # partial build (never into the live link)
        keep_groups = {it["pick_group"] for it in items if all(Path(s).is_file() for s in it["op"]["inputs"])}
        items = [it for it in items if all(Path(s).is_file() for s in it["op"]["inputs"])]
        rot_built = rotation_report(items)
        print("PARTIAL build (NOT FOR PARTICIPANTS): %d items in %d groups; rotation of the built items %s"
              % (len(items), len(keep_groups), "ok" if rot_built["ok"] else "BAD"))

    ids = [stim_id(it["file"]) for it in items]
    if len(set(ids)) != len(ids) or len({it["file"] for it in items}) != len(items):
        print("REFUSED: duplicate file names / stimulus ids inside the plan")
        return 2
    live_files = set()
    if (out / "manifest.json").is_file():
        try:
            live_files = {x["file"] for x in json.load(io.open(out / "manifest.json", encoding="utf-8")).get("items", [])}
        except Exception:                                                # noqa: BLE001
            live_files = set()
    clash = sorted(live_files & {it["file"] for it in items})
    if clash and not a.replace_same_names:
        print("REFUSED: %d new files share a name (= stimulus id) with files the current manifest at %s serves, e.g. %s; "
              "pass --replace-same-names only for a rebuild of this same study" % (len(clash), out, clash[:2]))
        return 2

    # source durations: the arms of one cell must be equally long (the pair mux uses -shortest)
    src_dur, src_dur_warn = {}, []
    for c, arm, p in need + need_ml:
        if p.is_file():
            src_dur.setdefault(c, {})[arm] = B.probe_dur(p)
    for c, d in sorted(src_dur.items()):
        vals = [v for v in d.values() if v is not None]
        if len(vals) != len(d) or (vals and max(vals) - min(vals) > ARM_DUR_TOL):
            src_dur_warn.append({"cell": c, "durations": d})

    stage = norm_path(a.stage_dir) / time.strftime("%Y%m%d_%H%M%S")
    t0 = time.time()
    fails = execute(items, stage, a.workers)
    if fails and a.require_complete:
        print("NOT BUILT: %d ffmpeg failures, e.g. %s -- %s untouched; staging kept at %s" % (len(fails), fails[:2], out, stage))
        return 3
    built = [it for it in items if it.get("_ok")]
    # Loudness gate (as build_lilt_v2a): an audible clip must measure inside a gross window, and every arm of one cell
    # (the same wav through the same mux) must be equally loud. normal vs strong per voice pair is recorded, not gated.
    qc_bad, lufs, by_cell, by_vp, voice_lufs = [], [], {}, {}, {}
    for it in built:
        it["dur_s"] = B.probe_dur(stage / it["file"])
        if not a.no_qc:
            q = media_qc(stage / it["file"])
            it["media_qc"] = q
            want_audio = it["block"] != "pair_mute"
            if (q["w"], q["h"]) != (1120, 630) or q["audio"] != want_audio or q["exposing_tags"]:
                qc_bad.append({"file": it["file"], "qc": q})
            if want_audio and (q.get("lufs") is None or not LUFS_WINDOW[0] <= q["lufs"] <= LUFS_WINDOW[1]):
                qc_bad.append({"file": it["file"], "lufs": q.get("lufs"), "window": LUFS_WINDOW})
            if it["block"] == "voice_only" and (q.get("lufs") is None or abs(q["lufs"] - VOICE_LUFS) > VOICE_LUFS_TOL):
                qc_bad.append({"file": it["file"], "lufs": q.get("lufs"), "voice_target": VOICE_LUFS, "tol": VOICE_LUFS_TOL})
            if q.get("lufs") is not None:
                lufs.append(q["lufs"])
                # the arms of one cell share a wav through the same mux -> equally loud; the voice-only clip of a cell
                # is a different mux (two-pass loudnorm on the bare wav) and is judged against VOICE_LUFS above
                by_cell.setdefault(it["utterance"] + ("|voice" if it["block"] == "voice_only" else ""), []).append(q["lufs"])
                if it["block"] in ("single", "pair"):
                    by_vp.setdefault(it["base_cell"], {}).setdefault(it["voice_level"], []).append(q["lufs"])
                if it["block"] == "voice_only":
                    voice_lufs[it["utterance"]] = q["lufs"]
    for key, vals in sorted(by_cell.items()):
        if max(vals) - min(vals) > ARM_LUFS_SPREAD:
            qc_bad.append({"cell": key, "lufs_spread_across_arms": [min(vals), max(vals)]})
    n_vs_s = {vp: {lv: round(sum(v) / len(v), 2) for lv, v in d.items()} for vp, d in sorted(by_vp.items())}
    for it in items:
        it.pop("op", None)
        it.pop("_ok", None)
        if a.provisional:
            ins = [x for x in ([it.get("source")] + it.get("sources", [])) if x]
            real = it["block"] == ML_BLOCK or it["utterance"] in real_cells
            it["stand_in"] = [] if (real or it["block"] == "voice_only") else sorted({Path(x).name for x in ins})
            it["real_render"] = bool(real) if it["block"] != "voice_only" else None
    if qc_bad and a.require_complete:
        print("NOT BUILT: media QC failed on %d files, e.g. %s -- %s untouched; staging kept at %s" % (len(qc_bad), qc_bad[:2], out, stage))
        return 3

    groups = {}
    for it in built:
        groups.setdefault(it["pick_group"], []).append(it)
    served = {}
    for g, arms in groups.items():
        served[arms[0]["block"]] = served.get(arms[0]["block"], 0) + 1
    dur_warn = []
    for vp in sel["voice_pairs"]:
        ds = [it["dur_s"] for it in built if it["block"] == "single" and it["base_cell"] == vp and it.get("dur_s")]
        if ds and max(ds) - min(ds) > ARM_DUR_TOL:
            dur_warn.append({"voice_pair": vp, "single_durations": ds})

    purpose = ("PROVISIONAL FLOW TEST -- NOT STUDY MEDIA: English clips are stand-ins (an existing held Danielle V9 render "
               "of a v4 cell, never RAVDESS, never human brows) except the English cells whose real renders were "
               "finished at build time (manifest real_render_cells)%s; metadata are the study cells'"
               % ("; the voice-pair SELECTION IS A PLACEHOLDER" if sel["placeholder"] else "") if a.provisional else
               "Does the upper face follow the voice (v6): EmoFace (E) vs E + the actor's own brow motion (H) vs E + the "
               "same brow track time-reversed (R); RAVDESS female voices at normal and strong intensity; Danielle V9, %s%s"
               % (sel["cond"], (", layer " + a.layer_label) if a.layer_label else ""))
    if partial:
        purpose = "PARTIAL -- NOT FOR PARTICIPANTS (%d sources missing): %s" % (len(absent), purpose)
    if opts["hr_normal"]:
        purpose += "; OPTION --hr-normal: + H_vs_R@normal (12 pair screens)"
    if opts["voice_check"]:
        purpose += ("; OPTION --voice-check: + audio-only voice check (12 screens, served last; server.py / app.js must "
                    "carry patches/server_v6.patch + app_js_v6.patch)")
    if opts["singles"] != N_VOICE_PAIRS:
        purpose += "; singles %d per participant (a priori: the silent-subset voice pairs)" % opts["singles"]
    if opts["ml"]:
        purpose += ("; EXPERIMENT 2 (--ml, 18 screens): LiltFace L (%s) vs EmoFace E and vs Lrev (%s) on %d existing "
                    "renders' cells in three stimulus languages %s, block pair_ml served after the English pairs and "
                    "before the silent pairs (server.py needs the pair_ml ordering hunk)"
                    % (ML_TOKENS["L"], ML_TOKENS["Lrev"], len(sel["ml_cells"]), sel["ml_cells"]))
    purpose += "; %d screens per participant" % sum(expected_served(**opts).values())
    hyps = dict(HYPOTHESES)
    if opts["ml"]:
        hyps.update(ML_HYPOTHESES)
    if opts["hr_normal"]:
        hyps["P2b"] = OPTIONAL_HYPOTHESES["P2b"]
    if opts["voice_check"]:
        hyps["MC2"] = OPTIONAL_HYPOTHESES["MC2"]
    man = {
        "built": "build_lilt_v6.py " + time.strftime("%Y-%m-%d %H:%M"), "builder": "build_lilt_v6.py",
        "study": "v6", "label": "Face comparison (separate link)", "purpose": purpose,
        "design_variant": ("full" + ("+singles%d" % opts["singles"] if opts["singles"] != N_VOICE_PAIRS else "")
                           + ("+ml" if opts["ml"] else "") + ("+hr_normal" if opts["hr_normal"] else "")
                           + ("+voice_check" if opts["voice_check"] else "")),
        "options": opts, "expected_served": expected_served(**opts),
        "partial": partial, "provisional": bool(a.provisional), "look": a.look, "cond": sel["cond"],
        "tokens": sel["tokens"], "layer_label": a.layer_label, "newer_than": a.newer_than,
        "selection": selection_summary(sel), "cells": sel["voice_pairs"], "practice_cells": sel["practice"],
        "mute_pairs": sel["mute_pairs"], "catch_cell": sel["catch_cell"],
        "voice_check_pairs": sel["voice_check_pairs"] if opts["voice_check"] else [], "hypotheses": hyps,
        "contrasts": list(av_contrasts(opts["hr_normal"])), "mute_contrast": MUTE_CONTRAST,
        "voice_check_arm": VOICE_CHECK_ARM if opts["voice_check"] else "",
        "server_patch_required": bool(opts["voice_check"] or opts["ml"]),
        "server_patch_note": ("pair_ml is served after pair + pair_catch and before pair_mute only with patches/server_v6.patch "
                              "(17:36 version) applied; the unpatched server.py drops block pair_ml entirely" if opts["ml"] else ""),
        "singles": opts["singles"], "single_pairs": sel["single_pairs"], "singles_rule": SINGLES_RULE[opts["singles"]],
        "ml": ({"rule": ML_RULE, "block": ML_BLOCK, "tokens": ML_TOKENS, "cond": sel["cond"], "look": a.look,
                "contrasts": ["%s@%s" % (c0, lg) for c0 in ML_CONTRASTS for lg in ML_LANGS], "cells": sel["ml_cells"],
                "min_px": ML_MIN_PX, "px_per_rig_unit": ml_info.get("px_per_rig_unit"), "px_from": ml_info.get("px_from"),
                "table": ml_rows, "summary": ml_summary(ml_rows, sel["ml_cells"]),
                "sources_present": {"%s|%s" % (c, arm): p.is_file() for c, arm, p in need_ml},
                "amplitude_note": "L's in-speech brow movement (own p95-p5, medians in summary) is compared with the English "
                                  "RAVDESS human reference 0.48 inner / 0.35 outer rig units (main-session measurement)"}
               if opts["ml"] else None),
        "real_render_cells": real_cells,
        "design": design_table(served, sel, opts), "served_per_block": served, "n_items": len(built),
        "n_served_per_participant": sum(served.values()), "est_minutes": est_minutes(built, served),
        "client_minutes": client_minutes(sum(served.values())),
        "rotation_check": rot, "rotation_check_24": {k: v for k, v in rot24.items() if k != "per_participant"},
        "rotation_check_built": rot_built,
        "media_qc": None if a.no_qc else {"lufs_min": min(lufs) if lufs else None, "lufs_max": max(lufs) if lufs else None,
                                           "lufs_window": LUFS_WINDOW, "arm_lufs_spread_max": ARM_LUFS_SPREAD,
                                           "lufs_by_cell": {k: [min(v), max(v)] for k, v in sorted(by_cell.items())},
                                           "lufs_normal_vs_strong": n_vs_s,
                                           "note_normal_vs_strong": "the pool_render mux loudness-normalises each clip, so "
                                                                    "normal and strong play at about the same loudness",
                                           "lufs_voice_only": voice_lufs, "voice_only_target": [VOICE_LUFS, VOICE_LUFS_TOL],
                                           "n_checked": len(built), "n_problems": len(qc_bad), "problems": qc_bad[:10]},
        "duration_warnings": dur_warn, "source_duration_warnings": src_dur_warn,
        "media_note": "v4 conventions (build_lilt_icassp.py): singles SINGLE_CROP 1120x630, pairs PAIR_CROP 560x630 x2 "
                      "hstack with the left input's audio (the same wav both sides), silent pairs no audio track, catch = "
                      "H vs the same clip frozen on its first frame; audio = pool_render.py mux, loudnorm -23 LUFS; "
                      "file names neutral (v6_<block>_<code>_<k>.mp4, k=1 H right / k=2 H left; catch k=1 moving left); "
                      "voice_only (option) = the RAVDESS wav under a black 1120x630 frame, two-pass loudnorm I=-23 LUFS, "
                      "aac 192k, no metadata (v6_voice_<code>_1.mp4)",
        "missing_sources": absent, "stale_sources": stale, "build_failures": fails,
        "required_renders": len(need), "required_wavs": [str(p) for c, p in wavs],
        "stand_in_map": {"%s|%s" % k: v for k, v in stand_used.items()} if a.provisional else {},
        "items": built,
    }
    pub = publish(out, stage, man, [it["file"] for it in built])
    if not is_live:
        with io.open(out / "design_table.md", "w", encoding="utf-8") as fh:
            fh.write(design_md(man))
    try:
        stage.rmdir()
    except OSError:
        pass
    print("items %d | served per participant %d %s | est %.1f min (instructions show %d) | %.0f s"
          % (len(built), man["n_served_per_participant"], served, man["est_minutes"], man["client_minutes"], time.time() - t0))
    if not a.no_qc:
        print("media QC: LUFS %s..%s, problems %d %s" % (man["media_qc"]["lufs_min"], man["media_qc"]["lufs_max"],
                                                          len(qc_bad), qc_bad[:2]))
    if dur_warn or src_dur_warn:
        print("duration warnings: %s %s" % (dur_warn[:3], src_dur_warn[:3]))
    if fails:
        print("ffmpeg failures: %s" % fails[:4])
    print("published", out / "manifest.json", pub)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

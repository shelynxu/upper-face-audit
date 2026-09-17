# -*- coding: utf-8 -*-
"""build_lilt_icassp.py -- the LiltFace online study as the /root/lilt manifest.

Face: Danielle, V9 look (pool_render.py --look danielle_v9). Audio: validated ACTOR recordings with long,
intonation-rich utterances (ESD 0002 parallel sentences, JVNV F2, RAVDESS A02 strong takes as two-statement
passages). One LiltFace configuration (--lilt, a pool_rigs.py token) against EmoFace as shipped.

Versions of each utterance (all through the identical post-step, same seed):
    E     EmoFace as shipped (d5, eye x1.8 / mouth x1.3)
    g1    LiltFace at gain 1 (below the ~20 % motion JND)    g2  gain 2 (the visible ceiling of this rig; g3 = g2 on video)
    rev   LiltFace g2 driven by the TIME-REVERSED prosody of the same wav: the same amount of upper-face
          motion, alignment with the voice destroyed (the GENEA 2023 mismatch control, adapted so the
          mouth stays synchronised)
    Ed2 / g2d2   EmoFace and LiltFace g2 at expression degree 2 (0.4 x the emotional deviation; the /faceval
          rung at 58 % recognition) -- the weak-expression end of the real-data floor/ceiling

Per participant (46 screens, ~14 min):
  block single      12  one version of each utterance (pick_group, 5 arms, rotates across participants):
                        7-choice emotion + vividness of the facial movement 1-7 + naturalness 1-7 (ACR); no radar.
                        (vividness travels in the CSV column `intensity`; see analyse_lilt.py)
  block practice     2  LiltFace g2 vs EmoFace on two other ESD sentences, same audio (unscored)
  block pair        24  side by side, one shared audio, 7-point Comparison Category Rating -3..+3
                        (ITU-T P.910 CCR / the CMOS of P.800) on "whose brow/eye/head movement follows the
                        voice's intonation and emphasis better":
                          pairA_<utt>  g2 vs E  or  g2 vs rev   (left/right counterbalanced, 4 arms)
                          pairB_<utt>  g1 vs E  or  g2 vs g1    (4 arms)
                     4  deg_<utt>   g2d2 vs Ed2 (2 arms)
  block pair_mute    6  g2 vs E WITHOUT SOUND, "which face's movement is more lively?" (visibility check, last block)
  block pair_catch   1  g2 moving vs the same clip frozen; its own question ("which face is moving?"), left / right buttons
  pairA for one utterance per language serves BOTH arms (g2 vs E and g2 vs rev) to every participant: 27 pairs in all
Pairs are face-cropped at 1:1 (560x630 per side on the 1120x630 stage), singles re-framed to the face (1120x630).
Nothing is pooled with the corpus gate: own directory, ids namespaced lilt/.

--provisional builds the same design from the first-screen Hadley renders (screen/mp4) with PLACEHOLDER
version slots, so the flow and the response screens can be tested before the pool renders exist.

RUN: cd ${PROJECT_ROOT}/pipeline/prosody_ladder && python build_lilt_icassp.py --lilt r5_lilt_nonod [--provisional]
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPJ = HERE.parents[1]
POOL = HERE / "pool"
OUT = SPJ / "pipeline" / "pilot" / "lilt_link"

# Selected from the 2026-09-13 audition verdicts (artifact "verdicts"), which superseded every earlier pick.
# Mandarin: the ESD 0002 parallel sentence 我总是控制不了它 in all five emotions (surprise too, for one text per
# language); English: the shipped anger / happy / sad short sentences; Japanese: the shipped anger / sad short
# sentences and the T3 surprise clone, plus the shipped happy short sentence in the slot freed by the duplicate
# Mandarin happy pick.
UTTERANCES = ["cn_esd0002_short3_ang", "cn_esd0002_short3_hap", "cn_esd0002_short3_sad", "cn_esd0002_short3_sur",
              "cn_esd0002_short3_neu", "en_ang_short_sentence", "en_hap_short_sentence", "en_sad_short_sentence",
              "jp_ang_short_sentence", "jp_sad_short_sentence", "jp_jvnvF2_long_sur", "jp_hap_short_sentence"]
DEG_UTT = ["cn_esd0002_short3_ang", "cn_esd0002_short3_hap", "cn_esd0002_short3_sad", "cn_esd0002_short3_sur"]
CATCH_UTT = ["cn_esd0002_short3_neu"]                                         # ONE catch per participant
BOTH_A_UTT = ["cn_esd0002_short3_hap", "en_ang_short_sentence", "jp_sad_short_sentence"]   # both pairA arms served
PRACTICE_UTT = ["cn_esd0002_short2_ang", "cn_esd0002_short2_sur"]           # a different ESD text from the study
MUTE_UTT = ["cn_esd0002_short3_hap", "cn_esd0002_short3_ang", "en_ang_short_sentence", "en_hap_short_sentence",
            "jp_ang_short_sentence", "jp_jvnvF2_long_sur"]                    # silent g2-vs-E pairs, last block

PROV_MAP = {"E": "before", "g1": "r3_prom_up", "g2": "r4_f0", "g3": "r3_prom", "rev": "r3_e2v", "Ed2": "before", "g2d2": "r4_f0"}
PROV_CELLS = ["cn_hap_long_sentence", "cn_ang_short_sentence", "jp_sad_short_sentence", "en_ang_short_sentence",
              "jp_hap_long_sentence", "en_hap_long_sentence"]


def sh(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def probe_dur(p):
    r = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)])
    try:
        return round(float(r.stdout.strip()), 3)
    except ValueError:
        return None


# Face framing: two full 1920x1080 frames side by side made the faces too small to see the
# brows). Danielle V9 renders put the head centre at y ~ 588 with the hair top at ~ 254 (pool_render framing gate), so:
# The app's stage is 1120 x 630 CSS px (style.css --stim-w/--stim-h), so these crops show the face at 1:1 render
# pixels instead of being downscaled (judge check: +29 % on-screen displacement for every arm at no cost).
PAIR_CROP = "crop=560:630:680:235"        # each side: face + jaw; 1120 x 630 once the two are stacked (16:9, 1:1)
SINGLE_CROP = "crop=1120:630:400:245"     # single face at 1:1


def sbs(left, right, out):
    """Left | right, unlabelled, face-cropped, one shared audio (both muxed from the same wav through the same loudnorm)."""
    vf = ("[0:v]%s[l];[1:v]%s[r];[l][r]hstack=inputs=2,"
          "drawbox=x=iw/2-2:y=0:w=4:h=ih:color=black@1:t=fill[v]" % (PAIR_CROP, PAIR_CROP))
    r = sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(left), "-i", str(right), "-filter_complex", vf,
            "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-c:a", "aac", "-movflags", "+faststart", "-shortest", str(out)])
    return out.is_file() and out.stat().st_size > 10000, (r.stderr or "")[-160:]


def sbs_mute(left, right, out):
    """sbs() without the audio track: the silent block asks which movement is more lively, not which fits the voice."""
    vf = ("[0:v]%s[l];[1:v]%s[r];[l][r]hstack=inputs=2,"
          "drawbox=x=iw/2-2:y=0:w=4:h=ih:color=black@1:t=fill[v]" % (PAIR_CROP, PAIR_CROP))
    r = sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(left), "-i", str(right), "-filter_complex", vf,
            "-map", "[v]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
            "-shortest", str(out)])
    return out.is_file() and out.stat().st_size > 10000, (r.stderr or "")[-160:]


def still_of(src, out):
    """The same clip with the face frozen on its first visible frame, same audio (the catch foil: the catch screen
    asks its own question, 'one face is frozen -- which face is moving?', with left/right buttons)."""
    png = out.with_suffix(".png")
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-frames:v", "1", str(png)])
    r = sh(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "60", "-i", str(png), "-i", str(src),
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-shortest", str(out)])
    png.unlink(missing_ok=True)
    return out.is_file() and out.stat().st_size > 10000, (r.stderr or "")[-160:]


def crop_single(src, out):
    """A single face clip re-framed to the face (SINGLE_CROP), audio copied."""
    r = sh(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", SINGLE_CROP, "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out)])
    return out.is_file() and out.stat().st_size > 10000, (r.stderr or "")[-160:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lilt", default="r5_lilt_nonod", help="pool_rigs.py strategy token of the LiltFace configuration")
    ap.add_argument("--amp", default="a1", help="amplification rung: a1 = shipped x1.8/x1.3 unheld; e11m10 = eye x1.1, "
                    "mouth x1.0 (1:1 = e10m10), expression held through the silent pads (conds <amp>_d<main>_h / <amp>_d<weak>_h)")
    ap.add_argument("--deg-main", type=int, default=5, help="EmoFace degree of the main arms (5 = native)")
    ap.add_argument("--deg-weak", type=int, default=2, help="EmoFace degree of the weak-face pair block (2 = /faceval 58%% rung)")
    ap.add_argument("--look", default="danielle_v9")
    ap.add_argument("--provisional", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--require-complete", action="store_true", help="build only when every source render exists")
    a = ap.parse_args()
    out = Path(a.out)
    media = out / "media"

    pool_man = json.load(io.open(POOL / "manifest.json", encoding="utf-8")) if (POOL / "manifest.json").is_file() else {"items": {}}
    if a.provisional:
        utts = PROV_CELLS
        deg_utts, catch_utts, prac_utts = utts[:2], utts[2:4], utts[4:6]

        def src(u, v):
            return HERE / "screen" / "mp4" / ("%s__%s.mp4" % (u, PROV_MAP[v]))
        character = "Hadley"
    else:
        utts, deg_utts, catch_utts, prac_utts = UTTERANCES, DEG_UTT, CATCH_UTT, PRACTICE_UTT
        d5, d2 = (("a1_d%d" % a.deg_main, "a1_d%d" % a.deg_weak) if a.amp == "a1"
                  else ("%s_d%d_h" % (a.amp, a.deg_main), "%s_d%d_h" % (a.amp, a.deg_weak)))   # names kept: main / weak rung
        tok = {"E": ("before", d5), "g1": (a.lilt + "-g1", d5), "g2": (a.lilt + "-g2", d5),
               "rev": (a.lilt + "_rev-g2", d5), "Ed2": ("before", d2), "g2d2": (a.lilt + "-g2", d2)}

        def src(u, v):
            t, c = tok[v]
            return POOL / "mp4" / ("%s__%s__%s__%s.mp4" % (u, t, c, a.look))
        character = "Danielle"

    need = ([(u, v) for u in utts for v in ("E", "g1", "g2", "rev")] + [(u, v) for u in deg_utts for v in ("Ed2", "g2d2")]
            + [(u, "g2") for u in catch_utts] + [(u, v) for u in prac_utts for v in ("E", "g2")])
    absent = sorted({str(src(u, v)) for u, v in need if not src(u, v).is_file()})
    if absent and a.require_complete:
        print("NOT BUILT: %d sources missing, the live manifest is untouched; e.g. %s" % (len(absent), absent[:3]))
        return 2
    if (out / "manifest.json").is_file() or media.is_dir():          # keep the previous build, never delete it
        old = out / "_old" / time.strftime("%Y%m%d_%H%M%S")
        old.mkdir(parents=True, exist_ok=True)
        for p in ("manifest.json", "media"):
            if (out / p).exists():
                shutil.move(str(out / p), str(old / p))
    media.mkdir(parents=True, exist_ok=True)

    def meta_of(u):
        rec = pool_man["items"].get(u, {})
        if a.provisional or not rec:
            lang, emo = u.split("_")[0], u.split("_")[1]
            return {"stim_language": lang, "label": emo, "text": "", "speaker": "", "length": "", "source": "clone"}
        return {"stim_language": rec["lang"], "label": rec["label"], "text": rec.get("text", ""),
                "speaker": u.split("_")[1], "length": rec.get("length", ""), "source": rec.get("source", "")}

    items, missing, log = [], [], []

    def base(u, block, presentation):
        m = meta_of(u)
        return {"block": block, "presentation": presentation, "stim_language": m["stim_language"],
                "text_type": m["length"], "text_content": m["text"], "character": character,
                "char_group": "ambiguous" if character == "Danielle" else "caucasian",
                "tts_model": "actor" if m["source"] == "actor" else "clone", "voice_id": m["speaker"],
                "is_static": 0, "utterance": u, "provisional": bool(a.provisional)}

    def have(u, v):
        p = src(u, v)
        if not p.is_file():
            missing.append(str(p))
            return None
        return p

    # ------------------------------------------------------------------ singles (5 arms)
    for u in utts:
        m = meta_of(u)
        for v in ("E", "g1", "g2", "rev"):
            p = have(u, v)
            if not p:
                continue
            f = "single_%s__%s.mp4" % (u, v)
            ok, why = crop_single(p, media / f)
            log.append({"file": f, "ok": ok, "why": why})
            if not ok:
                continue
            it = base(u, "single", "AV")
            it.update({"id": f[:-4], "file": f, "dur_s": probe_dur(media / f), "voice_emotion": m["label"],
                       "face_emotion": m["label"], "congruency": "congruent", "renderer": v, "face_motion": p.stem,
                       "face_deg": "d%d" % a.deg_main, "cue_manip": "", "voice_arm": "", "pick_group": "single_" + u, "source": str(p)})
            items.append(it)

    # ------------------------------------------------------------------ pairs
    def add_pair(u, A, B, group, arm_label, face_deg="d%d" % a.deg_main):
        pa, pb = have(u, A), have(u, B)
        if not (pa and pb):
            return
        for L, R in ((A, B), (B, A)):
            f = "pair_%s__%s_vs_%s.mp4" % (u, L, R)
            ok, why = sbs(src(u, L), src(u, R), media / f)
            log.append({"file": f, "ok": ok, "why": why})
            if not ok:
                continue
            it = base(u, "pair", "AV_pair")
            it.update({"id": f[:-4], "file": f, "dur_s": probe_dur(media / f), "voice_emotion": "", "face_emotion": "",
                       "congruency": "", "renderer": "pair", "face_motion": "%s|%s" % (src(u, L).stem, src(u, R).stem),
                       "face_deg": face_deg, "cue_manip": "L:%s|R:%s" % (L, R), "voice_arm": arm_label,
                       "pick_group": group, "source": ""})
            items.append(it)

    for u in utts:
        if u in BOTH_A_UTT:                       # both main-contrast arms to every participant (+3 pairs)
            add_pair(u, "g2", "E", "pairA_" + u + "_E", "g2_vs_E")
            add_pair(u, "g2", "rev", "pairA_" + u + "_rev", "g2_vs_rev")
        else:
            add_pair(u, "g2", "E", "pairA_" + u, "g2_vs_E")
            add_pair(u, "g2", "rev", "pairA_" + u, "g2_vs_rev")
        add_pair(u, "g1", "E", "pairB_" + u, "g1_vs_E")
        add_pair(u, "g2", "g1", "pairB_" + u, "g2_vs_g1")      # the dose step itself (g3 = g2 on video: retired)
    for u in deg_utts:
        add_pair(u, "g2d2", "Ed2", "deg_" + u, "d2:g2_vs_E", face_deg="d%d" % a.deg_weak)   # arm key kept for analyse_lilt

    # ------------------------------------------------------------------ practice: g3 vs E (the real contrast, large, unscored)
    for u in prac_utts:
        for A, B in (("g2", "E"), ("E", "g2")):
            pa, pb = have(u, A), have(u, B)
            if not (pa and pb):
                continue
            f = "practice_%s__%s_vs_%s.mp4" % (u, A, B)   # g2 vs E
            ok, why = sbs(pa, pb, media / f)
            log.append({"file": f, "ok": ok, "why": why})
            if not ok:
                continue
            it = base(u, "practice", "AV_pair")
            it.update({"id": f[:-4], "file": f, "dur_s": probe_dur(media / f), "voice_emotion": "", "face_emotion": "",
                       "congruency": "", "renderer": "pair", "face_motion": "%s|%s" % (pa.stem, pb.stem), "face_deg": "d%d" % a.deg_main,
                       "cue_manip": "L:%s|R:%s" % (A, B), "voice_arm": "practice:g2_vs_E",
                       "pick_group": "practice_" + u, "source": ""})
            items.append(it)
    # ------------------------------------------------------------------ silent block: g2 vs E without sound, last
    for u in (utts[:6] if a.provisional else MUTE_UTT):
        pa, pb = have(u, "g2"), have(u, "E")
        if not (pa and pb):
            continue
        for L, R in (("g2", "E"), ("E", "g2")):
            f = "mute_%s__%s_vs_%s.mp4" % (u, L, R)
            ok, why = sbs_mute(src(u, L), src(u, R), media / f)
            log.append({"file": f, "ok": ok, "why": why})
            if not ok:
                continue
            it = base(u, "pair_mute", "AV_pair")
            it.update({"id": f[:-4], "file": f, "dur_s": probe_dur(media / f), "voice_emotion": "", "face_emotion": "",
                       "congruency": "", "renderer": "pair", "face_motion": "%s|%s" % (src(u, L).stem, src(u, R).stem),
                       "face_deg": "d%d" % a.deg_main, "cue_manip": "L:%s|R:%s" % (L, R), "voice_arm": "mute:g2_vs_E",
                       "pick_group": "mute_" + u, "source": ""})
            items.append(it)
    # ------------------------------------------------------------------ catch: g2 moving vs the same clip frozen
    for u in catch_utts:
        mov = have(u, "g2")
        if not mov:
            continue
        stl = media / ("_still_%s__g2.mp4" % u)
        ok, why = still_of(mov, stl)
        log.append({"file": stl.name, "ok": ok, "why": why})
        if not ok:
            continue
        for L, R, tag in ((mov, stl, "mov_vs_still"), (stl, mov, "still_vs_mov")):
            f = "pair_catch_%s__%s.mp4" % (u, tag)
            ok, why = sbs(L, R, media / f)
            log.append({"file": f, "ok": ok, "why": why})
            if not ok:
                continue
            it = base(u, "pair_catch", "AV_pair")
            it.update({"id": f[:-4], "file": f, "dur_s": probe_dur(media / f), "voice_emotion": "", "face_emotion": "",
                       "congruency": "", "renderer": "pair", "face_motion": "g2", "face_deg": "d%d" % a.deg_main,
                       "cue_manip": "L:g2|R:still" if tag == "mov_vs_still" else "L:still|R:g2",
                       "voice_arm": "correct:" + ("left" if tag == "mov_vs_still" else "right"),
                       "pick_group": "pair_catch_%s" % u, "source": ""})
            items.append(it)
        stl.unlink(missing_ok=True)

    groups = {}
    for it in items:
        groups.setdefault(it["pick_group"], []).append(it)
    by_block = {}
    for g, arms in groups.items():
        by_block.setdefault(arms[0]["block"], []).append(g)
    served = {b: len(gs) for b, gs in by_block.items()}
    n_pair_main = len([g for g in by_block.get("pair", []) if not g.startswith("deg_")])
    n_deg = len([g for g in by_block.get("pair", []) if g.startswith("deg_")])

    def mean_dur(block):
        ds = [it["dur_s"] or 0 for it in items if it["block"] == block]
        return sum(ds) / len(ds) if ds else 0.0

    sec = (served.get("single", 0) * (mean_dur("single") + 14) + served.get("pair", 0) * (mean_dur("pair") + 6)
           + served.get("pair_mute", 0) * (mean_dur("pair_mute") + 6)
           + served.get("pair_catch", 0) * (mean_dur("pair_catch") + 5) + served.get("practice", 0) * (mean_dur("practice") + 10)
           + 240)
    design = [
        {"block": "single", "per_participant": served.get("single", 0), "arms": "E | g1 | g2 | rev (rotated)",
         "response": "7-choice emotion + vividness of the facial movement 1-7 + naturalness of the facial movement 1-7 (ACR); "
                     "no radar. NOTE: vividness is stored in the CSV column `intensity`.",
         "purpose": "first-exposure vividness (benefit) and naturalness (cost) MOS per version; recognition guard (does LiltFace change the emotion read?)"},
        {"block": "practice", "per_participant": served.get("practice", 0), "arms": "g2 vs E, left/right counterbalanced",
         "response": "CCR -3..+3", "purpose": "teach the comparison scale on the real contrast; unscored"},
        {"block": "pair (A)", "per_participant": len([g for g in by_block.get("pair", []) if g.startswith("pairA_")]),
         "arms": "g2 vs E | g2 vs rev, left/right counterbalanced", "response": "CCR -3..+3 (CMOS)",
         "purpose": "H1 appropriateness gain over EmoFace; H2 prosody-locking (same motion, broken alignment)"},
        {"block": "pair (B)", "per_participant": len([g for g in by_block.get("pair", []) if g.startswith("pairB_")]),
         "arms": "g1 vs E | g2 vs g1", "response": "CCR -3..+3 (CMOS)",
         "purpose": "H3 dose: g1 is below the motion JND (Hyde 2013), g2 above it; g3 is retired (identical to g2 on video)"},
        {"block": "pair (deg)", "per_participant": n_deg, "arms": "g2d2 vs Ed2", "response": "CCR -3..+3 (CMOS)",
         "purpose": "H4 interaction with expression strength (weak face, d2 = 58 % /faceval rung)"},
        {"block": "pair_mute", "per_participant": served.get("pair_mute", 0), "arms": "g2 vs E, NO SOUND, left/right counterbalanced",
         "response": "CCR -3..+3 on 'which face's movement is more lively and expressive?'",
         "purpose": "visibility check: is the motion difference seen at all, independent of fit to the voice (last block)"},
        {"block": "pair_catch", "per_participant": served.get("pair_catch", 0), "arms": "g2 moving vs the same clip frozen",
         "response": "left / right: 'one face is frozen -- which face is moving?'", "purpose": "attention (one per participant)"},
    ]
    man = {"built": "build_lilt_icassp.py " + time.strftime("%Y-%m-%d %H:%M"),
           "purpose": ("PROVISIONAL flow test: first-screen Hadley renders in placeholder version slots "
                       "(E=before g1=r3_prom_up g2=r4_f0 g3=r3_prom rev=r3_e2v)" if a.provisional else
                       "LiltFace study: EmoFace vs LiltFace (%s) gain ladder, reversed-prosody control and d2 on Danielle V9" % a.lilt),
           "label": "LiltFace comparison (separate link)", "lilt": a.lilt, "amp": a.amp, "look": a.look if not a.provisional else "hadley",
           "provisional": bool(a.provisional), "utterances": utts, "design": design,
           "n_items": len(items), "n_served_per_participant": sum(served.values()),
           "est_minutes": round(sec / 60.0, 1),
           "media_note": "singles = pool_render.py mux (60 fps, loudnorm -23 LUFS); pairs = hstack of two such clips, audio of the left input (the same wav both sides)",
           "missing_sources": missing, "build_log": [x for x in log if not x["ok"]], "items": items}
    tmp = out / "manifest.json.tmp"
    json.dump(man, io.open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, out / "manifest.json")
    print("items %d | served per participant %d %s | est %.1f min" % (len(items), sum(served.values()), served, man["est_minutes"]))
    print("pair groups: %d main + %d deg" % (n_pair_main, n_deg))
    if missing:
        print("MISSING sources (%d), e.g. %s" % (len(missing), missing[:4]))
    bad = [x for x in log if not x["ok"]]
    if bad:
        print("ffmpeg failures: %s" % bad[:4])
    print("written", out / "manifest.json")


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)

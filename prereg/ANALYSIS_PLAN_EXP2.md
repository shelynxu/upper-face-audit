# ANALYSIS PLAN v6, ADDENDUM: Experiment 2 (three-language AV pairs, hand-built rule layer L)

Status: **DRAFT TO FREEZE.** Written 2026-09-14, about 17:55 JST, before any participant. This file is part of the pre-registration of `ANALYSIS_PLAN_V6.md` and is hashed with it (main plan section 11 item 7: seven lines in `PLAN_FREEZE.sha256`). Where this file is silent, the main plan applies: exclusions (section 5), sample, rotation and stopping (section 6), freeze (section 11) and deviations (section 12).

Tags: [measured] read or computed from a named file; [code] read from code; [computed] a property of the test, not of data; [U] unverified, an estimate, or her decision.

Files this addendum relies on, all of which read the already rendered post arrays and mp4s read-only:

- `exp2/exp2_cell_rule.py` (+ `.csv`, `.json`, `.log`)
- `exp2/exp2_extra_checks.py` (+ `.log`)
- `exp2/exp2_power.py` (+ `exp2_power.log`)
- `exp2/diag_native_sim.py` (+ `.log`; simulation only)

---

## 1. Aim and link to Experiment 1 and the paper

**Question.** The voice is held identical in both faces.

- **P4:** do viewers judge the movement to fit the voice better when a hand-built rule layer moves the upper face with the utterance's own prosody (L) than with EmoFace alone (E)?
- **P5:** is that because L is timed with the voice rather than because it moves more? The comparison is L against its time reversal (Lrev), which has the same amplitude distribution but is out of step with the voice.

**Why a second experiment.** Experiment 1 (H vs E) needs a human brow track from the same recording, so it can only use English RAVDESS voices (main plan section 1.1). L needs only the audio, so it can be made for any language. Experiment 2 asks the Experiment 1 question for Mandarin, English and Japanese voices, with viewers of all three native languages.

**What L is and is not.**

- L is a hand-built rule layer on EmoFace's output (token `r5b_lilt_nonod-g2`). It is not a learned model and **not a human reference**.
- Its timing comes from each utterance's own prosody. That is the design statement; the rule's internals are not restated here [U].
- Its amplitude was calibrated so that its in-speech brow movement matches the English RAVDESS human reference [measured by the main session, 2026-09-14]:

  | median, rig units | inner brow | outer brow |
  |---|---|---|
  | L | 0.43 | 0.40 |
  | human reference | 0.48 | 0.35 |
  | E | 0.08 | 0.09 |

  On the 9 cells used (section 2.3), p95-p5 over the shown frames is 0.09-0.74 inner and 0.19-0.55 outer for L, and 0.06-0.19 inner and 0.02-0.30 outer for E [measured, `exp2/exp2_cell_rule.csv`].
- **L changes more than the brows** [measured, `exp2/exp2_extra_checks.log`]. On all 9 cells, L and Lrev differ from E in these rig columns and in no others:
  - 29/98, brow down [code, `_explore/clamp_check.py`]
  - 30/99, name not checked [U]
  - 31/100, inner brow raise
  - 32/101, outer brow raise
  - 38/107, name not checked [U]
  - 39/108, upper lid raise [code, `variants.py` EYELID_U]

  H in Experiment 1 changes only 31/100 and 32/101 [measured, `stimuli/build/verify_arrays_v6.csv`].

**Link to the paper.**

- Claim A says the audited models' brows do not follow controlled voice changes. P4 asks whether an upper face that does follow the voice, by rule, is preferred to EmoFace's.
- Claim C says a pre-registered **learned** layer failed its gates. L is not that layer, so Experiment 2 says nothing about claim C.
- A positive P4 says that, for these stimuli, a voice-timed rule layer is preferred to EmoFace. It does not say that L is human-like or that L is the right mapping from prosody to face.

**Scope.** One avatar (Danielle V9). Three utterances per language, one speaker per language, emotions anger / happiness / sadness. Native Mandarin, English and Japanese viewers, analysed pooled (main plan section 1.1).

## 2. Stimuli

### 2.1 Arms and media

| arm | token | what |
|---|---|---|
| E | `before` | EmoFace, condition `e10m10_d5_h` (as in Experiment 1) |
| L | `r5b_lilt_nonod-g2` | E + the rule layer |
| Lrev | `r5b_lilt_nonod_rev-g2` | E + the same rule layer driven by the time-reversed prosody [code, build_lilt_v6.py docstring and design table]; not re-derived in this lane, section 2.4 measures the result |

- **Media.** The clips are the ALREADY RENDERED `pool/mp4/<cell>__<token>__e10m10_d5_h__danielle_v9.mp4`. All 27 files for the 9 cells exist [measured, `exp2/exp2_extra_checks.log`, 2026-09-14 ~17:35].
- **Her check of the renders.** She viewed these renders on 14 Sep before this addendum was written: stare fixed, natural, differences visible. That stimulus check came before any participant, used no participant data, and is disclosed in section 8.
- **Presentation, identical to Experiment 1's pairs:**
  - face-cropped side by side, the same voice on both sides, played simultaneously;
  - the pool_render mux (loudnorm target -23 LUFS); measured -22.9 to -20.6 LUFS across cells, identical within each cell's pair (0.0 LU between arms) [measured, provisional manifest `media_qc.lufs_by_cell`];
  - neutral file names;
  - the same pair question and seven-point L3..R3 scale (main plan section 2).
- **Left/right** is counterbalanced by 2-arm pick groups (section 3).

### 2.2 Voices (the confound of section 8 starts here)

| stimulus language | cells | voice source [measured, `pool_set.json` source / speaker / dur_s] |
|---|---|---|
| cn | `cn_esd0002_short3_ang`, `_hap`, `_sad` | **actor recordings**: ESD speaker 0002, the same sentence 我总是控制不了它。 in all three; 2.58 / 2.54 / 3.74 s |
| en | `en_ang_short_sentence`, `en_hap_short_sentence`, `en_sad_short_sentence` | **synthetic clones** (tier A, l4b library), reference speaker `rav02`; 2.22 / 1.90 / 1.59 s; texts not stored in pool_set [U] |
| jp | `jp_ang_short_sentence`, `jp_hap_short_sentence`, `jp_sad_short_sentence` | **synthetic clones** (tier A, l4b library), reference speaker `jvnvF2`; 2.16 / 2.16 / 2.68 s; texts not stored in pool_set [U] |

The TTS model behind each clone is not re-checked here [U]; pool_set notes "VoxCPM2/chatterbox clones".

### 2.3 A-priori cell rule and the realised set

**The rule** (her decision, fixed before any participant):

1. **Parity:** anger, happiness and sadness in every stimulus language.
2. **Candidates:** the cells of the fixed v4 study in `pool_set.json`: `cn_esd0002_short3_{ang,hap,sad,sur,neu}`, `en_{ang,hap,sad}_short_sentence`, `jp_{ang,sad,hap}_short_sentence` and `jp_jvnvF2_long_sur`.
3. **Visibility:** a cell qualifies only if L - E has a brow p95-p5 of **at least 2.0 face-crop px on at least one brow channel**, measured over the frames where the layer is shown.
   - inner channel = mean of rig columns 31/100 x 8.1 px per rig unit
   - outer channel = mean of 32/101 x 8.0 px per rig unit
4. **No substitution:** a parity cell that fails is dropped, not replaced, and its language enters with fewer cells.

The candidates hold exactly one cell per language x emotion inside the parity set, so parity leaves no choice. `cn_esd0002_short3_sur`, `cn_esd0002_short3_neu` and `jp_jvnvF2_long_sur` are excluded by parity, not by the visibility rule.

**Which measurement decides.** The builder applies the rule at build time [code, build_lilt_v6.py `--ml`]: clip(L) - clip(E), L/R averaged, in the cue-bank speech span, at 8.14 / 8.03 px per rig unit from gains.json. It writes its table into the manifest (`ml.table`) and `design_table.md`. **That table decides.**

**Cross-check below** [measured, `exp2/exp2_cell_rule.py` on the post arrays]. Here the shown frames are those where |L - E| or |Lrev - E| > 1e-6 on any of 31/32/100/101 (the speech span with its hold ramp), at 8.1 / 8.0 px. If the build table selects different cells, this section is corrected before hashing.

| cell | L-E inner px | L-E outer px | qualifies | Lrev/L inner | Lrev/L outer | r(L-E, Lrev-E) inner | r outer | Lrev clean |
|---|---|---|---|---|---|---|---|---|
| cn_esd0002_short3_ang | 3.21 | 3.25 | yes | 1.063 | 0.986 | 0.359 | 0.402 | no |
| cn_esd0002_short3_hap | 0.44 | 2.86 | yes (outer only; inner clamped) | 1.084 | 0.960 | 0.310 | -0.173 | no |
| cn_esd0002_short3_sad | 5.98 | 1.60 | yes (inner) | 0.921 | 0.902 | -0.236 | -0.411 | **yes** |
| en_ang_short_sentence | 3.16 | 4.17 | yes | 0.953 | 0.998 | 0.247 | 0.014 | **yes** |
| en_hap_short_sentence | 2.20 | 3.46 | yes | 0.854 | 0.980 | -0.095 | 0.194 | **yes** |
| en_sad_short_sentence | 2.90 | 2.88 | yes | 0.795 | 1.023 | -0.155 | -0.460 | no (ratio inner 0.795) |
| jp_ang_short_sentence | 3.21 | 4.02 | yes | 1.047 | 1.031 | 0.479 | 0.380 | no |
| jp_hap_short_sentence | 4.92 | 3.98 | yes | 1.005 | 1.018 | -0.260 | -0.631 | **yes** |
| jp_sad_short_sentence | 6.04 | 1.92 | yes (inner) | 1.121 | 0.971 | -0.213 | 0.661 | no |

**Realised set:** all 9 parity cells qualify, so there are 3 cells x 3 languages x 2 contrasts = **18 Experiment 2 screens per participant** [measured].

### 2.4 Lrev as a timing control; the P5 sensitivity subset

This uses the same rule as P2's in Experiment 1 (main plan section 4.2), applied to the Lrev columns above.

**Rule:** Lrev is a clean motion-equated timing control when, on both brow channels, the Lrev/L amplitude ratio is in [0.80, 1.25] **and** r(L - E, Lrev - E) < 0.30.

**Clean, 4 of 9** [measured]: `cn_esd0002_short3_sad`, `en_ang_short_sentence`, `en_hap_short_sentence`, `jp_hap_short_sentence`.

**Not clean, 5 of 9:**

| cell | reason |
|---|---|
| cn_ang | r 0.36 inner / 0.40 outer |
| cn_hap | r 0.31 inner |
| en_sad | ratio 0.795 inner |
| jp_ang | r 0.48 / 0.38 |
| jp_sad | r 0.66 outer |

In most of these cells L's motion is partly symmetric in time, so its reversal resembles it. Lrev is therefore a weaker timing control than R in Experiment 1, where 8 of 12 strong cells are clean.

The subset analysis (section 6) repeats P5 on the 4 clean cells. Because they are named here, hashing this file fixes them. `analyse_lilt_v6.py` reads the same list from `exp2/exp2_cell_rule.json` (fields `lrev_clean` and `parity_emotion`).

## 3. Design facts the analysis relies on (the manifest contract)

`build_lilt_v6.py` is authoritative. Once it has the block, its rotation report must confirm every line below before hashing; if a line differs, fix this table first.

| item | value |
|---|---|
| Block | `pair_ml`, presentation `AV_pair`, 18 served per participant, two arms per pick group (L on the right / L on the left) |
| Serving order | After block `pair` (+ `pair_catch`) and **before** `pair_mute` [her decision]. `patches/server_v6.patch` (17:36) serves it there, shuffled with no adjacent repeat of stim_language, voice_emotion or voice_arm where possible [code]. The unpatched live server has no `pair_ml` line and **silently drops** the block [code, server.py mode lilt] |
| `voice_arm` | `L_vs_E@<lang>` or `L_vs_Lrev@<lang>`, where lang = `cn`, `en` or `jp` (the stimulus language) |
| `cue_manip` | `L:<arm>|R:<arm>`, with arm names `L`, `E`, `Lrev` |
| `stim_language` | `cn` / `en` / `jp`, equal to the language in `voice_arm` |
| `base_cell` | The pool cell name, e.g. `en_hap_short_sentence` (fallback: the text before `__` in `face_motion`) |
| `voice_emotion` | ang / hap / sad |
| Side balance within a participant | Each contrast has 9 screens, an odd number, so L is on the left in **4 or 5** of 9, mirrored between consecutive rotation slots, and L is on opposite sides in a cell's two contrasts. The builder guarantees the flip by group naming (`pair5_ml1_LE_<slot>_<cell>`, `pair5_ml2_LR_<slot>_<cell>`) and checks it in its rotation report [code, build_lilt_v6.py]; the analysis self-test checks both on the builder's items [measured, self-test G] |
| Rotation | Period of 6 for Experiment 1 + Experiment 2 (2-arm groups; the 18 groups are a multiple of 2 and 3). The offset is allocated **least-filled per link** (LILTROT v6): the offset held by the fewest non-test sessions of the same link, build and period that are completed or were started less than 60 minutes ago, ties to the lowest; stored in the session record `lilt_rotation` [code, rotation/apply_rotation_v6.py; main plan section 2 "Rotation"], so each listener group is balanced on its own counts |
| Screens | 81 per participant: 6 singles, 2 practice, 36 AV pairs + 1 catch, **18 pair_ml**, 6 silent pairs, 12 voice-only (main plan section 2) |

## 4. Dependent variable, hypotheses and tests (confirmatory family = P4, P5)

**Trial score.**

- v = -3..+3 from L3..R3; + means the right face is better.
- **s = +v when L was on the right, -v when L was on the left.**
- Only the first response (revision 0) counts.

**Participant DV.** Per contrast: the mean over the three stimulus languages of the participant's per-language mean s, so each language has equal weight. With complete data this equals the plain mean over 9 screens. After a focus drop, it stops one language's retained count from reweighting the others.

| | hypothesis | participant DV | H0 |
|---|---|---|---|
| **P4** | L's movement fits the voice better than E's | D4 = language-weighted mean s in L_vs_E | mean D4 = 0 |
| **P5** | L is preferred over Lrev: timing with the voice matters, not motion alone | D5 = language-weighted mean s in L_vs_Lrev | mean D5 = 0 |

**Test.** A two-sided one-sample t-test on the participant DVs decides. `analyse_lilt_v6.py` also reports the one-sided p, which is not used for the decision.

**Multiplicity.** Holm over P4 and P5, family-wise alpha .05. **This family is separate from Experiment 1's P1-P3.** The two experiments use different layers and different stimuli, and no conclusion combines them. The paper reports each experiment against its own family and says so in the Method.

**Support rule.** A hypothesis is supported when its Holm-adjusted p < .05 **and** its mean is > 0. A significant negative mean is reported as contradicting the hypothesis.

**Reported for each DV:**

- mean, SD and dz;
- the 95 % participant-bootstrap percentile CI (B = 10,000, seed from the contrast name);
- the Wilcoxon signed-rank p;
- the number of participants with DV > 0 and with DV < 0, and the sign-test p;
- the cell-level mean with a bootstrap CI over the 9 cells. If that CI includes 0 while the participant test is significant, the paper says the effect is not shown to generalise over utterances.

**Robustness.** The t-test decides. When the Wilcoxon test disagrees about significance, the result is called "not robust to the test choice".

**Sensitivity** [computed, `exp2/exp2_power.log`, `power_sensitivity.log`; two-sided, 80 % power]:

| pooled N | alpha .05 | alpha .025 (Holm step 1 of 2) |
|---|---|---|
| 24 | dz >= 0.60 | dz >= 0.67 |
| 36 | dz >= 0.48 | dz >= 0.53 |
| 48 | dz >= 0.41 | dz >= 0.46 |
| 72 | dz >= 0.34 | dz >= 0.37 |

## 5. Exclusions (shared with Experiment 1, plus one Experiment-2 rule)

Main plan section 5 applies, in its order.

1. **Not a participant:** as in the main plan.
2. **Not eligible:** the same step as in Experiment 1, now with three languages:
   - `native_lang` not in {cn, en, jp}, i.e. "other";
   - hearing = impaired;
   - vision = other.
3. **Catch:** the single catch screen decides the primary sample for **both** experiments.
4. **Focus:**
   - `pair_ml` screens are scored screens. Primary: drop screens with `focus_hidden` >= 1. Sensitivity: drop screens with `focus_lost` >= 1 and re-report P4 and P5.
   - A participant excluded by the Experiment 1 retention rule (< 75 % retained in any primary AV contrast) is out of **every** analysis, Experiment 2 included.
   - **Experiment 2 rule:** a participant with < 75 % retained in L_vs_E (fewer than 7 of 9) **or** in L_vs_Lrev is out of **Experiment 2 only**.
5. **Nothing else:** no exclusion by response time, response pattern or outcome.

## 6. Secondary and exploratory analyses (no confirmatory claims)

- **Per stimulus language (descriptive).** For L_vs_E and L_vs_Lrev in each language: participant means with participant-bootstrap CIs, and the trial preference and tie rates.
  - Each participant contributes 3 screens per contrast per language, so these are noisy.
  - No per-language claim is made; the paper reports only the pooled P4 / P5.
- **Listener groups (secondary).** P4 and P5 within each listener group, with Holm inside the group. The group is the **link** (infos `test_url`; main plan section 1.1), which is also the rotation counter and the stopping count; a participant whose native_lang differs from the link stays in the link's group and is listed. Reported as secondary, with the group N.
- **Group differences (exploratory).** For D4 and D5, and for Experiment 1's D1, D2, D3 and DM: the between-group sum of squares with a label-permutation p (5,000 permutations, seed from the DV name), plus Kruskal-Wallis when scipy is present.
- **Listener group x stimulus language (exploratory).**
  - Per contrast, a 3 x 3 table of participant means with bootstrap CIs.
  - The **native-stimulus effect**: within each participant, the mean s on the cells in their native language minus the mean s on the other two languages' cells. Tested with a one-sample t, bootstrap CI and sign counts.
  - Reading limit: the native stimulus language is fully confounded with that language's speaker, voice source (actor vs clone) and cells (section 8). A native-stimulus effect therefore cannot be attributed to language familiarity.
- **Side-bias-adjusted P4 / P5 (sensitivity).** Within a participant L sits 4:5 or 5:4, so that participant's side bias does not cancel exactly. The adjustment subtracts from v the participant's mean raw v over the 36 Experiment 1 AV pairs, where sides are balanced. It removes an always-right responder exactly [measured, self-test G].
- **P5 on the clean-Lrev subset (sensitivity, descriptive):** the 4 cells of section 2.4.
- **Per emotion (descriptive):** participant means per emotion per contrast.
- **Focus_lost sensitivity:** section 5 step 4.

## 7. What each outcome means

- **P4 and P5 supported.** A voice-timed rule layer made the face judged a better fit to the voice than EmoFace, and its timing mattered, pooled over Mandarin, English and Japanese utterances. This is not a claim that L is human-like, and not a claim about any single language.
- **P4 supported, P5 null.** The preference may rest on added upper-face motion rather than on its timing. Lrev is clean in only 4 of 9 cells, so the P5 subset estimate is read alongside. A positive subset estimate with a pooled null points to the weak control, not to "timing does not matter" (descriptive only).
- **P5 supported, P4 null.** Motion out of step with the voice was judged worse than motion in step, while motion in step was not judged better than no added motion.
- **Both null.** Not detected at the pooled N reached; only effects at or above the section 4 dz are excluded. Write "not detected", never "no effect".
  - **Experiment 2 has no silent visibility check of its own.** L's visibility rests on the measured amplitude (every cell >= 2 px; L comparable to H in rig units) and on Experiment 1's silent check of H.
  - A double null together with a failed Experiment 1 silent check is therefore uninformative for Experiment 2 as well.
- **Pooled result and per-language descriptives disagree.** Only the pooled result is claimed. The per-language pattern is reported as descriptive, together with the section 8 confounds.
- **Experiments 1 and 2 disagree** (e.g. P1 null, P4 positive). Both are reported as found; the difference between the experiments is not tested. The two differ in several ways:
  - **H:** human, brows only, pre-onset raise removed, clamp loss up to 42 %.
  - **L:** rule-based; changes brows + lids + brow-down + two further columns; amplitude-matched.
  - **Voices:** acted English RAVDESS in Experiment 1; recordings and clones in Experiment 2.

## 8. Confounds and limitations (written before data)

1. **Stimulus language is confounded with voice source and speaker.** The Mandarin cells are ESD actor recordings of one speaker and one sentence; the English and Japanese cells are TTS clones of one reference speaker each. Any stimulus-language difference, including the native-stimulus effect, may come from recording vs synthesis, the speaker or the sentence rather than the language.
2. **The English clone reference is RAVDESS actor 02** [measured, pool_set speaker `rav02`]. Actor 02's own recordings appear in Experiment 1 (`rav02_ang`, `rav02_dis`), which is served before Experiment 2, so familiarity with the timbre may carry over. Whether the clone sounds like the actor is not measured [U].
3. **L changes more than the brows:** brow down, brow raises, upper lid, and columns 30/99 and 38/107 [measured]. H changes only the brow raises, so the two experiments do not test the same layer content.
4. **L's amplitude is calibrated to the English human reference.** No human brow reference was measured for Mandarin or Japanese, so "amplitude comparable to the human" holds for English speakers' brows only.
5. **The rule layer was chosen by eye before data.** She chose the rung and gain and viewed the renders on 14 Sep. No participant data were used, but the stimuli are a selected best case of the rule layer, not a random draw.
6. **No silent check for L** (section 7).
7. **Order and fatigue.** Experiment 2 always follows the 36 Experiment 1 AV pairs and the catch, and precedes the silent pairs; the order is not counterbalanced. Carry-over (the same question over the preceding 37 screens) and fatigue can lower sensitivity. Both affect the two arms of every pair equally.
8. **Each Experiment 2 voice is heard twice** within the block: once in L_vs_E and once in L_vs_Lrev.
9. **The timing control is weak:** Lrev is clean in only 4 of 9 cells (section 2.4).
10. **There are few utterances:** 3 per language, one speaker per language, three emotions. Generalisation over utterances rests on a 9-cell bootstrap, which is coarse.
11. **One avatar, one camera, one crop, one condition** (`e10m10_d5_h`).
12. **Clamping:** `cn_esd0002_short3_hap` loses its inner-brow channel (0.44 px shown) and qualifies on the outer channel only [measured].
13. **Loudness differs between cells, never within a pair.** cn_ang -20.8, cn_hap -20.6, jp_sad -21.9, cn_sad -22.6 LUFS; the other five cells -22.8 / -22.9 (range 2.3 LU; the Mandarin cells about 1.5 LU louder on average) [measured, provisional build]. P4 and P5 are unaffected (0.0 LU between the two versions of each cell); the exploratory listener group x stimulus language analysis is not. The clips were not re-muxed, because she viewed them as they are.

## 9. Analysis code and freeze

`analyse_lilt_v6.py` implements sections 4-6:

- functions `exp2_summary`, `exp2_dv`, `exp2_clean_cells`, `perm_groups` and `listener_group`;
- the Experiment 2 retention rule inside `exclusions`, recorded in `flow["step4_dropped_from_exp2_only"]`;
- results under `results["samples"][<sample>]["exp2"]`, `["by_listener_group"]` and `["group_differences_exploratory"]`.

Its `--selftest` (section G) covers simulated 81-screen sessions from the three links, rotated per link as the patched server does. The sessions are built from the builder's own `plan_items(..., ml=True)` with `--singles 6` (an analysis-side stand-in is used only if the builder cannot be called that way), and the self-test log names the plan source.

**Freeze:** main plan section 11 item 7. `PLAN_FREEZE.sha256` holds seven lines: `ANALYSIS_PLAN_V6.md`, this file, `build_lilt_v6.py`, `analyse_lilt_v6.py`, the live `manifest.json`, `stimuli/build/p2_sensitivity_cells.json` and `exp2/exp2_cell_rule.json` (the P5 clean-Lrev subset the analysis reads).

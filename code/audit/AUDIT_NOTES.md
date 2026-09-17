# audit2: voice-dose audit of four talking-head models on ladder2 (plan section 3.2, E1 and E3)

Lane `voicedose/audit2/`, built 2026-09-14 04:55–05:20 on CPU.
Tags: [measured] = computed in this lane; [code] = read in code; [V-text] = read in a project document; [U] = unverified.
Every table value comes from `results.csv` via `analyse2.py`; `audit2_tables.md` holds the full tables T1–T10. None of Decisions 1–5 is taken here.

---

## 0. Summary

1. **No audited model shows a voice-driven upper-face dose response on the PSOLA ladder** [measured].
   - Brow amplitude slopes (mean of inner and outer, 12 cells, cell bootstrap): every span, dynamics and register CI of EmoFace, EmoTalk and EMOTE includes 0.
   - MEDTalk's brows go slightly *down* with register (−0.017/st [−0.030, −0.007]).
   - The effort composite (+4.7 st, rate 0.90, +7 dB), brow mean ln ratio vs c100:

     | model | ln ratio [95% CI] | cells > 0 |
     |---|---|---|
     | EmoFace | −0.007 [−0.104, +0.118] | 4/12 |
     | EmoTalk | +0.107 [−0.094, +0.382] | 7/12 |
     | EMOTE | −0.015 [−0.106, +0.070] | 7/12 |
     | MEDTalk | −0.072 [−0.137, −0.010] | 4/12 |

   - The human-predicted value is **+0.217 [0.089, 0.359]** (G2, E4_common, W2_full, slope-only, realised deltas).
   - EmoTalk's mean is one cell: en_rav02_take1_ang +1.43; without it −0.013, median +0.018.
2. **Both constructed followers pass the plan's effort rule** ("within [0.5, 2] × predicted and above the null floor") [measured].
   - F0-height follower: +0.423 [0.338, 0.513], 1.95× predicted.
   - Energy follower: +0.138 [0.102, 0.175], 0.64× predicted.
   - So the effort G2 rule alone does not exclude a trivial follower. The G0b baselines and the reversed-prosody control in plan §2.2 are what separate them. This is a measurement, not a gate decision.
3. **E3 metric validation.**
   - **The follow index separates the F0-height follower** from the human tracks and from all four models: excess over circular shift +0.295 [0.180, 0.418], vs human +0.037/+0.050 and models −0.11…+0.13.
   - **It does not separate the energy follower** (−0.016 [−0.190, +0.100]).
   - **The energy follower is identified instead by** 1.96 events per F0 peak (human 1.28–1.38) and by its dynamics slope.
   - **The dose slope separates both followers from the models**: F0 follower span +0.470 [0.293, 0.642], register +0.092/st [0.065, 0.117]; energy follower dynamics +0.542 [0.357, 0.719].
   - **On 12 cells with 43 frozen F0 peaks, the follow index cannot tell a deaf model from a human**: model CIs are ±0.1–0.2 against a human mean of +0.04.
4. **Injection control.**
   - With human thresholds, recovery is 0.61/0.60, reproducing human2.
   - With each model's own threshold, recovery is 0.30–0.82 clean and 0.36–0.76 with MediaPipe-matched noise. EMOTE (0.30/0.32 clean) and MEDTalk outer brow (0.46 clean, 0.36 noise) fall below 0.5.
   - The follow index is usable as a gate for gross F0 locking. For those two models it is under-powered, and against human-sized locking it is descriptive [measured].
5. **Level.** +6 dB changes no model's upper face: every |brow ln ratio| ≤ 0.006, all CIs include 0. EmoFace's jaw does not respond to span or dynamics either. It does respond to rate: rate090 +0.057 [0.030, 0.088], 11/12 cells; effort +0.090 [0.025, 0.158] [measured].

---

## 1. What ran

| stage | env | items | failures | wall time |
|---|---|---|---|---|
| `stage_acoustics.py` (extract_prosody grid 60 fps + G2 cue measures) | GPTSoVits | 204 | 0 | 319 s |
| `run_emoface.py` (measure-lane path: pool_rigs.Model decode + post_step) | emoface | 204 | 0 | 74 s |
| `run_ext.py --probe emotalk` (d1_audit probe, `run(wav)`) | d1_emotalk | 204 | 0 | 186 s |
| `run_ext.py --probe emote` (d1_audit probe, `run(wav, label)` + FLAME 68 landmarks) | d1_emote | 204 | 0 | 59 s |
| `run_ext.py --probe medtalk` (d1_audit probe, `run(wav, label)`, ASR 'zh' as shipped, raw decoder) | pl_medtalk | 204 | 0 | 538 s |
| `score.py` (metrics.py per item × source × channel × arm) | GPTSoVits | 10,812 rows | 0 | 42 s |
| `analyse2.py` (dose, ratios, G2, E3, figures) | GPTSoVits | — | first run crashed (column named `diff` clashed with `Series.diff`), fixed and rerun | ~40 s |

**Items.** All 204 ladder2 wavs (12 cells × 17 rungs) [measured].

**Checks.**
- EmoFace decode vs the probe's full `run()`: max |diff| 3.6e-7 [measured].
- Probe settings are the adapters' defaults:
  - EmoTalk: `level=1`, `person=0`, no label.
  - EMOTE: EMOTE_v2, intensity 2, identity M003, label = cell emotion.
  - MEDTalk: label = cell emotion, `asr="zh"`, savgol 0 [code].

**Isolation.**
- MEDTalk's ASR cache was redirected into `audit2/asr_cache_medtalk`.
- Every script ran with `PYTHONDONTWRITEBYTECODE=1`.
- No GPU, nothing queued.
- A final `find -newer` over d1_audit, human2, ladder2, audit, pipeline/, prosody_ladder/, lilthead/ and the EmoTalk/MEDTalk/inferno repos found no file written outside this lane [measured].

## 2. Channel mapping (brow-inner / brow-outer / upper-lid equivalents; jaw = positive control)

| model | frame rate, readout | brow_inner | brow_outer | lid_upper | jaw |
|---|---|---|---|---|---|
| EmoFace | 60 fps, MetaHuman rig after post_step (savgol 15/3, no blinks) | mean cols 31/100 `CTRL_[LR]_brow_raiseIn` | 32/101 `brow_raiseOut` | 39/108 `eye_eyelidU` | 4 `CTRL_C_jaw-translateY` |
| MEDTalk | 60 fps, raw decoder, same 174-column order [code: probe_medtalk.py; names checked in both metahuman_attr_names.txt] | 31/100 | 32/101 | 39/108 | 4 |
| EmoTalk | 30 fps, ARKit-52 raw (no demo savgol / blink injection) | `browInnerUp` (2) | mean `browOuterUp` L/R (3, 4) | mean `eyeWide` L/R (20, 21) − mean `eyeBlink` L/R (8, 9) | `jawOpen` (24) |
| EMOTE | 25 fps, FLAME 68 landmarks on the mean face, y up, / IOD = \|lm36 − lm45\| | (mean y lm21, 22 − eye line) / IOD | (mean y lm17, 26 − eye line) / IOD | mean \|37−41\|, \|38−40\|, \|43−47\|, \|44−46\| / IOD | \|lm30 − lm8\| / IOD |

Notes on the mapping:
- **EMOTE eye line** = mean y of lm36, 39, 42, 45. These are the human pose-normalised definitions (`human/analyze.py` build_signals) in iBUG indices [code].
- **Secondary "net" brow** (raise − brow_down: rig cols 29/98, ARKit 0/1) is in `results.csv` and `dose_tables.csv`. It changes no conclusion. Effort net-brow ln ratio: EmoFace −0.057 (5/12), EmoTalk +0.053 (6/12), MEDTalk −0.093 (2/12) [measured: diag.json].
- **Constructed followers** are built on the same 60-fps acoustic grid. Each is copied into both brow channels.
  - F0-height follower: 60-ms Gaussian of the phrase layer re the cell's c100 voiced median, rectified at 0 st.
  - Energy follower: 60-ms Gaussian of [30-ms dBFS − (c100 speech-span median − 20 dB)]+, zeroed outside the frozen span. It is speaker/cell referenced, so level, dynamics and rate all pass through.
- **MEDTalk s_t**: the per-frame audio-predicted intensity, reported as source `medtalk_st`.

## 3. Scoring rules

**Frozen structure** (ladder2 rule), computed once per cell on c100:
- Span = `metrics.speech_mask` (30 dB) on the c100 extract_prosody logE. This is window W2, the one the human G2 coefficients use.
- F0 peaks = `metrics.f0_peaks` (≥ 1 st) inside that span, the human2 `clipdata.py` "audit" path.
- There are 43 peaks over 12 cells: cn 1–5, en 1–3, jp 5–7 per cell [measured: refs.json].
- Both are reused on every rung.

**Rate rungs** (rate090, effort0, effort): every track — model output, follower, acoustic layer — is resampled onto the c100 time axis, x_tn(t) = x_item(t/rate) (ladder2 timemap rule). They are then scored like any other rung.

**Event thresholds.** `metrics.reference_threshold` over the 12 c100 items per (source, channel, arm), held fixed for all rungs.

**Noise arm.** White noise is added, then thresholds are recomputed.
- σ = (human median noise_sigma / human median amp) × this source's median c100 amp × √(fps/29.97).
- Human ratios [measured, human_audit_metrics.csv]: brow 0.0245, lid 0.0192.
- The √fps factor equalises smoothed noise at the detector's 40-ms smoothing. It is an assumption [U].

**Follow index.**
- Plan G3 window −0.25…+0.30 s, plus the post-peak window 0…+0.30 s.
- Nulls: circular shift (`metrics.follow_index`, 200 shifts) and cross-utterance. The cross-utterance null uses the same rung of the other cells in the same language, aligned at span onset, mirroring human2's "same actor × statement" null.

**Dose** (`metrics.dose_response`, group = cell: within-cell slope, cluster bootstrap over cells with 2000 draws, Spearman on cell-centred values, fraction of cells perfectly monotone):

| family | rungs | x |
|---|---|---|
| span | k050 k075 c100 k133 k167 | ln realised k, Praat (pyin as sensitivity) |
| dyn | g060 c100 g150 | ln realised_g_ip |
| combined | less c100 more | −1/0/+1 |
| register | c100 r+2 r+4 | register st, Praat |
| rate | c100 rate090 | ln(1/0.9) |

- y = ln amp (p95−p5), ln vel_rms, raw event rate.
- brow_mean = geometric mean of the inner and outer amplitudes, so its ln ratio is the plan's primary endpoint.
- **swing** = median over cells of (max − min over the family's rungs) / c100 (d1_audit `protocol.py` t4_stats rule) [code].
- **r_w** = within-cell Pearson r. It is printed beside every swing in T1.

**Null floor.** Median |ln ratio| of src and tgt100 vs c100 (24 contrasts).

## 4. E1 results (clean arm, 12 cells)

### 4.1 Brow mean: slopes and ratios [measured: T1, T3, T4]

| source | span slope / ln k | swing | dyn slope / ln g | swing | register / st | rate090 ln ratio | effort ln ratio | effort0 ln ratio | lvl+6 ln ratio | null floor (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| EmoFace | −0.008 [−0.102, +0.099] | 0.19 | −0.000 [−0.061, +0.061] | 0.09 | −0.007 [−0.037, +0.030] | −0.048 [−0.160, +0.047] | −0.007 [−0.104, +0.118] | −0.013 [−0.108, +0.109] | −0.004 [−0.010, +0.001] | 0.024 |
| EmoTalk | +0.080 [−0.045, +0.239] | 0.18 | −0.113 [−0.407, +0.164] | 0.37 | −0.017 [−0.050, +0.009] | +0.091 [−0.051, +0.236] | +0.107 [−0.094, +0.382] | +0.108 [−0.095, +0.385] | +0.004 [−0.005, +0.014] | 0.044 |
| EMOTE | +0.027 [−0.008, +0.072] | 0.08 | +0.007 [−0.064, +0.077] | 0.06 | −0.000 [−0.015, +0.015] | −0.003 [−0.076, +0.071] | −0.015 [−0.106, +0.070] | −0.016 [−0.109, +0.067] | −0.002 [−0.007, +0.002] | 0.015 |
| MEDTalk | −0.008 [−0.047, +0.033] | 0.10 | +0.002 [−0.054, +0.052] | 0.06 | **−0.017 [−0.030, −0.007]** | −0.035 [−0.073, +0.000] | **−0.072 [−0.137, −0.010]** | **−0.076 [−0.139, −0.014]** | +0.006 [−0.001, +0.019] | 0.023 |
| F0-height follower | **+0.470 [+0.293, +0.642]** | 0.64 | +0.096 [−0.009, +0.241] | 0.03 | **+0.092 [+0.065, +0.117]** | −0.242 [−0.496, −0.034] (tracker artefact, §9) | **+0.423 [+0.338, +0.513]** | **+0.497 [+0.394, +0.609]** | −0.010 [−0.066, +0.027] | 0.051 |
| Energy follower | +0.001 [−0.018, +0.023] | 0.02 | **+0.542 [+0.357, +0.719]** | 0.50 | −0.004 [−0.010, +0.001] | **+0.061 [+0.044, +0.076]** | **+0.138 [+0.102, +0.175]** | **+0.048 [+0.028, +0.070]** | **+0.081 [+0.052, +0.114]** | 0.003 |

Reading the table:
- Monotone cells (span, brow mean): EmoFace 1/12, EmoTalk 1/12, EMOTE 0/12, MEDTalk 0/12, F0 follower 7/12 (frac 0.58). Within-cell r for span: −0.03, +0.24, +0.23, −0.08, +0.67.
- Dropping the three ladder2 FAIL items (en_rav02_take1_sad k133, k167, more) moves no span or combined slope by more than 0.013 (T2).
- The pyin-dose span slopes agree in sign and CI coverage (T2) [measured].

### 4.2 Lid and jaw [measured: T1, T3]

| source | lid span | lid dyn | lid rate090 | lid effort | jaw span | jaw dyn | jaw rate090 | jaw effort |
|---|---|---|---|---|---|---|---|---|
| EmoFace | +0.062 [−0.049, +0.168] | +0.065 [−0.052, +0.203] | +0.020 [−0.056, +0.092] | −0.052 [−0.175, +0.066] | −0.005 [−0.043, +0.036] | +0.007 [−0.079, +0.112] | **+0.057 [+0.030, +0.088]** 11/12 | **+0.090 [+0.025, +0.158]** |
| EmoTalk | **+0.124 [+0.039, +0.199]** (pyin +0.034 [−0.017, +0.218]) | −0.001 [−0.176, +0.199] | **−0.173 [−0.333, −0.031]** | −0.069 [−0.182, +0.050] | +0.007 [−0.051, +0.054] | +0.030 [−0.044, +0.107] | −0.024 [−0.172, +0.137] | +0.008 [−0.116, +0.144] |
| EMOTE | −0.015 [−0.058, +0.030] | **+0.079 [+0.017, +0.140]** | −0.031 [−0.077, +0.026] | +0.016 [−0.053, +0.091] | +0.031 [−0.008, +0.072] | +0.058 [−0.007, +0.122] | +0.003 [−0.048, +0.057] | +0.000 [−0.078, +0.083] |
| MEDTalk | +0.019 [−0.007, +0.045] | **−0.127 [−0.214, −0.042]** | **−0.077 [−0.116, −0.037]** 1/12 | −0.035 [−0.101, +0.026] | +0.002 [−0.031, +0.034] | −0.008 [−0.067, +0.041] | −0.000 [−0.022, +0.018] | **−0.047 [−0.078, −0.019]** |

**The jaw positive control works only for rate and effort, and only in EmoFace.** No model's jaw follows PSOLA span or envelope dynamics. The measure lane's RAVDESS jaw +0.11 came from natural intensity pairs [V-text: plan P1], not from this ladder.

### 4.3 Superseding the D1 T4 numbers

D1 T4 reported MEDTalk upper-face dyn partial r +0.642 and EmoFace upper span +0.316 / dyn +0.326. Those came from a WORLD-resynthesised ladder, with the label-deviation amplitude metric (RMS of M_label − M_neu) and swing 0.02 / 0.17 [V-text: plan P1; code: protocol.py t4_stats].

On this PSOLA ladder, with raw channel amplitudes:
- MEDTalk brow dynamics slope +0.002 [−0.054, +0.052] (swing 0.06); MEDTalk lid dynamics −0.127.
- EmoFace brow span −0.008 and dynamics −0.000.

The metrics differ, so these are not a like-for-like replication [measured / code].

## 5. Effort composite vs the human-predicted interval (G2)

### 5.1 Predictions

Realised deltas use the G2 definitions (`g2_predict.measure_wav`, lane copy), effort vs c100, mean over the 12 cells [measured]:
- level +6.55 dB
- register +3.39 st (the extract_prosody/harvest front end; Praat on the same items reads +4.71)
- env SD +0.17 dB
- ln duration +0.103

| prediction (brow_mean unless stated) | estimate [95% CI] |
|---|---|
| **E4_common, W2_full, slope-only, realised** (used below) | **+0.217 [0.089, 0.359]** |
| same, with intercept | +0.271 [0.163, 0.374] |
| same, Praat register substituted | +0.213 [0.101, 0.342] |
| nominal (+7 dB, +4.7 st, rate 0.90) | +0.222 [0.107, 0.356] |
| E4_common, W2_pm (equal-length windows) | +0.149 [0.020, 0.291] |
| E7_common, W2_full | +0.145 [0.063, 0.251] |
| lid, E4_common W2_full slope-only | +0.189 [0.097, 0.303] |
| effort0 (level removed; realised level −0.24 dB) | +0.040 [−0.068, 0.173] |
| lvl+6 | +0.157 [−0.029, 0.309] |
| rate090 | +0.059 [−0.003, 0.129] |
| r+4 | −0.015 [−0.104, 0.099] |

### 5.2 Comparison

| source | arm | brow ln ratio effort [CI] | × predicted | in [0.5, 2]× | CI overlaps prediction CI | above null floor (mean > floor and CI lo > 0) |
|---|---|---|---|---|---|---|
| EmoFace | clean | −0.007 [−0.104, +0.118] | −0.03 | no | yes | no |
| EmoTalk | clean | +0.107 [−0.094, +0.382] | 0.49 | no (just below) | yes | no |
| EMOTE | clean | −0.015 [−0.106, +0.070] | −0.07 | no | no | no |
| MEDTalk | clean | −0.072 [−0.137, −0.010] | −0.33 | no | no | no |
| F0-height follower | clean | +0.423 [+0.338, +0.513] | 1.95 | **yes** | yes | **yes** |
| Energy follower | clean | +0.138 [+0.102, +0.175] | 0.64 | **yes** | yes | **yes** |
| (noise arm) | noise | EmoFace −0.004, EmoTalk +0.100, EMOTE −0.020, MEDTalk −0.078, F0 follower +0.403 (1.85×), energy follower +0.133 (0.61×) | | same verdicts | | |

- **Lid** vs the +0.189 lid prediction: EmoFace −0.052, EmoTalk −0.069, EMOTE +0.016, MEDTalk −0.035. None is in band [measured: g2_compare.csv].
- **Level-free twin (effort0)**, predicted +0.040: models −0.076…+0.108; F0 follower +0.497 (12× predicted); energy follower +0.048 (1.2×).
- **Reading.** The effort rung discriminates poorly on its own. Two content-free followers land inside the band: one through register, one through level plus rate. The F0 follower then fails the level-free twin, and the energy follower fails the rest of its rungs (§7). A GATES.md that uses the effort rung needs effort0 or the G0b baselines beside it. Whether to do that is a decision for the analyst.

## 6. Verdicts, one paragraph per model

**EmoFace.**
- Its brows and lid do not respond to any prosodic dose on the PSOLA ladder. Brow slopes: span −0.008 [−0.102, +0.099], dynamics −0.000 [−0.061, +0.061], register −0.007/st [−0.037, +0.030]. Effort −0.007 [−0.104, +0.118], 4/12 cells up.
- Swing is 0.09–0.23, the same size as its null floor p90 (0.11).
- +6 dB changes nothing (−0.004 [−0.010, +0.001]), consistent with input normalisation [plan: `do_normalize`].
- The one voice dependence is in the jaw: rate090 +0.057 [0.030, 0.088] (11/12) and effort +0.090 [0.025, 0.158], with no span or dynamics response. The upper face stays flat while the jaw tracks articulation timing.
- Follow index (c100): +0.038 / +0.004, CIs ±0.15, indistinguishable from humans (T8). Injection recovery at its own threshold is 0.82/0.73, so the index could have detected locking [measured].
- **Verdict:** deaf above the jaw on this ladder.

**EmoTalk.**
- Brow slopes are wide and centred near 0: span +0.080 [−0.045, +0.239], dynamics −0.113 [−0.407, +0.164], register −0.017 [−0.050, +0.009].
- Effort is +0.107 [−0.094, +0.382], but one cell (en_rav02_take1_ang, +1.43) makes the mean. Without it −0.013; median +0.018. It is the only model whose effort CI overlaps the human prediction, and only because of that width.
- Lid follows Praat-measured span (+0.124 [0.039, 0.199]) but not the pyin dose (+0.034 [−0.017, +0.218]). Lid falls on rate090 (−0.173 [−0.333, −0.031]).
- +6 dB: +0.004 [−0.005, +0.014].
- Highest detrended phrase coupling of the four: inner brow r − r_rev +0.168 [0.028, 0.326]. There is no event-level locking (follow excess −0.030 [−0.131, +0.065]).
- Injection recovery 0.62/0.73.
- **Verdict:** no consistent dose response. Its brow is phrase-correlated in continuous r but not peak-locked, and it responds erratically to single cells (a noisy, not graded, voice dependence) [measured].

**EMOTE.**
- The flattest model: brow span +0.027 [−0.008, +0.072], dynamics +0.007, register −0.000/st [−0.015, +0.015], effort −0.015 [−0.106, +0.070], lvl+6 −0.002. Null floor 0.015; swing 0.06–0.08.
- Upper lid shows a small positive dynamics slope, +0.079 [0.017, 0.140] (rho +0.58), and g150 +0.036 [0.004, 0.069]. This is the only upper-face dose signal among the four, and it is about 1/7 of the energy follower's slope.
- Outer-brow follow excess is +0.126 [0.038, 0.235] clean but +0.052 [−0.056, +0.178] with noise.
- **Weakest injection recovery: 0.30/0.32 clean.** Its brow motion is small relative to its threshold, so the follow index is under-powered for EMOTE.
- Units are IOD on the FLAME mean face, directly comparable to human MediaPipe IOD. c100 brow_inner amplitude median is 0.055 IOD vs human 0.024 [measured: score_meta.json; human_dose.md §1].
- **Verdict:** deaf at the brow; a weak envelope-dynamics effect on the lid [measured].

**MEDTalk.**
- The only model with CIs excluding 0, and they point the wrong way. Brow register −0.017/st [−0.030, −0.007] (rho −0.52); effort −0.072 [−0.137, −0.010]; effort0 −0.076 [−0.139, −0.014]. Lid: dynamics −0.127 [−0.214, −0.042], rate090 −0.077 (1/12 up). Jaw: effort −0.047.
- Its audio-predicted intensity s_t (mean over the span) does rise with effort (+0.016 [0.010, 0.023], 11/12) and register (r+4 +0.010 [0.002, 0.018]). The rise is small, and it does not reach the brows, which fall.
- Brow follow excess: −0.113 [−0.234, −0.009] inner (fewer hits than chance) and −0.028 outer. Injection recovery 0.55/0.46.
- **Confound:** Whisper forced to Chinese, as shipped, transcribes the en/jp rungs differently on almost every rung: 7–17 distinct transcripts per 17 items, vs 2–5 in cn (script variants 他/它, 總/总) [measured: diag.json]. So the text path that feeds s_t varies with the rung. cn cells are the clean MEDTalk test; per-language, cn brow effort is −0.07 [−0.19, +0.04].
- **Verdict:** no human-direction dose response; small anti-dose effects that are consistent across channels [measured].

**Constructed followers (validation rows, not models).**
- **F0-height follower** behaves as built:
  - Doses: span +0.470, register +0.092/st, effort +0.423, effort0 +0.497, +6 dB −0.010.
  - Peak locking: follow excess +0.295, 0.76 events per F0 peak, precision excess +0.367, detrended coupling +0.336 [0.171, 0.510]. Injection recovery 0.93.
  - Its span slope is 0.47 rather than about 1. The front end it is built on (extract_prosody, i.e. LiltHead v1's) reads k167 as a phrase-span ×1.18 (median) where Praat reads ×1.57 [measured: diag.json]. ladder2 also found this front end orders the span ladder in only 6/12 cells [V-text: ladder2 summary].
- **Energy follower:**
  - Doses: dynamics +0.542, rate090 +0.061, +6 dB +0.081, effort +0.138, span +0.001, register −0.004.
  - No peak locking (−0.016), about 2 events per F0 peak (syllable-rate firing); injection recovery only 0.40 because its base recall is already 0.88 (ceiling) [measured].

## 7. E3: do the metrics separate followers, humans and deaf models?

**Human reference** (human2 `follow_records.csv`, audit path, per-actor `reference_threshold`, 1362 clips with ≥ 1 peak, 24 actors, plan window):

| channel | recall | excess vs circular | excess vs cross-utterance | events/F0 peak | events/s |
|---|---|---|---|---|---|
| inner | 0.653 | +0.037 [0.016, 0.055] | +0.063 [0.041, 0.083] | 1.38 | 1.31 |
| outer | 0.619 | +0.050 [0.032, 0.066] | +0.058 [0.042, 0.073] | 1.28 | 1.21 |

- In the post-peak window (0…+0.30 s) the human excess is −0.005 / −0.008.
- Human detrended coupling r − r_rev: +0.016 [−0.001, 0.036] inner, +0.005 [−0.012, 0.024] outer (human_audit_metrics.csv span) [measured].

**c100 items, source minus human** (T8, plan window, clean):

| source | Δ excess circular, inner [CI] | AUC | Δ events/F0 peak, inner [CI] |
|---|---|---|---|
| EmoFace | +0.001 [−0.180, +0.174] | 0.49 | −0.32 [−0.64, +0.01] |
| EmoTalk | −0.067 [−0.167, +0.030] | 0.40 | +0.19 [−0.44, +0.99] |
| EMOTE | −0.034 [−0.122, +0.048] (outer +0.077 [−0.014, +0.192]) | 0.43 | −0.30 [−0.80, +0.34] |
| MEDTalk | **−0.150 [−0.282, −0.044]** | 0.32 | −0.19 [−0.60, +0.26] |
| F0-height follower | **+0.258 [+0.131, +0.390]** | 0.74 | **−0.62 [−0.76, −0.48]** |
| Energy follower | −0.053 [−0.213, +0.066] | 0.44 | **+0.59 [+0.10, +1.13]** |

**Injection positive control** (plan window, pooled recovery, T9):

| source | clean inner / outer | noise inner / outer |
|---|---|---|
| human (human thresholds) | 0.61 / 0.60 (0.5× size 0.37 / 0.31; 2× 0.82 / 0.83; random-time excess +0.030 / +0.039) | — |
| EmoFace | 0.82 / 0.73 | 0.76 / 0.71 |
| EmoTalk | 0.62 / 0.73 | 0.64 / 0.67 |
| EMOTE | 0.32 / 0.30 | 0.38 / 0.53 |
| MEDTalk | 0.55 / 0.46 | 0.52 / 0.36 |
| F0-height follower | 0.93 / 0.93 | 0.94 / 0.94 |
| Energy follower | 0.40 / 0.40 | 0.20 / 0.20 |

- Raise size = (human median event size / human median W2 amplitude = 0.376 inner, 0.358 outer) × the source's median c100 amplitude.
- Each source has 43 peaks, so these recoveries carry sizeable sampling error; no CI was computed [measured].

**Answers.**
1. **Follow index vs followers.** It separates the F0-height follower from humans and from all four models, under both nulls and with noise. It does not separate the energy follower, which is not peak-locked by construction. Events per F0 peak (≈ 2) and the dynamics slope identify that one [measured].
2. **Dose slope vs deaf models.** It separates both followers from the four models: follower slopes are 0.47–0.54 per ln unit or 0.092/st, with CIs excluding 0, while model brow slopes have |slope| ≤ 0.11 with CIs including 0 [measured].
3. **Follow index vs humans on this ladder.** It does not separate deaf models from humans: 12 cells with 43 peaks give CI half-widths of 0.1–0.2 against a human excess of +0.04. One negative exception is MEDTalk inner.
4. **Is the index usable as a gate?**
   - At human thresholds, human2's injection says yes (0.61/0.60), with a compressed range: base recall 0.66.
   - At model thresholds this lane's injection gives 0.30–0.82. EMOTE (0.30/0.32) and MEDTalk outer (0.46, noise 0.36) fall below 0.5.
   - Conclusion: usable for detecting follower-scale locking. Descriptive for human-scale locking, and for EMOTE/MEDTalk at their own thresholds. The noise arm moves recovery by ≤ 0.23 and changes no verdict [measured].
5. **Detrended coupling** (r − r_rev) is also highest for the F0 follower (+0.336) and within the model spread otherwise: EmoTalk inner +0.168 [0.028, 0.326], EmoFace +0.106, MEDTalk +0.051, EMOTE −0.028 [measured].

## 8. Per language (DESCRIPTIVE; 5 cn / 3 en / 4 jp cells; T10)

- **Effort brow ln ratio (cn / en / jp):**

  | source | cn | en | jp |
  |---|---|---|---|
  | EmoFace | +0.12 | −0.11 | −0.09 |
  | EmoTalk | +0.02 | +0.35 (one cell) | +0.03 |
  | EMOTE | −0.02 | +0.13 [0.05, 0.27] | −0.11 |
  | MEDTalk | −0.07 | −0.04 | −0.10 |
  | F0 follower | +0.44 | +0.41 | +0.41 |
  | Energy follower | +0.12 | +0.08 | +0.21 |

- **Brow span slope.** EmoFace cn −0.04 [−0.08, −0.00]; MEDTalk cn −0.05 [−0.07, −0.04]; every other model × language CI includes 0.
- **Energy-follower jp.** Its dynamics slope is small (+0.11) and its +6 dB response large (+0.16): one possible reason is that the long jp passages have internal pauses where the −20 dB floor clips [U, not checked].
- None of this is a language effect claim. Language is confounded with speaker and text, and cn/jp audio is unheard [V-text: ladder2].

## 9. Problems and caveats

1. **Few F0 peaks.** 43 peaks over 12 cells (cn 1–5, en 1–3); follow-index CIs are wide. Longer utterances would be needed to make G3 decisive on models [measured].
2. **Front-end artefacts in the F0 follower and G2 cues.**
   - extract_prosody's per-file F0 range moves the voiced median on some rate items (cn_hap rate090 −3.9 st). The follower is rectified at the c100 median, so its rate090 ratio reads −0.242 although the rate090 phrase-layer span ratio is 1.003 (median) [measured: diag.json].
   - The same front end gives the G2 register delta +3.39 st for effort, where Praat gives +4.71. The register coefficient is ≈ 0, so the prediction moves only 0.217 → 0.213 [measured].
   - The follower's span slope is attenuated by this front end (§6).
3. **MEDTalk text path varies across rungs** on en/jp (Whisper-zh as shipped) [measured]. `asr="auto"` was not run.
4. **EmoTalk's effort mean rests on one cell** [measured].
5. **Mapping choices.**
   - Rig and ARKit brows: primary = raise controls only; net variants are secondary.
   - EmoTalk lid = eyeWide − eyeBlink.
   - EMOTE landmarks are on the mean face, pose 0, jaw only.
   - EmoFace is scored after post_step (savgol 15/3, as the measure lane); MEDTalk, EmoTalk and EMOTE are scored raw. Smoothing lowers event counts on EmoFace only.
6. **Thresholds use different populations.** Human thresholds come per actor from normal-intensity clips; model thresholds come per source from the 12 c100 items (`reference_threshold`, 10% of median amplitude dominates on smooth model tracks). The √fps noise scaling is an assumption [U].
7. **Rate rungs** are scored on time-normalised frames. The human G2 W2_full coefficients include a window-length duration effect; W2_pm predicts +0.149 for effort. p95−p5 is invariant to a uniform warp in principle, but event rates and velocities are not [code].
8. **Level normalisation.**
   - The flat +6 dB response of all four is measured.
   - Mechanisms: EmoFace uses `do_normalize` [plan, code]; EmoTalk's processors use `do_normalize: True` [V-text: NOTES_emotalk.md].
   - MEDTalk's `Wav2Vec2Processor` / emotion2vec normalisation and EMOTE's processor settings were not checked [U].
9. **Not done.**
   - No reversed-prosody arm for the four models: G2's reversed control is for v2a.
   - r5b and LiltHead v1 were not rescored on ladder2 (plan E3 lists them); their ladder-v1 numbers are in `measure/dose.md`.
   - No per-cell human prediction intervals in the gate (per-cell point predictions exist in analyse2's memory only).
   - No CIs on the injection recoveries.
10. **Launch notes.**
    - The first analyse2 run crashed at the markdown step (`s.diff` is a pandas method); renamed to `diff_est` and rerun.
    - The CSVs written before the crash were overwritten identically.
    - funasr printed "Downloading Model from modelscope" at MEDTalk load; the load took 14 s and ran offline from the cached `iic/emotion2vec_base`. Whether it contacted the network is [U].
11. **Audio unheard.** The ladder audio has not been heard [V-text: ladder2]. Any cn/jp wording depends on a listening check.

## 10. Files (all in `voicedose/audit2/`)

| file | content |
|---|---|
| `common2.py` | ladder2 item list |
| `stage_acoustics.py` → `acoustic/*.npz`, `g2_measures.json` | 60-fps extract_prosody tracks, absolute dBFS, G2 cue measures |
| `run_emoface.py`, `run_ext.py` → `outputs/{emoface,emotalk,emote,medtalk}/*.npz` | raw model outputs (+ EMOTE landmarks, MEDTalk s_t and transcripts); `run_*.json` logs |
| `score.py` → `results.csv` (10,812 rows: item × source × channel × arm), `injection.csv`, `refs.json`, `score_meta.json` | per-item metrics, frozen structure, thresholds, noise σ |
| `analyse2.py` → `dose_tables.csv`, `ratio_tables.csv`, `null_floor.csv`, `g2_predictions.csv`, `g2_compare.csv`, `e3_follow.csv`, `e3_separation.csv`, `e3_injection.csv`, `audit2_tables.md`, `fig_dose_forest.png`, `fig_effort_vs_human.png`, `fig_e3_follow.png` | tables and figures |
| `diag.py` → `diag.json` | follower rate artefact, MEDTalk transcripts, net-brow and leave-one-out effort |
| `g2_predict.py`, `level_def.py` | copies of human2 files (coefficients and draws read from human2, read-only) |
| `asr_cache_medtalk/` | MEDTalk Whisper transcripts (redirected from d1_audit) |
| `*.sh`, `*.log` | launchers (nohup + disown), chain logs |

# ANALYSIS PLAN v6: does the voice-following upper face matter to viewers?

**Status: DRAFT TO FREEZE.**

- **Written** 2026-09-14 before any participant. First draft 14:20 JST; revision 14:40; **final-design revision about 17:55 JST**. The final revision covers: the 14:36 selection, 6 singles, three listener-group links, Experiment 2 as the addendum, the voice check ON, `--hr-normal` OFF, the P2 sensitivity subset and the anticipatory-brow limitation.
- **Companion:** `ANALYSIS_PLAN_V6_EXP2.md` (Experiment 2: three-language pairs, rule layer L; hypotheses P4 and P5). The two files are frozen together (section 11). After the freeze, any change goes into the deviation log (section 12).

Tags: [measured] read or computed from a named file; [code] read from code; [computed] a property of the test, not of data; [U] unverified, an estimate, or her decision.

---

## 1. Research question and its link to the paper

**Question.** With the voice held identical, do viewers judge a talking head's movement to fit that voice better when its brows follow the voice the way the human speaker's brows did?

**Link to claim A** (the audited brows do not follow controlled voice changes; EmoFace's upper face is set by the label):

- E is EmoFace's own output (condition `e10m10_d5_h`, token `before`).
- H keeps E and adds only the brow motion that the same actor produced in the same recording.
- H vs E asks whether what claim A says is missing is visible and matters to a viewer.

**Link to claim B** (strong delivery raises brow motion about x1.33 for the same actor and sentence):

- H carries the actor's own normal/strong brow dynamics, with three limits:
  - only the dynamics: the in-span median is removed before the gain, so a static level difference is not shown [code, `v2a_face.to_rig_units`; BUILD_NOTES section 2];
  - only inside the acoustic speech span (section 7.1);
  - after clamping to the rig's range: combined loss median 20 %, 0-42 %, and 5 of 24 study H clips lose more than 30 % [measured, `stimuli/build/build_stats.csv`].
- P3 asks whether H's advantage grows where the human reference says the brows should move more.

**Not tested here:** claim C (the NO-GO layer is not rendered); the other three models; which vocal cue drives the brows; lids.

H's brow track also follows the words and the articulation of that recording. The study therefore tests "follows this recording as the human did", not "follows prosody".

### 1.1 Voice language, listener groups, and what may be claimed

**Why English voices in Experiment 1.** H needs the actor's own brow track from the same recording, so the voices must come from an audiovisual corpus with the same actor and sentence at two intensities. In this project only RAVDESS provides that [code, build_lilt_v6.py docstring]. The statement is lexically neutral ("Kids are talking by the door", statement 01 [measured, pool_set_v6.json `text`]) and identical in every clip. The words therefore cannot differ between E, H and R, between intensities, or between emotions. The Mandarin and Japanese questions are asked with a rule layer in Experiment 2 (addendum).

**Three listener groups, one design.**

- **Links:** `/root/lilt_cn`, `/root/lilt_jp` and `/root/lilt_en` serve identical stimuli and the identical 81-screen session; only the interface language, the consent's payment section and the recruitment channel differ.
- **Recruitment and payment per link** (server/client patch PAYMENT v6, `payment/apply_payment_v6.py`; config `lilt_payment`; consent says about 20 minutes on all three) [code, payment/README.md]:
  - **cn:** 40 CNY (config `lilt_payment.cn` = "40 元"). The participant contacts the researcher with the username shown on the end screen to claim it; no contact detail is collected by the site.
  - **jp:** a 1,000 JPY Amazon gift card. A new session requires the email address typed twice (matching); the address is stored **only** in `data/payment/contacts.csv` (username, session_id, link), never in the session record, the response CSV or the infos JSON, so it is kept apart from the research data.
  - **en:** recruited through **Prolific**; the amount is set on Prolific (config `lilt_payment.en` = "£4.00 via Prolific"; the Prolific consent states the payment is made through the participant's Prolific account; the amount is her decision [U]). Prolific arrivals open `/root/lilt_en?PROLIFIC_PID=...&STUDY_ID=...&SESSION_ID=...`, get the username `pid_<PID>`, and the server stores the three Prolific ids in the background record. The completion code is released only to a completed `pid_` session on lilt_en with every trial answered (blank code in config = no code is shown). A session on the en link started with the access code instead of a Prolific ID never receives the completion code; it is analysed like any other session of the en group.
  - **Consequence for exclusions (no new rule):** a Prolific return, time-out or any other session that did not reach the last trial is an incomplete session and is excluded under section 5 step 1. Such a session holds its rotation offset for at most 60 minutes (section 2 "Rotation"). Payment status is never an analysis variable.
- **Group** = the **link** the participant entered by (cn / en / jp). The link is recorded exactly: the session record's `path` and the infos file's `test_url` [measured, test914: `/root/lilt_cn` in both]. The server allocates rotation offsets per link (least-filled, section 2 "Rotation") and the stopping rule counts per link, so the group, the rotation and the stopping count are the same variable [code, patches/server_v6.patch, rotation/apply_rotation_v6.py; analyse_lilt_v6.listener_group]. Without an infos link (not expected) the fallback is background `native_lang`, then CSV `culture`.
- A participant may choose a background native language that differs from the link. Such a participant stays in the link's group; the analysis reports the link x native_lang table and lists them (`flow.link_x_native_lang`, `flow.native_lang_differs_from_link`). native_lang "other" is ineligible (section 5).
- **Primary analyses** (P1-P3, MC, VC here; P4-P5 in the addendum) pool the three groups. Stimuli and rotation are identical, and the pair judgement compares two faces with one shared voice, so a group can scale an H-E difference but cannot create one on its own.
- **Secondary:** every hypothesis within each group, with Holm inside the group, reported with the group N.
- **Exploratory:** group x contrast. Group differences in each participant DV are tested by label permutation of the between-group sum of squares, with Kruskal-Wallis alongside.
- The English voices are native for the en group and non-native for the other two, so group differences in Experiment 1 mix culture and language familiarity. No group difference is interpreted as a culture effect.
- English level (`second_lang`, `second_level`) is recorded and used as an exploratory moderator for the cn and jp groups (section 8).

**What may be claimed.** Pooled results are for adult native speakers of Mandarin, English or Japanese recruited online (the cn and jp links by snowball invitation with payment per link, the en link through Prolific; section 1.1). They watched one avatar speak acted North American English [U: RAVDESS documentation] with brow-only edits. Nothing is claimed about a culture difference or about Mandarin or Japanese voices in this experiment. Every paper sentence that reports the study carries this scope (PAPER_INSERT_V6.md).

## 2. Design facts this plan relies on

`build_lilt_v6.py` is authoritative. If its rotation report contradicts this table, fix the table before freezing (section 11).

| item | value |
|---|---|
| Arms per voice clip | E = EmoFace; H = E + the same actor's brow motion from the same recording (inner to raiseIn 31/100, outer to raiseOut 32/101; lids stay E); R = E + that track time-reversed |
| Voice pairs | **12** same-actor normal/strong pairs of female RAVDESS actors: 4 emotions x 3 actors, statement 01 [measured, `stimuli/SELECTION_HISTORY.md` pass 2, 14:36, `selection.json`]: hap 04 / 16 / 14; dis 24 / 02 / 18; fea 10 / **20 (repetition 02)** / **12 (repetition 02)**; ang 06 / 08 / 02. Every pair is >= 2.0 px shown H-E at strong. The weakest pair is ang 02: 2.15 px strong, 1.95 px normal, human ln ratio 0.032 |
| Why two fear pairs use repetition 02 | **actor 20:** both repetition-01 clips failed tracking (the tracking fallback as first written). **Actor 12:** both repetition-01 clips were clean, but the human ln(strong/normal) brow ratio was -0.251, failing "ratio > 1"; repetition 02 gives +0.125 and passes [measured, SELECTION.md section 2; BUILD_NOTES section 6]. The second case is **take selection on the premise variable** (amendment 2 of 14:35, made before any participant). It is disclosed here and in the paper's limitations if the pair is reported |
| Actor changes between selection passes | Pass 1 (14:21): 11 pairs, fea 10 / 02 (shortfall 1, visibility relaxed to 0 px), dis 24 / 20 / 18, ang 06 / 08 / 12, practice rav22_ang_s only. Pass 2 (14:36, the 14:35 amendments: repetition-02 fallback precedence and a third practice tier): fear takes actor 20 (repetition 02) first at cap 1, so **dis replaces 20 with 02**; ang cannot then fill at cap 1 and takes **actor 02 at cap 2 instead of 12** (12 became the fear repetition-02 actor). Actor 02 therefore carries two pairs (dis and ang) [measured, SELECTION_HISTORY.md] |
| Practice clips | **2:** rav22_ang_s (tier 1: actor outside the study) and rav10_hap_s (tier 3: a study actor in a chosen emotion that is not that actor's study emotion; never a study cell). The catch screen uses rav10_hap_s [measured, BUILD_NOTES section 7] |
| Screens (81, identical on all three links) | **1. singles 6** (arm E / H / R rotated; 7-choice emotion + vividness + naturalness), then **2. practice 2** (H vs E), then **3. English AV pairs 36** with **1 catch** mixed in (H vs E @normal 12, H vs E @strong 12, H vs R @strong 12), then **4. Experiment 2 pair_ml 18** (addendum), then **5. silent H vs E @strong 6**, then **6. voice-only check 12**, served last |
| Singles 12 -> 6: the rule | A priori, fixed before any participant and dependent on nothing measured [code, `build_lilt_v6.SINGLES_RULE[6]`, `--singles 6`]: the singles are **the 6 voice pairs of the silent subset**, at strong: rav04_hap, rav06_ang, rav10_fea, rav16_hap, rav18_dis, rav20_fea_r2 [measured, silent subset of the real-selection manifest `study/lilt_link_v6_prov_real_full/manifest.json`]. Every study emotion keeps singles (hap 2, ang 1, fea 2, dis 1), with 2 per actor slot and arms E / H / R rotated 2 / 2 / 2 per participant. **Consequence, written now:** the strong video of these 6 pairs appears in 4 scored screens (single, two AV pairs, silent pair), while the strong video of the other 6 appears in 2 (the AV pairs only). Section 8 reports P1-P3 separately for the two sets (descriptive) |
| Silent subset (6) | a-priori `mute_rule`: within each emotion the actors are sorted by id and take slots k = 0, 1, 2; the pair is silent when (emotion index + k) is even [code, build_lilt_v6.py docstring]. Result above |
| Voice check (ON) | 12 audio-only screens: the normal + strong take of 6 voice pairs, **rav14_hap, rav04_hap, rav18_dis, rav20_fea_r2, rav12_fea_r2, rav06_ang** [measured, `stimuli/voice_check_wavs.txt`]. Rule (`voice_check_select.py`): score = z(strong - normal level, dB) + z(strong - normal median-F0 register, st) over the 12 pairs; the best pair per emotion first, then the 2 best remaining with at most 2 per emotion. Two-pass loudnorm to -23 LUFS, 48 kHz copies |
| `--hr-normal` | **OFF** (her decision). P2b is not analysed. The 12 R renders at normal voices exist in the render list but are not served |
| Renders | 64 served in Experiment 1 (E and H at 24 voices, R at 12 strong voices, E and H for the 2 practice clips), drawn from the 76-line render list [measured, verifier notes]. Experiment 2 uses already rendered clips (addendum section 2) |
| Questions (keys in app.js; en / jp / cn) | pair_q: "Both faces speak with the same voice. Which face's movement fits this voice better?" / 二つの顔は同じ声で話しています。どちらの顔の動きがこの声によく合っていますか？ / 两张脸说的是同一段声音。哪一张脸的动作更配合这段声音？ **mute_q:** "No sound this time. Which face's movement is more lively and expressive?" (jp / cn as in app.js). **catch_q:** "One of the two faces is frozen. Which face is moving?" [code, app.js] |
| Response codes | L3 L2 L1 S0 R1 R2 R3; L3 = the left face is much better [code, server.py `_options_for`] |
| Rotation | For each `pick_group` (sorted by name; i = its index over **all** groups), the server serves `arms[(i + o) mod n_arms]`, where o is the session's **rotation offset** in 0..P-1 and P = the lcm of the arm counts of the manifest actually loaded. Singles have 3 arms and pairs 2, so **P = 6**; 6 single groups and 18 pair_ml groups (group names `pair5_ml1_LE_*`, `pair5_ml2_LR_*`) keep it at 6. **Offset rule (least-filled, server patch LILTROT v6, `rotation/apply_rotation_v6.py`):** a new session on link L takes the offset held by the **fewest** non-test lilt sessions of the **same link**, the same build (a sha1 of the served group-to-arm structure) and the same P that are **completed, or not completed and created less than `lilt_rotation_active_min` = 60 minutes ago**; ties go to the lowest offset, so strictly sequential arrivals get 0, 1, 2, ... as before. Test users (`test*`) receive an offset but are never counted; sessions without a stamp are never counted. If the allocation fails, the session is built with the v6 rule (o = completed non-test sessions of the link) [code, rotation/apply_rotation_v6.py, rotation/README.md]. **Where it is recorded:** only in the session record `data/sessions/<key>.json`, key `lilt_rotation` (`offset`, `period`, `build`, `built`, `link`, `rule`, `active_min`, `counts_before`, `allocated_utc`), joinable to the CSV by `session_id`; it is not in the CSV or the infos JSON. The offset is also recoverable exactly from a participant's served `stimulus_file` set (offsets 0..5 serve 6 distinct 81-file sets [measured, rotation/verify_balance-analysis/offset_recover.log]). The rotation report must confirm P = 6 (section 11) |
| Focus record | per trial, counted only while the trial is on screen: `focus_hidden`, `focus_blurred`, `focus_lost` (their sum), `focus_away_ms` [code, app.js, server.py CSV] |
| Revisions | a re-answer after Back is stamped `revision` >= 1, and the first answer is kept [code, app.js] |
| Audio | the AV pairs carry the 16 kHz EmoFace copy of each wav, identical within every contrast; the voice check uses the 48 kHz copies [measured, BUILD_NOTES section 5]. Whether the bandwidth difference is audible is not measured [U] |

## 3. Dependent variable

**Trial score.**

- Map the response code to v = -3..+3, where + means the right face is better.
- **s = +v when H was on the right, s = -v when H was on the left.** H's side is read from the manifest `cue_manip` "L:x|R:y".
- Only the first response to each stimulus (revision 0) is used.

**Unit of analysis.** One score per participant per contrast.

| score | definition | screens |
|---|---|---|
| m_HE_normal | mean s, H vs E, normal voice | up to 12 |
| m_HE_strong | mean s, H vs E, strong voice | up to 12 |
| m_HR_strong | mean s, H vs R, strong voice | up to 12 |
| m_silent | mean s, silent H vs E, strong voice | up to 6 |
| d_VC | mean over voice-check pairs with both takes retained of intensity(strong) - intensity(normal) | up to 6 pairs |

Side bias cancels within a participant, because H sits on the left in 6 of the 12 screens of each contrast at every rotation slot, on the same side at normal and strong within a voice pair, and on the other side in H vs R [measured, `final/lilt_link_v6_final_prov2/manifest.json` `rotation_check` (81-screen build, 18:30): H left 6/6/6 per contrast, silent 3 of 6, singles 2/2/2, L left 4|5 of 9 per contrast and flipped in 9/9 cells]. The live build's own `rotation_check` is the record (section 11 item 2).

## 4. Hypotheses and tests (confirmatory family = P1, P2, P3)

| | hypothesis | participant DV | H0 |
|---|---|---|---|
| **P1** | H's movement fits the voice better than E's: the deaf upper face matters | D1 = (m_HE_normal + m_HE_strong) / 2 | mean D1 = 0 |
| **P2** | H is preferred over R: alignment with the voice matters, not motion alone | D2 = m_HR_strong | mean D2 = 0 |
| **P3** | H's advantage over E is larger for strong than for normal voices | D3 = mean over the voice pairs answered at both levels of s(H vs E @strong) - s(H vs E @normal) | mean D3 = 0 |
| **MC** | manipulation check: the brow layer is visible without sound | DM = m_silent | mean DM = 0 |

D3 is defined as the paired version, as implemented in `analyse_lilt_v6.p3_values`. With complete data it equals m_HE_strong - m_HE_normal, and after a focus drop it compares only voice pairs present at both levels. This resolves the verifier's note that the plan and the code differed.

**Test.** Two-sided one-sample t-test on the participant DVs.

**Multiplicity.** Holm over the three p-values of P1-P3, family-wise alpha .05. Experiment 2 (P4, P5) has its own Holm family (addendum section 4).

**Support rule.** Holm-adjusted p < .05 **and** mean > 0. A significant opposite sign is reported as contradicting the hypothesis.

**Manipulation check.** Outside the family. It passes when mean DM > 0 with unadjusted two-sided p < .05. It does not decide P1-P3, but it decides how their nulls are read (section 7).

**Reported for every DV:**

- mean and SD, and dz;
- 95 % participant-bootstrap percentile CI (B = 10,000, seed from the contrast name);
- Wilcoxon signed-rank p;
- the counts with DV > 0 and < 0, and the sign-test p.

**Robustness.** The t-test decides. When Wilcoxon disagrees about significance at the same Holm step, the text says "not robust to the test choice". The bootstrap CI is descriptive; its null coverage in this script's simulation is checked by the self-test (section 11).

**Generalisation over stimuli** (reported, not decisive): the same DVs as one mean per voice pair, with a bootstrap CI over the 12 pairs. If that CI includes 0 while the participant test is significant, the paper says the effect is not shown to generalise over voices.

**Also reported:** side bias (mean raw v over the AV pairs) and the tie rate.

### 4.1 Voice check (VC; ON): listeners hear the strong take as more intense than the normal take

- **Stimuli:** block `voice_only`, served last; section 2 lists the pairs and the rule. Audio over a black frame, in the same container as the AV clips. Each clip is gain-matched to -23 LUFS on the QC meter (ffmpeg ebur128) and the build refuses any clip outside -23 +/- 0.5 LUFS. The quiet wav of rav20_fea_r2 normal measured -24.0 LUFS with loudnorm and -23.2 LUFS with the 18:30 linear-gain fix [measured, `final/finish/voicefix_test`]; the live build's `media_qc.lufs_voice_only` is the record, so the normal and strong takes of a pair differ by at most 1 LU.
- **Level in the AV pairs (Experiment 1), recorded before hashing:** in the provisional build the normal take played 0.4-0.9 LU louder than the strong take in the real English renders (e.g. rav04_hap -20.1 vs -21.0 LUFS) [measured, provisional `media_qc.lufs_normal_vs_strong`]. This works against P3 (conservative). The main session copies the live build's realised normal-minus-strong range into section 12 before hashing.
- **Responses:** 7-choice emotion (ang sad hap sur fear dis neu) and intensity 1-7, anchored on the chosen emotion; naturalness is hidden [code, app_js_v6.patch].
- **DV:** d_VC. **Test:** H0 mean d_VC <= 0 against H1 > 0, one-sample t **one-sided**, alpha .05, with Wilcoxon as robustness, a participant-bootstrap CI, and the count with d_VC > 0. **Pass** = one-sided p < .05 and mean > 0. (The earlier 8a table's "two-sided" for this test is withdrawn; one-sided is the rule.)
- **Descriptive:** 7-AFC accuracy (response == the RAVDESS label) per level and per emotion, the confusion table, and the per-voice-pair difference.
- **Sensitivity:** one-sided .05 at 80 % power needs dz >= 0.52 at N = 24 and dz >= 0.77 at N = 12 [computed, power_sensitivity method].
- **Caveats written now:**
  - The mux loudness-normalises every clip, so the level cue of vocal intensity is removed. The step must be heard in F0, voice quality, rate and duration.
  - The 6 voice-check pairs were chosen for the largest level + register steps (section 2), so VC is a best-case check.
  - Every voice was heard before this block, and both takes sit in one block, so VC is a within-block manipulation check, not a naive intensity measurement.
- **What a failed VC means for P3:**
  - (a) a P3 null is **uninformative** about whether the face should follow vocal intensity;
  - (b) a P3 positive is read as the visual-difference-size effect of section 7;
  - (c) P1 and P2 are unaffected.

### 4.2 P2 amplitude sensitivity analysis (pre-registered, descriptive, outside the family)

- **Why.** R is built from H's input track reversed exactly inside the speech span (24 of 24 [measured, `stimuli/build/verify_arrays_v6.csv`]). The *shown* motion is not exactly equated, because the clamp bites differently after reversal: shown R/H amplitude 0.68-1.07, and r(shown H, shown R) median -0.33 inner / -0.24 outer, positive on 8 of 24 cells [measured, same file; BUILD_NOTES section 3].
- **Rule.** An R cell is a clean motion-equated timing control when the shown R/H amplitude ratio (p95-p5 in the speech span) is in **[0.80, 1.25]** on both brow channels **and** r(shown H, shown R) **< 0.30** on both channels.
- **Clean strong cells, 8 of 12** [measured, `stimuli/build/p2_sensitivity_cells.json`, written before any participant]: rav02_ang_s, rav06_ang_s, rav08_ang_s, rav12_fea_r2_s, rav16_hap_s, rav18_dis_s, rav20_fea_r2_s, rav24_dis_s.
- **Not clean:**
  - rav02_dis_s: r 0.72 inner / 0.63 outer
  - rav04_hap_s: r 0.36 inner
  - rav10_fea_s: R/H inner 0.68
  - rav14_hap_s: r 0.39 inner
- **Analysis.** P2's DV restricted to H vs R @strong screens on the 8 clean cells: participant mean, bootstrap CI, t p (descriptive). The cells are named here, so hashing this plan fixes them.
- **Reading.** A pooled P2 null with a clearly positive subset estimate points to the imperfect control, not to "timing does not matter". A positive P2 with a subset estimate near 0 is reported as "not robust to restricting R to motion-equated cells".

## 5. Exclusions (fixed order; every count reported; shared with Experiment 2)

1. **Not a participant:** usernames starting with "test"; sessions that never reached the last trial (the server files the CSV only on completion); a person's second session (keep the first, from her recruitment log).
2. **Not eligible** (background form): `native_lang` not in {cn, en, jp} (i.e. "other"); `hearing` = impaired; `vision` = other. "Prefer not to say" is kept. A missing background record is kept and counted.
3. **Catch failed or missing:** the one "which face is moving" screen answered with the frozen side, or not answered. Those participants leave the primary sample. Everyone surviving steps 1, 2 and 4 forms the sensitivity sample ("all").
4. **Focus, per scored screen** (singles, pairs, pair_ml, silent pairs, voice-only; never the catch):
   - **Primary:** drop a screen with `focus_hidden` >= 1. A blank field counts as no loss and is counted.
   - **Sensitivity:** drop every screen with `focus_lost` >= 1 and re-report P1-P3, MC and P4-P5.
   - **Participant rule:** a participant with fewer than **75 %** retained (9 of 12) in any of the three primary AV contrasts is excluded from **all** analyses, Experiment 2 included. Fewer than 4 of 6 retained silent screens drops them from MC only. Fewer than 4 voice pairs with both takes retained drops them from VC only. The Experiment 2 rule (< 75 % in L_vs_E or L_vs_Lrev, Experiment 2 only) is in the addendum, section 5.
5. **Nothing else:** no exclusion by response time, replays, response pattern or outcome; practice screens are never scored.

## 6. Sample size, listener groups, rotation and stopping

**Target.** N = 24 valid participants **per listener group** (after section 5), which is 4 full rotation cycles of 6 per group. The pooled target is 72.

- Rotation offsets are allocated **per link** (least-filled, section 2 "Rotation"), and the group is the link (section 1.1), so 24 valid participants in a group aim at 4 per offset within that group, and the pooled sample holds 3 x that balance. Report `realised_balance` pooled and per group: H-left share per contrast, single-arm counts per voice pair, participants with identical arm/side vectors, and the completed sessions per offset per link (from the `lilt_rotation` stamps, cross-checked against the offset inferred from the served files). The audit is descriptive only: no hypothesis test, estimate, exclusion or stopping rule depends on it. It joins each session's `lilt_rotation` stamp to the response data by `session_id` (usernames are not used); a session without a stamp gets the unique offset whose served stimulus set contains its stimuli. Because abandoned places hold an offset for up to `lilt_rotation_active_min` minutes and excluded participants still consume one, some imbalance is expected: per (link, build) the audit warns only when one offset holds more than ceil(n_link / P) + 2 participants (P = 6, from `lilt_rotation.period`) or a per-clip single-arm split exceeds 4 within a link; pooled over links both allowances are summed. A link where more than half of n >= 4 participants share one offset is flagged as a collapsed rotation, and a participant whose stamped offset differs from the offset actually served is flagged. Flagged imbalance is reported, not corrected, and excludes nobody [code, analyse_lilt_v6.py md5 d0c5e543; simulation W = 60, 20 % abandonment, 24 or 48 per link within 30 min or 3 h: least-filled warns in 0/60 runs per condition, the old completed-only rule in 60/60, `final/analysis_rotation/NOTE.md`].
- The allocation counts **completed sessions and sessions started less than 60 minutes ago** (not completed sessions only), so simultaneous starts on one link are spread over the offsets. A session abandoned for 60 minutes or more frees its offset. Because open and later-abandoned sessions hold an offset while they are recent, completers are expected to be near-equal rather than exactly equal per offset: a difference of up to 2 between offsets, and a per-clip arm split of up to 2, are the expected result of this allocation [measured, rotation/sim/sim_rotation.log and rotation/verify_balance-analysis/sim_verify.log: median worst split 2 for 24 arrivals within 30 min]. The balance warnings are descriptive; they exclude nobody.
- Catch and focus exclusions still consume a rotation slot.
- Balance within a participant (H left in 6 of 12 per contrast) holds at any N.
- Staggered invitations are not required for balance; they remain allowed.

**Stopping rule, per group.** Each group stops inviting at whichever comes first:

- **(a)** its 24th valid participant completes; or
- **(b)** **16 Sep 2026, 22:00 JST** (the same for all groups).

A group that reaches (a) stops while the others continue. At (b), all invitations stop at 22:00, and every session completed by 22:30 JST (last `response_ts`) is analysed. Report the N reached per group and a flow count for every exclusion step, pooled and per group.

**No interim look.** Before the stop she may monitor completions, per-group N, catch passes, focus counts, balance warnings and media errors only, never the pair scores. The analysis runs once, after the stop.

**Paper inclusion, decided now, independent of the outcome (her decision, 14 Sep 2026 ~19:40 JST, before any participant).**
- With at least 12 valid participants pooled (section 5 exclusions), a passed live self-test and a passed silent visibility check (MC), Experiment 1 enters the ICASSP paper whatever the results: positive, mixed or null, using the matching PAPER_INSERT_V6.md paragraph; a null is written as "not detected; only large effects are excluded".
- If the silent visibility check fails, the study does not enter the paper, which keeps "no perceptual experiment is reported" (the P1-P3 results are still analysed and reported elsewhere as uninformative).
- With fewer than 12 valid participants pooled at the stop, the paper keeps "no perceptual experiment is reported".
- Experiment 2 appears in the ICASSP paper in at most one sentence, whatever its result; its full report is left to a later paper.

**Sensitivity** [computed, two-sided one-sample t, 80 % power; `power_sensitivity.log`, `exp2/exp2_power.log`]:

| N | alpha .05 | alpha .025 | alpha .0167 (Holm step 1 of 3) |
|---|---|---|---|
| 12 | dz >= 0.89 | 1.01 | 1.08 |
| 24 (one group) | dz >= 0.60 | 0.67 | 0.71 |
| 48 | dz >= 0.41 | 0.46 | 0.48 |
| 72 (pooled target) | dz >= 0.34 | 0.37 | 0.39 |

A non-significant result excludes only effects above these values and is written "not detected at this sample size", never "no effect".

### 6.1 Session and rotation facts to confirm at the build

- **81 screens.** The client estimate is `Math.round((n*11 + 150)/60)` = **17 min** [code, app.js via build_lilt_v6.client_minutes]; the instructions must show that number.
- **Rotation period 6** per link (offset allocated least-filled per link by the patched server, stamped in `lilt_rotation.period`).
- **Exp 1:** H left 6 of 12 per contrast at every slot, H on the same side at normal and strong in a voice pair and on the other side in H vs R, singles 2 / 2 / 2 per participant, silent 3 / 3.
- **Exp 2:** L left 4 or 5 of 9 per contrast, mirrored in consecutive slots (addendum section 3).
- The rotation report of `build_lilt_v6.py` at 12 and 24 participants is authoritative (section 11 item 2).
- **Breaks differ by link (her decision, 14 Sep ~20:45):** `/root/lilt_cn` and `/root/lilt_jp` keep the pause button and have no fixed rest; `/root/lilt_en` (Prolific) has no pause button and one automatic, non-extendable 60-s rest after the last English pair screen. The screen order and content are identical on the three links (section 12).

## 7. What a null means, and limitations

**Reading each pattern:**

- **MC fails** (silent H vs E not reliably > 0). The brow layer was not visibly different at the rendered size, so P1-P3 nulls are **uninformative**, and the paper says so. A significant P result is still reported with that caveat.
- **MC passes, P1 null.** Visible brow motion from the actor's own recording did not make the face a better judged fit to the voice. Claim A then stands as a finding about model behaviour, not yet shown to matter to viewers, and only for these stimuli.
- **P1 supported, P2 null.** The preference may come from added motion rather than its alignment with the voice. Read section 4.2's subset estimate alongside.
- **P2 supported, P1 null.** Misaligned brow motion is judged worse than aligned motion, while aligned motion is not judged better than none.
- **P3 null.** No evidence that H's advantage grows with vocal intensity (read with VC, section 4.1).
- **P3 supported.** This is consistent with the human reference, but it cannot separate "the face should follow intensity" from "a larger H-E difference is easier to see":
  - H's added motion is larger at strong voices: shown H-E is smaller at normal than at strong in 11 of 12 pairs, the exception being rav24_dis [measured, SELECTION_HISTORY.md pass 2]; the medians are normal 3.26 / 1.94 px vs strong 4.79 / 4.41 px [measured, BUILD_NOTES section 4];
  - pairs were selected for a human ratio > 1;
  - silent pairs exist only at strong.
  The per-emotion human ratio CI includes 1 for sad and surprise [measured, ledger_dose H2], so P3 rests on the selected pairs.
- **VC fails:** section 4.1.

### 7.1 Limitations written before data (the paper carries the relevant ones)

1. **Anticipatory brow motion is not shown.** The H layer is multiplied by EmoFace's hold weight: 0 outside the acoustic speech span, with a raised-cosine ramp over the first and last 10 rig frames (167 ms) [code, `v2a_face.hold_weight`]. In the 300 ms before the span, the human brows deviate from their in-span median by a median 2.9 px (inner) and 2.6 px (outer), that is 57 % and 64 % of the in-span p95-p5. They exceed the in-span amplitude in 4 of 26 clips (inner) and 6 of 26 (outer), e.g. rav20_fea_r2_s inner 12.9 vs 4.9 px. The onset ramp attenuates deviations of up to 10.9 px (rav04_hap_s) [measured, `verify_stimulus-validity/pre_onset_motion.log`]. None of that pre-onset raise is shown. H is therefore "the actor's own brow motion during speech", and R reverses the same span, so P1-P3 stay internally valid. But **claim B's human brow response is tested with its anticipatory part removed**, and a null cannot speak for brow motion that leads the voice.
2. **Clamp loss:** a median of 20 % of the intended H motion is lost, 0-42 % [measured]; for fear and disgust the loss is largest (EmoFace holds raiseIn near 1).
3. **Take selection on the premise variable:** the actor-12 fear pair uses repetition 02 because repetition 01 failed "ratio > 1" (section 2).
4. **Retargeting gain:** the between-actor spread of the gain is x0.8 to x2.5 [measured, CALIBRATION via HANDOFF_V2A section 2]; two strong clips sit just above the 8 px top of the human band (rav12_fea_r2_s outer 8.81 px, rav04_hap_s inner 8.37 px) [measured].
5. **One avatar**, brows only, 12 acted English voices of female actors, one sentence, loudness-normalised audio.
6. **The alignment decision (lag 0).** Both streams start at 0 in every RAVDESS file. The pooled mouth track leads the audio by about 80 ms, which is normal articulatory anticipation, so no shift was applied [measured, `stimuli/align_pooled.json`; BUILD_NOTES section 1].
7. **Order:** singles before pairs, Experiment 2 between the English AV pairs and the silent pairs, and the voice check last. Fixed, not counterbalanced.
8. **Unequal exposure:** the 6 single / silent-subset voice pairs' strong videos are seen in 4 scored screens, the other 6 pairs' in 2 (section 2). A P1-P3 difference between the two sets is reported descriptively and would not be interpretable as a voice effect.

## 8. Exploratory (labelled exploratory; no multiplicity correction; no confirmatory claims)

- **Listener groups:** every DV per group (secondary, section 1.1), group differences by permutation, and realised balance per group.
- **Singles by arm E / H / R** (6 per participant, 2 per arm): emotion accuracy (7-AFC), vividness, naturalness, and the within-participant differences H-E, R-E, H-R with bootstrap CIs. Naturalness is the likely cost side of H.
- **Breakdowns:** P1-P3 and the silent pairs per emotion, per actor, per voice pair; P1-P3 for the 6 voice pairs with singles vs the 6 without (from the per-voice-pair table; section 7.1 item 8).
- **Per voice level:** m_HE_normal and m_HE_strong tested separately.
- **Motion size:** whether s tracks the H-E brow amplitude and the clamp loss of each clip (long CSV; e.g. s ~ amplitude + (1 | participant) + (1 | voice pair)).
- **Other responses:** response time, replays, tie rate; the "all participants" sample and the `focus_lost` sensitivity.
- **Background:** English level (`second_level`) as a moderator within the cn and jp groups.
- **VC:** per emotion and per voice pair; accuracy by level and emotion; whether a participant's d_VC predicts their D3 (descriptive).

## 9. How the paper would read

The paragraph versions for each outcome, the listener-group sentence, the Experiment 2 paragraphs and the page-budget cuts are maintained in `PAPER_INSERT_V6.md`. That file is wording only and is not frozen; it may never change a hypothesis, test, exclusion or stopping rule, which live here and in the addendum.

**Sentences that must change if the study is reported** (the paper is read-only from this lane):

| file | current text | change |
|---|---|---|
| `intro_related.tex` P7 | "no perceptual experiment is reported." | contribution (v): a perception test of whether a voice-following upper face matters to viewers |
| `main.tex` ethics | "No human participants were recruited." | name the online study, consent, ethics approval, and recruitment and payment per link: cn 40 CNY, jp a 1,000 JPY Amazon gift card (email kept apart from the data), en through Prolific (amount set on Prolific). **Blocker [U]:** confirm the approval covers the three links and English and Japanese participants **before** recruiting |
| `discussion_conclusion.tex` Limitations | "a perceptual test with rendered faces, ..." | drop that clause; add the scope sentence (PAPER_INSERT section 2 / 6.4) |
| `discussion_conclusion.tex` Summary / Conclusion | - | one clause per outcome |
| `abstract.tex` | 152 words [measured] | one clause, paid for inside the abstract |
| `HANDOFF_PAPER.md` section 7 (zh) | 本文不含感知实验 | update |

## 10. Page budget

See `PAPER_INSERT_V6.md` sections 5 and 6.5 [measured word counts there]. In short: Experiment 1 with Method C needs about 15-17 lines on pages 1-4, and Experiment 2 about 10 more. The HANDOFF_PAPER section 4 cuts cover roughly half of that. Which retreat option applies to Experiment 2 is decided before the result.

## 11. Before freezing (all of it before the link goes live)

1. **Stimulus record:**
   - Experiment 1: 12 voice pairs, 2 practice clips, the silent subset, the 6 singles and the 6 voice-check pairs exactly as in section 2 [measured: `selection.json` 14:36, `voice_check_wavs.txt`, BUILD_NOTES].
   - Experiment 2: the 9 cells of addendum section 2.3.
   - If the live manifest differs, correct sections 2 and 4.2 and the addendum before hashing.
2. **Builder and rotation.** The live build runs `build_lilt_v6.py --singles 6 --ml --voice-check` (`--hr-normal` OFF) [code, build_lilt_v6.py docstring, 17:42]. Its rotation report at 12 and 24 participants must show section 6.1: 81 screens, cycle 6, H balance, singles 2 / 2 / 2, Exp 2 L balance with L flipped between a cell's two contrasts. Its measured Exp 2 cell table (manifest `ml.table`) must select the 9 cells of the addendum section 2.3; if it does not, correct the addendum before hashing.
3. **Server and client.**
   - `patches/server_v6.patch` and `app_js_v6.patch` (17:36) are applied with a guarded restart. They serve `pair_ml` after `pair` + `pair_catch` and before `pair_mute`, `voice_only` last, and count n_prior per link (the fallback rule); on top of them `payment/apply_payment_v6.py` (payment per link, section 1.1) and `rotation/apply_rotation_v6.py` (least-filled offsets, section 2 "Rotation") are applied, with the config keys `lilt_payment`, `lilt_prolific` and `lilt_rotation_active_min` (60), before the same single guarded restart; the app splices the Exp 2 instruction bullet (speech in Chinese, Japanese and English) only when pair_ml screens are served [code, patches/README.md].
   - The unpatched live server drops `pair_ml` and `voice_only` silently [code].
   - The live self-test must pass on the patched server, with all 81 screens served in order on each of the three links: `study/lilt_selftest_v6.py --live --link cn|jp|en` (updated 18:30 for pair_ml, 81 screens and per-link paths). The dry run of the same script against the live manifest (`--manifest /mnt/d/SPJ/pipeline/pilot/lilt_link/manifest.json`) must end with every check passed.
4. **R check:** done. R is exactly reversed on the input track (24 of 24), and the sensitivity subset is fixed in section 4.2.
5. **Analysis script:** `analyse_lilt_v6.py --selftest` passes on simulated data only. At 18:3x JST, 78/78 checks passed, including section G: 81-screen sessions from the three links, Experiment 2, listener group = link (a jp native on the en link stays in en), the per-emotion voice-check difference for fear, and null CI coverage for P1-P5 [measured, `final/finish/run9_selftest_link_groups.log`]. After the rotation-audit edits, 85/85 passed on the final file (md5 d0c5e543, 19:29 JST) [measured, `final/analysis_rotation/run2_selftest.log`]. The freeze step (item 7) reruns `--selftest` on the exact file being hashed and cites that log (`final/finish/main_5_analysis_selftest.log`). Its report starts by re-checking `PLAN_FREEZE.sha256`.
6. **Her decisions [U]:**
   - the paper-inclusion threshold: DECIDED 19:40, >= 12 valid pooled, any result; silent check failed -> not in the paper (section 6);
   - the Experiment 2 paper option: DECIDED 19:40, at most one sentence (section 6);
   - ethics approval coverage (three links, three languages, payment per link as in section 1.1, Prolific for en);
   - the en amount on Prolific and the Prolific completion code in config `lilt_prolific.completion_code`;
   - the per-group stopping rule as written.
7. **Freeze** (Git Bash, main session, after the final live build and after the live swap; nothing in these files may change after this step):

       S=/d/SPJ/pipeline/prosody_ladder/_explore/voicedose/study_v6
       P=/d/SPJ/pipeline/prosody_ladder
       bash "$S/final/finish/main_5_freeze.sh"

   The script (1) refuses a provisional or partial live manifest, or one whose `ml.cells` differ from addendum section 2.3; (2) writes the live build's normal-minus-strong LU range and voice-only LUFS range into section 12 of this plan; (3) reruns `analyse_lilt_v6.py --selftest` on the file being hashed; (4) runs `sha256sum "$S/ANALYSIS_PLAN_V6.md" "$S/ANALYSIS_PLAN_V6_EXP2.md" "$P/build_lilt_v6.py" "$P/analyse_lilt_v6.py" /d/SPJ/pipeline/pilot/lilt_link/manifest.json "$S/stimuli/build/p2_sensitivity_cells.json" "$S/exp2/exp2_cell_rule.json" > "$S/PLAN_FREEZE.sha256"`; (5) writes `PLAN_FREEZE.frozen_at` (time, flags, served blocks, the analysis self-test log and count, and an md5 record of the deployed `webapp/server.py`, `static/app.js` and `config.json`; a record, not a gate, because the Prolific completion code may still be pasted into config.json); (6) runs `sha256sum -c`.

   `PLAN_FREEZE.sha256` holds exactly **seven** lines: this plan, the Experiment 2 addendum, the builder, the analysis script, the live manifest, and the two cell lists the analysis reads at run time (`stimuli/build/p2_sensitivity_cells.json` for P2 section 4.2, `exp2/exp2_cell_rule.json` for the P5 subset). `sha256sum -c` must print seven `OK`. The live self-test (item 3) runs right after the freeze and changes none of the seven files. The first participant must start after the time in `PLAN_FREEZE.frozen_at`, and the analysis reports that time next to the first `onset_ts`. `PAPER_INSERT_V6.md` and the `exp2/` measurement files are not hashed; the values that decide anything (the cells, the sensitivity subsets) are written into the two hashed plan files, and the two JSON cell lists the analysis reads are hashed too.
8. **Overwriting the freeze** is allowed only while no participant has started. After the first participant, the file is never edited: a later change writes `PLAN_FREEZE_2.sha256` and a section 12 entry.

## 12. Deviations

- Any change after the freeze goes into `study_v6/DEVIATIONS_V6.md` with the time, what changed, why, and whether any pair score had been seen. It is reported in the paper.
- A deviation made after the scores were seen turns the affected test into an exploratory one.
- `PLAN_FREEZE.sha256` is never edited after the first participant; a failed `sha256sum -c` at analysis time is itself a deviation to be explained.
- **Recorded pre-data changes** (not deviations; all before any participant):
  - the 14:35 selection amendments (repetition-02 fallback precedence, practice tier 3);
  - `validate_selection` aligned to the practice rule (15:15);
  - singles 12 -> 6;
  - 18:30: listener group = the link (was background native_lang; her design names the link groups); voice-only clips gain-matched on the QC meter (the rav20_fea_r2 normal clip was -24.0 LUFS); both before any participant;
  - the Experiment 2 addendum;
  - the three listener-group links, with n_prior counted per link (server patch 17:36);
  - ~19:00: rotation offsets allocated least-filled per link over completed and recent (< 60 min) sessions, stamped in the session record (LILTROT v6; replaces "completed sessions only", which put simultaneous starts on one list); payment and recruitment per link (PAYMENT v6: cn 40 CNY, jp 1,000 JPY Amazon gift card with the email kept separate, en through Prolific); both before any participant;
  - the voice check ON and `--hr-normal` OFF.
  - ~20:50 (before any participant): the six rav20_fea_r2 AV renders came out of the pool_render mux too loud (normal -18.5 LUFS, outside the build's -26..-19 window; strong -19.5; every other English render -20.3..-22.4), because single-pass loudnorm over-boosted a very quiet source take. One linear -2.0 dB gain was applied to the audio of all six (video stream copied, frame counts unchanged, the same gain for normal and strong, so the pair keeps its own normal-minus-strong difference); originals in `final/loudfix_rav20_backup/`. The live build then passed media QC (-23.2 to -19.6 LUFS, 0 problems).
  - ~20:45, her decisions (before any participant): the English reward is £5.00 on Prolific; sentences promising payment for a partly completed session are removed from the lilt links in all three languages; on `/root/lilt_en` only there is no pause button and one fixed, non-extendable 60-s rest after the last English pair screen (screen 45 of 81), applied before that link opens. `/root/lilt_cn` and `/root/lilt_jp` keep the pause button and have no fixed rest. This procedural difference between links bears only on the exploratory listener-group comparisons (section 8) and is reported with them. As implemented (app.js patch LILT UI2 v6, 21:20, before any participant): the lilt_en rest starts after the participant submits the screen just before the first pair_ml screen (after screen 45 of 81); it counts as taken when it starts (saved through the existing session rest_state), so a reload during it ends it and the rest is at most 60 s; it is logged as session events `lilt_rest_start` / `lilt_rest_end`; screen order, rotation, scoring and CSV columns are unchanged. On all three links the withdrawn screen no longer shows payment wording after a comment is sent. ~21:30, her decision (before any participant; only test* sessions existed): `/root/lilt_en` states **about 15 minutes** (its consent in the en, ja and zh interface, and its instructions screen), matching the 15-minute estimate entered on Prolific; `/root/lilt_cn` and `/root/lilt_jp` keep about 20 minutes in the consent and the computed 17 minutes on the instructions screen. The session content is unchanged; the build estimates about 19 minutes and her own test run took about 9 minutes.
  - LIVE BUILD RECORD (main_5_freeze, 2026-09-14 21:34; live manifest built build_lilt_v6.py 2026-09-14 20:56): normal-minus-strong level in the Experiment 1 AV pairs -0.1 to +1.2 LU over the 12 voice pairs (rav02_ang +0.4, rav02_dis +0.7, rav04_hap +0.9, rav06_ang -0.1, rav08_ang +0.6, rav10_fea +0.0, rav12_fea_r2 +0.4, rav14_hap +0.2, rav16_hap +0.3, rav18_dis +1.0, rav20_fea_r2 +1.2, rav24_dis +0.5); voice-only clips -23.2 to -23.0 LUFS (12 clips) [measured, live manifest `media_qc.lufs_normal_vs_strong`, `media_qc.lufs_voice_only`]; recorded before hashing, not a deviation.

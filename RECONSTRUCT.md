# Reconstructing the results

Two levels, depending on how far back you want to start.

## Level 1 — check every reported number (no licences needed)

Every audited measurement and every plotted value is in `data/`, and the de-identified perception
responses are in `data/perception/`. No corpus, model or render is required for them. The frozen
perception analyser reads the original per-session records, which carry recruitment-platform IDs
and are not released; `data/perception/perception_scored.csv` is its per-screen output for the
analysed sessions, so the perception results can be recomputed from it directly:

```python
import pandas as pd

d = pd.read_csv("data/perception/perception_scored.csv")
d = d[d.in_primary == 1]
m = d.pivot_table(index="participant", columns="contrast", values="score_H", aggfunc="mean")
print("visibility check (no sound)", round(m["mute:H_vs_E@strong"].mean(), 3))
print("P1 H vs E     ", round(((m["H_vs_E@normal"] + m["H_vs_E@strong"]) / 2).mean(), 3))
print("P2 H vs R     ", round(m["H_vs_R@strong"].mean(), 3))
print("P3 strong - normal", round((m["H_vs_E@strong"] - m["H_vs_E@normal"]).mean(), 3))
```

These are the participant means; the confidence intervals and Holm-adjusted p values come from
the analyser's bootstrap and are listed in `data/derived/figure_perception_values.csv`.

```bash
python -m pip install numpy scipy pandas statsmodels matplotlib

# the perception experiment, exactly as pre-registered -- reads the per-session response
# records, which are not in this repository; --help shows what it expects
python code/perception/analyse_perception.py --help

# the four checks below read the same per-session records, so they cannot be run from this
# repository either; they are here as the code behind the reported robustness results

# a score-blind engagement audit of every completed session
python code/perception/engagement_audit.py

# the same result under eight nested exclusion sets; it re-derives the frozen
# analyser's own output first and refuses to continue if it cannot
python code/perception/robustness_specification_curve.py

# crossed participant x item mixed model, min-F', and the normalisation variants
python code/perception/mixed_model.py
python code/perception/mixed_model_family.py

# redraw the figures from the released values, into figures_out/
python code/figures/make_figures.py figures_out/                # protocol diagram, dose response
python code/figures/make_perception_figure.py --check-scored    # the perception figure

# confirm the pre-registration is untouched
python verify_freeze.py
```

`data/derived/figure_dose_values.csv` and `data/derived/figure_perception_values.csv` carry every
value plotted in the figures, each with the key it came from, so a figure can be checked without
rerunning anything. `--check-scored` goes one step further and recomputes the perception figure's
values from `data/perception/perception_scored.csv` before drawing, stopping if any of them
disagrees. That figure's panel (a) is three crops of a rendered frame; the renders are not part of
this release, so the panel is drawn empty at its published size unless `--frames` points at them.

## Level 2 — rebuild the stimuli from the corpora

You need the corpora yourself; see `NOTICES.md` for how each is obtained. Then:

1. **The ladder.** `code/ladder/build_ladder.py` resynthesises each recording into its rungs with
   TD-PSOLA in Praat. `code/ladder/ladder_manifest.json` records every rung's target and what was
   actually achieved, so you can check the rebuild against ours rung by rung.
2. **The models.** `code/audit/run_emoface.py` and `run_ext.py` run each model on the ladder at
   its shipped settings. The models are not included; install them from their own repositories.
3. **The readout.** `code/audit/score.py` maps each model's output to inner brow, outer brow,
   upper lid and jaw, and computes the ln amplitude ratios against `c100`.
4. **The analysis.** `code/audit/analyse2.py` produces `data/derived/audit_results.csv` and the
   dose tables.
5. **The human reference.** `code/human/g2_predict.py` fits the same-actor prediction from
   RAVDESS video.

Paths outside the frozen files are written as `${PROJECT_ROOT}`, and paths into the corpora as
`${CORPUS_ROOT}`; set both to your own locations. The four frozen files that contain paths keep
the absolute paths they had when they were hashed — changing them would break `verify_freeze.py`,
which is the point of shipping them.

## What you cannot reconstruct from here

**The rendered stimulus videos.** They combine corpus audio with a rendered character. The IEMOCAP
and ESD audio may not be redistributed; the RAVDESS and JVNV audio could be, under their share-alike
licences, but is left to the source releases; and the renders themselves are not distributed — the
two silent, cropped demo clips in `media/` are the only rendered frames here.
`data/derived/served_stimuli_manifest.json` records exactly which file was served on every screen,
so the design is fully inspectable even though the media are not included.

**The per-session perception records.** They carry recruitment-platform IDs, session IDs and
timestamps, so they are not released, and the frozen analyser cannot be run end to end from here.
The de-identified responses and the analyser's per-screen scores are in `data/perception/`.

# Tool Condition Monitoring in CNC Milling using Sensor Data and Machine Learning

> Can a worn cutting tool be detected purely from a CNC machine's own motor signals — without ever stopping the job to inspect the tool? This project builds two complementary ML pipelines to answer that, and reports an honest result rather than an inflated one.

## Overview

CNC controllers log motor-level telemetry (position, velocity, current, voltage, power) at every axis during a job. This project tests whether that signal carries a learnable "fingerprint" of tool wear — the basis of real **Tool Condition Monitoring (TCM)** systems in Industry 4.0 used to cut unplanned downtime, scrap, and manual inspection cost.

Two modelling strategies are built, evaluated, and compared:

| | Approach A | Approach B |
|---|---|---|
| Granularity | One feature vector per **experiment** (18 samples) | One feature vector per **timestep** (~17,500 samples) |
| Features | Hand-engineered time/frequency-domain statistics (mean, RMS, skew, kurtosis, FFT) | Raw multi-sensor readings |
| Validation | Leave-One-Out CV with feature selection nested inside each fold | Stratified, **group-aware** train/test split (whole experiments held out) |
| Models | Logistic Regression, Random Forest | Random Forest, XGBoost |

## Dataset

**CNC Mill Tool Wear Dataset** — collected at the System-level Manufacturing and Automation Research Testbed (SMART Lab), University of Michigan.

- 18 experiments engraving an "S" pocket into a wax block, varying feed rate (3–20 mm/s) and clamp pressure (2.5–4.0 bar), using a fresh or visibly worn tool (8 unworn / 10 worn).
- `train.csv` — process parameters + ground-truth labels per experiment.
- `experiment_01.csv` … `experiment_18.csv` — 48-channel CNC motor telemetry at 100 ms resolution, including a machining-stage label.

**Source (no Kaggle API key needed — fully reproducible in Colab):**
`https://github.com/SaeedShurrab/Tool-Wear-Detection-in-CNC-Milling-Operartions`

## How to Run

1. Open the notebook in Google Colab or Jupyter.
2. Run all cells top to bottom — the first code cell `git clone`s the dataset automatically, no manual download or API key required.
3. `xgboost` is the only package not pre-installed on a fresh Colab runtime (`pip install xgboost -q`, commented at the top of the setup cell).

## Methodology

- **Data cleaning:** feature extraction is restricted to active cutting stages (`Layer 1–3 Up/Down`); idle/prep/repositioning rows are dropped as physically uninformative. Z-axis current is found to be identically zero throughout (the pocket is cut at one fixed depth) and excluded.
- **Approach A:** per-experiment statistical + FFT features → `StandardScaler` → `SelectKBest` → classifier, evaluated with Leave-One-Out CV (the correct way to handle only 18 samples — a single train/test split would be unreliable).
- **Approach B:** raw per-timestep sensor readings → classifier, with **train/test split done by whole experiment** (stratified on tool condition) so no experiment's rows appear on both sides of the split — avoiding a common and easy-to-miss leakage bug.
- Every result is benchmarked against a **majority-class baseline**, not reported in isolation.

## Results

| Metric | Approach A (LOOCV) | Approach B (held-out experiments) |
|---|---|---|
| Accuracy | 0.50 (chance level) | 0.40–0.42 (≤ majority-class baseline of 0.67) |
| ROC-AUC | — | 0.52–0.53 (chance ≈ 0.50) |
| Top-ranked signals | X/Y-axis current & voltage means | X/Y-axis & spindle output current |

**Headline finding, reported directly rather than hidden behind a favorable split:** neither approach reliably beats a trivial baseline on genuinely unseen experiments. Both consistently point to axis/spindle current as the most separable channel — physically sensible, since cutting load shows up first in motor current — but that ranking does not translate into reliable generalization.

## Why the Signal Is Weak — and What Would Fix It

- **Soft workpiece material.** Wax shows far smaller wear-induced load changes than a metal would.
- **Operating-point confounding.** Feed rate and clamp pressure shift signal magnitude independently of, and alongside, tool condition across experiments.
- **Coarse binary label.** `tool_condition` has no continuous flank-wear (VB) measurement — two "worn" tools could be at very different points in their wear life.
- **Only 18 independent trials exist.** Row-level features add data points, not independent observations.
- **Likely fix:** repeat this pipeline on a metal-machining dataset with continuous VB (e.g. the NASA/UC Berkeley milling dataset), framed as regression; explicitly control for feed rate/clamp pressure; add a true force/vibration/acoustic-emission sensor.

## Takeaway

The deliverable here is methodological rigor, not an inflated accuracy number: nested feature selection, Leave-One-Out CV for a small sample, group-aware train/test splitting to prevent leakage, and an explicit baseline comparison — all of which surfaced a negative result worth reporting honestly.

## Tech Stack

Python · pandas · NumPy · SciPy (FFT/stats) · scikit-learn · XGBoost · Matplotlib · Seaborn

## References

1. University of Michigan SMART Lab — *CNC Mill Tool Wear Dataset* (2018).
2. Original Kaggle listing: *"Tool Wear Detection in CNC Mill"*, uploader `shasun`.
3. Related datasets for follow-up work: NASA/UC Berkeley Milling Dataset, PHM Society 2010 Data Challenge.

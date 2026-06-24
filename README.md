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

## Repository Structure

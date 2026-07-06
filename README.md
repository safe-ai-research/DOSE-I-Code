# DOSE-I-Code

Code accompanying the [DOSE-I study](https://safe-ai-research.github.io/DOSE-I/) on EEG-based monitoring of consciousness in sedated patients undergoing gastrointestinal endoscopy. This repository provides two key elements of the analysis pipeline: 1) conversion of raw two-channel EEG waveforms to processed EEG (pEEG) parameters, and 2) machine-learning models applied to classify conscious state.

## Repository structure

### `Predictions_with_ML/`

Machine-learning pipeline for binary (conscious vs. unconscious) classification from pEEG features.

- **`extract_and_join.py`** — Reads pEEG CSVs, drops deprecated features, fills NaNs with median, computes rolling means (10 s, 60 s), intersperses features, joins patient demographics (`GENDER`, `HEIGHT`, `ASA`, `AGE`, `BMI`, `WEIGHT`), and creates 5 s sliding windows (step 4 s).

- **`helper_functions.py`** — Shared utilities: sliding-window generation, time-series splits, unified train/eval for four classifiers (dummy, decision-tree, BalancedRandomForest, Keras NN), feature-importance plotting.

- **`randomForestGlobal.py`** — Global model: 10-fold cross-validation by patient ID.

- **`randomForestLocal.py`** — Local (per-patient) model: 70/30 non-overlapping time-series split.

- **`summarize.py`** — Aggregates per-fold results into mean, std, and 95% CI for loss, AUC-ROC, accuracy, precision, recall, F1. Outputs `.txt` and `.json`.

### `pEEG_estimation/`

Tools for converting raw waveforms and computing pEEG parameter time series.

- **`conv-eegdata.awk`** — AWK script that extracts the two EEG channels from PHILIPS Intellivue CSV exports, handles missing data via placeholder insertion, and writes formatted time-series files for subsequent processing.

- **`pEEG_calculation.c`** — Custom C program (uses its own FFT implementation) that computes 49 time-varying EEG parameters from two-channel recordings: absolute, relative, and synchronization power in eight frequency bands (0.5–1, 1–2, 2–4, 4–7.5, 7.5–13, 13–20, 20–30, 30–49 Hz); Median Frequency (MF), Spectral Edge Frequency 95% (SEF95), and Weighted Spectral Median Frequencies (WSMF) following Jordan et al. (2007); three variants of Permutation Entropy (PE) per Olofsen et al. (2008) and Jordan et al. (2008); SynchFastSlow, CFS Bicoherence, and PowerFastSlow per Miller et al. (2004); and an optimised WSMF variant (Klimpel, 2020). Heart rate (ECG- and pulse-derived), respiratory rate, SpO₂, propofol dose, MOAAS score, and consciousness state are passed through from the source data. All parameters are computed with sliding windows of 8 s and 16 s.

- **`pEEG_parameter_description.txt`** — Full specification of every output column, including the original references for each method.


## Data

This code is intended for usage with the DOSE-I study. Using PHILIPS monitors (Intellivue system), EEG, ECG, pulse oximetry, and capnography signals were recorded during routine sedated endoscopies.

The DOSE-I dataset is available on [Zenodo](https://doi.org/10.5281/zenodo.18483291).

## Reference

Garbe, J., Nguyen, Q. V., Kantelhardt, J. W., Dünninghaus, F., Erffmeier, K., Seeliger, K., & Schmid, T. (2026). Towards predicting sedation depth in endoscopy with large clinically annotated EEG data of continuous propofol sedation. In Artificial Intelligence in Medicine (Lecture Notes in Computer Science, Vol. 16749). Springer.


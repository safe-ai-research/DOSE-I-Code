# Predictions with Basic ML — DOSE-I Study

Machine-learning pipeline for binary (conscious vs. unconscious) classification from processed EEG (pEEG) features in propofol-sedated patients as presented in the AIME contribution [Towards Predicting Sedation Depth in Endoscopy with Large Clinically Annotated EEG Data of Continuous Propofol Sedation](https://safe-ai-research.github.io/DOSE-I/).

** Code Author: Quang Vu Nguyen **
** Licence: GNU GPLv3 **

## Pipeline

```
Raw EEG → pEEG_estimation/ → patient CSVs
         → extract_and_join.py → sliding-window features + demographics
         → randomForestGlobal.py (cross-patient, K-Fold)
         → randomForestLocal.py  (per-patient, time-series split)
         → summarize.py → aggregated metrics (mean, std, 95% CI)
```

## File Reference

| File | Purpose |
|------|---------|
| `extract_and_join.py` | Reads pEEG CSVs from `pEEG/`, fills NaNs, computes rolling means (10s, 60s), intersperses features, joins demographics from `Klin_Parameter.csv`, creates 5s sliding windows (step 4s), writes to `timelag+rollingAvg+Step4s/` |
| `helper_functions.py` | Shared utilities: sliding-window generation, non-overlapping time-series splits, unified train/eval for 4 classifiers (dummy, decision-tree, RandomForest, Keras NN), feature-importance plotting |
| `randomForestGlobal.py` | **Global model** — 10-fold cross-validation by patient ID. Supports dummy, decision-tree, BalancedRandomForest, NN. Tracks AUC-ROC, accuracy, precision, recall, F1, confusion matrix |
| `randomForestLocal.py` | **Local (per-patient) model** — 70/30 non-overlapping time-series split. Set `labelcol=0` for binary, `labelcol=1` for multiclass (MOAAS) |
| `summarize.py` | Aggregates per-fold results: computes mean, std, and 95% CI for cross-entropy loss, AUC-ROC, accuracy, precision, recall, F1. Outputs `.txt` and `.json` |

## Usage Order

1. **Preprocess** — run `extract_and_join.py`
2. **Train models** — run `randomForestGlobal.py` and/or `randomForestLocal.py`
3. **Aggregate** — results are summarized automatically via `summarize.py`

## Dependencies

`numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `keras`, `tensorflow`, `matplotlib`, `seaborn`, `pathlib`

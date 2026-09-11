# testrepo

## Editing the file

Theres a markdown file in this repository

## `classification_models_tuned.py`

A tuned multi-model **classification** benchmark: the classification
counterpart of the regression workflow (`regression_models_tuned.py`), with
every stage re-derived for a categorical target rather than translated
mechanically.

### Running it

```bash
pip install -U scikit-learn xgboost lightgbm optuna tensorflow shap matplotlib seaborn
```

Then set three things near the top of Section A and run the file top to bottom
(it is written as a Colab/notebook script, and also runs as a plain script):

| Setting | What it is |
| --- | --- |
| `CSV_PATH` | path to your CSV (Google Drive path, local path, or the `CSV_PATH` env var) |
| `TARGET_COL` | the class-label column |
| `POSITIVE_LABEL` | for a binary target, which original label counts as "positive" |

Binary and multiclass targets are both handled; the file detects which from the
data and switches scorers, curves and averaging conventions accordingly.

### What it does

1. **Setup & metrics** — stratified CV, a threshold metric set (Accuracy,
   Balanced Accuracy, Precision, Recall, F1, MCC, Kappa) and a
   probability metric set (ROC-AUC, PR-AUC, LogLoss, Brier).
2. **Data & auto-encoding** — the target is label-encoded automatically
   (and one-hot encoded for the multiclass Keras heads); string, boolean,
   datetime, constant and identifier feature columns are encoded or dropped;
   the split seed is chosen on train/test distribution match, and the split is
   stratified.
3. **EDA** — class balance, per-class feature distributions, grouped box plots,
   feature correlation heatmaps, and ANOVA-F / mutual-information association
   with the target.
4. **Model space** — 9 tuned models: Random Forest, Shallow MLP, SVC, XGBoost,
   LightGBM, AdaBoost, KNN, CNN-LSTM, Sequential Logistic Regression, plus 11
   stacking and soft-voting ensembles built on their probabilities.
5. **Hyperparameter optimization** — GridSearchCV / RandomizedSearchCV / Optuna
   per model, all on one objective, plus a K-fold count stability sweep.
6. **Evaluation** — **confusion matrices** (counts and row-normalized),
   **ROC curves** (per model and overlaid; one-vs-rest with micro/macro
   averages for multiclass), precision-recall curves, calibration curves with
   ECE, a decision-threshold sweep, learning curves, a Taylor diagram on the
   predicted probabilities, a Kohavi-Wolpert 0-1 loss bias-variance
   decomposition, and seed sensitivity.
7. **Factor importance** — SHAP (bar, beeswarm, violin, waterfall, dependence)
   and ICE / 1-way and 2-way partial dependence.

Figures are written as PNGs and the tables as `results_*.csv`; the target's
label mapping is written to `target_label_encoding.csv`.

### Cost

The default search budgets (60 randomized draws, 25-80 Optuna trials per model)
are sized for a real run, not a quick look. To try it out fast, lower
`N_ITER_RANDOM` and `N_TRIALS` in Section 0 and `SEED_SCAN_N` in Section A, and
trim the `models` list. `SVC(probability=True)` and the three Keras models
dominate the runtime, and the ensembles refit their members.

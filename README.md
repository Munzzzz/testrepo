# testrepo

## Editing the file

Theres a markdown file in this repository

## Notebooks

Both workflows are also available as notebooks, with the documentation in
markdown cells and the code in code cells:

| Notebook | Cells |
| --- | --- |
| `classification_models_tuned.ipynb` | 79 markdown, 66 code |
| `regression_models_tuned.ipynb` | 55 markdown, 42 code |

They are generated from the scripts and carry the same code, unchanged and in
the same order, so either form can be run. The comments that documented a
section became markdown; short comments explaining the line they sit on stayed
with their code.

Note that `regression_models_tuned.py` is not itself in this branch (it was
added to `main` and then deleted), so `regression_models_tuned.ipynb` is the
copy of that workflow here.

## `classification_models_tuned.py`

A tuned multi-model **classification** benchmark: the classification
counterpart of the regression workflow (`regression_models_tuned.py`), with
every stage re-derived for a categorical target rather than translated
mechanically.

### Running it

```bash
pip install -U scikit-learn imbalanced-learn xgboost lightgbm optuna tensorflow shap lime matplotlib seaborn openpyxl
pip install pysr          # optional — symbolic regression (Section 16)
```

`lime` and `pysr` are only needed by the interpretability sections, and both
degrade cleanly if absent: the workflow prints what is missing and carries on.
PySR is the one to install deliberately — its first import downloads Julia and
precompiles a backend, which takes minutes and needs network access.

Then set three things near the top of Section A and run the file top to bottom
(it is written as a Colab/notebook script, and also runs as a plain script):

| Setting | What it is |
| --- | --- |
| `CSV_PATH` | path to your CSV (Google Drive path, local path, or the `CSV_PATH` env var) |
| `TARGET_COL` | the class-label column |
| `POSITIVE_LABEL` | for a binary target, which original label counts as "positive" |

To turn on class rebalancing, set `RESAMPLING` in Section B (e.g. `"smote"`).
Section B documents the trade it makes: recall usually rises, precision falls,
ROC-AUC barely moves, and the probabilities become miscalibrated — so if the
goal is simply to stop missing the minority class, moving the decision
threshold (Section 8E) does the same job for free and keeps calibration.

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
3. **Class rebalancing (optional)** — SMOTE and variants (Borderline, SVM,
   ADASYN, SMOTE-NC, SMOTE+Tomek, SMOTE+ENN) via the `RESAMPLING` knob in
   Section B, off by default. The sampler is a pipeline step, never a
   preprocessing call, so it is applied to each fold's training part only —
   never to a validation fold and never to the test set.
4. **EDA** — class balance, per-class feature distributions, grouped box plots,
   feature correlation heatmaps, and ANOVA-F / mutual-information association
   with the target.
5. **Model space** — 9 tuned models: Random Forest, Shallow MLP, SVC, XGBoost,
   LightGBM, AdaBoost, KNN, CNN-LSTM, Sequential Logistic Regression, plus 11
   stacking and soft-voting ensembles built on their probabilities.
6. **Hyperparameter optimization** — GridSearchCV / RandomizedSearchCV / Optuna
   per model, all on one objective, plus a K-fold count stability sweep.
7. **Evaluation** — **confusion matrices** (counts and row-normalized),
   **ROC curves** (per model and overlaid; one-vs-rest with micro/macro
   averages for multiclass), precision-recall curves, calibration curves with
   ECE, a decision-threshold sweep, learning curves, a Taylor diagram on the
   predicted probabilities, a Kohavi-Wolpert 0-1 loss bias-variance
   decomposition, and seed sensitivity.
8. **Factor importance** — four views, which answer different questions and
   are worth reading against each other:
   * **SHAP** (bar, beeswarm, violin, waterfall, dependence) — what the model
     attributes each prediction to.
   * **Permutation importance** — what a feature is worth to being *right*,
     scored on both splits, with a consensus ranking and a Spearman
     comparison against the SHAP ranking. A feature can rank high on SHAP and
     near zero here: the model uses it, and the use buys nothing.
   * **LIME** — why *this* row, with the local surrogate's own R-squared
     printed beside every explanation, since a LIME plot with a poor local
     fit is not a weak explanation but no explanation at all.
   * **ICE / 1-way and 2-way partial dependence** — how the prediction
     responds as a feature is swept.
9. **Symbolic regression (PySR)** — searches expression space directly and
   returns a formula rather than a black box. The whole Pareto front is
   printed as an aligned ladder (complexity, loss, marginal score, equation),
   and the selected equation is shown four ways: plain, sympy, LaTeX, and —
   for classification — as an explicit decision rule. The equation is then
   scored against the tuned models so the cost of readability is a number.
   Classification fits the log-odds of the best model by default (`"logit"`,
   a symbolic surrogate) or the labels directly with a margin loss
   (`PYSR_TARGET = "label"`); multiclass fits one equation per class.

Figures are written as PNGs and the tables as `results_*.csv`; the target's
label mapping is written to `target_label_encoding.csv`.

### Cost — set the tuning profile first

Runtime is dominated by the three Keras models. One Optuna trial trains a
network once per CV fold, and the neural models are refit again by the K-fold
sweep, the learning curves, the bias-variance bootstrap and every ensemble
containing them — roughly **1,000 network fits** at the `"full"` budget. That
is an afternoon on a GPU and 2-8 hours on a notebook CPU, where the first
symptom is the search appearing to hang on "Shallow MLP".

`TUNING_PROFILE` in Section 0 controls this:

| Profile | Optuna trials (MLP) | Keras epochs | Search folds | Budget per model |
| --- | --- | --- | --- | --- |
| `"fast"` | 8 | 120 | 3 | 300 s |
| `"balanced"` (default) | 20 | 200 | 3 | 900 s |
| `"full"` | 40 | 300 | 5 | unlimited |

The profile also scales the K-fold sweep, the learning-curve resolution and
the bias-variance bootstrap. `STUDY_TIMEOUT` is a hard wall-clock bound on each
model's search: Optuna finishes the trial in flight, keeps the best parameters
found, and says so — a shortened search is reported, never silently passed off
as a complete one. Every search prints progress, so a long run is
distinguishable from a stuck one.

**The biggest single saving** is dropping the neural models, which on small
tabular data rarely beat the boosted trees:

```python
models = [rf, svc, xgboost_model, lgbm, ada, knn]      # Section 4
```

That removes every Keras fit in the file. Ensemble membership follows the
`models` roster, so groups left with fewer than two members are skipped
automatically. `SVC(probability=True)` is then the slowest remaining model,
because calibration fits it once per fold.

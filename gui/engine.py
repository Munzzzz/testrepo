"""
Modelling engine for the GUI — everything that computes, nothing that draws.

Kept free of Streamlit on purpose: every number the app shows comes from a
function here, so each one can be tested directly, without a browser.

THE WORKSPACE BUNDLE
--------------------
One trained session — models, data split, predictions, metadata — travels as a
plain dict (a "bundle"), whether it came from the in-app trainer or from one of
the notebook workflows (`export_gui_bundle()` at the end of each notebook).
The app never cares which: `Workspace(bundle)` gives both the same interface.

    format            "ml-gui-bundle/1"
    task              "regression" | "classification"
    source            human-readable origin ("In-app training", "Notebook: ...")
    target            target column name
    features          RAW feature names — what the prediction form asks for
    feature_info      {name: {"kind": "numeric"|"categorical"|"text", ...stats}}
    encode            callable(raw DataFrame) -> whatever every model callable takes
    models            {name: callable(encoded) -> predictions (regression)
                                              or class probabilities (classification)}
    classes           classification: original labels, in encoded order 0..K-1
    positive_index    classification: encoded index of the positive class (binary)
    X_train, X_test   RAW feature DataFrames
    y_train, y_test   numeric targets (regression) or encoded labels (classification)
    test_pred, train_pred   {name: array} — cached so every chart is instant
    cv_score          {name: float | None}  in the primary metric's own units
    params            {name: dict}
    fit_time          {name: float}
    primary_metric    "RMSE" | "ROC_AUC"
    best_model        name of the model the prediction tab starts on
    selection_note    how best_model was chosen, in words
    refit             {name: unfitted sklearn estimator}  (in-app only: enables
                      learning curves, which need to retrain)
    extras            free-form (e.g. the notebook's PySR equation)

A model is always called as `models[name](encode(raw_df))`. That single rule is
what lets a notebook model — trained on one-hot encoded float32 arrays inside a
Keras closure — sit in the same dropdown as an in-app sklearn Pipeline that
takes raw DataFrames.
"""
from __future__ import annotations

import io
import math
import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, cohen_kappa_score,
                             f1_score, log_loss, matthews_corrcoef,
                             mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, learning_curve,
                                     train_test_split)
from sklearn.neighbors import (KNeighborsClassifier, KNeighborsRegressor,
                               NearestNeighbors)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, StandardScaler,
                                   label_binarize)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

BUNDLE_FORMAT = "ml-gui-bundle/1"
SEED = 42

# Tuning budgets, mirroring the notebooks' TUNING_PROFILE idea: the same model
# roster at every level, only the search depth changes.
BUDGETS = {
    "Quick":    dict(n_iter=4,  cv=3),
    "Balanced": dict(n_iter=12, cv=5),
    "Thorough": dict(n_iter=30, cv=5),
}

# SVMs scale roughly quadratically in rows; past this they dominate the
# training time for no reliable gain, so they are off by default.
SVM_ROW_LIMIT = 5000


# =============================================================================
#  METRICS — the notebooks' definitions, reproduced exactly
# =============================================================================
#  The GUI recomputes every metric from stored predictions rather than trusting
#  whatever a bundle carried, so a notebook bundle and an in-app run are scored
#  by literally the same code. The formulas match compute_metrics() in the two
#  workflow files: MAPE and SI in percent; binary precision/recall/F1 for the
#  positive class and macro-averaged for multiclass; ROC-AUC one-vs-rest,
#  prevalence-weighted for multiclass.

REGRESSION_METRICS = {
    # name: (higher_is_better, description)
    "R2":   (True,  "Coefficient of determination (1 = perfect)"),
    "RMSE": (False, "Root mean squared error, in target units"),
    "MAE":  (False, "Mean absolute error, in target units"),
    "MAPE": (False, "Mean absolute percentage error (%)"),
    "SI":   (False, "Scatter index: RMSE / mean(actual), in %"),
}

CLASSIFICATION_METRICS = {
    "Accuracy":     (True,  "Share of rows classified correctly"),
    "Balanced_Acc": (True,  "Mean recall over classes — robust to imbalance"),
    "Precision":    (True,  "Of predicted positives, share truly positive"),
    "Recall":       (True,  "Of true positives, share found"),
    "F1":           (True,  "Harmonic mean of precision and recall"),
    "MCC":          (True,  "Matthews correlation — uses all four cells"),
    "Kappa":        (True,  "Cohen's kappa — agreement above chance"),
    "ROC_AUC":      (True,  "Ranking quality of the probabilities"),
    "PR_AUC":       (True,  "Average precision — sensitive to the rare class"),
    "LogLoss":      (False, "Penalises confident wrong probabilities"),
    "Brier":        (False, "Squared error of the probability vector"),
}


METRIC_LABELS = {"R2": "R²", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC",
                 "Balanced_Acc": "Balanced accuracy", "LogLoss": "Log loss"}


def metric_label(m):
    """How a metric's name reads on screen (R2 -> R², ROC_AUC -> ROC-AUC)."""
    return METRIC_LABELS.get(m, m.replace("_", " "))


def metric_catalog(task):
    return REGRESSION_METRICS if task == "regression" else CLASSIFICATION_METRICS


def _safe(fn):
    """A metric that is undefined on this slice is NaN, not an exception."""
    try:
        v = float(fn())
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    rmse = _safe(lambda: np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "R2":   _safe(lambda: r2_score(y_true, y_pred)),
        "RMSE": rmse,
        "MAE":  _safe(lambda: mean_absolute_error(y_true, y_pred)),
        "MAPE": _safe(lambda: mean_absolute_percentage_error(y_true, y_pred) * 100),
        "SI":   _safe(lambda: rmse / np.mean(y_true) * 100),
    }


def classification_metrics(y_true, proba, n_classes, positive_index=1):
    y_true = np.asarray(y_true).astype(int).ravel()
    P = np.asarray(proba, dtype=np.float64)
    y_pred = P.argmax(axis=1)
    labels = list(range(n_classes))
    binary = n_classes == 2
    kw = dict(average="binary" if binary else "macro", zero_division=0, labels=labels)
    if binary:
        kw["pos_label"] = positive_index
    onehot = np.eye(n_classes)[y_true]
    return {
        "Accuracy":     _safe(lambda: accuracy_score(y_true, y_pred)),
        "Balanced_Acc": _safe(lambda: balanced_accuracy_score(y_true, y_pred)),
        "Precision":    _safe(lambda: precision_score(y_true, y_pred, **kw)),
        "Recall":       _safe(lambda: recall_score(y_true, y_pred, **kw)),
        "F1":           _safe(lambda: f1_score(y_true, y_pred, **kw)),
        "MCC":          _safe(lambda: matthews_corrcoef(y_true, y_pred)),
        "Kappa":        _safe(lambda: cohen_kappa_score(y_true, y_pred, labels=labels)),
        "ROC_AUC":      _safe(lambda: (
            roc_auc_score(y_true, P[:, positive_index]) if binary else
            roc_auc_score(y_true, P, multi_class="ovr", average="weighted",
                          labels=labels))),
        "PR_AUC":       _safe(lambda: (
            average_precision_score((y_true == positive_index).astype(int),
                                    P[:, positive_index]) if binary else
            average_precision_score(label_binarize(y_true, classes=labels), P,
                                    average="macro"))),
        "LogLoss":      _safe(lambda: log_loss(y_true, P, labels=labels)),
        "Brier":        _safe(lambda: np.mean(np.sum((P - onehot) ** 2, axis=1))),
    }


# =============================================================================
#  DATA — reading, profiling, task detection
# =============================================================================
def read_table(source, name=None, sheet=0):
    """
    CSV / TXT / Excel into a DataFrame.

    `source` is a path or a file-like object (what st.file_uploader returns).
    The CSV delimiter is sniffed, so semicolon- and tab-separated exports load
    without configuration.
    """
    name = str(name or getattr(source, "name", source))
    ext = os.path.splitext(name)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(source, sheet_name=sheet)
    if hasattr(source, "read"):
        raw = source.read()
        source = io.BytesIO(raw if isinstance(raw, bytes) else raw.encode())
    return pd.read_csv(source, sep=None, engine="python", encoding_errors="replace")


def excel_sheets(source):
    """Sheet names of an Excel file, or [] for anything else."""
    try:
        return list(pd.ExcelFile(source).sheet_names)
    except Exception:
        return []


def _is_numeric(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _looks_like_dates(s, sample=50):
    vals = s.dropna().astype(str).head(sample)
    if len(vals) < 5 or vals.str.fullmatch(r"-?\d+(\.\d+)?").all():
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(vals, errors="coerce")
    return parsed.notna().mean() > 0.8


def detect_task(y):
    """
    Regression or classification, with the reason in words.

    Text or boolean -> classification. Numeric -> regression, unless it holds
    at most 10 distinct whole numbers, which is how class labels are usually
    stored (0/1, 1..5 grades). The app shows the reason and lets it be
    overridden — a 1-10 rating could honestly go either way.
    """
    s = y.dropna()
    n = int(s.nunique())
    if not _is_numeric(s):
        return "classification", f"'{y.name}' holds text/boolean values ({n} classes)"
    whole = bool(np.all(np.isclose(s.astype(float), np.round(s.astype(float)))))
    if whole and n <= 10:
        return "classification", f"'{y.name}' holds {n} distinct whole numbers — read as class labels"
    return "regression", f"'{y.name}' is numeric with {n} distinct values"


def suggest_features(df, target):
    """
    Every column except the target, minus the ones that would only hurt.

    Returns (features, excluded) where excluded is [(column, reason), ...]. The
    app shows the exclusions and lets the user put any of them back — these
    are defaults, not decisions.
    """
    features, excluded = [], []
    n = len(df)
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if s.isna().all():
            excluded.append((c, "entirely empty"))
        elif s.nunique(dropna=True) <= 1:
            excluded.append((c, "constant — carries no information"))
        elif pd.api.types.is_datetime64_any_dtype(s) or (
                not _is_numeric(s) and _looks_like_dates(s)):
            excluded.append((c, "a date — not usable as a model input as-is"))
        elif not _is_numeric(s) and s.nunique() == n:
            excluded.append((c, "unique text on every row — an identifier"))
        elif (_is_numeric(s) and s.nunique() == n and n > 20
              and (str(c).lower().startswith("unnamed") or
                   (np.all(np.isclose(s, np.round(s))) and s.is_monotonic_increasing))):
            excluded.append((c, "a row index / counter — an identifier"))
        else:
            features.append(c)
    return features, excluded


def profile_features(X):
    """Per-feature facts the prediction form needs: kind, range, default."""
    info = {}
    for c in X.columns:
        s = X[c]
        if _is_numeric(s):
            v = s.dropna().astype(float)
            whole = bool(len(v)) and bool(np.all(np.isclose(v, np.round(v))))
            info[c] = {
                "kind": "numeric", "integer": whole,
                "min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()), "median": float(v.median()),
                "std": float(v.std(ddof=0)) if len(v) > 1 else 0.0,
                "q01": float(v.quantile(0.01)), "q99": float(v.quantile(0.99)),
                "n_missing": int(s.isna().sum()),
            }
        else:
            counts = s.dropna().astype(str).value_counts()
            info[c] = {
                "kind": "categorical",
                "categories": list(counts.index),
                "mode": counts.index[0] if len(counts) else "",
                "n_missing": int(s.isna().sum()),
            }
    return info


def baseline_row(features, info):
    """A 'typical' input: the median of each numeric feature, the mode of each category."""
    row = {}
    for f in features:
        fi = info[f]
        if fi["kind"] == "numeric":
            row[f] = fi["median"]
        elif fi["kind"] == "categorical":
            row[f] = fi["mode"]
        else:
            row[f] = fi.get("default", "")
    return pd.DataFrame([row], columns=features)


def normalize_types(df, features, info):
    """
    Put raw inputs in the dtypes the in-app pipelines were trained on.

    Numeric columns become float64 — integer columns included, which is what
    sklearn's partial-dependence and several imputers want. Categorical
    columns become object dtype with np.nan for missing: pandas 3 reads text
    as its own `str` dtype, and not every sklearn transformer handles that
    identically, so it is normalised once, here, on the way in.
    """
    out = pd.DataFrame(index=df.index)
    for f in features:
        s = df[f] if f in df.columns else pd.Series(np.nan, index=df.index)
        if info[f]["kind"] == "numeric":
            out[f] = pd.to_numeric(s, errors="coerce").astype(np.float64)
        else:
            out[f] = s.astype(object).where(s.notna(), np.nan).map(
                lambda v: v if isinstance(v, float) and np.isnan(v) else str(v))
    return out


class RawFrameEncoder:
    """
    The in-app `encode`: raw rows -> the typed DataFrame the pipelines take.

    A small class rather than a lambda so it pickles by reference cleanly and
    reads well in a traceback.
    """
    def __init__(self, features, info):
        self.features = list(features)
        self.info = info

    def __call__(self, df):
        return normalize_types(df, self.features, self.info)


# =============================================================================
#  DATA EXPLORATION — the numbers behind the Data tab
# =============================================================================
def summary_statistics(df, columns):
    """describe() plus the shape statistics the notebooks report."""
    rows = []
    for c in columns:
        s = df[c]
        if _is_numeric(s):
            v = s.dropna().astype(float)
            q1, q3 = v.quantile([0.25, 0.75]) if len(v) else (np.nan, np.nan)
            rows.append({
                "feature": c, "type": "numeric", "count": int(v.size),
                "missing": int(s.isna().sum()), "mean": v.mean(), "std": v.std(),
                "min": v.min(), "25%": q1, "median": v.median(), "75%": q3,
                "max": v.max(),
                "skewness": stats.skew(v) if len(v) > 2 else np.nan,
                "kurtosis": stats.kurtosis(v) if len(v) > 3 else np.nan,
                "CV %": v.std() / v.mean() * 100 if v.mean() else np.nan,
            })
        else:
            vc = s.dropna().astype(str).value_counts()
            rows.append({
                "feature": c, "type": "categorical", "count": int(s.notna().sum()),
                "missing": int(s.isna().sum()), "unique": int(vc.size),
                "top": vc.index[0] if len(vc) else None,
                "top freq": int(vc.iloc[0]) if len(vc) else None,
            })
    return pd.DataFrame(rows).set_index("feature")


def outlier_table(df, columns):
    """IQR-rule outlier counts (beyond 1.5 x IQR), per numeric feature."""
    rows = []
    for c in columns:
        if not _is_numeric(df[c]):
            continue
        v = df[c].dropna().astype(float)
        q1, q3 = v.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((v < lo) | (v > hi)).sum())
        rows.append({"feature": c, "lower fence": lo, "upper fence": hi,
                     "outliers": n_out, "outlier %": 100 * n_out / max(len(v), 1)})
    return pd.DataFrame(rows).set_index("feature") if rows else pd.DataFrame()


_CORR_TESTS = {"pearson": stats.pearsonr, "spearman": stats.spearmanr,
               "kendall": stats.kendalltau}
SIG_ONE_STAR = 0.01    # p < 0.01 -> "*"   (the notebooks' convention)
SIG_TWO_STAR = 0.05    # p < 0.05 -> "**"


def correlation_with_pvalues(df, columns, method="pearson"):
    """
    Pairwise correlation and its p-value, on pairwise-complete rows.

    Same convention as the notebooks: `*` marks p < 0.01 and `**` marks
    p < 0.05. That is the reverse of the usual journal habit, and it is kept
    deliberately so a figure from the GUI and one from the notebook can never
    disagree about what a star means — the legend under every heatmap spells
    it out.
    """
    cols = [c for c in columns if _is_numeric(df[c])]
    k = len(cols)
    R = np.full((k, k), np.nan)
    Pv = np.full((k, k), np.nan)
    test = _CORR_TESTS[method]
    for i in range(k):
        R[i, i], Pv[i, i] = 1.0, 0.0
        for j in range(i):
            pair = df[[cols[i], cols[j]]].dropna().astype(float)
            if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, p = test(pair.iloc[:, 0], pair.iloc[:, 1])
            R[i, j] = R[j, i] = float(r)
            Pv[i, j] = Pv[j, i] = float(p)
    return (pd.DataFrame(R, index=cols, columns=cols),
            pd.DataFrame(Pv, index=cols, columns=cols))


def significance_stars(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < SIG_ONE_STAR:
        return "*"
    if p < SIG_TWO_STAR:
        return "**"
    return ""


# =============================================================================
#  MODEL ZOO — the notebooks' roster, as fast sklearn-compatible estimators
# =============================================================================
#  Same families as the workflows. The three Keras networks become sklearn's
#  MLP (one net) and a regularised linear model (the Sequential Linear /
#  Logistic analogue): Keras trains an order of magnitude slower and a GUI has
#  to stay interactive. The notebooks' own tuned Keras models are still one
#  bundle away — `export_gui_bundle()` — and load into the same dropdowns.

@dataclass
class ModelDef:
    name: str
    estimator: object
    params: dict
    scale: bool = False            # standardise numeric inputs first
    scale_target: bool = False     # regression: fit on standardised y
    needs_rows_under: int | None = None


def _knn_k_max(n_train, cv):
    """KNN's n_neighbors must stay below the rows in a CV training fold."""
    return max(2, min(30, int(n_train * (cv - 1) / cv) - 1))


def regression_zoo(n_train, cv):
    kmax = _knn_k_max(n_train, cv)
    return [
        ModelDef("Linear (Ridge)", Ridge(), {"model__alpha": loguniform(1e-3, 1e3)},
                 scale=True),
        ModelDef("Random Forest",
                 RandomForestRegressor(random_state=SEED, n_jobs=1),
                 {"model__n_estimators": [200, 400, 700],
                  "model__max_depth": [None, 6, 10, 16],
                  "model__min_samples_leaf": [1, 2, 4],
                  "model__max_features": ["sqrt", 0.5, 1.0]}),
        ModelDef("XGBoost", _xgb("reg"),
                 {"model__n_estimators": randint(150, 800),
                  "model__learning_rate": loguniform(0.01, 0.3),
                  "model__max_depth": randint(2, 8),
                  "model__subsample": uniform(0.6, 0.4),
                  "model__colsample_bytree": uniform(0.6, 0.4),
                  "model__reg_lambda": loguniform(1e-3, 10)}),
        ModelDef("LightGBM", _lgbm("reg"),
                 {"model__n_estimators": randint(150, 800),
                  "model__learning_rate": loguniform(0.01, 0.3),
                  "model__num_leaves": randint(8, 64),
                  "model__min_child_samples": randint(5, 40),
                  "model__subsample": uniform(0.6, 0.4),
                  "model__subsample_freq": [1],
                  "model__colsample_bytree": uniform(0.6, 0.4)}),
        ModelDef("AdaBoost",
                 AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=SEED),
                 {"model__n_estimators": [50, 100, 200, 400],
                  "model__learning_rate": loguniform(0.01, 1.0),
                  "model__estimator__max_depth": [2, 3, 4, 6]}),
        ModelDef("KNN", KNeighborsRegressor(),
                 {"model__n_neighbors": randint(2, kmax + 1),
                  "model__weights": ["uniform", "distance"], "model__p": [1, 2]},
                 scale=True),
        ModelDef("SVR", SVR(),
                 {"model__C": loguniform(0.1, 1000), "model__gamma": loguniform(1e-4, 1),
                  "model__epsilon": loguniform(0.01, 1)},
                 scale=True, scale_target=True, needs_rows_under=SVM_ROW_LIMIT),
        ModelDef("Neural net (MLP)",
                 MLPRegressor(max_iter=2000, early_stopping=True, random_state=SEED),
                 {"model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                  "model__alpha": loguniform(1e-5, 1e-1),
                  "model__learning_rate_init": loguniform(1e-4, 1e-2)},
                 scale=True, scale_target=True),
    ]


def classification_zoo(n_train, cv):
    kmax = _knn_k_max(n_train, cv)
    return [
        ModelDef("Logistic regression", LogisticRegression(max_iter=5000),
                 {"model__C": loguniform(1e-3, 1e3)}, scale=True),
        ModelDef("Random Forest",
                 RandomForestClassifier(random_state=SEED, n_jobs=1),
                 {"model__n_estimators": [200, 400, 700],
                  "model__max_depth": [None, 6, 10, 16],
                  "model__min_samples_leaf": [1, 2, 4],
                  "model__max_features": ["sqrt", 0.5, 1.0]}),
        ModelDef("XGBoost", _xgb("clf"),
                 {"model__n_estimators": randint(150, 800),
                  "model__learning_rate": loguniform(0.01, 0.3),
                  "model__max_depth": randint(2, 8),
                  "model__subsample": uniform(0.6, 0.4),
                  "model__colsample_bytree": uniform(0.6, 0.4)}),
        ModelDef("LightGBM", _lgbm("clf"),
                 {"model__n_estimators": randint(150, 800),
                  "model__learning_rate": loguniform(0.01, 0.3),
                  "model__num_leaves": randint(8, 64),
                  "model__min_child_samples": randint(5, 40),
                  "model__subsample": uniform(0.6, 0.4),
                  "model__subsample_freq": [1],
                  "model__colsample_bytree": uniform(0.6, 0.4)}),
        # AdaBoost RANKS well but its predict_proba is squashed towards 1/K —
        # boosting pushes probabilities away from 0 and 1, a well-documented
        # effect (Niculescu-Mizil & Caruana, "Predicting good probabilities
        # with supervised learning", 2005). Left raw, it can win on ROC-AUC,
        # which only looks at the ordering, and then tell the user "38%
        # confident" about a class it is right about most of the time. Platt
        # scaling inside cross-validation is the paper's remedy and the same
        # treatment SVC gets below; it leaves the ranking essentially intact.
        ModelDef("AdaBoost",
                 CalibratedClassifierCV(
                     AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=SEED),
                     method="sigmoid", ensemble=False, cv=3),
                 {"model__estimator__n_estimators": [50, 100, 200, 400],
                  "model__estimator__learning_rate": loguniform(0.01, 1.0),
                  "model__estimator__estimator__max_depth": [1, 2, 3, 4]}),
        ModelDef("KNN", KNeighborsClassifier(),
                 {"model__n_neighbors": randint(2, kmax + 1),
                  "model__weights": ["uniform", "distance"], "model__p": [1, 2]},
                 scale=True),
        # SVC(probability=True) is deprecated in scikit-learn 1.9, so the
        # probabilities come from an explicit calibration wrapper — the same
        # route the classification notebook's make_probabilistic_svc takes.
        ModelDef("SVM (SVC)",
                 CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False, cv=3),
                 {"model__estimator__C": loguniform(0.1, 1000),
                  "model__estimator__gamma": loguniform(1e-4, 1)},
                 scale=True, needs_rows_under=SVM_ROW_LIMIT),
        ModelDef("Neural net (MLP)",
                 MLPClassifier(max_iter=2000, early_stopping=True, random_state=SEED),
                 {"model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                  "model__alpha": loguniform(1e-5, 1e-1),
                  "model__learning_rate_init": loguniform(1e-4, 1e-2)},
                 scale=True),
    ]


def _xgb(kind):
    import xgboost as xgb
    if kind == "reg":
        return xgb.XGBRegressor(random_state=SEED, n_jobs=1, tree_method="hist",
                                verbosity=0)
    return xgb.XGBClassifier(random_state=SEED, n_jobs=1, tree_method="hist",
                             eval_metric="logloss", verbosity=0)


def _lgbm(kind):
    import lightgbm as lgb
    cls = lgb.LGBMRegressor if kind == "reg" else lgb.LGBMClassifier
    return cls(random_state=SEED, n_jobs=1, verbose=-1)


def model_names(task):
    zoo = regression_zoo if task == "regression" else classification_zoo
    return [m.name for m in zoo(1000, 5)]


def make_preprocessor(numeric, categorical, scale):
    num = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("scale", StandardScaler()))
    parts = []
    if numeric:
        parts.append(("num", Pipeline(num), numeric))
    if categorical:
        # handle_unknown: a category the model never saw becomes all-zeros
        # rather than an exception. max_categories folds a long tail into one
        # "infrequent" column instead of exploding the matrix.
        parts.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     max_categories=30, sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(parts, remainder="drop")


def _smote_k(y, cv):
    """
    SMOTE's k_neighbors must be below the minority class's size in EVERY CV
    training fold, not in the full training set. The classification notebook's
    resolve_smote_k makes the same correction.
    """
    smallest = int(np.bincount(np.asarray(y, dtype=int)).min())
    in_fold = int(np.floor(smallest * (cv - 1) / cv))
    return max(1, min(5, in_fold - 1))


def build_estimator(mdef, task, numeric, categorical, smote_k=None):
    """Preprocessing + (optional SMOTE) + model, as one leakage-free pipeline."""
    pre = make_preprocessor(numeric, categorical, mdef.scale)
    if smote_k is not None:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        # Inside the pipeline, SMOTE runs on each CV training fold only and
        # never on the fold it is scored on — resampling before the split is
        # the classic leak that inflates every score.
        pipe = ImbPipeline([("prep", pre),
                            ("smote", SMOTE(k_neighbors=smote_k, random_state=SEED)),
                            ("model", clone(mdef.estimator))])
    else:
        pipe = Pipeline([("prep", pre), ("model", clone(mdef.estimator))])
    params = dict(mdef.params)
    if task == "regression" and mdef.scale_target:
        pipe = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
        params = {f"regressor__{k}": v for k, v in params.items()}
    return pipe, params


# =============================================================================
#  TRAINING
# =============================================================================
@dataclass
class TrainSettings:
    task: str
    target: str
    features: list
    budget: str = "Balanced"
    test_size: float = 0.2
    seed: int = SEED
    models: list | None = None          # names; None = the full roster
    smote: bool = False                 # classification only
    positive_label: object = None       # classification, binary only
    ensemble: bool = True


def prepare_xy(df, settings):
    """Drop rows without a target, encode the target, profile the features."""
    data = df[settings.features + [settings.target]].copy()
    n_before = len(data)
    data = data[data[settings.target].notna()]
    dropped = n_before - len(data)
    X = data[settings.features]
    info = profile_features(X)
    X = normalize_types(X, settings.features, info)
    y_raw = data[settings.target]
    if settings.task == "regression":
        y = pd.to_numeric(y_raw, errors="coerce").astype(np.float64).to_numpy()
        keep = np.isfinite(y)
        return X[keep], y[keep], None, 1, info, dropped + int((~keep).sum())
    # Classification: label-encode ("auto-encode the output"), keeping the
    # original labels so every screen can show them instead of 0/1/2.
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    classes = list(le.classes_)
    pos = 1
    if len(classes) == 2 and settings.positive_label is not None:
        pos = classes.index(str(settings.positive_label))
    return X, y, classes, pos, info, dropped


def train_workspace(df, settings, progress=None):
    """
    Tune and fit every requested model; return a bundle dict.

    `progress(fraction, message)` is called as work completes so the app can
    drive a progress bar. Each model is tuned by randomised search with
    cross-validation on the TRAINING split only; the test split is touched
    once, at the end, to score the finished models.
    """
    t_start = time.time()
    say = progress or (lambda f, m: None)
    task = settings.task
    X, y, classes, pos, info, dropped = prepare_xy(df, settings)
    n_classes = len(classes) if classes else None

    if task == "classification":
        counts = np.bincount(y)
        if counts.min() < 2:
            rare = [classes[i] for i in np.flatnonzero(counts < 2)]
            raise ValueError(f"class(es) {rare} have fewer than 2 rows — a model "
                             f"cannot learn a class it sees once. Remove them or "
                             f"merge them into a neighbouring class.")
    strat = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.seed, stratify=strat)

    budget = BUDGETS[settings.budget]
    cv_k = budget["cv"]
    if task == "classification":
        cv_k = max(2, min(cv_k, int(np.bincount(y_tr).min())))
        cv = StratifiedKFold(cv_k, shuffle=True, random_state=settings.seed)
        scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
        primary = "ROC_AUC"
    else:
        cv = KFold(cv_k, shuffle=True, random_state=settings.seed)
        scoring = "neg_root_mean_squared_error"
        primary = "RMSE"

    numeric = [f for f in settings.features if info[f]["kind"] == "numeric"]
    categorical = [f for f in settings.features if info[f]["kind"] != "numeric"]
    zoo = (regression_zoo if task == "regression" else classification_zoo)(len(X_tr), cv_k)
    if settings.models is not None:
        zoo = [m for m in zoo if m.name in settings.models]
    smote_k = (_smote_k(y_tr, cv_k)
               if task == "classification" and settings.smote else None)

    models, refit, test_pred, train_pred = {}, {}, {}, {}
    cv_score, params, fit_time, failures, skipped = {}, {}, {}, {}, []
    steps = len(zoo) + (1 if settings.ensemble else 0)

    def _predict(est, Xs):
        return (est.predict_proba(Xs) if task == "classification"
                else np.asarray(est.predict(Xs), dtype=np.float64).ravel())

    for i, mdef in enumerate(zoo):
        if mdef.needs_rows_under and len(X_tr) > mdef.needs_rows_under:
            skipped.append((mdef.name, f"over {mdef.needs_rows_under:,} training rows"))
            continue
        say(i / steps, f"Tuning {mdef.name} ({i + 1} of {steps})")
        est, space = build_estimator(mdef, task, numeric, categorical, smote_k)
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search = RandomizedSearchCV(
                    est, space, n_iter=budget["n_iter"], cv=cv, scoring=scoring,
                    n_jobs=-1, random_state=settings.seed, error_score=np.nan,
                    refit=True)
                search.fit(X_tr, y_tr)
        except Exception as exc:
            failures[mdef.name] = f"{type(exc).__name__}: {exc}"
            continue
        best = search.best_estimator_
        score = float(search.best_score_)
        models[mdef.name] = best
        refit[mdef.name] = clone(best)
        cv_score[mdef.name] = -score if primary == "RMSE" else score
        params[mdef.name] = {k.split("__", 1)[-1] if k.startswith("regressor__") else k:
                             v for k, v in search.best_params_.items()}
        params[mdef.name] = {k.replace("model__", ""): v
                             for k, v in params[mdef.name].items()}
        fit_time[mdef.name] = time.time() - t0
        test_pred[mdef.name] = _predict(best, X_te)
        train_pred[mdef.name] = _predict(best, X_tr)

    if not models:
        raise RuntimeError("every model failed to train: "
                           + "; ".join(f"{k}: {v}" for k, v in failures.items()))

    def _cv_rank(name):
        s = cv_score[name]
        return -s if primary == "RMSE" else s

    # A soft-voting ensemble of the three best models by cross-validation —
    # the notebooks' Section 4B idea in its simplest form. Its CV score is
    # computed the same way as every other model's, so it earns its place in
    # the ranking rather than being handed it.
    if settings.ensemble and len(models) >= 3:
        say(len(zoo) / steps, "Building the voting ensemble")
        top = sorted(models, key=_cv_rank, reverse=True)[:3]
        members = [(f"m{i}", clone(models[n])) for i, n in enumerate(top)]
        vote = (VotingRegressor(members) if task == "regression"
                else VotingClassifier(members, voting="soft"))
        name = "Voting ensemble (top 3)"
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = cross_val_score(vote, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
                vote.fit(X_tr, y_tr)
            models[name] = vote
            refit[name] = clone(vote)
            cv_score[name] = float(-np.mean(s) if primary == "RMSE" else np.mean(s))
            params[name] = {"members": ", ".join(top)}
            fit_time[name] = time.time() - t0
            test_pred[name] = _predict(vote, X_te)
            train_pred[name] = _predict(vote, X_tr)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"

    say(1.0, "Done")
    best_name = max(models, key=_cv_rank)
    direction = "lowest" if primary == "RMSE" else "highest"
    call = (lambda est: est.predict_proba) if task == "classification" else (
        lambda est: est.predict)
    return {
        "format": BUNDLE_FORMAT,
        "task": task,
        "source": "In-app training",
        "target": settings.target,
        "features": list(settings.features),
        "feature_info": info,
        "encode": RawFrameEncoder(settings.features, info),
        "models": {n: call(est) for n, est in models.items()},
        "classes": classes,
        "positive_index": pos,
        "X_train": X_tr, "X_test": X_te,
        "y_train": np.asarray(y_tr), "y_test": np.asarray(y_te),
        "test_pred": test_pred, "train_pred": train_pred,
        "cv_score": cv_score, "params": params, "fit_time": fit_time,
        "primary_metric": primary,
        "rank_by": "cv",
        "best_model": best_name,
        "selection_note": (f"{direction} {cv_k}-fold cross-validated "
                           f"{primary.replace('_', '-')} on the training split — "
                           f"chosen without looking at the test set, so its test "
                           f"score is an honest estimate"),
        "refit": refit,
        "failures": failures,
        "skipped": skipped,
        "settings": {"budget": settings.budget, "test_size": settings.test_size,
                     "seed": settings.seed, "cv_folds": cv_k,
                     "smote": bool(smote_k), "rows_dropped": dropped},
        "train_seconds": time.time() - t_start,
        "extras": {},
    }


# =============================================================================
#  WORKSPACE — one interface over any bundle
# =============================================================================
class Workspace:
    """Everything the app needs from a trained session, whatever its source."""

    def __init__(self, bundle):
        fmt = bundle.get("format")
        if fmt != BUNDLE_FORMAT:
            raise ValueError(f"not a GUI bundle (format {fmt!r}, expected {BUNDLE_FORMAT!r})")
        self.b = bundle
        self.task = bundle["task"]
        self.target = bundle["target"]
        self.features = list(bundle["features"])
        self.info = bundle["feature_info"]
        self.classes = bundle.get("classes")
        self.n_classes = len(self.classes) if self.classes else None
        self.pos = int(bundle.get("positive_index", 1))
        self.X_train, self.X_test = bundle["X_train"], bundle["X_test"]
        self.y_train = np.asarray(bundle["y_train"])
        self.y_test = np.asarray(bundle["y_test"])
        self.primary = bundle["primary_metric"]
        self.model_names = list(bundle["models"])
        self.best = (bundle["best_model"] if bundle["best_model"] in bundle["models"]
                     else self.model_names[0])
        self.source = bundle.get("source", "")
        self._cache = {}
        self.id = f"{self.source}|{self.target}|{len(self.y_train)}|{id(bundle)}"

    # -- prediction -----------------------------------------------------------
    def _raw(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.features)
        return X[self.features]

    def predict_raw(self, name, X):
        """Regression: predictions. Classification: an (n, K) probability matrix."""
        out = self.b["models"][name](self.b["encode"](self._raw(X)))
        out = np.asarray(out, dtype=np.float64)
        if self.task == "regression":
            return out.ravel()
        if out.ndim == 1:                       # a binary head returning P(class 1)
            out = np.column_stack([1.0 - out, out])
        return out

    def predict_scalar(self, name, X, class_index=None):
        """One number per row: the prediction, or P(class_index)."""
        out = self.predict_raw(name, X)
        return out if self.task == "regression" else out[:, class_index]

    # -- metrics --------------------------------------------------------------
    def _metrics(self, y, pred):
        if self.task == "regression":
            return regression_metrics(y, pred)
        return classification_metrics(y, pred, self.n_classes, self.pos)

    def metrics(self, name, split="test"):
        key = ("metrics", name, split)
        if key not in self._cache:
            y = self.y_test if split == "test" else self.y_train
            pred = self.b["test_pred" if split == "test" else "train_pred"].get(name)
            self._cache[key] = (self._metrics(y, pred) if pred is not None else
                                {m: float("nan") for m in metric_catalog(self.task)})
        return self._cache[key]

    def rank_value(self, name):
        """
        The number models are ranked by — higher is always better here.

        In-app runs rank by the cross-validated score (the test set plays no
        part in choosing); notebook bundles rank by the test metric, because
        that is how the notebooks choose and the two must agree. Either way
        the leaderboard and the "best model" come from this one function, so
        the table's top row and the highlighted model cannot disagree.
        """
        higher = metric_catalog(self.task)[self.primary][0]
        if self.b.get("rank_by", "cv") == "cv" and self.b["cv_score"].get(name) is not None:
            v = self.b["cv_score"][name]
        else:
            v = self.metrics(name, "test")[self.primary]
        if v is None or not np.isfinite(v):
            return -np.inf
        return v if higher else -v

    def leaderboard(self):
        """One row per model: CV score, then every test metric, best first."""
        rows = []
        for n in self.model_names:
            m = self.metrics(n, "test")
            cv = self.b["cv_score"].get(n)
            rows.append({"Model": n,
                         f"CV {self.primary}": np.nan if cv is None else cv,
                         **{f"{k}": v for k, v in m.items()},
                         "Fit time (s)": self.b["fit_time"].get(n, np.nan)})
        df = pd.DataFrame(rows).set_index("Model")
        order = sorted(self.model_names, key=self.rank_value, reverse=True)
        return df.loc[order]

    def pred(self, name, split="test"):
        return np.asarray(self.b["test_pred" if split == "test" else "train_pred"][name])

    @property
    def can_refit(self):
        return bool(self.b.get("refit"))

    # -- uncertainty & novelty ------------------------------------------------
    def conformal_halfwidth(self, name, coverage=0.90):
        """
        Split-conformal half-width from the model's held-out absolute errors.

        For a new point exchangeable with the test rows, the interval
        prediction ± q contains the true value with probability >= coverage.
        q is the ceil((n+1)*coverage)-th smallest absolute test residual; with
        too few test rows for the requested coverage the interval is
        unbounded, and that is reported instead of a made-up number.
        """
        if self.task != "regression":
            return None
        resid = np.sort(np.abs(self.y_test - self.pred(name)))
        n = len(resid)
        rank = math.ceil((n + 1) * coverage)
        return float("inf") if rank > n else float(resid[rank - 1])

    def extrapolation(self, row):
        """Features of one input row that fall outside what the model saw."""
        flags = []
        for f in self.features:
            fi, v = self.info[f], row[f].iloc[0]
            if fi["kind"] == "numeric":
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                span = fi["max"] - fi["min"] or 1.0
                if v < fi["min"]:
                    flags.append((f, f"{_fmt(v)} is below the training minimum "
                                     f"{_fmt(fi['min'])} ({(fi['min'] - v) / span:.0%} of the range)"))
                elif v > fi["max"]:
                    flags.append((f, f"{_fmt(v)} is above the training maximum "
                                     f"{_fmt(fi['max'])} ({(v - fi['max']) / span:.0%} of the range)"))
            elif fi["kind"] == "categorical" and str(v) not in fi["categories"]:
                flags.append((f, f"'{v}' never appeared in the training data"))
        return flags

    def out_of_range(self, frame):
        """extrapolation() for a whole table at once: True where a row has any flag."""
        bad = np.zeros(len(frame), dtype=bool)
        for f in self.features:
            fi = self.info[f]
            if fi["kind"] == "numeric":
                v = pd.to_numeric(frame[f], errors="coerce")
                bad |= ((v < fi["min"]) | (v > fi["max"])).to_numpy(dtype=bool)
            elif fi["kind"] == "categorical":
                bad |= ~frame[f].astype(str).isin(list(fi["categories"])).to_numpy(dtype=bool)
        return bad

    def novelty(self, row, k=5):
        """
        How unusual is this COMBINATION of values, even if each is in range?

        Every feature can sit comfortably inside its own training range while
        the combination is one the model never saw — a mix with the most
        cement AND the most water, say. The distance from the input to its
        k-th nearest training row (standardised numeric features) is ranked
        against the same distance for the training rows themselves; a
        percentile near 100 means the input is lonelier than almost any real
        row, and the model is extrapolating even though no single field is.
        """
        num = [f for f in self.features if self.info[f]["kind"] == "numeric"]
        if not num or len(self.X_train) <= k + 1:
            return None
        if "nn" not in self._cache:
            Z = self.X_train[num].astype(float)
            mu, sd = Z.mean(), Z.std(ddof=0).replace(0, 1.0)
            Zs = ((Z - mu) / sd).fillna(0.0).to_numpy()
            nn = NearestNeighbors(n_neighbors=k + 1).fit(Zs)
            d_train = nn.kneighbors(Zs)[0][:, k]      # column 0 is the row itself
            self._cache["nn"] = (nn, mu, sd, np.sort(d_train))
        nn, mu, sd, ref = self._cache["nn"]
        z = ((row[num].astype(float) - mu) / sd).fillna(0.0).to_numpy()
        d = nn.kneighbors(z, n_neighbors=k)[0][0, k - 1]
        return float(np.searchsorted(ref, d, side="right") / len(ref) * 100)

    # -- explanation ----------------------------------------------------------
    def baseline(self):
        return baseline_row(self.features, self.info)

    def shapley(self, name, row, class_index=None, max_exact=11, n_perm=160):
        """
        Exact 'baseline Shapley' contributions for one prediction.

        Each feature's contribution is its average marginal effect of switching
        from the typical value (training median / mode) to the entered value,
        over every order in which features could be switched. The
        contributions add up EXACTLY to prediction - baseline prediction, so
        the chart can be read as a ledger. Computed exactly (2^M evaluations
        in one batch) up to max_exact features, and by averaging random
        permutations beyond that — each permutation telescopes to the same
        total, so additivity holds exactly either way.
        """
        base = self.baseline()
        M = len(self.features)
        f = lambda X: self.predict_scalar(name, X, class_index)
        if M <= max_exact:
            masks = np.arange(2 ** M)
            rows = pd.concat([base] * len(masks), ignore_index=True)
            for j, feat in enumerate(self.features):
                take = ((masks >> j) & 1).astype(bool)
                col = rows[feat].astype(object)
                col[take] = row[feat].iloc[0]
                rows[feat] = col
            v = f(rows.infer_objects())
            size = np.array([bin(m).count("1") for m in masks])
            w = np.array([math.factorial(s) * math.factorial(M - s - 1) / math.factorial(M)
                          for s in range(M)])
            phi = np.zeros(M)
            for j in range(M):
                without = masks[((masks >> j) & 1) == 0]
                phi[j] = np.sum(w[size[without]] * (v[without | (1 << j)] - v[without]))
            return phi, float(v[0]), float(v[-1])
        rng = np.random.RandomState(SEED)
        orders = [rng.permutation(M) for _ in range(n_perm)]
        blocks = []
        for order in orders:
            cur = base.copy().astype(object)
            blocks.append(cur.copy())
            for j in order:
                cur[self.features[j]] = row[self.features[j]].iloc[0]
                blocks.append(cur.copy())
        v = f(pd.concat(blocks, ignore_index=True).infer_objects()).reshape(n_perm, M + 1)
        phi = np.zeros(M)
        for p, order in enumerate(orders):
            phi[order] += np.diff(v[p])
        return phi / n_perm, float(v[0, 0]), float(v[0, -1])

    def what_if(self, name, row, feature, n=60):
        """Prediction as one feature sweeps its training range, the rest held."""
        fi = self.info[feature]
        if fi["kind"] == "numeric":
            # Sweep a little PAST the training range on both sides, so the chart
            # shows where extrapolation begins instead of stopping exactly at the
            # edge of the data. Never below zero for a quantity that was never
            # negative in training (a mix proportion, an age).
            span = (fi["max"] - fi["min"]) or max(abs(fi["max"]), 1.0)
            lo, hi = fi["min"] - 0.1 * span, fi["max"] + 0.1 * span
            if fi["min"] >= 0:
                lo = max(lo, 0.0)
            cur = float(row[feature].iloc[0])
            lo, hi = min(lo, cur), max(hi, cur)
            if fi["integer"] and hi - lo <= n:
                grid = np.arange(math.floor(lo), math.ceil(hi) + 1, dtype=float)
            else:
                grid = np.unique(np.append(np.linspace(lo, hi, n), cur))
        elif fi["kind"] == "categorical":
            grid = np.array(fi["categories"][:40], dtype=object)
        else:                                  # free text (a date): nothing to sweep
            grid = np.array([row[feature].iloc[0]], dtype=object)
        rows = pd.concat([row] * len(grid), ignore_index=True)
        rows[feature] = grid
        out = self.predict_raw(name, rows)
        return grid, out

    # -- importance & learning curve -----------------------------------------
    def permutation_importance(self, name, n_repeats=5):
        """Drop in the primary score when each raw feature is shuffled (test split)."""
        key = ("perm", name, n_repeats)
        if key in self._cache:
            return self._cache[key]
        rng = np.random.RandomState(SEED)
        score_name = "R2" if self.task == "regression" else "ROC_AUC"
        X = self.X_test.reset_index(drop=True)
        base = self._metrics(self.y_test, self.predict_raw(name, X))[score_name]
        rows = []
        for f in self.features:
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[f] = rng.permutation(Xp[f].to_numpy())
                drops.append(base - self._metrics(self.y_test,
                                                  self.predict_raw(name, Xp))[score_name])
            rows.append({"feature": f, "importance": float(np.mean(drops)),
                         "std": float(np.std(drops))})
        out = (pd.DataFrame(rows).sort_values("importance", ascending=False)
               .reset_index(drop=True))
        out.attrs["score"] = score_name
        out.attrs["baseline"] = base
        self._cache[key] = out
        return out

    def learning_curve(self, name, points=5):
        """Train and CV score against training-set size (in-app models only)."""
        key = ("lc", name, points)
        if key in self._cache:
            return self._cache[key]
        if not self.can_refit or name not in self.b["refit"]:
            return None
        est = self.b["refit"][name]
        if self.task == "classification":
            k = max(2, min(5, int(np.bincount(self.y_train.astype(int)).min())))
            cv = StratifiedKFold(k, shuffle=True, random_state=SEED)
            scoring = "roc_auc" if self.n_classes == 2 else "roc_auc_ovr_weighted"
        else:
            cv = KFold(5, shuffle=True, random_state=SEED)
            scoring = "r2"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sizes, tr, va = learning_curve(
                est, self.X_train, self.y_train, train_sizes=np.linspace(0.2, 1.0, points),
                cv=cv, scoring=scoring, n_jobs=-1, shuffle=True, random_state=SEED)
        out = pd.DataFrame({"train_size": sizes,
                            "train_mean": tr.mean(1), "train_std": tr.std(1),
                            "cv_mean": va.mean(1), "cv_std": va.std(1)})
        out.attrs["score"] = "R2" if self.task == "regression" else "ROC_AUC"
        self._cache[key] = out
        return out


# =============================================================================
#  DIAGNOSTICS — the data behind the Model tab's graphs
# =============================================================================
def confusion(ws, name, split="test"):
    from sklearn.metrics import confusion_matrix
    y = ws.y_test if split == "test" else ws.y_train
    pred = ws.pred(name, split).argmax(axis=1)
    return confusion_matrix(y.astype(int), pred, labels=list(range(ws.n_classes)))


def roc_curves(ws, name):
    """[(label, fpr, tpr, auc, is_average)] — one curve for binary, OvR + macro otherwise."""
    from sklearn.metrics import auc, roc_curve
    y, P = ws.y_test.astype(int), ws.pred(name)
    if ws.n_classes == 2:
        fpr, tpr, _ = roc_curve(y == ws.pos, P[:, ws.pos])
        return [(f"{ws.classes[ws.pos]} vs rest", fpr, tpr, auc(fpr, tpr), False)]
    out, grid, tprs = [], np.linspace(0, 1, 201), []
    for k in range(ws.n_classes):
        if (y == k).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y == k, P[:, k])
        out.append((str(ws.classes[k]), fpr, tpr, auc(fpr, tpr), False))
        tprs.append(np.interp(grid, fpr, tpr))
    if len(out) > 8:          # past eight hues the classes fold into the average
        out = []
    if tprs:
        mean = np.mean(tprs, axis=0)
        mean[0] = 0.0
        out.append(("Macro average", grid, mean, auc(grid, mean), True))
    return out


def pr_curves(ws, name):
    """[(label, recall, precision, ap, prevalence, is_average)]"""
    from sklearn.metrics import precision_recall_curve
    y, P = ws.y_test.astype(int), ws.pred(name)
    ks = [ws.pos] if ws.n_classes == 2 else list(range(ws.n_classes))
    out = []
    for k in ks:
        if (y == k).sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y == k, P[:, k])
        ap = average_precision_score(y == k, P[:, k])
        out.append((str(ws.classes[k]), rec, prec, ap, float(np.mean(y == k)), False))
    return out if len(out) <= 8 else out[:0]


def calibration_curves(ws, name, bins=10):
    from sklearn.calibration import calibration_curve
    y, P = ws.y_test.astype(int), ws.pred(name)
    ks = [ws.pos] if ws.n_classes == 2 else list(range(min(ws.n_classes, 8)))
    out = []
    for k in ks:
        if (y == k).sum() == 0:
            continue
        frac, mean_pred = calibration_curve(y == k, P[:, k], n_bins=bins, strategy="quantile")
        out.append((f"{ws.classes[k]}", mean_pred, frac))
    return out


def threshold_sweep(ws, name):
    """Binary only: precision, recall and F1 across decision thresholds."""
    y = (ws.y_test.astype(int) == ws.pos).astype(int)
    p = ws.pred(name)[:, ws.pos]
    th = np.linspace(0.01, 0.99, 99)
    prec, rec, f1 = [], [], []
    for t in th:
        yh = (p >= t).astype(int)
        prec.append(precision_score(y, yh, zero_division=0))
        rec.append(recall_score(y, yh, zero_division=0))
        f1.append(f1_score(y, yh, zero_division=0))
    f1 = np.array(f1)
    return th, np.array(prec), np.array(rec), f1, float(th[int(np.argmax(f1))])


def per_class_report(ws, name):
    from sklearn.metrics import precision_recall_fscore_support
    y, pred = ws.y_test.astype(int), ws.pred(name).argmax(axis=1)
    p, r, f, sup = precision_recall_fscore_support(
        y, pred, labels=list(range(ws.n_classes)), zero_division=0)
    return pd.DataFrame({"Precision": p, "Recall": r, "F1": f, "Support": sup},
                        index=[str(c) for c in ws.classes])


def probability_groups(ws, name):
    """Binary: P(positive) split by true class. Multiclass: P(true class) per class."""
    y, P = ws.y_test.astype(int), ws.pred(name)
    if ws.n_classes == 2:
        return ([(f"true {ws.classes[k]}", P[y == k, ws.pos]) for k in range(2)],
                f"Predicted P({ws.classes[ws.pos]})")
    return ([(str(ws.classes[k]), P[y == k, k]) for k in range(ws.n_classes) if (y == k).any()],
            "Predicted probability of the TRUE class")


def taylor_stats(ws):
    """Correlation and SD ratio of every model's test predictions (regression)."""
    y = ws.y_test.astype(float)
    rows = {}
    for n in ws.model_names:
        p = ws.pred(n).astype(float)
        r = np.corrcoef(y, p)[0, 1] if np.std(p) > 0 else 0.0
        rows[n] = {"R": float(r), "SD ratio": float(np.std(p) / np.std(y)),
                   "Centred RMS": float(np.sqrt(np.mean(((p - p.mean()) - (y - y.mean())) ** 2)) / np.std(y))}
    return pd.DataFrame(rows).T


def _fmt(v):
    """Compact, honest number formatting for messages."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    a = abs(v)
    if a >= 1e5 or (a and a < 1e-3):
        return f"{v:.3g}"
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.4g}" if a < 1 else f"{v:,.3f}".rstrip("0").rstrip(".")


# =============================================================================
#  BUNDLES — save and load
# =============================================================================
def save_bundle(bundle):
    """
    Serialise a bundle to bytes, self-contained.

    This module is pickled BY VALUE (cloudpickle.register_pickle_by_value), so
    a saved file does not need gui/engine.py to be importable under the same
    name when it is loaded back — it only needs the same library versions,
    which is the usual pickle rule and worth saying out loud: load it in the
    environment that trained it.
    """
    import cloudpickle
    import sys
    mod = sys.modules[__name__]
    cloudpickle.register_pickle_by_value(mod)
    try:
        return cloudpickle.dumps(bundle)
    finally:
        cloudpickle.unregister_pickle_by_value(mod)


def load_bundle(source):
    """
    Bytes, a path, or a file-like object -> bundle dict.

    Loading a pickle can execute code: only open bundles you created (the
    notebook's export, or the app's own Save button).
    """
    import pickle
    import cloudpickle  # noqa: F401 — must be importable to rebuild closures
    if isinstance(source, (bytes, bytearray)):
        return pickle.loads(source)
    if hasattr(source, "read"):
        return pickle.loads(source.read())
    with open(source, "rb") as fh:
        return pickle.load(fh)


# =============================================================================
#  DEMO DATA — so the app can be tried before a file is ready
# =============================================================================
def demo_dataset(kind, n=400, seed=7):
    """
    A synthetic concrete-mix table with a known structure.

    Regression: compressive strength (MPa) from a mix design, rising with
    binder content and curing age and falling with the water/binder ratio.
    Classification: the same mixes graded 'low' / 'medium' / 'high'.
    """
    rng = np.random.RandomState(seed)
    cement = rng.uniform(150, 500, n)
    slag = rng.uniform(0, 250, n) * (rng.rand(n) > 0.4)
    fly_ash = rng.uniform(0, 180, n) * (rng.rand(n) > 0.5)
    water = rng.uniform(140, 230, n)
    sp = rng.uniform(0, 20, n).round(1)
    coarse = rng.uniform(800, 1150, n)
    fine = rng.uniform(600, 950, n)
    age = rng.choice([3, 7, 14, 28, 56, 90, 180], n)
    binder = cement + slag + 0.7 * fly_ash
    wb = water / binder
    strength = (95 * np.exp(-2.1 * wb) * (1 - np.exp(-0.18 * age ** 0.9))
                + 0.35 * sp + rng.normal(0, 3.0, n))
    df = pd.DataFrame({"Cement": cement.round(1), "Slag": slag.round(1),
                       "FlyAsh": fly_ash.round(1), "Water": water.round(1),
                       "Superplasticizer": sp, "CoarseAgg": coarse.round(1),
                       "FineAgg": fine.round(1), "Age": age})
    if kind == "regression":
        df["Strength"] = strength.round(2)
    else:
        df["Grade"] = pd.cut(strength, [-np.inf, 25, 45, np.inf],
                             labels=["low", "medium", "high"]).astype(str)
    return df

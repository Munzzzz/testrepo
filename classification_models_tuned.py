# =============================================================================
#  TUNED CLASSIFICATION BENCHMARK
#  9 models, leakage-free stratified CV, nested CV, uniform
#  fit/predict_proba/evaluate interface, Accuracy/Balanced-Accuracy/Precision/
#  Recall/F1/MCC/Kappa/ROC-AUC/PR-AUC/LogLoss/Brier metrics, confusion
#  matrices, ROC & PR curves, calibration, threshold analysis, and a K-fold
#  count stability sweep.
# =============================================================================
#  This is the classification counterpart of the regression benchmark. Every
#  stage was re-derived for a categorical target rather than translated
#  mechanically — the places where the two genuinely differ are flagged with
#  "REGRESSION -> CLASSIFICATION" notes so the reasoning is auditable:
#
#    * the target is LABEL-ENCODED automatically (Section A), and one-hot
#      encoded on top of that for the multiclass Keras heads. The regression
#      script's StandardScaler on y is GONE — scaling a class index is
#      meaningless, and encoding is what replaces it.
#    * KFold -> StratifiedKFold everywhere (class priors preserved per fold).
#    * RMSE/MAE/MAPE/SI/R2 -> a threshold metric set (Accuracy, Balanced
#      Accuracy, Precision, Recall, F1, MCC, Kappa) PLUS a ranking/probability
#      set (ROC-AUC, PR-AUC, LogLoss, Brier). Both are reported because they
#      answer different questions and disagree under class imbalance.
#    * actual-vs-predicted scatter -> CONFUSION MATRICES.
#    * residual analysis -> ROC curves, PR curves, calibration curves, and a
#      decision-threshold sweep.
#    * squared-error bias-variance -> Kohavi-Wolpert 0-1 loss decomposition.
#    * every model exposes predict_proba; hard labels are argmax of the
#      probabilities, so labels and probabilities can never disagree.
#
#  End result: a plain list you can loop over —
#      models = [rf, mlp, svc, xgboost_model, lgbm, ada, knn, cnn_lstm, seq_logit]
#      fit_all(models, X_tr, y_tr)
#      evaluate_all(models, X_te, y_te)
#  Any single model still works standalone: svc.fit(X_tr, y_tr); svc.predict(X_te)
#
#  pip install -U scikit-learn xgboost lightgbm optuna tensorflow shap matplotlib seaborn
# =============================================================================

# =============================================================================
# =============================================================================
#  This file is organized into 7 parts:
#    Part 1 — Prerequisites, library imports & common setup
#              (data loading + target auto-encoding + seed selection, global
#               imports/constants, metric functions, the ModelSpec interface)
#    Part 2 — Search strategies (why each model uses Grid/Random/Optuna)
#    Part 3 — Exploratory data analysis (class balance, per-class distributions,
#              box plots, correlation heatmaps, feature-target association)
#    Part 4 — Model space (the 9 model definitions + hyperparameter spaces)
#    Part 5 — Hyperparameter optimization (fit_all + K-fold stability sweep)
#    Part 6 — Evaluation metrics (test-set comparison, learning curves,
#              train-vs-test table, CONFUSION MATRICES, ROC curves, PR curves,
#              calibration, threshold analysis, Taylor diagram on probabilities,
#              bias-variance, seed sensitivity)
#    Part 7 — Factor importance (SHAP, ICE/PDP)
#  Section labels are kept from the regression file for continuity.
# =============================================================================
# =============================================================================

# =============================================================================
# =============================================================================
#  PART 1 — PREREQUISITES, LIBRARY IMPORTS & COMMON SETUP
# =============================================================================
# =============================================================================

# =============================================================================
#  HOW TO RUN IT: TOP TO BOTTOM, IN ORDER.
#
#  Every section builds on state the earlier ones created — the fitted models,
#  the split, the SHAP results, the configuration constants. Running a cell on
#  its own in a fresh kernel raises NameError on whichever of those it reaches
#  first, which points at the cell you are in rather than at the cells you
#  skipped. Function definitions are written so the `def` itself never needs
#  anything from another cell (defaults are resolved when the function is
#  CALLED, not when it is defined), but the data they operate on still has to
#  exist. Re-running a section after a kernel restart means re-running from
#  Section 0.
# =============================================================================
# =============================================================================
#  SECTION 0 — COMMON SETUP  (run this cell once, before everything else)
# =============================================================================
import warnings, time, os, re, textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# `display` is injected by IPython/Colab. The shim keeps the file runnable as a
# plain script too, so nothing below depends on being inside a notebook.
try:
    display
except NameError:
    def display(*objs):
        for o in objs:
            print(o)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, LabelEncoder,
                                   label_binarize)

# FEATURE-SCALER KNOB. Set to MinMaxScaler to min-max scale instead of
# standardizing. Used for every FEATURE scaler in this file — inside the
# KNN/SVC Pipelines and inside the per-fold Keras scaling — so switching it
# here changes every model at once WITHOUT any leakage: the scaler is still
# refit on the training part of each fold rather than on the full training set
# up front. (See Section A for why pre-scaling x_train/x_test globally is the
# thing to avoid.)
#
# REGRESSION -> CLASSIFICATION. The regression file also carried a TARGET
# scaler (StandardScaler on y, for the Keras nets). There is no such thing
# here: the target is a set of unordered class labels, and the encoding that
# replaces the scaler is LabelEncoder (+ one-hot for the multiclass Keras
# heads), applied in Section A. Standardizing a class index would invent an
# ordering and a distance between classes that do not exist.
SCALER_CLS = StandardScaler

from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score,
    average_precision_score, log_loss, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc,
)
from sklearn.utils.class_weight import compute_class_weight

# STRATIFIED K-Fold, not plain KFold. Rows are independent, so there is no
# temporal or grouping structure to protect against — but the CLASS BALANCE
# does have to be protected. With an imbalanced target, plain KFold can hand a
# fold a training split that is missing the minority class entirely, which
# makes ROC-AUC undefined and log_loss raise, and it makes every fold's score
# noisier for no benefit. Stratifying costs nothing and removes that failure
# mode; it is the correct default for classification.
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)   # picks hyperparameters
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)   # nested-CV generalisation estimate

# ---- Primary tuning objective ------------------------------------------------
#  Every search in this file — GridSearchCV, RandomizedSearchCV and Optuna —
#  optimizes THE SAME metric, so the models are ranked on one consistent
#  criterion. ROC-AUC is the default because it is threshold-free: it scores
#  the model's ability to RANK positives above negatives, so it does not
#  silently reward a model for being lucky at the arbitrary 0.5 cutoff, and it
#  is insensitive to class prevalence.
#
#  SWITCH IT HERE if your problem calls for something else:
#    "average_precision"  -> PR-AUC. Preferred over ROC-AUC when positives are
#                            rare and you care about the positive class only;
#                            ROC-AUC can look flattering at 1% prevalence.
#    "f1_macro"           -> if a single operating point is what you ship and
#                            every class matters equally.
#    "balanced_accuracy"  -> imbalance-corrected accuracy.
#    "neg_log_loss"       -> if calibrated probabilities are the deliverable.
#  SCORING is resolved in Section A, once the number of classes is known
#  (binary and multiclass need different sklearn scorer names).
PRIMARY_METRIC = "ROC_AUC"
SCORING = None          # set by resolve_scoring() in Section A


# =============================================================================
#  TUNING BUDGET — READ THIS BEFORE THE FIRST RUN
# =============================================================================
#  The dominant cost in this file is the three Keras models, and it is not
#  obvious from the code how large it gets. One Optuna trial trains a network
#  once PER CV FOLD, so the search alone is n_trials x n_folds networks, and
#  the neural models are then refit again by the K-fold sweep (Part 5), the
#  learning curves (Part 6), the bias-variance bootstrap (Part 6) and — most
#  expensively — by every ensemble that contains them (Section 4B), which
#  refits its members for each out-of-fold pass.
#
#  At the "full" budget that totals roughly 1,000 network fits. On a GPU that
#  is an afternoon; on a notebook CPU it is 2-8 hours, and the first symptom is
#  the search appearing to hang on "Shallow MLP" for 40 minutes. It is not
#  hung — it is doing 200 trainings.
#
#  So pick a profile. "balanced" is the default and is what most tabular
#  problems want; "fast" is for a first pass or a CPU-only machine; "full" is
#  the original budget and should be reserved for a final, GPU-backed run.
#
#  If you do not need the neural models at all — and on small tabular data the
#  boosted trees usually win — the single biggest saving is to drop them from
#  the roster in Section 4:
#      models = [rf, svc, xgboost_model, lgbm, ada, knn]
#  That removes every Keras fit in the file and typically takes the whole run
#  from hours to minutes.
# =============================================================================
TUNING_PROFILE = "balanced"     # "fast" | "balanced" | "full"

_PROFILES = {
    #                     trials per Optuna model              random  epochs pat folds  timeout
    "fast":     (dict(mlp=8,  xgb=25, lgbm=25, svc=15, cnn_lstm=5,  seq_logit=8),  20,  120,  8, 3,  300),
    "balanced": (dict(mlp=20, xgb=50, lgbm=50, svc=30, cnn_lstm=12, seq_logit=15), 40,  200, 15, 3,  900),
    "full":     (dict(mlp=40, xgb=80, lgbm=80, svc=60, cnn_lstm=30, seq_logit=25), 60,  300, 25, 5, None),
}
if TUNING_PROFILE not in _PROFILES:
    raise ValueError(f"TUNING_PROFILE must be one of {list(_PROFILES)}")
(N_TRIALS, N_ITER_RANDOM, KERAS_EPOCHS, KERAS_PATIENCE,
 KERAS_CV_SPLITS, STUDY_TIMEOUT) = _PROFILES[TUNING_PROFILE]

#  The searches are not the only repeated-refit loops. The K-fold stability
#  sweep (Part 5), the learning curves and the bias-variance bootstrap (Part 6)
#  each refit every model many times over, so they scale with the profile too.
_DOWNSTREAM = {
    "fast":     dict(sweep_max_k=4,  lc_sizes=3,  n_bootstrap=5),
    "balanced": dict(sweep_max_k=7,  lc_sizes=5,  n_bootstrap=10),
    "full":     dict(sweep_max_k=11, lc_sizes=10, n_bootstrap=20),
}
SWEEP_MAX_K  = _DOWNSTREAM[TUNING_PROFILE]["sweep_max_k"]
LC_SIZES     = _DOWNSTREAM[TUNING_PROFILE]["lc_sizes"]
N_BOOTSTRAP  = _DOWNSTREAM[TUNING_PROFILE]["n_bootstrap"]

print(f"Tuning profile: {TUNING_PROFILE}  |  Optuna trials: {N_TRIALS}  |  "
      f"random-search draws: {N_ITER_RANDOM}")
print(f"  Keras: {KERAS_EPOCHS} epochs, patience {KERAS_PATIENCE}, "
      f"{KERAS_CV_SPLITS} search folds  |  per-model search budget: "
      f"{'unlimited' if STUDY_TIMEOUT is None else str(STUDY_TIMEOUT) + 's'}")

#  The Keras searches get their own, usually smaller, fold count. Each fold
#  costs a full network training, so 5 -> 3 folds is a 40% saving on the single
#  most expensive part of the run. The tuning signal barely changes: fold count
#  affects the variance of the CV estimate, and the search only needs to RANK
#  configurations, not measure them precisely. The final evaluation in Part 6
#  is unaffected — it uses the held-out test set either way.
keras_cv = StratifiedKFold(n_splits=KERAS_CV_SPLITS, shuffle=True, random_state=SEED)

#  STUDY_TIMEOUT is a hard wall-clock bound, in seconds, on EACH model's
#  search. Optuna finishes the trial it is running and then stops, keeping the
#  best parameters found so far, so a slow model degrades into a shorter search
#  instead of stalling the notebook indefinitely. None = no limit.
#
#  A search that stops on the timeout rather than on n_trials is reported as
#  such below, so an under-searched model is never mistaken for a fully tuned
#  one in the results table.

RESULTS = {}   # name -> dict of test metrics, filled in by report()


# =============================================================================
#  SECTION C — OUTPUT COLLECTION  (every figure separately, every table in one
#  workbook)
# =============================================================================
#  Two rules for everything this file produces:
#
#  FIGURES. One file per analysis, and — where an analysis covers several
#  features, metrics or parameters — one file per FEATURE, METRIC or PARAMETER
#  as well. Multi-panel grids are still drawn, because they are how you scan a
#  dataset, but a grid is a contact sheet, not a deliverable: a single panel of
#  it cannot be dropped into a paper, resized, or referenced on its own. So the
#  grids are kept AND each panel is written separately, into a subdirectory per
#  analysis so the count stays navigable.
#
#  TABLES. Every result table is registered as it is produced and the whole set
#  is written to ONE Excel workbook at the end, one sheet per table. Scattering
#  a dozen CSVs across a working directory makes them easy to lose and easy to
#  mismatch; a single workbook keeps a run's numbers together and dated.
# =============================================================================
import re

OUTPUT_DIR   = "outputs"
FIGURE_DIR   = os.path.join(OUTPUT_DIR, "figures")
RESULTS_XLSX = os.path.join(OUTPUT_DIR, "results.xlsx")

RESULT_TABLES = {}      # sheet name -> DataFrame, written by export_tables()
SAVED_FIGURES = []      # every path written, reported at the end

#  Write one figure per feature / per metric / per pair in addition to the
#  scan grids. Costs a little time in the EDA (the data is already computed)
#  and one extra partial-dependence pass per model in Part 7; set False if you
#  only want the grids.
PER_ITEM_FIGURES = True


def _slug(name):
    """A filename that survives every OS: no spaces, slashes or punctuation."""
    out = re.sub(r"[^\w\-.]+", "_", str(name).strip())
    return re.sub(r"_+", "_", out).strip("_") or "figure"


def _fname(name):
    """A model name as a filename fragment. Defined here rather than in Part 6
    because Part 5 names files too."""
    return name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')


def save_fig(fig=None, name="figure", subdir="", dpi=300, close=False):
    """
    Write one figure to outputs/figures/<subdir>/<name>.png.

    Centralised so that resolution, background and bounding box are identical
    everywhere, and so the full list of outputs can be reported at the end
    rather than left for you to find.

    close defaults to False because the callers render with plt.show() and then
    close: the inline notebook backend releases a figure on show(), so writing
    it afterwards would produce a blank file.
    """
    fig = fig or plt.gcf()
    name = str(name)
    if name.lower().endswith(".png"):      # tolerate callers that pass a filename
        name = name[:-4]
    directory = os.path.join(FIGURE_DIR, subdir) if subdir else FIGURE_DIR
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{_slug(name)}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    SAVED_FIGURES.append(path)
    if close:
        plt.close(fig)
    return path


def _sheet_name(name, taken):
    """
    Excel sheet names: 31 characters, and none of : \ / ? * [ ].
    Truncating blindly collides (two long names share a prefix), so collisions
    are resolved with a numeric suffix rather than silently overwriting a sheet.
    """
    clean = re.sub(r"[:\\/?*\[\]]", "-", str(name)).strip() or "Sheet"
    clean = clean[:31]
    if clean not in taken:
        return clean
    stem = clean[:28]
    for i in range(2, 100):
        candidate = f"{stem}_{i}"
        if candidate not in taken:
            return candidate
    raise ValueError(f"cannot make a unique sheet name for {name!r}")


def register_table(name, df, index=False):
    """Keep a result table for the single Excel export at the end."""
    df = df.reset_index() if index else df
    RESULT_TABLES[name] = df.copy()
    return df


def export_tables(path=None):
    """Write every registered table to one workbook, one sheet per table."""
    path = RESULTS_XLSX if path is None else path
    if not RESULT_TABLES:
        print("No tables registered — nothing to export.")
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    taken = {}
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in RESULT_TABLES.items():
            sheet = _sheet_name(name, taken)
            taken[sheet] = name
            # Objects such as dicts of best params do not survive Excel's type
            # coercion, so anything non-scalar is written as its repr.
            safe = df.copy()
            for col in safe.columns:
                if safe[col].dtype == object:
                    safe[col] = safe[col].map(
                        lambda v: v if isinstance(v, (str, int, float, bool, type(None)))
                        else repr(v))
            safe.to_excel(writer, sheet_name=sheet, index=False)
    print(f"\n{len(RESULT_TABLES)} table(s) written to {path}")
    for sheet, name in taken.items():
        print(f"    [{sheet}] {name}  ({len(RESULT_TABLES[name])} rows)")
    return path

# ---- Target encoding state (populated by encode_target() in Section A) -------
#  These are module-level on purpose: the metric functions below, the Keras
#  heads in Part 4, and the plotting code in Part 6 all need to know how many
#  classes there are and what the original labels were called. They are read at
#  CALL time, not at definition time, so defining them here and filling them in
#  Section A is safe.
LABEL_ENCODER = None    # fitted sklearn LabelEncoder
CLASSES       = None    # original class labels, in encoded order (0, 1, 2, ...)
N_CLASSES     = None
IS_BINARY     = None
POS_LABEL     = 1       # encoded index treated as "positive" in binary reporting


# =============================================================================
#  AUTO-ENCODING THE OUTPUT
# =============================================================================
#  The single biggest structural difference from the regression pipeline. A
#  regression target is fed to the models as-is (optionally standardized); a
#  classification target has to be turned into contiguous integer class indices
#  0..K-1 first, because that is what sklearn, XGBoost, LightGBM and the Keras
#  losses all expect. Doing it by hand is where silent bugs come from:
#
#    * df['y'].replace({'no': 0, 'yes': 1}) breaks the moment an unseen or
#      misspelled label appears — it stays a string and everything downstream
#      either crashes or, worse, treats it as a third class.
#    * XGBoost requires labels to be exactly 0..K-1. Labels like {1, 2} or
#      {-1, +1} raise, and {0, 2} silently implies a 3-class problem.
#    * the mapping has to be REVERSIBLE, or the confusion matrix axes and the
#      classification report end up labelled 0/1/2 instead of the names a
#      reader can interpret.
#
#  encode_target() does all of that once, keeps the fitted encoder so every
#  later section can decode back to the original labels, and prints the mapping
#  so the direction of the encoding is on the record rather than assumed.
# =============================================================================
def encode_target(y_raw, positive_label=None, verbose=True):
    """
    Label-encode a target of ANY dtype (str / bool / int / float / Categorical)
    into contiguous integers 0..K-1.

    positive_label: for a BINARY target, the original label that should be
        treated as the positive class (encoded as 1) in precision/recall/F1,
        ROC/PR curves and threshold tuning. LabelEncoder sorts labels, so by
        default 'yes' > 'no' and 'PAIDOFF' < 'COLLECTION' — which is often NOT
        the class you care about. Name it explicitly whenever it matters.

    Returns (y_encoded, fitted_LabelEncoder).
    """
    y_ser = pd.Series(np.asarray(y_raw).ravel())

    n_missing = int(y_ser.isna().sum())
    if n_missing:
        raise ValueError(
            f"target has {n_missing} missing values. A row with no label cannot "
            f"be trained or scored on — drop those rows (df = df.dropna(subset=[TARGET_COL])) "
            f"or treat 'missing' as its own class deliberately."
        )

    # A float column holding only whole numbers (1.0, 2.0) is a class label
    # that pandas widened, not a continuous target. Casting to int first keeps
    # the printed mapping readable ('1' rather than '1.0').
    if y_ser.dtype.kind == "f" and np.allclose(y_ser.dropna() % 1, 0):
        y_ser = y_ser.astype("int64")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_ser).astype(np.int64)

    n_classes = len(le.classes_)
    if n_classes < 2:
        raise ValueError(f"target has only {n_classes} distinct value(s) — nothing to classify.")

    # Sanity check against a continuous target pasted in by mistake: 40 distinct
    # labels over 1000 rows is almost always a regression target.
    if n_classes > max(20, 0.05 * len(y_ser)):
        warnings.warn(
            f"target has {n_classes} distinct values over {len(y_ser)} rows. "
            f"That looks like a continuous variable — if so this is a regression "
            f"problem and belongs in the regression pipeline, not this one.",
            stacklevel=2)

    global LABEL_ENCODER, CLASSES, N_CLASSES, IS_BINARY, POS_LABEL
    LABEL_ENCODER = le
    CLASSES       = list(le.classes_)
    N_CLASSES     = n_classes
    IS_BINARY     = (n_classes == 2)

    if IS_BINARY and positive_label is not None:
        if positive_label not in CLASSES:
            raise ValueError(f"positive_label={positive_label!r} is not one of {CLASSES}")
        POS_LABEL = int(le.transform([positive_label])[0])
    else:
        POS_LABEL = 1 if IS_BINARY else None

    if verbose:
        counts = pd.Series(y_enc).value_counts().sort_index()
        table = pd.DataFrame({
            "encoded":  counts.index,
            "original": [CLASSES[i] for i in counts.index],
            "count":    counts.values,
            "share":    (counts.values / len(y_enc)).round(4),
        })
        print("\n" + "=" * 78)
        print(f"TARGET ENCODING  ({'binary' if IS_BINARY else f'{n_classes}-class multiclass'})")
        print("=" * 78)
        print(table.to_string(index=False))
        if IS_BINARY:
            print(f"positive class = {CLASSES[POS_LABEL]!r}  (encoded {POS_LABEL})")

        # Imbalance is not an error, but it decides which metrics are honest:
        # at 95/5, a model that predicts the majority class for every row scores
        # 95% accuracy and is worthless. Balanced accuracy, F1, MCC and PR-AUC
        # are the ones that catch it.
        share = counts.values / len(y_enc)
        ratio = share.max() / share.min()
        if ratio >= 1.5:
            print(f"\nCLASS IMBALANCE: majority/minority ratio = {ratio:.1f}:1.")
            print("  -> read Balanced Accuracy / F1 / MCC / PR-AUC, not raw Accuracy.")
            print("  -> class_weight='balanced' is in the search space of every model")
            print("     that supports it, so the tuner can buy back the minority class.")
    return y_enc, le


def decode_target(y_enc):
    """Encoded indices -> original labels. Used for readable plot axes/reports."""
    if LABEL_ENCODER is None:
        return np.asarray(y_enc)
    return LABEL_ENCODER.inverse_transform(np.asarray(y_enc).astype(int))


def one_hot(y_enc, n_classes=None):
    """
    Integer class indices -> one-hot matrix (n, K).

    The SECOND encoding of the output, needed only by the multiclass Keras
    heads: categorical_crossentropy expects a probability vector per row, not
    an index. (Binary heads keep the 0/1 index and use binary_crossentropy, and
    sklearn/XGBoost/LightGBM all want the index form too — so the one-hot form
    is deliberately produced at the point of use rather than globally.)
    """
    n_classes = n_classes or N_CLASSES
    y_enc = np.asarray(y_enc).astype(int).ravel()
    out = np.zeros((len(y_enc), n_classes), dtype=np.float32)
    out[np.arange(len(y_enc)), y_enc] = 1.0
    return out


def resolve_scoring(n_classes=None):
    """
    sklearn scorer string matching PRIMARY_METRIC, for the given task shape.
    Binary and multiclass need different names for the same idea, and getting
    this wrong is a silent 'ValueError: multiclass format is not supported'
    halfway through a 3-hour search.
    """
    n_classes = n_classes or N_CLASSES
    binary = (n_classes == 2)
    table = {
        "ROC_AUC":           "roc_auc"           if binary else "roc_auc_ovr_weighted",
        "PR_AUC":            "average_precision" if binary else "f1_macro",
        "F1":                "f1"                if binary else "f1_macro",
        "Balanced_Accuracy": "balanced_accuracy",
        "Accuracy":          "accuracy",
        "LogLoss":           "neg_log_loss",
    }
    if PRIMARY_METRIC not in table:
        raise ValueError(f"PRIMARY_METRIC={PRIMARY_METRIC!r} not one of {list(table)}")
    if PRIMARY_METRIC == "PR_AUC" and not binary:
        warnings.warn("average_precision has no direct multiclass scorer; using f1_macro "
                      "for the search. PR-AUC is still reported per class in Part 6.",
                      stacklevel=2)
    return table[PRIMARY_METRIC]


# ---- Metrics -----------------------------------------------------------------
#  Two families, both reported, because they answer different questions:
#
#  THRESHOLD METRICS (Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC,
#  Kappa) score the hard labels, i.e. the model AFTER a decision rule has been
#  applied. They are what the deployed system's behaviour looks like, and they
#  all move when you move the threshold.
#
#  RANKING / PROBABILITY METRICS (ROC-AUC, PR-AUC, LogLoss, Brier) score the
#  predicted probabilities directly, with no threshold. ROC-AUC and PR-AUC ask
#  "are positives ranked above negatives"; LogLoss and Brier are proper scoring
#  rules that also punish MIS-CALIBRATION — a model can rank perfectly
#  (AUC = 1.0) while every probability it emits is wrong.
#
#  Reporting only one family is how classification write-ups go wrong: 99%
#  accuracy on a 99/1 split, or an excellent AUC from a model whose 0.5
#  threshold never fires.
def _proba_pos(proba):
    """Binary: the positive-class column. Shape (n, 2) -> (n,)."""
    proba = np.asarray(proba)
    return proba[:, POS_LABEL] if proba.ndim == 2 else proba


def roc_auc(y_true, proba):
    """ROC-AUC, binary or multiclass (one-vs-rest, prevalence-weighted)."""
    y_true = np.asarray(y_true).astype(int)
    labels = list(range(N_CLASSES))
    # A y_true slice that happens to contain a single class (a tiny CV fold)
    # makes AUC mathematically undefined rather than zero — report NaN.
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if IS_BINARY:
        return float(roc_auc_score(y_true, _proba_pos(proba)))
    return float(roc_auc_score(y_true, np.asarray(proba), multi_class="ovr",
                               average="weighted", labels=labels))


def pr_auc(y_true, proba):
    """
    Average precision (area under the precision-recall curve).
    Preferred over ROC-AUC when the positive class is rare: ROC-AUC's false
    positive rate has the (large) negative count in its denominator, so a flood
    of false positives barely moves it, while precision collapses immediately.
    """
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if IS_BINARY:
        return float(average_precision_score((y_true == POS_LABEL).astype(int),
                                             _proba_pos(proba)))
    Y = label_binarize(y_true, classes=list(range(N_CLASSES)))
    return float(average_precision_score(Y, np.asarray(proba), average="macro"))


def brier(y_true, proba):
    """
    Brier score = mean squared error between the predicted probability vector
    and the one-hot truth. This IS the regression pipeline's MSE, applied to
    probabilities instead of to the target — the closest direct analogue in the
    whole file. Lower is better; it decomposes into calibration + refinement,
    which is why the calibration curves in Part 6 read as its diagnostic.
    """
    y_true = np.asarray(y_true).astype(int)
    P = np.asarray(proba, dtype=np.float64)
    if P.ndim == 1:
        P = np.column_stack([1 - P, P])
    return float(np.mean(np.sum((P - one_hot(y_true, P.shape[1])) ** 2, axis=1)))


def compute_metrics(y_true, y_pred, proba=None):
    """
    One row of the results table. `average` switches between the binary and
    macro conventions automatically: for a binary target the interesting
    precision/recall/F1 are those of the POSITIVE class, while for multiclass
    the macro average (every class weighted equally, regardless of size) is the
    honest default — a weighted average just re-reports accuracy.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(N_CLASSES))
    avg = "binary" if IS_BINARY else "macro"
    kw = dict(average=avg, zero_division=0, labels=labels)
    if IS_BINARY:
        kw["pos_label"] = POS_LABEL

    m = {
        "Accuracy":     float(accuracy_score(y_true, y_pred)),
        "Balanced_Acc": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision":    float(precision_score(y_true, y_pred, **kw)),
        "Recall":       float(recall_score(y_true, y_pred, **kw)),
        "F1":           float(f1_score(y_true, y_pred, **kw)),
        "F1_weighted":  float(f1_score(y_true, y_pred, average="weighted",
                                       zero_division=0, labels=labels)),
        # MCC is the most informative single number on an imbalanced binary
        # problem: it uses all four confusion-matrix cells and only goes high
        # when the model is right about BOTH classes. Kappa is the same idea
        # expressed as agreement above chance.
        "MCC":          float(matthews_corrcoef(y_true, y_pred)),
        "Kappa":        float(cohen_kappa_score(y_true, y_pred, labels=labels)),
    }
    if proba is not None:
        P = np.asarray(proba, dtype=np.float64)
        m["ROC_AUC"] = roc_auc(y_true, P)
        m["PR_AUC"]  = pr_auc(y_true, P)
        # labels= is mandatory: a CV fold whose y_true is missing a class would
        # otherwise make log_loss infer the wrong number of columns and raise.
        m["LogLoss"] = float(log_loss(y_true, P, labels=labels))
        m["Brier"]   = brier(y_true, P)
    else:
        m.update({"ROC_AUC": float("nan"), "PR_AUC": float("nan"),
                  "LogLoss": float("nan"), "Brier": float("nan")})
    return m


def primary_score(y_true, proba, y_pred=None):
    """The number every hyperparameter search maximises. Higher is better."""
    if PRIMARY_METRIC == "ROC_AUC":
        return roc_auc(y_true, proba)
    if PRIMARY_METRIC == "PR_AUC":
        return pr_auc(y_true, proba)
    if PRIMARY_METRIC == "LogLoss":
        return -float(log_loss(np.asarray(y_true).astype(int), np.asarray(proba),
                               labels=list(range(N_CLASSES))))
    y_pred = labels_from_proba(proba) if y_pred is None else y_pred
    return compute_metrics(y_true, y_pred, proba)[
        {"F1": "F1", "Balanced_Accuracy": "Balanced_Acc", "Accuracy": "Accuracy"}[PRIMARY_METRIC]]


def labels_from_proba(proba, threshold=None):
    """
    Probabilities -> hard labels.

    Every model in this file predicts through this one function, so a model's
    labels can never disagree with its own probabilities (they do disagree in
    sklearn's SVC, whose predict() uses the decision function while
    predict_proba() uses a separately-fitted Platt calibration).

    threshold: binary only. None = argmax, i.e. the usual 0.5 cutoff. Part 6's
    threshold analysis passes a tuned value here.
    """
    P = np.asarray(proba)
    if threshold is not None and P.ndim == 2 and P.shape[1] == 2:
        return np.where(P[:, POS_LABEL] >= threshold, POS_LABEL, 1 - POS_LABEL).astype(int)
    return P.argmax(axis=1).astype(int)


def report(name, y_true, y_pred, proba=None, best_params=None, cv_score=None, elapsed=None):
    m = compute_metrics(y_true, y_pred, proba)
    m[f"CV_{PRIMARY_METRIC}"] = cv_score
    m["fit_time_s"] = elapsed
    m["best_params"] = best_params
    RESULTS[name] = m
    print(f"\n=== {name} ===")
    if best_params:
        print("best params :", best_params)
    if cv_score is not None:
        print(f"inner-CV {PRIMARY_METRIC} : {cv_score:.4f}")
    print(f"test ACC : {m['Accuracy']:.4f} | BAL-ACC : {m['Balanced_Acc']:.4f} | "
          f"F1 : {m['F1']:.4f} | MCC : {m['MCC']:.4f}")
    print(f"     AUC : {m['ROC_AUC']:.4f} | PR-AUC : {m['PR_AUC']:.4f} | "
          f"LogLoss : {m['LogLoss']:.4f} | Brier : {m['Brier']:.4f}")
    return m


def nested_cv_score(make_search, X, y, cv=None, label=""):
    """
    Unbiased estimate of the ENTIRE tune-then-fit pipeline: re-runs the search
    from scratch inside each outer fold. Inner-CV scores alone are optimistic
    because they're what the search maximised; this corrects for that.
    `make_search` must be a zero-arg factory returning a *fresh* search object.
    Outer folds are stratified for the same reason the inner ones are.
    """
    cv = cv or outer_cv
    scores = []
    for k, (tr, te) in enumerate(cv.split(X, y), 1):
        s = make_search()
        s.fit(X[tr], y[tr])
        scores.append(primary_score(y[te], s.best_estimator_.predict_proba(X[te])))
        print(f"  [{label}] outer fold {k}: {PRIMARY_METRIC}={scores[-1]:.4f}")
    scores = np.array(scores)
    print(f"  [{label}] nested CV {PRIMARY_METRIC} = {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores.mean(), scores.std()


# -----------------------------------------------------------------------------
#  Scaling always lives inside a Pipeline (or is refit per-fold for Keras).
#  Fitting a scaler on the full training set before CV leaks each fold's
#  held-out mean/std into training. Mandatory for KNN / SVC / MLP / logistic;
#  harmless (unused) for tree models.
# -----------------------------------------------------------------------------


# =============================================================================
#  SECTION A — DATA LOADING, TARGET ENCODING, SEED SELECTION, TRAIN/TEST SPLIT
# =============================================================================
#  Runs right after the library-imports/constants half of Part 1's common setup
#  (so pandas/numpy are already available) and right before the data-array
#  casts (X_tr/X_te/y_tr/y_te), which need x_train/x_test/y_train/y_test to
#  exist first.
#
#  ---------------------------------------------------------------------------
#  WHY THE SEED IS SELECTED ON DISTRIBUTION, NOT ON MODEL PERFORMANCE
#  ---------------------------------------------------------------------------
#  Two very different ways to pick a split seed, and only one is defensible:
#
#    (a) Score each seed by how closely the TRAIN and TEST marginal
#        distributions match — per-column two-sample Kolmogorov-Smirnov
#        statistics on the features, plus a class-proportion check on the
#        target. This uses no model and no predictions. It is essentially
#        automated stratification, and it is what this section does.
#
#    (b) Score each seed by test-set accuracy/AUC of a fitted model and keep
#        the best. DO NOT DO THIS. Choosing the split that maximises the test
#        score turns the test set into a selection criterion, so the reported
#        metric is no longer an estimate of generalisation — it is the maximum
#        of many draws, biased upward, and it will not reproduce. That is
#        test-set leakage and reviewers do check for it.
#
#  ---------------------------------------------------------------------------
#  REGRESSION -> CLASSIFICATION: THE TARGET TERM CHANGES
#  ---------------------------------------------------------------------------
#  The regression version ran a KS test on the target alongside the features.
#  KS compares empirical CDFs, which is meaningless for unordered class labels
#  — it would depend on the arbitrary order LabelEncoder assigned. The correct
#  target-side check is whether the CLASS PROPORTIONS match, measured here as
#  total variation distance (half the L1 distance between the two class-share
#  vectors).
#
#  And because every split below passes stratify=Y_enc, that term is ~0 by
#  construction: stratification already guarantees matched class shares to
#  within one row per class. It is still computed and reported, as a check that
#  stratification did what it claims. The seed scan is therefore really
#  selecting on FEATURE balance, which stratification does NOT give you —
#  that is the gap this section fills.
# =============================================================================
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
from scipy.stats import (pearsonr as st_pearsonr, spearmanr as st_spearmanr,
                         kendalltau as st_kendalltau)

# ---- Load the dataset -------------------------------------------------------
#  Colab path first (as in the source notebook), with a plain-filesystem
#  fallback so the same file runs locally, in CI, or in a different notebook
#  service without edits.
CSV_PATH = os.environ.get("CSV_PATH", "/content/drive/MyDrive/your_dataset.csv")

try:
    from google.colab import drive
    drive.mount('/content/drive')
except (ImportError, ModuleNotFoundError):
    print("Not running in Colab — reading CSV_PATH from the local filesystem.")

try:
    cs = pd.read_csv(CSV_PATH)
    print(f"File loaded successfully: {CSV_PATH}")
    display(cs.head())
except FileNotFoundError:
    raise FileNotFoundError(
        f"No file at '{CSV_PATH}'. Set CSV_PATH (or the CSV_PATH environment "
        f"variable) to your dataset before running this file.")

print('First 5 rows:')
display(cs.head())

print('\nData types and non-null counts:')
cs.info()

print('\nMissing values per column:')
print(cs.isnull().sum())

# ---- `cs` -> X / Y -----------------------------------------------------------
#  TARGET_COL is the class label column. POSITIVE_LABEL names which of the two
#  original labels counts as "positive" for a binary problem — set it, don't
#  leave it to LabelEncoder's alphabetical order (see encode_target()).
TARGET_COL     = 'target'
POSITIVE_LABEL = None      # e.g. 'yes', 1, 'COLLECTION'; None = alphabetical
TEST_SIZE      = 0.3

if TARGET_COL not in cs.columns:
    raise KeyError(f"TARGET_COL={TARGET_COL!r} is not a column of the loaded file. "
                   f"Columns are: {list(cs.columns)}")

# A row with no label cannot be trained or scored on. Dropping is the only
# honest default; imputing a class label invents ground truth.
n_before = len(cs)
cs = cs.dropna(subset=[TARGET_COL]).reset_index(drop=True)
if len(cs) < n_before:
    print(f"Dropped {n_before - len(cs)} row(s) with a missing target.")

X_raw = cs.drop(TARGET_COL, axis=1)
Y_raw = cs[TARGET_COL]


# =============================================================================
#  AUTO-ENCODING THE INPUTS
# =============================================================================
#  Classification datasets routinely carry string/categorical FEATURES
#  (education level, region, device type) in a way that regression benchmark
#  datasets of physical measurements do not. Every model here ultimately sees a
#  float32 matrix, so those columns have to be encoded too.
#
#  WHY THIS IS SAFE TO DO BEFORE THE SPLIT, unlike scaling: one-hot and ordinal
#  encoding are per-column, unsupervised, and never consult the target or any
#  other row's value beyond the set of categories present. No held-out
#  information reaches the training rows. Scaling is different — it computes a
#  mean/std ACROSS rows — which is why it stays inside the Pipelines (see the
#  note at the end of this section).
#
#  The one real cost is that a category appearing only in the test split gets a
#  column of all-zeros in training. That is the same behaviour as
#  OneHotEncoder(handle_unknown='ignore') and is preferable to crashing.
# =============================================================================
ONEHOT_MAX_CARDINALITY = 15   # above this, one-hot would explode the matrix


def _is_text_like(s):
    """
    True for a column of labels rather than measurements.

    Tested on what the column IS, not on `dtype == object`: pandas 3 gives
    string columns a dedicated `str` dtype, so an `== object` check — the
    idiom most tutorials still use — silently classifies every text column as
    numeric and the whole frame blows up at the float cast further down.
    """
    return not (pd.api.types.is_numeric_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s))


def encode_features(X_df, verbose=True):
    """Mixed-dtype feature frame -> all-numeric feature frame."""
    X_df = X_df.copy()

    # Identifier-like and constant columns carry no signal and actively hurt:
    # a unique-per-row ID lets a tree memorise the training set.
    drop = []
    for c in X_df.columns:
        if X_df[c].nunique(dropna=False) <= 1:
            drop.append((c, "constant"))
        elif _is_text_like(X_df[c]) and X_df[c].nunique() == len(X_df):
            drop.append((c, "unique per row (identifier)"))
    if drop:
        if verbose:
            for c, why in drop:
                print(f"  dropping column {c!r}: {why}")
        X_df = X_df.drop(columns=[c for c, _ in drop])

    # Datetime columns become numeric calendar features rather than being
    # dropped or one-hot exploded: the day-of-week / month structure is usually
    # the part that predicts, and a raw timestamp is useless to a tree split.
    for c in list(X_df.columns):
        if pd.api.types.is_datetime64_any_dtype(X_df[c]) or (
                _is_text_like(X_df[c]) and _looks_like_dates(X_df[c])):
            dt = pd.to_datetime(X_df[c], errors="coerce")
            X_df[f"{c}_year"]      = dt.dt.year
            X_df[f"{c}_month"]     = dt.dt.month
            X_df[f"{c}_dayofweek"] = dt.dt.dayofweek
            X_df[f"{c}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
            X_df = X_df.drop(columns=[c])
            if verbose:
                print(f"  expanded datetime column {c!r} -> year/month/dayofweek/is_weekend")

    # Booleans are already a clean 0/1 encoding — cast rather than one-hot,
    # which would produce two perfectly complementary columns.
    for c in list(X_df.columns):
        if pd.api.types.is_bool_dtype(X_df[c]):
            X_df[c] = X_df[c].astype(np.int8)

    cat_cols = [c for c in X_df.columns if _is_text_like(X_df[c])]

    low_card  = [c for c in cat_cols if X_df[c].nunique() <= ONEHOT_MAX_CARDINALITY]
    high_card = [c for c in cat_cols if c not in low_card]

    if low_card:
        # drop_first=False on purpose. Dropping a level is needed only to avoid
        # perfect collinearity in an unregularised linear model; every model
        # here is either regularised or a tree, and keeping all levels makes
        # the SHAP/PDP output in Part 7 directly readable per category.
        X_df = pd.get_dummies(X_df, columns=low_card, drop_first=False, dtype=np.float64)
        if verbose:
            print(f"  one-hot encoded {len(low_card)} column(s): {low_card}")

    for c in high_card:
        # Ordinal for high-cardinality columns: one-hot would add hundreds of
        # near-empty columns. This DOES impose a fake ordering, which trees
        # tolerate (they can split the integer range into arbitrary groups) and
        # linear/distance models do not — so check anything that lands here.
        X_df[c] = LabelEncoder().fit_transform(X_df[c].astype(str))
        if verbose:
            print(f"  ordinal-encoded high-cardinality column {c!r} "
                  f"({X_df[c].nunique()} levels) — review if a linear/KNN model wins")

    # Median imputation for anything still missing. Median, not mean, because
    # it survives the skewed columns that are common in tabular data.
    n_na = int(X_df.isna().sum().sum())
    if n_na:
        X_df = X_df.fillna(X_df.median(numeric_only=True))
        if verbose:
            print(f"  median-imputed {n_na} missing feature value(s)")

    return X_df.astype(np.float64)


def _looks_like_dates(s, sample=50):
    """True if most of a sample of an object column parses as a date."""
    sample_vals = s.dropna().astype(str).head(sample)
    if len(sample_vals) == 0:
        return False
    parsed = pd.to_datetime(sample_vals, errors="coerce")
    return parsed.notna().mean() > 0.8


print("\nEncoding features:")
X = encode_features(X_raw)
print(f"Feature matrix: {X_raw.shape[1]} raw column(s) -> {X.shape[1]} numeric column(s)")

# ---- Encode the target ------------------------------------------------------
Y, LABEL_ENCODER = encode_target(Y_raw, positive_label=POSITIVE_LABEL)
Y = pd.Series(Y, name=TARGET_COL)

SCORING = resolve_scoring(N_CLASSES)
print(f"\nSearch objective: PRIMARY_METRIC={PRIMARY_METRIC} -> sklearn scoring={SCORING!r}")


def split_imbalance(X_df, y_ser, seed, test_size=None):
    """
    Per-column two-sample KS statistic between the train and test halves of the
    FEATURES, plus the class-proportion drift on the target. KS is
    distribution-free and catches shifts in location, spread AND shape, which a
    mean/std check alone would miss.

    Returns (worst_feature_KS, mean_feature_KS, class_share_drift).
    Lower is better; 0 would mean identical empirical distributions.
    """
    test_size = TEST_SIZE if test_size is None else test_size
    Xa, Xb, ya, yb = train_test_split(X_df, y_ser, test_size=test_size,
                                      random_state=seed, stratify=y_ser)
    ks = [ks_2samp(Xa[c], Xb[c]).statistic for c in X_df.columns]

    # Total variation distance between the two class-share vectors. ~0 whenever
    # stratify= is on; kept as a check that stratification actually happened.
    pa = np.bincount(ya, minlength=N_CLASSES) / len(ya)
    pb = np.bincount(yb, minlength=N_CLASSES) / len(yb)
    tvd = float(0.5 * np.abs(pa - pb).sum())
    return float(np.max(ks)), float(np.mean(ks)), tvd


SEED_SCAN_RANGE = range(int(os.environ.get("SEED_SCAN_N", 1000)))


def find_best_seed(X_df, y_ser, candidate_seeds=None,
                   test_size=None, top_n=5):
    """
    Minimax criterion: pick the seed whose WORST-matched feature column is best
    matched, tie-broken on the mean. Minimising the worst column (rather than
    the average) is the point — an average can stay low while one feature is
    badly split, which is exactly the 'biased split' case to avoid.

    Cost is len(candidate_seeds) x n_features KS tests. Trim SEED_SCAN_N on a
    wide dataset; the gain past a few hundred draws is small.
    """
    candidate_seeds = SEED_SCAN_RANGE if candidate_seeds is None else candidate_seeds
    test_size = TEST_SIZE if test_size is None else test_size
    rows = [(s, *split_imbalance(X_df, y_ser, s, test_size)) for s in candidate_seeds]
    scores = (pd.DataFrame(rows, columns=['seed', 'max_KS', 'mean_KS', 'class_drift'])
                .sort_values(['max_KS', 'mean_KS'])
                .reset_index(drop=True))
    print(f"\nScanned {len(rows)} seeds. Best {top_n} by worst-column KS:")
    print(scores.head(top_n).round(4).to_string(index=False))
    if (scores['seed'] == 42).any():
        ref = scores.loc[scores.seed == 42].iloc[0]
        print(f"\nFor reference, seed=42 -> max_KS={ref['max_KS']:.4f}, "
              f"mean_KS={ref['mean_KS']:.4f}, class_drift={ref['class_drift']:.4f}")
    print(f"Max class-share drift across all scanned seeds: {scores['class_drift'].max():.4f} "
          f"(~0 confirms stratification is doing its job)")
    return int(scores.iloc[0]['seed']), scores


BEST_SEED, seed_scan = find_best_seed(X, Y)
print(f"\n{'=' * 78}\nBEST_SEED = {BEST_SEED}   "
      f"(reuse this to reproduce the exact split)\n{'=' * 78}")

# ---- The split everything downstream uses ----------------------------------
#  stratify=Y is the one non-negotiable difference from the regression split.
#  Without it an unlucky draw can under-represent (or, on a small minority
#  class, entirely omit) a class from the test set, which makes the test
#  metrics meaningless for that class.
#
#  Kept as DataFrames on purpose. Column names are consumed by Part 7 (SHAP
#  feature_names, PDP) and Part 3 (the EDA table) — both check
#  hasattr(x_train, "columns") and silently fall back to Feature_0, Feature_1,
#  ... if they are lost.
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=BEST_SEED, stratify=Y)
x_train = pd.DataFrame(x_train, columns=X.columns)
x_test  = pd.DataFrame(x_test,  columns=X.columns)

print(f"\nTrain: {x_train.shape}   Test: {x_test.shape}")
print("Class balance —")
_balance = pd.DataFrame({
    "class":      [CLASSES[i] for i in range(N_CLASSES)],
    "train_n":    np.bincount(y_train, minlength=N_CLASSES),
    "train_share": (np.bincount(y_train, minlength=N_CLASSES) / len(y_train)).round(4),
    "test_n":     np.bincount(y_test, minlength=N_CLASSES),
    "test_share": (np.bincount(y_test, minlength=N_CLASSES) / len(y_test)).round(4),
})
print(_balance.to_string(index=False))
register_table("Class balance", _balance)
register_table("Target encoding",
               pd.DataFrame({"encoded": range(N_CLASSES), "original": CLASSES}))
register_table("Split seed scan", seed_scan.head(50))


# -----------------------------------------------------------------------------
#  ON PRE-SCALING THE FEATURES
# -----------------------------------------------------------------------------
#  The usual next cell is:
#
#      sc = MinMaxScaler()
#      x_train = sc.fit_transform(x_train)
#      x_test  = sc.transform(x_test)
#
#  That is deliberately NOT run here, because in this pipeline it breaks three
#  things at once:
#
#   1. LEAKAGE. It fits the scaler on the whole training set, then every
#      section below cross-validates inside that same training set — so every
#      validation fold's min and max have already leaked into the transform,
#      and the CV scores that drive all the hyperparameter tuning come out
#      optimistic. Per-fold refitting is the whole point of the Pipelines.
#   2. DOUBLE SCALING. The KNN and SVC Pipelines and the Keras helpers already
#      scale internally, so they would be scaling data that is already scaled.
#   3. LOST COLUMN NAMES. fit_transform returns a bare numpy array, so
#      x_train.columns disappears and SHAP/PDP (Part 7) and the EDA table
#      (Part 3) silently fall back to Feature_0, Feature_1, ...
#
#  To use min-max scaling instead of standardization, set the knob in
#  Section 0 — one line, applies to every model, and stays leakage-free
#  because each scaler is still refit inside each fold:
#
#      SCALER_CLS = MinMaxScaler
#
#  Tree models (rf, xgboost_model, lgbm, ada) are scale-invariant and are
#  unaffected either way.
# -----------------------------------------------------------------------------

X_tr = np.asarray(x_train, dtype=np.float32)
X_te = np.asarray(x_test,  dtype=np.float32)
# Labels stay INT64, not float32. Class indices are not measurements: float
# labels raise in XGBoost's classifier and silently break np.bincount.
y_tr = np.asarray(y_train, dtype=np.int64).ravel()
y_te = np.asarray(y_test,  dtype=np.int64).ravel()

N_FEATURES = X_tr.shape[1]


# =============================================================================
#  SECTION B — CLASS REBALANCING (SMOTE AND FRIENDS)
# =============================================================================
#  SMOTE (Synthetic Minority Over-sampling TEchnique, Chawla et al. 2002) grows
#  the minority class by interpolating between a minority point and one of its
#  k nearest minority neighbours, rather than duplicating rows the way naive
#  random oversampling does.
#
#  ---------------------------------------------------------------------------
#  THE ONE RULE: RESAMPLE INSIDE THE FOLD, NEVER BEFORE THE SPLIT
#  ---------------------------------------------------------------------------
#  This is the mistake that makes published SMOTE results irreproducible, and
#  it is worth being precise about WHY, because the leakage is not obvious:
#
#      X_res, y_res = SMOTE().fit_resample(X, y)      # <-- WRONG
#      cross_val_score(model, X_res, y_res, cv=5)
#
#  A synthetic point is a blend of a real minority row and one of its
#  neighbours. Resample first, and a synthetic row built partly from row 37 can
#  land in the training folds while row 37 itself lands in the validation fold.
#  The model has then effectively seen the answer, and every CV score rises —
#  often dramatically, and entirely fictitiously. Worse, duplicated-ish points
#  land on both sides of the split, so the validation fold is no longer
#  independent at all. The held-out test score does not move, and the gap gets
#  blamed on "overfitting" rather than on the leak that caused it.
#
#  So the sampler is a PIPELINE STEP here, never a preprocessing call. An
#  imblearn Pipeline applies its sampler during fit() and skips it during
#  predict()/predict_proba(), which is exactly the required behaviour:
#    * each CV fold resamples only its own training part;
#    * the validation fold keeps its real class balance;
#    * the held-out test set is never resampled, so every number in Part 6 is
#      measured at the prevalence the model will actually meet.
#  sklearn's own Pipeline cannot do this — it would try to apply the sampler at
#  predict time — which is why imblearn.pipeline.Pipeline is imported below.
#
#  ---------------------------------------------------------------------------
#  WHAT TO EXPECT, HONESTLY
#  ---------------------------------------------------------------------------
#  SMOTE is not a free accuracy upgrade, and this pipeline is instrumented well
#  enough to show you exactly what it does and does not buy:
#
#    * ROC-AUC usually barely moves. AUC measures RANKING and is insensitive to
#      class prevalence, and resampling does not give the model new information
#      — it re-weights what is already there. If SMOTE "fixes" your AUC, be
#      suspicious of a leak.
#    * RECALL on the minority class usually rises and PRECISION usually falls,
#      because the decision boundary moves toward the majority class. Watch F1,
#      MCC and the confusion matrices (Section 8), not accuracy.
#    * CALIBRATION GETS WORSE. Resampling changes the effective class prior, so
#      the model's probabilities no longer estimate P(class | x) on the real
#      population — they are systematically too high for the minority class.
#      Expect LogLoss, Brier and ECE to degrade in Section 8D. That is not a
#      bug, it is the trade.
#
#  Which means: if what you actually want is to stop missing the minority
#  class, MOVING THE DECISION THRESHOLD (Section 8E) does the same job, costs
#  nothing, adds no synthetic data, and leaves the probabilities calibrated.
#  Reach for SMOTE when the minority class is so small that the model cannot
#  learn its SHAPE at all, not merely when the counts look lopsided. Running
#  both settings and comparing the Part 6 tables is the way to decide, and this
#  file is set up to make that a one-line change.
#
#  ---------------------------------------------------------------------------
#  SMOTE AND class_weight ARE BOTH IMBALANCE CORRECTIONS
#  ---------------------------------------------------------------------------
#  Every model here already carries class_weight in its search space. Applying
#  'balanced' weights ON TOP of a balanced resample corrects the same imbalance
#  twice and pushes the boundary past the minority class. Both are left in the
#  space deliberately — the search can select class_weight=None once resampling
#  is on, and letting it choose against the CV objective is more defensible
#  than asserting either. But if you are comparing runs, know that this is a
#  2x2 of corrections, not a single switch.
# =============================================================================
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, SMOTENC
from imblearn.combine import SMOTETomek, SMOTEENN

#  THE KNOB. None disables resampling entirely (the sampler step becomes a
#  no-op, so nothing else in the file changes shape).
#    "smote"       vanilla SMOTE — interpolates between minority neighbours.
#    "borderline"  BorderlineSMOTE — synthesises only near the decision
#                  boundary, where the classifier is actually confused. Usually
#                  the better default when the minority class is not compact.
#    "svm"         SVMSMOTE — uses an SVM's support vectors to pick where to
#                  synthesise. Slower; good when the boundary is curved.
#    "adasyn"      ADASYN — allocates more synthetic points to minority rows
#                  that are HARD (many majority neighbours). Sharpens focus on
#                  the difficult region; also amplifies label noise there.
#    "smotenc"     SMOTE-NC — treats the flagged columns as categorical and
#                  copies the majority category of the neighbours instead of
#                  interpolating. See the one-hot note below.
#    "smote_tomek" / "smote_enn"
#                  oversample, then CLEAN: remove the borderline majority rows
#                  (Tomek links) or the misclassified ones (edited nearest
#                  neighbours). Useful when SMOTE alone leaves the classes
#                  smeared into each other. These shrink the majority class too,
#                  so they change both sides of the balance.
RESAMPLING = None
SMOTE_K    = 5          # requested k_neighbors; clamped below to what the folds allow


#  ONE-HOT COLUMNS AND SYNTHETIC ROWS.
#  Section A one-hot encoded the categorical features, so the matrix contains
#  0/1 indicator columns. Plain SMOTE interpolates them like any other number
#  and produces values such as region_north = 0.37 — a row that is 37% in a
#  category. Trees tolerate that (they just split it somewhere), but it is
#  meaningless for the distance and linear models, and it quietly changes what
#  the SHAP and PDP plots in Part 7 are describing.
#  SMOTENC is the fix that works post-encoding: it assigns each flagged column
#  the most common value among the neighbours, so the dummies stay 0/1. It is
#  still imperfect — the dummies of one original variable are decided
#  independently, so a synthetic row can end up in two categories or none. The
#  fully correct approach is to run SMOTENC on the data BEFORE one-hot
#  encoding; this is the practical approximation given where the encoding sits.
BINARY_FEATURE_IDX = [j for j in range(N_FEATURES)
                      if np.isin(np.unique(X_tr[:, j]), (0.0, 1.0)).all()]
if BINARY_FEATURE_IDX and RESAMPLING in ("smote", "borderline", "svm", "adasyn"):
    print(f"\nNote: {len(BINARY_FEATURE_IDX)} of {N_FEATURES} feature(s) are 0/1 indicators. "
          f"{RESAMPLING!r} will interpolate them into fractions; RESAMPLING='smotenc' "
          f"keeps them binary.")


def resolve_smote_k(y, n_splits=5, requested=None):
    """
    Largest usable k_neighbors.

    SMOTE needs k minority NEIGHBOURS, so it requires k < (minority count) in
    whatever data it is fitted on — and since it is fitted per fold, the
    binding constraint is the minority count in a TRAINING FOLD, not in the
    whole training set. Getting this wrong surfaces as
    "Expected n_neighbors <= n_samples" from deep inside a CV loop, hours in.
    """
    requested = SMOTE_K if requested is None else requested
    smallest = int(np.bincount(y, minlength=N_CLASSES).min())
    in_fold  = int(np.floor(smallest * (n_splits - 1) / n_splits))
    return max(1, min(requested, in_fold - 1)), smallest, in_fold


def make_sampler():
    """A FRESH sampler instance, or None when resampling is off."""
    if RESAMPLING is None:
        return None
    k, smallest, in_fold = resolve_smote_k(y_tr)
    if in_fold < 2:
        warnings.warn(
            f"rarest class has ~{in_fold} row(s) per training fold — too few to "
            f"synthesise from. Resampling disabled; collect more data for that "
            f"class, merge it, or rely on class_weight and threshold moving.",
            stacklevel=2)
        return None
    common = dict(random_state=SEED)
    if RESAMPLING == "smote":        return SMOTE(k_neighbors=k, **common)
    if RESAMPLING == "borderline":   return BorderlineSMOTE(k_neighbors=k, **common)
    if RESAMPLING == "svm":          return SVMSMOTE(k_neighbors=k, **common)
    if RESAMPLING == "adasyn":       return ADASYN(n_neighbors=k, **common)
    if RESAMPLING == "smotenc":
        if not BINARY_FEATURE_IDX:
            warnings.warn("RESAMPLING='smotenc' but no 0/1 columns were found; "
                          "falling back to plain SMOTE.", stacklevel=2)
            return SMOTE(k_neighbors=k, **common)
        return SMOTENC(categorical_features=BINARY_FEATURE_IDX, k_neighbors=k, **common)
    if RESAMPLING == "smote_tomek":  return SMOTETomek(smote=SMOTE(k_neighbors=k, **common), **common)
    if RESAMPLING == "smote_enn":    return SMOTEENN(smote=SMOTE(k_neighbors=k, **common), **common)
    raise ValueError(f"unknown RESAMPLING={RESAMPLING!r}")


def build_pipeline(steps):
    """
    Assemble a model pipeline with the sampler inserted immediately before the
    final estimator.

    `steps` is [... , ("model", estimator)] WITHOUT the sampler. The sampler
    step always exists — as "passthrough" when resampling is off — so every
    hyperparameter grid in Part 4 keeps one set of "model__" prefixes whether
    or not resampling is enabled.

    Position matters: the sampler sits AFTER any scaler, because SMOTE picks
    neighbours by Euclidean distance and unscaled features would let the
    largest-magnitude column decide who counts as a neighbour.
    """
    sampler = make_sampler() or "passthrough"
    return ImbPipeline(list(steps[:-1]) + [("sampler", sampler)] + list(steps[-1:]))


def resample_fit(X, y):
    """
    Resample a training split directly.

    Used only by the models that cannot go through a Pipeline: XGBoost and
    LightGBM early-stop against an eval_set (which must keep its real class
    balance, so it must not be resampled), and the Keras nets scale and fit
    fold by fold by hand. Call it on the TRAINING part of a split only.
    """
    sampler = make_sampler()
    if sampler is None:
        return X, y
    Xr, yr = sampler.fit_resample(X, y)
    return np.asarray(Xr, dtype=np.float32), np.asarray(yr, dtype=np.int64).ravel()


# ── WHAT THE RESAMPLING ACTUALLY DOES ────────────────────────────────────────
#  Illustration only. The real resampling happens inside each CV fold, on that
#  fold's training part; this applies it once to the whole training set purely
#  to report the shape of the change. Nothing below is fitted on it.
if RESAMPLING is not None:
    _k, _smallest, _in_fold = resolve_smote_k(y_tr)
    print("\n" + "=" * 78)
    print(f"CLASS REBALANCING — {RESAMPLING}")
    print("=" * 78)
    print(f"rarest class: {_smallest} training rows (~{_in_fold} per training fold) "
          f"-> k_neighbors={_k}")
    _Xr, _yr = resample_fit(X_tr, y_tr)
    _before = np.bincount(y_tr, minlength=N_CLASSES)
    _after  = np.bincount(_yr, minlength=N_CLASSES)
    print(pd.DataFrame({"class": CLASSES, "before": _before, "after": _after,
                        "change": _after - _before}).to_string(index=False))
    print(f"training rows: {len(y_tr)} -> {len(_yr)}")
    print("(illustration only — fitting resamples per fold, never on the test set)")
    del _Xr, _yr


# =============================================================================
#  SECTION 1 — UNIFORM MODEL INTERFACE
# =============================================================================
#  Every model below — sklearn estimator, Optuna-tuned booster, or Keras net —
#  gets wrapped in a ModelSpec so they can all be tuned, fit, and evaluated the
#  same way, regardless of what's happening internally:
#
#      spec.fit(X_tr, y_tr)          # runs hyperparameter search + final refit
#      spec.predict_proba(X_te)      # raw features in, (n, K) probabilities out
#      spec.predict(X_te)            # argmax of the above -> class indices
#      spec.evaluate(X_te, y_te)     # predict + score + store in RESULTS
#
#  REGRESSION -> CLASSIFICATION: PROBABILITIES ARE THE PRIMITIVE.
#  In the regression file the primitive was predict(); here it is
#  predict_proba(), and labels are derived from it by argmax. That ordering is
#  deliberate:
#    * ROC curves, PR curves, calibration, log-loss, Brier, soft voting,
#      stacking meta-features and SHAP-on-probability ALL need probabilities.
#      A pipeline built on hard labels cannot produce any of them.
#    * it guarantees labels and probabilities agree. sklearn's SVC is the
#      cautionary case: its predict() uses the decision function while its
#      predict_proba() uses a separately-fitted Platt calibration, so the two
#      can and do disagree on points near the boundary.
#    * the decision threshold becomes an explicit, tunable choice (Part 6)
#      rather than a hardcoded 0.5 buried inside each model.
#
#  Tuning and refitting are DELIBERATELY split into two functions:
#      search_fn(X, y)         -> (best_params, cv_score)
#      refit_fn(X, y, params)  -> (fitted_model_or_None, proba_fn)
#  `fit()` just calls them in sequence. The split matters for Part 5's K-fold
#  stability sweep: refitting a fresh model with hyperparameters that are
#  already known is cheap, whereas re-running Optuna/GridSearchCV at every one
#  of 9 fold counts for 9 models would mean thousands of extra searches.
#  spec.refit_on(X, y) reuses refit_fn directly, skipping the search.
# =============================================================================
class ModelSpec:
    def __init__(self, name, search_fn, refit_fn, search_factory=None):
        self.name = name
        self._search_fn = search_fn     # (X, y) -> (best_params, cv_score)
        self._refit_fn = refit_fn       # (X, y, params) -> (model, proba_fn)
        self.search_factory = search_factory   # optional: enables .nested_cv()
        self.model = None
        self.proba_fn = None
        self.best_params = None
        self.cv_score = None
        self.fit_time_s = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).ravel()
        t0 = time.time()
        self.best_params, self.cv_score = self._search_fn(X, y)
        self.model, self.proba_fn = self._refit_fn(X, y, self.best_params)
        self.fit_time_s = time.time() - t0
        return self

    def predict_proba(self, X):
        if self.proba_fn is None:
            raise RuntimeError(f"{self.name}: call .fit(X_tr, y_tr) before .predict_proba().")
        P = np.asarray(self.proba_fn(np.asarray(X, dtype=np.float32)), dtype=np.float64)
        if P.ndim == 1:                       # a binary head returning P(class 1)
            P = np.column_stack([1.0 - P, P])
        return P

    def predict(self, X, threshold=None):
        return labels_from_proba(self.predict_proba(X), threshold=threshold)

    def evaluate(self, X_te, y_te):
        proba = self.predict_proba(X_te)
        pred = labels_from_proba(proba)
        return report(self.name, y_te, pred, proba, self.best_params,
                      self.cv_score, self.fit_time_s)

    def refit_on(self, X, y, params=None):
        """Refit a FRESH model using already-known hyperparameters (skips the
        search entirely). Returns a proba_fn. Used by the Part 5 K-fold sweep,
        the ensembles, learning curves and the bias-variance bootstrap."""
        params = params or self.best_params
        if params is None:
            raise RuntimeError(f"{self.name}: no tuned hyperparameters yet — call .fit() first.")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).ravel()
        _, proba_fn = self._refit_fn(X, y, params)

        def wrapped(Xnew):
            P = np.asarray(proba_fn(np.asarray(Xnew, dtype=np.float32)), dtype=np.float64)
            return np.column_stack([1.0 - P, P]) if P.ndim == 1 else P
        return wrapped

    def nested_cv(self, X, y, cv=None):
        """Only available for models tuned via Grid/RandomizedSearchCV (rf, ada,
        knn) — re-running an Optuna study per outer fold is expensive, so it's
        opt-in rather than wired up by default for the boosted/NN models."""
        if self.search_factory is None:
            print(f"{self.name}: nested CV not wired up for this model "
                  f"(Optuna-tuned — rerun manually with nested_cv_score if needed).")
            return None
        return nested_cv_score(self.search_factory, np.asarray(X, dtype=np.float32),
                               np.asarray(y, dtype=np.int64).ravel(), cv=cv, label=self.name)

    def __repr__(self):
        return f"ModelSpec({self.name!r}, fitted={self.model is not None})"


def fit_all(models, X, y):
    for spec in models:
        print(f"\n>>> Tuning + fitting: {spec.name}")
        spec.fit(X, y)


def evaluate_all(models, X_te, y_te):
    for spec in models:
        spec.evaluate(X_te, y_te)
    return summary_table()


def summary_table():
    cols = [f"CV_{PRIMARY_METRIC}", "ROC_AUC", "PR_AUC", "Accuracy", "Balanced_Acc",
            "F1", "MCC", "LogLoss", "Brier", "fit_time_s"]
    df = (pd.DataFrame(RESULTS).T[cols]
            .astype(float)
            .sort_values("ROC_AUC", ascending=False))   # higher is better now
    return df


# =============================================================================
# =============================================================================
#  PART 2 — SEARCH STRATEGIES
# =============================================================================
# =============================================================================

# =============================================================================
#  SECTION 2 — SEARCH-STRATEGY CHOICES (justification)
# =============================================================================
# GridSearchCV       -> small, low-dimensional, mostly *discrete* spaces where
#                       exhaustive enumeration is cheap and guarantees the
#                       optimum of the grid. Used for: KNN, AdaBoost.
# RandomizedSearchCV -> moderate/large space, model is robust and its response
#                       surface is dominated by a few parameters. Bergstra &
#                       Bengio (2012): random sampling beats grid at equal
#                       budget once >2-3 dims matter. Used for: Random Forest.
# Optuna (TPE)       -> Bayesian / sequential model-based search. Best when the
#                       space is high-dimensional, mixes continuous log-scaled
#                       params with integers/categoricals, and each evaluation
#                       is expensive. TPE reuses past trials, MedianPruner
#                       kills bad trials mid-CV. Used for: XGBoost, LightGBM,
#                       SVC, MLP, CNN-LSTM, Sequential Logistic.
#
# CLASSIFICATION-SPECIFIC ADDITION: class_weight.
#   Every model that supports it carries class_weight in its search space
#   (None vs 'balanced', and 'balanced_subsample' for RF; scale_pos_weight for
#   XGBoost; a computed class_weight dict for the Keras nets). It is treated as
#   a hyperparameter rather than switched on by default because re-weighting is
#   a genuine trade: it buys minority-class recall at the cost of precision and
#   of probability calibration. Letting the search decide, against a metric
#   that is itself imbalance-aware, is more defensible than asserting it.
# =============================================================================


# =============================================================================
# =============================================================================
#  PART 3 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
# =============================================================================

# =============================================================================
#  SECTION 10 — EXPLORATORY DATA ANALYSIS
# =============================================================================
#  Pure exploratory/descriptive analysis of the dataset itself — nothing here
#  depends on any model being tuned or fitted, so this only needs Part 1 to
#  have run. It sits before any modeling because that is the natural place for
#  exploratory analysis: look at the data before building anything on top.
#
#  REGRESSION -> CLASSIFICATION. The regression EDA plotted each feature
#  against the target as a scatter with a fitted line and a Pearson r. None of
#  that survives a categorical target: there is no line to fit, and r is
#  undefined for unordered classes. The replacements answer the same underlying
#  question ("does this feature carry signal about the target?") in the form a
#  class label admits:
#
#      scatter(feature, target) + r   ->  per-class DISTRIBUTIONS of the
#                                         feature (KDE overlay + grouped box
#                                         plots). Separated distributions =
#                                         a feature the classifier can use.
#      correlation with the target    ->  ANOVA F statistic and MUTUAL
#                                         INFORMATION, which are defined for a
#                                         categorical target (and MI catches
#                                         non-monotonic relationships that F
#                                         misses).
#      (new) class balance bar chart  ->  the first thing to check in any
#                                         classification problem; it decides
#                                         which metrics are trustworthy.
# =============================================================================
import seaborn as sns
from sklearn.feature_selection import f_classif, mutual_info_classif
from matplotlib.colors import LinearSegmentedColormap

target_name = TARGET_COL

if hasattr(x_train, "columns"):
    X_full = pd.concat([x_train, x_test], axis=0, ignore_index=True)
else:
    fallback_names = [f"Feature_{i}" for i in range(N_FEATURES)]
    X_full = pd.DataFrame(np.vstack([X_tr, X_te]), columns=fallback_names)

df = X_full.copy()
y_full_enc = np.concatenate([y_tr, y_te])
# The DECODED labels go in the EDA frame so legends and axis ticks read
# 'PAIDOFF' / 'COLLECTION' rather than 0 / 1.
df[target_name] = decode_target(y_full_enc)

FEATURE_COLS = [c for c in df.columns if c != target_name]

# ── COLOR SYSTEM ─────────────────────────────────────────────────────────────
#  One validated categorical order, used for classes in Part 3 and for models
#  in Part 6. Hues are assigned in fixed slot order and never cycled or
#  generated: past 8 series the code folds to a capped subset instead (see
#  ROC_MAX_MODELS in Part 6), because a 9th generated hue is not separable.
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                 '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK           = '#0b0b0b'    # primary text
INK_SOFT      = '#52514e'    # secondary text / annotations
GRID_COLOR    = '#d8d7d2'

#  Sequential = ONE hue, light -> dark. Used for confusion matrices and any
#  magnitude heatmap. Never a rainbow: 'viridis'/'jet' encode magnitude as hue,
#  which readers cannot order without consulting the colorbar.
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", [
    '#f4f8fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
    '#256abf', '#184f95', '#0d366b'])

#  Diverging = two hues + a NEUTRAL GRAY midpoint, for correlations where the
#  sign matters and 0 must read as "nothing".
DIVERGING = LinearSegmentedColormap.from_list("div_br", [
    '#104281', '#2a78d6', '#9ec5f4', '#f0efec', '#f5b3b2', '#e34948', '#8f2322'])

CLASS_COLORS = {CLASSES[i]: SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(N_CLASSES)}
PER_CLASS_EDA = N_CLASSES <= len(SERIES_COLORS)
if not PER_CLASS_EDA:
    print(f"\nNote: {N_CLASSES} classes exceeds the 8-hue categorical palette; "
          f"per-class overlays are skipped in favour of pooled distributions. "
          f"Facet by class manually if you need them.")


# ── SUMMARY STATISTICS TABLE ──────────────────────────────────────────────────
def summary_stats(a):
    """
    count / mean / std / min / Q1 / median / Q3 / max (from describe()), plus
    mode and skewness per column.

    Names each quantile exactly once: Q1 / median / Q3. (The 50th percentile IS
    Q2 by convention, so labelling the 75th percentile 'Q2' — as the reference
    version did — leaves the third quartile mislabelled.)
    """
    desc_stats = a.describe().T
    modes = a.mode().iloc[0]
    skewness = a.skew()
    stats = pd.concat([desc_stats, modes, skewness], axis=1)
    stats.columns = ['count', 'mean', 'std', 'min', 'Q1', 'median', 'Q3', 'max',
                     'mode', 'skewness']
    return stats


print("\n" + "=" * 78)
print("SUMMARY STATISTICS (features)")
print("=" * 78)
summary_df = summary_stats(df[FEATURE_COLS])
print(summary_df.round(4).to_string())
register_table("EDA summary stats", summary_df, index=True)

# Per-class summary: a feature whose mean differs sharply across classes is one
# the models will lean on. This is the table form of the plots below.
print("\n" + "=" * 78)
print("PER-CLASS FEATURE MEANS")
print("=" * 78)
_per_class_means = df.groupby(target_name)[FEATURE_COLS].mean().T
print(_per_class_means.round(4).to_string())
register_table("Per-class feature means", _per_class_means, index=True)


def _eda_grid(n, ncols=2, row_height=4.3):
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, row_height * nrows))
    axs = np.array(axs).reshape(nrows, ncols)
    return fig, axs, nrows, ncols


def _hide_unused_axes(fig, axs, n, nrows, ncols):
    for j in range(n, nrows * ncols):
        fig.delaxes(axs[j // ncols, j % ncols])


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=12, fontweight='bold', color=INK)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=10, color=INK_SOFT)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10, color=INK_SOFT)
    ax.tick_params(labelsize=9, colors=INK_SOFT)
    ax.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


# ── CLASS BALANCE ─────────────────────────────────────────────────────────────
#  The first chart to look at. It decides which metrics are trustworthy for the
#  rest of the file: the more skewed this is, the less accuracy means.
def plot_class_balance(df, target):
    counts = df[target].value_counts().reindex(CLASSES)
    shares = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * N_CLASSES), 4.5))
    bars = ax.bar([str(c) for c in counts.index], counts.values,
                  color=[CLASS_COLORS[c] for c in counts.index], width=0.62)
    # Direct labels on every bar: <= 8 categories, so no legend is needed and
    # the reader never has to map a color back to a key.
    for bar, n, share in zip(bars, counts.values, shares.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{n:,}\n{share:.1%}", ha='center', va='bottom',
                fontsize=10, color=INK_SOFT)

    _style(ax, f"Class balance — {target}", target, "Rows")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.grid(axis='x', visible=False)

    ratio = shares.max() / shares.min()
    ax.annotate(f"majority:minority = {ratio:.1f}:1",
                xy=(0.99, 0.96), xycoords='axes fraction', ha='right', va='top',
                fontsize=10, color=INK_SOFT)

    plt.tight_layout()
    save_fig(plt.gcf(), "eda_class_balance", subdir="eda")
    plt.show()
    plt.close(fig)


# ── PER-CLASS DISTRIBUTIONS (KDE overlay) ────────────────────────────────────
#  The replacement for the regression file's histograms. A feature whose
#  per-class curves sit on top of each other carries no marginal signal; one
#  whose curves separate is directly usable by the classifier. (Overlap here
#  does not prove uselessness — the feature may still matter in combination
#  with another, which is what the 2-way PDPs in Part 7 show.)
def _draw_class_density(ax, df, target, column):
    """One feature's class-conditional density. Shared by the grid and the
    stand-alone per-feature figure so the two can never drift apart."""
    if PER_CLASS_EDA:
        for cls in CLASSES:
            vals = df.loc[df[target] == cls, column]
            if vals.nunique() > 1:
                sns.kdeplot(vals, ax=ax, color=CLASS_COLORS[cls], linewidth=2,
                            fill=True, alpha=0.18, label=str(cls))
            else:
                ax.axvline(vals.iloc[0], color=CLASS_COLORS[cls],
                           linewidth=2, label=str(cls))
    else:
        sns.kdeplot(df[column], ax=ax, color=SERIES_COLORS[0], linewidth=2,
                    fill=True, alpha=0.18, label="all rows")
    _style(ax, f"{column} by class", column, "Density")
    ax.legend(title=None, fontsize=8, frameon=False)


def plot_class_conditional_densities(df, target, ncols=2):
    columns = FEATURE_COLS
    n = len(columns)
    fig, axs, nrows, ncols = _eda_grid(n, ncols)
    for i, column in enumerate(columns):
        _draw_class_density(axs[i // ncols, i % ncols], df, target, column)
    _hide_unused_axes(fig, axs, n, nrows, ncols)
    plt.tight_layout()
    save_fig(plt.gcf(), "eda_class_conditional_densities", subdir="eda")
    plt.show()
    plt.close(fig)

    if PER_ITEM_FIGURES:
        for column in columns:
            f1, ax1 = plt.subplots(figsize=(6.4, 4.4))
            _draw_class_density(ax1, df, target, column)
            f1.tight_layout()
            save_fig(f1, f"density_{column}", subdir="eda/per_feature", close=True)


# ── GROUPED BOX PLOTS (feature vs class) ─────────────────────────────────────
#  The direct structural replacement for scatter(feature, target): same axes
#  roles (feature on y, target on x), with the target's categorical nature
#  respected. Non-overlapping boxes = a separable feature.
def _draw_class_boxplot(ax, df, target, column):
    """One feature's distribution grouped by class, plus its outlier count."""
    if PER_CLASS_EDA:
        sns.boxplot(data=df, x=target, y=column, ax=ax, hue=target,
                    palette=CLASS_COLORS, legend=False, width=0.55,
                    linecolor=INK, linewidth=1.0, fliersize=3)
    else:
        sns.boxplot(data=df, y=column, ax=ax, color=SERIES_COLORS[0],
                    width=0.4, linecolor=INK, linewidth=1.0, fliersize=3)
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = int(((df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)).sum())
    _style(ax, f"{column} by class  ({n_out} outliers overall)", target, column)
    return n_out


def plot_boxplots_by_class(df, target, ncols=2):
    columns = FEATURE_COLS
    n = len(columns)
    fig, axs, nrows, ncols = _eda_grid(n, ncols, row_height=4.0)
    outliers = {}
    for i, column in enumerate(columns):
        outliers[column] = _draw_class_boxplot(axs[i // ncols, i % ncols], df, target, column)
    _hide_unused_axes(fig, axs, n, nrows, ncols)
    plt.tight_layout()
    save_fig(plt.gcf(), "eda_boxplots_by_class", subdir="eda")
    plt.show()
    plt.close(fig)

    if PER_ITEM_FIGURES:
        for column in columns:
            f1, ax1 = plt.subplots(figsize=(6.0, 4.4))
            _draw_class_boxplot(ax1, df, target, column)
            f1.tight_layout()
            save_fig(f1, f"boxplot_{column}", subdir="eda/per_feature", close=True)

    register_table("EDA outlier counts",
                   pd.DataFrame({"feature": list(outliers), "outliers": list(outliers.values())}))


# ── CORRELATION HEATMAPS (features only) ─────────────────────────────────────
#  Three methods because they answer different questions: Pearson measures
#  LINEAR association, Spearman and Kendall measure MONOTONIC association and
#  are robust to outliers and non-linear-but-ordered relationships.
#
#  The encoded target is deliberately EXCLUDED. Correlating a feature with a
#  class index is only meaningful for a binary target (where Pearson r is the
#  point-biserial correlation); for 3+ classes the number depends entirely on
#  the arbitrary order LabelEncoder assigned, and reporting it is a real
#  mistake, not a cosmetic one. Feature-target association gets its own,
#  correctly-defined chart below.
_CORR_TESTS = {"pearson": st_pearsonr, "spearman": st_spearmanr, "kendall": st_kendalltau}

#  Significance thresholds, annotated on every correlation cell.
#  NOTE THE MAPPING: one star marks p < 0.01 and two stars p < 0.05, as
#  specified. That is the reverse of the common journal convention (where more
#  stars means a smaller p), so the legend is printed on every figure and the
#  exact p-values are exported alongside — a reader should never have to guess
#  which way round it is. Swap the two constants to use the usual convention.
SIG_ONE_STAR = 0.01     # p < 0.01  ->  *
SIG_TWO_STAR = 0.05     # p < 0.05  ->  **


def correlation_with_pvalues(data, method):
    """
    Correlation matrix AND the matrix of two-sided p-values for it.

    pandas .corr() gives coefficients only, so the tests are run pairwise here.
    A pair that cannot be tested — a constant column has no variance, so the
    correlation is undefined rather than zero — yields NaN and is left
    unannotated instead of being reported as a non-significant result.
    """
    cols = list(data.columns)
    r = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols, dtype=float)
    pv = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols, dtype=float)
    test = _CORR_TESTS[method]
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j >= i:
                continue
            try:
                stat, pval = test(data[a], data[b])
            except Exception:
                stat, pval = np.nan, np.nan
            r.loc[a, b] = r.loc[b, a] = stat
            pv.loc[a, b] = pv.loc[b, a] = pval
    return r, pv


def significance_stars(pval):
    """'' / '*' / '**' under the thresholds above."""
    if pval is None or not np.isfinite(pval):
        return ""
    if pval < SIG_ONE_STAR:
        return "*"
    if pval < SIG_TWO_STAR:
        return "**"
    return ""


def plot_correlation_heatmaps(df, methods=('pearson', 'spearman', 'kendall')):
    """
    Lower-triangle correlation heatmap per method, annotated with the
    coefficient and its significance stars.

    THE DIAGONAL IS REMOVED. Every variable correlates perfectly with itself,
    so the diagonal is a row of 1.00 that carries no information, anchors the
    colour scale at its extreme, and draws the eye away from the off-diagonal
    cells that are the point of the plot.

    Three methods because they answer different questions: Pearson measures
    LINEAR association, Spearman and Kendall measure MONOTONIC association and
    are robust to outliers and non-linear-but-ordered relationships.

    The encoded target is deliberately EXCLUDED. Correlating a feature with a
    class index is only meaningful for a binary target (where Pearson r is the
    point-biserial correlation); for 3+ classes the number depends entirely on
    the arbitrary order LabelEncoder assigned, and reporting it is a real
    mistake, not a cosmetic one. Feature-target association gets its own,
    correctly-defined chart below.
    """
    for method in methods:
        corr, pvals = correlation_with_pvalues(df[FEATURE_COLS], method)

        labels = corr.copy().astype(object)
        for a in FEATURE_COLS:
            for b in FEATURE_COLS:
                labels.loc[a, b] = ("" if not np.isfinite(corr.loc[a, b])
                                    else f"{corr.loc[a, b]:.2f}{significance_stars(pvals.loc[a, b])}")

        #  k=0 masks the diagonal as well as the upper triangle; k=1 would keep
        #  the diagonal, which is the default and is what we do not want here.
        mask = np.triu(np.ones_like(corr, dtype=bool), k=0)

        fig, ax = plt.subplots(figsize=(9.5, 7.5))
        sns.heatmap(corr, mask=mask, annot=labels if len(FEATURE_COLS) <= 15 else False,
                    cmap=DIVERGING, vmin=-1, vmax=1, center=0, square=True,
                    linewidths=1, linecolor='white', cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 8.5}, fmt='', cbar=True, ax=ax)
        ax.set_title(f"{method.capitalize()} correlation — features",
                     fontsize=13, fontweight='bold', color=INK)
        ax.tick_params(labelsize=9, colors=INK_SOFT)
        fig.text(0.5, 0.005,
                 f"*  p < {SIG_ONE_STAR}      **  p < {SIG_TWO_STAR}      "
                 f"unmarked: not significant at {SIG_TWO_STAR}      diagonal omitted",
                 ha='center', fontsize=9, color=INK_SOFT)
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        save_fig(plt.gcf(), f"eda_{method}_correlation_heatmap", subdir="eda")
        plt.show()
        plt.close(fig)

        #  Exported long-form: the heatmap shows the stars, the table carries
        #  the exact p-values a reviewer will ask for.
        rows = []
        for i, a in enumerate(FEATURE_COLS):
            for j, b in enumerate(FEATURE_COLS):
                if j < i:
                    rows.append({"feature_a": a, "feature_b": b,
                                 "coefficient": corr.loc[a, b], "p_value": pvals.loc[a, b],
                                 "stars": significance_stars(pvals.loc[a, b])})
        register_table(f"Corr {method}", pd.DataFrame(rows))


# ── FEATURE -> TARGET ASSOCIATION ────────────────────────────────────────────
#  The correctly-defined replacement for "correlation with the target".
#    ANOVA F  : does the feature's MEAN differ across classes? Linear, cheap,
#               and exactly what f_classif tests. Blind to a feature that
#               separates classes without shifting the mean.
#    Mutual information : any statistical dependence at all, including
#               non-monotonic ones (e.g. a class that occupies the MIDDLE of a
#               feature's range). Non-negative, in nats, 0 = independent.
#  Read them together: a feature high on MI but low on F is exactly the
#  non-linear structure the tree and network models can exploit and a logistic
#  baseline cannot.
def plot_feature_target_association(X_df, y_enc):
    f_stat, _ = f_classif(X_df, y_enc)
    mi = mutual_info_classif(X_df, y_enc, random_state=SEED)

    assoc = (pd.DataFrame({"feature": X_df.columns, "ANOVA_F": f_stat, "MutualInfo": mi})
               .sort_values("MutualInfo", ascending=False))
    print("\n" + "=" * 78)
    print("FEATURE -> TARGET ASSOCIATION")
    print("=" * 78)
    print(assoc.round(4).to_string(index=False))

    panels = [("MutualInfo", SERIES_COLORS[0], "Mutual information (nats)"),
              ("ANOVA_F",    SERIES_COLORS[1], "ANOVA F statistic")]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.32 * len(assoc))))
    for ax, (col, color, label) in zip(axes, panels):
        d = assoc.sort_values(col)
        ax.barh(d["feature"], d[col], color=color, height=0.62)
        _style(ax, label, label, None)
        ax.grid(axis='y', visible=False)
    fig.suptitle("Which features carry signal about the class?",
                 fontsize=13, fontweight='bold', color=INK)
    plt.tight_layout()
    save_fig(plt.gcf(), "eda_feature_target_association", subdir="eda")
    plt.show()
    plt.close(fig)

    if PER_ITEM_FIGURES:
        for col, color, label in panels:
            f1, ax1 = plt.subplots(figsize=(7, max(3.5, 0.32 * len(assoc))))
            d = assoc.sort_values(col)
            ax1.barh(d["feature"], d[col], color=color, height=0.62)
            _style(ax1, label, label, None)
            ax1.grid(axis='y', visible=False)
            f1.tight_layout()
            save_fig(f1, f"association_{col}", subdir="eda", close=True)
    register_table("Feature-target association", assoc)
    return assoc


plot_class_balance(df, target_name)
plot_class_conditional_densities(df, target_name)
plot_boxplots_by_class(df, target_name)
plot_correlation_heatmaps(df)
assoc_df = plot_feature_target_association(X_full, y_full_enc)


# =============================================================================
# =============================================================================
#  PART 4 — MODEL SPACE
# =============================================================================
# =============================================================================

# =============================================================================
#  SECTION 3 — KERAS HELPERS  (MLP, CNN-LSTM, Sequential Logistic)
# =============================================================================
#  REGRESSION -> CLASSIFICATION, three changes, all in this block:
#
#   1. OUTPUT HEAD. Dense(1, 'linear') + mse becomes Dense(1, 'sigmoid') +
#      binary_crossentropy for two classes, or Dense(K, 'softmax') +
#      categorical_crossentropy for K > 2. Cross-entropy is the loss that
#      matches a probability output; training a softmax head on MSE converges
#      badly because the gradient vanishes exactly where the model is most
#      wrong.
#   2. NO TARGET SCALER. The regression helpers standardized y and
#      inverse-transformed the predictions. Here the target is label-encoded
#      (Section A) and, for K > 2, one-hot encoded at the point of use. The
#      network's output is already in the units we want — probabilities.
#   3. CLASS WEIGHTS. Passed as per-row sample_weight computed from the
#      TRAINING FOLD only (never the validation fold), which is the Keras
#      equivalent of class_weight='balanced' and is what keeps the nets
#      comparable to the sklearn models under imbalance. sample_weight rather
#      than Keras's class_weight= because the latter does not accept one-hot
#      targets.
# =============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.get_logger().setLevel("ERROR")


def keras_head():
    """(units, activation, loss) for the output layer, from the task shape."""
    if IS_BINARY:
        return 1, "sigmoid", "binary_crossentropy"
    return N_CLASSES, "softmax", "categorical_crossentropy"


def keras_targets(y_enc):
    """Encoded labels -> the shape the chosen loss expects."""
    y_enc = np.asarray(y_enc).astype(np.int64).ravel()
    return y_enc.astype(np.float32) if IS_BINARY else one_hot(y_enc)


def keras_probabilities(raw):
    """Network output -> an (n, K) probability matrix, uniformly."""
    raw = np.asarray(raw, dtype=np.float64)
    if IS_BINARY:
        p = raw.ravel()
        return np.column_stack([1.0 - p, p])
    return raw


def balanced_sample_weight(y_enc, use_weights=True):
    """
    Per-row weights implementing class_weight='balanced' (n / (K * count_k)).
    Computed on the split it is called with, so a fold's weights never see the
    held-out rows.
    """
    y_enc = np.asarray(y_enc).astype(int).ravel()
    if not use_weights:
        return None
    present = np.unique(y_enc)
    w = compute_class_weight(class_weight="balanced", classes=present, y=y_enc)
    lookup = dict(zip(present, w))
    return np.array([lookup[v] for v in y_enc], dtype=np.float32)


def keras_predict(model, X):
    """
    Probabilities from a Keras model, the fast way for small inputs.

    model.predict() sets up a batched prediction loop and its own tf.function
    on every call. Across the hundreds of calls this file makes, that overhead
    dominates the actual arithmetic on tabular-sized data, so small arrays go
    through the model directly and only large ones pay for the loop.
    """
    X = np.asarray(X, dtype=np.float32)
    if len(X) <= 4096:
        return np.asarray(model(X, training=False))
    return model.predict(X, verbose=0)


def keras_cv_loss(build_fn, X, y, batch_size=32, epochs=None, cv=None, to3d=False,
                  trial=None, patience=None, balanced=False):
    """
    Manual stratified K-fold for Keras: refits the feature scaler inside every
    fold (no leakage), early-stops each fold on its own validation split, and
    reports intermediate values to Optuna for pruning.

    Returns a value to MINIMIZE: -PRIMARY_METRIC, so every Keras model is tuned
    on exactly the same objective as the sklearn and boosted models.
    """
    cv = cv or keras_cv
    epochs = KERAS_EPOCHS if epochs is None else epochs
    patience = KERAS_PATIENCE if patience is None else patience
    fold_scores = []
    for k, (tr, va) in enumerate(cv.split(X, y)):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        sx = SCALER_CLS().fit(Xtr)
        Xtr_s, Xva_s = sx.transform(Xtr), sx.transform(Xva)

        # Resample AFTER scaling and on the training part only — SMOTE chooses
        # neighbours by Euclidean distance, so it has to see comparable scales,
        # and the validation fold must keep its real class balance or the score
        # this function returns is measured on a population that does not exist.
        Xtr_s, ytr = resample_fit(Xtr_s, ytr)

        if to3d:   # (n, f) -> (n, f, 1): the features act as the "sequence" axis
            Xtr_s = Xtr_s.reshape(Xtr_s.shape[0], Xtr_s.shape[1], 1)
            Xva_s = Xva_s.reshape(Xva_s.shape[0], Xva_s.shape[1], 1)

        sw_tr = balanced_sample_weight(ytr, balanced)
        sw_va = balanced_sample_weight(yva, balanced)

        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED)
        model = build_fn(Xtr_s.shape[1:])
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                           restore_best_weights=True)
        val_data = ((Xva_s, keras_targets(yva)) if sw_va is None
                    else (Xva_s, keras_targets(yva), sw_va))
        model.fit(Xtr_s, keras_targets(ytr), validation_data=val_data,
                  epochs=epochs, batch_size=batch_size, verbose=0,
                  callbacks=[es], sample_weight=sw_tr)

        proba = keras_probabilities(keras_predict(model, Xva_s))
        fold_scores.append(primary_score(yva, proba))

        if trial is not None:                      # Optuna pruning hook
            trial.report(float(-np.nanmean(fold_scores)), step=k)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
    return float(-np.nanmean(fold_scores))


def keras_fit(build_fn, Xtr, ytr, batch_size, epochs=None, to3d=False,
              val_frac=0.15, patience=None, balanced=False):
    """
    Fit the tuned architecture on (Xtr, ytr). Returns (model, proba_fn) where
    proba_fn carries the fitted feature scaler so it can be called on ANY
    future array of raw features, exactly like a sklearn estimator.

    The internal validation split is STRATIFIED — an unstratified 15% holdout
    on an imbalanced target can end up with no minority rows at all, which
    makes val_loss a majority-class-only quantity and early stopping blind.
    """
    # The final fit gets a little more headroom than a search fold: it happens
    # once per model, and this is the network that is actually scored.
    epochs = int(KERAS_EPOCHS * 1.3) if epochs is None else epochs
    patience = KERAS_PATIENCE + 5 if patience is None else patience

    n_val = max(N_CLASSES, int(len(Xtr) * val_frac))
    idx = np.arange(len(Xtr))
    tr_i, va_i = train_test_split(idx, test_size=n_val, random_state=SEED, stratify=ytr)

    sx = SCALER_CLS().fit(Xtr[tr_i])
    Xt, Xv = sx.transform(Xtr[tr_i]), sx.transform(Xtr[va_i])

    # Training portion only; the internal validation split keeps its real
    # balance so early stopping is judged on realistic data.
    yt_fit, yv_fit = ytr[tr_i], ytr[va_i]
    Xt, yt_fit = resample_fit(Xt, yt_fit)

    if to3d:
        Xt = Xt.reshape(*Xt.shape, 1)
        Xv = Xv.reshape(*Xv.shape, 1)

    sw_t = balanced_sample_weight(yt_fit, balanced)
    sw_v = balanced_sample_weight(yv_fit, balanced)

    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    model = build_fn(Xt.shape[1:])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                       restore_best_weights=True)
    val_data = ((Xv, keras_targets(yv_fit)) if sw_v is None
                else (Xv, keras_targets(yv_fit), sw_v))
    model.fit(Xt, keras_targets(yt_fit), validation_data=val_data, epochs=epochs,
              batch_size=batch_size, verbose=0, callbacks=[es], sample_weight=sw_t)

    def proba_fn(Xnew):
        Xs = sx.transform(np.asarray(Xnew, dtype=np.float32))
        if to3d:
            Xs = Xs.reshape(*Xs.shape, 1)
        return keras_probabilities(keras_predict(model, Xs))

    return model, proba_fn


import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_study():
    #  direction="minimize" throughout, and every objective returns
    #  -PRIMARY_METRIC (or a loss), so "lower is better" holds uniformly even
    #  though the reported metric is one where higher is better.
    #
    #  n_warmup_steps=1: a trial may be pruned after its FIRST reported fold
    #  rather than its third. On the Keras models a fold is a whole network
    #  training, so this is the difference between abandoning a hopeless
    #  configuration after one training and after three.
    return optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )


def run_study(objective, key):
    """
    Run one model's search under the profile's trial count AND wall-clock
    budget, reporting progress as it goes.

    Two things this fixes. First, a search that prints nothing for forty
    minutes is indistinguishable from one that has hung, so every tenth trial
    reports the best score so far and the elapsed time. Second, `timeout`
    bounds the damage: Optuna finishes the trial in flight and then stops,
    keeping the best parameters found, so an expensive model degrades into a
    shorter search instead of stalling the notebook.

    A search cut short by the timeout says so, loudly — an under-searched model
    must not be mistaken for a fully tuned one when reading the results table.
    """
    n_target = N_TRIALS[key]
    study = make_study()
    t0 = time.time()

    def progress(st, trial):
        done = len(st.trials)
        if done % max(1, n_target // 10) and done != n_target:
            return
        try:
            best = f"{-st.best_value:.4f}"
        except ValueError:
            best = "n/a"
        print(f"    [{key}] trial {done}/{n_target}  best {PRIMARY_METRIC}={best}"
              f"  {(time.time() - t0) / 60:.1f} min", flush=True)

    study.optimize(objective, n_trials=n_target, timeout=STUDY_TIMEOUT,
                   callbacks=[progress])

    if STUDY_TIMEOUT is not None and len(study.trials) < n_target:
        print(f"    [{key}] hit the {STUDY_TIMEOUT}s budget after "
              f"{len(study.trials)}/{n_target} trials — these parameters are the "
              f"best of a SHORTENED search. Raise STUDY_TIMEOUT, or accept it and "
              f"say so when reporting this model.", flush=True)
    return study


# =============================================================================
#  1. SHALLOW MULTILAYER PERCEPTRON   —  Optuna (TPE)
# =============================================================================
#  Shallow = exactly ONE hidden layer. Essential knobs: hidden width,
#  activation, L2, dropout, learning rate, batch size — continuous/log-scaled,
#  so TPE beats a grid here. Epochs are NOT tuned: EarlyStopping picks them per
#  fold. `balanced` is tuned too, so the search decides whether re-weighting
#  the classes helps on this dataset rather than it being asserted.
# =============================================================================
def build_mlp(units, activation, l2, dropout, lr):
    out_units, out_act, loss = keras_head()

    def _b(input_shape):
        m = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(units, activation=activation,
                         kernel_regularizer=keras.regularizers.l2(l2)),
            layers.Dropout(dropout),
            layers.Dense(out_units, activation=out_act),
        ])
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)
        return m
    return _b


def mlp_search(X, y):
    def objective(trial):
        units      = trial.suggest_int("units", 8, 256, log=True)
        activation = trial.suggest_categorical("activation", ["relu", "tanh", "selu"])
        l2         = trial.suggest_float("l2", 1e-6, 1e-2, log=True)
        dropout    = trial.suggest_float("dropout", 0.0, 0.5)
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        balanced   = trial.suggest_categorical("balanced", [False, True])
        return keras_cv_loss(build_mlp(units, activation, l2, dropout, lr),
                             X, y, batch_size=batch_size, trial=trial, balanced=balanced)
    study = run_study(objective, "mlp")
    return study.best_params, -study.best_value


def mlp_refit(X, y, params):
    return keras_fit(
        build_mlp(params["units"], params["activation"], params["l2"],
                  params["dropout"], params["lr"]),
        X, y, batch_size=params["batch_size"], balanced=params.get("balanced", False))


mlp = ModelSpec("Shallow MLP", mlp_search, mlp_refit)


# =============================================================================
#  2. RANDOM FOREST   —  RandomizedSearchCV  (+ nested CV)
# =============================================================================
#  ~8 knobs but performance is driven mostly by max_features and the leaf-size
#  constraints. Random search dominates grid search at equal budget once >2-3
#  dims matter (Bergstra & Bengio, 2012), and RF is cheap enough for nested CV.
#  n_estimators is set high, not really "tuned": more trees only reduces
#  variance, it doesn't overfit RF.
#
#  CLASSIFICATION ADDITIONS: `criterion` (gini vs entropy — the impurity
#  measure, which has no regression analogue) and `class_weight`, including
#  'balanced_subsample', which recomputes the weights per bootstrap draw and is
#  usually the better of the two for RF.
# =============================================================================
from sklearn.ensemble import RandomForestClassifier

#  Grid keys carry the "model__" prefix because every sklearn model in this
#  file is assembled by build_pipeline() (Section B), which always includes a
#  sampler step — "passthrough" when RESAMPLING is None. One set of names
#  works whether or not resampling is on.
rf_space = {
    "model__n_estimators":      [300, 500, 800, 1200],
    "model__max_depth":         [None, 5, 10, 15, 20, 30],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf":  [1, 2, 4, 8],
    "model__max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
    "model__bootstrap":         [True, False],
    "model__criterion":         ["gini", "entropy"],
    "model__class_weight":      [None, "balanced", "balanced_subsample"],
}


def make_rf_estimator():
    return build_pipeline([("model", RandomForestClassifier(random_state=SEED, n_jobs=-1))])


def make_rf_search():
    return RandomizedSearchCV(
        make_rf_estimator(),
        rf_space, n_iter=N_ITER_RANDOM, scoring=SCORING,
        cv=inner_cv, random_state=SEED, n_jobs=-1, refit=True)


def rf_search(X, y):
    search = make_rf_search()
    search.fit(X, y)
    return search.best_params_, search.best_score_


def rf_refit(X, y, params):
    m = make_rf_estimator()
    m.set_params(**params)
    m.fit(X, y)
    return m, m.predict_proba


rf = ModelSpec("Random Forest", rf_search, rf_refit, search_factory=make_rf_search)


# =============================================================================
#  3. XGBOOST   —  Optuna (TPE) + per-fold early stopping
# =============================================================================
#  8 interacting hyperparameters, several log-scaled -> Bayesian search
#  territory. n_estimators is NOT sampled: each fold early-stops on its own
#  validation split; the median of the folds' best_iteration is reused for the
#  final refit (fixed, no early stopping needed on the full-data refit).
#
#  CLASSIFICATION ADDITIONS: the objective becomes logistic/softmax
#  automatically from the label vector (which is why Section A's encoding to
#  contiguous 0..K-1 ints is a hard requirement — XGBoost rejects anything
#  else), the early-stopping metric becomes logloss/mlogloss, and
#  scale_pos_weight is tuned for binary problems. scale_pos_weight is XGBoost's
#  imbalance knob: it multiplies the gradient of the positive class, and the
#  textbook starting value is n_negative/n_positive, so the search is centred
#  on that rather than on 1.
# =============================================================================
import xgboost as xgb

XGB_EVAL_METRIC = "logloss" if N_CLASSES == 2 else "mlogloss"


def xgb_search(X, y):
    neg_pos_ratio = float((y == 0).sum() / max((y == 1).sum(), 1)) if IS_BINARY else 1.0

    def objective(trial):
        params = dict(
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth        = trial.suggest_int("max_depth", 2, 12),
            min_child_weight = trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            gamma            = trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
        )
        if IS_BINARY:
            # Centred on the textbook n_neg/n_pos, spanning 'ignore imbalance'
            # (1.0) through 'over-correct it' (2x).
            params["scale_pos_weight"] = trial.suggest_float(
                "scale_pos_weight", max(1e-2, 0.5 * neg_pos_ratio), 2.0 * neg_pos_ratio, log=True)

        fold_scores, best_iters = [], []
        for k, (tr, va) in enumerate(inner_cv.split(X, y)):
            m = xgb.XGBClassifier(
                n_estimators=3000, early_stopping_rounds=50, eval_metric=XGB_EVAL_METRIC,
                tree_method="hist", random_state=SEED, n_jobs=-1, **params)
            # Resample the TRAINING part of the fold only. The eval_set drives
            # early stopping, so it has to keep the real class balance — stop
            # on a rebalanced set and the chosen n_estimators is tuned for a
            # population that does not exist.
            Xf, yf = resample_fit(X[tr], y[tr])
            m.fit(Xf, yf, eval_set=[(X[va], y[va])], verbose=False)
            fold_scores.append(primary_score(y[va], m.predict_proba(X[va])))
            best_iters.append(m.best_iteration)
            trial.report(float(-np.nanmean(fold_scores)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("n_estimators", int(np.median(best_iters)) + 1)
        return float(-np.nanmean(fold_scores))

    study = run_study(objective, "xgb")
    best = dict(study.best_params)
    best["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
    return best, -study.best_value


def xgb_refit(X, y, params):
    m = xgb.XGBClassifier(tree_method="hist", random_state=SEED, n_jobs=-1,
                          eval_metric=XGB_EVAL_METRIC, **params)
    Xf, yf = resample_fit(X, y)
    m.fit(Xf, yf, verbose=False)
    return m, m.predict_proba


xgboost_model = ModelSpec("XGBoost", xgb_search, xgb_refit)


# =============================================================================
#  4. ADABOOST   —  GridSearchCV  (+ nested CV)
# =============================================================================
#  Very few real knobs; the one people forget is the *weak learner's* depth,
#  which is tuned here jointly with learning_rate/n_estimators since those two
#  trade off directly. Small discrete space -> exhaustive grid is affordable.
#
#  REGRESSION -> CLASSIFICATION: the `loss` parameter
#  (linear/square/exponential) is regression-only and is gone. AdaBoost for
#  classification is SAMME, whose reweighting rule is fixed; the weak learner's
#  own class_weight takes its place in the grid. `algorithm` is deliberately
#  not passed — SAMME.R was removed in recent sklearn and the argument itself
#  is on the way out, so specifying it breaks across versions.
# =============================================================================
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_grid = {
    "model__estimator__max_depth":        [1, 2, 3, 4, 6, 8],
    "model__estimator__min_samples_leaf": [1, 5],
    "model__estimator__class_weight":     [None, "balanced"],
    "model__n_estimators":                [50, 100, 300, 600],
    "model__learning_rate":               [0.01, 0.05, 0.1, 0.5, 1.0],
}


def make_ada_estimator():
    return build_pipeline([("model", AdaBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=SEED), random_state=SEED))])


def make_ada_search():
    return GridSearchCV(make_ada_estimator(), ada_grid, scoring=SCORING,
                        cv=inner_cv, n_jobs=-1, refit=True)


def ada_search(X, y):
    search = make_ada_search()
    search.fit(X, y)
    return search.best_params_, search.best_score_


def ada_refit(X, y, params):
    m = make_ada_estimator()
    m.set_params(**params)     # routes model__estimator__* to the nested tree
    m.fit(X, y)
    return m, m.predict_proba


ada = ModelSpec("AdaBoost", ada_search, ada_refit, search_factory=make_ada_search)


# =============================================================================
#  5. LIGHT GRADIENT BOOSTING   —  Optuna (TPE) + per-fold early stopping
# =============================================================================
#  LightGBM grows leaf-wise, so num_leaves (not max_depth) is the primary
#  capacity knob and min_child_samples is the primary overfitting brake — on a
#  dataset this size they must be tuned jointly, which is exactly where a
#  multivariate Bayesian sampler beats independent grid axes.
#  class_weight is tuned alongside them: on a leaf-wise learner, re-weighting
#  interacts with min_child_samples (weighted leaves fill up differently), so
#  the two should not be chosen independently.
# =============================================================================
import lightgbm as lgb

LGB_EVAL_METRIC = "binary_logloss" if N_CLASSES == 2 else "multi_logloss"


def lgbm_search(X, y):
    def objective(trial):
        params = dict(
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 8, 256, log=True),
            max_depth         = trial.suggest_int("max_depth", 3, 15),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 100),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq    = trial.suggest_int("subsample_freq", 0, 7),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True),
            class_weight      = trial.suggest_categorical("class_weight", [None, "balanced"]),
        )
        fold_scores, best_iters = [], []
        for k, (tr, va) in enumerate(inner_cv.split(X, y)):
            m = lgb.LGBMClassifier(n_estimators=3000, random_state=SEED,
                                   n_jobs=-1, verbose=-1, **params)
            # Training part only — the eval_set must keep the real balance, for
            # the same reason as XGBoost above.
            Xf, yf = resample_fit(X[tr], y[tr])
            m.fit(Xf, yf, eval_set=[(X[va], y[va])], eval_metric=LGB_EVAL_METRIC,
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            fold_scores.append(primary_score(y[va], m.predict_proba(X[va])))
            best_iters.append(m.best_iteration_ or 100)
            trial.report(float(-np.nanmean(fold_scores)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("n_estimators", int(np.median(best_iters)) + 1)
        return float(-np.nanmean(fold_scores))

    study = run_study(objective, "lgbm")
    best = dict(study.best_params)
    best["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
    return best, -study.best_value


def lgbm_refit(X, y, params):
    m = lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, **params)
    X, y = resample_fit(X, y)
    m.fit(X, y)
    return m, m.predict_proba


lgbm = ModelSpec("LightGBM", lgbm_search, lgbm_refit)


# =============================================================================
#  6. CNN-LSTM   —  Optuna (TPE)
# =============================================================================
#  Tabular data has no time axis, so this is the standard workaround: reshape
#  (n, f) -> (n, f, 1) and let Conv1D + LSTM scan across the fixed feature
#  ordering as if it were a short sequence. It is a legitimate way to let the
#  model learn local interactions between adjacent features, but it is not
#  modelling real temporal dependence — treat its result as one more nonlinear
#  learner to compare, not as inherently superior to the MLP.
#
#  Note the consequence of the reshape for a classification problem
#  specifically: the "sequence" is your COLUMN ORDER, so one-hot dummy columns
#  from Section A end up adjacent and the convolution will happily pool across
#  levels of the same categorical variable. That is not wrong, but it is worth
#  knowing before reading anything into this model's feature attributions.
# =============================================================================
def build_cnn_lstm(filters, kernel_size, pool, lstm_units, dropout, rec_dropout,
                   dense_units, lr):
    out_units, out_act, loss = keras_head()

    def _b(input_shape):
        timesteps = input_shape[0]
        k = min(kernel_size, timesteps)
        m = keras.Sequential()
        m.add(layers.Input(shape=input_shape))
        m.add(layers.Conv1D(filters, k, padding="same", activation="relu"))
        if pool and timesteps >= 4:
            m.add(layers.MaxPooling1D(pool_size=2))
        m.add(layers.LSTM(lstm_units, dropout=dropout, recurrent_dropout=rec_dropout))
        m.add(layers.Dense(dense_units, activation="relu"))
        m.add(layers.Dropout(dropout))
        m.add(layers.Dense(out_units, activation=out_act))
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)
        return m
    return _b


def cnn_lstm_search(X, y):
    def objective(trial):
        filters     = trial.suggest_categorical("filters", [16, 32, 64, 128])
        kernel_size = trial.suggest_int("kernel_size", 2, 5)
        pool        = trial.suggest_categorical("pool", [True, False])
        lstm_units  = trial.suggest_categorical("lstm_units", [16, 32, 64, 128])
        dropout     = trial.suggest_float("dropout", 0.0, 0.5)
        rec_dropout = trial.suggest_float("rec_dropout", 0.0, 0.3)
        dense_units = trial.suggest_categorical("dense_units", [8, 16, 32, 64])
        lr          = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [16, 32, 64])
        balanced    = trial.suggest_categorical("balanced", [False, True])
        return keras_cv_loss(
            build_cnn_lstm(filters, kernel_size, pool, lstm_units, dropout,
                           rec_dropout, dense_units, lr),
            X, y, batch_size=batch_size, to3d=True, trial=trial, balanced=balanced)
    study = run_study(objective, "cnn_lstm")
    return study.best_params, -study.best_value


def cnn_lstm_refit(X, y, params):
    return keras_fit(
        build_cnn_lstm(params["filters"], params["kernel_size"], params["pool"],
                       params["lstm_units"], params["dropout"], params["rec_dropout"],
                       params["dense_units"], params["lr"]),
        X, y, batch_size=params["batch_size"], to3d=True,
        balanced=params.get("balanced", False))


cnn_lstm = ModelSpec("CNN-LSTM", cnn_lstm_search, cnn_lstm_refit)


# =============================================================================
#  7. K-NEAREST NEIGHBORS   —  GridSearchCV  (+ nested CV)
# =============================================================================
#  Tiny discrete space (k x weighting x distance metric) -> exhaustive search
#  is cheap and there's no reason to approximate it. Scaling is inside the
#  Pipeline because KNN is a pure distance method: features on wildly different
#  scales would let the largest-magnitude one dominate the metric.
#
#  CLASSIFICATION NOTE: KNN's predict_proba is the fraction of the k neighbours
#  voting for each class, so with k=5 it can only ever emit 0, 0.2, 0.4, ... —
#  a coarse, poorly-calibrated probability. That is fine for ROC-AUC (which
#  only needs a ranking) but it is why KNN usually looks bad on log-loss and
#  Brier, and why its calibration curve in Part 6 is a staircase. There is also
#  no class_weight parameter here; weights='distance' is the closest thing, and
#  it is in the grid.
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier

knn_pipe = build_pipeline([("scaler", SCALER_CLS()), ("model", KNeighborsClassifier())])

max_k = min(40, max(2, len(X_tr) // 2))
knn_grid = {
    "model__n_neighbors": list(range(1, max_k + 1)),
    "model__weights":     ["uniform", "distance"],
    "model__p":           [1, 2],
    "model__leaf_size":   [20, 30, 50],
}


def make_knn_search():
    return GridSearchCV(knn_pipe, knn_grid, scoring=SCORING,
                        cv=inner_cv, n_jobs=-1, refit=True)


def knn_search(X, y):
    search = make_knn_search()
    search.fit(X, y)
    return search.best_params_, search.best_score_


def knn_refit(X, y, params):
    m = build_pipeline([("scaler", SCALER_CLS()), ("model", KNeighborsClassifier())])
    m.set_params(**params)
    m.fit(X, y)
    return m, m.predict_proba


knn = ModelSpec("KNN", knn_search, knn_refit, search_factory=make_knn_search)


# =============================================================================
#  8. SUPPORT VECTOR MACHINE (SVC)   —  Optuna (TPE)
# =============================================================================
#  C and gamma each span 5-6 orders of magnitude and interact strongly (large C
#  + large gamma = memorisation). A log-uniform Bayesian search finds the ridge
#  in that space with far fewer evaluations than a brute grid, and SVC training
#  is O(n^2)-O(n^3) so evaluations aren't free.
#
#  REGRESSION -> CLASSIFICATION:
#    * `epsilon` (the width of SVR's insensitive tube) has no meaning here and
#      is gone; `class_weight` takes its place in the space.
#    * calibrated probabilities are REQUIRED, because this whole pipeline is
#      built on predict_proba. They are not free: a 5-fold calibration is fitted
#      on top of the SVM, so every SVC fit is really six, and this is the
#      slowest model in the file. It is also why labels come from argmax of the
#      calibrated probabilities here rather than from SVC.predict(), which uses
#      the uncalibrated decision function and can disagree near the boundary.
#      See make_probabilistic_svc() below for how this is obtained across
#      sklearn versions.
# =============================================================================
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import sklearn as _sklearn

#  HOW THE PROBABILITIES ARE OBTAINED, across sklearn versions.
#  SVC(probability=True) was deprecated in scikit-learn 1.9 and is scheduled
#  for removal in 1.11, with CalibratedClassifierCV(SVC(), ensemble=False) as
#  the named replacement. The two do the same thing — fit the SVM, then fit a
#  calibrator on cross-validated decision values — so this picks whichever the
#  installed version supports rather than emitting a deprecation warning on new
#  sklearn or crashing on old. Either way the cost is the same: one SVC fit per
#  calibration fold, which is what makes this the slowest model in the file.
_SKLEARN_VERSION = tuple(int(p) for p in _sklearn.__version__.split(".")[:2])
USE_CALIBRATED_SVC = _SKLEARN_VERSION >= (1, 9)


def make_probabilistic_svc(**params):
    """An SVC that exposes a usable predict_proba, on any sklearn version."""
    base = SVC(cache_size=1000, random_state=SEED, **params)
    if USE_CALIBRATED_SVC:
        return CalibratedClassifierCV(base, ensemble=False, cv=5)
    base.set_params(probability=True)
    return base


def svc_search(X, y):
    def objective(trial):
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid", "linear"])
        params = dict(
            kernel       = kernel,
            C            = trial.suggest_float("C", 1e-2, 1e4, log=True),
            class_weight = trial.suggest_categorical("class_weight", [None, "balanced"]),
        )
        if kernel != "linear":
            params["gamma"] = trial.suggest_float("gamma", 1e-5, 1e1, log=True)
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 4)
        if kernel in ("poly", "sigmoid"):
            params["coef0"] = trial.suggest_float("coef0", -1.0, 1.0)

        pipe = build_pipeline([("scaler", SCALER_CLS()),
                               ("model", make_probabilistic_svc(**params))])
        s = cross_val_score(pipe, X, y, cv=inner_cv, scoring=SCORING, n_jobs=-1)
        return float(-s.mean())

    study = run_study(objective, "svc")
    return study.best_params, -study.best_value


def svc_refit(X, y, params):
    m = build_pipeline([("scaler", SCALER_CLS()),
                        ("model", make_probabilistic_svc(**params))])
    m.fit(X, y)
    return m, m.predict_proba


svc = ModelSpec("SVM (SVC)", svc_search, svc_refit)


# =============================================================================
#  9. SEQUENTIAL LOGISTIC REGRESSION   —  Optuna (TPE)
# =============================================================================
#  A single dense layer with a sigmoid/softmax output, built with
#  keras.Sequential and fit by gradient descent — i.e. logistic regression as
#  the "zero hidden layer" baseline next to the MLP/CNN-LSTM. This is the
#  direct counterpart of the regression file's Sequential Linear Regression:
#  the same architecture with the loss and output activation swapped, which is
#  exactly the difference between linear and logistic regression.
#
#  Its job is to be the floor. If a 40-trial-tuned gradient-boosted ensemble
#  cannot beat a one-layer logistic model on your data, the honest conclusion
#  is that the signal is linear and the complex models are not earning their
#  interpretability cost.
# =============================================================================
def build_logit(lr, l1, l2):
    out_units, out_act, loss = keras_head()

    def _b(input_shape):
        m = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(out_units, activation=out_act,
                         kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2)),
        ])
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)
        return m
    return _b


def seq_logit_search(X, y):
    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        l1         = trial.suggest_float("l1", 1e-8, 1e-2, log=True)
        l2         = trial.suggest_float("l2", 1e-8, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        balanced   = trial.suggest_categorical("balanced", [False, True])
        return keras_cv_loss(build_logit(lr, l1, l2), X, y,
                             batch_size=batch_size, trial=trial, balanced=balanced)
    study = run_study(objective, "seq_logit")
    return study.best_params, -study.best_value


def seq_logit_refit(X, y, params):
    return keras_fit(build_logit(params["lr"], params["l1"], params["l2"]),
                     X, y, batch_size=params["batch_size"],
                     balanced=params.get("balanced", False))


seq_logit = ModelSpec("Sequential Logistic Regression", seq_logit_search, seq_logit_refit)


# =============================================================================
#  SECTION 4 — RUN EVERYTHING
# =============================================================================
#  A plain list of models — fit and evaluate any subset the same way. Comment
#  lines out to skip a model, or reorder freely — nothing below this line
#  depends on definition order.
# =============================================================================
models = [rf, mlp, svc, xgboost_model, lgbm, ada, knn, cnn_lstm, seq_logit]


# =============================================================================
#  SEARCH-SPACE REGISTRY  (for the hyperparameter report in Part 6)
# =============================================================================
#  The Grid/Randomized models carry their space as a dict already, so those
#  entries point straight at it and can never fall out of step with what is
#  actually searched. The Optuna models define their space IMPERATIVELY, inside
#  the objective, as a sequence of trial.suggest_* calls — there is no object to
#  introspect — so the ranges are mirrored declaratively here for reporting.
#
#  That mirror is the one thing in this file that can silently drift: change a
#  suggest_* bound in Part 4 and this table will keep quoting the old one. The
#  consistency check below catches the common half of that (a parameter that
#  appears in the tuned result but is missing here) and warns; it cannot detect
#  a bound that was edited in one place only, so change them together.
SEARCH_SPACES = {
    "Random Forest":                  rf_space,     # dict, as searched
    "AdaBoost":                       ada_grid,     # dict, as searched
    "KNN":                            knn_grid,     # dict, as searched
    "XGBoost": {
        "learning_rate":    "log-uniform [0.01, 0.3]",
        "max_depth":        "int [2, 12]",
        "min_child_weight": "log-uniform [0.01, 20.0]",
        "subsample":        "uniform [0.5, 1.0]",
        "colsample_bytree": "uniform [0.4, 1.0]",
        "gamma":            "log-uniform [1e-8, 5.0]",
        "reg_alpha":        "log-uniform [1e-8, 10.0]",
        "reg_lambda":       "log-uniform [1e-8, 20.0]",
        "scale_pos_weight": "log-uniform [0.5, 2.0] x n_neg/n_pos (binary only)",
        "n_estimators":     "not searched - median best_iteration from per-fold early stopping",
    },
    "LightGBM": {
        "learning_rate":     "log-uniform [0.01, 0.3]",
        "num_leaves":        "int, log [8, 256]",
        "max_depth":         "int [3, 15]",
        "min_child_samples": "int [5, 100]",
        "subsample":         "uniform [0.5, 1.0]",
        "subsample_freq":    "int [0, 7]",
        "colsample_bytree":  "uniform [0.4, 1.0]",
        "reg_alpha":         "log-uniform [1e-8, 10.0]",
        "reg_lambda":        "log-uniform [1e-8, 20.0]",
        "class_weight":      "{None, balanced}",
        "n_estimators":      "not searched - median best_iteration from per-fold early stopping",
    },
    "SVM (SVC)": {
        "kernel":       "{rbf, poly, sigmoid, linear}",
        "C":            "log-uniform [1e-2, 1e4]",
        "class_weight": "{None, balanced}",
        "gamma":        "log-uniform [1e-5, 1e1]  (non-linear kernels only)",
        "degree":       "int [2, 4]  (poly only)",
        "coef0":        "uniform [-1.0, 1.0]  (poly and sigmoid only)",
    },
    "Shallow MLP": {
        "units":      "int, log [8, 256]",
        "activation": "{relu, tanh, selu}",
        "l2":         "log-uniform [1e-6, 1e-2]",
        "dropout":    "uniform [0.0, 0.5]",
        "lr":         "log-uniform [1e-4, 1e-2]",
        "batch_size": "{16, 32, 64, 128}",
        "balanced":   "{False, True}  (class-weighted sample weights)",
    },
    "CNN-LSTM": {
        "filters":     "{16, 32, 64, 128}",
        "kernel_size": "int [2, 5]",
        "pool":        "{True, False}",
        "lstm_units":  "{16, 32, 64, 128}",
        "dropout":     "uniform [0.0, 0.5]",
        "rec_dropout": "uniform [0.0, 0.3]",
        "dense_units": "{8, 16, 32, 64}",
        "lr":          "log-uniform [1e-4, 5e-3]",
        "batch_size":  "{16, 32, 64}",
        "balanced":    "{False, True}  (class-weighted sample weights)",
    },
    "Sequential Logistic Regression": {
        "lr":         "log-uniform [1e-4, 1e-1]",
        "l1":         "log-uniform [1e-8, 1e-2]",
        "l2":         "log-uniform [1e-8, 1e-2]",
        "batch_size": "{16, 32, 64, 128}",
        "balanced":   "{False, True}  (class-weighted sample weights)",
    },
}

SEARCH_STRATEGY = {
    "Random Forest": "RandomizedSearchCV", "AdaBoost": "GridSearchCV",
    "KNN": "GridSearchCV", "XGBoost": "Optuna TPE", "LightGBM": "Optuna TPE",
    "SVM (SVC)": "Optuna TPE", "Shallow MLP": "Optuna TPE",
    "CNN-LSTM": "Optuna TPE", "Sequential Logistic Regression": "Optuna TPE",
}


# =============================================================================
#  SECTION 4B — CUSTOM ENSEMBLE LEARNING ALGORITHMS  (stacking & soft voting)
# =============================================================================
#  Meta-models built ON TOP of the 9 tuned base models above. Each one is
#  itself a ModelSpec, so every downstream Part treats it exactly like a base
#  model — .fit(), .predict_proba(), .evaluate(), .refit_on() all behave the same.
#
#  ---------------------------------------------------------------------------
#  REGRESSION -> CLASSIFICATION: THE MEMBERS CONTRIBUTE PROBABILITIES
#  ---------------------------------------------------------------------------
#  The regression ensembles blended point predictions. Blending HARD LABELS
#  would be the naive translation and it is strictly worse:
#    * hard-label (majority) voting throws away every member's confidence, so a
#      member that is 51% sure counts exactly as much as one that is 99% sure,
#      and ties have to be broken arbitrarily;
#    * it produces no probabilities, so the ensemble could not be scored on
#      ROC-AUC, PR-AUC, log-loss or Brier, and could not have an ROC curve —
#      i.e. most of Part 6 would go blank for exactly the models that are
#      supposed to be best.
#  So both families here are SOFT: members contribute predict_proba output.
#
#  ---------------------------------------------------------------------------
#  STACKING: OUT-OF-FOLD META-FEATURES ARE NOT OPTIONAL
#  ---------------------------------------------------------------------------
#  The meta-learner must be trained on OUT-OF-FOLD base predictions. If you
#  instead fit the base models on all of X and feed the meta-learner their
#  in-sample probabilities, those probabilities are fit to noise the base
#  models have already memorised — and in classification they are typically
#  saturated at 0.0/1.0, which is a far more extreme version of the problem
#  than in regression. The meta-learner then learns to trust whichever base
#  model OVERFITS HARDEST and the stack generalises worse than its own members.
#  _oof_probas() below does the stratified K-fold version properly.
#
#  Meta-learner is LogisticRegressionCV (the classification counterpart of the
#  regression file's RidgeCV): an L2-regularised multinomial logistic model
#  with the penalty chosen by internal CV. Regularisation is not optional here
#  — base-model probabilities are extremely collinear (every member estimates
#  the same posterior), so an unpenalised fit produces huge cancelling
#  coefficients that swing wildly with tiny data changes. Coefficients may come
#  out NEGATIVE; that is not necessarily a bug, a member can be useful as a
#  correction term to the others.
#
#  ---------------------------------------------------------------------------
#  COST WARNING — READ BEFORE RUNNING
#  ---------------------------------------------------------------------------
#  One stacking fit = (ENSEMBLE_CV folds x n_members) base refits for the OOF
#  pass, PLUS n_members full refits. For S1 (all 9 members, 5 folds) that is
#  54 base fits, three of which are Keras networks and one of which is an SVC
#  with probability=True (itself 6 fits). Every downstream call to .refit_on()
#  on an ensemble re-triggers that whole cascade, so the ensembles are
#  deliberately kept OUT of the expensive refit loops (the K-fold sweep in
#  Part 5, the learning curves and bias-variance bootstrap in Part 6, and SHAP
#  in Part 7) and included only in the cheap predict-only tables and plots.
# =============================================================================
from sklearn.linear_model import LogisticRegressionCV

ENSEMBLE_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

#  The OOF pass used purely to REPORT an honest cv_score for each ensemble is a
#  second, separate pass over the members. Set this False to skip it (the
#  ensembles still fit and predict identically; their CV column just shows
#  NaN), which roughly halves ensemble fitting time.
ENSEMBLE_REPORT_CV = True


def _require_tuned(name, members):
    """Ensembles reuse each member's ALREADY-tuned hyperparameters via
    refit_on(), so the base models must have been fitted first."""
    missing = [m.name for m in members if m.best_params is None]
    if missing:
        raise RuntimeError(
            f"{name}: these base models have not been tuned yet: {missing}. "
            f"Run fit_all(models, X_tr, y_tr) before fitting any ensemble.")


def _oof_probas(members, X, y, cv=None):
    """
    Out-of-fold PROBABILITY tensor, shape (n_members, n_samples, n_classes).
    Entry [j, i] is member j's predicted class distribution for row i, made by
    a copy of that member that never saw row i during training.
    Folds are stratified, so each member is refit on a class-balanced split.
    """
    cv = cv or ENSEMBLE_CV
    oof = np.zeros((len(members), len(X), N_CLASSES), dtype=np.float64)
    for j, member in enumerate(members):
        for tr, va in cv.split(X, y):
            proba_fn = member.refit_on(X[tr], y[tr])
            oof[j, va, :] = proba_fn(X[va])
    return oof


def _meta_features(proba_tensor):
    """
    (n_members, n, K) -> the design matrix the meta-learner sees.

    Binary: one column per member, P(positive). The complementary column is
    redundant (it is 1 - the first) and including it would make the design
    matrix exactly singular.
    Multiclass: all K columns per member, horizontally concatenated. They sum
    to 1 per member, which is collinear too — but dropping a class would make
    the result depend on which class was dropped, and the L2 penalty handles
    the collinearity cleanly.
    """
    M, n, K = proba_tensor.shape
    if IS_BINARY:
        return np.column_stack([proba_tensor[j][:, POS_LABEL] for j in range(M)])
    return np.concatenate([proba_tensor[j] for j in range(M)], axis=1)


#  CLASS WEIGHTING OF THE META-LEARNER — a knob worth knowing about.
#  The base models each tune their own class_weight, but the meta-learner does
#  not inherit any of that, and on an imbalanced target an unweighted logistic
#  stack will happily learn an intercept that puts EVERY row below 0.5. The
#  result is the failure this file is built to catch: an excellent ROC-AUC
#  (the ranking is fine) next to F1 = 0 and balanced accuracy = 0.50 (the
#  argmax decision never fires for the minority class). If you see that
#  combination in the summary table for a stack, this is why.
#
#  It is left at None by default because the fix should be deliberate, and
#  there are two different fixes depending on what you want:
#    * 'balanced' here — re-weights the meta-learner so its 0.5 cutoff lands
#      somewhere useful. Cheap, but it distorts the probabilities, so log-loss,
#      Brier and the calibration curves all get worse.
#    * leave None and move the THRESHOLD instead (Section 8E). This keeps the
#      probabilities calibrated and treats the operating point as the separate
#      decision it actually is. Usually the better answer.
STACK_META_CLASS_WEIGHT = None      # or 'balanced'


def _make_meta():
    return LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 7), cv=ENSEMBLE_CV, scoring=SCORING,
        class_weight=STACK_META_CLASS_WEIGHT,
        max_iter=5000, random_state=SEED, n_jobs=-1)


# -- STACKING FACTORY ---------------------------------------------------------
def make_stacking_spec(name, members):
    """Level-1 LogisticRegressionCV meta-learner on stratified K-fold OOF
    class probabilities of `members`."""
    member_names = [m.name for m in members]

    def _fit_meta(X, y):
        oof = _oof_probas(members, X, y)
        Z = _meta_features(oof)
        meta = _make_meta().fit(Z, y)
        return Z, meta

    def search_fn(X, y):
        _require_tuned(name, members)
        if not ENSEMBLE_REPORT_CV:
            return {"members": member_names, "meta": "LogisticRegressionCV"}, None
        Z, meta = _fit_meta(X, y)
        coefs = np.atleast_2d(meta.coef_)
        params = {
            "members": member_names,
            "meta": "LogisticRegressionCV",
            "meta_C": float(np.ravel(meta.C_)[0]),
            # For a binary stack there is one coefficient per member, so the
            # blend is directly readable. For multiclass there are K x M, so
            # the mean |coef| per member is reported as a contribution summary.
            "blend_weights": ({n: round(float(c), 4) for n, c in zip(member_names, coefs[0])}
                              if IS_BINARY else
                              {n: round(float(np.abs(coefs[:, j * N_CLASSES:(j + 1) * N_CLASSES]).mean()), 4)
                               for j, n in enumerate(member_names)}),
        }
        return params, primary_score(y, meta.predict_proba(Z))   # honest OOF blend score

    def refit_fn(X, y, params):
        _require_tuned(name, members)
        # The meta-learner is refit here rather than reused, because it is part
        # of the model: on a bootstrap resample (Part 6 bias-variance) the
        # correct blend weights are the ones learned from THAT resample.
        _, meta = _fit_meta(X, y)
        base_proba_fns = [m.refit_on(X, y) for m in members]

        def proba_fn(Xnew):
            tensor = np.stack([pf(Xnew) for pf in base_proba_fns], axis=0)
            return meta.predict_proba(_meta_features(tensor))

        return {"meta": meta, "members": member_names}, proba_fn

    return ModelSpec(name, search_fn, refit_fn)


# -- SOFT VOTING FACTORY ------------------------------------------------------
def make_voting_spec(name, members, weighting="uniform"):
    """
    Weighted average of member PROBABILITY MATRICES (soft voting), then argmax.
      weighting="uniform"         -> equal weights; no OOF pass needed to fit.
      weighting="inverse_logloss" -> weight_j proportional to 1 / OOF_logloss_j,
                                     so better-calibrated members count for
                                     more. The weights come from OUT-OF-FOLD
                                     log-loss, not training log-loss — training
                                     log-loss would just rank the members by how
                                     hard they overfit.

    Log-loss is the weighting criterion (rather than accuracy or AUC) because
    it is the metric that actually matters for an average of probabilities: a
    member with great AUC but wildly overconfident probabilities will drag a
    soft-voting blend around, and log-loss is what detects that.
    """
    member_names = [m.name for m in members]

    def search_fn(X, y):
        _require_tuned(name, members)
        need_oof = (weighting == "inverse_logloss") or ENSEMBLE_REPORT_CV
        oof = _oof_probas(members, X, y) if need_oof else None

        if weighting == "uniform":
            w = np.ones(len(members)) / len(members)
        elif weighting == "inverse_logloss":
            losses = np.array([log_loss(y, oof[j], labels=list(range(N_CLASSES)))
                               for j in range(len(members))])
            w = (1.0 / losses) / (1.0 / losses).sum()
        else:
            raise ValueError(f"unknown weighting: {weighting!r}")

        params = {
            "members": member_names,
            "weighting": weighting,
            # Full precision, NOT rounded: refit_fn reads these back and does
            # arithmetic with them. Rounding to 4 dp makes uniform weights
            # 0.3333... which sum to 0.9999 rather than 1.0 — which for
            # probabilities means the blended rows no longer sum to 1 and
            # log_loss silently renormalises them.
            "weights": {n: float(wi) for n, wi in zip(member_names, w)},
        }
        cv_score = (primary_score(y, np.tensordot(w, oof, axes=(0, 0)))
                    if ENSEMBLE_REPORT_CV else None)
        return params, cv_score

    def refit_fn(X, y, params):
        _require_tuned(name, members)
        w = np.array([params["weights"][n] for n in member_names], dtype=np.float64)
        w = w / w.sum()          # guard against drift; weights must sum to 1
        base_proba_fns = [m.refit_on(X, y) for m in members]

        def proba_fn(Xnew):
            tensor = np.stack([pf(Xnew) for pf in base_proba_fns], axis=0)
            blended = np.tensordot(w, tensor, axes=(0, 0))
            # Renormalise defensively: a member returning a row that does not
            # sum to exactly 1 (float error, or a clipped Keras sigmoid) would
            # otherwise propagate into log_loss and the calibration curves.
            return blended / blended.sum(axis=1, keepdims=True)

        return {"weights": dict(zip(member_names, w)), "members": member_names}, proba_fn

    return ModelSpec(name, search_fn, refit_fn)


# -- ENSEMBLE ROSTER ----------------------------------------------------------
#  Groupings are by INDUCTIVE BIAS, not by accuracy. Ensembling pays off when
#  members make DIFFERENT errors — averaging four gradient-boosted tree models
#  that all misclassify the same rows buys almost nothing, while combining a
#  tree, a kernel machine and a network can, because their failure modes
#  differ. That is why the family groups (S2-S4) are included alongside the
#  cross-family ones (S1, S5): the comparison shows whether diversity actually
#  bought anything on your data, which is a result worth reporting either way.
#  Membership follows the `models` roster in Section 4, so commenting a model
#  out there removes it from every ensemble too. Without this, dropping the
#  expensive Keras models from `models` would leave the ensembles holding
#  untuned specs and _require_tuned() would stop the run — which is exactly
#  what you do NOT want when you dropped them to save time.
_ACTIVE = {id(m) for m in models}
_pick = lambda group: [m for m in group if id(m) in _ACTIVE]

TREE_MEMBERS    = _pick([rf, xgboost_model, lgbm, ada])     # bagging + 3 boosting variants
NEURAL_MEMBERS  = _pick([mlp, cnn_lstm, seq_logit])         # the Keras family
KERNEL_MEMBERS  = _pick([svc, knn])                         # kernel + instance-based
DIVERSE_MEMBERS = _pick([rf, xgboost_model, mlp, svc, knn]) # one strong pick per family
ALL_MEMBERS     = list(models)

# Stacking (LogisticRegressionCV meta-learner on OOF probabilities)
S1 = make_stacking_spec("S1 Stack (All 9)",        ALL_MEMBERS)
S2 = make_stacking_spec("S2 Stack (Trees)",        TREE_MEMBERS)
S3 = make_stacking_spec("S3 Stack (Neural)",       NEURAL_MEMBERS)
S4 = make_stacking_spec("S4 Stack (Kernel+KNN)",   KERNEL_MEMBERS)
S5 = make_stacking_spec("S5 Stack (Cross-family)", DIVERSE_MEMBERS)

# Soft voting — same groupings, so stacking vs voting is a controlled
# comparison: identical members, different combination rule.
V1 = make_voting_spec("V1 Vote (All 9)",         ALL_MEMBERS)
V2 = make_voting_spec("V2 Vote (Trees)",         TREE_MEMBERS)
V3 = make_voting_spec("V3 Vote (Neural)",        NEURAL_MEMBERS)
V4 = make_voting_spec("V4 Vote (Kernel+KNN)",    KERNEL_MEMBERS)
V5 = make_voting_spec("V5 Vote (Cross-family)",  DIVERSE_MEMBERS)
V6 = make_voting_spec("V6 Vote (All 9, wtd)",    ALL_MEMBERS, weighting="inverse_logloss")

#  An ensemble of fewer than two members is just that member under another
#  name, so groups emptied by the roster filter above are dropped rather than
#  reported as duplicates.
_group_size = {"S1": len(ALL_MEMBERS), "S2": len(TREE_MEMBERS), "S3": len(NEURAL_MEMBERS),
               "S4": len(KERNEL_MEMBERS), "S5": len(DIVERSE_MEMBERS),
               "V1": len(ALL_MEMBERS), "V2": len(TREE_MEMBERS), "V3": len(NEURAL_MEMBERS),
               "V4": len(KERNEL_MEMBERS), "V5": len(DIVERSE_MEMBERS), "V6": len(ALL_MEMBERS)}
_keep = lambda tag, spec: spec if _group_size[tag] >= 2 else None

stacking_models = [m for m in (_keep("S1", S1), _keep("S2", S2), _keep("S3", S3),
                               _keep("S4", S4), _keep("S5", S5)) if m is not None]
voting_models   = [m for m in (_keep("V1", V1), _keep("V2", V2), _keep("V3", V3),
                               _keep("V4", V4), _keep("V5", V5), _keep("V6", V6)) if m is not None]
ensemble_models = stacking_models + voting_models
if len(ensemble_models) < 11:
    print(f"\n{11 - len(ensemble_models)} ensemble(s) skipped: their member group has "
          f"fewer than two models in the current `models` roster.")


# =============================================================================
# =============================================================================
#  PART 5 — HYPERPARAMETER OPTIMIZATION
# =============================================================================
# =============================================================================

# =============================================================================
#  RUN THE TUNING
#  fit_all() is where every model's hyperparameter search (Grid/Random/Optuna,
#  per Part 2's strategy) actually executes, then refits each on the full
#  training set with the best params found.
# =============================================================================
fit_all(models, X_tr, y_tr)

# ---- Ensembles (Section 4B) are fitted SECOND, and the order is mandatory: --
#  every ensemble reuses its members' tuned hyperparameters via refit_on(), so
#  the base models must already carry best_params. Fitting these first raises a
#  RuntimeError from _require_tuned() rather than silently using defaults.
fit_all(ensemble_models, X_tr, y_tr)

#  `all_models` = 9 base + 11 ensembles. Used below for the CHEAP,
#  predict-only tables and plots. `models` deliberately stays base-only so the
#  expensive refit loops (K-fold sweep just below, learning curves and
#  bias-variance in Part 6, SHAP in Part 7) do not each re-trigger the full
#  base-model cascade for every ensemble.
all_models = models + ensemble_models

# =============================================================================
#  SECTION 5 — K-FOLD STABILITY SWEEP  (per already-tuned model)
# =============================================================================
#  Checks how sensitive each model's CV score is to the *number* of folds,
#  reusing the hyperparameters found by fit_all() — re-running the full search
#  at every one of 9 fold counts for all 9 models would mean thousands of extra
#  Optuna trials, so this only refits a fresh model per fold via
#  spec.refit_on().
#
#  CLASSIFICATION NOTES:
#    * folds are STRATIFIED, so every fold keeps the class priors. Without
#      that, a k=10 fold on an imbalanced target can contain no minority rows
#      and its AUC is undefined, which turns the whole sweep into noise.
#    * the tracked metrics are Accuracy / F1 / ROC-AUC / MCC — one threshold
#      metric, one imbalance-aware threshold metric, one ranking metric, one
#      all-four-cells metric. Watching them together shows WHICH kind of
#      stability a model has: AUC is usually far steadier across fold counts
#      than F1, because F1 also inherits the 0.5-threshold's sensitivity to
#      each fold's class mix.
#    * k is capped at the smallest class count — StratifiedKFold cannot make
#      more folds than there are members of the rarest class.
# =============================================================================
SWEEP_METRICS = ("Accuracy", "F1", "ROC_AUC", "MCC")

_min_class_count = int(np.bincount(y_tr, minlength=N_CLASSES).min())
fold_range = range(2, min(SWEEP_MAX_K, _min_class_count + 1))
if fold_range.stop <= 2:
    raise ValueError(f"rarest class has only {_min_class_count} training row(s) — "
                     f"not enough to cross-validate. Collect more data for it, "
                     f"merge it into a neighbouring class, or drop it.")
print(f"\nK-fold sweep range: k = {fold_range.start}..{fold_range.stop - 1} "
      f"(capped by the rarest class, n={_min_class_count})")


def fold_sweep(spec, X, y, folds=None):
    """
    Returns {metric: {'means': [...], 'stds': [...]}} across the fold counts,
    refitting spec's already-tuned hyperparameters fresh on each fold split.

    The parameter is `folds`, not `fold_range`, so it does not shadow the
    module-level `fold_range` it defaults to. `def f(fold_range=fold_range)`
    captures the global when the def RUNS, which in a notebook means the def
    cannot even be defined before the cell that sets it; and the obvious
    late-binding rewrite is silently broken when the names match, because
    `fold_range = fold_range if fold_range is None else fold_range` resolves
    both sides to the parameter and leaves it None.
    """
    folds = fold_range if folds is None else folds
    out = {metric: {"means": [], "stds": []} for metric in SWEEP_METRICS}
    for k in folds:
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        fold_scores = {metric: [] for metric in SWEEP_METRICS}
        for tr, va in kfold.split(X, y):
            proba_fn = spec.refit_on(X[tr], y[tr])
            proba = proba_fn(X[va])
            m = compute_metrics(y[va], labels_from_proba(proba), proba)
            for metric in SWEEP_METRICS:
                fold_scores[metric].append(m[metric])
        line = f"{spec.name} | Folds={k}:"
        for metric in SWEEP_METRICS:
            arr = np.array(fold_scores[metric], dtype=float)
            out[metric]["means"].append(np.nanmean(arr))
            out[metric]["stds"].append(np.nanstd(arr))
            line += f"  {metric}={np.nanmean(arr):.4f}+/-{np.nanstd(arr):.4f}"
        print(line)
    return out


sweep_results = {spec.name: fold_sweep(spec, X_tr, y_tr) for spec in models}

def _draw_sweep(ax, res, metric, color):
    means, stds = res[metric]["means"], res[metric]["stds"]
    ax.errorbar(list(fold_range), means, yerr=stds, marker='o', markersize=5,
                capsize=4, linestyle='-', linewidth=2, color=color,
                ecolor=GRID_COLOR, elinewidth=2)
    _style(ax, metric, 'Number of folds (k)', metric)
    ax.set_xticks(list(fold_range))


# One grid per model to scan, plus one figure per metric to actually use.
_sweep_rows = []
for spec in models:
    res = sweep_results[spec.name]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric, color in zip(axes.ravel(), SWEEP_METRICS, SERIES_COLORS):
        _draw_sweep(ax, res, metric, color)
    fig.suptitle(f'K-Fold stability: {spec.name}', fontsize=13,
                 fontweight='bold', color=INK)
    fig.tight_layout()
    save_fig(fig, f"K-Fold_{spec.name.replace(' ', '_')}", subdir="kfold_stability")
    plt.show()
    plt.close(fig)

    if PER_ITEM_FIGURES:
        for metric, color in zip(SWEEP_METRICS, SERIES_COLORS):
            f1, ax1 = plt.subplots(figsize=(6.4, 4.4))
            _draw_sweep(ax1, res, metric, color)
            ax1.set_title(f"{spec.name} — {metric}", fontsize=12,
                          fontweight='bold', color=INK)
            f1.tight_layout()
            save_fig(f1, f"kfold_{_fname(spec.name)}_{metric}",
                     subdir="kfold_stability/per_metric", close=True)

    for metric in SWEEP_METRICS:
        for k, mean, std in zip(fold_range, res[metric]["means"], res[metric]["stds"]):
            _sweep_rows.append({"Model": spec.name, "Metric": metric, "k": k,
                                "mean": mean, "std": std})
register_table("K-fold stability", pd.DataFrame(_sweep_rows))


# =============================================================================
# =============================================================================
#  PART 6 — EVALUATION METRICS
# =============================================================================
# =============================================================================

# =============================================================================
#  TEST-SET EVALUATION
#  evaluate_all() scores every fitted model on the held-out test set and
#  populates RESULTS, which the summary table and every plot below reads from.
# =============================================================================
summary = evaluate_all(all_models, X_te, y_te)

print("\n" + "=" * 78)
print("HELD-OUT TEST SUMMARY  (sorted by ROC-AUC)")
print("=" * 78)
print(summary.round(4).to_string())
print("=" * 78)
register_table("Test summary", summary, index=True)

for spec in all_models:
    print(f"\n{spec.name}: {spec.best_params}")

#  Per-class breakdown for the single best model. The summary table above
#  averages across classes; this is where a model that is excellent overall and
#  useless on the minority class gets caught.
best_spec = max(all_models, key=lambda s: (RESULTS[s.name]["ROC_AUC"]
                                           if np.isfinite(RESULTS[s.name]["ROC_AUC"])
                                           else -np.inf))
print("\n" + "=" * 78)
print(f"PER-CLASS REPORT — {best_spec.name}")
print("=" * 78)
print(classification_report(y_te, best_spec.predict(X_te),
                            labels=list(range(N_CLASSES)),
                            target_names=[str(c) for c in CLASSES],
                            digits=4, zero_division=0))
register_table("Per-class report (best)", pd.DataFrame(classification_report(
    y_te, best_spec.predict(X_te), labels=list(range(N_CLASSES)),
    target_names=[str(c) for c in CLASSES], output_dict=True,
    zero_division=0)).T, index=True)

# ---- Optional: nested CV for the cheap, Grid/RandomizedSearchCV-tuned models
# for spec in (rf, ada, knn):
#     spec.nested_cv(X_tr, y_tr)

# ---- Using one model on its own, without touching the others:
# svc.fit(X_tr, y_tr)
# svc.evaluate(X_te, y_te)
# new_proba = svc.predict_proba(X_te)


# =============================================================================
#  SECTION 7B — HYPERPARAMETER REPORT  (search space vs. chosen value)
# =============================================================================
#  One row per hyperparameter: what was searched, over what range, and what the
#  search chose.
#
#  ---------------------------------------------------------------------------
#  WHY "TRAINING" AND "TESTING" HOLD THE SAME VALUE
#  ---------------------------------------------------------------------------
#  Both columns are reported, as asked, and they are deliberately identical.
#  Hyperparameters are chosen ONCE, by cross-validation inside the training set,
#  and then applied unchanged to the held-out test set. There is no separate
#  "best parameters for the test set", and producing one would mean selecting
#  hyperparameters by test score — which is test-set leakage: the reported test
#  metric would become the maximum over many configurations rather than an
#  estimate of generalisation, and it would not reproduce.
#
#  So the "Testing" column records what was actually APPLIED at test time. The
#  train-vs-test comparison that is worth having lives in the companion table
#  below, which pairs each model's CV score with its training and test scores —
#  the gap between them is the thing the two columns are usually wanted for.
# =============================================================================
def _fmt_value(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return "None" if v is None else str(v)


def _describe_space(space_entry):
    """A search space — dict entry or grid list — as one readable string."""
    if space_entry is None:
        return "not searched (fixed)"
    if isinstance(space_entry, str):
        return space_entry
    if isinstance(space_entry, (list, tuple, set)):
        vals = list(space_entry)
        shown = ", ".join(_fmt_value(v) for v in vals[:6])
        return "{" + shown + "}" if len(vals) <= 6 else \
               "{" + shown + f", ...}}  ({len(vals)} values)"
    return str(space_entry)


def _bare(param):
    """Drop the pipeline prefixes so the report reads as the model's own API."""
    return param.split("__")[-1] if param.startswith("model__") else param


def build_hyperparameter_table(specs):
    rows = []
    for spec in specs:
        space = SEARCH_SPACES.get(spec.name, {})
        best = spec.best_params or {}
        space_bare = {_bare(k): v for k, v in space.items()}
        best_bare = {_bare(k): v for k, v in best.items()}

        missing = sorted(set(best_bare) - set(space_bare))
        if missing:
            warnings.warn(f"{spec.name}: tuned parameter(s) {missing} are absent from "
                          f"SEARCH_SPACES — the registry has drifted from the code.",
                          stacklevel=2)

        for param in sorted(set(space_bare) | set(best_bare)):
            chosen = best_bare.get(param, None)
            rows.append({
                "Model": spec.name,
                "Search strategy": SEARCH_STRATEGY.get(spec.name, ""),
                "Hyperparameter": param,
                "Search Space or Range": _describe_space(space_bare.get(param)),
                "Best Parameter (Training)": _fmt_value(chosen) if param in best_bare else "not selected",
                "Best Parameter (Testing)": _fmt_value(chosen) if param in best_bare else "not selected",
            })
    return pd.DataFrame(rows)


hyperparam_df = build_hyperparameter_table(models)
print("\n" + "=" * 78)
print("HYPERPARAMETER SEARCH SPACE AND SELECTED VALUES")
print("  Training and Testing columns are identical by construction: selection")
print("  happens once on the training folds, and tuning on test would be leakage.")
print("=" * 78)
print(hyperparam_df.to_string(index=False))
register_table("Hyperparameters", hyperparam_df)

#  Companion: the model-level view, where train and test genuinely differ.
_model_rows = []
for spec in all_models:
    tr = compute_metrics(y_tr, spec.predict(X_tr), spec.predict_proba(X_tr))
    te = RESULTS[spec.name]
    _model_rows.append({
        "Model": spec.name,
        "Search strategy": SEARCH_STRATEGY.get(spec.name, "ensemble"),
        f"CV {PRIMARY_METRIC} (training folds)": spec.cv_score,
        f"Train {PRIMARY_METRIC}": tr["ROC_AUC"], f"Test {PRIMARY_METRIC}": te["ROC_AUC"],
        "Train F1": tr["F1"], "Test F1": te["F1"],
        "Train MCC": tr["MCC"], "Test MCC": te["MCC"],
        "Overfitting gap (train-test AUC)": tr["ROC_AUC"] - te["ROC_AUC"],
        "Fit time (s)": spec.fit_time_s,
        "Best parameters": spec.best_params,
    })
model_summary_df = pd.DataFrame(_model_rows).sort_values(f"Test {PRIMARY_METRIC}", ascending=False)
print("\n" + "=" * 78)
print("MODEL-LEVEL SUMMARY — where training and testing really do differ")
print("=" * 78)
print(model_summary_df.drop(columns=["Best parameters"]).round(4).to_string(index=False))
register_table("Model summary", model_summary_df)


# =============================================================================
#  PLOTTING CONVENTIONS FOR PART 6
# =============================================================================
#  Categorical hues are assigned in a fixed validated order and never cycled: a
#  9th generated hue is not reliably distinguishable, least of all for a
#  colour-blind reader. With 20 models in `all_models`, every MULTI-MODEL
#  overlay therefore shows the top ROC_MAX_MODELS by test AUC and points at the
#  summary table for the rest — which is also the only way an ROC overlay stays
#  readable. Single-model figures use one accent colour, since there is no
#  identity to encode.
# =============================================================================
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from sklearn.model_selection import learning_curve

plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'legend.frameon'   : False,
    'figure.dpi'       : 150,
})

ROC_MAX_MODELS = 8
ACCENT = SERIES_COLORS[0]


def top_models(specs=None, n=None, metric="ROC_AUC"):
    """The n best fitted models by a test metric, best first."""
    n = ROC_MAX_MODELS if n is None else n
    specs = specs if specs is not None else all_models
    scored = [(s, RESULTS[s.name].get(metric, float("nan"))) for s in specs
              if s.name in RESULTS]
    scored = [(s, v) for s, v in scored if np.isfinite(v)]
    return [s for s, _ in sorted(scored, key=lambda t: t[1], reverse=True)[:n]]


def comparison_colors(specs):
    """
    Fixed-slot hue assignment for one comparison figure.

    The palette holds 8 hues, and every overlay caller feeds this from
    top_models(n=ROC_MAX_MODELS), which caps at exactly that. A caller that
    passes the whole roster instead — 9 base models, or all_models with the
    ensembles — used to run off the end of the list and raise a bare
    IndexError from inside a dict comprehension, naming neither the palette
    nor the caller.

    Hues now repeat rather than crash, which is right for the per-model
    figures (each is its own figure, so a repeat is invisible) and wrong for
    an overlay, where two series would share a colour. So the wrap is
    reported once instead of happening silently: an overlay caller that
    trips it should be passing top_models() and is not.
    """
    if len(specs) > len(SERIES_COLORS) and not getattr(
            comparison_colors, "_warned", False):
        comparison_colors._warned = True
        print(f"Note: {len(specs)} models share a {len(SERIES_COLORS)}-hue "
              f"palette, so hues repeat. Fine for one-figure-per-model output; "
              f"for an overlay, pass top_models(n={len(SERIES_COLORS)}).")
    return {s.name: SERIES_COLORS[i % len(SERIES_COLORS)]
            for i, s in enumerate(specs)}


# =============================================================================
#  SECTION 6 — LEARNING CURVES  (per already-tuned model)
# =============================================================================
#  No hyperparameter search happens here — this only asks: given the
#  hyperparameters already found, how does train vs. validation score evolve
#  with training-set size?
#
#  REGRESSION -> CLASSIFICATION: the curve is plotted in the PRIMARY METRIC
#  (ROC-AUC by default) instead of R2, folds are stratified, and the chance
#  baseline is drawn at 0.5 rather than 0 — on a classification curve the
#  meaningful floor is random guessing, not "explains no variance".
#
#  sklearn's learning_curve() clones and refits an estimator internally, which
#  works directly for every sklearn-compatible model here (rf, svc, xgboost,
#  lgbm, ada, knn). It CANNOT clone a Keras model, so mlp / cnn_lstm /
#  seq_logit fall back to a manual sweep built on spec.refit_on(), which reuses
#  the same already-tuned hyperparameters without repeating the search.
# =============================================================================
train_sizes_pct = np.linspace(0.1, 1.0, LC_SIZES)
lc_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

#  A learning curve's smallest slice must still contain every class (and, for
#  KNN, at least n_neighbors rows). Below that, sklearn raises rather than
#  returning a low score, which would abort the whole sweep.
LC_MIN_TRAIN = max(2 * N_CLASSES, 20)


def get_learning_curve(spec, X, y, train_sizes=None, cv=None):
    """
    Unified learning-curve computation across all 9 models.
    Returns (abs_train_sizes, train_mean, train_std, val_mean, val_std) in the
    primary metric, or None if the model cannot produce one.
    """
    train_sizes = train_sizes_pct if train_sizes is None else train_sizes
    cv = lc_cv if cv is None else cv
    if hasattr(spec.model, "get_params"):     # sklearn-compatible estimator
        try:
            sizes, tr_scores, va_scores = learning_curve(
                spec.model, X, y, train_sizes=train_sizes, cv=cv, n_jobs=-1,
                scoring=SCORING, error_score=np.nan)
            return (sizes, np.nanmean(tr_scores, axis=1), np.nanstd(tr_scores, axis=1),
                    np.nanmean(va_scores, axis=1), np.nanstd(va_scores, axis=1))
        except Exception as exc:              # e.g. KNN with k > smallest slice
            print(f"  learning curve unavailable for {spec.name}: {exc}")
            return None

    # Keras fallback: manual per-size, per-fold refit via spec.refit_on()
    min_train_len = min(len(tr) for tr, _ in cv.split(X, y))
    sizes = np.unique((np.asarray(train_sizes) * min_train_len).astype(int))
    sizes = sizes[sizes >= LC_MIN_TRAIN]
    if len(sizes) == 0:
        return None

    try:
        tr_mean, tr_std, va_mean, va_std = [], [], [], []
        for size in sizes:
            tr_s, va_s = [], []
            for tr_idx, va_idx in cv.split(X, y):
                if len(tr_idx) - size < N_CLASSES:
                    # Two ways the largest train_sizes entry breaks a stratified
                    # subsample, both hit in practice:
                    #   size == len(tr_idx)      -> train_test_split rejects
                    #                               train_size == n_samples;
                    #   len(tr_idx) - size < K   -> the leftover is too small to
                    #                               hold one row per class, and
                    #                               stratify raises.
                    # Both mean "this is effectively the whole fold", so use it
                    # as-is. The plotted x value can then understate the rows
                    # actually used by up to K-1, on the smallest fold only.
                    sub = tr_idx
                else:
                    # STRATIFIED subsample, not a plain random choice: an
                    # unstratified slice of an imbalanced training set can
                    # contain a single class, which makes AUC undefined and the
                    # curve jagged for the wrong reason.
                    sub, _ = train_test_split(tr_idx, train_size=int(size),
                                              random_state=SEED, stratify=y[tr_idx])
                proba_fn = spec.refit_on(X[sub], y[sub])
                tr_s.append(primary_score(y[sub], proba_fn(X[sub])))
                va_s.append(primary_score(y[va_idx], proba_fn(X[va_idx])))
            tr_mean.append(np.nanmean(tr_s)); tr_std.append(np.nanstd(tr_s))
            va_mean.append(np.nanmean(va_s)); va_std.append(np.nanstd(va_s))
    except Exception as exc:
        # Fail soft, exactly like the sklearn branch above: a learning curve is
        # a diagnostic, and losing one model's curve must not abort the
        # evaluation of the other nineteen.
        print(f"  learning curve unavailable for {spec.name}: "
              f"{type(exc).__name__}: {exc}")
        return None

    return sizes, np.array(tr_mean), np.array(tr_std), np.array(va_mean), np.array(va_std)


# ── PER-MODEL LEARNING CURVES ────────────────────────────────────────────────
lc_results = {}   # name -> (sizes, train_mean, train_std, val_mean, val_std)

for spec in models:
    curve = get_learning_curve(spec, X_tr, y_tr)
    if curve is None:
        continue
    sizes, train_mean, train_std, val_mean, val_std = curve
    lc_results[spec.name] = curve

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_mean, color=ACCENT, linewidth=2, linestyle='-',
            marker='o', markersize=5, label='Training score')
    ax.plot(sizes, val_mean, color=SERIES_COLORS[1], linewidth=2, linestyle='--',
            marker='s', markersize=5, label='Validation score')
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.12, color=ACCENT, linewidth=0)
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.12, color=SERIES_COLORS[1], linewidth=0)

    final_gap = train_mean[-1] - val_mean[-1]
    ax.annotate(f'Gap: {final_gap:.3f}', xy=(sizes[-1], val_mean[-1]),
                xytext=(-60, -20), textcoords='offset points',
                fontsize=9, color=INK_SOFT,
                arrowprops=dict(arrowstyle='->', color=INK_SOFT, lw=0.8))

    # 0.5 = chance for AUC. A validation curve hugging this line means the
    # model has found no signal, which is a different failure from overfitting.
    if PRIMARY_METRIC in ("ROC_AUC", "PR_AUC"):
        ax.axhline(0.5, color=INK_SOFT, linewidth=0.8, linestyle=':', alpha=0.7)
        ax.annotate('chance', xy=(sizes[0], 0.5), xytext=(0, 4),
                    textcoords='offset points', fontsize=8, color=INK_SOFT)

    _style(ax, f"Learning curve — {spec.name}", 'Training samples',
           f'{PRIMARY_METRIC.replace("_", "-")} (CV=5)')
    ax.set_ylim(0.35, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    save_fig(plt.gcf(), f"learning_curve_{_fname(spec.name)}", subdir="learning_curves")
    plt.show()
    plt.close(fig)


# ── COMBINED VALIDATION CURVE ───────────────────────────────────────────────
lc_specs = [s for s in top_models(models, ROC_MAX_MODELS) if s.name in lc_results]
if lc_specs:
    lc_colors = comparison_colors(lc_specs)
    fig, ax = plt.subplots(figsize=(10, 6))
    for spec in lc_specs:
        sizes, _, _, val_mean, val_std = lc_results[spec.name]
        ax.plot(sizes, val_mean, color=lc_colors[spec.name], linewidth=2,
                marker='o', markersize=4, label=spec.name)
        ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.08, color=lc_colors[spec.name], linewidth=0)
    ax.axhline(0.5, color=INK_SOFT, linewidth=0.8, linestyle=':', alpha=0.7)
    _style(ax, f"Validation score comparison (top {len(lc_specs)} by test AUC)",
           "Training samples", f'{PRIMARY_METRIC.replace("_", "-")} (CV=5)')
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    plt.tight_layout()
    save_fig(plt.gcf(), "validation_curve_comparison", subdir="learning_curves")
    plt.show()
    plt.close(fig)


# ── TEST-METRIC BAR CHART ────────────────────────────────────────────────────
#  Two metrics side by side rather than one. A single ranking chart is where
#  classification write-ups mislead: ROC-AUC ranks models on their ability to
#  rank, MCC ranks them on the decisions they actually make at the deployed
#  threshold, and on an imbalanced problem the two orders differ.
fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, 0.34 * len(all_models))))
for ax, metric, color in [(axes[0], "ROC_AUC", SERIES_COLORS[0]),
                          (axes[1], "MCC",     SERIES_COLORS[1])]:
    ranked = sorted(all_models, key=lambda s: RESULTS[s.name][metric])
    names  = [s.name for s in ranked]
    scores = [RESULTS[s.name][metric] for s in ranked]
    bars = ax.barh(names, scores, color=color, height=0.62)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.006, bar.get_y() + bar.get_height() / 2,
                f'{score:.3f}', va='center', fontsize=8, color=INK_SOFT)
    _style(ax, f"Held-out test {metric.replace('_', '-')}",
           metric.replace('_', '-'), None)
    ax.set_xlim(min(0, min(scores) * 1.1), max(scores) * 1.16)
    ax.grid(axis='y', visible=False)
    ax.tick_params(labelsize=8)
fig.suptitle("Model comparison — ranking quality (AUC) vs decision quality (MCC)",
             fontsize=13, fontweight='bold', color=INK)
plt.tight_layout()
save_fig(plt.gcf(), "test_metric_comparison", subdir="model_comparison")
plt.show()
plt.close(fig)


# =============================================================================
#  SECTION 7 — TRAIN vs TEST METRICS TABLE  (per already-tuned model)
# =============================================================================
#  Reuses compute_metrics() from Part 1 so these numbers are defined
#  identically to everywhere else in the pipeline.
# =============================================================================
METRIC_COLS = ["Accuracy", "Balanced_Acc", "Precision", "Recall", "F1",
               "MCC", "Kappa", "ROC_AUC", "PR_AUC", "LogLoss", "Brier"]


def build_metrics_df(models, X, y):
    """Predict on (X, y) for each already-fitted model, one row per model."""
    rows = []
    for spec in models:
        proba = spec.predict_proba(X)
        m = compute_metrics(y, labels_from_proba(proba), proba)
        rows.append({"Model": spec.name, **m})
    return pd.DataFrame(rows)[["Model"] + METRIC_COLS]


results_df_Train = build_metrics_df(all_models, X_tr, y_tr)
results_df_Test  = build_metrics_df(all_models, X_te, y_te)

print("\n" + "=" * 78)
print("TRAINING SET METRICS")
print("=" * 78)
print(results_df_Train.round(4).to_string(index=False))
register_table("Train metrics", results_df_Train)

print("\n" + "=" * 78)
print("TEST SET METRICS")
print("=" * 78)
print(results_df_Test.round(4).to_string(index=False))
register_table("Test metrics", results_df_Test)

#  A model whose train AUC is much higher than its test AUC is fitting noise —
#  the same gap the learning curves visualize directly. On classification watch
#  the LogLoss gap too: a model can hold its AUC while its probabilities become
#  wildly overconfident, which only log-loss and Brier reveal.
train_test_gap = results_df_Train.merge(
    results_df_Test, on="Model", suffixes=(" (Train)", " (Test)"))
train_test_gap["AUC Gap (Train-Test)"] = (
    train_test_gap["ROC_AUC (Train)"] - train_test_gap["ROC_AUC (Test)"])
train_test_gap["LogLoss Gap (Test-Train)"] = (
    train_test_gap["LogLoss (Test)"] - train_test_gap["LogLoss (Train)"])
train_test_gap = train_test_gap.sort_values("AUC Gap (Train-Test)", ascending=False)

print("\n" + "=" * 78)
print("TRAIN vs TEST  (sorted by overfitting gap, largest first)")
print("=" * 78)
print(train_test_gap[["Model", "ROC_AUC (Train)", "ROC_AUC (Test)",
                      "AUC Gap (Train-Test)", "LogLoss (Train)", "LogLoss (Test)",
                      "LogLoss Gap (Test-Train)"]].round(4).to_string(index=False))
register_table("Train vs test gap", train_test_gap)


# =============================================================================
#  SECTION 8 — CONFUSION MATRICES  (per already-tuned model)
# =============================================================================
#  THE structural replacement for the regression file's actual-vs-predicted
#  scatter plots. Same role — "where exactly is this model wrong?" — in the
#  form a categorical target admits. The scatter's diagonal becomes the
#  matrix's diagonal, and the off-diagonal cells name the specific confusions
#  instead of showing an undifferentiated cloud of residuals.
#
#  TWO PANELS, ALWAYS, and the reason matters:
#    * COUNTS answer "how many rows land in each cell", which is what you need
#      to judge whether a cell is even worth acting on (3 errors out of 5000 is
#      not a pattern).
#    * ROW-NORMALIZED (recall per true class) answers "what fraction of each
#      TRUE class does the model recover". On an imbalanced problem the counts
#      panel is dominated by the majority class and a total failure on the
#      minority class is almost invisible in it; normalizing by row is what
#      exposes it. Rows sum to 100%.
#  Reading only the counts panel is the single most common way an imbalanced
#  classifier gets declared good.
#
#  Colour is a SEQUENTIAL single-hue ramp, light -> dark, because the cells
#  encode magnitude. A rainbow map ('viridis', 'jet') would force the reader to
#  consult the colorbar to order two cells; annotations carry the exact values
#  regardless, and the text flips to white on dark cells for contrast.
# =============================================================================
CLASS_NAMES = [str(c) for c in CLASSES]


def _draw_cm(ax, M, title, fmt, vmax, cbar_label=None):
    im = ax.imshow(M, cmap=SEQ_BLUE, vmin=0, vmax=vmax)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=10, color=INK_SOFT)
    ax.set_ylabel("True label", fontsize=10, color=INK_SOFT)
    ax.set_title(title, fontsize=11, fontweight='bold', color=INK)
    # Thin white gutters between cells, so adjacent fills never touch.
    ax.set_xticks(np.arange(-0.5, N_CLASSES, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_CLASSES, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.grid(which='major', visible=False)
    ax.tick_params(which='minor', length=0)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, format(M[i, j], fmt), ha='center', va='center',
                    fontsize=10 if N_CLASSES <= 6 else 8,
                    color='white' if M[i, j] > 0.62 * vmax else INK)
    return im


def plot_confusion_matrices(specs, X, y, dataset_label, threshold=None):
    """Counts + row-normalized confusion matrix, one figure per model."""
    for spec in specs:
        proba = spec.predict_proba(X)
        pred = labels_from_proba(proba, threshold=threshold)
        m = compute_metrics(y, pred, proba)

        cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
        with np.errstate(invalid='ignore'):
            cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        fig, axes = plt.subplots(1, 2, figsize=(4.6 + 1.3 * N_CLASSES, 4.8))
        _draw_cm(axes[0], cm, "Counts", "d", max(cm.max(), 1))
        _draw_cm(axes[1], cm_norm * 100, "Row-normalized (recall %)", ".1f", 100.0)

        fig.suptitle(f"Confusion matrix — {spec.name}  ({dataset_label})",
                     fontsize=13, fontweight='bold', color=INK)
        note = (f"ACC {m['Accuracy']:.3f}   BAL-ACC {m['Balanced_Acc']:.3f}   "
                f"F1 {m['F1']:.3f}   MCC {m['MCC']:.3f}   AUC {m['ROC_AUC']:.3f}")
        if threshold is not None:
            note += f"   (threshold {threshold:.3f})"
        fig.text(0.5, 0.01, note, ha='center', fontsize=9, color=INK_SOFT)

        plt.tight_layout(rect=(0, 0.03, 1, 0.96))
        suffix = "" if threshold is None else "_tuned_threshold"
        save_fig(plt.gcf(), f"confusion_matrix_{_fname(spec.name)}_{dataset_label}{suffix}", subdir="confusion_matrices")
        plt.show()
        plt.close(fig)


def plot_confusion_grid(specs, X, y, dataset_label, ncols=4):
    """All models' row-normalized matrices in one figure, for scanning."""
    n = len(specs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, spec in zip(axes, specs):
        pred = spec.predict(X)
        cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
        with np.errstate(invalid='ignore'):
            cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True)) * 100
        _draw_cm(ax, cm_norm, spec.name, ".0f", 100.0)
        ax.set_xlabel(""); ax.set_ylabel("")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Row-normalized confusion matrices — {dataset_label} set (recall %)",
                 fontsize=14, fontweight='bold', color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(plt.gcf(), f"confusion_matrix_grid_{dataset_label}", subdir="confusion_matrices")
    plt.show()
    plt.close(fig)


plot_confusion_matrices(all_models, X_te, y_te, "test")
plot_confusion_matrices(all_models, X_tr, y_tr, "train")
plot_confusion_grid(all_models, X_te, y_te, "test")

#  Misclassification table: the confusion matrices in list form, so the worst
#  confusions can be sorted and quoted rather than eyeballed off a heatmap.
print("\n" + "=" * 78)
print("MOST FREQUENT CONFUSIONS — test set, per model")
print("=" * 78)
conf_rows = []
for spec in all_models:
    cm = confusion_matrix(y_te, spec.predict(X_te), labels=list(range(N_CLASSES)))
    off = [(CLASSES[i], CLASSES[j], int(cm[i, j]))
           for i in range(N_CLASSES) for j in range(N_CLASSES)
           if i != j and cm[i, j] > 0]
    off.sort(key=lambda t: t[2], reverse=True)
    worst = off[0] if off else (None, None, 0)
    conf_rows.append({"Model": spec.name, "Errors": int(cm.sum() - np.trace(cm)),
                      "Worst confusion": (f"{worst[0]} -> {worst[1]}" if worst[0] is not None
                                          else "none"),
                      "n": worst[2]})
print(pd.DataFrame(conf_rows).to_string(index=False))
register_table("Worst confusions", pd.DataFrame(conf_rows))


# =============================================================================
#  SECTION 8B — ROC CURVES
# =============================================================================
#  The ROC curve traces the true-positive rate against the false-positive rate
#  as the decision threshold sweeps from 1 to 0, so it shows the model at EVERY
#  operating point rather than only at 0.5. Its area (AUC) equals the
#  probability that a randomly chosen positive is ranked above a randomly
#  chosen negative — a threshold-free, prevalence-independent summary of
#  ranking quality.
#
#  READ IT WITH THE PR CURVE (Section 8C), NOT ALONE. ROC's x-axis is
#  FP / (FP + TN), and on an imbalanced problem TN is enormous, so a model can
#  produce far more false positives than true positives while the curve still
#  hugs the top-left corner. That is the classic "0.95 AUC, useless in
#  production" result. Precision has no TN in its denominator and collapses
#  immediately, which is why both are here.
#
#  MULTICLASS: there is no single ROC curve for K > 2. The standard treatment,
#  used here, is one-vs-rest — one curve per class — plus two aggregates:
#    micro-average: pool every (row, class) decision into one binary problem.
#                   Dominated by the frequent classes.
#    macro-average: average the per-class curves with equal weight. A rare
#                   class counts as much as a common one, so this is the one
#                   that exposes a model that ignores a minority class.
# =============================================================================
def roc_points(y_true, proba, class_idx=None):
    """(fpr, tpr, auc) for one binary problem or one one-vs-rest slice."""
    y_true = np.asarray(y_true).astype(int)
    if class_idx is None:
        y_bin, score = (y_true == POS_LABEL).astype(int), _proba_pos(proba)
    else:
        y_bin, score = (y_true == class_idx).astype(int), np.asarray(proba)[:, class_idx]
    if len(np.unique(y_bin)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_bin, score)
    return fpr, tpr, float(auc(fpr, tpr))


def macro_roc(y_true, proba):
    """Macro-average one-vs-rest ROC: interpolate every class onto a shared
    FPR grid, then average. (Simply averaging the raw curves is wrong — they
    are sampled at different FPR values.)"""
    curves = [roc_points(y_true, proba, c) for c in range(N_CLASSES)]
    curves = [c for c in curves if c is not None]
    if not curves:
        return None
    grid = np.linspace(0, 1, 200)
    mean_tpr = np.mean([np.interp(grid, f, t) for f, t, _ in curves], axis=0)
    return grid, mean_tpr, float(auc(grid, mean_tpr))


def micro_roc(y_true, proba):
    """Micro-average one-vs-rest ROC: every (row, class) decision pooled."""
    Y = one_hot(np.asarray(y_true).astype(int)).ravel()
    S = np.asarray(proba).ravel()
    if len(np.unique(Y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(Y, S)
    return fpr, tpr, float(auc(fpr, tpr))


def _roc_axes(ax, title):
    # The chance diagonal is the reference the curve is read against: a model
    # on this line ranks no better than a coin flip.
    ax.plot([0, 1], [0, 1], linestyle=':', linewidth=1.2, color=INK_SOFT)
    ax.annotate('chance', xy=(0.62, 0.58), fontsize=8, color=INK_SOFT, rotation=38)
    _style(ax, title, "False positive rate  (FP / (FP + TN))",
           "True positive rate  (recall)")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
    ax.set_aspect('equal', adjustable='box')


def plot_roc_single(spec, X, y, dataset_label="test"):
    """One model's ROC. Binary: one curve. Multiclass: per-class + averages."""
    proba = spec.predict_proba(X)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    if IS_BINARY:
        pts = roc_points(y, proba)
        if pts is None:
            plt.close(fig); return
        fpr, tpr, a = pts
        ax.plot(fpr, tpr, color=ACCENT, linewidth=2,
                label=f"{CLASSES[POS_LABEL]} vs rest  (AUC = {a:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.10, color=ACCENT, linewidth=0)

        # Mark the operating point the model actually ships with (argmax, i.e.
        # threshold 0.5) — the single point on this curve the confusion matrix
        # in Section 8 corresponds to.
        pred = labels_from_proba(proba)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[1 - POS_LABEL, POS_LABEL]).ravel()
        ax.plot(fp / max(fp + tn, 1), tp / max(tp + fn, 1), marker='o', markersize=9,
                color=INK, markerfacecolor='white', markeredgewidth=2, linestyle='none',
                label="operating point (threshold 0.5)")
    else:
        for c in range(min(N_CLASSES, len(SERIES_COLORS))):
            pts = roc_points(y, proba, c)
            if pts is None:
                continue
            fpr, tpr, a = pts
            ax.plot(fpr, tpr, color=SERIES_COLORS[c], linewidth=1.8,
                    label=f"{CLASSES[c]}  (AUC = {a:.3f})")
        for fn_, style, label in [(micro_roc, '--', 'micro-average'),
                                  (macro_roc, '-.', 'macro-average')]:
            agg = fn_(y, proba)
            if agg is not None:
                f, t, a = agg
                ax.plot(f, t, linestyle=style, linewidth=2.4, color=INK,
                        label=f"{label}  (AUC = {a:.3f})")

    _roc_axes(ax, f"ROC — {spec.name}  ({dataset_label})")
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    save_fig(plt.gcf(), f"roc_{_fname(spec.name)}_{dataset_label}", subdir="roc_curves")
    plt.show()
    plt.close(fig)


def plot_roc_comparison(specs, X, y, dataset_label="test"):
    """All (top-N) models on one axes. Binary: their ROC. Multiclass: their
    macro-average OvR ROC, which is the like-for-like summary curve."""
    colors = comparison_colors(specs)
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    for spec in specs:
        proba = spec.predict_proba(X)
        pts = roc_points(y, proba) if IS_BINARY else macro_roc(y, proba)
        if pts is None:
            continue
        fpr, tpr, a = pts
        ax.plot(fpr, tpr, color=colors[spec.name], linewidth=2,
                label=f"{spec.name}  ({a:.4f})")
    _roc_axes(ax, f"ROC comparison — top {len(specs)} models ({dataset_label})"
                  + ("" if IS_BINARY else ", macro-average OvR"))
    leg = ax.legend(loc='lower right', fontsize=8, title="model (AUC)", frameon=True,
                    framealpha=0.95, edgecolor=GRID_COLOR)
    leg.get_title().set_fontsize(8)
    plt.tight_layout()
    save_fig(plt.gcf(), f"roc_comparison_{dataset_label}", subdir="roc_curves")
    plt.show()
    plt.close(fig)


roc_specs = top_models(all_models, ROC_MAX_MODELS)
for spec in roc_specs:
    plot_roc_single(spec, X_te, y_te, "test")
plot_roc_comparison(roc_specs, X_te, y_te, "test")
#  Train-set ROC for the best model only: the gap between this and the test
#  curve is the overfitting gap, in the same units as the headline number.
plot_roc_single(best_spec, X_tr, y_tr, "train")


# =============================================================================
#  SECTION 8C — PRECISION-RECALL CURVES
# =============================================================================
#  The imbalance-honest companion to ROC. Precision = TP / (TP + FP) has no
#  true-negative term, so every false positive costs it directly. The baseline
#  is not a diagonal but a HORIZONTAL line at the positive class's prevalence —
#  what a model that guesses at random achieves — so on a 5%-positive problem a
#  PR curve sitting at 0.3 precision is still six times better than chance,
#  while the same model's ROC curve might look near-perfect.
#
#  Average precision (PR-AUC) is reported rather than the trapezoidal area:
#  trapezoidal interpolation between PR points is optimistically biased because
#  the curve is not monotone.
# =============================================================================
def plot_pr_comparison(specs, X, y, dataset_label="test"):
    colors = comparison_colors(specs)
    fig, ax = plt.subplots(figsize=(7.2, 6.6))

    for spec in specs:
        proba = spec.predict_proba(X)
        if IS_BINARY:
            y_bin, score = (np.asarray(y) == POS_LABEL).astype(int), _proba_pos(proba)
            prec, rec, _ = precision_recall_curve(y_bin, score)
            ap = average_precision_score(y_bin, score)
            ax.plot(rec, prec, color=colors[spec.name], linewidth=2,
                    label=f"{spec.name}  ({ap:.4f})")
        else:
            # Micro-average PR: every (row, class) decision pooled, which is
            # the standard single-curve summary for multiclass.
            Y = one_hot(np.asarray(y).astype(int)).ravel()
            S = np.asarray(proba).ravel()
            prec, rec, _ = precision_recall_curve(Y, S)
            ap = average_precision_score(Y, S)
            ax.plot(rec, prec, color=colors[spec.name], linewidth=2,
                    label=f"{spec.name}  ({ap:.4f})")

    baseline = (float((np.asarray(y) == POS_LABEL).mean()) if IS_BINARY
                else 1.0 / N_CLASSES)
    ax.axhline(baseline, linestyle=':', linewidth=1.2, color=INK_SOFT)
    ax.annotate(f"chance = prevalence ({baseline:.3f})", xy=(0.02, baseline),
                xytext=(0, 5), textcoords='offset points', fontsize=8, color=INK_SOFT)

    _style(ax, f"Precision-recall — top {len(specs)} models ({dataset_label})"
               + ("" if IS_BINARY else ", micro-average"),
           "Recall  (TP / (TP + FN))", "Precision  (TP / (TP + FP))")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1.02)
    ax.set_aspect('equal', adjustable='box')
    leg = ax.legend(loc='lower left', fontsize=8, title="model (avg. precision)",
                    frameon=True, framealpha=0.95, edgecolor=GRID_COLOR)
    leg.get_title().set_fontsize(8)
    plt.tight_layout()
    save_fig(plt.gcf(), f"precision_recall_comparison_{dataset_label}", subdir="pr_curves")
    plt.show()
    plt.close(fig)


plot_pr_comparison(roc_specs, X_te, y_te, "test")


# =============================================================================
#  SECTION 8D — CALIBRATION (RELIABILITY) CURVES
# =============================================================================
#  ROC and PR score the RANKING. Calibration scores the NUMBERS: of the rows a
#  model called 70% likely, did about 70% actually turn out positive? A model
#  can have a perfect AUC of 1.0 and still be badly calibrated — ranking is
#  invariant to any monotone squashing of the probabilities, and AUC cannot
#  see it. It matters the moment a probability is used for anything other than
#  argmax: expected-cost decisions, triage thresholds, or anything a human
#  reads as a confidence.
#
#  This is the closest counterpart to the regression file's residual analysis:
#  both ask whether the model's error is systematic rather than merely large.
#
#  Perfect calibration is the diagonal. Below it = overconfident (the usual
#  direction for boosted trees and for any model tuned on AUC alone); above it
#  = underconfident (common for bagged models like random forests, whose vote
#  fractions are pulled toward the middle).
#
#  ECE (expected calibration error) is the bin-weighted mean gap from the
#  diagonal — one number for "how far off are the probabilities", reported
#  alongside Brier, which mixes calibration together with discrimination.
# =============================================================================
from sklearn.calibration import calibration_curve

N_CALIB_BINS = 10


def expected_calibration_error(y_true, proba, n_bins=None):
    """
    Binary: gap between predicted P(positive) and observed frequency.
    Multiclass: the standard confidence-vs-accuracy form — bin rows by the
    model's max probability (its confidence in whatever it predicted) and
    compare with how often that prediction was right.
    """
    n_bins = N_CALIB_BINS if n_bins is None else n_bins
    y_true = np.asarray(y_true).astype(int)
    P = np.asarray(proba)
    if IS_BINARY:
        conf = _proba_pos(P)
        correct = (y_true == POS_LABEL).astype(float)
    else:
        conf = P.max(axis=1)
        correct = (P.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        ece += (sel.sum() / len(conf)) * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def plot_calibration(specs, X, y, dataset_label="test"):
    colors = comparison_colors(specs)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 8.4), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.15]})
    ax, ax_hist = axes

    ax.plot([0, 1], [0, 1], linestyle=':', linewidth=1.2, color=INK_SOFT)
    ax.annotate('perfectly calibrated', xy=(0.55, 0.5), fontsize=8,
                color=INK_SOFT, rotation=38)

    rows = []
    for spec in specs:
        P = spec.predict_proba(X)
        if IS_BINARY:
            conf = _proba_pos(P)
            y_bin = (np.asarray(y) == POS_LABEL).astype(int)
        else:
            conf = P.max(axis=1)
            y_bin = (P.argmax(axis=1) == np.asarray(y)).astype(int)
        try:
            # strategy='quantile' puts an equal NUMBER of rows in each bin.
            # With 'uniform', models that pile every prediction near 0 or 1
            # (most boosted trees) leave the middle bins nearly empty, and
            # those near-empty bins dominate the visual impression.
            frac_pos, mean_pred = calibration_curve(y_bin, conf, n_bins=N_CALIB_BINS,
                                                    strategy="quantile")
        except ValueError:
            continue
        ax.plot(mean_pred, frac_pos, marker='o', markersize=6, linewidth=2,
                color=colors[spec.name], label=spec.name)
        ax_hist.hist(conf, bins=25, histtype="step", linewidth=1.6,
                     color=colors[spec.name])
        rows.append({"Model": spec.name,
                     "ECE": expected_calibration_error(y, P),
                     "Brier": brier(y, P),
                     "LogLoss": float(log_loss(y, P, labels=list(range(N_CLASSES))))})

    _style(ax, f"Calibration — top {len(specs)} models ({dataset_label})", None,
           "Observed frequency" if IS_BINARY else "Accuracy within bin")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='upper left', fontsize=8)
    _style(ax_hist, None,
           "Predicted P(%s)" % CLASSES[POS_LABEL] if IS_BINARY else "Predicted confidence (max probability)",
           "Rows")
    ax_hist.set_yscale('log')

    plt.tight_layout()
    save_fig(plt.gcf(), f"calibration_{dataset_label}", subdir="calibration")
    plt.show()
    plt.close(fig)

    calib_df = pd.DataFrame(rows).sort_values("ECE")
    print("\n" + "=" * 78)
    print("CALIBRATION QUALITY  (lower is better on all three)")
    print("=" * 78)
    print(calib_df.round(4).to_string(index=False))
    register_table(f"Calibration {dataset_label}", calib_df)
    return calib_df


calibration_df = plot_calibration(roc_specs, X_te, y_te, "test")


# =============================================================================
#  SECTION 8E — DECISION-THRESHOLD ANALYSIS  (binary only)
# =============================================================================
#  Every threshold metric reported so far (accuracy, precision, recall, F1,
#  MCC, the confusion matrices) is the model AT ONE CUTOFF — argmax, i.e. 0.5.
#  That cutoff is a default, not a result: it is optimal only when the classes
#  are balanced and the two error types cost the same. Neither is usually true.
#
#  This section makes the choice explicit. It sweeps the threshold over the
#  full range and reports three candidate operating points:
#
#    0.5          the default, for reference.
#    Youden's J   argmax(TPR - FPR). The point furthest from the chance
#                 diagonal on the ROC curve; treats the two error types as
#                 equally costly, which makes it the natural threshold-free
#                 default when you have no cost information.
#    Best F1      argmax of F1. Favours the positive class; appropriate when
#                 the positives are the rare, expensive-to-miss ones.
#
#  THE HONEST-USE RULE: a threshold tuned on the TEST set and then reported on
#  that same test set is optimistically biased, exactly like tuning
#  hyperparameters on test. What is printed below is diagnostic — to SHIP a
#  threshold, pick it on the training set's cross-validated out-of-fold
#  probabilities (the code is the same, applied to those) and then report it
#  once on test. That caveat is the whole reason this section is separate from
#  the headline table rather than silently replacing it.
# =============================================================================
def threshold_sweep(spec, X, y, n_points=200):
    proba = spec.predict_proba(X)
    score = _proba_pos(proba)
    y_bin = (np.asarray(y) == POS_LABEL).astype(int)

    fpr, tpr, roc_thr = roc_curve(y_bin, score)
    #  sklearn sets roc_curve's first threshold to +inf (the "predict nothing
    #  positive" corner). If that corner happens to maximise TPR - FPR, the raw
    #  value is unusable as a cutoff, so it is clipped into the unit interval.
    youden_thr = float(np.clip(roc_thr[np.argmax(tpr - fpr)], 0.0, 1.0))

    lo, hi = float(score.min()), float(score.max())
    if hi - lo < 1e-6:          # a model that emits one constant probability
        lo, hi = 0.0, 1.0
    grid = np.linspace(max(lo, 1e-6), min(hi, 1 - 1e-6), n_points)
    rows = []
    for t in grid:
        pred = labels_from_proba(proba, threshold=t)
        m = compute_metrics(y, pred, proba)
        rows.append({"threshold": t, **{k: m[k] for k in
                     ("Accuracy", "Balanced_Acc", "Precision", "Recall", "F1", "MCC")}})
    sweep = pd.DataFrame(rows)
    best_f1_thr  = float(sweep.loc[sweep["F1"].idxmax(), "threshold"])
    best_mcc_thr = float(sweep.loc[sweep["MCC"].idxmax(), "threshold"])
    return sweep, {"Youden J": youden_thr, "Best F1": best_f1_thr,
                   "Best MCC": best_mcc_thr, "Default (0.5)": 0.5}


def plot_threshold_analysis(spec, X, y, dataset_label="test"):
    sweep, candidates = threshold_sweep(spec, X, y)
    proba = spec.predict_proba(X)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for i, metric in enumerate(("Precision", "Recall", "F1", "MCC")):
        ax.plot(sweep["threshold"], sweep[metric], linewidth=2,
                color=SERIES_COLORS[i], label=metric)

    for j, (label, thr) in enumerate(candidates.items()):
        ax.axvline(thr, color=INK_SOFT, linestyle=(':' if label == "Default (0.5)" else '--'),
                   linewidth=1.2)
        ax.annotate(f"{label}\n{thr:.3f}", xy=(thr, 1.005), xytext=(0, 2),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7.5, color=INK_SOFT)

    _style(ax, f"Threshold sweep — {spec.name} ({dataset_label})",
           f"Decision threshold on P({CLASSES[POS_LABEL]})", "Metric value")
    ax.set_ylim(0, 1.14)
    ax.legend(loc='lower center', fontsize=9, ncol=4)
    plt.tight_layout()
    save_fig(plt.gcf(), f"threshold_sweep_{_fname(spec.name)}_{dataset_label}", subdir="threshold")
    plt.show()
    plt.close(fig)

    rows = []
    for label, thr in candidates.items():
        m = compute_metrics(y, labels_from_proba(proba, threshold=thr), proba)
        rows.append({"Operating point": label, "threshold": round(thr, 4),
                     **{k: round(m[k], 4) for k in
                        ("Accuracy", "Balanced_Acc", "Precision", "Recall", "F1", "MCC")}})
    table = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(f"OPERATING POINTS — {spec.name} ({dataset_label})")
    print("=" * 78)
    print(table.to_string(index=False))
    register_table(f"Operating points {dataset_label}", table)
    return table, candidates


if IS_BINARY:
    threshold_table, threshold_candidates = plot_threshold_analysis(
        best_spec, X_te, y_te, "test")
    #  The confusion matrix the model would produce at the F1-optimal cutoff —
    #  directly comparable with the 0.5 matrix from Section 8, so the cost of
    #  the default threshold is visible as cells that move.
    plot_confusion_matrices([best_spec], X_te, y_te, "test",
                            threshold=threshold_candidates["Best F1"])
else:
    print("\nThreshold analysis is binary-only; skipped for a "
          f"{N_CLASSES}-class target. The multiclass analogue is per-class "
          "one-vs-rest thresholding or a cost matrix — add it here if your "
          "classes have asymmetric costs.")


# =============================================================================
#  SECTION 13 — TAYLOR DIAGRAM  (on predicted probabilities)
# =============================================================================
#  Compares every model on three statistics at once, in one polar plot:
#     radius   = standard deviation of the model's predicted probabilities
#     azimuth  = Pearson correlation with the observed outcome, as arccos(R)
#     distance = centered RMSE
#     from the reference star
#
#  bound by the law of cosines, which is exactly WHY the geometry works:
#      CRMSE^2 = SD_obs^2 + SD_pred^2 - 2 * SD_obs * SD_pred * R
#
#  REGRESSION -> CLASSIFICATION. A Taylor diagram needs two continuous series.
#  Class labels are not continuous, so feeding it hard predictions would be
#  meaningless. What it is applied to here instead is the PROBABILISTIC
#  FORECAST: the model's predicted P(positive) against the 0/1 outcome
#  indicator — which is standard practice in forecast verification, where
#  Taylor diagrams come from in the first place. For multiclass the one-hot
#  truth and the probability matrix are pooled one-vs-rest.
#
#  How to read it here: a model INSIDE the dotted reference arc under-disperses
#  — its probabilities are squashed toward the base rate, which is what an
#  over-regularised or heavily-bagged model does. Outside it over-disperses:
#  probabilities pushed to 0/1, the signature of an overconfident boosted
#  model. The reference star is the outcome's own variability, sqrt(p(1-p)).
#
#  THE ONE THING THIS PLOT DOES NOT SHOW: BIAS. The identity holds only for the
#  CENTERED RMSE, and MSE_total = CRMSE^2 + Bias^2. A model with a systematic
#  offset (systematically over-predicting the positive rate) can sit next to
#  the star while being consistently wrong, so Bias is reported in the
#  companion table. Read both, and cross-check against the calibration curves
#  in Section 8D, which show the same information conditionally rather than in
#  aggregate.
# =============================================================================
def probability_series(y_true, proba):
    """(observed indicator, predicted probability) as two flat, aligned series."""
    y_true = np.asarray(y_true).astype(int)
    P = np.asarray(proba, dtype=np.float64)
    if IS_BINARY:
        return (y_true == POS_LABEL).astype(np.float64), _proba_pos(P)
    return one_hot(y_true).ravel().astype(np.float64), P.ravel()


def taylor_statistics(y_true, proba):
    """SD / R / centered-RMSE / bias for one model's probabilities."""
    obs, pred = probability_series(y_true, proba)
    sd_obs  = float(np.std(obs, ddof=0))
    sd_pred = float(np.std(pred, ddof=0))
    R       = float(np.corrcoef(obs, pred)[0, 1])
    # max(..., 0) guards against a tiny negative from floating-point round-off
    crmse   = float(np.sqrt(max(sd_obs**2 + sd_pred**2 - 2*sd_obs*sd_pred*R, 0.0)))
    return {"SD": sd_pred, "R": R, "CRMSE": crmse,
            "Bias": float(pred.mean() - obs.mean()), "SD_obs": sd_obs}


def taylor_diagram(entries, sd_obs, title="Taylor Diagram", normalize=True,
                   colors=None, figsize=(9.5, 7.5), savepath=None, zoom=False):
    """
    entries: list of dicts from taylor_statistics(), each with a "name" key.
    normalize=True divides every SD by SD_obs so the reference sits at radius 1.
    zoom=True crops the axes to the region the models actually occupy.
    """
    ref   = 1.0 if normalize else sd_obs
    scale = (1.0 / sd_obs) if normalize else 1.0
    rmax  = max([ref] + [e["SD"] * scale for e in entries]) * 1.35

    #  A standard Taylor diagram is a quarter circle (R from 1 down to 0). If
    #  any model is anticorrelated with the outcome — which for a classifier
    #  means it is worse than chance — arccos(R) exceeds 90 deg and would fall
    #  off the plot, so the arc extends to a half circle instead of silently
    #  dropping that model.
    min_R = min(e["R"] for e in entries)
    theta_max = np.pi if min_R < 0 else np.pi / 2
    if min_R < 0:
        print(f"Note: negative correlation present (min R = {min_R:.3f}); "
              f"extending the diagram to 180 degrees.")

    rmin = 0.0
    if zoom:
        sds = [e["SD"] * scale for e in entries] + [ref]
        theta_max = min(theta_max, float(np.arccos(min_R)) * 1.25 + 0.02)
        rmin = max(0.0, min(sds) - 0.12 * (max(sds) - min(sds) + 1e-6))
        rmax = max(sds) + 0.12 * (max(sds) - min(sds) + 1e-6)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(theta_max))
    ax.set_rlim(rmin, rmax)

    if zoom:
        lo = float(np.cos(theta_max))
        corr_ticks = sorted(set(np.round(np.linspace(lo, 1.0, 8), 3).tolist()))
    else:
        corr_ticks = ([-0.99, -0.9, -0.7, -0.4, 0, 0.4, 0.7, 0.9, 0.95, 0.99]
                      if min_R < 0 else [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    ax.set_thetagrids(np.degrees(np.arccos(np.clip(corr_ticks, -1, 1))),
                      labels=[f"{c:g}" for c in corr_ticks])

    # Dashed arcs of constant centered RMSE = circles centred on the reference.
    tg, rg = np.meshgrid(np.linspace(0, theta_max, 200), np.linspace(rmin, rmax, 200))
    crmse_grid = np.sqrt(ref**2 + rg**2 - 2 * ref * rg * np.cos(tg))
    _lv = np.linspace(float(crmse_grid.min()), float(crmse_grid.max()), 8)[1:-1]
    cs = ax.contour(tg, rg, crmse_grid, levels=_lv,
                    colors=GRID_COLOR, linestyles='--', linewidths=0.9)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f', colors=INK_SOFT)

    # Arc of "same variability as the observations".
    ax.plot(np.linspace(0, theta_max, 200), np.full(200, ref),
            color=INK, linestyle=':', linewidth=1.3)
    ax.plot([0], [ref], marker='*', markersize=18, color=INK,
            linestyle='none', label='Observed outcome (reference)', zorder=6)

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h', 'p', '*', '8']
    for i, e in enumerate(entries):
        ax.plot([np.arccos(np.clip(e["R"], -1, 1))], [e["SD"] * scale],
                marker=markers[i % len(markers)], markersize=8,
                color=(colors or {}).get(e["name"], SERIES_COLORS[i % len(SERIES_COLORS)]),
                markeredgecolor='white', markeredgewidth=1.2,
                linestyle='none', label=f'{i + 1}. {e["name"]}', zorder=5)
        # numbered so many markers stay legible; the legend maps number -> name
        ax.annotate(str(i + 1), (np.arccos(np.clip(e["R"], -1, 1)), e["SD"] * scale),
                    textcoords='offset points', xytext=(6, 4), fontsize=7.5,
                    color=INK_SOFT)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=26, color=INK)
    #  Positioned relative to the VISIBLE radial span. Using rmax * 1.16 breaks
    #  in zoomed mode, where rmin is large and that lands outside the axes.
    _r_label = rmax + 0.10 * (rmax - rmin)
    ax.text(theta_max * 0.5, _r_label, "Correlation with outcome",
            ha='center', va='center', fontsize=10, clip_on=False, color=INK_SOFT,
            rotation=(-np.degrees(theta_max) / 2 if min_R >= 0 else 0))
    ax.set_xlabel(("Normalized " if normalize else "") + "SD of predicted probability",
                  labelpad=18, color=INK_SOFT)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='upper left', bbox_to_anchor=(1.10, 1.02), fontsize=8,
              frameon=True, framealpha=0.95, edgecolor=GRID_COLOR)
    fig.tight_layout()
    if savepath:
        save_fig(fig, savepath, subdir="taylor")
    return fig, ax


def build_taylor(specs, X, y, dataset_label, normalize=True):
    """Stats table + diagram for one dataset (train or test)."""
    entries = []
    for spec in specs:
        s = taylor_statistics(y, spec.predict_proba(X))
        s["name"] = spec.name
        entries.append(s)

    sd_obs = entries[0]["SD_obs"]
    tdf = pd.DataFrame([{
        "Model": e["name"], "SD": e["SD"], "SD/SD_obs": e["SD"] / sd_obs,
        "R": e["R"], "CRMSE": e["CRMSE"], "Bias": e["Bias"],
    } for e in entries]).sort_values("CRMSE")

    print("\n" + "=" * 78)
    print(f"TAYLOR STATISTICS (probabilities) — {dataset_label.upper()}   "
          f"(SD_obs = {sd_obs:.4f})")
    print("=" * 78)
    print(tdf.round(4).to_string(index=False))
    register_table(f"Taylor {dataset_label}", tdf)

    tcolors = comparison_colors(specs)
    for zoom, suffix in [(False, ""), (True, "_zoom")]:
        fig, _ = taylor_diagram(
            entries, sd_obs,
            title=(f"Taylor Diagram — {dataset_label} probabilities"
                   + (" (zoomed)" if zoom else "")),
            normalize=normalize, colors=tcolors, zoom=zoom,
            savepath=f"taylor_diagram_{dataset_label}{suffix}.png")
        plt.show()
        plt.close(fig)
    return tdf


taylor_df_test  = build_taylor(roc_specs, X_te, y_te, "test")
taylor_df_train = build_taylor(roc_specs, X_tr, y_tr, "train")


# =============================================================================
#  SECTION 11 — BIAS-VARIANCE TRADEOFF  (0-1 loss decomposition)
# =============================================================================
#  REGRESSION -> CLASSIFICATION, and this one cannot be translated literally.
#  The regression file used the squared-error decomposition
#
#      E[(pred - y)^2] = (mean_pred - y)^2 + Var(pred)
#
#  which is a statement about the MEAN and VARIANCE of a real-valued
#  prediction. Neither quantity is defined for class labels: the mean of
#  {'cat', 'dog', 'cat'} is not a class, and its variance is not a number.
#  Averaging the PROBABILITIES instead would decompose the Brier score, not the
#  error rate the model is actually judged on.
#
#  The established replacement, implemented here, is the Kohavi-Wolpert /
#  Domingos decomposition of 0-1 loss, in which "mean" becomes MAJORITY VOTE
#  and "variance" becomes DISAGREEMENT WITH THAT VOTE:
#
#      main(x) = the modal prediction across B bootstrap-trained models
#      bias(x) = 1 if main(x) != y(x) else 0        systematic error
#      V(x)    = P_b[ pred_b(x) != main(x) ]        instability
#
#  and the expected 0-1 loss decomposes EXACTLY as
#
#      E_b[loss(x)] = bias(x) + c(x) * V(x),
#          c(x) = +1                            where the model is unbiased
#          c(x) = -P_b[pred_b(x) = y(x)] / V(x) where it is biased
#
#  The second case is why variance can HELP: on a row the model gets
#  systematically wrong, instability occasionally knocks a prediction onto the
#  right answer. So the aggregate reported below is NET variance (unbiased
#  variance minus biased variance), and bias + net variance reconstructs the
#  average error rate exactly — the residual is printed as a check.
#
#  Bootstraps are STRATIFIED (resampled within each class, preserving class
#  counts). A plain bootstrap of an imbalanced target can draw a resample with
#  no minority rows at all, which makes several of the models raise rather than
#  train.
# =============================================================================
from scipy.stats import mode as scipy_mode

bv_models = models          # e.g. [rf, xgboost_model, lgbm, knn] for a faster pass


def stratified_bootstrap_index(y, rng):
    """Resample with replacement WITHIN each class, keeping class counts."""
    idx = []
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        idx.append(rng.choice(members, size=len(members), replace=True))
    out = np.concatenate(idx)
    rng.shuffle(out)
    return out


def bias_variance_decomposition(spec, X_train, y_train, X_test, y_test,
                                n_bootstrap=None, params=None):
    """
    Kohavi-Wolpert / Domingos 0-1 loss decomposition.
    Returns (bias, net_variance, avg_loss, extras).
    """
    n_bootstrap = N_BOOTSTRAP if n_bootstrap is None else n_bootstrap
    rng = np.random.RandomState(SEED)
    preds = np.zeros((n_bootstrap, len(X_test)), dtype=np.int64)
    for b in range(n_bootstrap):
        idx = stratified_bootstrap_index(y_train, rng)
        proba_fn = spec.refit_on(X_train[idx], y_train[idx], params=params)
        preds[b] = labels_from_proba(proba_fn(X_test))

    main_pred = scipy_mode(preds, axis=0, keepdims=False).mode.astype(np.int64)
    bias_vec  = (main_pred != y_test).astype(float)               # 1 where systematically wrong
    var_vec   = (preds != main_pred).mean(axis=0)                 # instability
    p_correct = (preds == y_test).mean(axis=0)                    # P_b[pred = y]

    bias            = float(bias_vec.mean())
    unbiased_var    = float((var_vec * (1 - bias_vec)).mean())    # variance that HURTS
    biased_var      = float((p_correct * bias_vec).mean())        # variance that HELPS
    net_variance    = unbiased_var - biased_var
    avg_loss        = float((preds != y_test).mean())

    return bias, net_variance, avg_loss, {
        "unbiased_variance": unbiased_var, "biased_variance": biased_var,
        "total_variance": float(var_vec.mean()),
        "residual": abs(avg_loss - (bias + net_variance)),
    }


# ── PLOT 1: bias vs net variance across all tuned models ─────────────────────
bv_rows = []
for spec in bv_models:
    b, netv, loss, extra = bias_variance_decomposition(spec, X_tr, y_tr, X_te, y_te)
    bv_rows.append({"Model": spec.name, "Bias": b, "Net variance": netv,
                    "Avg 0-1 loss": loss, "Accuracy": 1 - loss,
                    "Unbiased var": extra["unbiased_variance"],
                    "Biased var": extra["biased_variance"],
                    "Identity residual": extra["residual"]})
    print(f"{spec.name:32s} bias={b:.4f}  net var={netv:+.4f}  "
          f"avg 0-1 loss={loss:.4f}  (residual {extra['residual']:.2e})")

bv_df = pd.DataFrame(bv_rows).sort_values("Avg 0-1 loss")
print("\n" + "=" * 78)
print("BIAS-VARIANCE DECOMPOSITION OF 0-1 LOSS")
print("  Bias + Net variance = Avg 0-1 loss (residual ~ 0 confirms the identity)")
print("=" * 78)
print(bv_df.round(4).to_string(index=False))
register_table("Bias-variance", bv_df)

fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.42 * len(bv_df))))
ypos = np.arange(len(bv_df))
h = 0.36
#  GROUPED, not stacked: net variance can be negative (variance that rescues a
#  biased prediction), and a stacked bar cannot render a negative segment
#  without lying about the total. The total is marked separately.
ax.barh(ypos - h / 2, bv_df["Bias"], height=h, color=SERIES_COLORS[0], label='Bias')
ax.barh(ypos + h / 2, bv_df["Net variance"], height=h, color=SERIES_COLORS[1],
        label='Net variance')
ax.plot(bv_df["Avg 0-1 loss"], ypos, marker='D', markersize=7, linestyle='none',
        color=INK, markerfacecolor='white', markeredgewidth=1.8,
        label='Avg 0-1 loss (= bias + net variance)')
for i, total in enumerate(bv_df["Avg 0-1 loss"]):
    ax.text(total + 0.008, i, f'{total:.3f}', va='center', fontsize=9, color=INK_SOFT)

ax.set_yticks(ypos); ax.set_yticklabels(bv_df["Model"], fontsize=9)
ax.axvline(0, color=INK_SOFT, linewidth=0.8)
_style(ax, "Bias-variance decomposition of 0-1 loss", "Error contribution", None)
ax.grid(axis='y', visible=False)
ax.invert_yaxis()
ax.legend(loc='lower right', fontsize=9, frameon=True, framealpha=0.95,
          edgecolor=GRID_COLOR)
plt.tight_layout()
save_fig(plt.gcf(), "bias_variance_decomposition", subdir="bias_variance")
plt.show()
plt.close(fig)


# ── PLOT 2: the classic tradeoff curve vs model complexity ───────────────────
def plot_bias_variance_curve(spec, param_name, values, xlabel, n_bootstrap=10,
                             invert_x=False):
    """
    Sweep ONE complexity hyperparameter, holding the rest at their tuned
    values, and plot bias / net variance / total 0-1 loss against it. The tuned
    value is marked so you can see where the search landed relative to the
    minimum.
    """
    biases, netvars, totals = [], [], []
    for v in values:
        params = dict(spec.best_params)
        params[param_name] = v
        b, netv, loss, _ = bias_variance_decomposition(
            spec, X_tr, y_tr, X_te, y_te, n_bootstrap=n_bootstrap, params=params)
        biases.append(b); netvars.append(netv); totals.append(loss)
        print(f"  {spec.name} | {param_name}={v}: bias={b:.4f} net var={netv:+.4f} "
              f"loss={loss:.4f}")

    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, biases,  marker='o', markersize=6, linewidth=2, color=SERIES_COLORS[0],
            label='Bias')
    ax.plot(x, netvars, marker='s', markersize=6, linewidth=2, color=SERIES_COLORS[1],
            label='Net variance')
    ax.plot(x, totals,  marker='^', markersize=7, linewidth=2.5, color=INK,
            linestyle='--', label='Avg 0-1 loss')

    tuned = spec.best_params.get(param_name)
    if tuned in list(values):
        ax.axvline(list(values).index(tuned), color=INK_SOFT, linestyle=':', linewidth=1.5)
        ax.annotate(' tuned', xy=(list(values).index(tuned), ax.get_ylim()[1]),
                    fontsize=9, color=INK_SOFT, va='top')

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values])
    if invert_x:
        ax.invert_xaxis()
    ax.axhline(0, color=INK_SOFT, linewidth=0.8)
    _style(ax, f"Bias-variance tradeoff — {spec.name}", xlabel, "Error")
    ax.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.95,
              edgecolor=GRID_COLOR)
    plt.tight_layout()
    save_fig(plt.gcf(), f"bias_variance_curve_{_fname(spec.name)}", subdir="bias_variance")
    plt.show()
    plt.close(fig)


#  These two sweeps name specific models, so they are guarded against a roster
#  that no longer contains them: Section 0 recommends trimming `models` to
#  control runtime, and a hardcoded reference here would otherwise fail on an
#  unfitted spec at the very end of a long run.
#    KNN: complexity DECREASES as k grows, so the x-axis is inverted to keep the
#         conventional "simple on the left, complex on the right" reading.
#    Random Forest: complexity increases with depth.
_COMPLEXITY_SWEEPS = [
    (knn, "model__n_neighbors",
     [v for v in [1, 2, 3, 5, 8, 12, 20, 30, 40] if v <= max_k],
     "k  (fewer neighbours = more complex)", True),
    (rf, "model__max_depth", [2, 3, 5, 8, 12, 20, None],
     "max_depth  (deeper = more complex)", False),
]
for _spec, _param, _values, _xlabel, _invert in _COMPLEXITY_SWEEPS:
    if any(m is _spec for m in bv_models):
        plot_bias_variance_curve(_spec, _param, _values, xlabel=_xlabel, invert_x=_invert)
    else:
        print(f"  complexity sweep skipped: {_spec.name} is not in the current roster.")


# =============================================================================
#  SECTION 12 — SEED SENSITIVITY  (optional honesty check)
# =============================================================================
#  Refits the tuned models across several different STRATIFIED train/test
#  splits and reports mean +/- std of the primary metric. If a model's headline
#  number holds up here, the split was not doing the work. Hyperparameters are
#  held at their already-tuned values — this measures split sensitivity, not
#  tuning stability (that is what nested CV is for).
#
#  This matters more for classification than it did for regression: with an
#  imbalanced target the test set may contain only a few dozen minority rows,
#  so a handful of them moving between splits can swing recall and F1 by a lot
#  while accuracy barely twitches. Both are reported for that reason.
# =============================================================================
SEED_SENSITIVITY_SEEDS = [BEST_SEED, 0, 1, 7, 42]


def seed_sensitivity(models, X_df, y_ser, seeds=None, test_size=None):
    seeds = SEED_SENSITIVITY_SEEDS if seeds is None else seeds
    test_size = TEST_SIZE if test_size is None else test_size
    rows = []
    for spec in models:
        aucs, f1s = [], []
        for s in seeds:
            Xa, Xb, ya, yb = train_test_split(X_df, y_ser, test_size=test_size,
                                              random_state=s, stratify=y_ser)
            Xa = np.asarray(Xa, dtype=np.float32); Xb = np.asarray(Xb, dtype=np.float32)
            ya = np.asarray(ya, dtype=np.int64).ravel(); yb = np.asarray(yb, dtype=np.int64).ravel()
            proba_fn = spec.refit_on(Xa, ya)
            proba = proba_fn(Xb)
            aucs.append(roc_auc(yb, proba))
            f1s.append(compute_metrics(yb, labels_from_proba(proba), proba)["F1"])
        rows.append({"Model": spec.name,
                     "AUC mean": np.nanmean(aucs), "AUC std": np.nanstd(aucs),
                     "F1 mean": np.nanmean(f1s), "F1 std": np.nanstd(f1s),
                     "AUC min": np.nanmin(aucs), "AUC max": np.nanmax(aucs)})
        print(f"{spec.name:32s} AUC {np.nanmean(aucs):.4f} +/- {np.nanstd(aucs):.4f}   "
              f"F1 {np.nanmean(f1s):.4f} +/- {np.nanstd(f1s):.4f}")
    return pd.DataFrame(rows).sort_values("AUC mean", ascending=False)


# seed_df = seed_sensitivity(models, X, Y)
# print(seed_df.round(4).to_string(index=False))


# =============================================================================
# =============================================================================
#  PART 7 — FACTOR IMPORTANCE
# =============================================================================
# =============================================================================

# =============================================================================
#  SECTION 9 — SHAP EXPLANATIONS  (bar, beeswarm, violin, waterfall, dependence)
# =============================================================================
#  Assumes fit_all() (Part 5) already ran, so every spec in `models` is fitted
#  and tuned.  pip install shap
#
#  REGRESSION -> CLASSIFICATION, two changes that matter:
#
#   1. SHAP VALUES ARE PER CLASS. A regression model has one output, so it has
#      one set of SHAP values. A classifier has K, and a tree explainer returns
#      a (n_rows, n_features, K) array (or a length-K list). Silently taking
#      the first one — which is what the regression code's
#      `if isinstance(shap_values, list): shap_values = shap_values[0]` would
#      do here — explains class 0, i.e. for a binary problem it explains the
#      NEGATIVE class and every sign in the plot is backwards. SHAP_CLASS below
#      is explicit, defaults to the positive class, and is printed.
#
#   2. THE BLACK-BOX EXPLAINER WRAPS predict_proba, NOT predict. Explaining
#      hard labels means explaining a step function: it is flat almost
#      everywhere, so the attributions are near-zero noise except right at the
#      decision boundary. The probability surface is smooth and is what SHAP's
#      additivity assumption is meant for. (Explaining the LOG-ODDS is the
#      other defensible choice, and is what TreeExplainer does natively for
#      boosted models — which is why tree SHAP values here are on a different
#      scale from the black-box ones and should not be compared across models
#      in absolute terms, only within a model.)
#
#  The black-box fallback re-evaluates the model many times per row, so its
#  background is capped at SHAP_BACKGROUND_SIZE and the explained rows at
#  SHAP_TEST_SIZE. TreeExplainer is exact (no sampling) so the caps do not
#  affect the tree models, but they matter a lot for the Keras nets, the two
#  Pipelines, and especially the SVC.
# =============================================================================
import shap

feature_names = (list(x_train.columns) if hasattr(x_train, "columns")
                 else [f"Feature_{i}" for i in range(N_FEATURES)])

TREE_KEYS = ("RandomForest", "ExtraTrees", "DecisionTree", "GradientBoosting",
             "HistGradientBoosting", "XGB", "LGBM", "CatBoost")

#  Which class is being explained. Binary -> the positive class, so a positive
#  SHAP value always reads as "pushes toward the positive class".
SHAP_CLASS = POS_LABEL if IS_BINARY else 0
print(f"\nSHAP explains class {SHAP_CLASS} = {CLASSES[SHAP_CLASS]!r}. "
      f"Positive SHAP value = pushes the prediction toward this class.")

SHAP_BACKGROUND_SIZE = 100             # background rows for the black-box fallback
SHAP_TEST_SIZE = min(200, len(X_te))   # rows actually explained/plotted

_shap_rng = np.random.RandomState(SEED)
_bg_idx = _shap_rng.choice(len(X_tr), size=min(SHAP_BACKGROUND_SIZE, len(X_tr)), replace=False)
_te_idx = _shap_rng.choice(len(X_te), size=SHAP_TEST_SIZE, replace=False)
X_bg   = X_tr[_bg_idx]      # background for the black-box explainer only
X_shap = X_te[_te_idx]      # rows explained/plotted for every model


def _select_class(values, cls):
    """
    Pull one class's SHAP values out of whatever shape the explainer returned.
    SHAP has used three layouts across versions for classifiers:
      list of K arrays, each (n, f)      -> older TreeExplainer
      one array (n, f, K)                -> current TreeExplainer
      one array (n, f)                   -> single-output (already one class)
    """
    if isinstance(values, list):
        return np.asarray(values[cls])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, cls]
    return values


def _select_base(expected_value, cls):
    ev = np.ravel(expected_value)
    return float(ev[cls]) if ev.size > 1 else float(ev[0])


def _raw_feature_estimator(model):
    """
    The estimator that may be explained on RAW features, or None.

    Section B wraps every sklearn model in a pipeline, so spec.model is usually
    a Pipeline and a plain type-name check would send the tree models down the
    slow, approximate black-box path instead of exact TreeExplainer.

    Unwrapping is only sound when nothing ahead of the final estimator CHANGES
    the features. A sampler qualifies: it acts during fit and is a no-op at
    transform time, so the fitted estimator still consumes raw columns. A
    scaler does not — explaining it on unscaled values would silently
    misattribute — so the KNN and SVC pipelines return None here and keep the
    black-box path they were already on.
    """
    if not hasattr(model, "steps"):
        return model
    for _, step in model.steps[:-1]:
        if step == "passthrough" or hasattr(step, "fit_resample"):
            continue
        return None
    return model.steps[-1][1]


# ── BUILD SHAP EXPLANATIONS FOR EACH MODEL ────────────────────────────────────
explanations, explained_specs = [], []

for spec in models:
    model = _raw_feature_estimator(spec.model)
    cls_name = type(model).__name__ if model is not None else ""
    try:
        if model is not None and any(k in cls_name for k in TREE_KEYS):
            explainer = shap.TreeExplainer(model, feature_names=feature_names)
            values = explainer.shap_values(X_shap)
            base = _select_base(explainer.expected_value, SHAP_CLASS)
        elif model is not None and hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_tr, feature_names=feature_names)
            values = explainer.shap_values(X_shap)
            base = _select_base(explainer.expected_value, SHAP_CLASS)
        else:
            # spec.predict_proba, never model.predict: the Keras nets and the
            # KNN/SVC Pipelines expect scaled inputs, and the scalers live
            # inside spec's proba_fn rather than on the model object. Calling
            # model.predict on raw features would feed them wildly
            # out-of-distribution values — garbage in, garbage SHAP out.
            def f(data, _spec=spec):
                return _spec.predict_proba(data)[:, SHAP_CLASS]
            explainer = shap.Explainer(f, X_bg, feature_names=feature_names)
            values = explainer(X_shap).values
            base = float(np.mean(spec.predict_proba(X_bg)[:, SHAP_CLASS]))

        sel = _select_class(values, SHAP_CLASS)
        explanations.append(shap.Explanation(
            values        = sel,
            # ONE BASE VALUE PER ROW, as an array — not a bare Python float.
            # shap's Explanation arithmetic (exp.abs.mean(0), used by the bar
            # plot) reads self.base_values.shape, so a scalar float raises
            # AttributeError deep inside the plotting call. A numpy scalar
            # happens to survive because it has .shape == (), but the per-row
            # array is what shap actually expects and it keeps the base value
            # aligned when rows are sliced (e.g. the waterfall plot below).
            base_values   = np.full(len(sel), base, dtype=np.float64),
            data          = X_shap,
            feature_names = feature_names))
        explained_specs.append(spec)
    except Exception as exc:
        print(f"  SHAP unavailable for {spec.name}: {type(exc).__name__}: {exc}")

SHAP_RESULTS = dict(zip((s.name for s in explained_specs), explanations))


def _shap_figure(plot_fn, explanation, title, filename):
    """
    Render one SHAP figure, failing soft.

    shap's plotting API moves between releases more than any other dependency
    in this file, and these plots are diagnostics — losing one must not abort
    the run, least of all in Part 7 after every model has already been tuned.
    """
    try:
        plt.figure()
        plot_fn(explanation, show=False)
        plt.title(title, fontsize=13, fontweight='bold', color=INK)
        plt.tight_layout()
        save_fig(plt.gcf(), filename, subdir="shap")
        plt.show()
    except Exception as exc:
        print(f"  SHAP plot skipped ({filename}): {type(exc).__name__}: {exc}")
    finally:
        plt.close()


# ── SUMMARY BAR PLOTS ─────────────────────────────────────────────────────────
for spec, explanation in zip(explained_specs, explanations):
    _shap_figure(shap.plots.bar, explanation,
                 f"Feature importance — {spec.name}  (class: {CLASSES[SHAP_CLASS]})",
                 f"shap_bar_{_fname(spec.name)}.png")

# ── BEESWARM PLOTS ────────────────────────────────────────────────────────────
for spec, explanation in zip(explained_specs, explanations):
    _shap_figure(shap.plots.beeswarm, explanation,
                 f"SHAP beeswarm — {spec.name}  (class: {CLASSES[SHAP_CLASS]})",
                 f"shap_beeswarm_{_fname(spec.name)}.png")

# ── VIOLIN PLOTS ──────────────────────────────────────────────────────────────
#  Same global information as the beeswarm, shown as a per-feature distribution
#  instead of individual points — easier to read with many test rows.
for spec, explanation in zip(explained_specs, explanations):
    _shap_figure(shap.plots.violin, explanation,
                 f"SHAP violin — {spec.name}  (class: {CLASSES[SHAP_CLASS]})",
                 f"shap_violin_{_fname(spec.name)}.png")

# ── WATERFALL PLOTS ───────────────────────────────────────────────────────────
#  One row's prediction, decomposed. For a classifier this reads as "why was
#  THIS case assigned this class", which is usually the explanation a
#  stakeholder actually asks for.
sample_idx = 0
for spec, explanation in zip(explained_specs, explanations):
    _shap_figure(lambda e, show: shap.plots.waterfall(e[sample_idx], show=show),
                 explanation,
                 f"SHAP waterfall — {spec.name}  (row {sample_idx}, "
                 f"true class: {decode_target([y_te[_te_idx[sample_idx]]])[0]})",
                 f"shap_waterfall_{_fname(spec.name)}.png")

# ── DEPENDENCE PLOTS ──────────────────────────────────────────────────────────
feature = feature_names[0]  # Replace with your feature of interest
for spec, explanation in zip(explained_specs, explanations):
    _shap_figure(lambda e, show: shap.plots.scatter(e[:, feature], show=show),
                 explanation,
                 f"SHAP dependence ({feature}) — {spec.name}",
                 f"shap_dependence_{_fname(spec.name)}.png")


# =============================================================================
#  SECTION 9B — PERMUTATION IMPORTANCE
# =============================================================================
#  The second view of feature importance in Part 7, and the bluntest question
#  of the three: if this feature were pure noise, how much worse would the
#  classifier be? Shuffle one column, re-score, and the drop IS the importance.
#
#  It answers something SHAP does not. SHAP attributes the predictions the
#  model actually makes, so it explains the model's BEHAVIOUR: a feature the
#  model leans on heavily gets a large attribution whether or not that
#  reliance helps. Permutation importance scores against the TRUTH, so it
#  measures how much a feature contributes to being RIGHT. A feature can rank
#  high on SHAP and near zero here — the model uses it, and the use buys
#  nothing. The rank comparison at the end of this section is where that
#  disagreement surfaces, and a disagreement is a finding, not an error.
#
#  SCORED ON PROBABILITIES, NOT LABELS. The drop is measured in ROC-AUC (or
#  whatever PRIMARY_METRIC is set to) rather than accuracy, and that choice
#  matters more in classification than the equivalent choice does in
#  regression. Accuracy is a step function of the probabilities: on an
#  imbalanced problem, shuffling a genuinely useful feature can move every
#  probability substantially and still flip almost no argmax decisions, so
#  accuracy reports ~0 importance for a feature the model depends on. A
#  ranking metric sees the movement. This is the single most common way
#  permutation importance is made to look empty on imbalanced data.
#
#  Both splits are computed. Train-set permutation measures how much a feature
#  was used to FIT; test-set permutation measures how much it carries to
#  unseen data. The gap between them is diagnostic on its own: a feature that
#  matters a lot on train and nothing on test was memorised, not learned.
#
#  THE CORRELATION CAVEAT — the same one the PDP section carries, for the same
#  reason. Shuffling one column builds rows that never occur in nature, and
#  the model is then scored on them. Worse, with two strongly correlated
#  features each can be shuffled with little damage because the other still
#  carries the signal, so BOTH look unimportant and the pair's real
#  contribution goes unreported. Read the Part 3 correlation heatmaps
#  alongside this, and treat a low score for a feature with a highly
#  correlated partner as "not UNIQUELY important" rather than "not important".
#
#  Cost is (n_features x n_repeats + 1) predict_proba calls per model. Nothing
#  is refitted, so this scales with the roster rather than with training.
# =============================================================================
PERM_N_REPEATS = 10          # shuffles per feature; more repeats = tighter error bars
PERM_MODELS    = models      # use all_models to also cover the Part 4B ensembles


def permutation_importance_spec(spec, X, y, n_repeats=None, seed=None):
    """
    Model-agnostic permutation importance for one already-fitted ModelSpec.

    Written against spec.predict_proba rather than sklearn's
    permutation_importance so the Keras nets and the Part 4B ensembles —
    neither of which is an sklearn estimator — travel exactly the same code
    path as the trees, and so the score is the probability-based
    PRIMARY_METRIC rather than a thresholded one.

    Returns (baseline, drops), where drops has shape (n_features, n_repeats)
    and every entry is baseline - score_after_shuffling. Positive means the
    model got worse without the feature.
    """
    n_repeats = PERM_N_REPEATS if n_repeats is None else n_repeats
    seed = SEED if seed is None else seed
    rng = np.random.RandomState(seed)
    X = np.array(X, dtype=np.float64, copy=True)    # never mutate the caller's array
    y = np.asarray(y).astype(int)
    baseline = primary_score(y, spec.predict_proba(X))
    drops = np.empty((X.shape[1], n_repeats), dtype=np.float64)
    for j in range(X.shape[1]):
        original = X[:, j].copy()
        for r in range(n_repeats):
            X[:, j] = rng.permutation(original)
            drops[j, r] = baseline - primary_score(y, spec.predict_proba(X))
        X[:, j] = original                          # restore before the next column
    return baseline, drops


def compute_permutation_importance(specs=None):
    """Run the permutation for every spec on both splits; returns a long frame."""
    specs = PERM_MODELS if specs is None else specs
    rows, store = [], {}
    for spec in specs:
        for split, (Xs, ys) in (("test", (X_te, y_te)), ("train", (X_tr, y_tr))):
            baseline, drops = permutation_importance_spec(spec, Xs, ys)
            store[(spec.name, split)] = (baseline, drops)
            order = np.argsort(drops.mean(axis=1))[::-1]
            rank = np.empty(len(order), dtype=int)
            rank[order] = np.arange(1, len(order) + 1)
            for j, feat in enumerate(feature_names):
                rows.append({"Model": spec.name, "Split": split, "Feature": feat,
                             f"Baseline {PRIMARY_METRIC}": baseline,
                             "Mean drop": drops[j].mean(), "Std": drops[j].std(),
                             "Rank": int(rank[j])})
        print(f"  {spec.name:32s} baseline {PRIMARY_METRIC}  "
              f"test={store[(spec.name,'test')][0]:.4f}  "
              f"train={store[(spec.name,'train')][0]:.4f}")
    return pd.DataFrame(rows), store


print("\n" + "=" * 78)
print(f"PERMUTATION IMPORTANCE  ({PERM_N_REPEATS} shuffles per feature, "
      f"scored on {PRIMARY_METRIC})")
print("=" * 78)
perm_df, PERM_RESULTS = compute_permutation_importance()
register_table("Permutation importance", perm_df)

_perm_colors = comparison_colors(PERM_MODELS)


# ── PER-MODEL BAR CHARTS (mean drop +/- std over the repeats) ─────────────────
def plot_permutation_importance(specs=None, split="test"):
    specs = PERM_MODELS if specs is None else specs
    for spec in specs:
        baseline, drops = PERM_RESULTS[(spec.name, split)]
        means, stds = drops.mean(axis=1), drops.std(axis=1)
        order = np.argsort(means)                    # ascending: biggest at the top
        fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.42 * len(order) + 1.4)))
        ax.barh([feature_names[i] for i in order], means[order],
                xerr=stds[order], color=_perm_colors[spec.name], edgecolor=INK,
                linewidth=0.6, error_kw=dict(ecolor=INK, lw=1.0, capsize=3))
        ax.axvline(0, color=INK, linewidth=0.9)
        ax.set_xlabel(f"Drop in {PRIMARY_METRIC} when the feature is shuffled")
        ax.set_title(f"Permutation Importance — {spec.name}  ({split}, "
                     f"baseline = {baseline:.3f})",
                     fontsize=12, fontweight='bold', color=INK)
        ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        save_fig(fig, f"permutation_{_fname(spec.name)}_{split}",
                 subdir="permutation/per_model")
        plt.show()
        plt.close(fig)


plot_permutation_importance(split="test")

# ── CONSENSUS ACROSS MODELS ──────────────────────────────────────────────────
#  One model's ranking is one model's opinion. Averaging the mean drop across
#  the roster is the closest thing to a model-independent statement the data
#  supports, and the spread across models says how much to trust it.
_perm_test = perm_df[perm_df.Split == "test"]
perm_consensus = (_perm_test.groupby("Feature")["Mean drop"]
                  .agg(["mean", "std", "min", "max"])
                  .sort_values("mean", ascending=False)
                  .rename(columns={"mean": "Mean drop (across models)",
                                   "std": "Std across models",
                                   "min": "Min", "max": "Max"}))
print("\nConsensus permutation importance (test, averaged over the roster):")
display(perm_consensus.round(4))
register_table("Permutation consensus", perm_consensus, index=True)

fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.45 * len(perm_consensus) + 1.4)))
_c = perm_consensus.iloc[::-1]
ax.barh(_c.index, _c["Mean drop (across models)"],
        xerr=_c["Std across models"].fillna(0.0), color=ACCENT,
        edgecolor=INK, linewidth=0.6,
        error_kw=dict(ecolor=INK, lw=1.0, capsize=3))
ax.axvline(0, color=INK, linewidth=0.9)
ax.set_xlabel(f"Mean drop in {PRIMARY_METRIC} (averaged across models)")
ax.set_title(f"Permutation Importance — consensus over {len(PERM_MODELS)} models",
             fontsize=12, fontweight='bold', color=INK)
ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
fig.tight_layout()
save_fig(fig, "permutation_consensus", subdir="permutation")
plt.show()
plt.close(fig)

# ── TRAIN vs TEST (memorised or learned?) ────────────────────────────────────
_perm_gap = (perm_df.pivot_table(index="Feature", columns="Split",
                                 values="Mean drop", aggfunc="mean")
             .rename(columns={"train": "Mean drop (train)",
                              "test": "Mean drop (test)"}))
_perm_gap["Train - Test"] = _perm_gap["Mean drop (train)"] - _perm_gap["Mean drop (test)"]
_perm_gap = _perm_gap.sort_values("Train - Test", ascending=False)
print("\nTrain vs test permutation importance — a large positive gap means the "
      "feature\nwas used to fit and did not carry to unseen data:")
display(_perm_gap.round(4))
register_table("Permutation train vs test", _perm_gap, index=True)

# ── PERMUTATION vs SHAP: do the two rankings agree? ──────────────────────────
#  Spearman over the ranks, per model. High agreement means the model's
#  reliance and the feature's usefulness point the same way. Low agreement is
#  the interesting case and is worth reading feature by feature.
def compare_shap_and_permutation():
    if not SHAP_RESULTS:
        print("\nNo SHAP results to compare against — skipping.")
        return None
    rows = []
    for spec in PERM_MODELS:
        exp = SHAP_RESULTS.get(spec.name)
        if exp is None:
            continue
        shap_imp = np.abs(exp.values).mean(axis=0)
        perm_imp = PERM_RESULTS[(spec.name, "test")][1].mean(axis=1)
        if len(shap_imp) != len(perm_imp):
            continue
        rho, pval = st_spearmanr(shap_imp, perm_imp)
        rows.append({"Model": spec.name, "Spearman rho": rho, "p-value": pval,
                     "SHAP top feature": feature_names[int(np.argmax(shap_imp))],
                     "Permutation top feature": feature_names[int(np.argmax(perm_imp))]})
    if not rows:
        print("\nNo model has both SHAP and permutation results — skipping.")
        return None
    out = pd.DataFrame(rows)
    print("\nDo SHAP and permutation importance rank the features the same way?")
    display(out.round(4))
    register_table("SHAP vs permutation", out)
    return out


shap_vs_perm = compare_shap_and_permutation()


# =============================================================================
#  SECTION 9C — LIME  (Local Interpretable Model-agnostic Explanations)
# =============================================================================
#  pip install lime
#
#  The third view in Part 7, and the only LOCAL one. SHAP and permutation
#  importance both answer "which features matter?" over a whole dataset. LIME
#  answers "why was THIS row classified that way?" — it perturbs a single row,
#  watches how the predicted probability moves, and fits a small weighted
#  linear model to that neighbourhood. Its coefficients are the explanation.
#
#  Why keep it when SHAP is already here. They are not the same calculation
#  dressed differently: SHAP's local explanations sum exactly to the
#  prediction (the efficiency axiom), LIME's do not and make no such promise.
#  What LIME gives instead is a readable IF-THEN reading of the neighbourhood
#  — "0.46 < cement <= 0.73 pushed P(class 1) up by 0.21" — because it
#  discretises the features into bins first. That phrasing is the one people
#  will actually argue with, and arguing with an explanation is the point.
#
#  READ exp.score BEFORE YOU BELIEVE ANY OF IT. That is the local surrogate's
#  own R-squared: how well the little linear model reproduces the real model
#  in that neighbourhood. It is printed beside every explanation below, and it
#  is often mediocre. A LIME explanation with a local R-squared of 0.15 is not
#  a weak explanation, it is not an explanation at all — the linear surrogate
#  does not describe the classifier there, and the weights are noise. This is
#  the most common way LIME is misread, so the number sits next to every
#  single plot rather than being buried.
#
#  WHICH ROWS, AND WHICH CLASS. Explaining row 0 for class 0 is a habit, not a
#  choice. The rows picked here are the ones worth looking at — a confident
#  correct prediction, a confident MISTAKE (the most valuable explanation in
#  the file: it shows what the model thought it saw), and the most uncertain
#  row — and each is explained for the class the model actually predicted,
#  because that is the decision being questioned.
# =============================================================================
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("lime is not installed — skipping Section 9C.  pip install lime")

LIME_MODELS      = models[:3]   # local explanations are per-row; keep the roster small
LIME_N_FEATURES  = min(8, N_FEATURES)
LIME_NUM_SAMPLES = 5000         # perturbations drawn per explanation


def select_instances_to_explain(spec, X, y):
    """
    Rows worth explaining: a confident hit, a confident miss, and a coin flip.

    Returns [(index, label), ...]. The confident mistake is the one to read
    first — a high-probability wrong answer is where the model's reasoning
    and reality part company, and it is the only case where an explanation
    can tell you something you could not get from the metrics.
    """
    proba = spec.predict_proba(X)
    pred = labels_from_proba(proba)
    y = np.asarray(y).astype(int)
    confidence = proba.max(axis=1)
    correct = pred == y
    picks = []
    if correct.any():
        idx = int(np.flatnonzero(correct)[np.argmax(confidence[correct])])
        picks.append((idx, "confident and correct"))
    if (~correct).any():
        idx = int(np.flatnonzero(~correct)[np.argmax(confidence[~correct])])
        picks.append((idx, "confident but wrong"))
    picks.append((int(np.argmin(confidence)), "least confident"))
    seen, unique = set(), []
    for idx, label in picks:                          # a tiny test set can collide
        if idx not in seen:
            seen.add(idx)
            unique.append((idx, label))
    return unique


def explain_one(explainer, spec, row, idx, label):
    """One LIME explanation, rendered and tabulated. Returns rows for the table."""
    proba = spec.predict_proba(row.reshape(1, -1))[0]
    predicted = int(np.argmax(proba))
    actual = int(np.asarray(y_te).astype(int)[idx])
    exp = explainer.explain_instance(
        np.asarray(row, dtype=np.float64), spec.predict_proba,
        num_features=LIME_N_FEATURES, num_samples=LIME_NUM_SAMPLES,
        labels=(predicted,))
    pairs = exp.as_list(label=predicted)
    local_pred = float(np.ravel(exp.local_pred)[0])
    intercept = float(exp.intercept[predicted])

    conditions = [c for c, _ in pairs][::-1]
    weights = np.array([w for _, w in pairs])[::-1]
    fig, ax = plt.subplots(figsize=(8.4, max(2.8, 0.44 * len(pairs) + 1.8)))
    ax.barh(conditions, weights,
            color=['#e34948' if w < 0 else '#2a78d6' for w in weights],
            edgecolor=INK, linewidth=0.6)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_xlabel(f"Local weight — effect on P(class = {CLASSES[predicted]})")
    ax.set_title(f"LIME — {spec.name}\nrow {idx} ({label}):  true = "
                 f"{CLASSES[actual]},  predicted = {CLASSES[predicted]} "
                 f"(p = {proba[predicted]:.3f})\n"
                 f"local surrogate $R^2$ = {exp.score:.3f}",
                 fontsize=11, fontweight='bold', color=INK)
    ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    save_fig(fig, f"lime_{_fname(spec.name)}_row{idx}_{_slug(label)}",
             subdir="lime/per_instance")
    plt.show()
    plt.close(fig)

    print(f"\n  {spec.name} — row {idx} ({label})")
    print(f"    true = {CLASSES[actual]}   predicted = {CLASSES[predicted]}   "
          f"P = {proba[predicted]:.4f}")
    print(f"    local R2 = {exp.score:.4f}   intercept = {intercept:.4f}"
          + ("   << surrogate does not fit here; treat the weights as unreliable"
             if exp.score < 0.3 else ""))
    for cond, w in pairs:
        print(f"      {w:+9.4f}   {cond}")

    return [{"Model": spec.name, "Row": idx, "Case": label,
             "True class": CLASSES[actual], "Predicted class": CLASSES[predicted],
             "P(predicted)": float(proba[predicted]),
             "Local surrogate pred": local_pred, "Local R2": float(exp.score),
             "Intercept": intercept, "Condition": cond, "Weight": w}
            for cond, w in pairs]


if LIME_AVAILABLE:
    print("\n" + "=" * 78)
    print("LIME — LOCAL EXPLANATIONS")
    print("=" * 78)

    #  Built on the TRAINING data: that is the distribution the perturbations
    #  are drawn from and the bins are cut from. Building it on the test set
    #  would explain each row against a neighbourhood the model never saw.
    #  training_labels is passed so LIME can report class names properly.
    lime_explainer = LimeTabularExplainer(
        training_data   = np.asarray(X_tr, dtype=np.float64),
        training_labels = np.asarray(y_tr).astype(int),
        feature_names   = list(feature_names),
        class_names     = [str(c) for c in CLASSES],
        mode            = "classification",
        discretize_continuous = True,
        random_state    = SEED,
    )

    lime_rows = []
    for spec in LIME_MODELS:
        for idx, label in select_instances_to_explain(spec, X_te, y_te):
            try:
                lime_rows += explain_one(lime_explainer, spec,
                                         np.asarray(X_te, dtype=np.float64)[idx],
                                         idx, label)
            except Exception as exc:
                print(f"  LIME failed for {spec.name} row {idx}: "
                      f"{type(exc).__name__}: {exc}")

    if lime_rows:
        lime_df = pd.DataFrame(lime_rows)
        register_table("LIME local weights", lime_df)

        #  GLOBAL FROM LOCAL. Averaging |weight| per feature over many explained
        #  rows turns the local method into a global ranking that can be put
        #  next to SHAP and permutation importance. It is a weaker statement
        #  than either — it is an average of local linear fits, several of
        #  which may have had a poor local R2 — so the mean local R2 behind
        #  each figure is carried alongside it rather than dropped.
        def feature_of(condition):
            """Recover the feature name from a LIME condition like '2.1 < age <= 8'."""
            hits = [f for f in feature_names if f in condition]
            return max(hits, key=len) if hits else condition

        lime_df["Feature"] = lime_df["Condition"].map(feature_of)
        lime_global = (lime_df.assign(AbsWeight=lime_df["Weight"].abs())
                       .groupby("Feature")
                       .agg(**{"Mean |weight|": ("AbsWeight", "mean"),
                               "Std": ("AbsWeight", "std"),
                               "Explanations": ("AbsWeight", "size"),
                               "Mean local R2": ("Local R2", "mean")})
                       .sort_values("Mean |weight|", ascending=False))
        print("\nGlobal-from-local importance (mean |LIME weight| over "
              f"{lime_df[['Model', 'Row']].drop_duplicates().shape[0]} explanations):")
        display(lime_global.round(4))
        register_table("LIME global from local", lime_global, index=True)

        fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.45 * len(lime_global) + 1.4)))
        _g = lime_global.iloc[::-1]
        ax.barh(_g.index, _g["Mean |weight|"], xerr=_g["Std"].fillna(0.0),
                color=SERIES_COLORS[4 % len(SERIES_COLORS)], edgecolor=INK,
                linewidth=0.6, error_kw=dict(ecolor=INK, lw=1.0, capsize=3))
        ax.set_xlabel("Mean |LIME weight| across explained rows")
        ax.set_title("LIME — global importance aggregated from local explanations",
                     fontsize=12, fontweight='bold', color=INK)
        ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        save_fig(fig, "lime_global_from_local", subdir="lime")
        plt.show()
        plt.close(fig)


# =============================================================================
#  SECTION 14 — ICE & PARTIAL DEPENDENCE PLOTS
# =============================================================================
#  Model-agnostic interpretability, alongside SHAP in Part 7. SHAP attributes a
#  single prediction to each feature; ICE/PDP instead sweep a feature across
#  its range and show how the prediction RESPONDS. They answer different
#  questions and are worth reporting together.
#
#  REGRESSION -> CLASSIFICATION:
#    * the y-axis is now P(class), not the target's own units, so every curve
#      is bounded in [0, 1] and a flat curve at the class prior means "this
#      feature moves nothing".
#    * response_method='predict_proba' is mandatory. The default ('auto')
#      prefers decision_function where one exists, which would put the curves
#      on an unbounded log-odds scale for some models and probabilities for
#      others — not comparable across a grid of panels.
#    * `target` selects which class's probability is plotted for a multiclass
#      problem; for binary, sklearn always plots the positive class.
#    * the wrapper below is a ClassifierMixin and must expose classes_ and
#      predict_proba, or sklearn rejects it.
#
#  ---------------------------------------------------------------------------
#  WHY THE ADAPTER IS NECESSARY
#  ---------------------------------------------------------------------------
#  PartialDependenceDisplay.from_estimator() demands a fitted sklearn
#  classifier. That excludes 3 of the 9 base models (mlp, cnn_lstm, seq_logit
#  are Keras) and ALL 11 ensembles from Section 4B. _SpecEstimator wraps any
#  ModelSpec in the minimal sklearn surface sklearn actually checks, so every
#  model in this pipeline plots through one code path.
#
#  Note the base-class order: `ClassifierMixin, BaseEstimator`, NOT the
#  reverse. Since sklearn 1.6, is_classifier() resolves through
#  __sklearn_tags__ along the MRO, so putting BaseEstimator first makes sklearn
#  fail to recognise the wrapper as a classifier and raise "'estimator' must be
#  a fitted regressor or classifier." This is easy to get backwards and the
#  error message does not point at the cause.
#
#  ---------------------------------------------------------------------------
#  INTERPRETIVE CAVEAT
#  ---------------------------------------------------------------------------
#  PDP holds every other feature fixed and sweeps one across its marginal
#  range, which assumes the features are INDEPENDENT. Correlated features make
#  a grid point correspond to a combination that never occurs in reality; the
#  model still returns a number for it and PDP dutifully averages it in. This
#  bites especially hard on the ONE-HOT COLUMNS created in Section A: sweeping
#  a single dummy from 0 to 1 while its sibling dummies stay fixed constructs
#  rows that are in two categories at once, or in none. Read those panels as
#  the contrast between "has this level" and "does not", not as a curve.
#
#  Practical rules: (1) trust the MIDDLE of each curve far more than its tails;
#  (2) read ICE spread as signal — if the ICE curves fan out rather than
#  running parallel, the feature's effect depends on the rest of the row, i.e.
#  there are interactions, and the averaged PDP line is hiding them. The 2-way
#  plots below are the direct way to look at those. ALE plots are the standard
#  fix for correlated features.
# =============================================================================
from itertools import combinations
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import PartialDependenceDisplay

PDP_SAMPLE_SIZE   = 200    # rows used for 1-way ICE/PDP
PDP_2WAY_SAMPLE   = 100    # rows used for 2-way PDP
PDP_GRID_1WAY     = 50
PDP_GRID_2WAY     = 20     # 20x20 = 400 grid points per pair
PDP_TOP_K         = 6      # top-K features by SHAP -> C(6,2) = 15 pairs
N_COLS            = 3
PDP_TARGET        = None if IS_BINARY else SHAP_CLASS


class _SpecEstimator(ClassifierMixin, BaseEstimator):
    """Read-only sklearn classifier facade over an already-fitted ModelSpec."""
    def __init__(self, spec=None):
        self.spec = spec

    @property
    def classes_(self):
        # sklearn reads this to label the outputs and to decide how many
        # columns predict_proba should have.
        return np.arange(N_CLASSES)

    def fit(self, X, y=None):
        # Never trains anything. Present only because check_is_fitted() requires
        # the attribute to exist; the wrapped spec is already fitted.
        return self

    def __sklearn_is_fitted__(self):
        return self.spec is not None and self.spec.proba_fn is not None

    def predict_proba(self, X):
        return self.spec.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        return labels_from_proba(self.predict_proba(X))


def _pdp_frame(X):
    """
    A float64 view of X for sklearn's partial-dependence machinery.

    sklearn refuses to compute partial dependence on an integer column:

        ValueError: The column 0 contains integer data. Partial dependence
        plots are not supported for integer data: this can lead to implicit
        rounding with NumPy arrays or even errors with newer pandas versions.
        Please convert numerical features to floating point dtypes ahead of
        time to avoid problems.

    That is a guard against silent corruption, not a limitation of the
    method. PDP builds a grid of values for one feature and writes it back
    into a copy of X; if the column is integer-typed, every grid point is
    rounded on assignment, so the curve is computed at the wrong places and
    nothing warns you.

    It fires on the DATA, not on anything this pipeline does: read a CSV
    whose curing-age column is whole numbers and pandas types it int64, and
    the same goes for count-like or already-encoded categorical features.
    Every integer flavour is rejected — signed, unsigned, and pandas'
    nullable Int64 — while bool and float pass, which is why the failure can
    hide until one particular dataset is loaded. Casting here is the fix
    rather than retyping the training data, because the models are trained
    and evaluated on exactly what was loaded.

    Nothing about the plots changes. Every model already receives float32
    through _SpecEstimator, and integers of this magnitude are exact in
    float64, so a feature with fewer distinct values than grid_resolution
    still produces exactly the same grid points it always did. Non-numeric
    columns are left alone: they are a different problem, and sklearn's own
    message for them is clearer than a failed cast would be.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X), columns=feature_names[:np.shape(X)[1]])
    numeric = {c: np.float64 for c in X.columns
               if pd.api.types.is_numeric_dtype(X[c])}
    return X.astype(numeric) if numeric else X


_pdp_rng = np.random.RandomState(SEED)
_pdp_idx = _pdp_rng.choice(len(x_train), size=min(PDP_SAMPLE_SIZE, len(x_train)), replace=False)
X_pdp      = _pdp_frame(x_train.iloc[_pdp_idx] if hasattr(x_train, "iloc")
                        else pd.DataFrame(X_tr[_pdp_idx], columns=feature_names))
X_pdp_2way = X_pdp.iloc[:PDP_2WAY_SAMPLE]


# ── 1-WAY ICE + PDP ──────────────────────────────────────────────────────────
#  kind='both' overlays the per-sample ICE curves on the averaged PDP line.
#  centered=True starts every curve at 0 so the SHAPE of each response is
#  comparable instead of the curves being vertically scattered by baseline
#  differences.
ice_models = models     # base 9. Use all_models to include ensembles (much slower).


def plot_ice(specs, X_plot, features=None, n_cols=None):
    n_cols = N_COLS if n_cols is None else n_cols
    X_plot = _pdp_frame(X_plot)
    features = features if features is not None else list(range(len(feature_names)))
    n_rows = int(np.ceil(len(features) / n_cols))

    for spec in specs:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
        axes_flat = np.atleast_1d(axes).ravel()
        try:
            PartialDependenceDisplay.from_estimator(
                _SpecEstimator(spec), X_plot,
                features        = features,
                feature_names   = feature_names,
                target          = PDP_TARGET,
                response_method = 'predict_proba',
                kind            = 'both',
                subsample       = 50,
                centered        = True,
                grid_resolution = PDP_GRID_1WAY,
                random_state    = SEED,
                ice_lines_kw    = {'color': GRID_COLOR, 'alpha': 0.55, 'linewidth': 0.8},
                pd_line_kw      = {'color': ACCENT, 'linewidth': 2.5, 'label': 'Average'},
                ax              = axes_flat[:len(features)])
        except Exception as exc:
            print(f"  ICE/PDP unavailable for {spec.name}: {type(exc).__name__}: {exc}")
            plt.close(fig)
            continue

        for ax in axes_flat:
            if not ax.has_data():
                ax.set_visible(False)
                continue
            ax.grid(color=GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.8)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        fig.suptitle(f"ICE / partial dependence — {spec.name}  "
                     f"(P({CLASSES[SHAP_CLASS if not IS_BINARY else POS_LABEL]}))",
                     fontsize=14, fontweight='bold', color=INK)
        fig.tight_layout()
        save_fig(fig, f"ice_{_fname(spec.name)}", subdir="ice_pdp")
        plt.show()
        plt.close(fig)

        #  One figure per FEATURE as well. sklearn recomputes the partial
        #  dependence for each call, so this costs a second pass over the
        #  features per model — the reason it sits behind PER_ITEM_FIGURES.
        if PER_ITEM_FIGURES:
            for feat in features:
                f1, ax1 = plt.subplots(figsize=(5.2, 4.0))
                try:
                    PartialDependenceDisplay.from_estimator(
                        _SpecEstimator(spec), X_plot, features=[feat],
                        feature_names=feature_names, target=PDP_TARGET,
                        response_method='predict_proba', kind='both', subsample=50,
                        centered=True, grid_resolution=PDP_GRID_1WAY, random_state=SEED,
                        ice_lines_kw={'color': GRID_COLOR, 'alpha': 0.55, 'linewidth': 0.8},
                        pd_line_kw={'color': ACCENT, 'linewidth': 2.5}, ax=ax1)
                except Exception as exc:
                    print(f"    per-feature ICE skipped for {spec.name}/"
                          f"{feature_names[feat]}: {type(exc).__name__}")
                    plt.close(f1)
                    continue
                ax1.set_title(f"{feature_names[feat]}", fontsize=11,
                              fontweight='bold', color=INK)
                ax1.grid(color=GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.8)
                ax1.set_axisbelow(True)
                if ax1.get_legend() is not None:
                    ax1.get_legend().remove()
                f1.tight_layout()
                save_fig(f1, f"ice_{_fname(spec.name)}_{feature_names[feat]}",
                         subdir="ice_pdp/per_feature", close=True)


plot_ice(ice_models, X_pdp)


# ── 2-WAY PDP (feature interactions) ─────────────────────────────────────────
#  Which pairs to plot: all C(n, 2) pairs is too many, so the top PDP_TOP_K
#  features are selected by mean |SHAP| (consensus across the models explained
#  in Section 9) and all pairs of those are plotted. That makes the truncation
#  principled — the pairs shown are the ones the models actually rely on.
def top_features_by_shap(k=None):
    k = PDP_TOP_K if k is None else k
    try:
        imp = np.mean([np.abs(e.values).mean(axis=0) for e in SHAP_RESULTS.values()], axis=0)
        order = np.argsort(imp)[::-1][:k]
        print("Top features by mean |SHAP|:",
              [f"{feature_names[i]} ({imp[i]:.3f})" for i in order])
        return sorted(order.tolist())
    except (NameError, ValueError, IndexError):
        print(f"SHAP results unavailable — falling back to the first {k} features.")
        return list(range(min(k, len(feature_names))))


top_feats = top_features_by_shap()
pdp_pairs = list(combinations(top_feats, 2))
print(f"Plotting {len(pdp_pairs)} two-way interactions.")

#  Trees only by default: 2-way PDP is (rows x grid^2) predictions per pair, so
#  Keras models and especially ensembles get expensive fast. Filtered against
#  the roster for the same reason as the complexity sweeps above.
pdp_2way_models = [m for m in (rf, xgboost_model, lgbm) if any(x is m for x in models)]


def plot_pdp_2way(specs, X_plot, pairs=None, n_cols=None):
    """
    Two-way partial dependence, ONE FIGURE PER PAIR.

    A grid of fifteen contour plots is unreadable at any printable size, and
    each pair is a separate analysis, so each is written as its own file.
    sklearn is still called once per model with every pair — it computes them
    in one pass — and the panels are then redrawn individually, so splitting
    the output costs nothing extra. Two-way PD is (rows x grid^2) predictions
    per pair, which is the expensive part and is paid either way.
    """
    n_cols = N_COLS if n_cols is None else n_cols
    X_plot = _pdp_frame(X_plot)
    pairs = pairs if pairs is not None else pdp_pairs
    if not pairs:
        print("  fewer than two features — nothing to plot.")
        return
    n_rows = int(np.ceil(len(pairs) / n_cols))

    for spec in specs:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 5.0 * n_rows),
                                 constrained_layout=True)
        axes_flat = np.atleast_1d(axes).ravel()
        try:
            # sklearn computes the partial dependence; its default rendering is
            # then discarded and redrawn below as filled contours.
            display_obj = PartialDependenceDisplay.from_estimator(
                _SpecEstimator(spec), X_plot, features=pairs,
                feature_names=feature_names, target=PDP_TARGET,
                response_method='predict_proba', grid_resolution=PDP_GRID_2WAY,
                ax=axes_flat[:len(pairs)])
        except Exception as exc:
            print(f"  2-way PDP unavailable for {spec.name}: {type(exc).__name__}: {exc}")
            plt.close(fig)
            continue

        for idx, (ax, pd_result) in enumerate(zip(axes_flat[:len(pairs)],
                                                  display_obj.pd_results)):
            ax.clear()
            grid_values = pd_result.get('grid_values', pd_result.get('values'))
            Z = np.asarray(pd_result['average'])[0]

            #  ORIENTATION. sklearn returns average[0] with shape
            #  (len(grid[0]), len(grid[1])) — axis 0 is the FIRST feature.
            #  np.meshgrid's default indexing='xy' returns the transpose of
            #  that, so pairing them directly silently swaps the two axes. The
            #  usual `if Z.shape != X.shape: Z = Z.T` guard does NOT catch it,
            #  because with one grid_resolution for both features the shapes
            #  are equal and the check passes while the data is still
            #  transposed. indexing='ij' matches sklearn's layout exactly.
            G0, G1 = np.meshgrid(grid_values[0], grid_values[1], indexing='ij')

            #  Sequential single-hue fill: the surface encodes P(class), a
            #  magnitude, so it gets the same light->dark ramp as the confusion
            #  matrices rather than a rainbow.
            cf = ax.contourf(G0, G1, Z, levels=12, cmap=SEQ_BLUE)
            cbar = fig.colorbar(cf, ax=ax, pad=0.04)
            cbar.ax.tick_params(labelsize=9, colors=INK_SOFT)
            cbar.set_label(f"P({CLASSES[PDP_TARGET if PDP_TARGET is not None else POS_LABEL]})",
                           fontsize=9, color=INK_SOFT)

            f0, f1 = pairs[idx]
            ax.set_xlabel(feature_names[f0], fontsize=11, color=INK_SOFT)
            ax.set_ylabel(feature_names[f1], fontsize=11, color=INK_SOFT)
            ax.tick_params(axis='both', which='major', direction='out', length=4,
                           labelsize=9, colors=INK_SOFT)

        for j in range(len(pairs), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"2D partial dependence — {spec.name}",
                     fontsize=16, fontweight='bold', color=INK)
        save_fig(fig, f"pdp_2way_{_fname(spec.name)}", subdir="ice_pdp")
        plt.show()
        plt.close(fig)

        #  Re-plot each pair on its own, reusing the partial dependence sklearn
        #  already computed above — no extra model evaluations.
        if PER_ITEM_FIGURES:
            for idx, pd_result in enumerate(display_obj.pd_results):
                f0, f1i = pairs[idx]
                grid_values = pd_result.get('grid_values', pd_result.get('values'))
                Z = np.asarray(pd_result['average'])[0]
                G0, G1 = np.meshgrid(grid_values[0], grid_values[1], indexing='ij')
                fp, axp = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
                cf = axp.contourf(G0, G1, Z, levels=12, cmap=SEQ_BLUE)
                cb = fp.colorbar(cf, ax=axp, pad=0.04)
                cb.ax.tick_params(labelsize=9, colors=INK_SOFT)
                cb.set_label(
                    f"P({CLASSES[PDP_TARGET if PDP_TARGET is not None else POS_LABEL]})",
                    fontsize=9, color=INK_SOFT)
                axp.set_xlabel(feature_names[f0], fontsize=11, color=INK_SOFT)
                axp.set_ylabel(feature_names[f1i], fontsize=11, color=INK_SOFT)
                axp.set_title(f"{spec.name}: {feature_names[f0]} x {feature_names[f1i]}",
                              fontsize=11, fontweight='bold', color=INK)
                axp.tick_params(labelsize=9, colors=INK_SOFT)
                save_fig(fp, f"pdp2_{_fname(spec.name)}_{feature_names[f0]}"
                             f"_x_{feature_names[f1i]}",
                         subdir="ice_pdp/per_pair", close=True)


plot_pdp_2way(pdp_2way_models, X_pdp_2way)


# =============================================================================
#  SECTION 16 — PySR  (SYMBOLIC REGRESSION: A DECISION RULE YOU CAN READ)
# =============================================================================
#  pip install pysr
#
#  Everything above this point explains a model after the fact — SHAP,
#  permutation, LIME and PDP all take a fitted black box and interrogate it.
#  Symbolic regression skips that step: it searches the space of algebraic
#  expressions directly and returns a formula you can read, publish,
#  differentiate, and check against domain knowledge.
#
#  THE OBVIOUS PROBLEM: PySR IS A REGRESSOR. It fits a real-valued expression
#  by minimising a real-valued loss, and the target here is a class label.
#  Two ways out, and this section supports both through PYSR_TARGET:
#
#  "logit" (default) — fit the expression to the LOG-ODDS of the best model's
#      predicted probability. The result is a symbolic SURROGATE: a closed
#      form for the decision function the tuned pipeline already learned,
#      where f(x) > 0 means "predict the positive class" and |f(x)| is the
#      confidence. It fits the rest of Part 7, which is all about explaining
#      the fitted model, and it converges far better than fitting labels
#      directly, because log-odds is a smooth target while a 0/1 label is a
#      cliff. The surrogate is then scored against the TRUE labels below, so
#      its honesty as a classifier is measured, not assumed.
#
#  "label" — fit the labels directly with a margin loss (LogitDistLoss on
#      targets recoded to -1/+1), so the search optimises classification
#      rather than imitation. Use this when the equation is meant to stand on
#      its own as a model rather than explain another one. It is noisier and
#      usually needs a bigger budget.
#
#  Probabilities are clipped before the logit: p = 0 or 1 gives an infinite
#  target, and one infinity poisons every expression's loss. PYSR_LOGIT_CLIP
#  caps the log-odds at a finite, still-very-confident value.
#
#  MULTICLASS is handled one-vs-rest: one equation per class, each fitted to
#  that class's log-odds, and a prediction is the argmax over them. That means
#  K searches, so the budget below is per class.
#
#  WHAT PySR ACTUALLY RETURNS. Not one equation: a PARETO FRONT. Genetic
#  search evolves a population of expression trees and keeps, for every
#  complexity level, the best-scoring expression at that complexity. Choosing
#  a rung is a modelling decision, not a numerical one, so the whole front is
#  printed below. The interesting equation is usually two or three rungs BELOW
#  the most accurate one, where the loss has stopped improving much but the
#  formula still fits on a line. `score` is the marginal value of complexity:
#  the drop in log-loss per unit of extra complexity at that rung.
#
#  THE JULIA BACKEND. PySR wraps SymbolicRegression.jl, so the first
#  `import pysr` in a fresh environment downloads Julia and precompiles the
#  backend — minutes, and network access. That is why the import sits inside
#  the guard below rather than at the top of the file: an environment without
#  it should skip this section, not fail at import time hours into a run.
#
#  REPRODUCIBILITY. PySR is only deterministic with BOTH deterministic=True
#  AND parallelism="serial" AND a fixed random_state — it raises if you ask
#  for the first without the second, and merely WARNS if you set the seed
#  without either, which is the trap: a seeded-looking search that still
#  wanders between runs. PYSR_DETERMINISTIC sets all three together. Serial is
#  materially slower, so it is off by default and on when the number has to be
#  quotable.
# =============================================================================
PYSR_ENABLED       = True      # set False to skip this section entirely
PYSR_TARGET        = "logit"   # "logit" (surrogate) | "label" (direct classifier)
PYSR_NITERATIONS   = 40        # search iterations per class; the main quality knob
PYSR_POPULATIONS   = 15
PYSR_POPULATION_SZ = 33
PYSR_MAXSIZE       = 25        # largest expression the search may build
PYSR_TIMEOUT       = 300       # seconds per class; None for no cap
PYSR_DETERMINISTIC = False     # True -> serial + seeded + reproducible (slower)
PYSR_TOP_ROWS      = 15        # rungs of the Pareto front to print
PYSR_LOGIT_CLIP    = 12.0      # |log-odds| cap; 12 is p ~ 0.999994

#  OPERATORS. Every operator added multiplies the search space, so keep the set
#  small and sensible.
#
#  The names below are the ones PySR actually accepts, transcribed from its own
#  docs/src/operators.md. That distinction matters because PySR does NOT
#  validate operator names in Python — it passes them straight through to
#  Julia, where an unknown name fails inside the genetic search and comes back
#  as a `JuliaError` at fit() with a stack trace full of PythonCall frames and
#  no mention of the operator. `sqrt_abs` and `log_abs` are the specific trap:
#  they appear in pysr's export_sympy.py and look like "the protected
#  variants", but that table maps Julia output names BACK to sympy — it is the
#  read direction. Neither name is a valid input, and passing them is exactly
#  how you get that JuliaError.
#
#  Plain `sqrt` and `log` are the right choice and need no protection here.
#  SymbolicRegression.jl returns NaN rather than raising on a negative
#  argument, and the search "will preferentially select expressions which
#  avoid any invalid values over the training dataset" — a NaN costs an
#  expression its fitness, it does not poison the population.
PYSR_BINARY_OPS = ["+", "-", "*", "/"]
PYSR_UNARY_OPS  = ["square", "sqrt", "log"]

#  The full pre-defined sets, so a bad name is caught here with a useful
#  message instead of deep inside Julia. A string containing "=" is a custom
#  Julia definition (e.g. "myop(x) = x^2") and is passed through unchecked.
PYSR_UNARY_NAMES = {
    "neg", "square", "cube", "cbrt", "sqrt", "abs", "sign", "inv",
    "exp", "log", "log10", "log2", "log1p",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "erf", "erfc", "gamma", "relu", "sinc", "round", "floor", "ceil",
}
PYSR_BINARY_NAMES = {
    "+", "-", "*", "/", "^",
    "max", "min", ">", ">=", "<", "<=", "cond", "mod",
    "logical_or", "logical_and",
}


def check_pysr_operators(binary=None, unary=None):
    """Reject an unknown operator name here rather than inside Julia."""
    for ops, valid, kind in ((binary or [], PYSR_BINARY_NAMES, "binary"),
                             (unary or [], PYSR_UNARY_NAMES, "unary")):
        for op in ops:
            if "=" in op:          # a custom Julia definition, not a name
                continue
            if op not in valid:
                raise ValueError(
                    f"{op!r} is not a pre-defined PySR {kind} operator, and "
                    f"PySR would only fail on it inside Julia.\n"
                    f"Valid {kind} operators: {', '.join(sorted(valid))}.\n"
                    f"To use something else, pass it as Julia source with a "
                    f"sympy mapping, e.g. unary_operators=['myop(x) = x^2'] "
                    f"with extra_sympy_mappings={{'myop': lambda x: x**2}}.")


check_pysr_operators(PYSR_BINARY_OPS, PYSR_UNARY_OPS)


#  ── FEATURE NAMES PySR WILL ACCEPT ───────────────────────────────────────────
#  PySR validates variable names against ^[a-zA-Z0-9_]+$ and additionally
#  rejects anything colliding with a sympy function or one of its own operator
#  names. That is stricter than it sounds: a column called 'w/b' raises
#  "Invalid variable name", and so would 'max', 'abs', 'sign', 're' or 'beta'.
#  One-hot columns produced by encode_features are a common casualty, since
#  they carry the original category text. Renaming is therefore not optional,
#  and a silent rename would be worse than the crash — so the mapping is
#  printed whenever one happens, and the equations are shown with the safe
#  names so they stay copy-pasteable.
_PYSR_RESERVED = {
    "div", "inv", "mult", "sqrt", "sqrt_abs", "cbrt", "square", "cube", "plus",
    "sub", "neg", "pow", "pow_abs", "cos", "sin", "tan", "cosh", "sinh", "tanh",
    "exp", "acos", "asin", "atan", "acosh", "acosh_abs", "asinh", "atanh",
    "atanh_clip", "abs", "mod", "erf", "erfc", "log", "log10", "log2", "log1p",
    "log_abs", "log10_abs", "log2_abs", "log1p_abs", "floor", "ceil", "sign",
    "gamma", "round", "max", "min", "greater", "less", "greater_equal",
    "less_equal", "cond", "logical_or", "logical_and", "relu", "fma", "muladd",
    "clamp",
}


def pysr_safe_names(names):
    """
    Map feature names to ones PySR accepts. Returns (safe_names, renamed_dict).

    Non-alphanumerics become underscores, a leading digit gets an 'x' prefix,
    collisions with sympy/PySR function names get a trailing underscore, and
    any duplicate produced by that mangling is numbered so the mapping stays
    one-to-one.
    """
    import sympy
    safe, seen, renamed = [], set(), {}
    for original in names:
        name = re.sub(r"[^0-9a-zA-Z_]", "_", str(original)).strip("_")
        if not name or name[0].isdigit():
            name = "x_" + name
        while name in _PYSR_RESERVED or hasattr(sympy, name):
            name += "_"
        base, n = name, 2
        while name in seen:
            name, n = f"{base}_{n}", n + 1
        seen.add(name)
        safe.append(name)
        if name != str(original):
            renamed[name] = str(original)
    return safe, renamed


#  Equations are the point of this section, so the front is laid out as a real
#  table: the numbers in fixed columns and the expression starting at a fixed
#  offset, wrapping onto a hanging indent when it is too long for one line.
#  Printing the equation on its own line under the numbers (the obvious first
#  attempt) destroys the column alignment and makes two rungs impossible to
#  compare at a glance, which is the one thing this table exists for.
def _equation_row(mark, complexity, loss, score, equation, width=78):
    """
    One Pareto-front row: fixed numeric columns, equation hanging-indented.

    The continuation indent is measured from the row's own prefix rather than
    written as a constant, so it stays aligned even when a loss value renders
    wider than its column — a hardcoded offset drifts by a character the first
    time that happens, and a misaligned wrap is exactly as hard to read as no
    wrap at all.
    """
    head = f"{mark:2s} {complexity:>4d} {loss:>11.5g} {score:>7.4f}  "
    body = textwrap.wrap(str(equation), width=max(24, width - len(head)),
                         break_long_words=False, break_on_hyphens=False) or [""]
    return "\n".join([head + body[0]] + [" " * len(head) + b for b in body[1:]])


def _wrap_equation(text, indent=8, width=78):
    """Wrap a long equation on its spaces so no token is ever split."""
    return textwrap.fill(str(text), width=width,
                         initial_indent=" " * indent,
                         subsequent_indent=" " * (indent + 4),
                         break_long_words=False, break_on_hyphens=False)


def clipped_logit(p, clip=None):
    """
    log(p / (1-p)), finite everywhere.

    A model that returns a hard 0 or 1 — every tree ensemble does, on rows it
    finds easy — would otherwise give an infinite target, and a single
    infinity makes every candidate expression's loss infinite too, so the
    search returns nothing at all and gives no hint why.
    """
    clip = PYSR_LOGIT_CLIP if clip is None else clip
    lo = 1.0 / (1.0 + np.exp(clip))            # the p that maps to -clip
    p = np.clip(np.asarray(p, dtype=np.float64), lo, 1.0 - lo)
    return np.log(p / (1.0 - p))


def print_pareto_front(model, title, top=None):
    """
    Print the Pareto front as a readable ladder, simplest rung first.

    This is the part of a PySR run worth reading, and its default repr is wide
    enough to wrap badly in a notebook, so it is reformatted here: fixed-width
    columns, the equation on its own line so it can run long, and a marker on
    the rung `model_selection` actually chose.
    """
    top = PYSR_TOP_ROWS if top is None else top
    eqs = model.equations_
    if isinstance(eqs, list):
        eqs = eqs[0]
    chosen = model.get_best()
    chosen_complexity = int(chosen["complexity"])

    shown = eqs.tail(top) if len(eqs) > top else eqs
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"{'':2s} {'cplx':>4s} {'loss':>11s} {'score':>7s}  equation")
    print("-" * 78)
    for _, row in shown.iterrows():
        mark = ">>" if int(row["complexity"]) == chosen_complexity else "  "
        score = row["score"] if "score" in row else float("nan")
        print(_equation_row(mark, int(row["complexity"]), float(row["loss"]),
                            float(score), row["equation"]))
    print("-" * 78)
    print(f">> marks the rung chosen by model_selection='{model.model_selection}'.")
    print("   `score` is the drop in log-loss per unit of added complexity — a "
          "high\n   score means that rung paid for the extra terms.")
    return eqs, chosen


def print_chosen_equation(model, chosen, lhs, renamed, decision_note=None):
    """The selected equation, four ways: plain, sympy, LaTeX, and the rule."""
    print("\n" + "=" * 78)
    print(f"SELECTED EQUATION   (complexity {int(chosen['complexity'])}, "
          f"loss {chosen['loss']:.6g})")
    print("=" * 78)
    print(f"\n  {lhs} =")
    print(_wrap_equation(chosen["equation"], indent=6))

    try:
        print("\n  sympy (simplified):")
        print(_wrap_equation(model.sympy(), indent=6))
    except Exception as exc:
        print(f"    sympy form unavailable: {type(exc).__name__}: {exc}")

    try:
        print("\n  LaTeX (paste straight into a paper):")
        print(_wrap_equation(f"{lhs} = {model.latex(precision=4)}", indent=6))
    except Exception as exc:
        print(f"    LaTeX form unavailable: {type(exc).__name__}: {exc}")

    if decision_note:
        print(f"\n  Decision rule:\n      {decision_note}")
    if renamed:
        print("\n  Renamed for PySR (it rejects non-alphanumeric and "
              "function-name variables):")
        for safe, original in renamed.items():
            print(f"      {safe:>20s}  <-  {original}")
    print("\n" + "=" * 78)


def plot_pareto_front(eqs, chosen, title, filename):
    """Loss against complexity, with the selected rung marked."""
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(eqs["complexity"], eqs["loss"], marker='o', color=ACCENT,
            linewidth=1.8, markersize=5, label="Pareto front")
    ax.scatter([chosen["complexity"]], [chosen["loss"]], s=170, marker='*',
               color=SERIES_COLORS[1], zorder=5, edgecolor=INK, linewidth=0.7,
               label=f"selected (complexity {int(chosen['complexity'])})")
    ax.set_yscale("log")
    ax.set_xlabel("Complexity (number of nodes in the expression tree)")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(title, fontsize=12, fontweight='bold', color=INK)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, filename, subdir="pysr")
    plt.show()
    plt.close(fig)


def make_pysr_model():
    """One configured PySRRegressor. Built per class so searches stay independent."""
    parallelism = "serial" if PYSR_DETERMINISTIC else "multithreading"
    return PySRRegressor(
        niterations      = PYSR_NITERATIONS,
        populations      = PYSR_POPULATIONS,
        population_size  = PYSR_POPULATION_SZ,
        maxsize          = PYSR_MAXSIZE,
        binary_operators = PYSR_BINARY_OPS,
        unary_operators  = PYSR_UNARY_OPS,
        elementwise_loss = ("L2DistLoss()" if PYSR_TARGET == "logit"
                            else "LogitDistLoss()"),
        model_selection  = "best",
        timeout_in_seconds = PYSR_TIMEOUT,
        parallelism      = parallelism,
        deterministic    = PYSR_DETERMINISTIC,
        random_state     = SEED if PYSR_DETERMINISTIC else None,
        progress         = False,
        verbosity        = 0,
        temp_equation_file = True,
    )


def fit_pysr(model, X, y, label=""):
    """
    Run one PySR search, turning a Julia-side failure into something readable.

    A JuliaError arrives as a wall of PythonCall frames with the actual cause
    buried, and in a notebook the middle of that traceback is usually elided
    entirely — so the one line that says what went wrong is the line you do
    not get to see. The causes are few and each has a specific fix, so they
    are named here rather than left to be rediscovered.
    """
    t0 = time.time()
    try:
        model.fit(X, y)
    except Exception as exc:
        name = type(exc).__name__
        print(f"\n  PySR search failed for {label or 'the target'} "
              f"({name}).\n"
              f"  The message below is Julia's; the usual causes are:\n"
              f"    - an operator name Julia does not define (check_pysr_"
              f"operators above catches\n"
              f"      the pre-defined ones, but a typo inside a custom "
              f"'myop(x) = ...' string reaches Julia);\n"
              f"    - a non-finite value in X or y — PySR rejects NaN and inf "
              f"in the target;\n"
              f"    - a Julia backend that did not finish precompiling; "
              f"re-running the import often\n"
              f"      clears it, and `python -c \"import pysr\"` shows the "
              f"real setup error on its own.\n")
        finite_X = bool(np.isfinite(np.asarray(X, dtype=np.float64)).all())
        finite_y = bool(np.isfinite(np.asarray(y, dtype=np.float64)).all())
        print(f"  Checked for you: X all finite = {finite_X}, "
              f"y all finite = {finite_y}, "
              f"n = {len(np.asarray(y).ravel())}, "
              f"unary = {model.unary_operators}, binary = {model.binary_operators}")
        raise
    print(f"Search finished in {time.time() - t0:.1f}s.")
    return model


if PYSR_ENABLED:
    try:
        from pysr import PySRRegressor
        PYSR_AVAILABLE = True
    except Exception as exc:
        PYSR_AVAILABLE = False
        print(f"PySR unavailable ({type(exc).__name__}: {exc}).\n"
              "  pip install pysr   — note the first import also downloads "
              "Julia and precompiles\n  the backend, which needs network "
              "access and a few minutes.")
else:
    PYSR_AVAILABLE = False
    print("PYSR_ENABLED is False — skipping Section 16.")


if PYSR_AVAILABLE:
    pysr_names, pysr_renamed = pysr_safe_names(feature_names)
    X_sym_tr = pd.DataFrame(np.asarray(X_tr, dtype=np.float64), columns=pysr_names)
    X_sym_te = pd.DataFrame(np.asarray(X_te, dtype=np.float64), columns=pysr_names)
    if pysr_renamed:
        print("Renamed for PySR:  "
              + ",  ".join(f"{o} -> {s}" for s, o in pysr_renamed.items()))

    #  One search per class for multiclass; binary needs only the positive one,
    #  because the negative class's log-odds is its exact negation.
    _classes_to_fit = [POS_LABEL] if IS_BINARY else list(range(N_CLASSES))
    _surrogate_of = best_spec if PYSR_TARGET == "logit" else None
    if _surrogate_of is not None:
        _proba_tr = _surrogate_of.predict_proba(X_tr)
        print(f"\nFitting a symbolic surrogate to the log-odds of "
              f"{_surrogate_of.name} (the best model by ROC-AUC).")
    else:
        print("\nFitting the labels directly with a margin loss "
              "(PYSR_TARGET='label').")

    PYSR_MODELS, pysr_front_rows = {}, []
    for k in _classes_to_fit:
        label = str(CLASSES[k])
        if PYSR_TARGET == "logit":
            target = clipped_logit(_proba_tr[:, k])
            lhs = f"logit P({TARGET_COL} = {label})"
        else:
            target = np.where(np.asarray(y_tr).astype(int) == k, 1.0, -1.0)
            lhs = f"margin for {TARGET_COL} = {label}"

        print(f"\nSearching for class {label} "
              f"({PYSR_NITERATIONS} iterations"
              + (f", {PYSR_TIMEOUT}s cap" if PYSR_TIMEOUT else "") + ") ...")
        model = make_pysr_model()
        fit_pysr(model, X_sym_tr, target, label=f"class {label}")
        PYSR_MODELS[k] = model

        eqs, best = print_pareto_front(
            model, f"PySR PARETO FRONT — class {label}")
        print_chosen_equation(
            model, best, lhs, pysr_renamed,
            decision_note=(f"predict {label} when the expression is greater than "
                           f"every other class's" if not IS_BINARY else
                           f"predict {label} when the expression > 0, "
                           f"otherwise {CLASSES[1 - POS_LABEL]}"))

        front = eqs[["complexity", "loss", "equation"]].copy()
        if "score" in eqs.columns:
            front["score"] = eqs["score"]
        front.insert(0, "Class", label)
        front["selected"] = (front["complexity"].astype(int)
                             == int(best["complexity"]))
        pysr_front_rows.append(front)
        plot_pareto_front(eqs, best,
                          f"PySR Pareto Front — class {label}",
                          f"pysr_pareto_front_{_slug(label)}")

    register_table("PySR Pareto front", pd.concat(pysr_front_rows,
                                                  ignore_index=True))
    if pysr_renamed:
        register_table("PySR variable names", pd.DataFrame(
            [{"PySR name": s, "Original feature": o}
             for s, o in pysr_renamed.items()]))

    # ── DOES THE EQUATION ACTUALLY CLASSIFY? ─────────────────────────────────
    #  The surrogate was fitted to imitate a model, so the number that matters
    #  is not how well it reproduces the log-odds but how well the resulting
    #  RULE classifies the real labels. Scored the same way as every model in
    #  Part 6, so it drops straight into the comparison table.
    def symbolic_proba(X_sym):
        """Class probabilities implied by the symbolic expressions."""
        raw = np.column_stack([PYSR_MODELS[k].predict(X_sym)
                               for k in _classes_to_fit])
        if IS_BINARY:
            p_pos = 1.0 / (1.0 + np.exp(-np.clip(raw[:, 0], -PYSR_LOGIT_CLIP,
                                                 PYSR_LOGIT_CLIP)))
            P = np.column_stack([1.0 - p_pos, p_pos])
            return P if POS_LABEL == 1 else P[:, ::-1]
        raw = raw - raw.max(axis=1, keepdims=True)    # softmax over the one-vs-rest scores
        e = np.exp(np.clip(raw, -PYSR_LOGIT_CLIP, PYSR_LOGIT_CLIP))
        return e / e.sum(axis=1, keepdims=True)

    _sym_proba_te = symbolic_proba(X_sym_te)
    _sym_proba_tr = symbolic_proba(X_sym_tr)
    pysr_metrics = pd.DataFrame([
        dict(Split="train", **compute_metrics(np.asarray(y_tr).astype(int),
                                              labels_from_proba(_sym_proba_tr),
                                              _sym_proba_tr)),
        dict(Split="test", **compute_metrics(np.asarray(y_te).astype(int),
                                             labels_from_proba(_sym_proba_te),
                                             _sym_proba_te)),
    ])
    print("\nSymbolic rule scored against the TRUE labels:")
    display(pysr_metrics.round(4))
    register_table("PySR metrics", pysr_metrics)

    if PYSR_TARGET == "logit":
        _agree = float(np.mean(labels_from_proba(_sym_proba_te)
                               == _surrogate_of.predict(X_te)))
        print(f"\nThe surrogate reproduces {_surrogate_of.name}'s own test "
              f"predictions {_agree:.1%} of the time.")
        print("Fidelity to the model and accuracy against the truth are "
              "different numbers;\nboth are reported because a surrogate can "
              "be faithful to a wrong model.")

    _bench = summary_table()[["ROC_AUC", "F1"]].copy()
    _bench.loc["PySR (symbolic)"] = [pysr_metrics.loc[1, "ROC_AUC"],
                                     pysr_metrics.loc[1, "F1"]]
    _bench = _bench.sort_values("ROC_AUC", ascending=False)
    print("\nWhere the equation lands against the tuned models (test ROC-AUC):")
    display(_bench.round(4))
    register_table("PySR vs models", _bench, index=True)

    # ── CONFUSION MATRIX FOR THE SYMBOLIC RULE ───────────────────────────────
    #  Drawn through the same _draw_cm helper as every other confusion matrix
    #  in the file — counts beside row-normalised recall — so the equation can
    #  be compared with the tuned models panel for panel.
    _sym_pred = labels_from_proba(_sym_proba_te)
    _cm = confusion_matrix(np.asarray(y_te).astype(int), _sym_pred,
                           labels=list(range(N_CLASSES)))
    with np.errstate(invalid='ignore'):
        _cm_norm = np.nan_to_num(_cm / _cm.sum(axis=1, keepdims=True)) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    _draw_cm(axes[0], _cm, "Counts", "d", max(1, _cm.max()))
    _draw_cm(axes[1], _cm_norm, "Row-normalised (recall %)", ".1f", 100.0)
    fig.suptitle("PySR symbolic rule — test", fontsize=13, fontweight='bold',
                 color=INK)
    fig.tight_layout()
    save_fig(fig, "pysr_confusion_test", subdir="pysr")
    plt.show()
    plt.close(fig)


# =============================================================================
#  SECTION 15 — EXPORT
# =============================================================================
#  Every table registered anywhere above goes into ONE workbook, one sheet per
#  table, rather than a scatter of CSVs: a run's numbers stay together, cannot
#  be mismatched across runs, and open in the tool most people actually use to
#  read them. Figures are already written per analysis (and per feature, metric
#  or pair) under outputs/figures/.
# =============================================================================
register_table("Ensemble composition", pd.DataFrame([
    {"Ensemble": spec.name,
     "Members": ", ".join((spec.best_params or {}).get("members", [])),
     "Combination": (spec.best_params or {}).get("meta")
                    or (spec.best_params or {}).get("weighting", ""),
     "Weights / blend": (spec.best_params or {}).get("weights")
                        or (spec.best_params or {}).get("blend_weights", "")}
    for spec in ensemble_models]))

register_table("Run configuration", pd.DataFrame([
    {"setting": "tuning profile",        "value": TUNING_PROFILE},
    {"setting": "primary metric",        "value": PRIMARY_METRIC},
    {"setting": "sklearn scoring",       "value": SCORING},
    {"setting": "resampling",            "value": str(RESAMPLING)},
    {"setting": "split seed",            "value": BEST_SEED},
    {"setting": "test size",             "value": TEST_SIZE},
    {"setting": "classes",               "value": ", ".join(map(str, CLASSES))},
    {"setting": "positive class",        "value": str(CLASSES[POS_LABEL]) if IS_BINARY else "n/a"},
    {"setting": "features",              "value": N_FEATURES},
    {"setting": "train rows",            "value": len(y_tr)},
    {"setting": "test rows",             "value": len(y_te)},
    {"setting": "models",                "value": ", ".join(m.name for m in models)},
]))

export_tables()

print(f"\n{len(SAVED_FIGURES)} figure(s) written under {FIGURE_DIR}/")
_by_dir = {}
for _path in SAVED_FIGURES:
    _by_dir.setdefault(os.path.dirname(_path), []).append(_path)
for _d in sorted(_by_dir):
    print(f"    {_d}/  ({len(_by_dir[_d])} figures)")

print("\n" + "=" * 78)
print(f"DONE. Tables -> {RESULTS_XLSX}   Figures -> {FIGURE_DIR}/")
print(f"Best model by test ROC-AUC: {best_spec.name} "
      f"({RESULTS[best_spec.name]['ROC_AUC']:.4f})")
print("=" * 78)

# =============================================================================
#  TUNED REGRESSION BENCHMARK — Concrete Compressive Strength
#  9 models, leakage-free CV, nested CV, uniform fit/predict/evaluate interface,
#  RMSE/MAE/MAPE/SI/R2 metrics, and a K-fold count stability sweep.
# =============================================================================
#  Assumes x_train, y_train, x_test, y_test already exist (as in your notebook).
#  Target: compressive strength (MPa). Typical UCI Concrete features (in this
#  order): Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer,
#  Coarse Aggregate, Fine Aggregate, Age. Tabular, i.i.d. mix designs — plain
#  shuffled K-Fold is the correct CV scheme throughout.
#
#  End result: a plain list you can loop over —
#      models = [rf, mlp, svr, xgboost_model, lgbm, ada, knn, cnn_lstm, seq_linear]
#      fit_all(models, X_tr, y_tr)
#      evaluate_all(models, X_te, y_te)
#  Any single model still works standalone: svr.fit(X_tr, y_tr); svr.predict(X_te)
#
#  pip install -U scikit-learn xgboost lightgbm optuna tensorflow matplotlib
# =============================================================================

# =============================================================================
# =============================================================================
#  This file is organized into 7 parts:
#    Part 1 — Prerequisites, library imports & common setup
#              (data loading + seed selection, global imports/constants,
#               metric functions, the ModelSpec interface)
#    Part 2 — Search strategies (why each model uses Grid/Random/Optuna)
#    Part 3 — Exploratory data analysis (summary stats, histograms, scatter,
#              box plots, correlation heatmaps)
#    Part 4 — Model space (the 9 model definitions + hyperparameter spaces)
#    Part 5 — Hyperparameter optimization (fit_all + K-fold stability sweep)
#    Part 6 — Evaluation metrics (test-set comparison, learning curves,
#              train-vs-test table, actual-vs-predicted plots,
#              bias-variance tradeoff, seed sensitivity)
#    Part 7 — Factor importance (SHAP)
#  Original Section 0-12/A labels are kept as sub-headers within each Part
#  for continuity; they no longer run in numeric order.
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

# `display` is injected by IPython/Colab. The shim keeps the file runnable as a
# plain script too, so nothing below depends on being inside a notebook.
try:
    display
except NameError:
    def display(*objs):
        for o in objs:
            print(o)


SEED = 42
np.random.seed(SEED)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# FEATURE-SCALER KNOB. Set to MinMaxScaler to min-max scale instead of
# standardizing. Used for every FEATURE scaler in this file — inside the KNN/SVR
# Pipelines and inside the per-fold Keras scaling — so switching it here changes
# every model at once WITHOUT any leakage: the scaler is still refit on the
# training part of each fold rather than on the full training set up front.
# (See Section A for why pre-scaling x_train/x_test globally is the thing to avoid.)
# The Keras TARGET scaler stays StandardScaler regardless: zero-mean/unit-variance
# targets are what the networks' initialization and learning rates are tuned around.
SCALER_CLS = StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
)


# Shuffled K-Fold: rows are independent mix-design trials, no temporal or
# grouping structure to protect against, so plain KFold is appropriate.
inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)   # picks hyperparameters
outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)   # nested-CV generalisation estimate

SCORING = "neg_root_mean_squared_error"   # sklearn maximises, so RMSE is negated
N_ITER_RANDOM = 60      # RandomizedSearchCV draws
N_TRIALS = {"mlp": 40, "xgb": 80, "lgbm": 80, "svr": 60, "cnn_lstm": 30, "seq_linear": 25}

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
#  mismatch; a single workbook keeps a run's numbers together.
# =============================================================================
OUTPUT_DIR   = "outputs"
FIGURE_DIR   = os.path.join(OUTPUT_DIR, "figures")
RESULTS_XLSX = os.path.join(OUTPUT_DIR, "results.xlsx")

RESULT_TABLES = {}      # sheet name -> DataFrame, written by export_tables()
SAVED_FIGURES = []      # every path written, reported at the end

#  Write one figure per feature / per metric / per pair in addition to the scan
#  grids. Costs a little time in the EDA (the data is already computed) and one
#  extra partial-dependence pass per model in Part 7; set False for grids only.
PER_ITEM_FIGURES = True


def _slug(name):
    """A filename that survives every OS: no spaces, slashes or punctuation."""
    out = re.sub(r"[^\w\-.]+", "_", str(name).strip())
    return re.sub(r"_+", "_", out).strip("_") or "figure"


def _fname(name):
    """A model name as a filename fragment."""
    return name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')


def save_fig(fig=None, name="figure", subdir="", dpi=300, close=False):
    """
    Write one figure to outputs/figures/<subdir>/<name>.png.

    Centralised so resolution, background and bounding box are identical
    everywhere, and so the full list of outputs can be reported at the end.

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
    Excel sheet names: 31 characters, and none of : \\ / ? * [ ].
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


# ---- Metrics -----------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error, as a percentage. Assumes y_true > 0,
    which holds for compressive strength (MPa)."""
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100)


def scatter_index(y_true, y_pred):
    """
    SI = RMSE / mean(y_true), expressed as a percentage. A normalized-error
    measure common in engineering/materials regression papers because, unlike
    raw RMSE, it's comparable across targets/datasets with different scales.
    Rough rule of thumb in the concrete-strength literature: SI below ~10-15%
    is considered a good fit — a loose guide, not a hard threshold.
    """
    return float(rmse(y_true, y_pred) / np.mean(y_true) * 100)


def compute_metrics(y_true, y_pred):
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "SI":   scatter_index(y_true, y_pred),
    }


def report(name, y_true, y_pred, best_params=None, cv_rmse=None, elapsed=None):
    m = compute_metrics(y_true, y_pred)
    m["CV_RMSE"] = cv_rmse
    m["fit_time_s"] = elapsed
    m["best_params"] = best_params
    RESULTS[name] = m
    print(f"\n=== {name} ===")
    if best_params:
        print("best params :", best_params)
    if cv_rmse is not None:
        print(f"inner-CV RMSE : {cv_rmse:.4f}")
    print(f"test RMSE : {m['RMSE']:.4f} MPa | MAE : {m['MAE']:.4f} MPa | "
          f"MAPE : {m['MAPE']:.2f}% | SI : {m['SI']:.2f}% | R2 : {m['R2']:.4f}")
    return m


def nested_cv_rmse(make_search, X, y, cv=None, label=""):
    """
    Unbiased estimate of the ENTIRE tune-then-fit pipeline: re-runs the search
    from scratch inside each outer fold. Inner-CV scores alone are optimistic
    because they're what the search maximised; this corrects for that.
    `make_search` must be a zero-arg factory returning a *fresh* search object.
    """
    cv = cv or outer_cv
    scores = []
    for k, (tr, te) in enumerate(cv.split(X), 1):
        s = make_search()
        s.fit(X[tr], y[tr])
        scores.append(rmse(y[te], s.best_estimator_.predict(X[te])))
        print(f"  [{label}] outer fold {k}: RMSE={scores[-1]:.4f}")
    scores = np.array(scores)
    print(f"  [{label}] nested CV RMSE = {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores.mean(), scores.std()


# -----------------------------------------------------------------------------
#  Scaling always lives inside a Pipeline (or is refit per-fold for Keras).
#  Fitting a scaler on the full training set before CV leaks each fold's
#  held-out mean/std into training. Mandatory for KNN / SVR / MLP / linear;
#  harmless (unused) for tree models.
# -----------------------------------------------------------------------------



# =============================================================================
#  SECTION A — DATA LOADING, SEED SELECTION, AND TRAIN/TEST SPLIT
# =============================================================================
#  Runs right after the library-imports/constants half of Part 1's common setup
#  (so pandas/numpy are already available) and right before the data-array casts
#  (X_tr/X_te/y_tr/y_te), which need x_train/x_test/y_train/y_test to exist
#  first. This section produces those four DataFrames/Series; the array casts
#  immediately below convert them into what every later Part actually uses.
#
#  ---------------------------------------------------------------------------
#  WHY THE SEED IS SELECTED ON DISTRIBUTION, NOT ON MODEL PERFORMANCE
#  ---------------------------------------------------------------------------
#  You asked for a seed that keeps the data "representative and not biased".
#  There are two very different ways to pick one, and only one is defensible:
#
#    (a) Score each seed by how closely the TRAIN and TEST marginal
#        distributions match — per-column two-sample Kolmogorov-Smirnov
#        statistics on the features and the target. This uses no model and no
#        predictions. It is essentially automated stratification, and it is
#        what this section does.
#
#    (b) Score each seed by test-set R2/RMSE of a fitted model and keep the
#        best. DO NOT DO THIS. Choosing the split that maximises test score
#        turns the test set into a selection criterion, so the reported test
#        metric is no longer an estimate of generalisation — it is the maximum
#        of many draws, biased upward, and it will not reproduce. That is
#        test-set leakage and reviewers do check for it.
#
#  Even with the honest version (a), there is a caveat worth stating in your
#  write-up: metrics from a deliberately well-balanced split are mildly
#  optimistic relative to a single arbitrary split, because you removed the
#  unlucky draws. The most defensible reporting is to fix BEST_SEED for all
#  the plots and tables (so everything is mutually consistent and
#  reproducible), and separately report mean +/- std of test metrics across
#  several seeds via SEED_SENSITIVITY below, so the reader can see the split
#  did not do the work. Both are provided.
# =============================================================================
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
from scipy.stats import (pearsonr as st_pearsonr, spearmanr as st_spearmanr,
                         kendalltau as st_kendalltau)

# ---- Load the dataset -------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# Replace with the actual path to your CSV file in Google Drive.
file_path = '/content/drive/MyDrive/CS_ML_JR_DT.csv'

try:
    cs = pd.read_csv(file_path)
    print("File loaded successfully from Google Drive:")
    display(cs.head())
except FileNotFoundError:
    print(f"Error: The file at '{file_path}' was not found. Please check the path.")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")

# >>> FLAG: `concrete` is not defined anywhere in the code you gave me — only
# `cs` (just loaded above) exists at this point, so this line will raise
# NameError as written. Two likely fixes, depending on what you meant:
#   - If you meant to drop 'w/b' from the DataFrame you just loaded:
#         cs = cs.drop('w/b', axis=1)
#   - If `concrete` (and `concrete_2`, per the commented-out line below) are
#     separate raw DataFrames loaded elsewhere in your notebook, that loading
#     code needs to run before this line.
# Left as you wrote it so nothing is silently changed — fix the line below
# once you confirm which case applies.
cs = concrete.drop('w/b', axis=1)
#cs_2 = concrete_2.drop('w/b', axis=1)

print('First 5 rows of the concrete DataFrame:')
display(cs.head())

print('\nData types and non-null counts of the concrete DataFrame:')
cs.info()

cs.isnull().sum()

# ---- `cs` -> X / Y -----------------------------------------------------------
# `cs` is your raw DataFrame. Adjust TARGET_COL to your target column name.
TARGET_COL = 'CS'
TEST_SIZE  = 0.3

X = cs.drop(TARGET_COL, axis=1)
Y = cs[TARGET_COL]


def split_imbalance(X_df, y_ser, seed, test_size=None):
    """
    Two-sample KS statistic between the train and test halves, computed
    per column (all features + the target). KS is distribution-free and
    catches shifts in location, spread, AND shape, which a mean/std check
    alone would miss. Returns (worst_column_KS, mean_KS).
    Lower is better; 0 would mean identical empirical distributions.
    """
    test_size = TEST_SIZE if test_size is None else test_size
    Xa, Xb, ya, yb = train_test_split(X_df, y_ser, test_size=test_size, random_state=seed)
    ks = [ks_2samp(Xa[c], Xb[c]).statistic for c in X_df.columns]
    ks.append(ks_2samp(ya, yb).statistic)
    return float(np.max(ks)), float(np.mean(ks))


def find_best_seed(X_df, y_ser, candidate_seeds=range(1000), test_size=None, top_n=5):
    """
    Minimax criterion: pick the seed whose WORST-matched column is best
    matched, tie-broken on the mean. Minimising the worst column (rather than
    the average) is the point — an average can stay low while one feature is
    badly split, which is exactly the 'biased split' case you want to avoid.
    """
    test_size = TEST_SIZE if test_size is None else test_size
    rows = [(s, *split_imbalance(X_df, y_ser, s, test_size)) for s in candidate_seeds]
    scores = (pd.DataFrame(rows, columns=['seed', 'max_KS', 'mean_KS'])
                .sort_values(['max_KS', 'mean_KS'])
                .reset_index(drop=True))
    print(f"\nScanned {len(rows)} seeds. Best {top_n} by worst-column KS:")
    print(scores.head(top_n).round(4).to_string(index=False))
    #  Guarded: narrowing candidate_seeds to shorten the scan is the obvious way
    #  to speed this section up, and if the narrowed range excludes 42 this
    #  reference line would raise IndexError on an empty selection.
    if (scores['seed'] == 42).any():
        ref = scores.loc[scores.seed == 42].iloc[0]
        print(f"\nFor reference, seed=42 -> max_KS={ref['max_KS']:.4f}, "
              f"mean_KS={ref['mean_KS']:.4f}")
    return int(scores.iloc[0]['seed']), scores


BEST_SEED, seed_scan = find_best_seed(X, Y)
print(f"\n{'=' * 78}\nBEST_SEED = {BEST_SEED}   "
      f"(reuse this to reproduce the exact split)\n{'=' * 78}")

# ---- The split everything downstream uses ----------------------------------
#  Kept as DataFrames on purpose. Column names are consumed by Section 9 (SHAP
#  feature_names) and Section 10 (the EDA table) — both check
#  hasattr(x_train, "columns") and silently fall back to Feature_0, Feature_1,
#  ... if they are lost.
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=BEST_SEED)
x_train = pd.DataFrame(x_train, columns=X.columns)
x_test  = pd.DataFrame(x_test,  columns=X.columns)

print(f"\nTrain: {x_train.shape}   Test: {x_test.shape}")
print(f"Target mean  — train: {y_train.mean():.3f}   test: {y_test.mean():.3f}")
print(f"Target std   — train: {y_train.std():.3f}   test: {y_test.std():.3f}")


# -----------------------------------------------------------------------------
#  ON MinMaxScaler
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
#   1. LEAKAGE. It fits the scaler on the whole training set, then Sections
#      1-9 cross-validate inside that same training set — so every validation
#      fold's min and max have already leaked into the transform, and the CV
#      scores that drive all the hyperparameter tuning come out optimistic.
#      This is the exact failure the Pipeline-wrapped scalers were built to
#      avoid; per-fold refitting is the whole point.
#   2. DOUBLE SCALING. The KNN and SVR Pipelines and the Keras helpers already
#      scale internally, so they would be scaling data that is already scaled.
#   3. LOST COLUMN NAMES. fit_transform returns a bare numpy array, so
#      x_train.columns disappears and SHAP (Section 9) and the EDA table
#      (Section 10) silently fall back to Feature_0, Feature_1, ...
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
y_tr = np.asarray(y_train, dtype=np.float32).ravel()
y_te = np.asarray(y_test,  dtype=np.float32).ravel()

N_FEATURES = X_tr.shape[1]

# =============================================================================
#  SECTION 1 — UNIFORM MODEL INTERFACE
# =============================================================================
#  Every model below — sklearn estimator, Optuna-tuned booster, or Keras net —
#  gets wrapped in a ModelSpec so they can all be tuned, fit, and evaluated the
#  same way, regardless of what's happening internally:
#
#      spec.fit(X_tr, y_tr)        # runs hyperparameter search + final refit
#      spec.predict(X_te)          # raw features in, raw-units predictions out
#      spec.evaluate(X_te, y_te)   # predict + score + store in RESULTS
#
#  Tuning and refitting are DELIBERATELY split into two functions:
#      search_fn(X, y)         -> (best_params, cv_rmse)
#      refit_fn(X, y, params)  -> (fitted_model_or_None, predict_fn)
#  `fit()` just calls them in sequence. The split matters for Section 5's
#  K-fold stability sweep: refitting a fresh model with hyperparameters that
#  are already known is cheap, whereas re-running Optuna/GridSearchCV at every
#  one of 9 fold counts for 9 models would mean thousands of extra searches.
#  spec.refit_on(X, y) reuses refit_fn directly, skipping the search.
# =============================================================================
class ModelSpec:
    def __init__(self, name, search_fn, refit_fn, search_factory=None):
        self.name = name
        self._search_fn = search_fn     # (X, y) -> (best_params, cv_rmse)
        self._refit_fn = refit_fn       # (X, y, params) -> (model, predict_fn)
        self.search_factory = search_factory   # optional: enables .nested_cv()
        self.model = None
        self.predict_fn = None
        self.best_params = None
        self.cv_rmse = None
        self.fit_time_s = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        t0 = time.time()
        self.best_params, self.cv_rmse = self._search_fn(X, y)
        self.model, self.predict_fn = self._refit_fn(X, y, self.best_params)
        self.fit_time_s = time.time() - t0
        return self

    def predict(self, X):
        if self.predict_fn is None:
            raise RuntimeError(f"{self.name}: call .fit(X_tr, y_tr) before .predict().")
        return self.predict_fn(np.asarray(X, dtype=np.float32))

    def evaluate(self, X_te, y_te):
        pred = self.predict(X_te)
        return report(self.name, y_te, pred, self.best_params, self.cv_rmse, self.fit_time_s)

    def refit_on(self, X, y, params=None):
        """Refit a FRESH model using already-known hyperparameters (skips the
        search entirely). Used by the Section 5 K-fold stability sweep."""
        params = params or self.best_params
        if params is None:
            raise RuntimeError(f"{self.name}: no tuned hyperparameters yet — call .fit() first.")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        _, predict_fn = self._refit_fn(X, y, params)
        return predict_fn

    def nested_cv(self, X, y, cv=None):
        """Only available for models tuned via Grid/RandomizedSearchCV (rf, ada,
        knn) — re-running an Optuna study per outer fold is expensive, so it's
        opt-in rather than wired up by default for the boosted/NN models."""
        if self.search_factory is None:
            print(f"{self.name}: nested CV not wired up for this model "
                  f"(Optuna-tuned — rerun manually with nested_cv_rmse if needed).")
            return None
        return nested_cv_rmse(self.search_factory, np.asarray(X, dtype=np.float32),
                              np.asarray(y, dtype=np.float32).ravel(), cv=cv, label=self.name)

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
    df = (pd.DataFrame(RESULTS).T[["CV_RMSE", "RMSE", "MAE", "MAPE", "SI", "R2", "fit_time_s"]]
            .astype(float).sort_values("RMSE"))
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
#                       SVR, MLP, CNN-LSTM, Sequential Linear (Keras).
# =============================================================================


# =============================================================================
# =============================================================================
#  PART 3 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
# =============================================================================

# =============================================================================
#  SECTION 10 — EXPLORATORY DATA ANALYSIS
#  (summary statistics, histograms, scatter plots, box plots)
# =============================================================================
#  Pure exploratory/descriptive analysis of the dataset itself — nothing here
#  depends on any model being tuned or fitted, so this only needs Part 1
#  (prerequisites/setup) to have already run: X_tr/X_te/y_tr/y_te and the
#  original x_train/x_test/y_train/y_test. It deliberately sits here, right
#  after Part 1 and before any modeling (Parts 4-7), because that is the
#  natural place for exploratory analysis — look at the data before building
#  anything on top of it.
#
#  `df` combines train+test back into one table purely for description (it
#  mirrors how `cs` was used in the reference code — the whole dataset before
#  any split) and plays no role in fitting or tuning, all of which already
#  happened in Sections 1-9.
#
#  Note: the reference code's column names (age, w/b, BFA, FA, CBBF, BCA, CA,
#  SP, CS) describe a brick-aggregate concrete mix design, not the standard
#  UCI Cement/Slag/FlyAsh/Water/Superplasticizer/CoarseAgg/FineAgg/Age feature
#  set assumed in a few comments earlier in this file (Sections 6 and 9). The
#  functions below read column names straight out of `df`, so they work with
#  whatever your real columns are regardless — but you may want to update
#  those earlier comments if your dataset is actually the brick-aggregate
#  variant.
# =============================================================================
import seaborn as sns

target_name = getattr(y_train, "name", None) or "Compressive_Strength"

if hasattr(x_train, "columns"):
    X_full = pd.concat([x_train, x_test], axis=0, ignore_index=True)
else:
    fallback_names = [f"Feature_{i}" for i in range(N_FEATURES)]
    X_full = pd.DataFrame(np.vstack([X_tr, X_te]), columns=fallback_names)

df = X_full.copy()
df[target_name] = np.concatenate([y_tr, y_te])


# ── SUMMARY STATISTICS TABLE ──────────────────────────────────────────────────
def summary_stats(a):
    """
    count / mean / std / min / Q1 / median / Q3 / max (from describe()), plus
    mode and skewness per column.

    Fixes a labeling bug in the reference version: it named the 25th and 50th
    percentile columns 'Q1' and 'median', then reused 'Q2' for the 75th
    percentile. Q2 conventionally IS the median (50th percentile) — the 75th
    percentile is Q3, not a second Q2 — so relabeling 'median' as 'Q2' would
    just restate the same quantity under two names while leaving the 75th
    percentile column mislabeled. Using 'Q1' / 'median' / 'Q3' names each
    quantile exactly once.
    """
    desc_stats = a.describe().T
    modes = a.mode().iloc[0]
    skewness = a.skew()
    stats = pd.concat([desc_stats, modes, skewness], axis=1)
    stats.columns = ['count', 'mean', 'std', 'min', 'Q1', 'median', 'Q3', 'max',
                     'mode', 'skewness']
    return stats


print("\n" + "=" * 78)
print("SUMMARY STATISTICS")
print("=" * 78)
summary_df = summary_stats(df)
print(summary_df.round(4).to_string())
register_table("EDA summary stats", summary_df, index=True)
register_table("Split seed scan", seed_scan.head(50))


EDA_EDGE_COLOR = "#1D3557"   # one consistent dark navy for titles/edges/text across all EDA plots


def _eda_grid(n, ncols=2, row_height=4.3):
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, row_height * nrows))
    axs = np.array(axs).reshape(nrows, ncols)
    return fig, axs, nrows, ncols


def _hide_unused_axes(fig, axs, n, nrows, ncols):
    for j in range(n, nrows * ncols):
        fig.delaxes(axs[j // ncols, j % ncols])


# ── HISTOGRAMS (with KDE overlay) ────────────────────────────────────────────
def _draw_histogram(ax, df, column, color):
    """One column's distribution. Shared by the grid and the per-feature figure
    so the two can never drift apart."""
    sns.kdeplot(df[column], color=color, linewidth=2, ax=ax)
    ax.hist(df[column], color=color, edgecolor=EDA_EDGE_COLOR, linewidth=0.8,
            density=True, alpha=0.45)
    ax.set_title(f"Distribution of {column}", fontsize=13, fontweight='bold',
                 color=EDA_EDGE_COLOR)
    ax.set_xlabel(column, fontsize=11, color=EDA_EDGE_COLOR)
    ax.set_ylabel('Density', fontsize=11, color=EDA_EDGE_COLOR)
    ax.tick_params(labelsize=9, colors=EDA_EDGE_COLOR)
    ax.grid(True, linestyle='--', color='#E0E0E0', alpha=0.7)


def plot_histograms(df, ncols=2):
    columns = list(df.columns)
    n = len(columns)
    fig, axs, nrows, ncols = _eda_grid(n, ncols)
    palette = sns.color_palette("colorblind", n_colors=n)

    for i, column in enumerate(columns):
        _draw_histogram(axs[i // ncols, i % ncols], df, column, palette[i])

    _hide_unused_axes(fig, axs, n, nrows, ncols)
    plt.tight_layout()
    save_fig(plt.gcf(), "eda_histograms", subdir="eda")
    plt.show()
    plt.close(fig)

    if PER_ITEM_FIGURES:
        for i, column in enumerate(columns):
            f1, ax1 = plt.subplots(figsize=(6.4, 4.4))
            _draw_histogram(ax1, df, column, palette[i])
            f1.tight_layout()
            save_fig(f1, f"histogram_{column}", subdir="eda/per_feature", close=True)


# ── SCATTER PLOTS (each feature vs. the target) ──────────────────────────────
def _draw_scatter(ax, df, column, target, color):
    """One feature against the target, with its fitted line and Pearson r."""
    ax.scatter(df[column], df[target], color=color, alpha=0.5,
               edgecolors=EDA_EDGE_COLOR, linewidths=0.5, s=25)
    r = df[column].corr(df[target])
    coeffs = np.polyfit(df[column], df[target], 1)
    x_line = np.linspace(df[column].min(), df[column].max(), 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), color=EDA_EDGE_COLOR,
            linewidth=1.5, linestyle='--')
    ax.set_title(f"{column} vs {target}  (r = {r:.2f})", fontsize=13,
                 fontweight='bold', color=EDA_EDGE_COLOR)
    ax.set_xlabel(column, fontsize=11, color=EDA_EDGE_COLOR)
    ax.set_ylabel(target, fontsize=11, color=EDA_EDGE_COLOR)
    ax.tick_params(labelsize=9, colors=EDA_EDGE_COLOR)
    ax.grid(True, linestyle='--', color='#E0E0E0', alpha=0.7)
    return r


def plot_scatterplots(df, target, ncols=2):
    columns = [c for c in df.columns if c != target]
    n = len(columns)
    fig, axs, nrows, ncols = _eda_grid(n, ncols)
    palette = sns.color_palette("colorblind", n_colors=n)

    rs = {}
    for i, column in enumerate(columns):
        rs[column] = _draw_scatter(axs[i // ncols, i % ncols], df, column, target, palette[i])

    _hide_unused_axes(fig, axs, n, nrows, ncols)
    plt.tight_layout()
    save_fig(plt.gcf(), "eda_scatterplots", subdir="eda")
    plt.show()
    plt.close(fig)

    if PER_ITEM_FIGURES:
        for i, column in enumerate(columns):
            f1, ax1 = plt.subplots(figsize=(6.4, 4.6))
            _draw_scatter(ax1, df, column, target, palette[i])
            f1.tight_layout()
            save_fig(f1, f"scatter_{column}_vs_{target}", subdir="eda/per_feature", close=True)

    register_table("Feature-target correlation",
                   pd.DataFrame({"feature": list(rs), "pearson_r": list(rs.values())}))


# ── BOX PLOTS (outlier / spread check per column) ────────────────────────────
def _draw_boxplot(ax, df, column, color):
    """One column's spread, titled with its outlier count."""
    sns.boxplot(y=df[column], ax=ax, color=color,
                boxprops=dict(edgecolor=EDA_EDGE_COLOR),
                medianprops=dict(color=EDA_EDGE_COLOR, linewidth=2),
                whiskerprops=dict(color=EDA_EDGE_COLOR),
                capprops=dict(color=EDA_EDGE_COLOR),
                flierprops=dict(markerfacecolor=color, markeredgecolor=EDA_EDGE_COLOR,
                                markersize=5, alpha=0.6))
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_outliers = int(((df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)).sum())
    ax.set_title(f"{column}  ({n_outliers} outliers)", fontsize=13,
                 fontweight='bold', color=EDA_EDGE_COLOR)
    ax.set_ylabel(column, fontsize=11, color=EDA_EDGE_COLOR)
    ax.set_xlabel('')
    ax.tick_params(labelsize=9, colors=EDA_EDGE_COLOR)
    ax.grid(True, axis='y', linestyle='--', color='#E0E0E0', alpha=0.7)
    return n_outliers


def plot_boxplots(df, ncols=2):
    columns = list(df.columns)
    n = len(columns)
    fig, axs, nrows, ncols = _eda_grid(n, ncols, row_height=3.8)
    palette = sns.color_palette("colorblind", n_colors=n)

    outliers = {}
    for i, column in enumerate(columns):
        outliers[column] = _draw_boxplot(axs[i // ncols, i % ncols], df, column, palette[i])

    _hide_unused_axes(fig, axs, n, nrows, ncols)
    plt.tight_layout()
    save_fig(plt.gcf(), "eda_boxplots", subdir="eda")
    plt.show()
    plt.close(fig)

    if PER_ITEM_FIGURES:
        for i, column in enumerate(columns):
            f1, ax1 = plt.subplots(figsize=(5.0, 4.4))
            _draw_boxplot(ax1, df, column, palette[i])
            f1.tight_layout()
            save_fig(f1, f"boxplot_{column}", subdir="eda/per_feature", close=True)

    register_table("EDA outlier counts",
                   pd.DataFrame({"column": list(outliers), "outliers": list(outliers.values())}))


# ── CORRELATION HEATMAPS ─────────────────────────────────────────────────────
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


#  Three methods because they answer different questions: Pearson measures
#  LINEAR association, Spearman and Kendall measure MONOTONIC association and
#  are robust to outliers and non-linear-but-ordered relationships. A feature
#  that looks uncorrelated under Pearson but strong under Spearman is
#  non-linearly related to the target — which is exactly the kind of structure
#  the tree/NN models in this pipeline can exploit and the linear baseline
#  cannot.
#
#  THE DIAGONAL IS REMOVED. Every variable correlates perfectly with itself, so
#  the diagonal is a row of 1.00 that carries no information, anchors the colour
#  scale at its extreme, and draws the eye away from the off-diagonal cells that
#  are the point of the plot.
def plot_correlation_heatmaps(df, methods=('pearson', 'spearman', 'kendall')):
    cols = list(df.columns)
    for method in methods:
        corr, pvals = correlation_with_pvalues(df, method)

        labels = corr.copy().astype(object)
        for a in cols:
            for b in cols:
                labels.loc[a, b] = ("" if not np.isfinite(corr.loc[a, b])
                                    else f"{corr.loc[a, b]:.2f}{significance_stars(pvals.loc[a, b])}")

        #  k=0 masks the diagonal as well as the upper triangle; k=1 would keep
        #  the diagonal, which is the default and is what we do not want here.
        mask = np.triu(np.ones_like(corr, dtype=bool), k=0)

        fig, ax = plt.subplots(figsize=(9.5, 7.5))
        sns.heatmap(corr, mask=mask, annot=labels if len(cols) <= 15 else False,
                    cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True,
                    linewidths=1, linecolor='white', cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 8.5}, fmt='', cbar=True, ax=ax)
        ax.set_title(f"{method.capitalize()} Correlation", fontsize=14,
                     fontweight='bold', color=EDA_EDGE_COLOR)
        ax.tick_params(labelsize=9, colors=EDA_EDGE_COLOR)
        fig.text(0.5, 0.005,
                 f"*  p < {SIG_ONE_STAR}      **  p < {SIG_TWO_STAR}      "
                 f"unmarked: not significant at {SIG_TWO_STAR}      diagonal omitted",
                 ha='center', fontsize=9, color=EDA_EDGE_COLOR)
        plt.tight_layout(rect=(0, 0.03, 1, 1))
        save_fig(plt.gcf(), f"eda_{method}_correlation_heatmap", subdir="eda")
        plt.show()
        plt.close(fig)

        #  Exported long-form: the heatmap shows the stars, the table carries
        #  the exact p-values a reviewer will ask for.
        rows = []
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                if j < i:
                    rows.append({"variable_a": a, "variable_b": b,
                                 "coefficient": corr.loc[a, b], "p_value": pvals.loc[a, b],
                                 "stars": significance_stars(pvals.loc[a, b])})
        register_table(f"Corr {method}", pd.DataFrame(rows))


plot_histograms(df)
plot_scatterplots(df, target_name)
plot_boxplots(df)
plot_correlation_heatmaps(df)


# =============================================================================
# =============================================================================
#  PART 4 — MODEL SPACE
# =============================================================================
# =============================================================================

# =============================================================================
#  SECTION 3 — KERAS HELPERS  (MLP, CNN-LSTM, Sequential Linear)
# =============================================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.get_logger().setLevel("ERROR")


def keras_cv_rmse(build_fn, X, y, batch_size=32, epochs=300,
                  cv=None, to3d=False, trial=None, patience=25):
    """
    Manual K-fold for Keras: refits the X/y scalers inside every fold (no
    leakage), early-stops each fold on its own validation split, and reports
    intermediate values to Optuna for pruning. Returns mean CV RMSE in the
    original MPa units.
    """
    cv = cv or inner_cv
    fold_rmse = []
    for k, (tr, va) in enumerate(cv.split(X)):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        sx = SCALER_CLS().fit(Xtr)
        Xtr_s, Xva_s = sx.transform(Xtr), sx.transform(Xva)
        sy = StandardScaler().fit(ytr.reshape(-1, 1))
        ytr_s = sy.transform(ytr.reshape(-1, 1)).ravel()
        yva_s = sy.transform(yva.reshape(-1, 1)).ravel()

        if to3d:   # (n, f) -> (n, f, 1): the 8 mix-design features act as the "sequence" axis
            Xtr_s = Xtr_s.reshape(Xtr_s.shape[0], Xtr_s.shape[1], 1)
            Xva_s = Xva_s.reshape(Xva_s.shape[0], Xva_s.shape[1], 1)

        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED)
        model = build_fn(Xtr_s.shape[1:])
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                           restore_best_weights=True)
        model.fit(Xtr_s, ytr_s, validation_data=(Xva_s, yva_s),
                  epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

        pred = sy.inverse_transform(model.predict(Xva_s, verbose=0).reshape(-1, 1)).ravel()
        fold_rmse.append(rmse(yva, pred))

        if trial is not None:                      # Optuna pruning hook
            trial.report(float(np.mean(fold_rmse)), step=k)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
    return float(np.mean(fold_rmse))


def keras_fit(build_fn, Xtr, ytr, batch_size, epochs=400, to3d=False,
             val_frac=0.15, patience=30):
    """
    Fit the tuned architecture on (Xtr, ytr). Returns (model, predict_fn) where
    predict_fn carries the fitted X/y scalers so it can be called on ANY future
    array of raw features, exactly like a sklearn estimator's .predict.
    """
    n_val = max(1, int(len(Xtr) * val_frac))
    idx = np.random.RandomState(SEED).permutation(len(Xtr))
    va_i, tr_i = idx[:n_val], idx[n_val:]

    sx = SCALER_CLS().fit(Xtr[tr_i])
    sy = StandardScaler().fit(ytr[tr_i].reshape(-1, 1))
    Xt, Xv = sx.transform(Xtr[tr_i]), sx.transform(Xtr[va_i])
    yt = sy.transform(ytr[tr_i].reshape(-1, 1)).ravel()
    yv = sy.transform(ytr[va_i].reshape(-1, 1)).ravel()

    if to3d:
        Xt = Xt.reshape(*Xt.shape, 1)
        Xv = Xv.reshape(*Xv.shape, 1)

    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    model = build_fn(Xt.shape[1:])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                       restore_best_weights=True)
    model.fit(Xt, yt, validation_data=(Xv, yv), epochs=epochs,
              batch_size=batch_size, verbose=0, callbacks=[es])

    def predict_fn(Xnew):
        Xs = sx.transform(np.asarray(Xnew, dtype=np.float32))
        if to3d:
            Xs = Xs.reshape(*Xs.shape, 1)
        pred_s = model.predict(Xs, verbose=0).reshape(-1, 1)
        return sy.inverse_transform(pred_s).ravel()

    return model, predict_fn


import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_study():
    return optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )


# =============================================================================
#  1. SHALLOW MULTILAYER PERCEPTRON   —  Optuna (TPE)
# =============================================================================
#  Shallow = exactly ONE hidden layer. Essential knobs: hidden width, activation,
#  L2, dropout, learning rate, batch size — continuous/log-scaled, so TPE beats
#  a grid here. Epochs are NOT tuned: EarlyStopping picks them per fold.
# =============================================================================
def build_mlp(units, activation, l2, dropout, lr):
    def _b(input_shape):
        m = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(units, activation=activation,
                         kernel_regularizer=keras.regularizers.l2(l2)),
            layers.Dropout(dropout),
            layers.Dense(1, activation="linear"),
        ])
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
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
        return keras_cv_rmse(build_mlp(units, activation, l2, dropout, lr),
                             X, y, batch_size=batch_size, trial=trial)
    study = make_study()
    study.optimize(objective, n_trials=N_TRIALS["mlp"])
    return study.best_params, study.best_value


def mlp_refit(X, y, params):
    model, predict_fn = keras_fit(
        build_mlp(params["units"], params["activation"], params["l2"],
                 params["dropout"], params["lr"]),
        X, y, batch_size=params["batch_size"])
    return model, predict_fn


mlp = ModelSpec("Shallow MLP", mlp_search, mlp_refit)


# =============================================================================
#  2. RANDOM FOREST   —  RandomizedSearchCV  (+ nested CV)
# =============================================================================
#  ~7 knobs but performance is driven mostly by max_features and the leaf-size
#  constraints. Random search dominates grid search at equal budget once >2-3
#  dims matter (Bergstra & Bengio, 2012), and RF is cheap enough for nested CV.
#  n_estimators is set high, not really "tuned": more trees only reduces
#  variance, it doesn't overfit RF.
# =============================================================================
from sklearn.ensemble import RandomForestRegressor

rf_space = {
    "n_estimators":      [300, 500, 800, 1200],
    "max_depth":         [None, 5, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
    "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
    "bootstrap":         [True, False],
    "max_samples":       [None, 0.6, 0.8],   # only used when bootstrap=True
}


def make_rf_search():
    return RandomizedSearchCV(
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        rf_space, n_iter=N_ITER_RANDOM, scoring=SCORING,
        cv=inner_cv, random_state=SEED, n_jobs=-1, refit=True)


def rf_search(X, y):
    search = make_rf_search()
    search.fit(X, y)
    return search.best_params_, -search.best_score_


def rf_refit(X, y, params):
    m = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
    m.fit(X, y)
    return m, m.predict


rf = ModelSpec("Random Forest", rf_search, rf_refit, search_factory=make_rf_search)


# =============================================================================
#  3. XGBOOST   —  Optuna (TPE) + per-fold early stopping
# =============================================================================
#  8 interacting hyperparameters, several log-scaled -> Bayesian search
#  territory. n_estimators is NOT sampled: each fold early-stops on its own
#  validation split; the median of the folds' best_iteration is reused for the
#  final refit (fixed, no early stopping needed on the full-data refit).
# =============================================================================
import xgboost as xgb


def xgb_search(X, y):
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
        fold_rmse, best_iters = [], []
        for k, (tr, va) in enumerate(inner_cv.split(X)):
            m = xgb.XGBRegressor(
                n_estimators=3000, early_stopping_rounds=50, eval_metric="rmse",
                tree_method="hist", random_state=SEED, n_jobs=-1, **params)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            fold_rmse.append(rmse(y[va], m.predict(X[va])))
            best_iters.append(m.best_iteration)
            trial.report(float(np.mean(fold_rmse)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("n_estimators", int(np.median(best_iters)) + 1)
        return float(np.mean(fold_rmse))

    study = make_study()
    study.optimize(objective, n_trials=N_TRIALS["xgb"])
    best = dict(study.best_params)
    best["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
    return best, study.best_value


def xgb_refit(X, y, params):
    m = xgb.XGBRegressor(tree_method="hist", random_state=SEED, n_jobs=-1, **params)
    m.fit(X, y, verbose=False)
    return m, m.predict


xgboost_model = ModelSpec("XGBoost", xgb_search, xgb_refit)


# =============================================================================
#  4. ADABOOST   —  GridSearchCV  (+ nested CV)
# =============================================================================
#  Very few real knobs; the one people forget is the *weak learner's* depth,
#  which is tuned here jointly with learning_rate/n_estimators since those two
#  trade off directly. Small discrete space -> exhaustive grid is affordable.
# =============================================================================
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

ada_grid = {
    "estimator__max_depth":        [1, 2, 3, 4, 6, 8],
    "estimator__min_samples_leaf": [1, 5],
    "n_estimators":                [50, 100, 300, 600],
    "learning_rate":               [0.01, 0.05, 0.1, 0.5, 1.0],
    "loss":                        ["linear", "square", "exponential"],
}


def make_ada_search():
    return GridSearchCV(
        AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=SEED),
                          random_state=SEED),
        ada_grid, scoring=SCORING, cv=inner_cv, n_jobs=-1, refit=True)


def ada_search(X, y):
    search = make_ada_search()
    search.fit(X, y)
    return search.best_params_, -search.best_score_


def ada_refit(X, y, params):
    m = AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=SEED), random_state=SEED)
    m.set_params(**params)     # routes estimator__* to the nested DecisionTreeRegressor
    m.fit(X, y)
    return m, m.predict


ada = ModelSpec("AdaBoost", ada_search, ada_refit, search_factory=make_ada_search)


# =============================================================================
#  5. LIGHT GRADIENT BOOSTING   —  Optuna (TPE) + per-fold early stopping
# =============================================================================
#  LightGBM grows leaf-wise, so num_leaves (not max_depth) is the primary
#  capacity knob and min_child_samples is the primary overfitting brake — on a
#  dataset this size they must be tuned jointly, which is exactly where a
#  multivariate Bayesian sampler beats independent grid axes.
# =============================================================================
import lightgbm as lgb


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
        )
        fold_rmse, best_iters = [], []
        for k, (tr, va) in enumerate(inner_cv.split(X)):
            m = lgb.LGBMRegressor(n_estimators=3000, random_state=SEED,
                                  n_jobs=-1, verbose=-1, **params)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            fold_rmse.append(rmse(y[va], m.predict(X[va])))
            best_iters.append(m.best_iteration_)
            trial.report(float(np.mean(fold_rmse)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("n_estimators", int(np.median(best_iters)) + 1)
        return float(np.mean(fold_rmse))

    study = make_study()
    study.optimize(objective, n_trials=N_TRIALS["lgbm"])
    best = dict(study.best_params)
    best["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
    return best, study.best_value


def lgbm_refit(X, y, params):
    m = lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1, **params)
    m.fit(X, y)
    return m, m.predict


lgbm = ModelSpec("LightGBM", lgbm_search, lgbm_refit)


# =============================================================================
#  6. CNN-LSTM   —  Optuna (TPE)
# =============================================================================
#  Concrete mix design has no time axis, so this is the standard tabular
#  workaround: reshape (n, 8) -> (n, 8, 1) and let Conv1D + LSTM scan across
#  the fixed feature ordering (Cement, Slag, Fly Ash, Water, Superplasticizer,
#  Coarse Agg, Fine Agg, Age) as if it were a short sequence. It's a legitimate
#  way to let the model learn local interactions between adjacent mix
#  components, but it is not modelling real temporal dependence — treat its
#  result as one more nonlinear learner to compare, not as inherently superior
#  to the MLP for this problem.
# =============================================================================
def build_cnn_lstm(filters, kernel_size, pool, lstm_units, dropout, rec_dropout,
                   dense_units, lr):
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
        m.add(layers.Dense(1, activation="linear"))
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
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
        return keras_cv_rmse(
            build_cnn_lstm(filters, kernel_size, pool, lstm_units, dropout,
                           rec_dropout, dense_units, lr),
            X, y, batch_size=batch_size, to3d=True, trial=trial)
    study = make_study()
    study.optimize(objective, n_trials=N_TRIALS["cnn_lstm"])
    return study.best_params, study.best_value


def cnn_lstm_refit(X, y, params):
    model, predict_fn = keras_fit(
        build_cnn_lstm(params["filters"], params["kernel_size"], params["pool"],
                       params["lstm_units"], params["dropout"], params["rec_dropout"],
                       params["dense_units"], params["lr"]),
        X, y, batch_size=params["batch_size"], to3d=True)
    return model, predict_fn


cnn_lstm = ModelSpec("CNN-LSTM", cnn_lstm_search, cnn_lstm_refit)


# =============================================================================
#  7. K-NEAREST NEIGHBORS   —  GridSearchCV  (+ nested CV)
# =============================================================================
#  Tiny discrete space (k x weighting x distance metric) -> exhaustive search
#  is cheap and there's no reason to approximate it. Scaling is inside the
#  Pipeline because KNN is a pure distance method: cement (kg/m^3, hundreds)
#  and age (days, single/double digits) are on wildly different scales, and
#  unscaled Euclidean distance would let cement dominate.
# =============================================================================
from sklearn.neighbors import KNeighborsRegressor

knn_pipe = Pipeline([("scaler", SCALER_CLS()), ("model", KNeighborsRegressor())])

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
    return search.best_params_, -search.best_score_


def knn_refit(X, y, params):
    m = Pipeline([("scaler", SCALER_CLS()), ("model", KNeighborsRegressor())])
    m.set_params(**params)
    m.fit(X, y)
    return m, m.predict


knn = ModelSpec("KNN", knn_search, knn_refit, search_factory=make_knn_search)


# =============================================================================
#  8. SVM (SVR)   —  Optuna (TPE)
# =============================================================================
#  C, gamma and epsilon each span 5-6 orders of magnitude and interact strongly
#  (large C + large gamma = memorisation). A log-uniform Bayesian search finds
#  the ridge in that space with far fewer evaluations than a brute grid, and
#  SVR training is O(n^2)-O(n^3) so evaluations aren't free.
# =============================================================================
from sklearn.svm import SVR


def svr_search(X, y):
    def objective(trial):
        kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid", "linear"])
        params = dict(
            kernel  = kernel,
            C       = trial.suggest_float("C", 1e-2, 1e4, log=True),
            epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
        )
        if kernel != "linear":
            params["gamma"] = trial.suggest_float("gamma", 1e-5, 1e1, log=True)
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 4)
        if kernel in ("poly", "sigmoid"):
            params["coef0"] = trial.suggest_float("coef0", -1.0, 1.0)

        pipe = Pipeline([("scaler", SCALER_CLS()),
                         ("model", SVR(cache_size=1000, **params))])
        s = cross_val_score(pipe, X, y, cv=inner_cv, scoring=SCORING, n_jobs=-1)
        return float(-s.mean())

    study = make_study()
    study.optimize(objective, n_trials=N_TRIALS["svr"])
    return study.best_params, study.best_value


def svr_refit(X, y, params):
    m = Pipeline([("scaler", SCALER_CLS()), ("model", SVR(cache_size=1000, **params))])
    m.fit(X, y)
    return m, m.predict


svr = ModelSpec("SVM (SVR)", svr_search, svr_refit)


# =============================================================================
#  9. SEQUENTIAL LINEAR REGRESSION   —  Optuna (TPE)
# =============================================================================
#  A single linear unit built with keras.Sequential and fit by gradient
#  descent — i.e. linear regression as the "zero hidden layer" baseline next
#  to the MLP/CNN-LSTM. Tuned knobs: learning rate, L1/L2 penalty, batch size.
# =============================================================================
def build_linear(lr, l1, l2):
    def _b(input_shape):
        m = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(1, activation="linear",
                         kernel_regularizer=keras.regularizers.l1_l2(l1=l1, l2=l2)),
        ])
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
        return m
    return _b


def seq_linear_search(X, y):
    def objective(trial):
        lr         = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        l1         = trial.suggest_float("l1", 1e-8, 1e-2, log=True)
        l2         = trial.suggest_float("l2", 1e-8, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        return keras_cv_rmse(build_linear(lr, l1, l2), X, y,
                             batch_size=batch_size, trial=trial)
    study = make_study()
    study.optimize(objective, n_trials=N_TRIALS["seq_linear"])
    return study.best_params, study.best_value


def seq_linear_refit(X, y, params):
    model, predict_fn = keras_fit(build_linear(params["lr"], params["l1"], params["l2"]),
                                  X, y, batch_size=params["batch_size"])
    return model, predict_fn


seq_linear = ModelSpec("Sequential Linear Regression", seq_linear_search, seq_linear_refit)



# =============================================================================
#  SECTION 4 — RUN EVERYTHING
# =============================================================================
#  A plain list of models — fit and evaluate any subset the same way. Comment
#  lines out to skip a model, or reorder freely — nothing below this line
#  depends on definition order.
# =============================================================================
models = [rf, mlp, svr, xgboost_model, lgbm, ada, knn, cnn_lstm, seq_linear]


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
#  consistency check in build_hyperparameter_table() catches the common half of
#  that (a tuned parameter missing from the registry) and warns; it cannot
#  detect a bound edited in one place only, so change them together.
SEARCH_SPACES = {
    "Random Forest": rf_space,      # dict, as searched
    "AdaBoost":      ada_grid,      # dict, as searched
    "KNN":           knn_grid,      # dict, as searched
    "XGBoost": {
        "learning_rate":    "log-uniform [0.01, 0.3]",
        "max_depth":        "int [2, 12]",
        "min_child_weight": "log-uniform [0.01, 20.0]",
        "subsample":        "uniform [0.5, 1.0]",
        "colsample_bytree": "uniform [0.4, 1.0]",
        "gamma":            "log-uniform [1e-8, 5.0]",
        "reg_alpha":        "log-uniform [1e-8, 10.0]",
        "reg_lambda":       "log-uniform [1e-8, 20.0]",
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
        "n_estimators":      "not searched - median best_iteration from per-fold early stopping",
    },
    "SVM (SVR)": {
        "kernel":  "{rbf, poly, sigmoid, linear}",
        "C":       "log-uniform [1e-2, 1e4]",
        "epsilon": "log-uniform [1e-3, 1.0]",
        "gamma":   "log-uniform [1e-5, 1e1]  (non-linear kernels only)",
        "degree":  "int [2, 4]  (poly only)",
        "coef0":   "uniform [-1.0, 1.0]  (poly and sigmoid only)",
    },
    "Shallow MLP": {
        "units":      "int, log [8, 256]",
        "activation": "{relu, tanh, selu}",
        "l2":         "log-uniform [1e-6, 1e-2]",
        "dropout":    "uniform [0.0, 0.5]",
        "lr":         "log-uniform [1e-4, 1e-2]",
        "batch_size": "{16, 32, 64, 128}",
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
    },
    "Sequential Linear Regression": {
        "lr":         "log-uniform [1e-4, 1e-1]",
        "l1":         "log-uniform [1e-8, 1e-2]",
        "l2":         "log-uniform [1e-8, 1e-2]",
        "batch_size": "{16, 32, 64, 128}",
    },
}

SEARCH_STRATEGY = {
    "Random Forest": "RandomizedSearchCV", "AdaBoost": "GridSearchCV",
    "KNN": "GridSearchCV", "XGBoost": "Optuna TPE", "LightGBM": "Optuna TPE",
    "SVM (SVR)": "Optuna TPE", "Shallow MLP": "Optuna TPE",
    "CNN-LSTM": "Optuna TPE", "Sequential Linear Regression": "Optuna TPE",
}


# =============================================================================
#  SECTION 4B — CUSTOM ENSEMBLE LEARNING ALGORITHMS  (stacking & voting)
# =============================================================================
#  Meta-models built ON TOP of the 9 tuned base models above. Each one is
#  itself a ModelSpec, so every downstream Part treats it exactly like a base
#  model — .fit(), .predict(), .evaluate(), .refit_on() all behave the same.
#
#  ---------------------------------------------------------------------------
#  WHY THESE ARE HAND-ROLLED INSTEAD OF sklearn's Stacking/VotingRegressor
#  ---------------------------------------------------------------------------
#  sklearn's StackingRegressor and VotingRegressor require every member to be
#  a clone-able sklearn estimator. Three of the nine base models (mlp,
#  cnn_lstm, seq_linear) are Keras networks that sklearn cannot clone, and
#  their fitted feature/target scalers live inside spec.predict_fn rather than
#  on the model object. Passing them to sklearn would either raise or, worse,
#  silently feed the networks unscaled inputs. Building the ensembles against
#  the ModelSpec interface instead means all nine model types combine
#  uniformly, and every member is called through spec.predict — the same
#  scaling-aware path used everywhere else in this file.
#
#  ---------------------------------------------------------------------------
#  STACKING: OUT-OF-FOLD META-FEATURES ARE NOT OPTIONAL
#  ---------------------------------------------------------------------------
#  The meta-learner must be trained on OUT-OF-FOLD base predictions. If you
#  instead fit the base models on all of X and feed the meta-learner their
#  in-sample predictions, those predictions are fit to noise the base models
#  have already memorised. The meta-learner then learns to trust whichever
#  base model OVERFITS HARDEST — the one whose in-sample predictions look
#  near-perfect — and the stack generalises worse than its own members. That
#  is the classic stacking failure, and it produces a beautiful training score
#  on the way down. _oof_matrix() below does the K-fold version properly.
#
#  Meta-learner is RidgeCV. Base-model predictions are extremely collinear
#  (every member is estimating the same target), so unregularised OLS on them
#  produces huge cancelling coefficients that swing wildly with tiny data
#  changes. Ridge shrinks them into something stable. Ridge coefficients may
#  come out NEGATIVE — that is not necessarily a bug: a member can be useful
#  as a correction term to the others. If you want blend weights that read as
#  interpretable "contributions" for a paper, swap RidgeCV for
#  sklearn.linear_model.LinearRegression(positive=True) instead.
#
#  ---------------------------------------------------------------------------
#  COST WARNING — READ BEFORE RUNNING
#  ---------------------------------------------------------------------------
#  One stacking fit = (ENSEMBLE_CV folds x n_members) base refits for the OOF
#  pass, PLUS n_members full refits. For S1 (all 9 members, 5 folds) that is
#  54 base fits, three of which are Keras networks. Every downstream call to
#  .refit_on() on an ensemble re-triggers that whole cascade, so the ensembles
#  are deliberately kept OUT of the expensive refit loops (the K-fold sweep in
#  Part 5, the learning curves and bias-variance bootstrap in Part 6, and SHAP
#  in Part 7) and included only in the cheap predict-only tables and plots.
#  Each of those sections says how to opt in if you want to pay for it.
# =============================================================================
from sklearn.linear_model import RidgeCV

ENSEMBLE_CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

#  The OOF pass used purely to REPORT an honest cv_rmse for each ensemble is a
#  second, separate pass over the members. Set this False to skip it (the
#  ensembles still fit and predict identically; their CV_RMSE column just shows
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


def _oof_matrix(members, X, y, cv=None):
    """
    Out-of-fold prediction matrix, shape (n_samples, n_members). Column j holds
    member j's predictions for each row, made by a copy of that member that
    never saw the row during training. These are the meta-features.
    """
    cv = cv or ENSEMBLE_CV
    oof = np.zeros((len(X), len(members)), dtype=np.float64)
    for j, member in enumerate(members):
        for tr, va in cv.split(X):
            predict_fn = member.refit_on(X[tr], y[tr])
            oof[va, j] = predict_fn(X[va])
    return oof


# -- STACKING FACTORY ---------------------------------------------------------
def make_stacking_spec(name, members, meta_alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Level-1 RidgeCV meta-learner trained on K-fold OOF predictions of `members`."""
    member_names = [m.name for m in members]

    def _fit_meta(X, y):
        oof = _oof_matrix(members, X, y)
        meta = RidgeCV(alphas=meta_alphas).fit(oof, y)
        return oof, meta

    def search_fn(X, y):
        _require_tuned(name, members)
        if not ENSEMBLE_REPORT_CV:
            return {"members": member_names, "meta": "RidgeCV"}, None
        oof, meta = _fit_meta(X, y)
        params = {
            "members": member_names,
            "meta": "RidgeCV",
            "meta_alpha": float(meta.alpha_),
            "blend_weights": {n: round(float(c), 4) for n, c in zip(member_names, meta.coef_)},
            "intercept": round(float(meta.intercept_), 4),
        }
        return params, rmse(y, meta.predict(oof))   # honest OOF blend RMSE

    def refit_fn(X, y, params):
        _require_tuned(name, members)
        # The meta-learner is refit here rather than reused, because it is part
        # of the model: on a bootstrap resample (Part 6 bias-variance) the
        # correct blend weights are the ones learned from THAT resample.
        _, meta = _fit_meta(X, y)
        base_predict_fns = [m.refit_on(X, y) for m in members]

        def predict_fn(Xnew):
            meta_features = np.column_stack([pf(Xnew) for pf in base_predict_fns])
            return meta.predict(meta_features)

        return {"meta": meta, "members": member_names}, predict_fn

    return ModelSpec(name, search_fn, refit_fn)


# -- VOTING FACTORY -----------------------------------------------------------
def make_voting_spec(name, members, weighting="uniform"):
    """
    Weighted average of member predictions.
      weighting="uniform"      -> equal weights; no OOF pass needed to fit.
      weighting="inverse_rmse" -> weight_j proportional to 1 / OOF_RMSE_j, so
                                  more accurate members count for more. The
                                  weights come from OUT-OF-FOLD errors, not
                                  training errors — training errors would just
                                  rank the members by how hard they overfit.
    Weights are treated as tuned hyperparameters: computed once during the
    search, then reused on refit, so a bootstrap refit does not pay for another
    OOF pass.
    """
    member_names = [m.name for m in members]

    def search_fn(X, y):
        _require_tuned(name, members)
        need_oof = (weighting == "inverse_rmse") or ENSEMBLE_REPORT_CV
        oof = _oof_matrix(members, X, y) if need_oof else None

        if weighting == "uniform":
            w = np.ones(len(members)) / len(members)
        elif weighting == "inverse_rmse":
            errs = np.array([rmse(y, oof[:, j]) for j in range(len(members))])
            w = (1.0 / errs) / (1.0 / errs).sum()
        else:
            raise ValueError(f"unknown weighting: {weighting!r}")

        params = {
            "members": member_names,
            "weighting": weighting,
            # Full precision, NOT rounded: refit_fn reads these back and does
            # arithmetic with them. Rounding to 4 dp makes uniform weights
            # 0.3333... which sum to 0.9999 rather than 1.0, quietly shrinking
            # every prediction toward zero. Round only for display.
            "weights": {n: float(wi) for n, wi in zip(member_names, w)},
        }
        cv_rmse = rmse(y, oof @ w) if ENSEMBLE_REPORT_CV else None
        return params, cv_rmse

    def refit_fn(X, y, params):
        _require_tuned(name, members)
        w = np.array([params["weights"][n] for n in member_names], dtype=np.float64)
        w = w / w.sum()          # guard against any drift; weights must sum to 1
        base_predict_fns = [m.refit_on(X, y) for m in members]

        def predict_fn(Xnew):
            preds = np.column_stack([pf(Xnew) for pf in base_predict_fns])
            return preds @ w

        return {"weights": dict(zip(member_names, w)), "members": member_names}, predict_fn

    return ModelSpec(name, search_fn, refit_fn)


# -- ENSEMBLE ROSTER ----------------------------------------------------------
#  Groupings are by INDUCTIVE BIAS, not by accuracy. Ensembling pays off when
#  members make DIFFERENT errors — averaging four gradient-boosted tree models
#  that all fail on the same rows buys almost nothing, while combining a tree,
#  a kernel machine and a network can, because their failure modes differ.
#  That is why the family groups (S2-S4) are included alongside the
#  cross-family ones (S1, S5): the comparison shows whether diversity actually
#  bought anything on your data, which is a result worth reporting either way.
#
#  All 9 base models are covered — S1/V1 contain every one of them, and each
#  model also appears in exactly one family group.
TREE_MEMBERS    = [rf, xgboost_model, lgbm, ada]        # bagging + 3 boosting variants
NEURAL_MEMBERS  = [mlp, cnn_lstm, seq_linear]           # the Keras family
KERNEL_MEMBERS  = [svr, knn]                            # kernel + instance-based
DIVERSE_MEMBERS = [rf, xgboost_model, mlp, svr, knn]    # one strong pick per family
ALL_MEMBERS     = [rf, mlp, svr, xgboost_model, lgbm, ada, knn, cnn_lstm, seq_linear]

# Stacking (RidgeCV meta-learner on OOF predictions)
S1 = make_stacking_spec("S1 Stack (All 9)",        ALL_MEMBERS)
S2 = make_stacking_spec("S2 Stack (Trees)",        TREE_MEMBERS)
S3 = make_stacking_spec("S3 Stack (Neural)",       NEURAL_MEMBERS)
S4 = make_stacking_spec("S4 Stack (Kernel+KNN)",   KERNEL_MEMBERS)
S5 = make_stacking_spec("S5 Stack (Cross-family)", DIVERSE_MEMBERS)

# Voting (weighted average) — same groupings, so stacking vs voting is a
# controlled comparison: identical members, different combination rule.
V1 = make_voting_spec("V1 Vote (All 9)",           ALL_MEMBERS)
V2 = make_voting_spec("V2 Vote (Trees)",           TREE_MEMBERS)
V3 = make_voting_spec("V3 Vote (Neural)",          NEURAL_MEMBERS)
V4 = make_voting_spec("V4 Vote (Kernel+KNN)",      KERNEL_MEMBERS)
V5 = make_voting_spec("V5 Vote (Cross-family)",    DIVERSE_MEMBERS)
V6 = make_voting_spec("V6 Vote (All 9, wtd)",      ALL_MEMBERS, weighting="inverse_rmse")

stacking_models = [S1, S2, S3, S4, S5]
voting_models   = [V1, V2, V3, V4, V5, V6]
ensemble_models = stacking_models + voting_models
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
#  base-model cascade for every ensemble. To include ensembles in any of those,
#  swap `models` for `all_models` in that section and expect a large slowdown.
all_models = models + ensemble_models

# =============================================================================
#  SECTION 5 — K-FOLD STABILITY SWEEP  (per already-tuned model)
# =============================================================================
#  Checks how sensitive each model's CV score is to the *number* of folds,
#  reusing the hyperparameters found by fit_all() (Part 5, just above) —
#  re-running the full
#  hyperparameter search at every one of 9 fold counts for all 9 models would
#  mean thousands of extra Optuna trials, so this only refits a fresh model
#  per fold via spec.refit_on() and tracks R2 / RMSE / MAPE / SI.
#  Run fit_all() first (just above) so best_params is populated. Note this
#  still trains
#  9 models x sum(2..10)=54 folds = 486 fresh fits; the three Keras models
#  (mlp, cnn_lstm, seq_linear) will dominate runtime since each fit trains a
#  network with early stopping. Trim fold_range, or restrict to
#  models_sklearn_only = [rf, svr, xgboost_model, lgbm, ada, knn] for a
#  faster pass.
# =============================================================================
import matplotlib.pyplot as plt

fold_range = range(2, 11)     # k = 2 ... 10
SWEEP_METRICS = ("R2", "RMSE", "MAPE", "SI")


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
        kfold = KFold(n_splits=k, shuffle=True, random_state=SEED)
        fold_scores = {metric: [] for metric in SWEEP_METRICS}
        for tr, va in kfold.split(X):
            predict_fn = spec.refit_on(X[tr], y[tr])
            pred = predict_fn(X[va])
            m = compute_metrics(y[va], pred)
            for metric in SWEEP_METRICS:
                fold_scores[metric].append(m[metric])
        line = f"{spec.name} | Folds={k}:"
        for metric in SWEEP_METRICS:
            arr = np.array(fold_scores[metric])
            out[metric]["means"].append(arr.mean())
            out[metric]["stds"].append(arr.std())
            line += f"  {metric}={arr.mean():.4f}+/-{arr.std():.4f}"
        print(line)
    return out


sweep_results = {spec.name: fold_sweep(spec, X_tr, y_tr) for spec in models}

def _draw_sweep(ax, res, metric):
    means, stds = res[metric]["means"], res[metric]["stds"]
    ax.errorbar(list(fold_range), means, yerr=stds, marker='o', capsize=4,
                linestyle='-', color='steelblue', ecolor='gray')
    ax.set_title(metric)
    ax.set_xlabel('Number of Folds (k)')
    ax.set_ylabel(metric + (" (%)" if metric in ("MAPE", "SI") else ""))
    ax.set_xticks(list(fold_range))
    ax.grid(True, alpha=0.3)


# One grid per model to scan, plus one figure per metric to actually use.
_sweep_rows = []
for spec in models:
    res = sweep_results[spec.name]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric in zip(axes.ravel(), SWEEP_METRICS):
        _draw_sweep(ax, res, metric)
    fig.suptitle(f'K-Fold Stability: {spec.name}')
    fig.tight_layout()
    save_fig(fig, f"K-Fold_{spec.name.replace(' ', '_')}", subdir="kfold_stability")
    plt.show()
    plt.close(fig)

    if PER_ITEM_FIGURES:
        for metric in SWEEP_METRICS:
            f1, ax1 = plt.subplots(figsize=(6.4, 4.4))
            _draw_sweep(ax1, res, metric)
            ax1.set_title(f"{spec.name} — {metric}")
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
#  populates RESULTS, which the summary table and several plots below read from.
# =============================================================================
summary = evaluate_all(all_models, X_te, y_te)

print("\n" + "=" * 78)
print(summary.round(4).to_string())
register_table("Test summary", summary, index=True)
print("=" * 78)

for spec in all_models:
    print(f"\n{spec.name}: {spec.best_params}")

# ---- Optional: nested CV for the cheap, Grid/RandomizedSearchCV-tuned models
# for spec in (rf, ada, knn):
#     spec.nested_cv(X_tr, y_tr)

# ---- Using one model on its own, without touching the others:
# svr.fit(X_tr, y_tr)
# svr.evaluate(X_te, y_te)
# new_preds = svr.predict(X_te)



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
#  estimate of generalisation, and it would not reproduce. (It is the same
#  mistake Section A refuses to make when picking the split seed.)
#
#  So the "Testing" column records what was actually APPLIED at test time. The
#  train-vs-test comparison worth having lives in the companion table below,
#  which pairs each model's CV score with its training and test scores.
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
    """Drop Pipeline prefixes so the report reads as the model's own API."""
    return param.split("__")[-1] if "__" in param else param


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
    tr = compute_metrics(y_tr, spec.predict(X_tr))
    te = RESULTS[spec.name]
    _model_rows.append({
        "Model": spec.name,
        "Search strategy": SEARCH_STRATEGY.get(spec.name, "ensemble"),
        "CV RMSE (training folds)": spec.cv_rmse,
        "Train RMSE": tr["RMSE"], "Test RMSE": te["RMSE"],
        "Train R2": tr["R2"], "Test R2": te["R2"],
        "Train MAE": tr["MAE"], "Test MAE": te["MAE"],
        "Overfitting gap (test-train RMSE)": te["RMSE"] - tr["RMSE"],
        "Fit time (s)": spec.fit_time_s,
        "Best parameters": spec.best_params,
    })
model_summary_df = pd.DataFrame(_model_rows).sort_values("Test RMSE")
print("\n" + "=" * 78)
print("MODEL-LEVEL SUMMARY — where training and testing really do differ")
print("=" * 78)
print(model_summary_df.drop(columns=["Best parameters"]).round(4).to_string(index=False))
register_table("Model summary", model_summary_df)


# =============================================================================
#  SECTION 6 — LEARNING CURVES  (per already-tuned model)
# =============================================================================
#  Assumes fit_all() (Part 5) and evaluate_all() (just above, Part 6) already
#  ran, so every spec in
#  `models` has .model / .best_params / .cv_rmse set, and RESULTS[name] holds
#  the held-out test metrics. No hyperparameter search happens here — the old
#  GridSearchCV/RandomizedSearchCV block from the original script is gone
#  because it would just repeat the tuning fit_all() already did, with a
#  different, unrelated
#  set of grids. This section only asks: given the hyperparameters already
#  found, how does train vs. validation score evolve with training-set size?
#
#  sklearn's learning_curve() clones and refits an estimator internally, which
#  works directly for every sklearn-compatible model here (rf, svr,
#  xgboost_model, lgbm, ada, knn) — clone(spec.model) reproduces the tuned
#  hyperparameters with the fitted state stripped. It CANNOT clone a Keras
#  model, so mlp / cnn_lstm / seq_linear fall back to a manual sweep built on
#  spec.refit_on(), which reuses the same already-tuned hyperparameters
#  without repeating the search — just like the Part 5 K-fold stability sweep
#  (Section 5).
#
#  Cost note: the Keras fallback still means up to ~10 sizes x 5 folds = 50
#  fresh fits per Keras model. For a faster pass, restrict to the
#  sklearn-only subset:
#      models_for_lc = [rf, svr, xgboost_model, lgbm, ada, knn]
# =============================================================================
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import learning_curve

# ── COLOR PALETTE (Okabe-Ito, colorblind-friendly) — one color per model,
#    generated from the `models` list instead of a hand-maintained dict ──────
_PALETTE = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
           '#D55E00', '#CC79A7', '#999999', '#000000']
COLORS = {spec.name: _PALETTE[i % len(_PALETTE)] for i, spec in enumerate(all_models)}

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

train_sizes_pct = np.linspace(0.1, 1.0, 10)
lc_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)


def get_learning_curve(spec, X, y, train_sizes=None, cv=None):
    """
    Unified learning-curve computation across all 9 models.
    Returns (abs_train_sizes, train_mean, train_std, val_mean, val_std) — R2.
    """
    train_sizes = train_sizes_pct if train_sizes is None else train_sizes
    cv = lc_cv if cv is None else cv
    if hasattr(spec.model, "get_params"):     # sklearn-compatible estimator
        sizes, tr_scores, va_scores = learning_curve(
            spec.model, X, y, train_sizes=train_sizes, cv=cv, n_jobs=-1, scoring='r2')
        return (sizes, tr_scores.mean(axis=1), tr_scores.std(axis=1),
                va_scores.mean(axis=1), va_scores.std(axis=1))

    # Keras fallback: manual per-size, per-fold refit via spec.refit_on()
    rng = np.random.RandomState(SEED)
    min_train_len = min(len(tr) for tr, _ in cv.split(X))
    sizes = np.unique((np.asarray(train_sizes) * min_train_len).astype(int))
    sizes = sizes[sizes >= 5]

    tr_mean, tr_std, va_mean, va_std = [], [], [], []
    for size in sizes:
        tr_s, va_s = [], []
        for tr_idx, va_idx in cv.split(X):
            sub = rng.choice(tr_idx, size=size, replace=False)
            predict_fn = spec.refit_on(X[sub], y[sub])
            tr_s.append(r2_score(y[sub], predict_fn(X[sub])))
            va_s.append(r2_score(y[va_idx], predict_fn(X[va_idx])))
        tr_mean.append(np.mean(tr_s)); tr_std.append(np.std(tr_s))
        va_mean.append(np.mean(va_s)); va_std.append(np.std(va_s))
    return sizes, np.array(tr_mean), np.array(tr_std), np.array(va_mean), np.array(va_std)


# ── PER-MODEL LEARNING CURVES ────────────────────────────────────────────────
lc_results = {}   # name -> (sizes, train_mean, train_std, val_mean, val_std)

for spec in models:
    color = COLORS[spec.name]
    sizes, train_mean, train_std, val_mean, val_std = get_learning_curve(spec, X_tr, y_tr)
    lc_results[spec.name] = (sizes, train_mean, train_std, val_mean, val_std)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(sizes, train_mean, color=color, linewidth=2, linestyle='-',
           marker='o', markersize=4, label='Training Score')
    ax.plot(sizes, val_mean, color=color, linewidth=2, linestyle='--',
           marker='s', markersize=4, alpha=0.75, label='Validation Score')
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.12, color=color, label='Train ±1 std')
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.12, color=color, label='Val ±1 std')

    final_gap = train_mean[-1] - val_mean[-1]
    ax.annotate(f'Gap: {final_gap:.3f}',
               xy=(sizes[-1], val_mean[-1]),
               xytext=(-60, -18), textcoords='offset points',
               fontsize=9, color='dimgray',
               arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8))

    ax.set_title(f"Learning Curve ({spec.name})", fontsize=14,
                fontweight='bold', pad=10, fontfamily='serif')
    ax.set_xlabel('Training Samples', labelpad=8)
    ax.set_ylabel('R² Score', labelpad=8)
    ax.set_ylim(-0.1, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.axhline(y=1.0, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9, frameon=True, framealpha=0.9,
             edgecolor='lightgray', borderpad=1, labelspacing=0.6, handlelength=2.5)
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')

    plt.tight_layout()
    save_fig(plt.gcf(), f"learning_curve_{spec.name.replace(' ', '_')}", subdir="learning_curves")
    plt.show()
    plt.close()


# ── COMBINED VALIDATION CURVE (all models on one plot) ──────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for spec in models:
    color = COLORS[spec.name]
    sizes, _, _, val_mean, val_std = lc_results[spec.name]
    ax.plot(sizes, val_mean, color=color, linewidth=2, marker='o', markersize=4, label=spec.name)
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.08, color=color)

ax.set_title("Validation Score Comparison", fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel("Training Samples", labelpad=8)
ax.set_ylabel("R² Score (CV=5)", labelpad=8)
ax.set_ylim(-0.1, 1.05)
ax.axhline(y=1.0, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
ax.legend(loc='lower right', fontsize=8, frameon=True, framealpha=0.9,
         edgecolor='lightgray', borderpad=1, labelspacing=0.5, handlelength=2.5, ncol=2)
ax.grid(True, linestyle='--', alpha=0.4, color='gray')

plt.tight_layout()
save_fig(plt.gcf(), "validation_curve_comparison", subdir="learning_curves")
plt.show()
plt.close()


# ── TEST R² BAR CHART ─────────────────────────────────────────────────────────
#  Uses the held-out test R2 already computed by evaluate_all() just above,
#  instead of re-deriving a separate CV score here.
fig, ax = plt.subplots(figsize=(8, 5.5))

sorted_specs = sorted(models, key=lambda s: RESULTS[s.name]['R2'], reverse=True)
names  = [s.name for s in sorted_specs]
scores = [RESULTS[s.name]['R2'] for s in sorted_specs]
colors = [COLORS[n] for n in names]

bars = ax.barh(names, scores, color=colors, edgecolor='white', linewidth=0.8, height=0.6)
for bar, score in zip(bars, scores):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
           f'{score:.4f}', va='center', fontsize=10, color='dimgray')

ax.set_xlim(0, max(scores) * 1.18)
ax.set_xlabel("Test R² Score", labelpad=8)
ax.set_title("Model Comparison — Held-Out Test R²", fontsize=14, fontweight='bold', pad=10)
ax.grid(True, axis='x', linestyle='--', alpha=0.4, color='gray')
ax.invert_yaxis()

plt.tight_layout()
save_fig(plt.gcf(), "test_r2_comparison", subdir="model_comparison")
plt.show()
plt.close()


# =============================================================================
#  SECTION 7 — TRAIN vs TEST METRICS TABLE  (per already-tuned model)
# =============================================================================
#  Assumes fit_all() (Part 5) already ran, so every spec in `models` has a
#  working .predict(). Reuses compute_metrics() from Part 1 so these numbers
#  are defined identically to everywhere else in the pipeline — same
#  RMSE/MAE/MAPE/SI/R2. F1 score and accuracy from the original snippet are
#  dropped rather than carried along commented-out: those are classification
#  metrics and don't apply to a regression target like compressive strength.
# =============================================================================
def build_metrics_df(models, X, y):
    """Predict on (X, y) for each already-fitted model and return one row per
    model with RMSE / MAE / MAPE / SI / R2."""
    rows = []
    for spec in models:
        pred = spec.predict(X)
        m = compute_metrics(y, pred)
        rows.append({"Model": spec.name, **m})
    return pd.DataFrame(rows)[["Model", "RMSE", "MAE", "MAPE", "SI", "R2"]]


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

# ---- Optional: side-by-side train/test view, useful for spotting overfitting.
#  A model whose Train R2 is much higher than its Test R2 (or whose Train RMSE
#  is much lower than its Test RMSE) is fitting noise rather than signal — the
#  same gap the learning curves in Section 6 visualize directly.
train_test_gap = results_df_Train.merge(
    results_df_Test, on="Model", suffixes=(" (Train)", " (Test)")
)
train_test_gap["R2 Gap (Train-Test)"] = (
    train_test_gap["R2 (Train)"] - train_test_gap["R2 (Test)"]
)
train_test_gap = train_test_gap.sort_values("R2 Gap (Train-Test)", ascending=False)

print("\n" + "=" * 78)
print("TRAIN vs TEST  (sorted by overfitting gap, largest first)")
print("=" * 78)
print(train_test_gap.round(4).to_string(index=False))
register_table("Train vs test gap", train_test_gap)


# =============================================================================
#  SECTION 8 — ACTUAL vs PREDICTED SCATTER PLOTS  (per already-tuned model)
# =============================================================================
#  Assumes fit_all() (Part 5) already ran, so every spec in `models` is
#  already fitted and tuned. The original snippet called model.fit(x_train,
#  y_train) again inside the plotting loop — for this pipeline that would
#  silently repeat each model's hyperparameter search (another 40-80 Optuna
#  trials for xgboost_model, lgbm, mlp, cnn_lstm, svr, seq_linear, on top of
#  what fit_all() already paid for), so it's dropped: this section only reads
#  predictions, it never refits.
#
#  Metrics in each legend box are computed fresh via compute_metrics() (same
#  definitions as Sections 4/6/7) rather than looked up by row position from
#  results_df_Train, since a positional lookup silently breaks if `models` is
#  ever reordered or filtered. SI is included alongside RMSE/MAE/MAPE/R2 for
#  consistency with the rest of the pipeline.
# =============================================================================
from matplotlib.lines import Line2D


def plot_actual_vs_predicted(models, X, y, dataset_label):
    """Actual-vs-predicted scatter, fitted-line diagonal, and a metrics box,
    one figure per model, for whichever dataset (train or test) is passed."""
    for spec in models:
        predictions = spec.predict(X)
        m = compute_metrics(y, predictions)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y, predictions, alpha=0.5, edgecolors='black',
                  facecolors='white', linewidths=1.5)

        # Diagonal spans the full range of actual AND predicted values, not
        # just y.min()/y.max(), so predictions outside the observed range
        # still land on the line rather than off the edge of the plot.
        lo = min(y.min(), predictions.min())
        hi = max(y.max(), predictions.max())
        ax.plot([lo, hi], [lo, hi], color='black', linestyle='-', linewidth=2)
        ax.set_aspect('equal', adjustable='box')   # square axes: a true 45° diagonal

        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(spec.name, fontweight='bold')
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        # Spines/tick styling already comes from the rcParams set in Section 6.

        legend_label = (f"RMSE: {m['RMSE']:.3f}\nMAE: {m['MAE']:.3f}\n"
                        f"MAPE: {m['MAPE']:.2f}%\nSI: {m['SI']:.2f}%\n"
                        f"R\u00b2: {m['R2']:.3f}")
        legend_item = Line2D([], [], color='none', marker=None, label=legend_label)
        legend = ax.legend(handles=[legend_item], loc='upper left',
                          handlelength=0, frameon=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.0)

        plt.tight_layout()
        save_fig(plt.gcf(), f"{spec.name.replace(' ', '_')}_{dataset_label}", subdir="predictions")
        plt.show()
        plt.close()


plot_actual_vs_predicted(all_models, X_tr, y_tr, "train")
plot_actual_vs_predicted(all_models, X_te, y_te, "test")


# =============================================================================
#  SECTION 13 — TAYLOR DIAGRAM
# =============================================================================
#  Compares every model on three statistics at once, in one polar plot:
#     radius            = standard deviation of the model's predictions
#     azimuth           = Pearson correlation with the observations, as
#                         arccos(R), so R=1 lies on the horizontal axis
#     distance from the = centered RMSE
#     reference star
#
#  These three are not independent — they are bound by the law of cosines,
#  which is exactly WHY the geometry works:
#
#      CRMSE^2 = SD_obs^2 + SD_pred^2 - 2 * SD_obs * SD_pred * R
#
#  The reference star sits at R=1 and SD=SD_obs (normalized: radius 1). The
#  closer a marker is to that star, the better the model. A marker INSIDE the
#  dotted SD_obs arc under-disperses (predictions too flat, variance
#  under-estimated — typical of an over-regularized or over-smoothed model);
#  outside it over-disperses.
#
#  ---------------------------------------------------------------------------
#  THE ONE THING THIS PLOT DOES NOT SHOW: BIAS
#  ---------------------------------------------------------------------------
#  The identity above holds only for the CENTERED RMSE, i.e. after removing
#  each series' mean. Total error decomposes as
#
#      MSE_total = CRMSE^2 + Bias^2
#
#  so a model with a large systematic offset can sit right next to the
#  reference star while being consistently wrong by several MPa. Feeding TOTAL
#  RMSE into a Taylor diagram instead of centered RMSE is a common error and
#  silently breaks the geometry. The companion table below therefore reports
#  Bias explicitly alongside CRMSE — read them together, and cross-check
#  against the RMSE/MAPE columns from Section 7.
#
#  All standard deviations use ddof=0 (population), consistently with the
#  covariance implied by R, which is what makes the identity hold exactly.
# =============================================================================
def taylor_statistics(y_true, y_pred):
    """SD / R / centered-RMSE / bias for one model against the observations."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    sd_obs  = float(np.std(y_true, ddof=0))
    sd_pred = float(np.std(y_pred, ddof=0))
    R       = float(np.corrcoef(y_true, y_pred)[0, 1])
    # max(..., 0) guards against a tiny negative from floating-point round-off
    crmse   = float(np.sqrt(max(sd_obs**2 + sd_pred**2 - 2*sd_obs*sd_pred*R, 0.0)))
    return {"SD": sd_pred, "R": R, "CRMSE": crmse,
            "Bias": float(y_pred.mean() - y_true.mean()), "SD_obs": sd_obs}


def taylor_diagram(entries, sd_obs, title="Taylor Diagram", normalize=True,
                   colors=None, figsize=(9.5, 7.5), savepath=None, zoom=False):
    """
    entries: list of dicts from taylor_statistics(), each with a "name" key.
    normalize=True divides every SD by SD_obs so the reference sits at radius
    1 — the usual choice when comparing many models, since it makes the
    dispersion ratio readable directly off the radial axis.
    zoom=True crops the axes to the region the models actually occupy. Well
    tuned models all land at R > 0.9 with SD near SD_obs, so on the full
    quarter circle they pile into one indistinguishable blob; the zoomed view
    is what actually lets you rank them. Keep the full view for context and
    the zoomed one for reading off differences.
    """
    ref   = 1.0 if normalize else sd_obs
    scale = (1.0 / sd_obs) if normalize else 1.0
    rmax  = max([ref] + [e["SD"] * scale for e in entries]) * 1.35

    #  A standard Taylor diagram is a quarter circle (R from 1 down to 0). If
    #  any model is anticorrelated, arccos(R) exceeds 90 deg and would fall
    #  off the plot entirely, so the arc is extended to a half circle instead
    #  of silently dropping that model.
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
        # Tick only inside the visible wedge, else every label collapses at R=1.
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
                   colors='#888888', linestyles='--', linewidths=0.9)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.2f', colors='#555555')

    # Arc of "same variability as the observations".
    ax.plot(np.linspace(0, theta_max, 200), np.full(200, ref),
           color='#333333', linestyle=':', linewidth=1.3)
    ax.plot([0], [ref], marker='*', markersize=18, color='black',
           linestyle='none', label='Observed (reference)', zorder=6)

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h', 'p', '*', '8']
    for i, e in enumerate(entries):
        ax.plot([np.arccos(e["R"])], [e["SD"] * scale],
               marker=markers[i % len(markers)], markersize=8,
               color=(colors or {}).get(e["name"], f"C{i % 10}"),
               markeredgecolor='black', markeredgewidth=0.6,
               linestyle='none', label=f'{i + 1}. {e["name"]}', zorder=5)
        # numbered so 20 markers stay legible; the legend maps number -> name
        ax.annotate(str(i + 1), (np.arccos(e["R"]), e["SD"] * scale),
                   textcoords='offset points', xytext=(6, 4), fontsize=7.5)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=26)
    #  Positioned relative to the VISIBLE radial span. Using rmax * 1.16 breaks
    #  in zoomed mode, where rmin is large and that lands far outside the axes.
    _r_label = rmax + 0.10 * (rmax - rmin)
    ax.text(theta_max * 0.5, _r_label, "Correlation coefficient",
           ha='center', va='center', fontsize=10, clip_on=False,
           rotation=(-np.degrees(theta_max) / 2 if min_R >= 0 else 0))
    ax.set_xlabel(("Normalized " if normalize else "") + "Standard deviation", labelpad=18)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='upper left', bbox_to_anchor=(1.10, 1.02), fontsize=8,
             frameon=True, framealpha=0.95, edgecolor='lightgray')
    fig.tight_layout()
    if savepath:
        save_fig(fig, savepath, subdir="taylor")
    return fig, ax


def build_taylor(specs, X, y, dataset_label, normalize=True):
    """Stats table + diagram for one dataset (train or test)."""
    entries = []
    for spec in specs:
        s = taylor_statistics(y, spec.predict(X))
        s["name"] = spec.name
        entries.append(s)

    sd_obs = entries[0]["SD_obs"]
    df = pd.DataFrame([{
        "Model": e["name"], "SD": e["SD"], "SD/SD_obs": e["SD"] / sd_obs,
        "R": e["R"], "CRMSE": e["CRMSE"], "Bias": e["Bias"],
    } for e in entries]).sort_values("CRMSE")

    print("\n" + "=" * 78)
    print(f"TAYLOR STATISTICS — {dataset_label.upper()}   (SD_obs = {sd_obs:.4f} MPa)")
    print("=" * 78)
    print(df.round(4).to_string(index=False))
    register_table(f"Taylor {dataset_label}", df)

    for zoom, suffix in [(False, ""), (True, "_zoom")]:
        fig, _ = taylor_diagram(
            entries, sd_obs,
            title=(f"Taylor Diagram — {dataset_label.capitalize()} Set"
                   + (" (zoomed)" if zoom else "")),
            normalize=normalize, colors=COLORS, zoom=zoom,
            savepath=f"taylor_diagram_{dataset_label}{suffix}.png")
        plt.show()
        plt.close(fig)
    return df


taylor_df_test  = build_taylor(all_models, X_te, y_te, "test")
taylor_df_train = build_taylor(all_models, X_tr, y_tr, "train")


# =============================================================================
#  SECTION 11 — BIAS-VARIANCE TRADEOFF
# =============================================================================
#  Assumes Sections 4 and 6 have run (models fitted; COLORS defined).
#
#  Bootstrap decomposition of expected squared error. Train each model on B
#  bootstrap resamples of the training set, predict the same fixed test set
#  each time, then for every test point:
#
#      bias^2   = ( mean_b[ pred_b(x) ] - y(x) )^2      systematic error
#      variance = Var_b[ pred_b(x) ]                    sensitivity to the sample
#
#  Averaged over test points these sum EXACTLY to the expected test MSE
#  (E_b[(pred - y)^2] = (mean_pred - y)^2 + Var(pred) pointwise), which is why
#  the stacked bars below total the model's error with nothing left over.
#  Irreducible noise is not separately identifiable from a single realisation
#  of y, so it sits inside the bias^2 term — read bias^2 as "bias + noise".
#
#  Refits use spec.refit_on(), so the already-tuned hyperparameters are reused
#  and no search is repeated. Cost is n_bootstrap x len(models) fits; the three
#  Keras models dominate. Trim N_BOOTSTRAP or restrict `bv_models` if slow.
# =============================================================================
N_BOOTSTRAP = 20
bv_models = models          # e.g. [rf, svr, xgboost_model, lgbm, ada, knn] for a faster pass


def bias_variance_decomposition(spec, X_train, y_train, X_test, y_test,
                                n_bootstrap=None, params=None):
    """Returns (bias_sq, variance, total_mse) via bootstrap resampling."""
    n_bootstrap = N_BOOTSTRAP if n_bootstrap is None else n_bootstrap
    rng = np.random.RandomState(SEED)
    preds = np.zeros((n_bootstrap, len(X_test)))
    for b in range(n_bootstrap):
        idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        predict_fn = spec.refit_on(X_train[idx], y_train[idx], params=params)
        preds[b] = predict_fn(X_test)
    main_pred = preds.mean(axis=0)
    bias_sq  = float(np.mean((main_pred - y_test) ** 2))
    variance = float(np.mean(preds.var(axis=0)))
    return bias_sq, variance, bias_sq + variance


# ── PLOT 1: bias^2 vs variance across all tuned models ───────────────────────
bv_rows = []
for spec in bv_models:
    b2, var, total = bias_variance_decomposition(spec, X_tr, y_tr, X_te, y_te)
    bv_rows.append({"Model": spec.name, "Bias^2": b2, "Variance": var,
                    "Total MSE": total, "Variance %": 100 * var / total})
    print(f"{spec.name:30s} bias^2={b2:8.3f}  variance={var:8.3f}  total={total:8.3f}")

bv_df = pd.DataFrame(bv_rows).sort_values("Total MSE")
print("\n" + "=" * 78)
print("BIAS-VARIANCE DECOMPOSITION")
print("=" * 78)
print(bv_df.round(4).to_string(index=False))
register_table("Bias-variance", bv_df)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(bv_df["Model"], bv_df["Bias^2"], color='#4C72B0',
       edgecolor='white', linewidth=0.8, height=0.6, label='Bias$^2$')
ax.barh(bv_df["Model"], bv_df["Variance"], left=bv_df["Bias^2"], color='#DD8452',
       edgecolor='white', linewidth=0.8, height=0.6, label='Variance')

for i, total in enumerate(bv_df["Total MSE"]):
    ax.text(total * 1.01, i, f'{total:.2f}', va='center', fontsize=9, color='dimgray')

ax.set_xlabel("Expected Test MSE  (Bias$^2$ + Variance)", labelpad=8)
ax.set_title("Bias-Variance Decomposition by Model", fontsize=14, fontweight='bold', pad=10)
ax.set_xlim(0, bv_df["Total MSE"].max() * 1.15)
ax.legend(loc='lower right', fontsize=9, frameon=True, framealpha=0.9, edgecolor='lightgray')
ax.grid(True, axis='x', linestyle='--', alpha=0.4, color='gray')
ax.invert_yaxis()
plt.tight_layout()
save_fig(plt.gcf(), "bias_variance_decomposition", subdir="bias_variance")
plt.show()
plt.close()


# ── PLOT 2: the classic tradeoff curve vs model complexity ───────────────────
def plot_bias_variance_curve(spec, param_name, values, xlabel, n_bootstrap=10,
                             invert_x=False):
    """
    Sweep ONE complexity hyperparameter, holding the rest at their tuned
    values, and plot bias^2 / variance / total against it. The tuned value is
    marked so you can see where the search landed relative to the minimum.
    """
    b2s, variances, totals = [], [], []
    for v in values:
        params = dict(spec.best_params)
        params[param_name] = v
        b2, var, total = bias_variance_decomposition(
            spec, X_tr, y_tr, X_te, y_te, n_bootstrap=n_bootstrap, params=params)
        b2s.append(b2); variances.append(var); totals.append(total)
        print(f"  {spec.name} | {param_name}={v}: bias^2={b2:.3f} variance={var:.3f} total={total:.3f}")

    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, b2s, marker='o', markersize=5, linewidth=2, color='#4C72B0', label='Bias$^2$')
    ax.plot(x, variances, marker='s', markersize=5, linewidth=2, color='#DD8452', label='Variance')
    ax.plot(x, totals, marker='^', markersize=5, linewidth=2.5, color=EDA_EDGE_COLOR,
           linestyle='--', label='Total MSE')

    tuned = spec.best_params.get(param_name)
    if tuned in values:
        ax.axvline(list(values).index(tuned), color='gray', linestyle=':', linewidth=1.5)
        ax.text(list(values).index(tuned), ax.get_ylim()[1] * 0.95, ' tuned',
               fontsize=9, color='gray', va='top')

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values])
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel("Error", labelpad=8)
    ax.set_title(f"Bias-Variance Tradeoff ({spec.name})", fontsize=14, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.9, edgecolor='lightgray')
    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
    plt.tight_layout()
    save_fig(plt.gcf(), f"bias_variance_curve_{spec.name.replace(' ', '_')}", subdir="bias_variance")
    plt.show()
    plt.close()


# KNN: complexity DECREASES as k grows, so the x-axis is inverted to keep the
# conventional "simple on the left, complex on the right" reading.
plot_bias_variance_curve(
    knn, "model__n_neighbors", [1, 2, 3, 5, 8, 12, 20, 30, 40],
    xlabel="k  (fewer neighbours = more complex)", invert_x=True)

# Random Forest: complexity increases with depth.
plot_bias_variance_curve(
    rf, "max_depth", [2, 3, 5, 8, 12, 20, None],
    xlabel="max_depth  (deeper = more complex)")


# =============================================================================
#  SECTION 12 — SEED SENSITIVITY  (optional honesty check)
# =============================================================================
#  Refits the tuned models across several different train/test splits and
#  reports mean +/- std of test RMSE. If a model's headline number holds up
#  here, the split was not doing the work. Hyperparameters are held at their
#  their already-tuned (fit_all(), Part 5) values — this measures split
#  sensitivity, not tuning stability
#  (that is what the nested CV in Sections 2/4 is for).
# =============================================================================
SEED_SENSITIVITY_SEEDS = [BEST_SEED, 0, 1, 7, 42]


def seed_sensitivity(models, X_df, y_ser, seeds=None, test_size=None):
    seeds = SEED_SENSITIVITY_SEEDS if seeds is None else seeds
    test_size = TEST_SIZE if test_size is None else test_size
    rows = []
    for spec in models:
        scores = []
        for s in seeds:
            Xa, Xb, ya, yb = train_test_split(X_df, y_ser, test_size=test_size, random_state=s)
            Xa = np.asarray(Xa, dtype=np.float32); Xb = np.asarray(Xb, dtype=np.float32)
            ya = np.asarray(ya, dtype=np.float32).ravel(); yb = np.asarray(yb, dtype=np.float32).ravel()
            predict_fn = spec.refit_on(Xa, ya)
            scores.append(rmse(yb, predict_fn(Xb)))
        rows.append({"Model": spec.name, "RMSE mean": np.mean(scores),
                     "RMSE std": np.std(scores), "RMSE min": np.min(scores),
                     "RMSE max": np.max(scores)})
        print(f"{spec.name:30s} {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return pd.DataFrame(rows).sort_values("RMSE mean")


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
#  Assumes fit_all() (Part 5) already ran, so every spec in `models` is
#  already fitted and tuned.  pip install shap
#
#  Explainer choice mirrors the original logic — tree models -> TreeExplainer,
#  linear models -> LinearExplainer, everything else -> a generic black-box
#  Explainer — with one important fix: the black-box path now wraps
#  spec.predict, not model.predict. For the KNN/SVR Pipelines and all three
#  Keras models (mlp, cnn_lstm, seq_linear), model.predict expects data in a
#  different form than the raw features: the Keras nets were trained on
#  STANDARDIZED inputs via a scaler that lives inside spec.predict_fn, not on
#  the model object itself. Calling model.predict directly on raw feature
#  values would silently feed wildly out-of-distribution inputs to the
#  network — garbage in, garbage SHAP values out. spec.predict already
#  handles this scaling for every model type uniformly, exactly like it does
#  everywhere else in this pipeline.
#
#  X_tr/X_te (numpy) are used throughout instead of x_train/x_test, so results
#  match exactly what every other section evaluates on — x_test_arr from the
#  original snippet is just X_te and doesn't need recomputing.
#
#  The black-box fallback re-evaluates the model many times per row it
#  explains, so its background is capped at SHAP_BACKGROUND_SIZE and the rows
#  actually explained are capped at SHAP_TEST_SIZE. TreeExplainer is exact
#  (no sampling) so this doesn't affect the tree models, but it matters a lot
#  for the three Keras nets and the two Pipelines. Trim these further, or
#  restrict `models` to the tree-based subset, if this is still too slow.
# =============================================================================
import shap

feature_names = (list(x_train.columns) if hasattr(x_train, "columns")
                 else [f"Feature_{i}" for i in range(N_FEATURES)])

TREE_KEYS = ("RandomForest", "ExtraTrees", "DecisionTree", "GradientBoosting",
             "HistGradientBoosting", "XGB", "LGBM", "CatBoost")

SHAP_BACKGROUND_SIZE = 100             # background rows for the black-box fallback
SHAP_TEST_SIZE = min(200, len(X_te))   # rows actually explained/plotted

_shap_rng = np.random.RandomState(SEED)
_bg_idx = _shap_rng.choice(len(X_tr), size=min(SHAP_BACKGROUND_SIZE, len(X_tr)), replace=False)
_te_idx = _shap_rng.choice(len(X_te), size=SHAP_TEST_SIZE, replace=False)
X_bg   = X_tr[_bg_idx]      # background for the black-box explainer only
X_shap = X_te[_te_idx]      # rows explained/plotted for every model

# ── BUILD SHAP EXPLANATIONS FOR EACH MODEL ────────────────────────────────────
#  Two things here are less obvious than they look, and both are version
#  sensitive rather than model sensitive:
#
#  1. `.shap_values()` is the OLD interface. TreeExplainer and LinearExplainer
#     still expose it, but the generic `shap.Explainer` dispatches to whatever
#     algorithm suits the model — Exact, Permutation, Partition — and those
#     classes only implement `__call__`, so the black-box path must be called
#     as `explainer(X)` and read `.values` off the Explanation it returns.
#     Calling `.shap_values()` there raises
#     `AttributeError: 'ExactExplainer' object has no attribute 'shap_values'`,
#     which reads like a SHAP bug and is really an API-generation mismatch.
#     It only bites the non-tree, non-linear models — here the KNN and SVR
#     Pipelines and the three Keras nets — so it hides until one of those is
#     in `models`.
#
#  2. base_values must be ONE VALUE PER ROW, as an array. shap's Explanation
#     arithmetic (exp.abs.mean(0), which the bar plot uses) reads
#     self.base_values.shape, and a bare Python float has no .shape. The
#     per-row array also keeps the base value aligned when rows are sliced,
#     which is what the waterfall plot below does.
#
#  The whole construction is wrapped per model: an explainer that fails is a
#  lost diagnostic, not a reason to abort Part 7 after every model has already
#  been tuned.
explanations, explained_specs = [], []

for spec in models:
    model = spec.model
    cls = type(model).__name__
    try:
        if any(k in cls for k in TREE_KEYS):
            explainer = shap.TreeExplainer(model, feature_names=feature_names)
            values = explainer.shap_values(X_shap)
            base = float(np.ravel(explainer.expected_value)[0])
        elif hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_tr, feature_names=feature_names)
            values = explainer.shap_values(X_shap)
            base = float(np.ravel(explainer.expected_value)[0])
        else:
            # spec.predict, never model.predict — see the note above.
            explainer = shap.Explainer(spec.predict, X_bg, feature_names=feature_names)
            values = explainer(X_shap).values
            base = float(np.mean(spec.predict(X_bg)))

        values = np.asarray(values[0] if isinstance(values, list) else values)
        if values.ndim == 3:               # (n, f, outputs) -> single output
            values = values[..., 0]

        explanations.append(shap.Explanation(
            values        = values,
            base_values   = np.full(len(values), base, dtype=np.float64),
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
        plt.title(title, fontsize=14, fontweight='bold')
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
                 f"Feature Importance ({spec.name})",
                 f"shap_bar_{spec.name.replace(' ', '_')}")

# ── BEESWARM PLOTS ────────────────────────────────────────────────────────────
for spec, explanation in zip(explained_specs, explanations):
    _shap_figure(shap.plots.beeswarm, explanation,
                 f"SHAP Beeswarm ({spec.name})",
                 f"shap_beeswarm_{spec.name.replace(' ', '_')}")

# ── VIOLIN PLOTS ──────────────────────────────────────────────────────────────
#  Same global feature-importance information as the beeswarm plots, shown as
#  a per-feature distribution (violin shape) instead of individual scatter
#  points — easier to read the overall spread when there are many test rows.
for spec, explanation in zip(explained_specs, explanations):
    _shap_figure(shap.plots.violin, explanation,
                 f"SHAP Violin ({spec.name})",
                 f"shap_violin_{spec.name.replace(' ', '_')}")

# ── WATERFALL PLOTS ───────────────────────────────────────────────────────────
sample_idx = 0

for spec, explanation in zip(explained_specs, explanations):
    _shap_figure(lambda e, show: shap.plots.waterfall(e[sample_idx], show=show),
                 explanation,
                 f"SHAP Waterfall ({spec.name})",
                 f"shap_waterfall_{spec.name.replace(' ', '_')}")

# ── DEPENDENCE PLOTS ──────────────────────────────────────────────────────────
feature = feature_names[0]  # Replace with your feature of interest

for spec, explanation in zip(explained_specs, explanations):
    _shap_figure(lambda e, show: shap.plots.scatter(e[:, feature], show=show),
                 explanation,
                 f"SHAP Dependence ({feature}) — {spec.name}",
                 f"shap_dependence_{_fname(feature)}_{spec.name.replace(' ', '_')}")




# =============================================================================
#  SECTION 9B — PERMUTATION IMPORTANCE
# =============================================================================
#  The second view of feature importance in Part 7, and the bluntest question
#  of the three: if this feature were pure noise, how much worse would the
#  model be? Shuffle one column, re-score, and the drop IS the importance.
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
#  Both splits are computed. Train-set permutation measures how much a feature
#  was used to FIT; test-set permutation measures how much it carries to
#  unseen data, which is almost always the question being asked. The gap
#  between them is itself diagnostic: a feature that matters a lot on train
#  and nothing on test was memorised, not learned.
#
#  THE CORRELATION CAVEAT — the same one the PDP section carries, for the same
#  reason. Shuffling one column of a mix design produces rows that are
#  physically impossible (cement from one mix, water from another, violating
#  the unit-volume and w/b constraints), and the model is then scored on them.
#  Worse, with two strongly correlated features each can be shuffled with
#  little damage because the other still carries the signal, so BOTH look
#  unimportant and the pair's real contribution goes unreported. Read the
#  correlation heatmaps from Part 3 alongside this, and treat a low score for
#  a feature that has a highly correlated partner as "not UNIQUELY important"
#  rather than "not important".
#
#  Cost is (n_features x n_repeats + 1) predictions per model. Nothing is
#  refitted, so this scales with the roster rather than with training.
# =============================================================================
PERM_N_REPEATS = 10          # shuffles per feature; more repeats = tighter error bars
PERM_MODELS    = models      # use all_models to also cover the Section 4B ensembles


def permutation_importance_spec(spec, X, y, n_repeats=None, seed=None):
    """
    Model-agnostic permutation importance for one already-fitted ModelSpec.

    Written against spec.predict rather than sklearn's permutation_importance
    so the Keras nets and the Section 4B ensembles — neither of which is an
    sklearn estimator — travel exactly the same code path as the trees.

    Returns (baseline_r2, drops), where drops has shape (n_features, n_repeats)
    and every entry is baseline - R2_after_shuffling. Positive means the model
    got worse without the feature, i.e. the feature was carrying something.
    """
    n_repeats = PERM_N_REPEATS if n_repeats is None else n_repeats
    seed = SEED if seed is None else seed
    rng = np.random.RandomState(seed)
    X = np.array(X, dtype=np.float64, copy=True)    # never mutate the caller's array
    baseline = r2_score(y, spec.predict(X))
    drops = np.empty((X.shape[1], n_repeats), dtype=np.float64)
    for j in range(X.shape[1]):
        original = X[:, j].copy()
        for r in range(n_repeats):
            X[:, j] = rng.permutation(original)
            drops[j, r] = baseline - r2_score(y, spec.predict(X))
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
                             "Baseline R2": baseline,
                             "Mean drop": drops[j].mean(), "Std": drops[j].std(),
                             "Rank": int(rank[j])})
        print(f"  {spec.name:32s} baseline R2  "
              f"test={store[(spec.name,'test')][0]:.4f}  "
              f"train={store[(spec.name,'train')][0]:.4f}")
    return pd.DataFrame(rows), store


print("\n" + "=" * 78)
print(f"PERMUTATION IMPORTANCE  ({PERM_N_REPEATS} shuffles per feature)")
print("=" * 78)
perm_df, PERM_RESULTS = compute_permutation_importance()
register_table("Permutation importance", perm_df)


# ── PER-MODEL BAR CHARTS (mean drop +/- std over the repeats) ─────────────────
def plot_permutation_importance(specs=None, split="test"):
    specs = PERM_MODELS if specs is None else specs
    for spec in specs:
        baseline, drops = PERM_RESULTS[(spec.name, split)]
        means, stds = drops.mean(axis=1), drops.std(axis=1)
        order = np.argsort(means)                    # ascending: biggest at the top
        fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.42 * len(order) + 1.4)))
        ax.barh([feature_names[i] for i in order], means[order],
                xerr=stds[order], color=COLORS[spec.name], edgecolor='black',
                linewidth=0.6, error_kw=dict(ecolor='black', lw=1.0, capsize=3))
        ax.axvline(0, color='black', linewidth=0.9)
        ax.set_xlabel("Drop in $R^2$ when the feature is shuffled")
        ax.set_title(f"Permutation Importance — {spec.name}  ({split}, "
                     f"baseline $R^2$ = {baseline:.3f})",
                     fontsize=12, fontweight='bold')
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
        xerr=_c["Std across models"].fillna(0.0), color='#0072B2',
        edgecolor='black', linewidth=0.6,
        error_kw=dict(ecolor='black', lw=1.0, capsize=3))
ax.axvline(0, color='black', linewidth=0.9)
ax.set_xlabel("Mean drop in $R^2$ (averaged across models)")
ax.set_title(f"Permutation Importance — consensus over {len(PERM_MODELS)} models",
             fontsize=12, fontweight='bold')
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
#  answers "why THIS prediction?" — it perturbs a single row, watches how the
#  model's output moves, and fits a small weighted linear model to that local
#  neighbourhood. The coefficients of that little model are the explanation.
#
#  Why keep it when SHAP is already here. They are not the same calculation
#  dressed differently: SHAP's local explanations sum exactly to the
#  prediction (the efficiency axiom), LIME's do not and make no such promise.
#  What LIME gives instead is a readable IF-THEN reading of the neighbourhood
#  — "0.46 < cement <= 0.73 pushed this prediction up by 4.2" — because it
#  discretises the features into bins first. Practitioners find that phrasing
#  easier to argue with, and arguing with an explanation is the point.
#
#  READ exp.score BEFORE YOU BELIEVE ANY OF IT. That is the local surrogate's
#  own R-squared: how well the little linear model reproduces the real model
#  in that neighbourhood. It is reported for every explanation below, and it
#  is often mediocre. A LIME explanation with a local R-squared of 0.15 is not
#  a weak explanation, it is not an explanation at all — the linear surrogate
#  simply does not describe the model there, and the weights are noise. This
#  is the most common way LIME is misread, so the number is printed next to
#  every single plot rather than buried.
#
#  WHICH ROWS. Explaining row 0 is a habit, not a choice, and row 0 is rarely
#  interesting. The rows picked here are the ones worth looking at: the
#  median-error case (typical behaviour), and the largest over- and
#  under-prediction (where the model fails, which is where an explanation
#  earns its keep).
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


def select_instances_to_explain(spec, X, y, k_extreme=1):
    """
    Rows worth explaining: the typical case and the failures.

    Returns [(index, label), ...] — the median absolute residual, then the
    k_extreme largest over-predictions and under-predictions.
    """
    resid = np.asarray(y).ravel() - spec.predict(X)
    order = np.argsort(np.abs(resid))
    picks = [(int(order[len(order) // 2]), "median error")]
    for i in np.argsort(resid)[:k_extreme]:          # most negative -> over-predicted
        picks.append((int(i), "worst over-prediction"))
    for i in np.argsort(resid)[::-1][:k_extreme]:    # most positive -> under-predicted
        picks.append((int(i), "worst under-prediction"))
    seen, unique = set(), []
    for idx, label in picks:                          # a tiny test set can collide
        if idx not in seen:
            seen.add(idx)
            unique.append((idx, label))
    return unique


def explain_one(explainer, spec, row, idx, label):
    """One LIME explanation, rendered and tabulated. Returns rows for the table."""
    exp = explainer.explain_instance(
        np.asarray(row, dtype=np.float64), spec.predict,
        num_features=LIME_N_FEATURES, num_samples=LIME_NUM_SAMPLES)
    pairs = exp.as_list()
    actual = float(np.asarray(y_te).ravel()[idx])
    predicted = float(spec.predict(row.reshape(1, -1))[0])
    local_pred = float(np.ravel(exp.local_pred)[0])
    intercept = float(exp.intercept[1])

    conditions = [c for c, _ in pairs][::-1]
    weights = np.array([w for _, w in pairs])[::-1]
    fig, ax = plt.subplots(figsize=(8.2, max(2.8, 0.44 * len(pairs) + 1.6)))
    ax.barh(conditions, weights,
            color=['#D55E00' if w < 0 else '#009E73' for w in weights],
            edgecolor='black', linewidth=0.6)
    ax.axvline(0, color='black', linewidth=0.9)
    ax.set_xlabel("Local weight (effect on this one prediction)")
    ax.set_title(f"LIME — {spec.name}\nrow {idx} ({label}):  actual = {actual:.3f},  "
                 f"model = {predicted:.3f}\nlocal surrogate $R^2$ = {exp.score:.3f}",
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    save_fig(fig, f"lime_{_fname(spec.name)}_row{idx}_{_slug(label)}",
             subdir="lime/per_instance")
    plt.show()
    plt.close(fig)

    print(f"\n  {spec.name} — row {idx} ({label})")
    print(f"    actual = {actual:.4f}   model = {predicted:.4f}   "
          f"local surrogate = {local_pred:.4f}")
    print(f"    local R2 = {exp.score:.4f}   intercept = {intercept:.4f}"
          + ("   << surrogate does not fit here; treat the weights as unreliable"
             if exp.score < 0.3 else ""))
    for cond, w in pairs:
        print(f"      {w:+9.4f}   {cond}")

    return [{"Model": spec.name, "Row": idx, "Case": label,
             "Actual": actual, "Predicted": predicted,
             "Local surrogate pred": local_pred, "Local R2": float(exp.score),
             "Intercept": intercept, "Condition": cond, "Weight": w}
            for cond, w in pairs]


if LIME_AVAILABLE:
    print("\n" + "=" * 78)
    print("LIME — LOCAL EXPLANATIONS")
    print("=" * 78)

    #  The explainer is built on the TRAINING data: that is the distribution
    #  the perturbations are drawn from and the bins are cut from. Building it
    #  on the test set would explain each row against a neighbourhood the
    #  model never saw.
    lime_explainer = LimeTabularExplainer(
        training_data   = np.asarray(X_tr, dtype=np.float64),
        training_labels = np.asarray(y_tr, dtype=np.float64).ravel(),
        feature_names   = list(feature_names),
        mode            = "regression",
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
                color='#CC79A7', edgecolor='black', linewidth=0.6,
                error_kw=dict(ecolor='black', lw=1.0, capsize=3))
        ax.set_xlabel("Mean |LIME weight| across explained rows")
        ax.set_title("LIME — global importance aggregated from local explanations",
                     fontsize=12, fontweight='bold')
        ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        fig.tight_layout()
        save_fig(fig, "lime_global_from_local", subdir="lime")
        plt.show()
        plt.close(fig)


# =============================================================================
#  SECTION 14 — ICE & PARTIAL DEPENDENCE PLOTS
# =============================================================================
#  Model-agnostic interpretability, sitting alongside SHAP in Part 7. SHAP
#  attributes a single prediction to each feature; ICE/PDP instead sweep a
#  feature across its range and show how the prediction RESPONDS. They answer
#  different questions and are worth reporting together.
#
#  Requires scikit-learn >= 1.1 (for `centered=True`).
#
#  ---------------------------------------------------------------------------
#  WHY THE ADAPTER BELOW IS NECESSARY
#  ---------------------------------------------------------------------------
#  PartialDependenceDisplay.from_estimator() demands a fitted sklearn
#  regressor. That excludes 3 of the 9 base models (mlp, cnn_lstm, seq_linear
#  are Keras) and ALL 11 ensembles from Section 4B. _SpecEstimator wraps any
#  ModelSpec in the minimal sklearn surface sklearn actually checks, so every
#  model in this pipeline plots through one code path.
#
#  Note the base-class order: `RegressorMixin, BaseEstimator`, NOT the reverse.
#  Since sklearn 1.6, is_regressor() resolves through __sklearn_tags__ along
#  the MRO, so putting BaseEstimator first makes sklearn fail to recognise the
#  wrapper as a regressor and raise "'estimator' must be a fitted regressor or
#  classifier." This is easy to get backwards and the error message does not
#  point at the cause.
#
#  ---------------------------------------------------------------------------
#  INTERPRETIVE CAVEAT — IMPORTANT FOR A CONCRETE MIX DATASET
#  ---------------------------------------------------------------------------
#  PDP works by holding every other feature fixed and sweeping one feature
#  across its marginal range, which assumes the features are INDEPENDENT. Mix
#  design badly violates that: the components are compositionally constrained
#  (they must sum to a unit volume) and the water/binder ratio ties water to
#  cementitious content. So a PDP grid point can correspond to a mix that is
#  physically impossible — e.g. more cement with every aggregate fraction and
#  the water held constant. The model still returns a number for it, and PDP
#  dutifully averages that number in.
#
#  Practical consequences: (1) trust the MIDDLE of each PDP curve far more
#  than its tails, where extrapolation is worst; (2) read ICE spread as
#  signal, not noise — if the ICE curves fan out rather than running parallel,
#  the feature's effect depends on the rest of the mix, i.e. there are
#  interactions, and the averaged PDP line is hiding them. The 2-way plots
#  below are the direct way to look at those interactions. Accumulated Local
#  Effects (ALE) plots are the standard fix for correlated features if you
#  want a version that does not fabricate impossible mixes.
# =============================================================================
from itertools import combinations
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import PartialDependenceDisplay

#  Brute-force PDP costs (n_rows x grid_resolution) predictions PER FEATURE,
#  and every ensemble prediction fans out into all of its member models. These
#  caps keep that tractable; PDP averaging over a few hundred representative
#  rows is statistically fine.
PDP_SAMPLE_SIZE   = 200    # rows used for 1-way ICE/PDP
PDP_2WAY_SAMPLE   = 100    # rows used for 2-way PDP
PDP_GRID_1WAY     = 50
PDP_GRID_2WAY     = 20     # 20x20 = 400 grid points per pair
PDP_TOP_K         = 6      # top-K features by SHAP -> C(6,2) = 15 pairs
N_COLS            = 3


class _SpecEstimator(RegressorMixin, BaseEstimator):
    """Read-only sklearn facade over an already-fitted ModelSpec."""
    def __init__(self, spec=None):
        self.spec = spec

    def fit(self, X, y=None):
        # Never trains anything. Present only because check_is_fitted() requires
        # the attribute to exist; the wrapped spec is already fitted.
        return self

    def __sklearn_is_fitted__(self):
        return self.spec is not None and self.spec.predict_fn is not None

    def predict(self, X):
        return self.spec.predict(np.asarray(X, dtype=np.float32))


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

        PartialDependenceDisplay.from_estimator(
            _SpecEstimator(spec),
            X_plot,
            features        = features,
            feature_names   = feature_names,
            kind            = 'both',
            subsample       = 50,
            centered        = True,
            grid_resolution = PDP_GRID_1WAY,
            random_state    = SEED,
            ice_lines_kw    = {'color': 'gray', 'alpha': 0.3, 'linewidth': 0.8},
            pd_line_kw      = {'color': 'black', 'linewidth': 2.5, 'label': 'Average'},
            ax              = axes_flat[:len(features)],
        )

        for ax in axes_flat:
            if not ax.has_data():
                ax.set_visible(False)
                continue
            ax.set_facecolor('white')
            ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.tick_params(axis='both', direction='out', bottom=True, top=False,
                          left=True, right=False)
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        fig.suptitle(f"ICE Plots ({spec.name})", fontsize=14, fontweight='bold')
        fig.tight_layout()
        save_fig(fig, f"ice_{spec.name.replace(' ', '_')}", subdir="ice_pdp")
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
                        feature_names=feature_names, kind='both', subsample=50,
                        centered=True, grid_resolution=PDP_GRID_1WAY, random_state=SEED,
                        ice_lines_kw={'color': 'gray', 'alpha': 0.3, 'linewidth': 0.8},
                        pd_line_kw={'color': 'black', 'linewidth': 2.5}, ax=ax1)
                except Exception as exc:
                    print(f"    per-feature ICE skipped for {spec.name}/"
                          f"{feature_names[feat]}: {type(exc).__name__}")
                    plt.close(f1)
                    continue
                ax1.set_title(feature_names[feat], fontsize=11, fontweight='bold')
                ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                if ax1.get_legend() is not None:
                    ax1.get_legend().remove()
                f1.tight_layout()
                save_fig(f1, f"ice_{_fname(spec.name)}_{feature_names[feat]}",
                         subdir="ice_pdp/per_feature", close=True)


plot_ice(ice_models, X_pdp)


# ── 2-WAY PDP (feature interactions) ─────────────────────────────────────────
#  Which pairs to plot: C(8,2) = 28 pairs is too many, so rather than keeping
#  an arbitrary first-N slice, the top PDP_TOP_K features are selected by mean
#  |SHAP| (consensus across the models explained in Section 9) and all pairs of
#  those are plotted. That makes the truncation principled — the pairs shown
#  are the ones the models actually rely on.
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

#  Trees only by default: 2-way PDP is (rows x grid^2) predictions per pair,
#  so Keras models and especially ensembles get expensive fast.
pdp_2way_models = [rf, xgboost_model, lgbm]


def plot_pdp_2way(specs, X_plot, pairs=None, n_cols=None):
    n_cols = N_COLS if n_cols is None else n_cols
    X_plot = _pdp_frame(X_plot)
    pairs = pairs if pairs is not None else pdp_pairs
    n_rows = int(np.ceil(len(pairs) / n_cols))

    for spec in specs:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 5.0 * n_rows),
                                constrained_layout=True)
        axes_flat = np.atleast_1d(axes).ravel()

        # sklearn computes the partial dependence; its default rendering is
        # then discarded and redrawn below as filled contours.
        display_obj = PartialDependenceDisplay.from_estimator(
            _SpecEstimator(spec), X_plot, features=pairs,
            feature_names=feature_names, grid_resolution=PDP_GRID_2WAY,
            ax=axes_flat[:len(pairs)])

        for idx, (ax, pd_result) in enumerate(zip(axes_flat[:len(pairs)], display_obj.pd_results)):
            ax.clear()

            grid_values = pd_result.get('grid_values', pd_result.get('values'))
            Z = np.asarray(pd_result['average'])[0]

            #  ORIENTATION. sklearn returns average[0] with shape
            #  (len(grid[0]), len(grid[1])) — axis 0 is the FIRST feature.
            #  np.meshgrid's default indexing='xy' returns the transpose of
            #  that, so pairing them directly silently swaps the two axes.
            #  The usual `if Z.shape != X.shape: Z = Z.T` guard does NOT catch
            #  it, because with one grid_resolution for both features the
            #  shapes are equal and the check passes while the data is still
            #  transposed. indexing='ij' matches sklearn's layout exactly and
            #  removes the ambiguity.
            G0, G1 = np.meshgrid(grid_values[0], grid_values[1], indexing='ij')

            cf = ax.contourf(G0, G1, Z, levels=12, cmap='viridis', alpha=0.95)

            cbar = fig.colorbar(cf, ax=ax, pad=0.04)
            cbar.outline.set_visible(True)
            cbar.outline.set_edgecolor('black')
            cbar.outline.set_linewidth(1.0)
            cbar.ax.tick_params(axis='y', direction='out', length=10, width=1,
                               colors='black', labelsize=10)

            f0, f1 = pairs[idx]
            ax.set_xlabel(feature_names[f0], fontsize=11, fontweight='bold', color='#333333')
            ax.set_ylabel(feature_names[f1], fontsize=11, fontweight='bold', color='#333333')
            ax.tick_params(axis='both', which='major', direction='out', length=4)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(1.0)

        for j in range(len(pairs), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f"2D Partial Dependence Plots ({spec.name})",
                    fontsize=17, fontweight='bold', color='#1a1a1a')
        save_fig(fig, f"pdp_2way_{spec.name.replace(' ', '_')}", subdir="ice_pdp")
        plt.show()
        plt.close(fig)

        #  Re-plot each pair on its own, reusing the partial dependence sklearn
        #  already computed above — no extra model evaluations. A grid of
        #  fifteen contour plots is unreadable at any printable size, and each
        #  pair is a separate analysis.
        if PER_ITEM_FIGURES:
            for idx, pd_result in enumerate(display_obj.pd_results):
                fa, fb = pairs[idx]
                grid_values = pd_result.get('grid_values', pd_result.get('values'))
                Z = np.asarray(pd_result['average'])[0]
                G0, G1 = np.meshgrid(grid_values[0], grid_values[1], indexing='ij')
                fp, axp = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
                cf = axp.contourf(G0, G1, Z, levels=12, cmap='viridis', alpha=0.95)
                cb = fp.colorbar(cf, ax=axp, pad=0.04)
                cb.ax.tick_params(labelsize=9)
                cb.set_label(f"Predicted {target_name}", fontsize=9)
                axp.set_xlabel(feature_names[fa], fontsize=11, fontweight='bold')
                axp.set_ylabel(feature_names[fb], fontsize=11, fontweight='bold')
                axp.set_title(f"{spec.name}: {feature_names[fa]} x {feature_names[fb]}",
                              fontsize=11, fontweight='bold')
                save_fig(fp, f"pdp2_{_fname(spec.name)}_{feature_names[fa]}"
                             f"_x_{feature_names[fb]}",
                         subdir="ice_pdp/per_pair", close=True)


plot_pdp_2way(pdp_2way_models, X_pdp_2way)


# =============================================================================
#  SECTION 16 — PySR  (SYMBOLIC REGRESSION: AN EQUATION, NOT A BLACK BOX)
# =============================================================================
#  pip install pysr
#
#  Everything above this point explains a model after the fact — SHAP,
#  permutation, LIME and PDP all take a fitted black box and interrogate it.
#  Symbolic regression skips that step: it searches the space of algebraic
#  expressions directly and returns a formula you can read, publish,
#  differentiate, and check against physics. For a mix-design problem that is
#  not a novelty, it is the deliverable — a closed-form strength model sits in
#  a paper in a way a 600-tree forest never will.
#
#  WHAT PySR ACTUALLY RETURNS. Not one equation: a PARETO FRONT. Genetic
#  search evolves a population of expression trees and keeps, for every
#  complexity level, the best-scoring expression at that complexity. So the
#  output is a ladder from `y = c` up to something long and accurate, and
#  choosing a rung is a modelling decision rather than a numerical one. The
#  whole front is printed below, not just the winner, because the interesting
#  equation is usually two or three rungs BELOW the most accurate one — where
#  the loss has stopped improving much but the formula still fits on a line.
#  `score` in that table is the marginal value of complexity: the drop in
#  log-loss per unit of extra complexity at that rung. A big score is a rung
#  that paid for itself.
#
#  THE JULIA BACKEND. PySR is a Python wrapper over SymbolicRegression.jl, so
#  the first `import pysr` in a fresh environment downloads Julia and
#  precompiles the backend. That takes minutes and needs network access, which
#  is why the import is inside the guard below rather than at the top of the
#  file: an environment without it should skip this section, not fail at
#  import time three hours into a run.
#
#  REPRODUCIBILITY. PySR is only deterministic with BOTH `deterministic=True`
#  AND `parallelism="serial"` AND a fixed `random_state` — it raises if you
#  ask for the first without the second, and merely warns if you set the seed
#  without either, which is the trap: a seeded-looking search that still
#  wanders between runs. PYSR_DETERMINISTIC below sets all three together.
#  Serial search is materially slower, so it is off by default and on when the
#  number has to be quotable.
#
#  COST. Roughly niterations x populations x population_size expression
#  evaluations. The defaults here are a working budget, not a publication
#  budget; PYSR_TIMEOUT caps the wall clock either way.
# =============================================================================
PYSR_ENABLED       = True      # set False to skip this section entirely
PYSR_NITERATIONS   = 40        # search iterations; the main quality/time knob
PYSR_POPULATIONS   = 15
PYSR_POPULATION_SZ = 33
PYSR_MAXSIZE       = 25        # largest expression the search may build
PYSR_TIMEOUT       = 300       # seconds; None for no cap
PYSR_DETERMINISTIC = False     # True -> serial + seeded + reproducible (slower)
PYSR_TOP_ROWS      = 15        # rungs of the Pareto front to print

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
#  rejects anything that collides with a sympy function or one of its own
#  operator names. That is stricter than it sounds: a column called 'w/b'
#  raises "Invalid variable name", and so would 'max', 'abs', 'sign', 're' or
#  'beta'. Renaming is therefore not optional, and a silent rename would be
#  worse than the crash — so the mapping is printed whenever one happens, and
#  the equations are shown with the safe names so they stay copy-pasteable.
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


def print_pareto_front(model, title, top=None):
    """
    Print the Pareto front as a readable ladder, simplest rung first.

    This is the part of a PySR run that is worth reading, and its default
    repr is wide enough to wrap badly in a notebook, so it is reformatted
    here: fixed-width columns, the equation last so it can run long, and a
    marker on the rung `model_selection` actually chose.
    """
    top = PYSR_TOP_ROWS if top is None else top
    eqs = model.equations_
    if isinstance(eqs, list):                 # multi-output; this file is single
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


def print_chosen_equation(model, chosen, target_name, renamed):
    """The selected equation, four ways: plain, sympy, LaTeX, and callable."""
    print("\n" + "=" * 78)
    print(f"SELECTED EQUATION   (complexity {int(chosen['complexity'])}, "
          f"loss {chosen['loss']:.6g})")
    print("=" * 78)
    print(f"\n  {target_name} =")
    print(_wrap_equation(chosen["equation"], indent=6))

    try:
        print("\n  sympy (simplified):")
        print(_wrap_equation(model.sympy(), indent=6))
    except Exception as exc:
        print(f"    sympy form unavailable: {type(exc).__name__}: {exc}")

    try:
        print("\n  LaTeX (paste straight into a paper):")
        print(_wrap_equation(f"{target_name} = {model.latex(precision=4)}", indent=6))
    except Exception as exc:
        print(f"    LaTeX form unavailable: {type(exc).__name__}: {exc}")

    if renamed:
        print("\n  Renamed for PySR (it rejects non-alphanumeric and "
              "function-name variables):")
        for safe, original in renamed.items():
            print(f"      {safe:>20s}  <-  {original}")
    print("\n" + "=" * 78)


def plot_pareto_front(eqs, chosen, title, filename):
    """Loss against complexity, with the selected rung marked."""
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(eqs["complexity"], eqs["loss"], marker='o', color='#0072B2',
            linewidth=1.8, markersize=5, label="Pareto front")
    ax.scatter([chosen["complexity"]], [chosen["loss"]], s=170, marker='*',
               color='#D55E00', zorder=5, edgecolor='black', linewidth=0.7,
               label=f"selected (complexity {int(chosen['complexity'])})")
    ax.set_yscale("log")
    ax.set_xlabel("Complexity (number of nodes in the expression tree)")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, filename, subdir="pysr")
    plt.show()
    plt.close(fig)


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

    _parallelism = "serial" if PYSR_DETERMINISTIC else "multithreading"
    print(f"\nSearching for an equation for {TARGET_COL} "
          f"({PYSR_NITERATIONS} iterations, {_parallelism}"
          + (f", {PYSR_TIMEOUT}s cap" if PYSR_TIMEOUT else "") + ") ...")

    pysr_model = PySRRegressor(
        niterations      = PYSR_NITERATIONS,
        populations      = PYSR_POPULATIONS,
        population_size  = PYSR_POPULATION_SZ,
        maxsize          = PYSR_MAXSIZE,
        binary_operators = PYSR_BINARY_OPS,
        unary_operators  = PYSR_UNARY_OPS,
        elementwise_loss = "L2DistLoss()",        # plain squared error
        model_selection  = "best",                # accuracy/complexity trade-off
        timeout_in_seconds = PYSR_TIMEOUT,
        parallelism      = _parallelism,
        deterministic    = PYSR_DETERMINISTIC,
        random_state     = SEED if PYSR_DETERMINISTIC else None,
        progress         = False,                 # tidy output in a notebook
        verbosity        = 0,
        temp_equation_file = True,                # no stray files in the cwd
    )

    fit_pysr(pysr_model, X_sym_tr, np.asarray(y_tr, dtype=np.float64).ravel(),
             label=TARGET_COL)

    pysr_eqs, pysr_best = print_pareto_front(
        pysr_model, f"PySR PARETO FRONT — every rung from constant to complex")
    print_chosen_equation(pysr_model, pysr_best, TARGET_COL, pysr_renamed)

    # ── HOW GOOD IS THE EQUATION, NEXT TO THE BLACK BOXES? ───────────────────
    #  The whole point is to know what readability costs. If the formula is
    #  within a couple of RMSE points of the best tuned model, that is the
    #  result worth reporting.
    _sym_tr = pysr_model.predict(X_sym_tr)
    _sym_te = pysr_model.predict(X_sym_te)
    pysr_metrics = pd.DataFrame([
        {"Split": "train",
         "RMSE": float(np.sqrt(mean_squared_error(y_tr, _sym_tr))),
         "MAE":  float(mean_absolute_error(y_tr, _sym_tr)),
         "R2":   float(r2_score(y_tr, _sym_tr))},
        {"Split": "test",
         "RMSE": float(np.sqrt(mean_squared_error(y_te, _sym_te))),
         "MAE":  float(mean_absolute_error(y_te, _sym_te)),
         "R2":   float(r2_score(y_te, _sym_te))},
    ])
    print("\nSymbolic model performance:")
    display(pysr_metrics.round(4))

    _bench = summary_table()[["RMSE", "R2"]].copy()
    _bench.loc["PySR (symbolic)"] = [pysr_metrics.loc[1, "RMSE"],
                                     pysr_metrics.loc[1, "R2"]]
    _bench = _bench.sort_values("RMSE")
    print("\nWhere the equation lands against the tuned models (test RMSE):")
    display(_bench.round(4))

    # ── TABLES + FIGURES ─────────────────────────────────────────────────────
    _front = pysr_eqs[["complexity", "loss", "equation"]].copy()
    if "score" in pysr_eqs.columns:
        _front["score"] = pysr_eqs["score"]
    _front["selected"] = (_front["complexity"].astype(int)
                          == int(pysr_best["complexity"]))
    register_table("PySR Pareto front", _front)
    register_table("PySR metrics", pysr_metrics)
    register_table("PySR vs models", _bench, index=True)
    if pysr_renamed:
        register_table("PySR variable names", pd.DataFrame(
            [{"PySR name": s, "Original feature": o}
             for s, o in pysr_renamed.items()]))

    plot_pareto_front(pysr_eqs, pysr_best,
                      "PySR Pareto Front — accuracy against complexity",
                      "pysr_pareto_front")

    #  Predicted vs actual for the equation alone, on the same axes style as
    #  Section 8 so the two can be compared directly.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, (split, yt, yp) in zip(axes, (("train", y_tr, _sym_tr),
                                          ("test", y_te, _sym_te))):
        yt = np.asarray(yt).ravel()
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        ax.scatter(yt, yp, s=26, alpha=0.65, color='#0072B2', edgecolor='none')
        ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=1.2)
        ax.set_xlabel(f"Actual {TARGET_COL}")
        ax.set_ylabel(f"Predicted {TARGET_COL}")
        ax.set_title(f"PySR equation — {split}  "
                     f"($R^2$ = {r2_score(yt, yp):.3f})",
                     fontsize=12, fontweight='bold')
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    save_fig(fig, "pysr_actual_vs_predicted", subdir="pysr")
    plt.show()
    plt.close(fig)

    #  Every rung as its own figure: the point of a Pareto front is to choose
    #  from it, and comparing rungs is much easier side by side than in a list.
    if PER_ITEM_FIGURES:
        for _, row in pysr_eqs.iterrows():
            try:
                pred = pysr_model.predict(X_sym_te, index=int(row.name))
            except Exception:
                continue
            fig, ax = plt.subplots(figsize=(5.6, 5.2))
            yt = np.asarray(y_te).ravel()
            lo, hi = float(min(yt.min(), pred.min())), float(max(yt.max(), pred.max()))
            ax.scatter(yt, pred, s=24, alpha=0.65, color='#009E73', edgecolor='none')
            ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=1.1)
            ax.set_xlabel(f"Actual {TARGET_COL}")
            ax.set_ylabel(f"Predicted {TARGET_COL}")
            ax.set_title(f"complexity {int(row['complexity'])}  "
                         f"($R^2$ = {r2_score(yt, pred):.3f})",
                         fontsize=11, fontweight='bold')
            ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            fig.tight_layout()
            save_fig(fig, f"pysr_complexity_{int(row['complexity']):02d}",
                     subdir="pysr/per_equation")
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
    {"setting": "target column",  "value": TARGET_COL},
    {"setting": "scoring",        "value": SCORING},
    {"setting": "split seed",     "value": BEST_SEED},
    {"setting": "test size",      "value": TEST_SIZE},
    {"setting": "features",       "value": N_FEATURES},
    {"setting": "train rows",     "value": len(y_tr)},
    {"setting": "test rows",      "value": len(y_te)},
    {"setting": "models",         "value": ", ".join(m.name for m in models)},
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
print("=" * 78)
